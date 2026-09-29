"""Unit tests for chiral_interface_analysis: the geometry, trimming, reduction
and statistics helpers, and the CLI's argument handling. None of these need a
trajectory."""

import numpy as np
import pytest
import typer

from chiroflux import chiral_interface_analysis as cia

# ── Stereochemistry ────────────────────────────────────────────────────────

def _tetrahedral(center, clockwise):
    """a, b, c tilted towards a viewer on +z; the lowest-priority group points
    to -z. clockwise=True traces a->b->c clockwise as seen from +z (R)."""
    angles = np.radians([0, -120, -240] if clockwise else [0, 120, 240])
    subs = np.stack([np.cos(angles), np.sin(angles), np.full(3, 0.33)], axis=1)
    return np.asarray(center, float), np.asarray(center, float) + subs


@pytest.mark.parametrize("clockwise, label", [(True, "R"), (False, "S")])
def test_cip_label_follows_the_viewing_rule(clockwise, label):
    center, subs = _tetrahedral([5.0, 5.0, 5.0], clockwise)
    L = np.array([50.0, 50.0, 50.0])
    assert cia.cip_label(cia.signed_volume(center, subs, L)) == label


def test_mirror_image_inverts_the_label():
    center, subs = _tetrahedral([5.0, 5.0, 5.0], clockwise=True)
    L = np.array([50.0, 50.0, 50.0])
    mirror = subs * np.array([-1.0, 1.0, 1.0])
    c_mirror = center * np.array([-1.0, 1.0, 1.0])
    assert cia.cip_label(cia.signed_volume(center, subs, L)) != \
        cia.cip_label(cia.signed_volume(c_mirror, mirror, L))


def test_signed_volume_survives_the_periodic_boundary():
    L = np.array([10.0, 10.0, 10.0])
    center, subs = _tetrahedral([0.2, 5.0, 5.0], clockwise=True)
    wrapped = subs % L                      # one substituent crosses x = 0
    assert np.isclose(cia.signed_volume(center, wrapped, L),
                      cia.signed_volume(center, subs, L))


# ── Midplane ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("centre", [2.0, 40.0, 78.5])
def test_midplane_is_found_wherever_the_bilayer_sits(centre):
    Lz = 80.0
    slab = (centre + np.linspace(-20, 20, 401)) % Lz
    p = (centre + np.r_[np.full(50, -18.5), np.full(50, 18.5)]) % Lz
    mid = cia._midplane(slab, p, Lz)
    d = (mid - centre + Lz / 2) % Lz - Lz / 2
    assert abs(d) < 1e-6


# ── Trimming ───────────────────────────────────────────────────────────────

def test_plus_path_drops_end_points_below_lambda_a():
    op = np.array([-0.5, 0.5, 1.0, -0.2])
    assert cia.path_ensemble(op, 0.0, -1.0) == "plus"
    keep = cia.endpoint_keep_mask(op, "plus", 0.0, -1.0)
    assert keep.tolist() == [False, True, True, False]


def test_minus_path_drops_end_points_outside_its_window():
    op = np.array([0.5, -0.5, -1.5, 0.4])
    assert cia.path_ensemble(op, 0.0, -1.0) == "minus"
    keep = cia.endpoint_keep_mask(op, "minus", 0.0, -1.0)
    assert keep.tolist() == [False, True, True, False]


def test_frame_multiplier_scales_frames_at_or_below_lambda_a():
    unit = cia.Unit(1, [], lam_a=0.0, below_la=2.0)
    assert cia.frame_multiplier(np.array([-1.0, 0.0, 1.0]), unit).tolist() == [2.0, 2.0, 1.0]
    assert cia.frame_multiplier(np.array([-1.0]), cia.Unit(1, [])).tolist() == [1.0]


# ── Reduction ──────────────────────────────────────────────────────────────

def test_runs_splits_on_changes():
    vals, lens = cia._runs(np.array([1, 1, 2, 2, 2, 1]))
    assert vals.tolist() == [1, 2, 1] and lens.tolist() == [2, 3, 1]
    vals, lens = cia._runs(np.array([], dtype=int))
    assert vals.size == 0 and lens.size == 0


def _path(n):
    F = np.zeros((n, len(cia.FLOAT_COLS)), dtype=np.float32)
    F[:, cia.FLOAT_COLS.index("d_c2")] = 0.5          # in contact throughout
    I = np.full((n, 1 + 2 * 2), -1, dtype=np.int32)   # two donors, no H-bonds
    I[:, 0] = 7                                       # always the same lipid
    G = np.zeros((n, 2, cia.N_GLYC_BINS), dtype=np.uint16)
    return {"F": F, "I": I, "G": G}


def test_depth_histogram_carries_the_frame_weights():
    grid = cia.Grid(depth_edges=np.array([0.0, 1.0, 2.0]))
    op = np.array([0.5, 0.5, 1.5, 5.0])
    out = cia.reduce_path(_path(4), op, 0.2, grid, mult=np.array([1.0, 2.0, 3.0, 4.0]))
    assert out["N"].tolist() == [3.0, 3.0, 4.0]
    assert out["N_raw"].tolist() == [2, 1, 1]


def test_contact_runs_break_at_cut_frames():
    grid = cia.Grid()
    k = cia.LIFE_KINDS.index("lipid_contact")
    out = cia.reduce_path(_path(5), np.zeros(5), 0.2, grid, mult=np.ones(5))
    assert out["life_n"][k] == 1 and out["life_sum"][k] == 5
    out = cia.reduce_path(_path(5), np.zeros(5), 0.2, grid,
                          mult=np.array([1.0, 1.0, 0.0, 1.0, 1.0]))
    assert out["life_n"][k] == 2 and out["life_sum"][k] == 4


# ── Statistics ─────────────────────────────────────────────────────────────

def test_block_bootstrap_draws_every_unit_and_caps_the_block():
    rng = np.random.default_rng(0)
    draws = np.concatenate([cia.block_bootstrap_indices(10, 50, rng) for _ in range(200)])
    assert set(draws.tolist()) == set(range(10))
    assert len(cia.block_bootstrap_indices(10, 50, rng)) == 10
    assert cia.block_bootstrap_indices(1, 50, rng).tolist() == [0]


def test_benjamini_hochberg_matches_hand_calculation():
    q = cia.benjamini_hochberg([0.01, 0.04, 0.03, 0.2])
    np.testing.assert_allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.2])


def test_jsd_bounds():
    assert cia.jsd([1, 2, 3], [2, 4, 6]) == pytest.approx(0.0)
    assert cia.jsd([1, 0], [0, 1]) == pytest.approx(np.log(2))
    assert np.isnan(cia.jsd([0, 0], [1, 1]))


# ── CLI argument handling ──────────────────────────────────────────────────

def test_per_run_looks_inside_each_run_directory(tmp_path):
    runs = [tmp_path / "entry", tmp_path / "escape"]
    for r in runs:
        (r / "post").mkdir(parents=True)
        (r / "post" / "infretis_data.txt").write_text("")
    got = cia._per_run("post/infretis_data.txt", [str(r) for r in runs], "-data-l")
    assert got == [str(r / "post" / "infretis_data.txt") for r in runs]


def test_per_run_accepts_one_file_per_run(tmp_path):
    files = [tmp_path / "a.txt", tmp_path / "b.txt"]
    for f in files:
        f.write_text("")
    got = cia._per_run(",".join(map(str, files)), ["run1", "run2"], "-data-l")
    assert got == [str(f) for f in files]


def test_per_run_rejects_a_count_mismatch_and_missing_files(tmp_path):
    with pytest.raises(typer.BadParameter, match="one path for every run"):
        cia._per_run("a,b", ["r1", "r2", "r3"], "-data-l")
    with pytest.raises(typer.BadParameter, match="exists"):
        cia._per_run("missing.txt", [str(tmp_path)], "-data-l")


def test_parse_paths():
    assert cia._parse_paths("1-3,7") == [1, 2, 3, 7]
    assert cia._parse_paths(None) is None


def test_extract_requires_its_inputs():
    with pytest.raises(typer.BadParameter, match="required"):
        cia.chiral_extract()
    with pytest.raises(typer.BadParameter, match="choose from"):
        cia.chiral_extract(tpr="t.tpr", permeant="nope", cache="c", xtc="*.xtc")
    with pytest.raises(typer.BadParameter, match="exactly one"):
        cia.chiral_extract(tpr="t.tpr", permeant="pro", cache="c")


def test_compare_needs_exactly_one_source_per_side():
    with pytest.raises(typer.BadParameter, match="exactly one of -runs-l"):
        cia.chiral_compare(runs_l="a", xtc_l="b", xtc_d="c", cache_l="x", cache_d="y")


def test_compare_refuses_to_overwrite_without_O(tmp_path):
    (tmp_path / "ddG_depth.csv").write_text("")
    with pytest.raises(ValueError, match="already exists"):
        cia.chiral_compare(xtc_l="none*.xtc", xtc_d="none*.xtc", cache_l="x",
                           cache_d="y", out_dir=str(tmp_path))
