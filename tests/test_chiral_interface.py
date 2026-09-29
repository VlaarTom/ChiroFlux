"""Unit tests for chiral_interface_analysis: the geometry, trimming, reduction
and statistics helpers, and the CLI's argument handling. None of these need a
trajectory."""

import glob
import os

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


# ── Extraction and the chunked cache, end to end ───────────────────────────
#
# A tiny but complete system: the neutral-proline permeant, four lipids with
# every atom the analysis selects (two per leaflet), and two waters, written
# as a .gro topology plus real .xtc segments in an infretis-style load/ folder.

_PRO = [("CA", (0, 0, 0)), ("N", (1, 0, 0)), ("C", (-1, 0.5, 0.3)),
        ("CB", (0, -1, -0.8)), ("HA", (0, 0.3, 1)), ("H01", (2, 0, 0)),
        ("O01", (-2, 1, 0.5)), ("H02", (-2.5, 1.5, 0.5)), ("O", (-1, -0.2, 1.3))]
# (name, dx, dy, depth into the membrane)
_LIPID = [("P", 0, 0, 0), ("O13", 1, 0, 0), ("O14", -1, 0, 0), ("O12", 0, -1, 0.8),
          ("O11", 0, 0, 1), ("C1", 0.5, 0, 2.5), ("C2", 0, 0, 3.5), ("C3", -0.5, 0, 4.5),
          ("O21", 0, 1, 3.8), ("O22", 0, 2, 4.5), ("O31", -1, 0, 5.5), ("O32", -1.5, 0, 6)]
_LIPIDS = [(20, 21, 1), (30, 30, 1), (20, 20, -1), (10, 10, -1)]   # x, y, leaflet


def _system():
    """(atom names, residue names per residue, atom -> residue, reference positions)."""
    names, resnames, resix, pos = [], [], [], []
    resnames.append("ORP")
    for n, d in _PRO:
        names.append(n)
        resix.append(0)
        pos.append(np.add((20, 20, 60), d))
    for x, y, s in _LIPIDS:
        resnames.append("DOPC")
        z0 = 58 if s > 0 else 22
        for n, dx, dy, depth in _LIPID:
            names.append(n)
            resix.append(len(resnames) - 1)
            pos.append((x + dx, y + dy, z0 - s * depth))
    for p in ((5, 5, 70), (35, 35, 10)):
        resnames.append("TIP3")
        names.append("OH2")
        resix.append(len(resnames) - 1)
        pos.append(p)
    return names, resnames, np.array(resix), np.array(pos, dtype=float)


BOX = [40.0, 40.0, 80.0, 90.0, 90.0, 90.0]


def _universe(names, resnames, resix):
    import MDAnalysis as mda

    u = mda.Universe.empty(len(names), n_residues=len(resnames), atom_resindex=resix,
                           trajectory=True)
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("resname", resnames)
    u.add_TopologyAttr("resid", np.arange(1, len(resnames) + 1))
    return u


def _write_xtc(u, path, frames, t0):
    import MDAnalysis as mda

    with mda.Writer(str(path), u.atoms.n_atoms) as w:
        for k, p in enumerate(frames):
            u.atoms.positions = p
            u.dimensions = BOX
            u.trajectory.ts.time = t0 + 2.0 * k
            w.write(u)


@pytest.fixture
def run(tmp_path):
    """A run directory with a .gro topology and three paths of two segments
    each; each segment has frames the path does not use."""
    names, resnames, resix, ref = _system()
    u = _universe(names, resnames, resix)
    u.atoms.positions = ref
    u.dimensions = BOX
    gro = tmp_path / "topol.gro"
    u.atoms.write(str(gro))
    rng = np.random.default_rng(1)
    run_dir = tmp_path / "infinit_entry"
    for pn in (3, 7, 12):
        acc = run_dir / "load" / str(pn) / "accepted"
        acc.mkdir(parents=True)
        seg = {n: [ref + rng.normal(0, 0.3, ref.shape) for _ in range(5)]
               for n in ("a.xtc", "b.xtc")}
        for n, frames in seg.items():
            _write_xtc(u, acc / n, frames, 0.0)
        # forward through a.xtc 0-2, then a reversed segment b.xtc 3..1
        plan = [("a.xtc", 0), ("a.xtc", 1), ("a.xtc", 2),
                ("b.xtc", 3), ("b.xtc", 2), ("b.xtc", 1)]
        (acc.parent / "traj.txt").write_text(
            "# step file index vel\n"
            + "".join(f"{k} {f} {i} 1\n" for k, (f, i) in enumerate(plan)))
        (acc.parent / "order.txt").write_text(
            "# time op\n" + "".join(f"{k} {op}\n" for k, op in
                                    enumerate([-0.5, 0.5, 1.5, 2.5, 1.0, -0.2])))
    return str(gro), run_dir


def _direct_features(gro, unit):
    """Reference: the same features through MDAnalysis's standard reader."""
    import MDAnalysis as mda

    T = cia._Topology(gro, cia.PERMEANTS["pro"])
    rows = []
    for fname, idx in unit.plan:
        u = mda.Universe(gro, fname)
        ts = u.trajectory[idx]
        rows.append(cia._frame_features(T, ts.positions, ts.dimensions))
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


def _hidden_files(run_dir):
    return [p for p in run_dir.rglob(".*") if p.is_file()]


def test_extract_matches_the_standard_reader_and_leaves_no_files(run):
    gro, run_dir = run
    units = cia.units_from_load_dir(run_dir / "load", cache_dir=str(run_dir / "cache"))
    errors = cia.extract_many(units, gro, "pro", str(run_dir / "cache"), "infinit_entry",
                              workers=2, per_file=500)
    assert errors == []
    assert _hidden_files(run_dir) == []       # no .xtc_offsets.npz / .lock
    chunks = sorted((run_dir / "cache").iterdir())
    assert [c.name for c in chunks] == ["infinit_entry_0000000-0000499.chiral.npz"]

    for u in units:
        arr, dt, stereo = cia.load_unit(u)
        f_ref, i_ref = _direct_features(gro, u)
        assert arr["F"].shape == (6, len(cia.FLOAT_COLS))
        np.testing.assert_allclose(arr["F"], f_ref, atol=1e-4)
        np.testing.assert_array_equal(arr["I"], i_ref)
        assert dt == pytest.approx(2.0)
        assert stereo["lipids_R"] + stereo["lipids_S"] == len(_LIPIDS)


def test_rerun_skips_cached_paths_and_redoes_changed_ones(run, capsys):
    gro, run_dir = run
    cache = str(run_dir / "cache")

    def extract():
        units = cia.units_from_load_dir(run_dir / "load", cache_dir=cache)
        assert cia.extract_many(units, gro, "pro", cache, "entry", workers=1,
                                per_file=2) == []
        return units

    extract()
    names = sorted(p.name for p in (run_dir / "cache").iterdir())
    assert names == ["entry_0000002-0000003.chiral.npz",
                     "entry_0000006-0000007.chiral.npz",
                     "entry_0000012-0000013.chiral.npz"]
    capsys.readouterr()

    extract()
    assert "3 units cached already, 0 to extract" in capsys.readouterr().out

    # path 7 is rewritten: drop its last phase point
    p7 = run_dir / "load" / "7"
    for f in ("traj.txt", "order.txt"):
        lines = (p7 / f).read_text().splitlines(keepends=True)
        (p7 / f).write_text("".join(lines[:-1]))
    units = extract()
    assert "2 units cached already, 1 to extract" in capsys.readouterr().out
    lengths = {u.uid: len(cia.load_unit(u)[0]["F"]) for u in units}
    assert lengths == {3: 6, 7: 5, 12: 6}


def test_a_different_permeant_invalidates_the_cache(run, capsys):
    gro, run_dir = run
    cache = str(run_dir / "cache")
    units = cia.units_from_load_dir(run_dir / "load", cache_dir=cache)
    cia.extract_many(units, gro, "pro", cache, "entry", workers=1)
    capsys.readouterr()
    # pro_yne's atom names do not exist in this topology: none of the paths
    # counts as cached, and the bad permeant fails with its own message
    with pytest.raises(ValueError, match="ORP atom"):
        cia.extract_many(units, gro, "pro_yne", cache, "entry", workers=1)
    assert "0 units cached already, 3 to extract" in capsys.readouterr().out


def test_reduce_units_from_the_cache(run):
    gro, run_dir = run
    cache = str(run_dir / "cache")
    units = cia.units_from_load_dir(run_dir / "load", cache_dir=cache)
    cia.extract_many(units, gro, "pro", cache, "entry", workers=1)
    for u in units:
        u.trim = (0.0, -1.0)
    missing = cia.Unit(99, [], order_file="x", signature="nope", cache_dir=cache)
    kept, reds, meta = cia.reduce_units(units + [missing], cia.Grid(), workers=2)
    assert [u.uid for u in kept] == [3, 7, 12]
    ok = meta[meta["uid"] != 99]
    assert ok["phase_points"].tolist() == [6, 6, 6]
    assert ok["deleted_first"].all() and ok["deleted_last"].all()   # -0.5 and -0.2
    assert sum(r["N_raw"].sum() for r in reds) == 3 * 4
    assert "not cached" in meta.loc[meta["uid"] == 99, "error"].item()


def test_plain_md_units_are_cached_whole(run):
    gro, run_dir = run
    cache = str(run_dir / "md_cache")
    xtcs = sorted(str(p) for p in (run_dir / "load" / "3" / "accepted").glob("*.xtc"))
    units = cia.units_from_files(xtcs, cache)
    assert cia.extract_many(units, gro, "pro", cache, "md", workers=1) == []
    kept, reds, meta = cia.reduce_units(units, cia.Grid(), workers=1)
    assert meta["phase_points"].tolist() == [5, 5]


def test_cli_extract_defaults_to_a_cache_inside_the_run_directory(run):
    gro, run_dir = run
    cia.chiral_extract(tpr=gro, permeant="pro", load_dir=str(run_dir / "load"), workers=1)
    assert [p.name for p in (run_dir / cia.DEFAULT_CACHE).iterdir()] == \
        ["infinit_entry_0000000-0000499.chiral.npz"]


def test_compare_runs_end_to_end_without_the_trajectories(run, tmp_path):
    gro, run_dir = run
    pattern = str(run_dir / "load" / "*" / "accepted" / "*.xtc")
    cache = str(tmp_path / "md_cache")
    cia.chiral_extract(tpr=gro, permeant="pro", xtc=pattern, cache=cache, workers=1)
    # plain-MD units are keyed on the xtc's size and mtime, so the files stay;
    # but nothing may read them any more: make them unreadable
    for x in glob.glob(pattern):
        os.chmod(x, 0)
    try:
        cia.chiral_compare(xtc_l=pattern, xtc_d=pattern, cache_l=cache, cache_d=cache,
                           temperature=310.0, n_boot=10, block=2, workers=1,
                           out_dir=str(tmp_path / "out"))
    finally:
        for x in glob.glob(pattern):
            os.chmod(x, 0o644)
    written = {p.name for p in (tmp_path / "out").iterdir()}
    assert written == set(cia.COMPARE_OUTPUTS) - {f"stages_{s}.csv" for s in "LD"}
