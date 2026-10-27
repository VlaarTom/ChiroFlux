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


def test_frame_parts_split_at_lambda_a():
    op = np.array([-1.0, 0.0, 1.0])
    above, below = cia.frame_parts(op, cia.Unit(1, [], lam_a=0.0))
    assert above.tolist() == [0.0, 0.0, 1.0] and below.tolist() == [1.0, 1.0, 0.0]
    above, below = cia.frame_parts(op, cia.Unit(1, [], lam_a=0.0, keep_below=False))
    assert above.tolist() == [0.0, 0.0, 1.0] and below.tolist() == [0.0, 0.0, 0.0]
    above, below = cia.frame_parts(op, cia.Unit(1, []))
    assert above.tolist() == [1.0, 1.0, 1.0] and not below.any()


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


def _stage0_units(seed=0, n=12, frames=60, cos_chi=0.3, chi_lz=0.0):
    """Stage-0 paths with frames on both sides of lambda_A = 0, half of them
    [0-] paths ending at lambda_A (R) and half at lambda_-1 (L)."""
    rng = np.random.default_rng(seed)
    grid = cia.Grid(depth_edges=np.array([0.0, 1.0, 2.0]))
    units, reds, ops = [], [], []
    for i in range(n):
        arr = _path(frames)
        arr["F"][:, cia.FLOAT_COLS.index("cos_chi_n")] = cos_chi
        arr["F"][:, cia.FLOAT_COLS.index("chi_lz")] = chi_lz
        op = rng.uniform(-1.5, 1.5, frames)
        u = cia.Unit(i, [], weight=rng.uniform(0.5, 2.0), lam_a=0.0,
                     xi_end="R" if i % 3 else "L")
        above, below = cia.frame_parts(op, u)
        live = (above + below) > 0
        red_b = cia.reduce_path(arr, op, 0.2, grid, below, live)
        for k in cia.RAW_KEYS + ("dt",):
            del red_b[k]
        units.append(u)
        reds.append((cia.reduce_path(arr, op, 0.2, grid, above, live), red_b))
        ops.append((arr, op))
    return grid, units, reds, ops


def test_ensemble_split_equals_scaling_the_below_frames_by_one_over_xi():
    grid, units, reds, ops = _stage0_units()
    e = cia.Ensemble(units, reds)
    w = np.array([u.weight for u in units])
    is_r = np.array([u.xi_end == "R" for u in units])
    xi = w[is_r].sum() / w.sum()
    assert e.xi() == pytest.approx(xi)
    # the old way: 1/xi baked into the frame weights of every path
    old = [cia.reduce_path(arr, op, 0.2, grid, np.where(op <= 0.0, 1.0 / xi, 1.0))
           for arr, op in ops]
    new = e.agg(keys=e.keys)                     # including the lazy pose_depth
    for k in e.keys:
        tol = 1e-6 if e.A[k].dtype == np.float32 else 1e-12   # hist, maps: float32
        np.testing.assert_allclose(new[k], sum(wi * r[k] for wi, r in zip(w, old)),
                                   rtol=tol, atol=tol)


def test_bootstrap_resamples_xi_and_the_stage_factors():
    grid, units, reds, _ = _stage0_units()
    rng = np.random.default_rng(1)
    e = cia.Ensemble(units, reds)
    xis = {round(e.xi(e.draw(rng, 2)), 12) for _ in range(30)}
    assert len(xis) > 1                                  # xi varies per replicate

    class Fixed:
        def draw(self, rng):
            return np.array([3.0])
    e = cia.Ensemble(units, reds, Fixed())
    we = e.draw(rng, 2)
    counts = we / (3.0 * e.w)
    assert np.allclose(counts, np.round(counts)) and counts.sum() == pytest.approx(len(units))


def test_stage_factor_model_draws_around_the_point_factors():
    from chiroflux.permeation_process import Stage, stage_factors

    def stage(a, b, p):
        return Stage(name=f"{a}", centers=np.zeros(1), counts=np.ones(1),
                     lam_a=a, lam_b=b, pcross=p, n_paths=1.0)
    stages = [stage(-25.0, -13.5, 0.003), stage(-13.5, 13.5, 1e-5), stage(13.5, 25.0, 4e-4)]
    factors = np.asarray(stage_factors(stages))
    rng = np.random.default_rng(0)
    fixed = cia.StageFactorModel(stages, np.array([0.0, np.nan, 0.0]), factors)
    assert fixed.draw(rng).tolist() == [1.0, 1.0, 1.0]
    noisy = cia.StageFactorModel(stages, np.array([0.3, 0.3, 0.3]), factors)
    ratios = np.array([noisy.draw(rng) for _ in range(2000)])
    assert np.all(ratios[:, 0] == 1.0)                  # the entry stage is the reference
    # factor = P_entry / P_escape here (P_internal cancels): log-SD sqrt(2) * sigma
    sigma = np.sqrt(np.log1p(0.3 ** 2))
    assert np.std(np.log(ratios[:, 1])) == pytest.approx(np.sqrt(2) * sigma, rel=0.1)
    assert np.median(np.log(ratios[:, 2])) == pytest.approx(0.0, abs=0.05)


def test_read_pcross_rel_err_parses_both_inftools_headers(tmp_path):
    wham = tmp_path / "wham"
    wham.mkdir()
    (wham / "errPtot.txt").write_text(
        "#averaged rel-error PM, WHAM: [np.float64(1.48), np.float64(0.79)]\n1 0.1 0.2\n")
    assert cia.read_pcross_rel_err(str(tmp_path), pcross_col=3) == 0.79
    assert cia.read_pcross_rel_err(str(tmp_path), pcross_col=1) == 0.79
    assert cia.read_pcross_rel_err(str(tmp_path), pcross_col=2) == 1.48
    (wham / "errPtot.txt").write_text("#averaged rel-error PM, WHAM: [0.38, 3.5e-1]\n")
    assert cia.read_pcross_rel_err(str(tmp_path)) == 0.35
    assert np.isnan(cia.read_pcross_rel_err(str(tmp_path / "missing")))


def test_pseudovector_observables_are_tested_against_the_mirror_image():
    # D the mirror image of L with no recognition: the permeant's pseudovector
    # flips against the normal (cos_chi_n), and through the tilt of the
    # lipid's e_z its projection on e_z (chi_lz) flips too
    grid, uL, rL, _ = _stage0_units(seed=0, cos_chi=0.3, chi_lz=0.06)
    _, uD, rD, _ = _stage0_units(seed=0, cos_chi=-0.3, chi_lz=-0.06)
    eL, eD = cia.Ensemble(uL, rL), cia.Ensemble(uD, rD)
    df = cia.depth_profiles(eL, eD, grid, min_frames=1, n_boot=20, block=2)
    for obs, diff in (("cos_chi_n", 0.6), ("chi_lz", 0.12)):
        chi = df[df["observable"] == obs].dropna(subset=["L"])
        assert len(chi) and (chi["tested"] == "L+D").all()
        np.testing.assert_allclose(chi["L_minus_D"], diff, atol=1e-6)
        np.testing.assert_allclose(chi["tested_value"], 0.0, atol=1e-6)
        assert (chi["p_boot"] > 0.5).all()              # no departure from the mirror image
    for obs in ("cos_axis_n", "axis_lz", "pose_z"):
        assert (df[df["observable"] == obs]["tested"] == "L-D").all()


def _pose_units(shallow_share, n=20, frames=100, seed=0):
    """Paths whose pose depends only on depth: pose_z = +0.8 at |OP| 2,
    -0.8 at |OP| 12, with `shallow_share` of the frames at |OP| 2."""
    rng = np.random.default_rng(seed)
    grid = cia.Grid()
    units, reds = [], []
    for i in range(n):
        arr = _path(frames)
        shallow = rng.random(frames) < shallow_share
        op = np.where(shallow, 2.0, 12.0)
        F = arr["F"]
        F[:, cia.FLOAT_COLS.index("pose_x")] = 0.6
        F[:, cia.FLOAT_COLS.index("pose_z")] = np.where(shallow, 0.8, -0.8)
        units.append(cia.Unit(i, [], weight=1.0))
        reds.append((cia.reduce_path(arr, op, 0.2, grid), None))
    return grid, cia.Ensemble(units, reds)


def test_depth_matched_pose_removes_a_pure_depth_mix_difference():
    grid, eL = _pose_units(0.8, seed=0)
    _, eD = _pose_units(0.2, seed=1)
    maps, df = cia.pose_maps_depth(eL, eD, grid, block=2, n_perm=50, seed=0, min_frames=50)
    rows = df.set_index("map")
    assert rows.loc["pooled", "jsd"] > 0.1             # pooled maps differ by depth mix
    assert rows.loc["depth_matched", "jsd"] == pytest.approx(0.0, abs=1e-12)
    per = df[df["map"] == "depth"]
    assert per["depth"].tolist() == ["0.00-5.00", "10.00-15.00"]
    np.testing.assert_allclose(per["jsd"], 0.0, atol=1e-12)
    assert maps["contact_share_L"][0] == pytest.approx(0.8, abs=0.05)
    np.testing.assert_allclose(maps["match_weight"].sum(), 1.0)


def test_pose_depth_edges_split_at_the_stage_boundaries():
    import pandas as pd

    st = pd.DataFrame({"lam_a": [-25.0, -13.5, 13.5], "lam_b": [-13.5, 13.5, 25.0]})
    edges = cia.pose_depth_edges(np.arange(0.0, 30.01, 5.0), [st, st.copy(), None])
    assert edges.tolist() == [0.0, 5.0, 10.0, 13.5, 15.0, 20.0, 25.0, 30.0]
    assert cia.pose_depth_edges([0.0, 10.0], [None]).tolist() == [0.0, 10.0]


def test_missing_paths_leave_their_stage_weight_to_the_others():
    units = [cia.Unit(i, [], weight=w, stage=s)
             for i, (w, s) in enumerate([(1.0, 0), (3.0, 0), (2.0, 0), (5.0, 1)])]
    kept = [units[0], units[2], units[3]]                # stage-0 path with weight 3 lost
    new, table = cia.rescale_missing(units, kept)
    assert [u.weight for u in new] == [2.0, 4.0, 5.0]   # stage 0 back to its total, 6
    assert [u.weight for u in kept] == [1.0, 2.0, 5.0]  # the inputs are untouched
    row = table.set_index("stage").loc[0]
    assert row["paths_missing"] == 1 and row["weight_missing_share"] == pytest.approx(0.5)
    assert table.set_index("stage").loc[1, "weight_rescale"] == 1.0


def test_path_class_from_the_raw_order_parameter():
    # lambda_-1 = -1, lambda_A = 0, lambda_B = 2
    assert cia.path_class(np.array([-0.5, 0.5, 2.1]), 0.0, 2.0, -1.0) == "reactive"
    assert cia.path_class(np.array([-0.5, 0.5, 1.5, -0.2]), 0.0, 2.0, -1.0) == "unreactive"
    assert cia.path_class(np.array([0.5, -0.5, -1.5]), 0.0, 2.0, -1.0) == "minus"


def test_uniform_weights_keep_the_paths_and_the_normalisation_of_wftot():
    plus = {1: 0.7, 2: 0.3, 3: 0.0}
    minus = {4: 0.9, 5: 0.1}
    u = cia.uniform_weights(plus, minus)
    assert u == {1: 0.5, 2: 0.5, 4: 0.5, 5: 0.5}       # path 3 has no weight either way
    w = cia.combine_weights(plus, minus)
    assert sum(v for p, v in w.items() if p in plus) == pytest.approx(1.0)
    assert sum(u.values()) == pytest.approx(sum(w.values()))


def test_a_subset_view_equals_an_ensemble_of_just_those_units():
    grid, units, reds, _ = _stage0_units()
    full = cia.Ensemble(units, reds)
    keep = np.array([i % 3 != 0 for i in range(len(units))])   # drops every 'L' end
    view = full.subset(keep)
    own = cia.Ensemble([u for u, k in zip(units, keep) if k],
                       [r for r, k in zip(reds, keep) if k])
    # the view keeps the whole run's xi; give the stand-alone ensemble the same
    own.xi = lambda we=None: full.xi()
    a, b = view.agg(keys=full.keys), own.agg(keys=full.keys)
    for k in full.keys:
        np.testing.assert_allclose(a[k], b[k], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(view.raw("N_raw"), own.raw("N_raw"))
    assert view.n_counted == keep.sum() and view.n == len(units)


def test_split_half_finds_a_difference_present_in_both_halves():
    grid = cia.Grid()

    def ens(pose_z, n=40, seed=0):
        rng = np.random.default_rng(seed)
        units, reds = [], []
        for i in range(n):
            arr = _path(60)
            F = arr["F"]
            F[:, cia.FLOAT_COLS.index("pose_x")] = 0.6
            F[:, cia.FLOAT_COLS.index("pose_z")] = pose_z + 0.1 * rng.standard_normal(60)
            units.append(cia.Unit(i, [], weight=1.0))
            reds.append((cia.reduce_path(arr, np.full(60, 12.0), 0.2, grid), None))
        return cia.Ensemble(units, reds)

    eL, eD = ens(0.6, seed=0), ens(-0.6, seed=1)
    df, maps = cia.split_half_pose(eL, eD, grid, block=2, n_perm=50, min_frames=100)
    row = df.set_index("depth").loc["10.00-15.00"]
    assert row["corr_halves"] > 0.9 and row["p_perm"] < 0.05
    assert maps["half1"].shape == maps["half2"].shape
    assert maps["labels"].dtype.kind == "U"        # loadable without pickle


def test_a_stage_view_equals_an_ensemble_of_that_stage_alone():
    grid, units, reds, _ = _stage0_units(n=12)
    for u in units[6:]:                       # second half becomes stage 1
        u.stage, u.xi_end = 1, None
    reds = [(r, None) if i >= 6 else (r, b) for i, (r, b) in enumerate(reds)]
    full = cia.Ensemble(units, reds)
    for s, (a, b) in ((0, (0, 6)), (1, (6, 12))):
        assert full.stage_range(s) == (a, b)
        view = full.restrict(a, b)
        own = cia.Ensemble(units[a:b], reds[a:b])
        assert view.n == b - a and view.xi() == pytest.approx(own.xi())
        x, y = view.agg(keys=full.keys), own.agg(keys=full.keys)
        for k in full.keys:
            np.testing.assert_allclose(x[k], y[k], rtol=1e-12, atol=1e-12)
        # a class subset inside the stage view
        keep = np.arange(b - a) % 2 == 0
        sub = view.subset(keep)
        ref = cia.Ensemble([u for u, k in zip(units[a:b], keep) if k],
                           [r for r, k in zip(reds[a:b], keep) if k])
        ref.xi = lambda we=None, v=view: v.xi()
        np.testing.assert_allclose(sub.agg()["N"], ref.agg()["N"], rtol=1e-12)
        rng = np.random.default_rng(0)
        assert view.draw(rng, 2).shape == (b - a,)


def test_stage_folder_names_and_result_folders(tmp_path):
    import pandas as pd
    from chiroflux import chiral_plot

    st = pd.DataFrame({"stage": [0, 1], "run": ["/x/L/infinit_entry/", "/x/L/infinit_internal"]})
    names = cia.stage_folder_names(st)
    assert names == {0: "stage0_infinit_entry", 1: "stage1_infinit_internal"}
    for cls in ("reactive", "unreactive"):
        for n in names.values():
            (tmp_path / cls / n).mkdir(parents=True)
    found = chiral_plot.result_folders(str(tmp_path))
    assert found[0] == str(tmp_path) and len(found) == 1 + 2 * 3
    assert chiral_plot._subset(str(tmp_path / "reactive")) == "reactive paths only"
    assert chiral_plot._subset(str(tmp_path / "unreactive" / "stage1_infinit_internal")) \
        == "unreactive paths only, stage1_infinit_internal"
    assert chiral_plot._subset(str(tmp_path)) is None
    (tmp_path / "stage0_infinit_entry").mkdir()
    assert chiral_plot._subset(str(tmp_path / "stage0_infinit_entry")) \
        == "all paths, stage0_infinit_entry"
    assert str(tmp_path / "stage0_infinit_entry") in chiral_plot.result_folders(str(tmp_path))
    (tmp_path / "unweighted" / "reactive" / "stage1_infinit_internal").mkdir(parents=True)
    assert chiral_plot._subset(str(tmp_path / "unweighted")) == "unweighted (no WHAM reweighting)"
    assert chiral_plot._subset(str(tmp_path / "unweighted" / "reactive" / "stage1_infinit_internal")) \
        == "unweighted (no WHAM reweighting), reactive paths only, stage1_infinit_internal"
    found = chiral_plot.result_folders(str(tmp_path))
    assert str(tmp_path / "unweighted" / "reactive" / "stage1_infinit_internal") in found


def _contact_units(fractions, frames=50):
    """One path per value: that fraction of its frames in contact."""
    grid = cia.Grid()
    units, reds = [], []
    for i, f in enumerate(fractions):
        arr = _path(frames)
        k = int(round(f * frames))
        arr["F"][k:, cia.FLOAT_COLS.index("d_c2")] = 2.0          # out of contact
        units.append(cia.Unit(i, [], weight=1.0))
        reds.append((cia.reduce_path(arr, np.full(frames, 12.0), 0.2, grid), None))
    return cia.Ensemble(units, reds)


def test_blocking_finds_the_correlation_length():
    rng = np.random.default_rng(0)
    independent = _contact_units(rng.uniform(0.2, 0.8, 1000))
    runs = _contact_units(np.repeat(rng.uniform(0.2, 0.8, 50), 20))   # 20 identical paths
    t_ind, q_ind = cia.blocking_analysis(independent, "L")
    t_run, _ = cia.blocking_analysis(runs, "L")
    pick = lambda t, B: t[(t.observable == "contact fraction") & (t.block == B)].inefficiency.item()
    assert pick(t_ind, 1) == pytest.approx(1.0) and pick(t_run, 1) == pytest.approx(1.0)
    assert 0.5 < pick(t_ind, 50) < 1.6                 # independent: stays near 1
    assert pick(t_run, 100) == pytest.approx(20, rel=0.25)   # runs of 20: levels off at 20
    row = q_ind[q_ind.observable == "contact fraction"].iloc[0]
    assert abs(row.quarter1 - row.whole_stage) < 0.05


def test_an_ensemble_with_other_weights_shares_the_paths():
    grid, units, reds, _ = _stage0_units()
    e = cia.Ensemble(units, reds)
    flat = e.with_weights(np.ones(e.n))
    assert flat.A is e.A and flat.n == e.n
    own = cia.Ensemble([cia.Unit(u.uid, [], weight=1.0, lam_a=u.lam_a, xi_end=u.xi_end)
                        for u in units], reds)
    for k in ("N", "pose", "glyc_bulk"):
        np.testing.assert_allclose(flat.agg(keys=[k])[k], own.agg(keys=[k])[k], rtol=1e-12)
    assert flat.xi() == pytest.approx(own.xi()) and flat.xi() != pytest.approx(e.xi())
    with pytest.raises(ValueError):
        e.with_weights([1.0])


def test_the_unweighted_analysis_writes_no_free_energies(tmp_path, capsys):
    grid, units, reds, _ = _stage0_units(n=12)
    e = cia.Ensemble(units, reds)
    flat = e.with_weights(np.ones(e.n))
    boot = dict(n_boot=5, block=2, seed=0)
    tables = cia._analyse(flat, flat, units, units, str(tmp_path), grid=grid, boot=boot,
                          by_class=False, split_half=False, names={}, temperature=None)
    written = {p.name for p in tmp_path.iterdir()}
    assert "ddG_depth.csv" not in written and "permeability_ratio.csv" not in written
    assert {"depth_profiles.csv", "pose_jsd_depth.csv", "blocking.csv"} <= written
    assert "ddG_depth" not in tables


def test_choose_block_takes_the_slowest_quantity_that_levels_off():
    import pandas as pd

    def rows(obs, curve, n=10000, stage=0, side="L"):
        return [{"side": side, "stage": stage, "observable": obs, "block": B,
                 "n_blocks": n // B, "inefficiency": s, "reliable": n // B >= 10}
                for B, s in curve.items()]
    fast = {1: 1.0, 10: 1.6, 25: 1.7, 50: 1.75, 100: 1.7, 250: 1.8, 500: 1.75}
    slow = {1: 1.0, 10: 1.2, 25: 1.8, 50: 2.6, 100: 3.5, 250: 3.7, 500: 3.8}
    t = pd.DataFrame(rows("fast", fast) + rows("slow", slow))
    block, rec, info = cia.choose_block(t)
    assert block == 100 and info["levelled"] and not info["capped"]
    assert rec.set_index("observable").loc["fast", "needed_block"] == 10
    # never levels off: the largest size that keeps 15 blocks, flagged
    rising = {1: 1, 10: 2, 25: 3, 50: 4, 100: 6, 250: 9, 500: 14}
    block, rec, info = cia.choose_block(pd.DataFrame(rows("rising", rising)))
    assert block == 500 and not info["levelled"] and "rising" in info["not_levelled"]
    # a short stage caps the folder's block so it keeps 15 blocks
    t = pd.DataFrame(rows("slow", slow) + rows("slow", slow, n=2000, stage=1))
    block, _, info = cia.choose_block(t)
    assert block == 100 and 2000 // block >= 15


def _blocking_rows(obs, curve, n, stage=0, side="L"):
    return [{"side": side, "stage": stage, "observable": obs, "block": B,
             "n_blocks": n // B, "inefficiency": s, "reliable": n // B >= 10}
            for B, s in curve.items()]


def test_choose_block_reads_a_noisy_flat_curve_as_independent_paths():
    import pandas as pd
    # ~1 everywhere, one noisy 1.56 at 17 blocks (scatter there ~35%): block 1
    noisy = {1: 1.0, 10: 1.05, 25: 1.11, 50: 1.15, 100: 1.12, 250: 1.56}
    block, rec, _ = cia.choose_block(pd.DataFrame(_blocking_rows("q", noisy, n=4283)))
    assert block == 1 and bool(rec["independent"].iloc[0])
    # the same rise measured with many blocks is real: not independent
    block, rec, _ = cia.choose_block(pd.DataFrame(_blocking_rows("q", noisy, n=100000)))
    assert block == 250 and not bool(rec["independent"].iloc[0])


def test_bulk_lipid_quantities_get_their_own_block(tmp_path):
    import pandas as pd
    flat = {1: 1.0, 10: 1.02, 25: 0.98, 50: 1.03, 100: 1.0, 250: 1.01, 500: 0.99}
    slow = {1: 1.0, 10: 1.5, 25: 2.2, 50: 3.0, 100: 3.9, 250: 4.0, 500: 4.1}
    t = pd.DataFrame(_blocking_rows("contact fraction", flat, n=10000)
                     + _blocking_rows("bulk glycerol b trans", slow, n=10000))
    block, lipid_block = cia._choose_and_report(t, None, str(tmp_path), "test")
    assert block == 1 and lipid_block == 100
    choice = pd.read_csv(tmp_path / "block_choice.csv").set_index("observable")
    assert choice.loc["bulk glycerol b trans", "used_for"] == "glycerol_conformers"
    assert choice.loc["contact fraction", "chosen_block"] == 1
    # a fixed block is used for both
    assert cia._choose_and_report(t, 50, str(tmp_path), "test") == (50, 50)


def test_blocking_of_a_subset_keeps_the_chain_positions():
    rng = np.random.default_rng(0)
    e = _contact_units(np.repeat(rng.uniform(0.2, 0.8, 50), 20))     # runs of 20
    keep = np.arange(e.n) % 2 == 0                                   # every other path
    t, _ = cia.blocking_analysis(e.subset(keep), "L")
    s = t[(t.observable == "contact fraction")].set_index("block").inefficiency
    # half the paths of each run of 20: 10 correlated paths per run
    assert s[1] == pytest.approx(1.0) and s[100] == pytest.approx(10, rel=0.3)


def _axis_units(scale, n=60, frames=80, seed=0, col="cos_axis_n", extra=0.0):
    """Paths spread over depth whose `col` (cos_axis_n) follows
    scale * pattern(depth) + extra * a second pattern."""
    rng = np.random.default_rng(seed)
    grid = cia.Grid()
    units, reds = [], []
    for i in range(n):
        arr = _path(frames)
        op = rng.uniform(0.0, 30.0, frames)
        pattern = np.sin(op / 5.0)
        arr["F"][:, cia.FLOAT_COLS.index(col)] = scale * pattern \
            + extra * np.cos(op / 7.0) + 0.05 * rng.standard_normal(frames)
        units.append(cia.Unit(i, [], weight=1.0))
        reds.append((cia.reduce_path(arr, op, 0.2, grid), None))
    return grid, cia.Ensemble(units, reds)


def test_split_half_profiles_finds_a_difference_in_both_halves():
    grid, eL = _axis_units(0.6, seed=0)
    _, eD = _axis_units(0.2, seed=1)
    rows, summary = cia.split_half_profiles(eL, eD, grid, block=3, n_perm=50, n_boot=30)
    r = summary.set_index("observable").loc["cos_axis_n"]
    assert r["corr_halves"] > 0.9 and r["p_perm"] < 0.05
    ax = rows[rows.observable == "cos_axis_n"]
    assert ax["same_sign"].mean() > 0.8 and ax["both_clear_of_zero"].sum() >= 3
    # identical L and D: nothing replicates beyond chance
    _, eD2 = _axis_units(0.6, seed=2)
    _, summary2 = cia.split_half_profiles(eL, eD2, grid, block=3, n_perm=50, n_boot=30)
    assert summary2.set_index("observable").loc["cos_axis_n", "p_perm"] > 0.05


def test_bootstrap_p_goes_below_its_resolution_for_a_clear_difference():
    # every replicate on one side: the p is the normal tail, not 1/n_boot, so
    # one strong result does not need others at the floor to pass BH
    grid, eL = _axis_units(0.6, seed=0)
    _, eD = _axis_units(0.2, seed=1)
    df = cia.depth_profiles(eL, eD, grid, n_boot=20, block=3)
    ax = df[(df.observable == "cos_axis_n") & df["p_normal_tail"]]
    assert len(ax) and (ax["p_boot"] < 1.0 / 20).all()
    # where replicates fall on both sides the empirical p stands
    rest = df[~df["p_normal_tail"] & np.isfinite(df["p_boot"])]
    assert len(rest) and (rest["p_boot"] >= 1.0 / 20).all()


def test_split_half_mirror_null_for_the_pseudoscalar():
    # D the exact mirror image of L (cos_chi_n -> -cos_chi_n): L + D is noise.
    # With more D blocks than L, relabelling unmirrored blocks would leave
    # part of the orientation profile in both halves' "L + D" (null_95 ~ 1);
    # the mirrored null is pure noise.
    grid, eL = _axis_units(0.6, seed=0, col="cos_chi_n")
    _, eD = _axis_units(-0.6, n=120, seed=1, col="cos_chi_n")
    rows, summary = cia.split_half_profiles(eL, eD, grid, block=3, n_perm=50, n_boot=30)
    assert (rows[rows.observable == "cos_chi_n"]["tested"] == "L+D").all()
    r = summary.set_index("observable").loc["cos_chi_n"]
    assert r["p_perm"] > 0.05 and r["null_95"] < 0.8
    # a departure from the mirror image with its own depth shape replicates
    _, eD2 = _axis_units(-0.6, seed=1, col="cos_chi_n", extra=0.3)
    _, summary2 = cia.split_half_profiles(eL, eD2, grid, block=3, n_perm=50, n_boot=30)
    r = summary2.set_index("observable").loc["cos_chi_n"]
    assert r["corr_halves"] > 0.9 and r["p_perm"] < 0.05


def test_analyse_writes_a_folder_per_stage_for_all_paths(tmp_path):
    grid, units, reds, _ = _stage0_units(n=12)
    e = cia.Ensemble(units, reds)
    boot = dict(n_boot=5, block=2, seed=0)
    cia._analyse(e, e, units, units, str(tmp_path), grid=grid, boot=boot, by_class=True,
                 split_half=False, names={0: "stage0_infinit_entry"}, temperature=None)
    stage = tmp_path / "stage0_infinit_entry"
    assert (stage / "depth_profiles.csv").exists() and (stage / "block_choice.csv").exists()


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
    assert sum(r[0]["N_raw"].sum() for r in reds) == 3 * 4      # raw counts: above part
    assert "not cached" in meta.loc[meta["uid"] == 99, "error"].item()


def test_compare_units_without_a_plan_reduce_the_same(run):
    gro, run_dir = run
    cache = str(run_dir / "cache")
    full = cia.units_from_load_dir(run_dir / "load", cache_dir=cache)
    cia.extract_many(full, gro, "pro", cache, "entry", workers=1)
    bare = cia.units_from_load_dir(run_dir / "load", cache_dir=cache, keep_plan=False)
    assert all(u.plan == [] for u in bare)
    assert [(u.signature, u.g96_index) for u in bare] == \
        [(u.signature, u.g96_index) for u in full]
    _, red_full, _ = cia.reduce_units(full, cia.Grid(), workers=1)
    _, red_bare, _ = cia.reduce_units(bare, cia.Grid(), workers=1)
    for a, b in zip(red_full, red_bare):
        for part_a, part_b in zip(a, b):
            assert (part_a is None) == (part_b is None)
            for k in part_a or {}:
                np.testing.assert_array_equal(part_a[k], part_b[k])


def test_a_cache_folder_without_chunks_fails_before_setup(run):
    gro, run_dir = run
    cache = run_dir / "analysis" / "chiral_cache"
    units = cia.units_from_load_dir(run_dir / "load", cache_dir=str(cache))
    cia.extract_many(units, gro, "pro", str(cache), "entry", workers=1)
    cia._check_cache_dirs([("l", "entry", str(cache))])          # the right folder
    with pytest.raises(typer.BadParameter, match="-cache-l for entry"):
        cia._check_cache_dirs([("l", "entry", str(cache)),
                               ("l", "entry", str(run_dir / "analysis"))])


def test_check_cached_reports_missing_and_stale_paths(run, capsys):
    gro, run_dir = run
    cache = str(run_dir / "cache")
    units = cia.units_from_load_dir(run_dir / "load", cache_dir=cache, keep_plan=False)
    with pytest.raises(typer.BadParameter, match="none of the paths"):
        cia.check_cached(units, "L")                              # nothing extracted
    full = cia.units_from_load_dir(run_dir / "load", cache_dir=cache)
    cia.extract_many(full[:2], gro, "pro", cache, "entry", workers=1)
    units[0].signature = "changed"
    capsys.readouterr()
    cia.check_cached(units, "L")
    out = capsys.readouterr().out
    assert "L stage 0: 1/3 paths cached" in out
    assert "1 not in the cache: 12" in out
    assert "1 changed since extraction (traj.txt differs): 3" in out


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


def test_compare_runs_end_to_end_without_the_trajectories(run, tmp_path, capsys):
    from chiroflux import chiral_plot

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
    written = {p.name for p in (tmp_path / "out").iterdir() if p.is_file()}
    figures = written & set(chiral_plot.PLOT_OUTPUTS)
    # the histogram figures go to their own subfolder
    assert (tmp_path / "out" / chiral_plot.HIST_DIR).is_dir()
    # plain MD has no stages and no path classes
    assert written - figures == set(cia.COMPARE_OUTPUTS) - {
        "stages_L.csv", "stages_D.csv", "path_classes.csv"}
    assert "plotting failed" not in capsys.readouterr().out
    # the tiny test system has too few frames for a free-energy profile, but
    # every figure whose tables have data is drawn
    assert {"chiral_profiles.png", "chiral_pose.png", "chiral_lifetimes.png"} <= figures

    with pytest.raises(ValueError, match="already exists"):
        chiral_plot.chiral_plot(in_dir=str(tmp_path / "out"))
    chiral_plot.chiral_plot(in_dir=str(tmp_path / "out"), overw=True)


# ── Histograms ────────────────────────────────────────────────────────────────

def test_histograms_hold_every_weighted_frame_of_the_profiles():
    grid, units, reds, _ = _stage0_units()
    e = cia.Ensemble(units, reds)
    a = e.agg(keys=("hist", "C"))
    names = [n for n, _, _ in cia.PROFILE_OBS]
    for name, _, _, sl in cia._hist_layout():
        np.testing.assert_allclose(a["hist"][:, sl].sum(axis=1), a["C"][names.index(name)],
                                   rtol=1e-6)


def test_plot_module_knows_every_histogram():
    from chiroflux import chiral_plot

    assert chiral_plot.HIST_NAMES == tuple(n for n, _, _ in cia.HIST_OBS)


def test_histograms_find_a_shifted_distribution_and_respect_the_mirror_null():
    grid, eL = _axis_units(0.6, seed=0)
    _, eD = _axis_units(0.2, seed=1)
    df, arrays = cia.histograms(eL, eD, grid, min_frames=50, n_boot=30, block=3)
    ax = df[df.observable == "cos_axis_n"]
    assert (ax["q_bh"] < 0.05).any()
    # each window's histogram is a distribution
    L1 = arrays["L_cos_axis_n_1d"]
    np.testing.assert_allclose(np.nansum(L1, axis=1)[np.isfinite(L1).any(axis=1)], 1.0)
    # D the mirror image of L for the pseudovector (up to the noise): nothing
    # to find, although L - D itself is large
    _, gL = _axis_units(0.6, seed=0, col="cos_chi_n")
    _, gD = _axis_units(-0.6, seed=1, col="cos_chi_n")
    df, arrays = cia.histograms(gL, gD, grid, min_frames=50, n_boot=30, block=3)
    chi = df[df.observable == "cos_chi_n"]
    assert (chi["tested"] == "L vs mirrored D").all()
    assert not (chi["q_bh"] < 0.05).any()
    raw = np.nanmax(np.abs(arrays["L_cos_chi_n_1d"] - arrays["D_cos_chi_n_1d"]))
    assert chi["L_minus_D"].abs().max() < raw / 2


def test_histogram_figures_are_drawn(tmp_path):
    from chiroflux import chiral_plot

    grid, eL = _axis_units(0.6, seed=0)
    _, eD = _axis_units(0.2, seed=1)
    _, arrays = cia.histograms(eL, eD, grid, min_frames=50, n_boot=10, block=3)
    np.savez(tmp_path / "histograms.npz", **arrays)
    chiral_plot._style()
    chiral_plot.plot_hist1d(str(tmp_path))
    chiral_plot.plot_hist2d(str(tmp_path))
    written = {p.name for p in (tmp_path / chiral_plot.HIST_DIR).iterdir()}
    assert {"chiral_hist1d_cos_axis_n.png", "chiral_hist2d_cos_axis_n.png",
            "chiral_hist1d_cos_chi_n.png"} <= written


# ── Orientation maps ─────────────────────────────────────────────────────────

def _orient_units(axis, chi, n=40, frames=60, seed=0, noise=0.15):
    """Paths in contact at depths 0-30 whose axis and chirality vectors, in
    the lipid frame, scatter around `axis` and `chi`."""
    rng = np.random.default_rng(seed)
    grid = cia.Grid()
    units, reds = [], []
    for i in range(n):
        arr = _path(frames)
        op = rng.uniform(0.0, 30.0, frames)
        for c, v in (("axis_l", axis), ("chi_l", chi)):
            vec = np.asarray(v, float) + noise * rng.standard_normal((frames, 3))
            vec /= np.linalg.norm(vec, axis=1, keepdims=True)
            for k, comp in zip("xyz", vec.T):
                arr["F"][:, cia.FLOAT_COLS.index(c + k)] = comp
        units.append(cia.Unit(i, [], weight=1.0))
        reds.append((cia.reduce_path(arr, op, 0.2, grid), None))
    return grid, cia.Ensemble(units, reds)


def test_inverting_a_map_twice_gives_it_back():
    m = np.random.default_rng(0).random((3, 24, 12))
    np.testing.assert_allclose(cia._invert(cia._invert(m)), m)
    # a vector straight along +e_z lands at cos = -1 once inverted
    up = np.zeros((24, 12)); up[3, -1] = 1.0
    assert cia._invert(up)[15, 0] == 1.0


def test_orientation_maps_compare_the_axis_directly_and_the_chirality_vector_inverted():
    tilt = [0.6, 0.0, 0.8]
    grid, eL = _orient_units(axis=tilt, chi=tilt, seed=0)
    # D: axis leaning the other way around e_z (a real difference), chirality
    # vector the exact opposite of L's (what no chiral recognition gives)
    _, eD = _orient_units(axis=[-0.6, 0.0, 0.8], chi=[-0.6, 0.0, -0.8], seed=1)
    df, maps = cia.orientation_maps(eL, eD, grid, block=3, n_perm=100, min_frames=50)
    m = df[df["map"] == "depth_matched"].set_index("vector")
    assert m.loc["axis", "p_perm"] < 0.05
    assert m.loc["chirality", "p_perm"] > 0.05
    assert set(df["tested"]) == {"L vs D", "L vs D inverted (-n)"}
    assert maps["axis_L_matched"].shape == (grid.n_phi, grid.n_cos)


def test_orientation_figures_are_drawn(tmp_path):
    from chiroflux import chiral_plot

    tilt = [0.6, 0.0, 0.8]
    grid, eL = _orient_units(axis=tilt, chi=tilt, seed=0)
    _, eD = _orient_units(axis=[-0.6, 0.0, 0.8], chi=[-0.6, 0.0, -0.8], seed=1)
    df, maps = cia.orientation_maps(eL, eD, grid, block=3, n_perm=20, min_frames=50)
    np.savez(tmp_path / "orient_maps_depth.npz", **maps)
    df.to_csv(tmp_path / "orient_jsd_depth.csv", index=False)
    chiral_plot._style()
    chiral_plot.plot_orient(str(tmp_path))
    assert (tmp_path / "chiral_orient_axis.png").exists()
    assert (tmp_path / "chiral_orient_chirality.png").exists()


def test_split_half_of_the_orientation_maps(tmp_path):
    from chiroflux import chiral_plot

    tilt = [0.6, 0.0, 0.8]
    # enough contact frames per half and depth for the 1000-frame minimum
    grid, eL = _orient_units(axis=tilt, chi=tilt, n=200, frames=100, seed=0)
    _, eD = _orient_units(axis=[-0.6, 0.0, 0.8], chi=[-0.6, 0.0, -0.8], n=200, frames=100,
                          seed=1)
    tables = cia._split_half_tables(eL, eD, grid, str(tmp_path), block=3, seed=0)
    t = tables["split_half_orient"]
    assert set(t["vector"]) == {"axis", "chirality"}
    # the real axis difference shows in both halves, the mirror image does not
    axis = t[(t["vector"] == "axis") & (t["map"] == "depth_matched")].iloc[0]
    assert axis["corr_halves"] > 0.5
    t.to_csv(tmp_path / "split_half_orient.csv", index=False)
    tables["split_half"].to_csv(tmp_path / "split_half.csv", index=False)
    chiral_plot._style()
    chiral_plot.plot_split_half_orient(str(tmp_path))
    chiral_plot.plot_split_half(str(tmp_path))
    assert (tmp_path / "chiral_split_half_orient_axis.png").exists()
    assert (tmp_path / "chiral_split_half.png").exists()


def test_histograms_stop_at_the_depth_limit():
    import pandas as pd
    from dataclasses import replace

    grid, eL = _axis_units(0.6, seed=0)
    _, eD = _axis_units(0.2, seed=1)
    for limit in (22.5, 25.0, None):
        g = replace(grid, hist_max_depth=limit)
        df, arrays = cia.histograms(eL, eD, g, min_frames=50, n_boot=5, block=3)
        wins = list(arrays["windows"])
        # windows are labelled by the depth bins they pool, all below the limit
        edges = {f"{x:.2f}" for x in g.depth_edges}
        assert all(w.split("-")[0] in edges and w.split("-")[1] in edges for w in wins)
        assert float(wins[-1].split("-")[1]) <= (limit or np.inf)
        assert df["window"].iloc[-1] == wins[-1]
        mids = 0.5 * (g.depth_edges[:-1] + g.depth_edges[1:])
        beyond = np.r_[mids >= (limit or np.inf), True]          # + the open bulk row
        assert np.isnan(arrays["L_cos_axis_n_2d"][beyond]).all()
        assert np.isfinite(arrays["L_cos_axis_n_2d"][~beyond]).any()
    # compare's default: |lambda_A| of the entry stage, unless set
    stages = pd.DataFrame({"stage": [1, 0], "lam_a": [-13.5, -25.0]})
    assert cia.with_hist_limit(grid, stages).hist_max_depth == 25.0
    assert cia.with_hist_limit(replace(grid, hist_max_depth=30.0), stages).hist_max_depth == 30.0
    assert cia.with_hist_limit(grid, None).hist_max_depth is None


# ── Reorientation, orientation free energy, barrier ──────────────────────────

def _turning_units(step, n=40, frames=2100, seed=0):
    """Paths in contact with one lipid throughout, whose axis vector does a
    random walk on the sphere with steps of size `step` (small: slow turning)."""
    rng = np.random.default_rng(seed)
    grid = cia.Grid()
    units, reds = [], []
    for i in range(n):
        arr = _path(frames)
        u = np.empty((frames, 3))
        u[0] = rng.standard_normal(3)
        u[0] /= np.linalg.norm(u[0])
        for t in range(1, frames):
            v = u[t - 1] + step * rng.standard_normal(3)
            u[t] = v / np.linalg.norm(v)
        for k, comp in zip("xyz", u.T):
            arr["F"][:, cia.FLOAT_COLS.index("axis_l" + k)] = comp
            arr["F"][:, cia.FLOAT_COLS.index("chi_l" + k)] = comp
            arr["F"][:, cia.FLOAT_COLS.index("pose_" + k)] = comp
        op = np.full(frames, rng.uniform(0.0, 30.0))
        units.append(cia.Unit(i, [], weight=1.0))
        reds.append((cia.reduce_path(arr, op, 0.2, grid), None))
    return grid, cia.Ensemble(units, reds)


def test_reorientation_finds_the_slower_turning_enantiomer():
    grid, eL = _turning_units(0.05, seed=0)
    _, eD = _turning_units(0.4, seed=1)
    df, curves = cia.reorientation(eL, eD, grid, min_pairs=10, n_boot=30, block=3)
    r = df[(df.quantity == "tau_ps") & (df.vector == "axis") & (df.depth == "all contact")].iloc[0]
    assert r.L > 2 * r.D and r.q_bh < 0.05
    assert {"tau_ps", "C(4 ps)", "C(40 ps)"} <= set(df.quantity)
    # C(t) starts near 1 and decays, faster for the larger steps
    cL, cD = curves["C_L"][0, -1], curves["C_D"][0, -1]
    k = int(np.searchsorted(cia.ROT_LAGS, 20))
    assert cL[0] > 0.95 and cD[k] < cL[k]
    # identical turning: no difference over all contact depths (a single depth
    # bin holds only a handful of these toy paths)
    grid, eA = _turning_units(0.05, n=80, seed=3)
    _, eB = _turning_units(0.05, n=80, seed=4)
    df2, _ = cia.reorientation(eA, eB, grid, min_pairs=10, n_boot=50, block=3)
    assert not (df2[df2.depth == "all contact"]["q_bh"] < 0.05).any()


def test_orientation_free_energy_averages_back_to_the_depth_profile():
    grid, units, reds, _ = _stage0_units()
    e = cia.Ensemble(units, reds)
    df, arrays = cia.orientation_free_energy(e, e, grid, 310.0, min_frames=10, n_boot=5, block=2)
    ddg = cia.depth_ddg(e, e, grid, 310.0, n_boot=5, block=2).set_index("depth")["G_L"]
    kt = cia.KB_KCAL * 310.0
    g = arrays["G_L_cos_axis_n"]
    nb = g.shape[1]
    for z, lab in enumerate(grid.depth_labels()):
        if np.isfinite(g[z]).any() and np.isfinite(ddg[lab]):
            back = -kt * np.log(np.nansum(np.exp(-g[z] / kt)) / nb)
            assert back == pytest.approx(ddg[lab], abs=1e-6)
    # the same ensemble on both sides: no difference anywhere
    d = df["ddG_kcal"][df["tested"] == "L-D"]
    assert np.isfinite(d).any()
    np.testing.assert_allclose(d[np.isfinite(d)], 0.0, atol=1e-9)


def test_barrier_depths_and_comparison():
    import pandas as pd

    shares = pd.DataFrame({"depth": ["a", "b", "c", "d"], "share_L": [0.1, 0.5, 0.3, 0.1],
                           "share_D": [0.1, 0.4, 0.4, 0.1]})
    assert cia.barrier_depths(shares) == ["b", "c"]          # b alone holds 0.45 < 0.5
    assert cia.barrier_depths(shares, cover=0.4) == ["b"]
    grid, eL = _axis_units(0.6, seed=0)
    _, eD = _axis_units(0.2, seed=1)
    labels = grid.depth_labels()
    shares = pd.DataFrame({"depth": labels, "share_L": 0.0, "share_D": 0.0})
    shares.loc[shares["depth"] == "2.50-5.00", ["share_L", "share_D"]] = 0.8
    shares.loc[shares["depth"] == "5.00-7.50", ["share_L", "share_D"]] = 0.2
    crossing = (np.arange(eL.n) % 2 == 0, np.arange(eD.n) % 2 == 0)
    t = cia.barrier_comparison(eL, eD, grid, shares, crossing, n_boot=30, block=3)
    assert set(t["paths"]) == {"all paths", "crossing paths"}
    assert t["barrier_depths"].iloc[0] == "2.50-5.00"
    ax = t[(t.observable == "cos_axis_n") & (t.paths == "all paths")].iloc[0]
    assert ax.q_bh < 0.05


def test_new_figures_are_drawn(tmp_path):
    import pandas as pd
    from chiroflux import chiral_plot

    grid, eL = _turning_units(0.05, seed=0)
    _, eD = _turning_units(0.4, seed=1)
    df, curves = cia.reorientation(eL, eD, grid, min_pairs=10, n_boot=10, block=3)
    df.to_csv(tmp_path / "reorientation.csv", index=False)
    np.savez(tmp_path / "reorientation.npz", **curves)
    grid, eA = _axis_units(0.6, seed=0)
    _, eB = _axis_units(0.2, seed=1)
    shares = pd.DataFrame({"depth": grid.depth_labels(), "share_L": 0.07, "share_D": 0.07})
    cia.barrier_comparison(eA, eB, grid, shares, n_boot=10, block=3).to_csv(
        tmp_path / "barrier.csv", index=False)
    g2, units, reds, _ = _stage0_units()
    e = cia.Ensemble(units, reds)
    _, arrays = cia.orientation_free_energy(e, e, g2, 310.0, min_frames=10, n_boot=5, block=2)
    np.savez(tmp_path / "orientation_free_energy.npz", **arrays)
    chiral_plot._style()
    chiral_plot.plot_reorientation(str(tmp_path))
    chiral_plot.plot_barrier(str(tmp_path))
    chiral_plot.plot_orientation_free_energy(str(tmp_path))
    for name in ("chiral_reorientation.png", "chiral_barrier.png", "chiral_orient_fe_cos_axis_n.png"):
        assert (tmp_path / name).exists(), name


def _glycerol_units(centre_b, n=40, frames=80, seed=0):
    """Paths in contact throughout whose touched lipid's dihedral b scatters
    around `centre_b` (deg); every lipid's histogram (G) peaks at 60 deg."""
    rng = np.random.default_rng(seed)
    grid = cia.Grid()
    units, reds = [], []
    bulk = np.zeros((2, cia.N_GLYC_BINS), dtype=np.uint16)
    bulk[:, 24] = 30                                  # 60-70 deg
    for i in range(n):
        arr = _path(frames)
        arr["G"][:] = bulk
        arr["F"][:, cia.FLOAT_COLS.index("glyc_a")] = rng.normal(-60, 15, frames)
        b = rng.normal(centre_b, 15, frames)
        arr["F"][:, cia.FLOAT_COLS.index("glyc_b")] = (b + 180) % 360 - 180
        op = rng.uniform(5.0, 20.0, frames)
        units.append(cia.Unit(i, [], weight=1.0))
        reds.append((cia.reduce_path(arr, op, 0.2, grid), None))
    return grid, cia.Ensemble(units, reds)


def test_glycerol_dihedral_distributions_find_a_shifted_touched_lipid(tmp_path):
    from chiroflux import chiral_plot

    grid, eL = _glycerol_units(60.0, seed=0)
    _, eD = _glycerol_units(180.0, seed=1)
    df, arrays = cia.glycerol_dihedrals(eL, eD, min_frames=50, n_boot=30, block=3)
    # every distribution is normalised; the bulk ones are identical
    np.testing.assert_allclose(np.nansum(arrays["L"], axis=-1), 1.0)
    np.testing.assert_allclose(arrays["L"][1], arrays["D"][1])
    b = df[(df.lipid == "contact") & df.dihedral.str.startswith("b")]
    assert (b.q_bh < 0.05).any()
    assert not (df[df.lipid == "bulk"].q_bh < 0.05).any()
    np.savez(tmp_path / "glycerol_dihedrals.npz", **arrays)
    chiral_plot._style()
    chiral_plot.plot_glycerol_dihedrals(str(tmp_path))
    assert (tmp_path / "chiral_glycerol_dihedrals.png").exists()
