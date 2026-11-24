"""cv-compare: the column plan, the corrections, the binned sums, the
distribution statistics, and the whole command on synthetic runs with
planted differences it must find (and a mirror-image one it must not)."""

import json
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from chiroflux import cv_compare as cc

LAM_M1, LAM_A, LAM_MID, LAM_B = -35.0, -25.0, -20.0, -13.5
CVS = ("Mem_u_def", "Mem_l_def", "PRO_N_hb", "PRO_ang_C_CG", "PRO_hCN_u_P", "noise")


# ── Columns and corrections ────────────────────────────────────────────────

def test_swap_leaflets_is_its_own_inverse():
    assert cc.swap_leaflets("CA_C2_u_DOPC") == "CA_C2_l_DOPC"
    assert cc.swap_leaflets("Mem_l_defDeep") == "Mem_u_defDeep"
    assert cc.swap_leaflets("Mem_thick_loc") == "Mem_thick_loc"     # not a leaflet field
    assert cc.swap_leaflets(cc.swap_leaflets("PRO_hCN_u_P")) == "PRO_hCN_u_P"


def test_plan_lines_b_up_by_its_swapped_names():
    a = ["Mem_u_def", "Mem_l_def", "PRO_ang_C_CG", "PRO_hCN_u_P", "z_PRO", "only_a"]
    b = ["Mem_l_def", "Mem_u_def", "PRO_ang_C_CG", "PRO_hCN_l_P", "z_PRO"]
    plan = cc.plan_columns(a, b, exclude=["z_"], angle_cols=["PRO_ang_C_CG"], swap_b=True,
                           mirror_b=["_def"], flip_b=["PRO_ang"], mirror_null=["hCN"])
    assert plan.names == ["Mem_u_def", "Mem_l_def", "cos(PRO_ang_C_CG)", "PRO_hCN_u_P"]
    assert plan.src_b == ["Mem_l_def", "Mem_u_def", "PRO_ang_C_CG", "PRO_hCN_l_P"]
    assert plan.sym_b == [("mirror",), ("mirror",), ("flip",), ()]
    assert plan.mirror_null.tolist() == [False, False, False, True]
    assert plan.families() == ["Mem", "Mem", "PRO", "PRO"]


def test_anchored_pattern_tells_tilt_from_its_count():
    """'_tilt' is a substring of Mem_u_tiltN, so only '_tilt$' can pick the
    cosine without negating the glycerol count next to it."""
    a = ["Mem_u_tilt", "Mem_u_tiltN", "Mem_l_tilt", "Mem_l_tiltN", "PRO_ang_C_CG"]
    loose = cc.plan_columns(a, a, mirror_b=["_tilt"])
    assert loose.sym_b[:4] == [("mirror",)] * 4                    # the trap
    plan = cc.plan_columns(a, a, angle_cols=["PRO_ang_C_CG"], mirror_b=["_tilt$"],
                           flip_b=["C_CG$"], exclude=["l_tiltN$"])
    assert plan.names == ["Mem_u_tilt", "Mem_u_tiltN", "Mem_l_tilt", "cos(PRO_ang_C_CG)"]
    assert plan.sym_b == [("mirror",), (), ("mirror",), ("flip",)]


def test_matches_pattern():
    from chiroflux.pathdata import _matches_any, _matches_exclude, _matches_pattern
    assert _matches_pattern("Mem_u_tiltN", "_tilt")
    assert not _matches_pattern("Mem_u_tiltN", "_tilt$")
    assert _matches_pattern("Mem_u_tilt", "_tilt$")
    assert _matches_pattern("cos(PRO_ang_C_CG)", "C_CG$")      # through the cos( )
    assert _matches_pattern("cos2(PRO_r_plane)", "plane$")
    assert not _matches_pattern("PRO_r_plane_chiral", "plane$")
    assert not _matches_pattern("anything", "$")                # empty anchor: nothing
    assert _matches_any("z_PRO", ["Box", "z_"]) and not _matches_any("z_PRO", [])
    assert _matches_exclude("Mem_l_tiltN", ["tiltN$"])


def test_mirror_and_flip_in_cvs_honour_the_anchor():
    from chiroflux.cvs import _apply_cv_flip, _apply_cv_mirror
    names = ["Mem_u_tilt", "Mem_u_tiltN", "PRO_ang_C_CG"]
    arr = np.array([[[0.8], [5.0], [30.0]]])
    out, _ = _apply_cv_mirror(arr, names, ["_tilt$"])
    assert out[0, :, 0].tolist() == [-0.8, 5.0, 30.0]
    out, _ = _apply_cv_flip(arr, names, ["C_CG$"])
    assert out[0, :, 0].tolist() == [0.8, 5.0, 150.0]


def _header_file(path, cols):
    path.write_text("# reactive\n# plus ensemble\n# x\nOP_Lamb " + " ".join(cols) + "\n")
    return str(path)


def test_columns_come_from_every_weighted_file_not_the_first(tmp_path, capsys):
    """Stage 1's paths 1000-1004 were written before generate-cvs gained
    'new' (and its unweighted burn-in path 3 never was rewritten); stage 2
    has 'new' nowhere.  Only stage 2 can drop a column."""
    s1 = {pn: _header_file(tmp_path / f"s1_{pn}.txt", ["a", "new"] if pn >= 1005 else ["a"])
          for pn in range(1000, 1010)}
    s2 = {pn: _header_file(tmp_path / f"s2_{pn}.txt", ["a", "both", "late"])
          for pn in range(1, 4)}
    s1b = {pn: _header_file(tmp_path / f"s1b_{pn}.txt",
                            ["a", "both", "late"] if pn != 1000 else ["a", "both"])
           for pn in range(1000, 1004)}
    kept, table = cc.side_header([("entry", s1b), ("internal", s2)], "OP_Lamb", "-cv-dir-a")
    assert kept == ["a", "both", "late"]                 # path 1000 lacking 'late' drops nothing
    assert table[["stage", "path", "missing"]].values.tolist() == [["entry", 1000, "late"]]
    out = capsys.readouterr().out
    assert "entry: 1 of 4 weighted paths have no late" in out and "paths 1000." in out

    kept, table = cc.side_header([("entry", s1), ("internal", s2)], "OP_Lamb", "-cv-dir-a")
    out = capsys.readouterr().out
    assert kept == ["a"]                                 # 'new' is in no file of internal
    assert "internal: new" in out
    assert "5 of 10 weighted paths have no new" not in out   # dropped anyway: not repeated
    # only what entry has in some files: 'new'; 'both'/'late' it has nowhere
    assert table[["stage", "missing"]].drop_duplicates().values.tolist() == [["entry", "new"]]
    assert sorted(table["path"]) == [1000, 1001, 1002, 1003, 1004]
    assert cc._ranges([1, 2, 3, 7, 9, 10]) == "1-3, 7, 9-10"


def test_corrections_on_degrees_and_on_cosines():
    X = np.array([[30.0, 30.0, 170.0, 0.5]])
    cc.transform(X, ["", "cos", "", ""], [("flip",), ("flip",), ("phase",), ("mirror",)])
    assert X[0, 0] == pytest.approx(150.0)                 # 180 - theta
    assert X[0, 1] == pytest.approx(-np.cos(np.deg2rad(30)))  # cos(180 - t) = -cos t
    assert X[0, 2] == pytest.approx(-10.0)                 # 170 + 180 wrapped
    assert X[0, 3] == pytest.approx(-0.5)


def test_value_bins_are_integers_for_counts_and_symmetric_for_mirror_nulls():
    plan = cc.plan_columns(["n", "x", "h"], mirror_null=["h"])
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.integers(0, 4, 500), rng.normal(2, 1, 500),
                         rng.normal(0.4, 0.1, 500)])
    vb = cc.value_bins([X], plan, 10, overrides={"x": (0, 5, 5)})
    assert vb.discrete.tolist() == [True, False, False]
    np.testing.assert_allclose(np.diff(vb.edges[0]), 1.0)
    assert vb.edges[0][0] == -0.5 and vb.edges[0][-1] >= 3.5
    np.testing.assert_allclose(vb.edges[1], np.linspace(0, 5, 6))
    assert vb.edges[2][0] == pytest.approx(-vb.edges[2][-1])
    idx = vb.index(np.array([[1.0, 99.0, np.nan]]))
    assert idx[0, 0] == 1 and idx[0, 1] == vb.offsets[2] - 1 and idx[0, 2] == -1


# ── Reduction ──────────────────────────────────────────────────────────────

def _cfg(n_cv, edges=(-30.0, -25.0, -20.0), wins=(-25.0,)):
    plan = cc.plan_columns([f"c{k}" for k in range(n_cv)])
    vb = cc.ValueBins([np.linspace(-3, 3, 7)] * n_cv, np.zeros(n_cv, bool))
    return cc.SideConfig("OP_Lamb", plan.src_a, plan.angle, None, 1.0,
                         cc.Grid(np.array(edges), np.array(wins)), vb, True)


def test_reduce_path_sums_match_the_frames_and_split_at_lambda_a():
    rng = np.random.default_rng(1)
    op = np.r_[rng.uniform(-29, -25.5, 30), rng.uniform(-24, -21, 50)]
    X = rng.normal(size=(80, 2))
    X[3, 1] = np.nan
    cfg = _cfg(2)
    unit = cc.cia.Unit(0, [], lam_a=-25.0, keep_below=True)
    red = cc.reduce_path(X, op, unit, cfg)
    parts = {p[0]: p for p in red["prof"]}
    _, bins, N, C, S, S2 = parts["below"]
    assert bins.tolist() == [1] and N.tolist() == [30]
    assert C[0].tolist() == [30, 29]
    np.testing.assert_allclose(S[0], np.nansum(X[:30], axis=0))
    np.testing.assert_allclose(S2[0], np.nansum(X[:30] ** 2, axis=0))
    _, bins, N, *_ = parts["above"]
    assert bins.tolist() == [2] and N.tolist() == [50]
    # every finite value lands in exactly one map cell and one histogram bin
    total = sum(m.sum() for m in red["maps"].values())
    assert total == np.isfinite(X).sum()
    assert sum(h[3].sum() for h in red["hist"]) == np.isfinite(X).sum()
    # later stages cut the frames below their lambda_A
    late = cc.reduce_path(X, op, cc.cia.Unit(0, [], lam_a=-25.0, keep_below=False), cfg)
    assert [p[0] for p in late["prof"]] == ["above"]


def test_direction_mask_splits_a_path_and_reversal_swaps_the_two():
    op = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    assert cc.direction_mask(op, "inbound").tolist() == [True, True, False, False, False]
    assert cc.direction_mask(op, "outbound").tolist() == [False, False, False, True, True]
    assert cc.direction_mask(op, "all").all()
    walk = np.cumsum(np.random.default_rng(5).normal(size=200))
    inb, outb = cc.direction_mask(walk, "inbound"), cc.direction_mask(walk, "outbound")
    assert not (inb & outb).any() and (inb | outb).all()
    # the time-reversed path has exactly the other frames inbound
    np.testing.assert_array_equal(cc.direction_mask(walk[::-1], "inbound"), outb[::-1])
    np.testing.assert_array_equal(cc.direction_mask(walk[::-1], "outbound"), inb[::-1])
    assert not cc.direction_mask([3.0], "inbound").any()      # one frame: no direction
    with pytest.raises(ValueError, match="direction"):
        cc.direction_mask(op, "sideways")


def test_reduce_path_keeps_only_the_frames_moving_the_chosen_way():
    op = np.r_[np.linspace(-24, -21, 30), np.linspace(-21.1, -24, 30)]   # in, then back
    X = np.random.default_rng(6).normal(size=(60, 2))
    unit = cc.cia.Unit(0, [], lam_a=-25.0, keep_below=True)
    sums = {}
    for d in cc.DIRECTIONS:
        (_, _, N, _, S, _), = cc.reduce_path(X, op, unit, replace(_cfg(2), direction=d))["prof"]
        sums[d] = (N.sum(), S.sum(0))
    assert sums["inbound"][0] == sums["outbound"][0] == 30
    np.testing.assert_allclose(sums["inbound"][1], X[:30].sum(0))
    np.testing.assert_allclose(sums["outbound"][1], X[30:].sum(0))
    assert sums["inbound"][0] + sums["outbound"][0] == sums["all"][0]


def test_quantiles_and_wasserstein_of_histograms():
    e = np.linspace(0, 10, 11)
    flat = np.full(10, 0.1)
    p10, med, p90 = cc.quantiles(flat, e, [0.1, 0.5, 0.9])
    assert (p10, med, p90) == pytest.approx((1.0, 5.0, 9.0))
    a = np.zeros(10)
    a[2] = 1.0
    b = np.zeros(10)
    b[5] = 1.0
    assert cc.wasserstein(a, b, e) == pytest.approx(3.0)       # moved by three bins
    assert np.isnan(cc.quantiles(np.full(10, np.nan), e, [0.5])[0])


def test_within_window_correlation_ignores_a_common_trend():
    rng = np.random.default_rng(2)
    acc = np.zeros((3, 4, 2, 2))
    for w in range(3):
        x = rng.normal(size=4000)
        y = -0.6 * x + 0.8 * rng.normal(size=4000)              # r = -0.6 inside a window
        X = np.column_stack([x + 10 * w, y + 10 * w])           # both rise along the OP
        M = np.ones_like(X)
        acc[w] = [X.T @ X, X.T @ M, (X ** 2).T @ M, M.T @ M]
    r_win, pooled, n = cc.correlation_matrices(acc)
    assert pooled[0, 1] == pytest.approx(-0.6, abs=0.03)
    np.testing.assert_allclose(r_win[:, 0, 1], -0.6, atol=0.04)
    assert n[0].tolist() == [4000, 4000]


# ── Synthetic runs ─────────────────────────────────────────────────────────

def _write_stage(run, seed, def_shift=0.0, hcn_sign=1.0, hb_reactive=0.0, n=52, frames=40,
                 old=(), turn=False):
    """One entry-like stage: plus paths from lambda_A, every 4th path a [0-]
    path (half ending at lambda_A, half at lambda_-1), CV files in ML_mda.
    def_shift lifts Mem_u_def beyond OP -20; hb_reactive adds H-bonds in the
    reactive paths; hcn_sign sets the sign of the handed CN; the files of
    the paths in `old` have no Mem_l_def (written by an older generate-cvs).
    Plus paths rise monotonically; with `turn` the unreactive ones rise for
    their first half and come back down in the second, as an A -> A path
    does, and the reactive ones recross once (five frames run backwards)."""
    rng = np.random.default_rng(seed)
    for sub in ("ML_mda", "wham", "load"):
        (run / sub).mkdir(parents=True, exist_ok=True)
    (run / "infretis.toml").write_text(
        f"[simulation]\ninterfaces = [{LAM_A}, {LAM_MID}, {LAM_B}]\n"
        f"[simulation.tis_set]\nlambda_minus_one = {LAM_M1}\n[engine]\ntemperature = 310.0\n")
    (run / "wham" / "histo_probability.txt").write_text(
        f"# lm1={LAM_M1}, lA={LAM_A}, lB={LAM_B}\n-30 1\n-20 1\n")
    (run / "wham" / "Pcross.txt").write_text("#lam P-wham P-pm P-wham2\n-13.5 0.01 0.01 0.01\n")
    data = ["# pnr len maxop f- f1 f2 w- w1 w2"]
    for pn in range(n):
        minus = pn % 4 == 3
        if minus:
            op = rng.uniform(LAM_M1 + 0.5, LAM_A - 0.5, frames)
            end = LAM_A + 0.2 if pn % 8 == 3 else LAM_M1 - 0.1
            order = np.r_[LAM_A + 0.1, op, end]
            label, ens = "non-reactive", "minus"
            data.append(f"{pn} {frames} {order.max():.3f} 1.0 ---- ---- 1.0 ---- ----")
        else:
            reactive = rng.random() < 0.4
            top = LAM_B + 0.5 if reactive else rng.uniform(LAM_A + 2, LAM_B - 0.5)
            op = np.sort(rng.uniform(LAM_A + 0.1, top, frames))
            if turn and not reactive:
                op = np.r_[np.sort(op[::2]), np.sort(op[1::2])[::-1]]
            elif turn:
                k = frames // 2
                op[k:k + 5] = op[k:k + 5][::-1]
            order = np.r_[LAM_A - 0.1, op]
            label, ens = ("reactive" if reactive else "non-reactive"), "plus"
            f2 = "1.0" if op.max() >= LAM_MID else "----"
            data.append(f"{pn} {frames} {op.max():.3f} ---- 1.0 {f2} ---- 1.0 {f2}")
        is_r = label == "reactive"
        cols = {
            "Mem_u_def": rng.normal(0, 0.3, frames) + def_shift * (op > LAM_MID),
            "Mem_l_def": rng.normal(0, 0.3, frames),
            "PRO_N_hb": rng.poisson(1.0 + hb_reactive * is_r, frames).astype(float),
            "PRO_ang_C_CG": np.degrees(np.arccos(rng.uniform(-1, 1, frames))),
            "PRO_hCN_u_P": hcn_sign * 0.3 + rng.normal(0, 0.05, frames),
            "noise": rng.normal(5, 1, frames),
        }
        load = run / "load" / str(pn)
        load.mkdir(parents=True, exist_ok=True)
        np.savetxt(load / "order.txt", np.column_stack([np.arange(len(order)), order]))
        if pn in old:
            del cols["Mem_l_def"]
        df = pd.DataFrame({"OP_Lamb": op, **cols})
        with open(run / "ML_mda" / f"{pn}.txt", "w") as fh:
            fh.write(f"# {label}\n# {ens} ensemble\n# {frames} phase points mapped\n")
            df.to_csv(fh, sep=" ", index=False, float_format="%.5f")
    (run / "infretis_data.txt").write_text("\n".join(data) + "\n")


def _run(tmp_path, out, **kw):
    args = dict(runs_b=None, label_a="A", label_b="B", cv_dir_a="ML_mda", cv_dir_b="ML_mda",
                data_a="infretis_data.txt", data_b="infretis_data.txt", toml_a=None,
                toml_b=None, op_col="OP_Lamb", path_numbers=None, nskip=0,
                wham_subdir="wham", pcross_col=3, op_range="auto", op_step=2.5,
                window_step=5.0, cv_cols=None, exclude=None, angle_cols="PRO_ang_C_CG",
                sym_angle_cols=None, hist_bins=12, ranges=None, range_sample=100,
                flip_op_b=False, swap_leaflets_b=False, mirror_b=None, flip_b=None,
                phase_shift_b=None, mirror_null=None, n_boot=60, block="2", seed=0,
                min_frames=50, hist_min_frames=150, correlations=True, unweighted=True,
                workers=2, paths_per_job=10, out_dir=str(tmp_path / out), plot=False,
                overw=False)
    args.update(kw)
    cc.cv_compare(**args)
    return tmp_path / out


@pytest.fixture(scope="module")
def two_sims(tmp_path_factory):
    root = tmp_path_factory.mktemp("sims")
    _write_stage(root / "A" / "infinit_entry", seed=1)
    _write_stage(root / "B" / "infinit_entry", seed=2, def_shift=0.8, hcn_sign=-1.0)
    out = _run(root, "cmp", runs_a=str(root / "A" / "infinit_entry"),
               runs_b=str(root / "B" / "infinit_entry"), label_a="L", label_b="D",
               mirror_null="hCN")
    return root, out


def test_the_planted_difference_ranks_first_and_where_it_was_planted(two_sims):
    _, out = two_sims
    rank = pd.read_csv(out / "ranking.csv")
    top = rank.iloc[0]
    assert top["cv"] == "Mem_u_def" and top["significant_at_max"]
    assert top["d_at_max"] < -1.5                       # B lifted by 0.8 over SD 0.3
    assert top["op_lo_at_max"] >= LAM_MID
    prof = pd.read_csv(out / "profiles.csv")
    u = prof[(prof["cv"] == "Mem_u_def") & prof["q_bh"].notna()]
    assert (u.loc[u["op_lo"] >= LAM_MID, "q_bh"] < 0.05).all()
    assert (u.loc[u["op_hi"] <= LAM_MID, "q_bh"] > 0.05).all()
    np.testing.assert_allclose(u.loc[u["op_lo"] >= LAM_MID, "diff"], -0.8, atol=0.15)


def test_a_mirror_image_cv_is_not_a_difference(two_sims):
    _, out = two_sims
    prof = pd.read_csv(out / "profiles.csv")
    h = prof[(prof["cv"] == "PRO_hCN_u_P") & prof["p_boot"].notna()]
    assert len(h) and (h["tested"] == "A+B").all()
    np.testing.assert_allclose(h["A"], 0.3, atol=0.03)
    np.testing.assert_allclose(h["B"], -0.3, atol=0.03)
    assert (h["q_bh"] > 0.05).all()
    shift = pd.read_csv(out / "shifts.csv")
    s = shift[(shift["cv"] == "PRO_hCN_u_P") & shift["mean_p"].notna()]
    np.testing.assert_allclose(s["mean_diff"], 0.0, atol=0.03)


def test_shift_table_tells_a_shift_from_a_spread(two_sims):
    _, out = two_sims
    shift = pd.read_csv(out / "shifts.csv")
    s = shift[(shift["cv"] == "Mem_u_def") & (shift["win_lo"] >= LAM_MID)
              & shift["mean_p"].notna()]
    assert len(s) and (s["mean_q"] < 0.05).all() and (s["median_q"] < 0.05).all()
    assert (s["sd_q"] > 0.05).all()
    assert (s["w1"] > 0.5).all()


def test_every_folder_and_table_is_written(two_sims):
    root, out = two_sims
    for sub in ("", "reactive", "unreactive", "unweighted", "unweighted/reactive"):
        for name in ("profiles.csv", "distributions.csv", "distributions.npz", "shifts.csv",
                     "ranking.csv", "blocking.csv", "block_choice.csv", "meta.json"):
            assert (out / sub / name).exists(), f"{sub}/{name}"
    assert (out / "correlations.npz").exists()
    assert not (out / "unweighted" / "correlations.npz").exists()
    assert not (out / "stage0_infinit_entry").exists()   # one stage: no per-stage folder
    meta = json.loads((out / "meta.json").read_text())
    assert meta["labels"] == ["L", "D"] and meta["block"] == 2
    cols = pd.read_csv(out / "columns.csv").set_index("cv")
    assert cols.loc["PRO_N_hb", "discrete"] and cols.loc["cos(PRO_ang_C_CG)", "angle"] == "cos"
    classes = pd.read_csv(out / "path_classes.csv")
    assert set(classes["side"]) == {"L", "D"}
    with np.load(out / "distributions.npz") as z:
        n = len(z["names"])
        assert z["A_1d"].shape[1] == z["bin_offsets"][-1] and n == len(CVS)
        assert z["frames_A_win"].shape == (len(z["win_edges"]) + 1, n)   # per window and CV
        for j in range(n):
            sl = slice(z["bin_offsets"][j], z["bin_offsets"][j + 1])
            rows = z["A_1d"][z["frames_A_win"][:, j] > 0]
            np.testing.assert_allclose(np.nansum(rows[:, sl], axis=1), 1.0)
    # rerunning into the same folder needs -O
    with pytest.raises(ValueError, match="already exists"):
        _run(root, "cmp", runs_a=str(root / "A" / "infinit_entry"),
             runs_b=str(root / "B" / "infinit_entry"))


def test_reactive_vs_unreactive_within_one_simulation(tmp_path):
    _write_stage(tmp_path / "S" / "infinit_entry", seed=3, hb_reactive=1.5, n=80)
    out = _run(tmp_path, "cls", runs_a=str(tmp_path / "S" / "infinit_entry"), block="auto",
               unweighted=False)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["labels"] == ["reactive", "unreactive"]
    rank = pd.read_csv(out / "ranking.csv")
    assert rank.iloc[0]["cv"] == "PRO_N_hb" and rank.iloc[0]["d_at_max"] > 0
    assert not (out / "reactive").exists()
    assert (out / "block_choice.csv").exists()


def test_paths_without_a_column_leave_it_to_the_others(tmp_path):
    """The first 20 paths' files predate Mem_l_def: it is still compared,
    from the other paths, and the 20 are listed for regeneration."""
    _write_stage(tmp_path / "S" / "infinit_entry", seed=4, hb_reactive=1.5, n=80,
                 old=range(20))
    out = _run(tmp_path, "old", runs_a=str(tmp_path / "S" / "infinit_entry"),
               unweighted=False)
    assert "Mem_l_def" in pd.read_csv(out / "columns.csv")["cv"].tolist()
    miss = pd.read_csv(out / "paths_missing_columns.csv")
    assert sorted(miss["path"]) == list(range(20)) and (miss["missing"] == "Mem_l_def").all()
    prof = pd.read_csv(out / "profiles.csv")
    hb, ld = (prof[prof["cv"] == c].set_index("op_lo") for c in ("PRO_N_hb", "Mem_l_def"))
    # frames and paths count only those with a value, so the 20 are not counted
    assert (ld["frames_A"] <= hb["frames_A"]).all() and (ld["frames_A"] < hb["frames_A"]).any()
    assert (ld["paths_A"] <= hb["paths_A"]).all() and (ld["paths_A"] < hb["paths_A"]).any()
    assert ld["A"].notna().any()


def test_cv_plot_draws_from_the_tables(two_sims):
    from chiroflux import cv_plot

    _, out = two_sims
    cv_plot.cv_plot(in_dir=str(out / "reactive"), top=1, overw=True)
    for name in ("cv_overview.png", "cv_overview_all.png", "cv_ranking.png", "cv_shape.png",
                 "cv_correlations.png", "cv_blocking.png"):
        assert (out / "reactive" / name).exists(), name
    assert (out / "reactive" / "profiles" / "cv_profiles_Mem.png").exists()
    assert len(os.listdir(out / "reactive" / "distributions")) == 1


def test_direction_splits_every_frame_once_and_is_recorded(tmp_path):
    """Unreactive paths turn back halfway, reactive ones recross briefly:
    inbound and outbound share the frames of both out, with none lost."""
    run = tmp_path / "S" / "infinit_entry"
    _write_stage(run, seed=7, hb_reactive=1.5, n=60, turn=True)
    kw = dict(runs_a=str(run), unweighted=False, correlations=False)
    frames = {}
    for d in cc.DIRECTIONS:
        out = _run(tmp_path, d, direction=d, **kw)
        meta = json.loads((out / "meta.json").read_text())
        assert meta["direction"] == d
        assert (f"{d} frames" in meta["title"]) == (d != "all")
        prof = pd.read_csv(out / "profiles.csv")
        frames[d] = prof[prof["cv"] == "noise"][["frames_A", "frames_B"]].sum()
    for col in ("frames_A", "frames_B"):                  # reactive, unreactive
        assert 0 < frames["outbound"][col] < frames["all"][col]
        assert frames["inbound"][col] + frames["outbound"][col] == frames["all"][col]
    assert frames["outbound"]["frames_A"] < frames["inbound"]["frames_A"]   # recrossings only
    with pytest.raises(Exception, match="-direction"):
        _run(tmp_path, "bad", direction="sideways", **kw)


def test_a_side_left_without_frames_is_skipped(tmp_path, capsys):
    """Paths that only rise have no outbound frames: the folder is skipped
    with a message, not written full of NaN."""
    run = tmp_path / "S" / "infinit_entry"
    _write_stage(run, seed=8, n=40)
    out = _run(tmp_path, "out", runs_a=str(run), unweighted=False, correlations=False,
               direction="outbound")
    assert "no outbound frames in reactive; skipped" in capsys.readouterr().out
    assert not (out / "profiles.csv").exists()


def test_distribution_windows_are_every_stage_window_one_row_per_side():
    from chiroflux import cv_plot

    edges = np.array([-35, -30, -25, -20, -15, -13.5, -10, -5, 0, 5, 10, 13.5, 15, 20, 25.0])
    lo, hi = np.r_[-np.inf, edges], np.r_[edges, np.inf]
    rows = [(label, [(lo[w], hi[w]) for w in ws])
            for label, ws in cv_plot.window_rows(edges, [-25, -13.5, 13.5, 25])]
    assert rows == [
        ("OP < 0", [(-25, -20), (-20, -15), (-15, -13.5), (-13.5, -10), (-10, -5), (-5, 0)]),
        ("OP > 0", [(0, 5), (5, 10), (10, 13.5), (13.5, 15), (15, 20), (20, 25)])]
    # a finer grid wraps each side onto a second row
    fine = cv_plot.window_rows(np.arange(-25, 25.1, 2.5), [-25, 25])
    assert [len(ws) for _, ws in fine] == [6, 4, 6, 4]
    # the midplane is a grid edge even when the regular steps miss it
    assert 0.0 in cc.make_edges(-33.0, 25.0, 5.0, [-25.0, cc.MIDPLANE])


def _blas_threads(_):
    from threadpoolctl import threadpool_info
    return max((p["num_threads"] for p in threadpool_info()), default=1)


def test_workers_run_one_blas_thread_each():
    """Forked workers inherit the parent's per-core BLAS pool; the
    initializer must bring each down to one thread."""
    from concurrent.futures import ProcessPoolExecutor
    pytest.importorskip("threadpoolctl")
    with ProcessPoolExecutor(max_workers=2, initializer=cc._one_blas_thread) as ex:
        assert set(ex.map(_blas_threads, range(4))) == {1}


# ── Indicators: populations of a state ─────────────────────────────────────

def test_indicator_forms_and_their_bounds():
    anti = cc.parse_indicator("PRO_anti=abs(PRO_dih_OH)>90")
    (c,) = anti.clauses
    assert (anti.name, c.column, c.absolute, c.lo, c.hi, anti.flip) == \
        ("PRO_anti", "PRO_dih_OH", True, 90.0, None, False)
    np.testing.assert_array_equal(anti([np.array([0.0, 120.0, -170.0, 90.0, np.nan])]),
                                  [0.0, 1.0, 1.0, 0.0, np.nan])
    basin = cc.parse_indicator("basin = -107 < PRO_CP_phi2 <= 41")
    (c,) = basin.clauses
    assert (c.lo, c.lo_incl, c.hi, c.hi_incl) == (-107.0, False, 41.0, True)
    np.testing.assert_array_equal(basin([np.array([-107.0, -30.0, 41.0, 60.0])]), [0, 1, 1, 0])
    low = cc.parse_indicator("low=x<=1e-3")
    assert (low.clauses[0].lo, low.clauses[0].hi, low.clauses[0].hi_incl) == (None, 1e-3, True)
    for bad in ("abs(x)>90", "a=x", "a=x>>1", "a=5<x<2", "a=3>2", "1a=x>0", "a=x>0&",
                "a=flip()", "a=x>0&&y<1"):
        with pytest.raises(ValueError):
            cc.parse_indicator(bad)


def test_a_joint_state_needs_every_test_and_is_unknown_without_a_value():
    both = cc.parse_indicator("endo_cis=PRO_dih_chi1>0&abs(PRO_dih_NH)<90")
    assert both.columns == ("PRO_dih_chi1", "PRO_dih_NH") and not both.flip
    chi = np.array([30.0, 30.0, -30.0, -30.0, 30.0])
    nh = np.array([40.0, 150.0, 40.0, -150.0, np.nan])
    np.testing.assert_array_equal(both([chi, nh]), [1.0, 0.0, 0.0, 0.0, np.nan])


def test_flip_counts_the_frames_where_the_state_changes():
    flips = cc.parse_indicator("pucker_flips = flip(PRO_dih_chi1>0)")
    assert flips.flip and flips.columns == ("PRO_dih_chi1",)
    chi = np.array([30.0, 25.0, -20.0, -35.0, 28.0, 31.0])
    np.testing.assert_array_equal(flips([chi]), [0.0, 1.0, 0.0, 1.0, 0.0, np.nan])
    assert np.nanmean(flips([chi])) == pytest.approx(2 / 5)       # per frame step
    joint = cc.parse_indicator("f=flip(x>0&y>0)")
    np.testing.assert_array_equal(joint([np.array([1.0, 1.0, 1.0]), np.array([1.0, -1.0, -1.0])]),
                                  [1.0, 0.0, np.nan])


def test_indicator_takes_its_cvs_corrections_and_comes_after_the_cvs():
    a = ["PRO_dih_chi1", "Mem_u_def", "noise"]
    b = ["PRO_dih_chi1", "Mem_l_def", "noise"]
    endo = cc.parse_indicator("PRO_endo=PRO_dih_chi1>0")
    up = cc.parse_indicator("up=Mem_u_def>0")
    both = cc.parse_indicator("endo_up=PRO_dih_chi1>0&Mem_u_def>0")
    plan = cc.plan_columns(a, b, exclude=["PRO_dih_"], swap_b=True,
                           mirror_b=["PRO_dih_", "_def"], indicators=[endo, up, both])
    assert plan.names == ["Mem_u_def", "noise", "PRO_endo", "up", "endo_up"]
    assert plan.src_a[:2] == ["Mem_u_def", "noise"] and plan.src_a[2:] == [None] * 3
    # every CV a clause needs is read once, with its own B corrections
    assert plan.aux_a == ["PRO_dih_chi1", "Mem_u_def"]
    assert plan.aux_b == ["PRO_dih_chi1", "Mem_l_def"]
    assert plan.aux_sym == [("mirror",), ("mirror",)]
    assert [s[1] for s in plan.indicator[2:]] == [(0,), (1,), (0, 1)]
    assert plan.sym_b[2:] == [(), (), ()]
    # B's chi1 and def are negated before the tests, so they mean the same in both
    X = np.zeros((3, 5))
    aux = np.array([[-40.0, 0.1], [40.0, -0.1], [np.nan, 0.1]])
    cc.transform(X, plan.angle, plan.sym_b, plan.indicator, aux, plan.aux_sym)
    np.testing.assert_array_equal(X[:, 2], [1.0, 0.0, np.nan])
    np.testing.assert_array_equal(X[:, 3], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(X[:, 4], [0.0, 0.0, np.nan])
    # side A: no corrections
    XA = np.zeros((3, 5))
    cc.transform(XA, plan.angle, None, plan.indicator, aux, None)
    np.testing.assert_array_equal(XA[:, 2], [0.0, 1.0, np.nan])
    # -cv-cols may list the indicator; it is added either way
    only = cc.plan_columns(a, cv_cols=["noise", "PRO_endo"], indicators=[endo])
    assert only.names == ["noise", "PRO_endo"]
    with pytest.raises(Exception, match="already the name"):
        cc.plan_columns(a, indicators=[cc.parse_indicator("noise=noise>0")])
    with pytest.raises(Exception, match="not a column"):
        cc.plan_columns(a, indicators=[cc.parse_indicator("x=noise>0&nothere>0")])


def test_indicator_bins_are_zero_and_one_whatever_the_ranges_say():
    plan = cc.plan_columns(["PRO_dih_OH"], indicators=[
        cc.parse_indicator("anti=abs(PRO_dih_OH)>90")])
    X = np.column_stack([np.linspace(-170, 170, 50), np.zeros(50)])   # no anti in the sample
    vb = cc.value_bins([X], plan, 12, overrides={"PRO_dih_OH": (-180, 180, 360)})
    assert len(vb.edges[0]) == 361
    np.testing.assert_array_equal(vb.edges[1], [-0.5, 0.5, 1.5])
    assert vb.discrete.tolist() == [False, True]


def test_indicator_population_along_the_path(two_sims):
    """B's Mem_u_def is lifted by 0.8 past LAM_MID, so P(Mem_u_def > 0.4)
    goes from ~0.09 to ~0.91 there; PRO_hCN_u_P is +0.3 in A and -0.3 in B,
    so with -mirror-b its sign is the same state on both sides.  The noise
    column is drawn afresh every frame, so its state flips at about
    2 p (1 - p) of the frame steps, the same on both sides."""
    root, _ = two_sims
    out = _run(root, "cmp_ind", runs_a=str(root / "A" / "infinit_entry"),
               runs_b=str(root / "B" / "infinit_entry"), label_a="L", label_b="D",
               mirror_b="hCN",
               indicators="lifted=Mem_u_def>0.4,pos=PRO_hCN_u_P>0,"
                          "lifted_pos=Mem_u_def>0.4&PRO_hCN_u_P>0,hops=flip(noise>5)",
               unweighted=False)
    cols = pd.read_csv(out / "columns.csv").set_index("cv")
    assert cols.loc["lifted", "indicator"] == "lifted=Mem_u_def>0.4"
    assert cols.loc["pos", "corrections_B"] == "PRO_hCN_u_P:mirror"
    assert cols.loc["lifted_pos", "column_A"] == "Mem_u_def&PRO_hCN_u_P"
    assert cols.loc["pos", "discrete"]
    prof = pd.read_csv(out / "profiles.csv")
    lifted = prof[(prof["cv"] == "lifted") & prof["q_bh"].notna()]
    past = lifted[lifted["op_lo"] >= LAM_MID]
    np.testing.assert_allclose(past["A"], 0.09, atol=0.05)
    np.testing.assert_allclose(past["B"], 0.91, atol=0.05)
    assert (past["q_bh"] < 0.05).all()
    np.testing.assert_allclose(lifted.loc[lifted["op_hi"] <= LAM_MID, "diff"], 0.0, atol=0.06)
    pos = prof[(prof["cv"] == "pos") & prof["p_boot"].notna()]
    np.testing.assert_allclose(pos[["A", "B"]].to_numpy(), 1.0, atol=1e-9)
    joint = prof[(prof["cv"] == "lifted_pos") & prof["q_bh"].notna()]
    np.testing.assert_allclose(joint[["A", "B"]].to_numpy(),
                               lifted.set_index("op_lo").loc[joint["op_lo"], ["A", "B"]]
                               .to_numpy(), atol=1e-9)
    hops = prof[(prof["cv"] == "hops") & prof["p_boot"].notna()]
    for side in ("A", "B"):
        assert np.average(hops[side], weights=hops[f"frames_{side}"]) == \
            pytest.approx(0.5, abs=0.03)
    dense = hops[(hops["frames_A"] >= 250) & (hops["frames_B"] >= 250)]
    assert len(dense) >= 2 and not (dense["q_bh"] < 0.05).any()
