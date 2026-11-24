"""Reactive vs unreactive membrane distortion: -paths and membrane-compare.

What has to hold: a path's class comes back from its file, [0-] paths are
never counted as unreactive, a -paths subset aggregates exactly the paths of
that class, and the comparison's class means agree with that aggregation. The
error on the difference is the joint block jackknife - checked against a case
small enough to do by hand - and bins one class barely reaches are left out
rather than compared.
"""

import os

import numpy as np
import pandas as pd
import pytest
import typer

from chiroflux import membrane_compare as mc
from chiroflux import membrane_spatial as ms

N_SLABS = 8
AXIS = np.arange(N_SLABS, dtype=float) - N_SLABS / 2 + 0.5


def _table(values, count=1.0, name="thickness"):
    """A slab table whose mean in every bin is `values` (scalar or per bin)."""
    values = np.broadcast_to(np.asarray(values, dtype=float), (N_SLABS,))
    count = np.broadcast_to(np.asarray(count, dtype=float), (N_SLABS,))
    return pd.DataFrame({"slab_center": AXIS,
                         f"{name}_wsum": values * count,
                         f"{name}_wtot": count})


def _spatial(value, n_depth=2, nx=3, ny=3, x=None):
    m = nx * ny
    acc = {"thickness": [np.full((n_depth, m), float(value)),
                         np.ones((n_depth, m))]}
    x = np.linspace(-5.0, 5.0, nx) if x is None else x
    y = np.linspace(-5.0, 5.0, ny)
    return acc, x, y, np.linspace(-10.0, 10.0, n_depth + 1)


DEPTH = np.array([-5.0, 5.0])        # two coarse permeant-depth bins
RADII = (1.0, 3.0)


def _depth_tables(value):
    """deform_depth / water_z_depth whose every cell has mean `value`."""
    v = float(np.mean(value))
    deform = {"depth_center": DEPTH}
    for r in RADII:
        col = ms.bin_column("def_upper", "r", r)
        deform[f"{col}_wsum"] = np.full(len(DEPTH), v)
        deform[f"{col}_wtot"] = np.ones(len(DEPTH))
    col = ms.bin_column("water_z", "z", -1.5)
    water = {"depth_center": DEPTH, f"{col}_wsum": np.full(len(DEPTH), v),
             f"{col}_wtot": np.ones(len(DEPTH))}
    return {"deform_depth": pd.DataFrame(deform),
            "water_z_depth": pd.DataFrame(water)}


def _write(out_dir, pn, cls, value, count=1.0, spatial=None):
    meta = {"ensemble": "minus" if cls == "minus" else "plus",
            "reactive": cls == "reactive"}
    tables = {"thick": _table(value, count), "p2": _table(value, count,
                                                          "sn2_p2_chain")}
    tables.update(_depth_tables(value))
    ms.write_path_output(ms.path_output_file(pn, out_dir), tables=tables,
                         spatial=spatial, meta=meta)


class TestPathClass:
    @pytest.mark.parametrize("meta, expected", [
        ({"ensemble": "plus", "reactive": True}, "reactive"),
        ({"ensemble": "plus", "reactive": False}, "unreactive"),
        ({"ensemble": "minus", "reactive": False}, "minus"),
        ({}, None),
    ])
    def test_the_class_follows_the_metadata(self, meta, expected):
        assert ms.path_class_of(meta) == expected

    def test_a_minus_path_is_never_unreactive(self):
        """It never reaches lambda_B, but it is another ensemble."""
        assert ms.path_class_of({"ensemble": "minus", "reactive": False}) == "minus"

    def test_the_class_comes_back_from_the_file(self, tmp_path):
        _write(tmp_path, 1, "reactive", 1.0)
        meta = ms.read_path_meta(ms.path_output_file(1, tmp_path))
        assert meta == {"ensemble": "plus", "reactive": True}
        assert ms.path_class_of(meta) == "reactive"

    def test_no_file_has_no_class(self, tmp_path):
        assert ms.read_path_meta(tmp_path / "9.npz") == {}

    def test_an_unknown_subset_is_refused_before_any_work(self):
        with pytest.raises(typer.BadParameter):
            ms.parse_path_subsets("reactive,reactiv")
        assert ms.parse_path_subsets(" all, reactive ") == ("all", "reactive")

    @pytest.mark.parametrize("stored, is_minus, expected", [
        ("plus", True, "minus"),            # a swapped [0-] path, old file
        ("minus", False, "unreactive"),
        ("minus", None, "minus"),           # no WHAM list: the file decides
    ])
    def test_wham_membership_overrides_the_stored_ensemble(
            self, stored, is_minus, expected):
        meta = {"ensemble": stored, "reactive": False}
        assert ms.path_class_of(meta, is_minus) == expected

    def test_a_reactive_path_wrongly_stored_as_minus_is_reactive(self):
        meta = {"ensemble": "minus", "reactive": True}
        assert ms.path_class_of(meta, False) == "reactive"


class TestPathEnsemble:
    """membrane-spatial read the ensemble off order.txt after dropping the
    .g96 row; a swapped path starts or ends on it."""
    LM1, LA, LB = -35.0, -25.0, -13.5

    def _aligned(self, order, g96_step):
        plan = [("traj.xtc", 0)] * (len(order) - 1)
        out, problem = ms.align_order_with_plan(order, plan, str(g96_step))
        assert problem is None
        return out

    def test_a_swapped_minus_path_stays_minus(self):
        # [0-] path made by a swap: it ends on the .g96 point beyond lambda_A,
        # and the frame before that lies between lambda_-1 and lambda_A.
        order = np.array([[0, -40.0], [1, -33.0], [2, -27.0], [3, -24.0]])
        assert ms.path_ensemble(order, self.LM1, self.LA) == "minus"
        cut = self._aligned(order, 3)
        assert ms.path_ensemble(cut, self.LM1, self.LA) == "plus"   # the bug

    def test_a_swapped_reactive_plus_path_stays_plus(self):
        # [0+] path made by a swap: it starts on the .g96 point below lambda_A
        # and ends beyond lambda_B, so without that row no endpoint is below
        # lambda_A.
        order = np.array([[0, -26.0], [1, -24.0], [2, -18.0], [3, -12.0]])
        assert ms.path_ensemble(order, self.LM1, self.LA) == "plus"
        cut = self._aligned(order, 0)
        assert ms.path_ensemble(cut, self.LM1, self.LA) == "minus"  # the bug

    @pytest.mark.parametrize("ends, expected", [
        ((-40.0, -24.0), "minus"), ((-24.0, -24.0), "minus"),
        ((-40.0, -40.0), "minus"), ((-26.0, -24.0), "plus"),
        ((-26.0, -12.0), "plus"), ((-26.0, -26.0), "plus"),
    ])
    def test_the_endpoint_rule(self, ends, expected):
        order = np.array([[0, ends[0]], [1, -30.0], [2, ends[1]]])
        assert ms.path_ensemble(order, self.LM1, self.LA) == expected


class TestMinusList:
    def test_it_is_read_next_to_the_weights_file(self, tmp_path):
        wham = tmp_path / "wham"
        wham.mkdir()
        (wham / "path_weights.txt").write_text("1 0.5\n2 0.5\n3 0.5\n")
        (wham / ms.MINUS_WEIGHTS_FILE).write_text(
            "#path_label, weight, sum_weight: {1.0}\n2      1.0e-03\n")
        minus = ms.load_minus_paths(str(wham / "path_weights.txt"))
        assert minus == {2}
        assert "lists" in ms.describe_minus_source(
            str(wham / "path_weights.txt"), minus)

    def test_the_newer_wham_header_is_skipped(self, tmp_path):
        (tmp_path / ms.MINUS_WEIGHTS_FILE).write_text(
            "#Sum of path types:{'LML': 0, 'RML': 0, 'LMR': 0, 'RMR': 2}\n"
            "Sum of path count:2\n"
            "#path_label, weight, sum_weight: {1.0}\n"
            "17      5.0e-01\n42      5.0e-01\n")
        assert ms.load_minus_paths(str(tmp_path / "path_weights.txt")) == {17, 42}

    def test_without_it_the_files_decide_and_it_says_so(self, tmp_path):
        w = tmp_path / "path_weights.txt"
        w.write_text("1 1.0\n")
        assert ms.load_minus_paths(str(w)) is None
        assert ms.describe_minus_source(str(w), None).startswith("WARNING")


class TestSelectPaths:
    @pytest.fixture
    def four(self, tmp_path):
        for pn, cls in zip((1, 2, 3, 4), ("reactive", "unreactive", "minus",
                                          "reactive")):
            _write(tmp_path, pn, cls, float(pn))
        return tmp_path

    def test_all_is_the_weights_unchanged(self, four):
        w = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
        assert ms.select_paths(w, 1, 4, "all", out_dir=four) is w

    def test_a_class_keeps_only_its_paths(self, four):
        w = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
        assert ms.select_paths(w, 1, 4, "reactive", out_dir=four) == {1: 1.0, 4: 4.0}
        assert ms.select_paths(w, 1, 4, "unreactive", out_dir=four) == {2: 2.0}
        assert ms.select_paths(w, 1, 4, "minus", out_dir=four) == {3: 3.0}

    def test_the_wham_minus_list_decides(self, four):
        """Path 2 is stored unreactive but WHAM lists it as [0-]."""
        w = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
        minus = {2, 3}
        assert ms.select_paths(w, 1, 4, "unreactive", out_dir=four,
                               minus=minus) == {}
        assert ms.select_paths(w, 1, 4, "minus", out_dir=four,
                               minus=minus) == {2: 2.0, 3: 3.0}

    def test_unclassified_paths_are_counted_and_left_out(self, four, capsys):
        ms.write_path_output(ms.path_output_file(5, four),
                             tables={"thick": _table(1.0)})
        w = {pn: 1.0 for pn in range(1, 6)}
        kept = ms.select_paths(w, 1, 5, "reactive", out_dir=four)
        assert 5 not in kept
        assert "1 weighted path(s) have no class" in capsys.readouterr().out

    def test_a_subset_aggregates_exactly_its_own_paths(self, four, monkeypatch):
        monkeypatch.setattr(ms, "ORDER_OUT", str(four))
        tables = ms.read_path_tables(ms.path_output_file(1, four))
        for pn in range(1, 5):            # every table, so the aggregation runs
            full = {k: tables["thick"] for k in ms._TABLE_KEYS}
            full["thick"] = _table(float(pn))
            ms.write_path_output(ms.path_output_file(pn, four), tables=full)
        w = {1: 1.0, 2: 1.0, 3: 1.0, 4: 3.0}
        sub = ms.select_paths(w, 1, 4, "reactive")
        thick = ms.aggregate_all_results(1, 4, sub, n_blocks=2)[2]
        # paths 1 (value 1, weight 1) and 4 (value 4, weight 3)
        np.testing.assert_allclose(thick["thickness"], (1 + 4 * 3) / 4)


class TestJointJackknife:
    def test_the_difference_and_its_error_by_hand(self):
        """Blocks {1 a, 2 b} and {3 a, 4 b}; one bin, one observable.

        a: 2/1 and 4/1 -> 3.  b: 6/1 and 12/2 -> 6.  delta = 3.
        Leaving block 0 out: 4 vs 6 -> 2; block 1 out: 2 vs 6 -> 4.
        Jackknife: sqrt((K-1)/K * sum (d_k - mean)^2) = sqrt(1/2 * 2) = 1.
        """
        S = {"a": np.array([2.0, 4.0]).reshape(2, 1, 1),
             "b": np.array([6.0, 12.0]).reshape(2, 1, 1)}
        T = {"a": np.array([1.0, 1.0]).reshape(2, 1, 1),
             "b": np.array([1.0, 2.0]).reshape(2, 1, 1)}
        Q = {c: (T[c] ** 2).sum(axis=0) for c in T}
        res = mc.compare_sums(S, T, Q, "a", "b", min_neff=0.0, t_crit=2.0)
        assert res["mean_a"].item() == pytest.approx(3.0)
        assert res["mean_b"].item() == pytest.approx(6.0)
        assert res["delta"].item() == pytest.approx(3.0)
        assert res["delta_err"].item() == pytest.approx(1.0)
        assert res["t"].item() == pytest.approx(3.0)
        assert res["significant"].item()

    def test_a_class_in_one_block_gets_no_error(self):
        """Its leave-one-out spread is all from one block: no error at all."""
        S = {"a": np.array([2.0, 4.0]).reshape(2, 1, 1),
             "b": np.array([6.0, 0.0]).reshape(2, 1, 1)}
        T = {"a": np.ones((2, 1, 1)),
             "b": np.array([1.0, 0.0]).reshape(2, 1, 1)}
        Q = {c: (T[c] ** 2).sum(axis=0) for c in T}
        res = mc.compare_sums(S, T, Q, "a", "b", min_neff=0.0, t_crit=2.0)
        assert np.isfinite(res["delta"]).all()
        assert np.isnan(res["delta_err"]).all()
        assert not res["significant"].any()

    def test_a_thin_bin_is_not_compared(self):
        S = {"a": np.ones((2, 1, 1)), "b": np.ones((2, 1, 1))}
        T = {"a": np.ones((2, 1, 1)),
             "b": np.array([1.0, 1e-6]).reshape(2, 1, 1)}   # n_eff ~ 1
        Q = {c: (T[c] ** 2).sum(axis=0) for c in T}
        res = mc.compare_sums(S, T, Q, "a", "b", min_neff=1.5, t_crit=2.0)
        assert np.isnan(res["mean_b"]).all()
        assert np.isnan(res["delta"]).all()


class TestFromFiles:
    @pytest.fixture
    def paths(self, tmp_path):
        _fill(tmp_path)
        return tmp_path

    def _sums(self, paths, weights):
        cls_of, _ = mc.classify_paths(weights, ("unreactive", "reactive"),
                                      paths)
        pns = sorted(cls_of)
        block_of, K = ms._assign_blocks(pns, 5)
        centers, _, obs = mc._scan(pns, "thick", paths)
        S, T, Q = mc._block_sums(pns, block_of, K, cls_of, weights,
                                 ("unreactive", "reactive"), "thick",
                                 centers, obs, paths)
        return mc.compare_sums(S, T, Q, "unreactive", "reactive", 2.0, 2.8), obs

    def test_a_real_difference_is_found(self, paths):
        res, obs = self._sums(paths, {pn: 1.0 for pn in range(1, 22)})
        j = obs.index("thickness")
        np.testing.assert_allclose(res["delta"][:, j], 2.0, atol=0.15)
        assert res["significant"][:, j].all()

    def test_minus_paths_are_in_neither_class(self, paths):
        cls_of, _ = mc.classify_paths({pn: 1.0 for pn in range(1, 22)},
                                      ("unreactive", "reactive"), paths)
        assert 21 not in cls_of

    def test_the_wham_minus_list_decides_the_classes(self, paths):
        weights = {pn: 1.0 for pn in range(1, 22)}
        cls_of, info = mc.classify_paths(weights, ms.PATH_CLASSES, paths,
                                         minus={2, 21})
        assert cls_of[2] == "minus"           # stored unreactive
        assert cls_of[21] == "minus"
        assert info["relabelled"] == 1
        _, info = mc.classify_paths(weights, ms.PATH_CLASSES, paths)
        assert info["relabelled"] == 0

    def test_every_weighted_file_in_the_folder_is_used(self, paths):
        """No path range: the folder and the weights decide. A file without a
        weight, a weighted path without a file, a temporary file and the old
        layout are each handled without a range to set."""
        (paths / "5.npz.tmp").write_bytes(b"partial")
        (paths / "7_spatial.npz").write_bytes(b"old layout")
        weights = {pn: 1.0 for pn in range(3, 22) if pn != 4}
        weights[40] = 2.0                                 # weighted, no file
        cls_of, info = mc.classify_paths(weights, ("unreactive", "reactive"),
                                         paths)
        assert sorted(cls_of) == [pn for pn in range(3, 21) if pn != 4]
        assert info == {"files": 21, "no_weight": 3, "unknown": 0,
                        "no_file": 1, "relabelled": 0}

    def test_a_missing_folder_is_refused(self, tmp_path):
        with pytest.raises(typer.BadParameter, match="no such folder"):
            mc.classify_paths({1: 1.0}, ("reactive",), tmp_path / "nope")

    def test_the_class_mean_is_what_membrane_spatial_paths_gives(
            self, paths, monkeypatch):
        """-paths reactive and membrane-compare must agree on the mean."""
        rng = np.random.default_rng(1)
        weights = {pn: float(rng.uniform(0.1, 5.0)) for pn in range(1, 22)}
        res, obs = self._sums(paths, weights)
        monkeypatch.setattr(ms, "ORDER_OUT", str(paths))
        sub = ms.select_paths(weights, 1, 21, "reactive")
        pns = sorted(sub)
        dfs = [ms._weighted(ms.read_path_tables(ms.path_output_file(p))["thick"],
                            sub[p]) for p in pns]
        ws, wt = ms._wsum_wtot_cols(dfs)
        agg = ms._pool_weighted_means(dfs, ws, wt, path_numbers=pns)
        np.testing.assert_allclose(res["mean_reactive"][:, obs.index("thickness")],
                                   agg["thickness"], rtol=1e-6)

    def test_maps_are_compared_cell_by_cell(self, paths):
        weights = {pn: 1.0 for pn in range(1, 22)}
        cls_of, _ = mc.classify_paths(weights, ("unreactive", "reactive"),
                                      paths)
        pns = sorted(cls_of)
        block_of, K = ms._assign_blocks(pns, 5)
        sums = mc.map_block_sums(pns, block_of, K, cls_of, weights,
                                 ("unreactive", "reactive"), list(mc.DEFAULT_MAP_OBS), paths)
        assert sums["names"] == ["thickness"]
        res = mc.compare_sums(sums["S"], sums["T"], sums["Q"], "unreactive",
                              "reactive", 2.0, 2.8)
        np.testing.assert_allclose(res["delta"], 2.0)

    def test_a_map_on_another_grid_is_left_out(self, paths):
        _write(paths, 22, "reactive", 3.0,
               spatial=_spatial(99.0, x=np.linspace(-9.0, 9.0, 3)))
        weights = {pn: 1.0 for pn in range(1, 23)}
        cls_of, _ = mc.classify_paths(weights, ("unreactive", "reactive"),
                                      paths)
        pns = sorted(cls_of)
        block_of, K = ms._assign_blocks(pns, 5)
        sums = mc.map_block_sums(pns, block_of, K, cls_of, weights,
                                 ("unreactive", "reactive"), list(mc.DEFAULT_MAP_OBS), paths)
        assert sums["skipped"] == 1
        res = mc.compare_sums(sums["S"], sums["T"], sums["Q"], "unreactive",
                              "reactive", 2.0, 2.8)
        np.testing.assert_allclose(res["mean_reactive"], 3.0)


class TestCommand:
    def _run(self, paths, out, **kw):
        """`paths` is the per-path folder; its parent is the run directory."""
        (paths / "w.txt").write_text("".join(f"{pn} 1.0\n" for pn in range(1, 22)))
        args = dict(run_a=str(paths.parent), run_b=None, data_dir=paths.name,
                    weights=f"{paths.name}/w.txt", a="unreactive",
                    b="reactive", label_a="A", label_b="B", paths="all",
                    rotate_b=False, n_blocks=5, min_neff=2.0,
                    alpha=0.05, unweighted=False, map_memory=2.0,
                    out_dir=str(out), map_obs="default", overw=False)
        args.update(kw)
        mc.membrane_compare(**args)

    def test_it_writes_the_tables_ranking_and_figures(self, tmp_path):
        paths = tmp_path / "Memb_OP"
        _fill(paths)
        out = tmp_path / "cmp"
        self._run(paths, out, unweighted=True)
        for name in ("thick_compare.csv", "p2_compare.csv", "paths.csv",
                     "deform_depth_compare.csv", "water_z_depth_compare.csv",
                     "distortion_ranking.csv", "overview_t.png",
                     "compare_thickness.png", "compare_sn2_p2_chain.png",
                     "compare_def_upper_vs_depth.png",
                     "compare_water_z_vs_depth.png",
                     "spatial_compare.npz", "map_ranking.csv",
                     "map_compare_thickness.png"):
            assert (out / name).exists(), name
        assert (out / "unweighted" / "distortion_ranking.csv").exists()
        rank = pd.read_csv(out / "distortion_ranking.csv")
        assert set(rank["observable"]) == {
            "thickness", "sn2_p2_chain", "def_upper_r1.00", "def_upper_r3.00",
            "water_z_z-1.50"}
        slab = rank["table"].isin(["thick", "p2"])
        assert (rank.loc[slab, "significant_bins"] == N_SLABS).all()
        assert (rank.loc[~slab, "significant_bins"] == len(DEPTH)).all()
        assert set(pd.read_csv(out / "paths.csv")["class"]) == {"reactive",
                                                                "unreactive"}

    def test_existing_output_needs_O(self, tmp_path):
        paths = tmp_path / "Memb_OP"
        _fill(paths)
        out = tmp_path / "cmp"
        self._run(paths, out)
        with pytest.raises(ValueError):
            self._run(paths, out)
        self._run(paths, out, overw=True)

    @pytest.mark.parametrize("a, b", [("reactive", "reactive"),
                                      ("all", "reactive")])
    def test_bad_classes_are_refused(self, tmp_path, a, b):
        with pytest.raises(typer.BadParameter):
            self._run(tmp_path, tmp_path / "cmp", a=a, b=b)

    def test_a_class_without_paths_is_refused(self, tmp_path):
        paths = tmp_path / "Memb_OP"
        _fill(paths)
        with pytest.raises(typer.BadParameter, match="minus"):
            self._run(paths, tmp_path / "cmp", a="minus",
                      weights=_weights_file(paths, range(1, 21)))


def _weights_file(folder, pns):
    """A weights file in `folder` giving weight 1 to `pns`, and only to them;
    returned as -weights takes it, relative to the run directory above."""
    (folder / "w_subset.txt").write_text("".join(f"{pn} 1.0\n" for pn in pns))
    return f"{folder.name}/w_subset.txt"


def _fill(out_dir):
    """20 paths alternating reactive (thickness ~3) and unreactive (~1), and
    one [0-] path far off both, which must stay out of either class."""
    rng = np.random.default_rng(0)
    for pn in range(1, 21):
        cls = "reactive" if pn % 2 else "unreactive"
        base = 3.0 if cls == "reactive" else 1.0
        _write(out_dir, pn, cls, base + 0.1 * rng.standard_normal(N_SLABS),
               spatial=_spatial(base))
    _write(out_dir, 21, "minus", 50.0)


class TestBinColumns:
    @pytest.mark.parametrize("name, axis, center", [
        ("def_upper", "r", 3.0), ("water_radial", "r", 0.5),
        ("water_z", "z", -39.333), ("water_z", "z", 12.5)])
    def test_the_bin_comes_back_from_the_name(self, name, axis, center):
        assert ms.split_bin_column(ms.bin_column(name, axis, center)) == \
            (name, axis, round(center, 2))

    def test_minus_zero_is_written_as_zero(self):
        assert ms.bin_column("water_z", "z", -0.0) == "water_z_z0.00"

    def test_other_columns_are_not_bins(self):
        for col in ("thickness", "def_upper", "def_upper_r1.00_neff",
                    "D_z_lag2"):
            assert ms.split_bin_column(col) is None

    def test_the_aggregate_reads_back_as_a_depth_by_radius_matrix(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(ms, "ORDER_OUT", str(tmp_path))
        for pn, v in ((1, 1.0), (2, 3.0)):
            _write(tmp_path, pn, "reactive", v)
        agg = ms.aggregate_radial_depth(1, 2, {1: 1.0, 2: 3.0}, n_blocks=2)
        x, cols, vals = ms.depth_binned_matrix(agg["deform_depth"], "def_upper")
        assert x.tolist() == list(RADII)
        np.testing.assert_allclose(vals, (1.0 + 3 * 3.0) / 4)
        _, _, neff = ms.depth_binned_matrix(agg["deform_depth"], "def_upper",
                                            "_neff")
        assert neff.shape == vals.shape
        assert ms.plot_depth_binned(agg["deform_depth"], "def_upper", "d", "r",
                                    output_path=str(tmp_path / "d.png"))
        assert (tmp_path / "d.png").exists()

    def test_files_without_the_depth_binned_tables_aggregate_to_nothing(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(ms, "ORDER_OUT", str(tmp_path))
        ms.write_path_output(ms.path_output_file(1, tmp_path),
                             tables={"thick": _table(1.0)})
        assert ms.aggregate_radial_depth(1, 1, {1: 1.0}) == {}


class TestMapSelection:
    AVAILABLE = ["p2_sn2_chain", "p2_sn2_chain_upper", "p2_sn2_bond_0_DOPC",
                 "thickness", "thickness_POPC_lower", "phi_sn1"]

    def test_default_is_the_curated_set(self):
        assert mc.select_map_names("default", self.AVAILABLE) == \
            ["p2_sn2_chain", "p2_sn2_chain_upper", "thickness"]

    def test_all_is_everything_in_file_order(self):
        assert mc.select_map_names("all", self.AVAILABLE) == self.AVAILABLE

    def test_patterns_pick_by_name(self):
        assert mc.select_map_names("thickness*, *_DOPC", self.AVAILABLE) == \
            ["p2_sn2_bond_0_DOPC", "thickness", "thickness_POPC_lower"]

    def test_chunks_respect_the_memory_and_keep_every_name(self):
        names = [f"o{i}" for i in range(10)]
        shape, K = (10, 169), 20
        per_obs = (9 * K + 6) * 10 * 169 * 8
        chunks = mc.map_chunks(names, shape, K, memory_gb=3.5 * per_obs / 1e9)
        assert [len(c) for c in chunks] == [3, 3, 3, 1]
        assert sum(chunks, []) == names
        assert mc.map_chunks(names, shape, K, memory_gb=0.0) == \
            [[n] for n in names]

    def test_chunked_is_identical_to_one_pass(self, tmp_path):
        rng = np.random.default_rng(3)
        names = ["thickness", "curvature_H", "p2_sn2_chain"]
        for pn in range(1, 13):
            cls = "reactive" if pn % 2 else "unreactive"
            acc = {n: [rng.random((2, 9)), rng.random((2, 9)) + 0.5]
                   for n in names}
            ms.write_path_output(
                ms.path_output_file(pn, tmp_path),
                spatial=(acc, np.linspace(-5, 5, 3), np.linspace(-5, 5, 3),
                         np.array([-10.0, 0.0, 10.0])),
                meta={"ensemble": "plus", "reactive": cls == "reactive"})
        cls_of = {pn: ("reactive" if pn % 2 else "unreactive")
                  for pn in range(1, 13)}
        w_of = {pn: float(pn) for pn in cls_of}
        pns = sorted(cls_of)
        block_of, K = ms._assign_blocks(pns, 4)
        got = {}
        for tag, mem in (("one", 10.0), ("chunked", 1e-9)):
            out = tmp_path / tag
            out.mkdir()
            mc._compare_maps(pns, block_of, K, cls_of, w_of, "unreactive",
                             "reactive", 0.0, 2.0, 0.05, names, mem,
                             tmp_path, str(out), False)
            got[tag] = dict(np.load(out / "spatial_compare.npz"))
        assert got["one"].keys() == got["chunked"].keys()
        for key in got["one"]:
            np.testing.assert_array_equal(got["one"][key], got["chunked"][key])

    def test_a_pattern_matching_nothing_is_refused_up_front(self, tmp_path):
        paths = tmp_path / "Memb_OP"
        _fill(paths)
        with pytest.raises(typer.BadParameter, match="matches none"):
            TestCommand()._run(paths, tmp_path / "cmp", map_obs="nosuch*")
        assert not (tmp_path / "cmp").exists()

    def test_none_skips_the_maps(self, tmp_path):
        paths = tmp_path / "Memb_OP"
        _fill(paths)
        TestCommand()._run(paths, tmp_path / "cmp", map_obs="none")
        assert not (tmp_path / "cmp" / "spatial_compare.npz").exists()
        assert (tmp_path / "cmp" / "distortion_ranking.csv").exists()


# ═════════════════════════════════════════════════════════════════════════════
# Two simulations (-dir-b)
# ═════════════════════════════════════════════════════════════════════════════

class TestRotatedObservable:
    @pytest.mark.parametrize("name, expected", [
        ("thickness", ("thickness", 0.0, 1.0)),
        ("upper_thickness", ("lower_thickness", 0.0, 1.0)),
        ("H_upper", ("H_lower", 0.0, 1.0)),                  # stored as |H|
        ("upper_DOPC_sn2_p2_chain", ("lower_DOPC_sn2_p2_chain", 0.0, 1.0)),
        ("lower_sn1_theta", ("upper_sn1_theta", 0.0, 1.0)),  # vs inward normal
        ("near5_sn2_phi", ("near5_sn2_phi", 180.0, -1.0)),
        ("phi_sn2_POPC_upper", ("phi_sn2_POPC_lower", 180.0, -1.0)),
        ("curvature_H_lower", ("curvature_H_upper", 0.0, 1.0)),
        ("def_upper_r3.00", ("def_lower_r3.00", 0.0, -1.0)),
        ("water_radial_r1.00", ("water_radial_r1.00", 0.0, 1.0)),
        ("water_z_z-39.33", ("water_z_z39.33", 0.0, 1.0)),
        ("water_z_z0.00", ("water_z_z0.00", 0.0, 1.0)),
        ("D_z_lag2", ("D_z_lag2", 0.0, 1.0)),
    ])
    def test_each_kind_of_observable(self, name, expected):
        assert mc.rotated_observable(name) == expected

    @pytest.mark.parametrize("name", ["K_upper", "DOPC_K_upper", "H_radial"])
    def test_upper_leaflet_only_quantities_have_no_counterpart(self, name):
        assert mc.rotated_observable(name) is None

    @pytest.mark.parametrize("name", ["upper_sn2_phi", "def_lower_r1.00",
                                      "water_z_z-1.50", "thickness_DOPC_upper"])
    def test_rotating_twice_is_the_identity(self, name):
        once, off1, s1 = mc.rotated_observable(name)
        twice, off2, s2 = mc.rotated_observable(once)
        assert twice == name
        # v -> off1 + s1 v -> off2 + s2 (off1 + s1 v) must be v again
        assert off2 + s2 * off1 == 0.0 and s1 * s2 == 1.0


class TestIndependentRuns:
    def test_the_error_adds_in_quadrature_with_each_runs_own_blocks(self):
        """a: 2 blocks, 2/1 and 4/1 -> err 1 (see the joint test above).
        b: 3 blocks, sums 6/1, 12/2, 3/1 -> 21/4; leave-one-out 5, 4.5, 6,
        so err_b² = 2/3 * 7/6 = 7/9. err = sqrt(1 + 7/9) = 4/3."""
        S = {"a": np.array([2.0, 4.0]).reshape(2, 1, 1),
             "b": np.array([6.0, 12.0, 3.0]).reshape(3, 1, 1)}
        T = {"a": np.ones((2, 1, 1)),
             "b": np.array([1.0, 2.0, 1.0]).reshape(3, 1, 1)}
        Q = {c: (T[c] ** 2).sum(axis=0) for c in T}
        res = mc.compare_sums(S, T, Q, "a", "b", min_neff=0.0, t_crit=2.0,
                              independent=True)
        assert res["delta"].item() == pytest.approx(21 / 4 - 3)
        assert res["err_b"].item() == pytest.approx(np.sqrt(7 / 9))
        assert res["delta_err"].item() == pytest.approx(4 / 3)

    def test_each_run_is_blocked_along_its_own_path_numbers(self):
        cls_of = {("L", pn): "L" for pn in range(100, 110)}
        cls_of.update({("D", pn): "D" for pn in range(1, 5)})
        pns = sorted(cls_of)
        block_of, K, dof = mc.assign_blocks(pns, cls_of, "L", "D", 5, True)
        assert K == {"L": 5, "D": 4} and dof == 3
        got = {p: k for p, k in zip(pns, block_of)}
        assert [got[("D", pn)] for pn in range(1, 5)] == [0, 1, 2, 3]
        assert [got[("L", pn)] for pn in range(100, 110)] == \
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


# A run whose depth profile, leaflets, signs and lateral maps are all
# asymmetric, and the same run seen after the 180° turn about x - written out
# by hand here, not with the module's own transform, so that the test checks
# the transform against the physics rather than against itself.

def _a_tables(z, d):
    return {
        "thick": pd.DataFrame({
            "slab_center": z,
            "upper_thickness_wsum": 1.0 + 0.1 * z, "upper_thickness_wtot": 1.0 + 0 * z,
            "lower_thickness_wsum": 2.0 - 0.05 * z, "lower_thickness_wtot": 1.0 + 0 * z,
            "K_upper_wsum": 0.3 + 0 * z, "K_upper_wtot": 1.0 + 0 * z}),
        "tilt": pd.DataFrame({
            "slab_center": z,
            "upper_sn2_phi_wsum": 30.0 + z, "upper_sn2_phi_wtot": 1.0 + 0 * z,
            "lower_sn2_phi_wsum": 100.0 + 0 * z, "lower_sn2_phi_wtot": 1.0 + 0 * z}),
        "deform_depth": pd.DataFrame({
            "depth_center": d,
            "def_upper_r1.00_wsum": 0.3 + 0.01 * d, "def_upper_r1.00_wtot": 1.0 + 0 * d,
            "def_lower_r1.00_wsum": -0.2 + 0 * d, "def_lower_r1.00_wtot": 1.0 + 0 * d}),
        "water_z_depth": pd.DataFrame({
            "depth_center": d,
            "water_z_z-1.50_wsum": 1.0 + 0.01 * d, "water_z_z-1.50_wtot": 1.0 + 0 * d,
            "water_z_z1.50_wsum": 5.0 + 0 * d, "water_z_z1.50_wtot": 1.0 + 0 * d}),
    }


def _b_tables(z, d):
    """The same membrane after (x, y, z) -> (x, -y, -z): B at depth z' sees
    what A saw at -z', with the leaflets traded."""
    A = _a_tables(-z, -d)
    one = 1.0 + 0 * z
    return {
        "thick": pd.DataFrame({
            "slab_center": z,
            "lower_thickness_wsum": A["thick"]["upper_thickness_wsum"], "lower_thickness_wtot": one,
            "upper_thickness_wsum": A["thick"]["lower_thickness_wsum"], "upper_thickness_wtot": one,
            "K_upper_wsum": 99.0 + 0 * z, "K_upper_wtot": one}),   # A's unseen lower K
        "tilt": pd.DataFrame({
            "slab_center": z,
            "lower_sn2_phi_wsum": 180.0 - A["tilt"]["upper_sn2_phi_wsum"], "lower_sn2_phi_wtot": one,
            "upper_sn2_phi_wsum": 180.0 - A["tilt"]["lower_sn2_phi_wsum"], "upper_sn2_phi_wtot": one}),
        "deform_depth": pd.DataFrame({
            "depth_center": d,
            "def_lower_r1.00_wsum": -A["deform_depth"]["def_upper_r1.00_wsum"],
            "def_lower_r1.00_wtot": 1.0 + 0 * d,
            "def_upper_r1.00_wsum": -A["deform_depth"]["def_lower_r1.00_wsum"],
            "def_upper_r1.00_wtot": 1.0 + 0 * d}),
        "water_z_depth": pd.DataFrame({
            "depth_center": d,
            "water_z_z1.50_wsum": A["water_z_depth"]["water_z_z-1.50_wsum"],
            "water_z_z1.50_wtot": 1.0 + 0 * d,
            "water_z_z-1.50_wsum": A["water_z_depth"]["water_z_z1.50_wsum"],
            "water_z_z-1.50_wtot": 1.0 + 0 * d}),
    }


_NX = _NY = 3
_EDGES = np.array([-10.0, 0.0, 10.0])
_GRID = np.linspace(-5.0, 5.0, 3)


def _a_maps():
    """(n_depth, nx, ny) maps that differ along every axis."""
    d, i, j = np.meshgrid(np.arange(2), np.arange(_NX), np.arange(_NY),
                          indexing="ij")
    return {"phi_sn2_upper": 10.0 * d + 3.0 * i + j,
            "thickness_lower": 40.0 + 5.0 * d - i + 2.0 * j}


def _b_maps():
    A = _a_maps()
    flip = lambda m: m[::-1, :, ::-1]      # depth and y reversed, x kept
    return {"phi_sn2_lower": 180.0 - flip(A["phi_sn2_upper"]),
            "thickness_upper": flip(A["thickness_lower"])}


def _write_run(out_dir, tables, maps, n=6):
    acc = {k: [v.reshape(2, _NX * _NY), np.ones((2, _NX * _NY))]
           for k, v in maps.items()}
    for pn in range(1, n + 1):
        ms.write_path_output(ms.path_output_file(pn, out_dir), tables=tables,
                             spatial=(acc, _GRID, _GRID, _EDGES),
                             meta={"ensemble": "plus", "reactive": True})


class TestRotation:
    @pytest.fixture
    def runs(self, tmp_path):
        z = AXIS.copy()
        d = DEPTH.copy()
        _write_run(tmp_path / "A", _a_tables(z, d), _a_maps())
        _write_run(tmp_path / "B", _b_tables(z, d), _b_maps())
        return tmp_path

    def _compare(self, runs, kind, rotate):
        where = {}
        cls_of, w_of = {}, {}
        for label, rot in (("A", False), ("B", rotate)):
            c, w, src, _ = mc.collect_run(label, runs / label,
                                          {pn: 1.0 for pn in range(1, 7)},
                                          ms.PATH_CLASSES, rot)
            cls_of.update(c), w_of.update(w), where.update(src)
        pns = sorted(cls_of)
        block_of, K, _ = mc.assign_blocks(pns, cls_of, "A", "B", 3, True)
        centers, _, obs = mc._scan(pns, kind, where)
        S, T, Q = mc._block_sums(pns, block_of, K, cls_of, w_of, ("A", "B"),
                                 kind, centers, obs, where)
        return mc.compare_sums(S, T, Q, "A", "B", 2.0, 2.0, True), obs, \
            (pns, block_of, K, cls_of, w_of, where)

    @pytest.mark.parametrize("kind", ["thick", "tilt", "deform_depth",
                                      "water_z_depth"])
    def test_a_rotated_copy_lands_on_the_original(self, runs, kind):
        res, obs, _ = self._compare(runs, kind, rotate=True)
        for j, o in enumerate(obs):
            if o == "K_upper":
                continue
            np.testing.assert_allclose(res["mean_B"][:, j], res["mean_A"][:, j],
                                       err_msg=o)
            np.testing.assert_allclose(res["delta"][:, j], 0.0, atol=1e-12,
                                       err_msg=o)

    def test_the_upper_leaflet_gaussian_curvature_is_not_compared(self, runs):
        res, obs, _ = self._compare(runs, "thick", rotate=True)
        assert "K_upper" in obs                            # A has it
        assert np.isnan(res["delta"][:, obs.index("K_upper")]).all()
        assert not any(o == "K_lower" for o in obs)

    def test_without_rotation_the_same_membrane_reads_as_different(self, runs):
        res, obs, _ = self._compare(runs, "thick", rotate=False)
        j = obs.index("upper_thickness")
        assert np.nanmax(np.abs(res["delta"][:, j])) > 0.5

    def test_rotated_maps_land_on_the_original(self, runs):
        _, _, (pns, block_of, K, cls_of, w_of, where) = \
            self._compare(runs, "thick", rotate=True)
        names = ["phi_sn2_upper", "thickness_lower"]
        assert mc.spatial_names(pns, where)[0] == names
        sums = mc.map_block_sums(pns, block_of, K, cls_of, w_of, ("A", "B"),
                                 names, where)
        assert sums["skipped"] == 0
        res = mc.compare_sums(sums["S"], sums["T"], sums["Q"], "A", "B", 2.0,
                              2.0, True)
        np.testing.assert_allclose(res["mean_B"], res["mean_A"])
        np.testing.assert_allclose(res["mean_A"][0].reshape(2, 3, 3),
                                   _a_maps()["phi_sn2_upper"])

    def test_an_asymmetric_map_grid_cannot_be_rotated(self):
        sp = {"names": ["thickness"], "wsum": np.zeros((1, 2, 9)),
              "wcount": np.ones((1, 2, 9)), "x_coords": _GRID,
              "y_coords": np.array([0.0, 5.0, 10.0]), "depth_edges": _EDGES}
        with pytest.raises(ValueError, match="symmetric"):
            mc._rotate_spatial(sp)


def _fill_run(out_dir, base, seed):
    rng = np.random.default_rng(seed)
    for pn in range(1, 21):
        cls = "reactive" if pn % 2 else "unreactive"
        _write(out_dir, pn, cls, base + 0.1 * rng.standard_normal(N_SLABS),
               spatial=_spatial(base))


class TestTwoRunCommand:
    def _run(self, tmp_path, base_b=1.0, weighted_b=range(1, 21), **kw):
        """Two runs laid out as on disk: <run>/analysis/Memb_OP and
        <run>/wham/path_weights.txt, found through the -dir/-weights defaults.
        All 20 paths of L are weighted, `weighted_b` of D."""
        for run, base, seed, pns in (("L", 1.0, 0, range(1, 21)),
                                     ("D", base_b, 1, weighted_b)):
            _fill_run(tmp_path / run / mc.DEFAULT_DATA_DIR, base, seed)
            w = tmp_path / run / mc.DEFAULT_WEIGHTS
            w.parent.mkdir(parents=True, exist_ok=True)
            w.write_text("".join(f"{pn} 1.0\n" for pn in pns))
        args = dict(run_a=str(tmp_path / "L"), run_b=str(tmp_path / "D"),
                    data_dir=mc.DEFAULT_DATA_DIR, weights=mc.DEFAULT_WEIGHTS,
                    a=None, b=None, label_a="L", label_b="D", paths="all", rotate_b=False, n_blocks=5, min_neff=2.0,
                    alpha=0.05, unweighted=False, map_memory=2.0,
                    out_dir=str(tmp_path / "cmp"), map_obs="default",
                    overw=False)
        args.update(kw)
        mc.membrane_compare(**args)
        return tmp_path / "cmp"

    def test_a_shifted_run_differs_everywhere(self, tmp_path):
        out = self._run(tmp_path, base_b=2.0, paths="unreactive")
        rank = pd.read_csv(out / "distortion_ranking.csv")
        slab = rank["table"].isin(["thick", "p2"])
        assert (rank.loc[slab, "significant_bins"] == N_SLABS).all()
        thick = pd.read_csv(out / "thick_compare.csv")
        np.testing.assert_allclose(thick["thickness_delta"], 1.0, atol=0.15)
        listing = pd.read_csv(out / "paths.csv")
        assert set(listing["class"]) == {"L", "D"}
        assert {"file", "rotated"} <= set(listing.columns)
        assert len(listing) == 20                          # 10 unreactive each
        assert (out / "map_compare_thickness.png").exists()

    def test_the_same_run_twice_gives_no_difference(self, tmp_path):
        out = self._run(tmp_path, base_b=1.0, paths="unreactive")
        thick = pd.read_csv(out / "thick_compare.csv")
        assert np.abs(thick["thickness_delta"]).max() < 0.15
        assert not thick["thickness_significant"].any()

    @pytest.mark.parametrize("kw, match", [
        ({"a": "reactive"}, "-paths"),
        ({"label_b": "L"}, "label"),
        ({"paths": "reactiv"}, "reactiv"),
        ({"weights": "/abs/path_weights.txt"}, "relative"),
        ({"data_dir": "/abs/Memb_OP"}, "relative"),
        ({"weights": "wham/other.txt"}, "no such weights file"),
        ({"data_dir": "analysis/nothing"}, "no such folder"),
        ({"run_b": "/no/such/run"}, "no such run directory"),
    ])
    def test_bad_two_run_options_are_refused(self, tmp_path, kw, match):
        with pytest.raises(typer.BadParameter, match=match):
            self._run(tmp_path, **kw)

    def test_dir_and_weights_are_read_inside_each_run(self, tmp_path):
        """One -dir and one -weights, resolved in both runs: B's own weights
        file decides B's paths, and B's files come from B's folder."""
        out = self._run(tmp_path, paths="unreactive", weighted_b=range(1, 11))
        listing = pd.read_csv(out / "paths.csv")
        assert (listing["class"] == "L").sum() == 10       # even 2..20
        assert (listing["class"] == "D").sum() == 5        # even 2..10
        files = listing.set_index("class")["file"]
        assert files["L"].str.startswith(
            str(tmp_path / "L" / mc.DEFAULT_DATA_DIR)).all()
        assert files["D"].str.startswith(
            str(tmp_path / "D" / mc.DEFAULT_DATA_DIR)).all()

    def test_each_runs_wham_minus_list_is_used(self, tmp_path, capsys):
        """D's WHAM lists paths 2 and 4 as [0-]: they leave D's unreactive
        class although their files say unreactive. L has no list, so its
        files decide, and the output says so."""
        wham_d = tmp_path / "D" / os.path.dirname(mc.DEFAULT_WEIGHTS)
        wham_d.mkdir(parents=True)
        (wham_d / ms.MINUS_WEIGHTS_FILE).write_text("2 0.5\n4 0.5\n")
        out = self._run(tmp_path, paths="unreactive")
        listing = pd.read_csv(out / "paths.csv")
        assert (listing["class"] == "L").sum() == 10
        assert (listing["class"] == "D").sum() == 8
        text = capsys.readouterr().out
        assert "L: WARNING" in text
        assert "2 weighted path(s) classed differently" in text

    @pytest.mark.parametrize("kw", [{"rotate_b": True}, {"paths": "reactive"}])
    def test_two_run_options_need_run_b(self, tmp_path, kw):
        with pytest.raises(typer.BadParameter, match="-run-b"):
            self._run(tmp_path, run_b=None, **kw)
