"""Reactive vs unreactive membrane distortion: -paths and membrane-compare.

What has to hold: a path's class comes back from its file, [0-] paths are
never counted as unreactive, a -paths subset aggregates exactly the paths of
that class, and the comparison's class means agree with that aggregation. The
error on the difference is the joint block jackknife - checked against a case
small enough to do by hand - and bins one class barely reaches are left out
rather than compared.
"""

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
        cls_of, _, _ = mc.classify_paths(1, 21, weights,
                                         ("unreactive", "reactive"), paths)
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
        cls_of, _, _ = mc.classify_paths(1, 21, {pn: 1.0 for pn in range(1, 22)},
                                         ("unreactive", "reactive"), paths)
        assert 21 not in cls_of

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
        cls_of, _, _ = mc.classify_paths(1, 21, weights,
                                         ("unreactive", "reactive"), paths)
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
        cls_of, _, _ = mc.classify_paths(1, 22, weights,
                                         ("unreactive", "reactive"), paths)
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
        weights = paths / "w.txt"
        weights.write_text("".join(f"{pn} 1.0\n" for pn in range(1, 22)))
        args = dict(weights=str(weights), data_dir=str(paths), a="unreactive",
                    b="reactive", start=1, end=21, n_blocks=5, min_neff=2.0,
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
            self._run(paths, tmp_path / "cmp", a="minus", end=20)


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
