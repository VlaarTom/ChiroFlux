"""One compressed .npz per path replaces seven CSVs and a spatial .npz.

What has to hold: tables and spatial maps come back as written (to the 7
significant digits the CSVs carried), a partial recompute merges into the file
instead of wiping the groups it did not touch, a reader never sees a half-
written file, and the aggregation over paths gives the same numbers it did
from the CSVs.
"""

import numpy as np
import pandas as pd
import pytest

from chiroflux import membrane_spatial as ms
from chiroflux.membrane_spatial import (
    _needed_groups,
    aggregate_all_results,
    aggregate_spatial_maps,
    path_output_contents,
    path_output_file,
    read_path_columns,
    read_path_spatial,
    read_path_tables,
    write_path_output,
)

ALL_TABLES = ("p2", "tilt", "thick", "deform", "water_z", "diff", "diff_xy")


def _table(axis_name, n_rows, columns, rng, scale=1.0):
    df = pd.DataFrame(rng.random((n_rows, len(columns))) * scale,
                      columns=columns)
    df.insert(0, axis_name, np.arange(n_rows, dtype=float) - n_rows / 2 + 0.5)
    return df


def _all_tables(rng, n_slabs=8):
    """The seven tables with the axes and wsum/wtot pairing they really have."""
    slab_cols = ["a_wsum", "a_wtot", "b_wsum", "b_wtot"]
    return {
        "p2":      _table("slab_center", n_slabs, slab_cols, rng),
        "tilt":    _table("slab_center", n_slabs, slab_cols, rng),
        "thick":   _table("slab_center", n_slabs, slab_cols, rng),
        "deform":  _table("r_center", 5, ["def_upper_wsum", "def_upper_wtot"], rng),
        "water_z": _table("z_center", 6, ["water_z_wsum", "water_z_wtot"], rng),
        "diff":    _table("slab_center", n_slabs, ["D_z_wsum", "D_z_wtot"], rng),
        "diff_xy": _table("slab_center", n_slabs, ["D_xy_wsum", "D_xy_wtot"], rng),
    }


def _spatial(rng, n_depth=3, nx=4, ny=4, names=("p2_sn1_chain", "thickness")):
    m = nx * ny
    acc = {n: [rng.random((n_depth, m)), rng.random((n_depth, m))] for n in names}
    x = np.linspace(-15.0, 15.0, nx)
    y = np.linspace(-15.0, 15.0, ny)
    depth = np.linspace(-40.0, 40.0, n_depth + 1)
    return acc, x, y, depth


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "ORDER_OUT", str(tmp_path))
    return tmp_path


class TestRoundTrip:
    def test_tables_come_back_as_written(self, tmp_path):
        rng = np.random.default_rng(0)
        tables = _all_tables(rng)
        path = tmp_path / "1.npz"
        write_path_output(path, tables=tables)
        back = read_path_tables(path)
        assert set(back) == set(ALL_TABLES)
        for key, df in tables.items():
            assert back[key].columns.tolist() == df.columns.tolist()
            np.testing.assert_allclose(back[key].to_numpy(), df.to_numpy(),
                                       rtol=1e-6)

    def test_the_axis_is_stored_exactly(self, tmp_path):
        """It is matched across paths by rounding, so it must not drift."""
        axis = np.linspace(-40.0, 40.0, 61)                  # not float32-exact
        df = pd.DataFrame({"z_center": axis, "w_wsum": np.ones(61),
                           "w_wtot": np.ones(61)})
        path = tmp_path / "1.npz"
        write_path_output(path, tables={"water_z": df})
        assert np.array_equal(read_path_tables(path)["water_z"]["z_center"], axis)

    def test_float32_keeps_what_the_csvs_kept(self, tmp_path):
        """The CSVs were %.6e: 7 significant digits, which float32 holds."""
        values = np.array([1.234567e-12, 9.876543e3, -3.141593e-1, 7.0e30])
        df = pd.DataFrame({"slab_center": np.arange(4.0),
                           "x_wsum": values, "x_wtot": values})
        path = tmp_path / "1.npz"
        write_path_output(path, tables={"diff": df})
        back = read_path_tables(path)["diff"]["x_wsum"].to_numpy()
        as_csv = np.array([float(f"{v:.6e}") for v in values])
        np.testing.assert_allclose(back, as_csv, rtol=1e-6)

    def test_the_spatial_maps_come_back_as_written(self, tmp_path):
        rng = np.random.default_rng(1)
        acc, x, y, depth = _spatial(rng)
        path = tmp_path / "1.npz"
        write_path_output(path, spatial=(acc, x, y, depth))
        sp = read_path_spatial(path)
        assert sp["names"] == list(acc)
        for i, name in enumerate(acc):
            np.testing.assert_allclose(sp["wsum"][i], acc[name][0], rtol=1e-6)
            np.testing.assert_allclose(sp["wcount"][i], acc[name][1], rtol=1e-6)
        assert np.array_equal(sp["x_coords"], x)
        assert np.array_equal(sp["depth_edges"], depth)

    def test_metadata_is_kept(self, tmp_path):
        path = tmp_path / "1.npz"
        write_path_output(path, tables=_all_tables(np.random.default_rng(2)),
                          meta={"reactive": True, "ensemble": "plus"})
        with np.load(path) as data:
            assert bool(data["meta__reactive"]) is True
            assert str(data["meta__ensemble"]) == "plus"
            assert int(data["meta__version"]) == ms.PATH_FILE_VERSION

    def test_it_is_one_file(self, tmp_path):
        rng = np.random.default_rng(3)
        write_path_output(tmp_path / "7.npz", tables=_all_tables(rng),
                          spatial=_spatial(rng))
        assert [p.name for p in tmp_path.iterdir()] == ["7.npz"]

    def test_the_file_is_named_after_the_path(self, out_dir):
        assert path_output_file(42) == out_dir / "42.npz"


class TestPartialUpdates:
    def test_a_recomputed_group_merges_into_the_file(self, tmp_path):
        """Recomputing only the diffusion must not lose the order parameters."""
        rng = np.random.default_rng(4)
        tables = _all_tables(rng)
        path = tmp_path / "1.npz"
        write_path_output(path, tables={k: tables[k] for k in ("p2", "tilt")},
                          spatial=_spatial(rng))
        write_path_output(path, tables={"diff": tables["diff"]})
        assert path_output_contents(path) == {"p2", "tilt", "diff", "spatial"}
        np.testing.assert_allclose(read_path_tables(path, ["p2"])["p2"].to_numpy(),
                                   tables["p2"].to_numpy(), rtol=1e-6)

    def test_a_rewritten_group_replaces_the_old_one(self, tmp_path):
        rng = np.random.default_rng(5)
        path = tmp_path / "1.npz"
        first = _all_tables(rng)["diff"]
        second = first.copy()
        second.iloc[:, 1:] *= 3.0
        write_path_output(path, tables={"diff": first})
        write_path_output(path, tables={"diff": second})
        np.testing.assert_allclose(read_path_tables(path, ["diff"])["diff"].to_numpy(),
                                   second.to_numpy(), rtol=1e-6)

    def test_new_columns_are_not_mixed_with_the_old_schema(self, tmp_path):
        """A rewritten table takes its columns from the new frame only."""
        rng = np.random.default_rng(6)
        path = tmp_path / "1.npz"
        write_path_output(path, tables={"p2": _table("slab_center", 4,
                                                     ["old_wsum", "old_wtot"], rng)})
        write_path_output(path, tables={"p2": _table("slab_center", 4,
                                                     ["new_wsum", "new_wtot"], rng)})
        assert read_path_columns(path, "p2") == ["slab_center", "new_wsum", "new_wtot"]

    def test_no_temporary_file_is_left_behind(self, tmp_path):
        write_path_output(tmp_path / "1.npz",
                          tables=_all_tables(np.random.default_rng(7)))
        assert not list(tmp_path.glob("*.tmp"))

    def test_an_unknown_table_is_refused(self, tmp_path):
        df = _table("slab_center", 3, ["x_wsum", "x_wtot"], np.random.default_rng(8))
        with pytest.raises(ValueError, match="Unknown table"):
            write_path_output(tmp_path / "1.npz", tables={"p3": df})


class TestMissingAndBrokenFiles:
    def test_a_missing_file_holds_nothing(self, tmp_path):
        path = tmp_path / "nope.npz"
        assert path_output_contents(path) == set()
        assert read_path_tables(path) == {}
        assert read_path_spatial(path) is None
        assert read_path_columns(path, "p2") == []

    def test_a_corrupt_file_is_treated_as_absent_and_rewritten(self, tmp_path):
        path = tmp_path / "1.npz"
        path.write_bytes(b"not a zip file")
        with pytest.warns(UserWarning, match="unreadable"):
            assert path_output_contents(path) == set()
        with pytest.warns(UserWarning, match="unreadable"):
            write_path_output(path, tables=_all_tables(np.random.default_rng(9)))
        assert path_output_contents(path) == set(ALL_TABLES)

    def test_a_file_without_spatial_maps_says_so(self, tmp_path):
        path = tmp_path / "1.npz"
        write_path_output(path, tables=_all_tables(np.random.default_rng(10)))
        assert read_path_spatial(path) is None


class TestNeededGroups:
    def test_a_missing_file_needs_everything(self, tmp_path):
        assert _needed_groups(tmp_path / "1.npz", overwrite=False) == \
            set(ms.OBSERVABLE_GROUPS)

    def test_overwrite_needs_everything_even_when_complete(self, tmp_path):
        rng = np.random.default_rng(11)
        path = tmp_path / "1.npz"
        write_path_output(path, tables=_all_tables(rng), spatial=_spatial(rng))
        assert _needed_groups(path, overwrite=False) == set()
        assert _needed_groups(path, overwrite=True) == set(ms.OBSERVABLE_GROUPS)

    def test_only_the_absent_groups_are_needed(self, tmp_path):
        rng = np.random.default_rng(12)
        tables = _all_tables(rng)
        path = tmp_path / "1.npz"
        write_path_output(path, tables={k: tables[k] for k in
                                        ("p2", "tilt", "thick", "deform", "water_z")})
        assert _needed_groups(path, overwrite=False) == \
            {"diffusion", "diffusion_xy", "spatial"}

    def test_a_group_with_one_of_its_tables_missing_is_needed(self, tmp_path):
        rng = np.random.default_rng(13)
        tables = _all_tables(rng)
        path = tmp_path / "1.npz"
        write_path_output(path, tables={"p2": tables["p2"]})   # no 'tilt'
        assert "order" in _needed_groups(path, overwrite=False)

    def test_missing_near_columns_trigger_a_recompute(self, tmp_path):
        rng = np.random.default_rng(14)
        path = tmp_path / "1.npz"
        write_path_output(path, tables=_all_tables(rng), spatial=_spatial(rng))
        needed = _needed_groups(path, overwrite=False, near_n=(5,))
        assert {"order", "structural"} <= needed

    def test_present_near_columns_do_not(self, tmp_path):
        rng = np.random.default_rng(15)
        tables = _all_tables(rng)
        chain = next(iter(ms.CHAIN_BONDS))
        tables["p2"][f"near5_{chain}_p2_chain_wsum"] = 1.0
        tables["thick"]["near5_thickness_wsum"] = 1.0
        path = tmp_path / "1.npz"
        write_path_output(path, tables=tables, spatial=_spatial(rng))
        assert _needed_groups(path, overwrite=False, near_n=(5,)) == set()


class TestAggregation:
    def test_the_pooled_table_is_the_ratio_of_summed_wsum_and_wtot(self, out_dir):
        rng = np.random.default_rng(16)
        written = {}
        for pn in (1, 2, 3):
            tables = _all_tables(rng)
            write_path_output(path_output_file(pn), tables=tables)
            written[pn] = tables
        weights = {1: 1.0, 2: 0.5, 3: 0.25}
        diff = aggregate_all_results(1, 3, weights, n_blocks=3)[5]
        wsum = sum(written[pn]["diff"]["D_z_wsum"].to_numpy() for pn in (1, 2, 3))
        wtot = sum(written[pn]["diff"]["D_z_wtot"].to_numpy() for pn in (1, 2, 3))
        np.testing.assert_allclose(diff["D_z"].to_numpy(), wsum / wtot, rtol=1e-5)

    def test_paths_without_a_weight_or_a_file_are_skipped(self, out_dir, capsys):
        rng = np.random.default_rng(17)
        for pn in (1, 2):
            write_path_output(path_output_file(pn), tables=_all_tables(rng))
        aggregate_all_results(1, 4, {1: 1.0, 3: 1.0}, n_blocks=1)
        out = capsys.readouterr().out
        assert "2 path(s) without a weight" in out
        assert "1 without an output file" in out

    def test_spatial_maps_pool_by_summing(self, out_dir):
        rng = np.random.default_rng(18)
        spatials = {}
        for pn in (1, 2):
            spatials[pn] = _spatial(rng)
            write_path_output(path_output_file(pn), spatial=spatials[pn])
        acc, x, y, shape, depth = aggregate_spatial_maps(1, 2, {1: 1.0, 2: 1.0})
        for name in spatials[1][0]:
            expected = spatials[1][0][name][0] + spatials[2][0][name][0]
            np.testing.assert_allclose(acc[name][0], expected, rtol=1e-5)
        assert shape == (4, 4)

    def test_spatial_paths_with_other_depth_bins_are_not_pooled(self, out_dir):
        rng = np.random.default_rng(19)
        write_path_output(path_output_file(1), spatial=_spatial(rng, n_depth=3))
        write_path_output(path_output_file(2), spatial=_spatial(rng, n_depth=5))
        acc, *_ = aggregate_spatial_maps(1, 2, {1: 1.0, 2: 1.0})
        first = read_path_spatial(path_output_file(1))
        np.testing.assert_allclose(acc["thickness"][0],
                                   first["wsum"][first["names"].index("thickness")])

    def test_no_spatial_maps_at_all_is_an_error(self, out_dir):
        with pytest.raises(RuntimeError, match="No spatial maps"):
            aggregate_spatial_maps(1, 2, {1: 1.0, 2: 1.0})
