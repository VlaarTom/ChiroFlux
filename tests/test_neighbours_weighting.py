"""neighbours stores per-path counts at weight 1 and applies the WHAM weights
when it aggregates.

The weights renormalise whenever the simulations are extended. With them baked
into the per-path CSVs every path had to be recomputed; now a new weights file
only needs a re-aggregation. What has to hold: the weights are applied exactly
once, to both count kinds; paths without a positive weight are left out; and a
CSV from before the change - weight already multiplied in - is never read as
unweighted.
"""

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from chiroflux import neighbours as nb
from chiroflux.neighbours import (
    UNWEIGHTED_MARKER,
    _is_current,
    aggregate_neighbour_results,
    compute_enrichment_statistics,
    load_weighted_paths,
    process_single_path_neighbour,
)

LABELS = ("3-DOPC", "3-POPC", "Mix")


def _counts(rng, n_slabs=6):
    raw = rng.integers(0, 20, size=(n_slabs, 3)).astype(float)
    df = pd.DataFrame({"slab_center": np.arange(n_slabs) - n_slabs / 2 + 0.5})
    for i, lbl in enumerate(LABELS):
        df[f"weighted_count_{lbl}"] = raw[:, i]
    total = raw.sum()
    for i, lbl in enumerate(LABELS):
        df[f"norm_count_{lbl}"] = raw[:, i] / total
    return df


def _write(out_dir, pn, df, reactive=True, ensemble="plus", current=True):
    path = out_dir / f"{pn}_neighbour.csv"
    with open(path, "w") as fh:
        fh.write(f"# {'reactive' if reactive else 'non-reactive'}\n")
        fh.write(f"# {ensemble} ensemble\n")
        if current:
            fh.write(f"{UNWEIGHTED_MARKER}\n")
    df.to_csv(path, mode="a", index=False, float_format="%.6e")
    return path


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(nb, "NEIGH_OUT", str(tmp_path))
    monkeypatch.setattr(nb, "PLOT_OUT", str(tmp_path / "plots"))
    (tmp_path / "plots").mkdir()
    return tmp_path


class TestLoadingWithWeights:
    def test_both_count_kinds_are_multiplied_by_the_weight(self, out_dir):
        rng = np.random.default_rng(0)
        df = _counts(rng)
        _write(out_dir, 1, df)
        ((pn, loaded, _, _),) = load_weighted_paths(1, 1, {1: 0.25}, quiet=True)
        assert pn == 1
        for lbl in LABELS:
            np.testing.assert_allclose(loaded[f"weighted_count_{lbl}"],
                                       0.25 * df[f"weighted_count_{lbl}"], rtol=1e-6)
            np.testing.assert_allclose(loaded[f"norm_count_{lbl}"],
                                       0.25 * df[f"norm_count_{lbl}"], rtol=1e-5)
        assert loaded["slab_center"].tolist() == df["slab_center"].tolist()

    def test_a_path_normalised_contribution_sums_to_its_weight(self, out_dir):
        """What the equal-path scheme relies on, now that w goes on at load."""
        _write(out_dir, 1, _counts(np.random.default_rng(1)))
        ((_, loaded, _, _),) = load_weighted_paths(1, 1, {1: 0.3}, quiet=True)
        total = sum(loaded[f"norm_count_{lbl}"].sum() for lbl in LABELS)
        assert total == pytest.approx(0.3, rel=1e-5)

    def test_reactivity_and_ensemble_come_from_the_header(self, out_dir):
        rng = np.random.default_rng(2)
        _write(out_dir, 1, _counts(rng), reactive=True, ensemble="plus")
        _write(out_dir, 2, _counts(rng), reactive=False, ensemble="plus")
        _write(out_dir, 3, _counts(rng), reactive=False, ensemble="minus")
        got = {pn: (r, e) for pn, _, r, e in
               load_weighted_paths(1, 3, {1: 1.0, 2: 1.0, 3: 1.0}, quiet=True)}
        assert got == {1: ("reactive", "plus"), 2: ("non-reactive", "plus"),
                       3: ("non-reactive", "minus")}

    def test_paths_without_a_positive_weight_or_a_csv_are_left_out(self, out_dir, capsys):
        rng = np.random.default_rng(3)
        for pn in (1, 2, 3):
            _write(out_dir, pn, _counts(rng))
        loaded = load_weighted_paths(1, 4, {1: 1.0, 2: 0.0, 4: 1.0})
        assert [pn for pn, *_ in loaded] == [1]
        out = capsys.readouterr().out
        assert "skipped 2 without a positive weight and 1 without a CSV" in out

    def test_a_csv_from_before_the_change_is_skipped_not_reweighted(self, out_dir, capsys):
        """Its counts already carry a weight; applying another would square it."""
        rng = np.random.default_rng(4)
        _write(out_dir, 1, _counts(rng), current=True)
        _write(out_dir, 2, _counts(rng), current=False)
        loaded = load_weighted_paths(1, 2, {1: 1.0, 2: 1.0})
        assert [pn for pn, *_ in loaded] == [1]
        assert "skipped 1 CSV(s) written before" in capsys.readouterr().out

    def test_only_a_current_csv_counts_as_done(self, out_dir):
        rng = np.random.default_rng(5)
        assert not _is_current(out_dir / "missing.csv")
        assert _is_current(_write(out_dir, 1, _counts(rng), current=True))
        assert not _is_current(_write(out_dir, 2, _counts(rng), current=False))


class TestAggregation:
    def test_the_pool_is_the_weighted_sum_normalised_per_slab(self, out_dir):
        rng = np.random.default_rng(6)
        frames = {pn: _counts(rng) for pn in (1, 2, 3)}
        for pn, df in frames.items():
            _write(out_dir, pn, df)
        weights = {1: 1.0, 2: 0.5, 3: 0.1}
        agg = aggregate_neighbour_results(1, 3, weights)

        pooled = {lbl: sum(weights[pn] * frames[pn][f"weighted_count_{lbl}"].to_numpy()
                           for pn in frames) for lbl in LABELS}
        total = sum(pooled.values())
        for lbl in LABELS:
            expected = np.where(total > 0, pooled[lbl] / np.where(total > 0, total, 1), 0)
            np.testing.assert_allclose(agg[lbl].to_numpy(), expected, rtol=1e-5)

    def test_new_weights_need_no_recomputation(self, out_dir):
        rng = np.random.default_rng(7)
        paths = [_write(out_dir, pn, _counts(rng)) for pn in (1, 2)]
        before = [p.read_bytes() for p in paths]
        a = aggregate_neighbour_results(1, 2, {1: 1.0, 2: 1.0})
        b = aggregate_neighbour_results(1, 2, {1: 1.0, 2: 50.0})
        assert not np.allclose(a["Mix"], b["Mix"])
        assert [p.read_bytes() for p in paths] == before

    def test_the_ensemble_split_writes_each_group(self, out_dir):
        rng = np.random.default_rng(8)
        _write(out_dir, 1, _counts(rng), reactive=True, ensemble="plus")
        _write(out_dir, 2, _counts(rng), reactive=False, ensemble="plus")
        _write(out_dir, 3, _counts(rng), reactive=False, ensemble="minus")
        nb.aggregate_ensemble_reactivity_specific_results(
            1, 3, {1: 1.0, 2: 1.0, 3: 1.0})
        written = sorted(p.name for p in (out_dir / "plots").glob("*.csv"))
        assert written == ["neighbours_minus.csv", "neighbours_plus_non_reactive.csv",
                           "neighbours_plus_reactive.csv"]

    def test_the_statistics_load_weighted_paths_themselves(self, out_dir):
        rng = np.random.default_rng(9)
        for pn in (1, 2, 3, 4):
            _write(out_dir, pn, _counts(rng))
        weights = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
        from_disk = compute_enrichment_statistics(1, 4, weights, n_bootstrap=20,
                                                  min_count=0.0)
        preloaded = compute_enrichment_statistics(
            1, 4, weights, n_bootstrap=20, min_count=0.0,
            path_dfs=[(pn, df) for pn, df, _, _ in
                      load_weighted_paths(1, 4, weights, quiet=True)])
        pd.testing.assert_frame_equal(from_disk, preloaded)


class TestWorker:
    def test_a_current_csv_is_skipped(self, out_dir, monkeypatch):
        monkeypatch.chdir(out_dir)
        _write(out_dir, 1, _counts(np.random.default_rng(10)), current=True)
        result = process_single_path_neighbour(1, False, -25.0, -13.5, -35.0, "x")
        assert result == (1, "skipped", "Output CSV already exists.")

    def test_a_csv_from_before_the_change_is_recomputed(self, out_dir, monkeypatch):
        """It gets past the existence check and on to the trajectories."""
        monkeypatch.chdir(out_dir)
        _write(out_dir, 1, _counts(np.random.default_rng(11)), current=False)
        result = process_single_path_neighbour(1, False, -25.0, -13.5, -35.0, "x")
        assert result[1] == "skipped"
        assert "Path folder does not exist" in result[2]


class TestCommandLine:
    def test_the_options_reach_the_command(self, tmp_path, monkeypatch):
        """Both old breakages: a string default Typer rejected before running,
        and an argparse block that re-parsed sys.argv over every option."""
        from chiroflux.cli import app

        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(app, ["neighbours", "-data", "d.txt", "-start", "1"])
        assert isinstance(result.exception, FileNotFoundError)
        assert "infretis.toml" in str(result.exception)

    def test_the_bulk_fraction_default_is_a_number(self):
        import inspect
        default = inspect.signature(nb.neighbours).parameters["bulk_dopc"].default
        assert default == pytest.approx(5.0 / 6.0)
