"""Tests for the shared plotting helpers and the output-overwrite guard."""

import numpy as np
import pytest

from chiroflux.pathdata import _check_overwrite
from chiroflux.plotting import (
    MAX_FIG_INCHES,
    _fig_inches,
    _plot_importance_bar,
    _plot_interface_heatmap,
    _top_n_indices,
    _truncation_note,
)


class TestCheckOverwrite:
    def test_allows_a_path_that_does_not_exist(self, tmp_path):
        _check_overwrite(str(tmp_path / "new.png"), overw=False)

    def test_refuses_to_clobber_an_existing_file(self, tmp_path):
        existing = tmp_path / "out.png"
        existing.write_text("x")
        with pytest.raises(ValueError, match="already exists"):
            _check_overwrite(str(existing), overw=False)

    def test_overwrite_flag_permits_clobbering(self, tmp_path):
        existing = tmp_path / "out.png"
        existing.write_text("x")
        _check_overwrite(str(existing), overw=True)

    def test_empty_path_means_not_written_and_is_allowed(self):
        _check_overwrite("", overw=False)


class TestImportanceBar:
    def test_writes_the_figure(self, tmp_path):
        out = tmp_path / "bar.png"
        _plot_importance_bar(np.array([0.3, 0.9, 0.1]), ["a", "b", "c"], str(out))
        assert out.stat().st_size > 0

    def test_does_not_take_the_absolute_value(self, tmp_path):
        """SVM permutation importance can be negative; the shared helper must
        plot what it is given, or those bars would silently flip sign."""
        signed = tmp_path / "signed.png"
        absolute = tmp_path / "abs.png"
        values = np.array([0.4, -0.1, 0.9])
        _plot_importance_bar(values, ["a", "b", "c"], str(signed))
        _plot_importance_bar(np.abs(values), ["a", "b", "c"], str(absolute))
        assert signed.read_bytes() != absolute.read_bytes()

    def test_respects_the_overwrite_guard(self, tmp_path):
        out = tmp_path / "bar.png"
        out.write_text("x")
        with pytest.raises(ValueError):
            _plot_importance_bar(np.array([1.0]), ["a"], str(out))


class TestInterfaceHeatmap:
    @staticmethod
    def _results():
        return [
            {"lambda": 0.1, "ranking": [("cv_a", 0.5), ("cv_b", 0.2)]},
            {"lambda": 0.9, "ranking": [("cv_a", 0.1), ("cv_b", 0.7)]},
        ]

    def test_writes_the_figure(self, tmp_path):
        out = tmp_path / "heat.png"
        _plot_interface_heatmap(self._results(), ["cv_a", "cv_b"], str(out))
        assert out.stat().st_size > 0

    def test_writes_nothing_when_no_interface_has_a_ranking(self, tmp_path):
        out = tmp_path / "heat.png"
        _plot_interface_heatmap(
            [{"lambda": 0.1, "ranking": None}], ["cv_a"], str(out)
        )
        assert not out.exists()

    def test_labels_are_parameterised(self, tmp_path):
        """The two callers differ only by labels, so those must change output."""
        shap_png = tmp_path / "shap.png"
        cohen_png = tmp_path / "cohen.png"
        results, names = self._results(), ["cv_a", "cv_b"]
        _plot_interface_heatmap(results, names, str(shap_png))
        _plot_interface_heatmap(
            results, names, str(cohen_png),
            value_label="|Cohen's d|", title="|Cohen's d| across interfaces",
        )
        assert shap_png.read_bytes() != cohen_png.read_bytes()

    def test_warns_and_skips_a_cv_missing_from_cv_names(self, tmp_path):
        """Previously one copy raised KeyError and the other dropped it
        silently; the unified helper skips it but says so."""
        out = tmp_path / "heat.png"
        results = [{"lambda": 0.1, "ranking": [("cv_a", 0.5), ("ghost_cv", 0.9)]}]
        with pytest.warns(UserWarning, match="ghost_cv"):
            _plot_interface_heatmap(results, ["cv_a"], str(out))
        assert out.stat().st_size > 0


class TestFigureSizeCap:
    """A dynamics run has N_cvs x N_kinds features, not N_cvs.

    At 0.9 in per heatmap column and 300 dpi, ~240 features is already 64800
    pixels wide and anything past 242 exceeds matplotlib's 2**16 Agg limit —
    which raised at save time, at the very end of a run, leaving a broken file
    behind. These pin both halves of the fix: pick fewer features, and never
    ask for a figure the backend cannot draw.
    """

    def test_never_asks_for_more_than_the_backend_can_draw(self):
        assert _fig_inches(0.9, 800, 6) == MAX_FIG_INCHES
        assert MAX_FIG_INCHES * 300 < 65536

    def test_small_inputs_are_untouched(self):
        assert _fig_inches(0.9, 20, 6) == pytest.approx(18.0)
        assert _fig_inches(0.9, 2, 6) == pytest.approx(6.0)  # the floor

    def test_top_n_keeps_the_largest_in_original_order(self):
        scores = np.array([0.1, 5.0, 0.3, 4.0, 0.2])
        assert _top_n_indices(scores, 2).tolist() == [1, 3]

    def test_top_n_of_zero_or_none_keeps_everything(self):
        scores = np.arange(5.0)
        assert _top_n_indices(scores, 0).tolist() == [0, 1, 2, 3, 4]
        assert _top_n_indices(scores, None).tolist() == [0, 1, 2, 3, 4]

    def test_nan_scores_do_not_win_a_slot(self):
        scores = np.array([np.nan, 1.0, np.nan, 2.0])
        assert _top_n_indices(scores, 2).tolist() == [1, 3]

    def test_bar_chart_survives_a_feature_count_that_used_to_crash(self, tmp_path):
        out = tmp_path / "bar.png"
        names = [f"cv{i}_var" for i in range(400)]
        _plot_importance_bar(
            np.linspace(0, 1, 400), names, str(out), top_n=40
        )
        assert out.exists() and out.stat().st_size > 0

    def test_heatmap_survives_a_feature_count_that_used_to_crash(self, tmp_path):
        out = tmp_path / "heat.png"
        names = [f"cv{i}_var" for i in range(400)]
        results = [
            {"lambda": 0.1 * i, "ranking": [(n, float(j)) for j, n in enumerate(names)]}
            for i in range(3)
        ]
        _plot_interface_heatmap(results, names, str(out), top_n=40)
        assert out.exists() and out.stat().st_size > 0

    def test_truncation_is_declared_in_the_title(self):
        assert _truncation_note(40, 400) == "  (top 40 of 400)"
        assert _truncation_note(40, 40) == ""
