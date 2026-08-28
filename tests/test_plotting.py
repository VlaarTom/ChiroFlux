"""Tests for the shared plotting helpers and the output-overwrite guard."""

import numpy as np
import pytest

from chiroflux.pathdata import _check_overwrite
from chiroflux.plotting import _plot_importance_bar, _plot_interface_heatmap


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
