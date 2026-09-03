"""SVM probabilities come from post-hoc Platt scaling, not from SVC itself.

`probability=True` fits an internal 5-fold calibration inside every SVM fit
(~5x the plain fit) and GPU SVM backends do not offer it at all. The same
sigmoid is recovered afterwards from the out-of-fold decision scores.
"""

import numpy as np
import pytest

from chiroflux.shap_analysis import _calibration_metrics, _platt_calibrate_oof


@pytest.fixture
def scores_and_labels():
    rng = np.random.default_rng(0)
    n = 600
    y = (rng.random(n) > 0.5).astype(int)
    scores = y + rng.normal(scale=0.8, size=n)   # separable but overlapping
    return scores, y


class TestPlattCalibration:
    def test_returns_probabilities_in_the_unit_interval(self, scores_and_labels):
        s, y = scores_and_labels
        p = _platt_calibrate_oof(s, y)
        assert np.isfinite(p).all()
        assert p.min() >= 0.0 and p.max() <= 1.0

    def test_preserves_ranking_so_auc_is_unchanged(self, scores_and_labels):
        """Platt is monotone: calibrating must not change discrimination."""
        from sklearn.metrics import roc_auc_score

        s, y = scores_and_labels
        p = _platt_calibrate_oof(s, y)
        assert roc_auc_score(y, p) == pytest.approx(roc_auc_score(y, s), abs=0.02)

    def test_probabilities_track_observed_frequency(self, scores_and_labels):
        """The point of calibrating: high-probability cases are mostly positive."""
        s, y = scores_and_labels
        p = _platt_calibrate_oof(s, y)
        assert y[p > 0.8].mean() > y[p < 0.2].mean()
        _, ece = _calibration_metrics(y, p)
        assert ece < 0.1

    def test_missing_scores_stay_nan(self, scores_and_labels):
        s, y = scores_and_labels
        s = s.copy()
        s[:20] = np.nan
        p = _platt_calibrate_oof(s, y)
        assert np.isnan(p[:20]).all()
        assert np.isfinite(p[20:]).sum() > 0

    def test_single_class_input_warns_and_returns_nan(self):
        s = np.linspace(-2, 2, 100)
        y = np.zeros(100, dtype=int)
        with pytest.warns(UserWarning, match="calibrate"):
            p = _platt_calibrate_oof(s, y)
        assert np.isnan(p).all()

    def test_too_few_scores_warns_rather_than_raising(self):
        with pytest.warns(UserWarning, match="calibrate"):
            p = _platt_calibrate_oof(np.array([0.1, -0.2]), np.array([0, 1]))
        assert np.isnan(p).all()


class TestCalibrationMetrics:
    def test_perfect_probabilities_score_near_zero(self):
        y = np.array([0, 0, 1, 1] * 40)
        brier, ece = _calibration_metrics(y, y.astype(float))
        assert brier == pytest.approx(0.0)
        assert ece == pytest.approx(0.0)

    def test_worst_case_probabilities_score_one(self):
        y = np.array([0, 1] * 40)
        brier, _ = _calibration_metrics(y, 1.0 - y)
        assert brier == pytest.approx(1.0)

    def test_confident_and_wrong_beats_nothing(self):
        """Brier must punish confident errors more than hedging."""
        rng = np.random.default_rng(0)
        y = (rng.random(400) > 0.5).astype(int)
        hedged, _ = _calibration_metrics(y, np.full(400, 0.5))
        confident_wrong, _ = _calibration_metrics(y, (1 - y) * 0.99 + 0.005)
        assert confident_wrong > hedged

    def test_returns_nan_when_not_computable(self):
        brier, ece = _calibration_metrics(np.zeros(5), np.full(5, 0.5))
        assert np.isnan(brier) and np.isnan(ece)
