"""Unit tests for the weighting/statistics helpers.

These are the pure-numeric pieces the analyses lean on, so they can be
checked without any simulation input.
"""

import warnings

import numpy as np
import pytest

from chiroflux.cvs import (
    _apply_cv_flip,
    _apply_cv_phase_shift,
    _apply_z_corrections,
    _frame_count_after_subsample,
    _subsample_frames,
)
from chiroflux.prepare_deeptda_data import _proximity_weight
from chiroflux.statistical_analysis import (
    _weighted_cohens_d,
    _weighted_ks,
    _weighted_mean_var,
    _weighted_spearman,
)


class TestProximityWeight:
    def test_clamps_outside_the_ramp(self):
        op = np.array([-5.0, 0.0, 10.0, 99.0])
        w = _proximity_weight(op, 0.0, 10.0)
        assert w[0] == 0.0 and w[1] == 0.0
        assert w[2] == 1.0 and w[3] == 1.0

    def test_smoothstep_is_one_half_at_the_midpoint(self):
        assert _proximity_weight(np.array([5.0]), 0.0, 10.0)[0] == pytest.approx(0.5)

    def test_is_monotonically_increasing(self):
        w = _proximity_weight(np.linspace(-1, 11, 200), 0.0, 10.0)
        assert np.all(np.diff(w) >= 0)


class TestSubsampling:
    @pytest.mark.parametrize("stride", [1, 2, 3, 7])
    @pytest.mark.parametrize("max_frames", [None, 1, 4, 50])
    @pytest.mark.parametrize("n_frames", [1, 5, 20])
    def test_predicted_count_matches_actual(self, n_frames, stride, max_frames):
        """_frame_count_after_subsample sizes the output arrays up front, so it
        must agree exactly with what _subsample_frames really returns."""
        op = np.arange(n_frames, dtype=float)
        cvs = np.zeros((n_frames, 3))
        kept_op, kept_cvs = _subsample_frames(op, cvs, stride, max_frames)
        predicted = _frame_count_after_subsample(n_frames, stride, max_frames)
        assert len(kept_op) == predicted
        assert len(kept_cvs) == predicted

    def test_keeps_op_and_cvs_aligned(self):
        op = np.arange(10, dtype=float)
        cvs = np.arange(10, dtype=float).reshape(10, 1) * 100
        kept_op, kept_cvs = _subsample_frames(op, cvs, 3, None)
        assert np.allclose(kept_cvs[:, 0], kept_op * 100)


class TestWeightedMoments:
    def test_uniform_weights_match_numpy(self):
        x = np.array([1.0, 2.0, 4.0, 8.0])
        mu, var = _weighted_mean_var(x, np.ones_like(x))
        assert mu == pytest.approx(x.mean())
        assert var == pytest.approx(x.var())  # population variance

    def test_weights_are_normalised_internally(self):
        x = np.array([1.0, 2.0, 4.0, 8.0])
        w = np.array([1.0, 1.0, 2.0, 2.0])
        assert _weighted_mean_var(x, w) == pytest.approx(_weighted_mean_var(x, w * 10))

    def test_zero_weight_samples_are_ignored(self):
        x = np.array([1.0, 2.0, 1000.0])
        w = np.array([1.0, 1.0, 0.0])
        mu, _ = _weighted_mean_var(x, w)
        assert mu == pytest.approx(1.5)


class TestCohensD:
    def test_sign_is_positive_when_reactive_mean_is_higher(self):
        x = np.array([0.0, 1.0, 4.0, 5.0])
        labels = np.array([0, 0, 1, 1])
        assert _weighted_cohens_d(x, labels, np.ones(4)) > 0

    def test_zero_within_class_spread_gives_zero(self):
        """pooled_std == 0 short-circuits to 0.0, however far apart the means."""
        x = np.array([0.0, 0.0, 1000.0, 1000.0])
        assert _weighted_cohens_d(x, np.array([0, 0, 1, 1]), np.ones(4)) == 0.0

    def test_sign_flips_with_the_labels(self):
        x = np.array([0.0, 1.0, 4.0, 5.0])
        w = np.ones(4)
        d_pos = _weighted_cohens_d(x, np.array([0, 0, 1, 1]), w)
        d_neg = _weighted_cohens_d(x, np.array([1, 1, 0, 0]), w)
        assert d_pos == pytest.approx(-d_neg)

    def test_identical_classes_give_zero(self):
        x = np.array([1.0, 2.0, 1.0, 2.0])
        assert _weighted_cohens_d(x, np.array([0, 0, 1, 1]), np.ones(4)) == 0.0

    def test_returns_nan_when_a_class_carries_no_weight(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        w = np.array([1.0, 1.0, 0.0, 0.0])
        assert np.isnan(_weighted_cohens_d(x, np.array([0, 0, 1, 1]), w))

    def test_weights_shift_the_effect_size(self):
        x = np.array([0.0, 10.0, 1.0, 11.0])
        labels = np.array([0, 0, 1, 1])
        equal = _weighted_cohens_d(x, labels, np.ones(4))
        skewed = _weighted_cohens_d(x, labels, np.array([1.0, 0.01, 1.0, 0.01]))
        assert not np.isclose(equal, skewed)


class TestSpearman:
    def test_perfect_separation_is_strongly_positive(self):
        """Correlating ranks against a *binary* label caps below 1.0 (a binary
        variable cannot reproduce evenly spaced ranks), so check the sign and
        that separated classes score far above mixed ones."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        separated = _weighted_spearman(x, np.array([0, 0, 1, 1]), np.ones(4))
        mixed = _weighted_spearman(x, np.array([0, 1, 0, 1]), np.ones(4))
        assert separated > 0
        assert separated > abs(mixed)

    def test_sign_flips_with_the_labels(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones(4)
        forward = _weighted_spearman(x, np.array([0, 0, 1, 1]), w)
        reversed_ = _weighted_spearman(x, np.array([1, 1, 0, 0]), w)
        assert forward == pytest.approx(-reversed_)

    def test_is_invariant_to_monotone_rescaling_of_x(self):
        labels = np.array([0, 0, 1, 1])
        w = np.ones(4)
        base = _weighted_spearman(np.array([1.0, 2.0, 3.0, 4.0]), labels, w)
        stretched = _weighted_spearman(np.array([1.0, 2.0, 30.0, 4000.0]), labels, w)
        assert base == pytest.approx(stretched)

    @pytest.mark.xfail(
        reason="known bug: _weighted_spearman does not average tied ranks. "
        "np.argsort(np.argsort(x)) hands a constant column the distinct ranks "
        "0,1,2,...,n-1, so std_r is never 0 and the `std_r < 1e-12` guard that "
        "was meant to catch this can never fire. A degenerate CV therefore "
        "reports a large spurious rho instead of 0.",
        strict=True,
    )
    def test_constant_feature_gives_zero(self):
        x = np.ones(4)
        assert _weighted_spearman(x, np.array([0, 0, 1, 1]), np.ones(4)) == 0.0


class TestWeightedKS:
    def test_disjoint_distributions_give_one(self):
        x = np.array([0.0, 0.1, 5.0, 5.1])
        assert _weighted_ks(x, np.array([0, 0, 1, 1]), np.ones(4)) == pytest.approx(1.0)

    def test_identical_distributions_give_zero(self):
        x = np.array([1.0, 2.0, 1.0, 2.0])
        assert _weighted_ks(x, np.array([0, 0, 1, 1]), np.ones(4)) == pytest.approx(0.0)

    def test_is_bounded_to_the_unit_interval(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=200)
        labels = rng.integers(0, 2, size=200)
        w = rng.uniform(0.1, 5.0, size=200)
        assert 0.0 <= _weighted_ks(x, labels, w) <= 1.0

    def test_returns_nan_when_a_class_carries_no_weight(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        w = np.array([1.0, 1.0, 0.0, 0.0])
        assert np.isnan(_weighted_ks(x, np.array([0, 0, 1, 1]), w))


class TestEntryFlip:
    """theta -> 180 - theta: opposite-leaflet entry, a reflection about 90 degrees."""

    @staticmethod
    def _arr(values):
        # (N_paths, N_cvs, N_interfaces) with one CV and one interface
        return np.array(values, dtype=float).reshape(-1, 1, 1)

    def test_maps_theta_to_180_minus_theta(self):
        out, _ = _apply_cv_flip(self._arr([0, 30, 90, 150, 180]), ["tilt"], ["tilt"])
        assert out.ravel() == pytest.approx([180, 150, 90, 30, 0])

    def test_is_not_a_plus_90_shift(self):
        """The name and the old help text said +90; that would give 120, not 60."""
        out, _ = _apply_cv_flip(self._arr([120.0]), ["tilt"], ["tilt"])
        assert out.ravel()[0] == pytest.approx(60.0)

    def test_ninety_degrees_is_the_fixed_point(self):
        out, _ = _apply_cv_flip(self._arr([90.0]), ["tilt"], ["tilt"])
        assert out.ravel()[0] == pytest.approx(90.0)

    def test_is_its_own_inverse(self):
        original = self._arr([0, 17, 90, 133, 180])
        once, names = _apply_cv_flip(original, ["tilt"], ["tilt"])
        twice, _ = _apply_cv_flip(once, names, ["tilt"])
        assert twice == pytest.approx(original)

    def test_preserves_the_zero_to_180_domain(self):
        out, _ = _apply_cv_flip(self._arr(np.linspace(0, 180, 50)), ["tilt"], ["tilt"])
        assert out.min() >= 0.0 and out.max() <= 180.0

    def test_flips_the_sign_of_the_cosine(self):
        angles = np.linspace(0, 180, 25)
        out, _ = _apply_cv_flip(self._arr(angles), ["tilt"], ["tilt"])
        assert np.cos(np.deg2rad(out.ravel())) == pytest.approx(-np.cos(np.deg2rad(angles)))

    def test_leaves_unmatched_columns_alone(self):
        arr = np.array([[[10.0], [20.0]]])  # 1 path, 2 CVs, 1 interface
        out, _ = _apply_cv_flip(arr, ["tilt", "other"], ["tilt"])
        assert out[0, 0, 0] == pytest.approx(170.0)
        assert out[0, 1, 0] == pytest.approx(20.0)

    def test_does_not_mutate_the_input(self):
        arr = self._arr([30.0])
        _apply_cv_flip(arr, ["tilt"], ["tilt"])
        assert arr.ravel()[0] == pytest.approx(30.0)

    def test_empty_substring_list_is_a_no_op(self):
        arr = self._arr([30.0])
        out, names = _apply_cv_flip(arr, ["tilt"], [])
        assert out is arr and names == ["tilt"]

    def test_skips_already_cosine_folded_columns(self):
        """_apply_angle_transforms renames to 'cos(<name>)' and runs first, so a
        substring match can otherwise hit a value in [-1,1] and return ~179."""
        arr = self._arr([0.5])
        with pytest.warns(UserWarning, match="already in cosine form"):
            out, _ = _apply_cv_flip(arr, ["cos(tilt)"], ["tilt"])
        assert out.ravel()[0] == pytest.approx(0.5)


class TestZCorrectionExcludeGuard:
    """A z-column missing because -exclude dropped it is expected, not an error.

    -exclude matches by SUBSTRING, so the guard has to use the same predicate
    the column discovery used; exact list membership silently misses.
    """

    NAMES = ["z_Memb", "z_O3_B"]          # z_O2_T already dropped by -exclude
    Z_COLS = ["z_O2_T", "z_O3_B"]

    def _run(self, exclude_list, capsys=None):
        arr = np.zeros((2, len(self.NAMES), 1))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _apply_z_corrections(arr, list(self.NAMES), self.Z_COLS, exclude_list)
        return [str(w.message) for w in caught]

    def test_substring_exclude_does_not_warn(self):
        """The reported bug: '-exclude O2' must silence z_O2_T."""
        assert self._run(["O2"]) == []

    def test_exact_name_exclude_does_not_warn(self):
        assert self._run(["z_O2_T"]) == []

    def test_prints_a_single_note_instead(self, capsys):
        self._run(["O2"])
        out = capsys.readouterr().out
        assert out.count("\n") == 1
        assert "z_O2_T" in out and "-exclude" in out

    def test_a_genuine_typo_still_warns(self):
        """Only excluded columns get a pass; a typo must stay loud."""
        messages = self._run(["something_else"])
        assert len(messages) == 1
        assert "z_O2_T" in messages[0]

    def test_none_exclude_list_does_not_crash(self):
        """`col in None` used to raise TypeError for every caller that
        omitted -exclude, which is most of them."""
        messages = self._run(None)
        assert len(messages) == 1  # warns, but does not raise


class TestPhaseShift:
    """phi -> phi + 180 (mod 360): the mirror of a periodic phase whose
    reference vector is a pseudovector (e.g. Cremer-Pople phi2)."""

    @staticmethod
    def _arr(values):
        return np.array(values, dtype=float).reshape(-1, 1, 1)

    def _shift(self, values, names=("phi2",), subs=("phi2",)):
        out, _ = _apply_cv_phase_shift(self._arr(values), list(names), list(subs))
        return out.ravel()

    def test_advances_by_half_a_turn(self):
        assert self._shift([0.0, 45.0, -90.0]) == pytest.approx([180.0, -135.0, 90.0])

    def test_result_stays_in_the_arctan2_interval(self):
        out = self._shift(np.linspace(-179.9, 180.0, 200))
        assert out.min() > -180.0 and out.max() <= 180.0

    def test_is_its_own_inverse(self):
        original = [-179.0, -90.0, 0.0, 45.0, 179.0, 180.0]
        assert self._shift(self._shift(original)) == pytest.approx(original)

    def test_really_is_a_180_degree_rotation(self):
        """Both sine and cosine must negate - a half turn, not a reflection."""
        xs = np.array([-170.0, -33.0, 0.0, 12.0, 175.0])
        out = self._shift(xs)
        assert np.cos(np.radians(out)) == pytest.approx(-np.cos(np.radians(xs)))
        assert np.sin(np.radians(out)) == pytest.approx(-np.sin(np.radians(xs)))

    def test_is_not_negation_and_not_the_180_minus_theta_flip(self):
        """The whole reason this exists: neither of the other two is correct."""
        x = 45.0
        assert self._shift([x])[0] == pytest.approx(-135.0)
        assert self._shift([x])[0] != pytest.approx(-x)        # not the mirror
        assert self._shift([x])[0] != pytest.approx(180.0 - x)  # not the entry flip

    def test_leaves_unmatched_columns_alone(self):
        arr = np.array([[[10.0], [20.0]]])
        out, _ = _apply_cv_phase_shift(arr, ["phi2", "other"], ["phi2"])
        assert out[0, 0, 0] == pytest.approx(-170.0)
        assert out[0, 1, 0] == pytest.approx(20.0)

    def test_does_not_mutate_the_input(self):
        arr = self._arr([45.0])
        _apply_cv_phase_shift(arr, ["phi2"], ["phi2"])
        assert arr.ravel()[0] == pytest.approx(45.0)

    def test_empty_substring_list_is_a_no_op(self):
        arr = self._arr([45.0])
        out, names = _apply_cv_phase_shift(arr, ["phi2"], [])
        assert out is arr and names == ["phi2"]

    def test_warns_when_the_column_is_not_on_minus180_to_180(self):
        """A [0,360) column would be silently re-based onto a different
        convention from the simulation it is being compared against."""
        with pytest.warns(UserWarning, match=r"outside \(-180, 180\]"):
            _apply_cv_phase_shift(self._arr([0.0, 90.0, 270.0]), ["phi2"], ["phi2"])
