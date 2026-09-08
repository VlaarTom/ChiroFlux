"""The dynamical features must measure what they claim, on signals whose answer
is known in advance.

Two of these tests exist to keep the module honest about its own premise. The
spectral-energy test pins down that ``specE`` *is* the variance rather than
merely correlating with it, which is the reason the redundancy report exists;
and the entropy test pins down the direction that makes spectral entropy
meaningful at all (noise high, single tone low). If those ever drift, the
redundancy heatmap stops being interpretable.
"""

import numpy as np
import pytest

from chiroflux.dynamics import (
    _autocorr_time,
    _detrend,
    _kind_redundancy,
    _lead_lag,
    _spectral_features,
    _window_bounds,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestDetrend:
    def test_removes_a_pure_linear_ramp(self):
        x = 3.0 + 0.7 * np.arange(200, dtype=float)
        assert np.allclose(_detrend(x), 0.0, atol=1e-9)

    def test_leaves_fluctuations_behind(self, rng):
        noise = rng.normal(size=400)
        x = 5.0 + 0.3 * np.arange(400) + noise
        out = _detrend(x)
        # The ramp is gone but the noise variance survives essentially intact.
        assert out.mean() == pytest.approx(0.0, abs=1e-9)
        assert np.var(out) == pytest.approx(np.var(noise), rel=0.1)


class TestSpectralEnergy:
    def test_spectral_energy_equals_the_variance(self, rng):
        """Parseval: this is why specE cannot add information over var."""
        x = _detrend(rng.normal(scale=2.0, size=501))
        energy, *_ = _spectral_features(x, dt=1.0)
        assert energy == pytest.approx(float(np.mean(x**2)), rel=0.15)

    def test_holds_for_an_even_length_window(self, rng):
        """The Nyquist bin must not be double-counted when folding."""
        x = _detrend(rng.normal(scale=1.5, size=500))
        energy, *_ = _spectral_features(x, dt=1.0)
        assert energy == pytest.approx(float(np.mean(x**2)), rel=0.15)


class TestSpectralShape:
    def test_main_frequency_finds_a_planted_tone(self):
        n, dt, f0 = 512, 0.5, 0.13
        t = np.arange(n) * dt
        x = _detrend(np.sin(2 * np.pi * f0 * t))
        _, fmain, centroid, _ = _spectral_features(x, dt)
        df = 1.0 / (n * dt)
        assert abs(fmain - f0) <= 2 * df
        assert abs(centroid - f0) <= 10 * df

    def test_entropy_is_high_for_noise_and_low_for_a_tone(self, rng):
        n, dt = 512, 1.0
        tone = _detrend(np.sin(2 * np.pi * 0.11 * np.arange(n) * dt))
        noise = _detrend(rng.normal(size=n))
        _, _, _, ent_tone = _spectral_features(tone, dt)
        _, _, _, ent_noise = _spectral_features(noise, dt)
        assert ent_tone < 0.5
        assert ent_noise > 0.8

    def test_too_short_a_window_returns_nan_rather_than_a_number(self):
        vals = _spectral_features(np.arange(5, dtype=float), dt=1.0)
        assert all(np.isnan(v) for v in vals)


def _ou_series(tau_true, n, rng, dt=1.0):
    """Ornstein-Uhlenbeck series with a known correlation time."""
    a = np.exp(-dt / tau_true)
    noise = rng.normal(scale=np.sqrt(1.0 - a**2), size=n)
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = a * x[i - 1] + noise[i]
    return x


class TestAutocorrelationTime:
    def test_recovers_a_known_correlation_time(self, rng):
        tau_true = 12.0
        x = _detrend(_ou_series(tau_true, 8000, rng))
        tau = _autocorr_time(x, dt=1.0)
        # The first-zero-crossing truncation biases low; the estimate should
        # still land in the right neighbourhood rather than merely be finite.
        assert 0.5 * tau_true < tau < 1.6 * tau_true

    def test_orders_a_slow_process_above_a_fast_one(self, rng):
        slow = _autocorr_time(_detrend(_ou_series(30.0, 8000, rng)), dt=1.0)
        fast = _autocorr_time(_detrend(_ou_series(3.0, 8000, rng)), dt=1.0)
        assert slow > 3 * fast

    def test_scales_with_the_time_step(self, rng):
        x = _detrend(_ou_series(10.0, 4000, rng))
        assert _autocorr_time(x, dt=2.0) == pytest.approx(
            2.0 * _autocorr_time(x, dt=1.0), rel=1e-9
        )

    def test_flat_input_is_nan_not_zero_division(self):
        assert np.isnan(_autocorr_time(np.zeros(100), dt=1.0))


class TestLeadLag:
    def test_recovers_a_planted_delay(self):
        """cv is op delayed by L frames, so the reported lag must be +L."""
        n, L = 201, 7
        t = np.arange(400, dtype=float)
        s = np.tanh((t - 200.0) / 15.0)
        k0 = 100
        op_win = s[k0 : k0 + n]
        cv_win = s[k0 - L : k0 - L + n]
        lag, r = _lead_lag(cv_win, op_win, max_lag=25, dt=1.0)
        assert lag == pytest.approx(float(L))
        assert r > 0.99

    def test_sign_flips_when_the_cv_leads(self):
        n, L = 201, 6
        t = np.arange(400, dtype=float)
        s = np.tanh((t - 200.0) / 15.0)
        k0 = 100
        op_win = s[k0 : k0 + n]
        cv_win = s[k0 + L : k0 + L + n]
        lag, _ = _lead_lag(cv_win, op_win, max_lag=25, dt=1.0)
        assert lag == pytest.approx(-float(L))

    def test_lag_is_reported_in_time_units(self):
        n, L = 201, 5
        t = np.arange(400, dtype=float)
        s = np.tanh((t - 200.0) / 15.0)
        op_win = s[100 : 100 + n]
        cv_win = s[100 - L : 100 - L + n]
        lag, _ = _lead_lag(cv_win, op_win, max_lag=25, dt=0.2)
        assert lag == pytest.approx(L * 0.2)

    def test_anticorrelated_signal_keeps_its_negative_sign(self):
        n, L = 201, 4
        t = np.arange(400, dtype=float)
        s = np.tanh((t - 200.0) / 15.0)
        op_win = s[100 : 100 + n]
        cv_win = -s[100 - L : 100 - L + n]
        lag, r = _lead_lag(cv_win, op_win, max_lag=25, dt=1.0)
        assert lag == pytest.approx(float(L))
        assert r < -0.99

    def test_constant_order_parameter_gives_nan(self):
        lag, r = _lead_lag(np.arange(50, dtype=float), np.ones(50), max_lag=10, dt=1.0)
        assert np.isnan(lag) and np.isnan(r)


class TestWindowBounds:
    def test_full_window_is_required_by_default(self):
        # A crossing 3 frames into the path cannot supply 5 frames on each side.
        assert _window_bounds(3, 100, width=11, mode="centered", min_frames=0) is None
        assert _window_bounds(50, 100, width=11, mode="centered", min_frames=0) == (45, 56)

    def test_pre_mode_ends_on_the_crossing(self):
        assert _window_bounds(50, 100, width=11, mode="pre", min_frames=0) == (40, 51)

    def test_pre_mode_never_reads_the_future(self):
        lo, hi = _window_bounds(60, 100, width=21, mode="pre", min_frames=0)
        assert hi == 61  # half-open: last frame read is the crossing itself

    def test_min_frames_accepts_a_clipped_window(self):
        assert _window_bounds(3, 100, width=11, mode="centered", min_frames=8) == (0, 9)
        assert _window_bounds(3, 100, width=11, mode="centered", min_frames=10) is None

    def test_clipping_respects_the_end_of_the_path(self):
        assert _window_bounds(97, 100, width=11, mode="centered", min_frames=5) == (92, 100)


class TestKindRedundancy:
    def test_a_duplicated_kind_scores_one(self, rng):
        """The check must actually flag a restatement, not just run."""
        n_paths, n_int = 300, 4
        kinds = ("var", "copy", "other")
        feats = np.empty((n_paths, 2 * len(kinds), n_int))
        for cv in range(2):
            base = cv * len(kinds)
            v = rng.normal(size=(n_paths, n_int))
            feats[:, base + 0, :] = v
            feats[:, base + 1, :] = 3.0 * v + 1.0        # monotone restatement
            feats[:, base + 2, :] = rng.normal(size=(n_paths, n_int))

        red = _kind_redundancy(feats, kinds, n_cvs=2)
        assert red[0, 1] == pytest.approx(1.0, abs=1e-6)
        assert red[0, 2] < 0.3
        assert np.allclose(np.diag(red), 1.0)
