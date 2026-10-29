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


# ---------------------------------------------------------------------------
# -align event
# ---------------------------------------------------------------------------

from chiroflux.dynamics import (  # noqa: E402
    _anchor_frames,
    _event_anchor,
    _event_traces,
    _trace_offsets,
)


class TestEventAnchor:
    OP = np.array([0.0, 0.5, 1.2, 1.9, 2.4, 1.0, 0.1])

    def test_reactive_anchors_on_the_first_crossing(self):
        assert _event_anchor(self.OP, True, event_op=1.5) == 3

    def test_reactive_without_a_crossing_has_no_event(self):
        assert _event_anchor(self.OP, True, event_op=3.0) is None

    def test_non_reactive_anchors_on_the_turning_point(self):
        assert _event_anchor(self.OP, False, event_op=1.5) == 4

    def test_shallow_turning_points_can_be_excluded(self):
        assert _event_anchor(self.OP, False, 1.5, event_min_op=2.4) == 4
        assert _event_anchor(self.OP, False, 1.5, event_min_op=2.5) is None

    def test_interface_mode_is_unchanged(self):
        assert _anchor_frames(self.OP, [0.4, 2.0, 9.0], "interface") == [1, 4, None]
        assert _anchor_frames(self.OP, [0.4, 2.0], "event", True, 1.5) == [3]

    def test_trace_offsets_match_the_window(self):
        assert list(_trace_offsets(2, "centered")) == [-2, -1, 0, 1, 2]
        assert list(_trace_offsets(2, "pre")) == [-4, -3, -2, -1, 0]


INTERFACES_EV = [0.0, 1.0, 2.0]
N_FRAMES_EV = 60
SAG_FROM = 8  # frames before the event at which the reactive CV drops


def _write_event_sim(root, n_per_class=12, seed=0):
    """Reactive paths ramp through lambda_B and their CV drops SAG_FROM frames
    before the crossing; non-reactive paths peak at OP 1.5 with a flat CV."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    (root / "infretis.toml").write_text(f"[simulation]\ninterfaces = {INTERFACES_EV}\n")
    ml = root / "ML"
    ml.mkdir(exist_ok=True)
    rows = []
    t = np.arange(N_FRAMES_EV)
    for pnr in range(2 * n_per_class):
        reactive = pnr < n_per_class
        if reactive:
            op = np.linspace(-0.2, 2.05, N_FRAMES_EV)
            event = int(np.argmax(op >= 2.0))
            cv = np.where(t >= event - SAG_FROM, -1.0, 0.0)
            rows.append(f"{pnr} {N_FRAMES_EV} {op.max():.3f} ---- 1.0 1.0 ---- 1.0 1.0")
        else:
            op = 1.5 - 1.7 * np.abs(t - 30) / 30
            cv = np.zeros(N_FRAMES_EV)
            rows.append(f"{pnr} {N_FRAMES_EV} {op.max():.3f} ---- 1.0 ---- ---- 1.0 ----")
        cv = cv + rng.normal(scale=0.01, size=N_FRAMES_EV)
        noise = rng.normal(size=N_FRAMES_EV)
        lines = ["# reactive" if reactive else "# non-reactive", "# plus ensemble",
                 "# info", "OP_Lamb cvA cvB"]
        lines += [f"{a:.5f} {b:.5f} {c:.5f}" for a, b, c in zip(op, cv, noise)]
        (ml / f"{pnr}.txt").write_text("\n".join(lines) + "\n")
    (root / "infretis_data.txt").write_text("\n".join(rows) + "\n")
    return root


class TestEventTraces:
    def test_traces_line_up_on_the_event(self, tmp_path):
        root = _write_event_sim(tmp_path / "sim")
        pnr = np.arange(24)
        labels = (pnr < 12).astype(float)
        offsets, names, mean, err, neff = _event_traces(
            root / "ML", pnr, "OP_Lamb", ["cvA"], labels, np.ones(24),
            half_window=10, window_mode="pre", min_frames=0,
            event_op=2.0, event_min_op=None,
        )
        assert names == ["OP_Lamb", "cvA"]
        assert offsets[-1] == 0
        cv = names.index("cvA")
        at = {off: i for i, off in enumerate(offsets)}
        # Reactive: the drop starts exactly SAG_FROM frames before the event.
        assert mean[1, cv, at[-SAG_FROM]] == pytest.approx(-1.0, abs=0.02)
        assert mean[1, cv, at[-SAG_FROM - 1]] == pytest.approx(0.0, abs=0.02)
        assert np.allclose(mean[0, cv], 0.0, atol=0.02)
        # The OP trace shows the alignment itself: reactive at lambda_B,
        # non-reactive at its turning point.
        assert mean[1, 0, at[0]] >= 2.0
        assert mean[0, 0, at[0]] == pytest.approx(1.5)
        assert np.all(neff[:, :, at[0]] == pytest.approx(12.0))

    def test_min_op_drops_shallow_non_reactive_paths(self, tmp_path):
        root = _write_event_sim(tmp_path / "sim")
        pnr = np.arange(24)
        labels = (pnr < 12).astype(float)
        *_, neff = _event_traces(
            root / "ML", pnr, "OP_Lamb", ["cvA"], labels, np.ones(24),
            half_window=10, window_mode="pre", min_frames=0,
            event_op=2.0, event_min_op=1.6,
        )
        assert np.all(neff[0] == 0)
        assert np.all(neff[1, :, -1] > 0)

    def test_command_end_to_end(self, tmp_path):
        from chiroflux.dynamics import dynamics

        root = _write_event_sim(tmp_path / "sim")
        out = tmp_path / "plots"
        dynamics(
            toml=str(root / "infretis.toml"), data=str(root / "infretis_data.txt"),
            cv_dir=str(root / "ML"), nskip=0, half_window=10, window_mode="pre",
            min_class=2, align="event", pair_lags=2, trace_top=2,
            plot_dir=str(out),
        )
        for name in ("event_effect_bar.png", "event_traces.png", "event_traces.txt",
                     "event_pairlag.png", "lag_profile.png", "dynamics_ranking.txt"):
            assert (out / name).exists(), name
        assert not (out / "interface_heatmap.png").exists()
        assert "align = event" in (out / "dynamics_ranking.txt").read_text()

    def test_unknown_align_mode_is_rejected(self, tmp_path):
        from chiroflux.dynamics import dynamics

        with pytest.raises(ValueError, match="-align"):
            dynamics(align="centre", plot_dir=str(tmp_path))
