"""The local diffusion profile has to come back with the right *value*, so
these tests run the estimator on walks whose D is known by construction.

The estimator this replaced cut the trajectory into "sojourns" - runs of
consecutive frames in one slab - and fitted each. That conditions the sample on
the permeant not leaving, which keeps the slowest excursions and discards the
rest, and it does so hardest where the motion is fastest: on the two-region
walk below it recovered 66% of the true D in the slow region against 22% in the
fast one, flattening a 6:1 contrast to 2:1. Binning windows by where they
*start* has no such condition, and the tests here pin that it recovers both
regions and their ratio.
"""

import numpy as np
import pytest

from chiroflux.membrane_spatial import (
    _unwrap_xy_pbc,
    _whole_centroid,
    build_slab_grid,
    diffusion_from_msd,
    fit_diffusion_intercept,
    lag_column,
    parse_lag_selection,
    slab_binned_msd,
)


def brownian(n, d, dt, rng, start=0.0):
    """A 1-D walk with MSD = 2*d*t by construction."""
    steps = rng.normal(scale=np.sqrt(2.0 * d * dt), size=n)
    steps[0] = start
    return np.cumsum(steps)


def slab_index(z, edges):
    return np.clip(np.searchsorted(edges, z, side="right") - 1,
                   0, len(edges) - 2)


def recover_d(z, edges, dt, lag=1, n_dim=1):
    """D per slab, the way the frame loop accumulates it."""
    si = slab_index(z, edges)
    total, count = slab_binned_msd(z, si, len(edges) - 1, lag)
    return diffusion_from_msd(total, count, lag, dt, n_dim=n_dim), count


class TestMSDAccumulation:
    def test_windows_are_binned_by_where_they_start(self):
        """Not by where they end, and not by where they stayed."""
        z = np.array([0.0, 5.0, 5.0, 5.0])
        si = np.array([0, 1, 1, 1])
        total, count = slab_binned_msd(z, si, 2, lag=1)
        assert count.tolist() == [1.0, 2.0]
        assert total[0] == pytest.approx(25.0)      # the 0 -> 5 jump
        assert total[1] == pytest.approx(0.0)

    def test_every_consecutive_pair_contributes(self):
        z = np.arange(11.0)
        si = np.zeros(11, dtype=int)
        _, count = slab_binned_msd(z, si, 1, lag=1)
        assert count[0] == 10.0

    def test_a_longer_lag_uses_fewer_windows(self):
        z = np.arange(11.0)
        si = np.zeros(11, dtype=int)
        assert slab_binned_msd(z, si, 1, lag=3)[1][0] == 8.0

    def test_a_lag_longer_than_the_path_contributes_nothing(self):
        z = np.arange(4.0)
        si = np.zeros(4, dtype=int)
        total, count = slab_binned_msd(z, si, 1, lag=10)
        assert total.sum() == 0.0 and count.sum() == 0.0

    def test_the_lateral_series_sums_over_both_axes(self):
        xy = np.array([[0.0, 0.0], [3.0, 4.0]])
        total, count = slab_binned_msd(xy, np.array([0, 0]), 1, lag=1)
        assert total[0] == pytest.approx(25.0)      # 3-4-5 triangle
        assert count[0] == 1.0

    def test_unvisited_slabs_are_nan_not_zero(self):
        """No windows is not a measurement of zero mobility."""
        d = diffusion_from_msd(np.zeros(3), np.array([4.0, 0.0, 2.0]), 1, 1.0)
        assert np.isnan(d[1])
        assert np.isfinite(d[0]) and np.isfinite(d[2])

    def test_the_conversion_is_msd_over_2_n_dim_tau(self):
        total = np.array([100.0])
        count = np.array([10.0])                    # <dz^2> = 10
        assert diffusion_from_msd(total, count, 1, 1.0)[0] == pytest.approx(5.0)
        assert diffusion_from_msd(total, count, 5, 1.0)[0] == pytest.approx(1.0)
        assert diffusion_from_msd(total, count, 1, 2.0)[0] == pytest.approx(2.5)
        assert diffusion_from_msd(total, count, 1, 1.0, n_dim=2)[0] \
            == pytest.approx(2.5)


class TestRecoveringAKnownD:
    def test_a_free_walk_comes_back_at_its_own_D(self):
        rng = np.random.default_rng(0)
        z = brownian(300_000, 0.2, 1.0, rng)
        edges = np.arange(z.min() - 1, z.max() + 2, 1.0)
        d, count = recover_d(z, edges, 1.0)
        assert np.nansum(d * count) / count.sum() == pytest.approx(0.2, rel=0.02)

    def test_the_answer_does_not_depend_on_the_slab_width(self):
        """The old estimator's answer did: 31% of the truth at 1 A, 88% at 5."""
        rng = np.random.default_rng(1)
        z = brownian(300_000, 0.2, 1.0, rng)
        for width in (1.0, 2.0, 5.0):
            edges = np.arange(z.min() - width, z.max() + 2 * width, width)
            d, count = recover_d(z, edges, 1.0)
            pooled = np.nansum(d * count) / count.sum()
            assert pooled == pytest.approx(0.2, rel=0.03), f"width {width}"

    def test_the_answer_does_not_depend_on_the_lag(self):
        rng = np.random.default_rng(2)
        z = brownian(300_000, 0.2, 1.0, rng)
        edges = np.arange(z.min() - 1, z.max() + 2, 1.0)
        for lag in (1, 2, 5):
            d, count = recover_d(z, edges, 1.0, lag=lag)
            pooled = np.nansum(d * count) / count.sum()
            assert pooled == pytest.approx(0.2, rel=0.03), f"lag {lag}"

    def test_a_lateral_walk_uses_the_2D_normalisation(self):
        rng = np.random.default_rng(3)
        d_true, dt = 0.2, 1.0
        xy = np.cumsum(
            rng.normal(scale=np.sqrt(2.0 * d_true * dt), size=(200_000, 2)),
            axis=0,
        )
        si = np.zeros(len(xy), dtype=int)
        total, count = slab_binned_msd(xy, si, 1, lag=1)
        got = diffusion_from_msd(total, count, 1, dt, n_dim=2)[0]
        assert got == pytest.approx(d_true, rel=0.02)


class TestPositionDependentD:
    """The case the old estimator got wrong: D that varies with depth."""

    @staticmethod
    def _two_region_walk(seed, d_core=0.05, d_water=0.30, n=600_000):
        rng = np.random.default_rng(seed)
        z = np.empty(n)
        z[0] = 0.0
        for i in range(1, n):
            d = d_core if abs(z[i - 1]) < 15.0 else d_water
            z[i] = z[i - 1] + rng.normal(scale=np.sqrt(2.0 * d))
            if abs(z[i]) > 40.0:
                z[i] = z[i - 1]
        return z

    def test_both_regions_come_back_at_their_own_D(self):
        z = self._two_region_walk(4)
        edges = np.arange(-40.0, 41.0, 1.0)
        centers = 0.5 * (edges[:-1] + edges[1:])
        d, _ = recover_d(z, edges, 1.0)
        core = np.nanmean(d[np.abs(centers) < 13.0])
        water = np.nanmean(d[(np.abs(centers) > 17.0) & (np.abs(centers) < 30.0)])
        assert core == pytest.approx(0.05, rel=0.05)
        assert water == pytest.approx(0.30, rel=0.05)

    def test_the_contrast_between_them_survives(self):
        """The sojourn estimator flattened this 0.167 to 0.49."""
        z = self._two_region_walk(5)
        edges = np.arange(-40.0, 41.0, 1.0)
        centers = 0.5 * (edges[:-1] + edges[1:])
        d, _ = recover_d(z, edges, 1.0)
        core = np.nanmean(d[np.abs(centers) < 13.0])
        water = np.nanmean(d[(np.abs(centers) > 17.0) & (np.abs(centers) < 30.0)])
        assert core / water == pytest.approx(0.05 / 0.30, rel=0.08)


class TestSlabGrid:
    def test_a_given_range_is_used_verbatim(self):
        edges, centers = build_slab_grid(None, [], "ORP", "z", 1.0,
                                         slab_range=(-40.0, 40.0))
        assert edges[0] == pytest.approx(-40.0)
        assert centers[0] == pytest.approx(-39.5)
        assert len(centers) == 80

    def test_the_width_sets_the_number_of_slabs(self):
        _, centers = build_slab_grid(None, [], "ORP", "z", 2.0,
                                     slab_range=(-10.0, 10.0))
        assert len(centers) == 10

    def test_deriving_the_grid_warns_because_it_is_path_dependent(self):
        """Per-path grids cannot be pooled bin by bin afterwards."""
        with pytest.raises(Exception):  # noqa: B017 - no trajectories to read
            with pytest.warns(UserWarning, match="slab-range"):
                build_slab_grid("missing.gro", ["missing.xtc"], "ORP", "z", 1.0)


class TestLateralPBC:
    """The permeant's lateral position is an *origin*, not just a value.

    Two ways it used to be wrong, both worth a regression test because both
    are silent: these trajectories are not written with ``-pbc mol/whole``, so
    the permeant's own atoms can sit on opposite sides of the box; and a path
    is several .xtc segments that the unwrapper has to see as one continuous
    series. Either defect injects a single ~L displacement, which in a 60 A
    box is ~3600 A^2 against a real per-frame dr^2 of a few A^2 - an 800-fold
    outlier landing in whichever slab the crossing happened in.
    """

    BOX = np.array([60.0, 60.0, 100.0])

    def test_a_molecule_split_across_the_boundary_is_made_whole(self):
        # Atoms at x = -1, 0, +1 about the boundary, stored wrapped. The
        # unwrapped centroid is x = 0; a naive mean lands 20 A away.
        pos = np.array([[59.0, 30.0, 50.0],
                        [0.0,  30.0, 50.0],
                        [1.0,  30.0, 50.0]])
        assert pos[:, 0].mean() == pytest.approx(20.0)

        got = _whole_centroid(pos, self.BOX)
        assert got[0] % 60.0 == pytest.approx(0.0, abs=1e-9)
        assert got[1] == pytest.approx(30.0)
        assert got[2] == pytest.approx(50.0)

    def test_it_is_a_no_op_on_a_molecule_that_is_already_whole(self):
        """So it only ever changes the frames that were wrong."""
        pos = np.array([[10.0, 20.0, 30.0], [12.0, 24.0, 33.0]])
        assert _whole_centroid(pos, self.BOX) == pytest.approx(pos.mean(axis=0))

    def test_degenerate_groups_do_not_raise(self):
        one = np.array([[1.0, 2.0, 3.0]])
        assert _whole_centroid(one, self.BOX) == pytest.approx(one[0])
        assert np.all(np.isnan(_whole_centroid(np.empty((0, 3)), self.BOX)))
        # A zero box length means "not periodic here", not a division by zero.
        pos = np.array([[10.0, 20.0, 30.0], [12.0, 24.0, 33.0]])
        assert _whole_centroid(pos, np.zeros(3)) == pytest.approx(pos.mean(axis=0))

    @staticmethod
    def _unwrap_series(segments, box_xy, reset_per_segment):
        """The frame loop's unwrapping, with the reset where it used to be."""
        jump, prev, out = np.zeros(2), None, []
        for seg in segments:
            if reset_per_segment:
                jump, prev = np.zeros(2), None
            for raw in seg:
                if prev is None:
                    prev, jump = raw.copy(), np.zeros(2)
                    out.append(raw.copy())
                else:
                    out.append(_unwrap_xy_pbc(raw, prev, jump, box_xy).copy())
        return np.array(out)

    def test_the_unwrap_state_survives_a_segment_junction(self):
        box_xy = self.BOX[:2]
        # A permeant drifting +2 A/frame in x, crossing the boundary inside
        # the first segment, then continuing across the junction.
        raw = [np.array([57.0, 5.0]), np.array([59.0, 5.0]), np.array([1.0, 5.0]),
               np.array([3.0, 5.0]),  np.array([5.0, 5.0]),  np.array([7.0, 5.0])]
        segments = [raw[:3], raw[3:]]

        carried = self._unwrap_series(segments, box_xy, reset_per_segment=False)
        dr2 = np.sum(np.diff(carried, axis=0) ** 2, axis=1)
        assert dr2 == pytest.approx(np.full(5, 4.0))

        # What the per-file reset did: the jump accumulated in segment one is
        # discarded, so the junction window reports a whole box length.
        reset = self._unwrap_series(segments, box_xy, reset_per_segment=True)
        dr2_reset = np.sum(np.diff(reset, axis=0) ** 2, axis=1)
        assert dr2_reset[2] == pytest.approx(58.0 ** 2)

    def test_a_whole_centroid_keeps_the_jumps_clean_multiples_of_L(self):
        """Which is what makes _unwrap_xy_pbc's 0.5*L cutoff trustworthy.

        A split molecule puts a ~L/2 glitch into the raw series - sitting
        right on the cutoff, so it can equally trigger a phantom correction
        or slip under a real one.
        """
        box_xy = self.BOX[:2]
        # Two atoms 2 A apart, walking across the boundary one A at a time.
        frames = []
        for x in (57.0, 58.0, 59.0, 60.0, 61.0, 62.0):
            atoms = np.array([[x % 60.0, 5.0], [(x + 2.0) % 60.0, 5.0]])
            frames.append(atoms)

        naive = np.array([f.mean(axis=0) for f in frames])
        whole = np.array([_whole_centroid(f, box_xy) for f in frames])

        naive_out = self._unwrap_series([naive], box_xy, reset_per_segment=False)
        whole_out = self._unwrap_series([whole], box_xy, reset_per_segment=False)

        # The molecule moves 1 A per frame throughout.
        assert np.sum(np.diff(whole_out, axis=0) ** 2, axis=1) == pytest.approx(
            np.full(5, 1.0)
        )
        assert np.max(np.sum(np.diff(naive_out, axis=0) ** 2, axis=1)) > 100.0


def caged_walk(n, d, sigma2, dt, rng):
    """Brownian(d) plus an iid bounded jitter of variance sigma2.

    MSD(tau) = 2 d tau + 2 sigma2 exactly, for every tau >= dt. That is the
    textbook model of what a short lag actually sees in a membrane: a genuinely
    diffusive component plus a fast, BOUNDED one (vibration, ballistic flight,
    rattling inside a transient cage of lipid tails). D_app = d + sigma2/tau,
    so the single-lag estimate reads high and gets worse as tau shrinks, while
    the intercept as 1/tau -> 0 is d.
    """
    walk = np.cumsum(rng.normal(scale=np.sqrt(2.0 * d * dt), size=n))
    return walk + rng.normal(scale=np.sqrt(sigma2), size=n)


def lag_scan(z, si, n_slabs, dt, n_lags=6):
    """(D_app, window counts) per slab for lags 1..n_lags, as the worker does."""
    d_app, counts = [], []
    for k in range(1, n_lags + 1):
        total, count = slab_binned_msd(z, si, n_slabs, k)
        d_app.append(diffusion_from_msd(total, count, k, dt))
        counts.append(count)
    return np.stack(d_app, axis=-1), np.stack(counts, axis=-1)


class TestTheLagScanFit:
    """A single short lag cannot tell D from a fast bounded component.

    The whole point of scanning lags is that D_app(tau) = D + c/(2 tau) is a
    straight line against 1/tau whose intercept is D, so the contamination
    separates itself out instead of having to be assumed absent.
    """

    def test_the_intercept_recovers_D_where_one_lag_cannot(self):
        rng = np.random.default_rng(10)
        d_true, sigma2, dt = 0.05, 0.05, 1.0
        z = caged_walk(2_000_000, d_true, sigma2, dt, rng)
        si = np.zeros(len(z), dtype=int)

        d_app, counts = lag_scan(z, si, 1, dt)
        # The single lag is off by 100% here, by construction.
        assert d_app[0, 0] == pytest.approx(d_true + sigma2, rel=0.02)

        fit, _, _ = fit_diffusion_intercept(d_app, weights=counts)
        assert fit[0] == pytest.approx(d_true, rel=0.02)

    def test_the_intercept_is_exact_on_a_clean_ladder(self):
        ks = np.arange(1, 7, dtype=float)
        d, c = 0.037, 0.21
        d_app = d + c / (2.0 * ks)
        fit, slope, resid = fit_diffusion_intercept(d_app, ks=ks)
        assert fit == pytest.approx(d)
        assert 2.0 * slope == pytest.approx(c)     # slope is c/2 at dt=lag=1
        assert resid == pytest.approx(0.0, abs=1e-12)

    def test_a_purely_diffusive_ladder_is_left_alone(self):
        """No fast component, nothing for the intercept to remove."""
        d_app = np.full(6, 0.2)
        fit, slope, _ = fit_diffusion_intercept(d_app)
        assert fit == pytest.approx(0.2)
        assert slope == pytest.approx(0.0, abs=1e-12)

    def test_two_lags_extrapolate_in_one_over_tau_not_tau(self):
        """2*D(2tau) - D(tau). The tau-space reflex goes the wrong way."""
        d, c = 0.05, 0.10
        y1 = d + c / 2.0
        y2 = d + c / 4.0
        fit, _, _ = fit_diffusion_intercept(np.array([y1, y2]))
        assert fit == pytest.approx(2.0 * y2 - y1)
        assert fit == pytest.approx(d)
        assert 2.0 * y1 - y2 != pytest.approx(d)

    def test_the_fit_is_weighted_by_window_count(self):
        """Long lags have fewer windows and must not drag the intercept."""
        ks = np.array([1.0, 2.0, 3.0])
        d_app = np.array([0.15, 0.10, 0.0833333])      # d = 0.05, c = 0.2
        heavy = fit_diffusion_intercept(d_app, weights=np.array([100.0, 50.0, 1.0]), ks=ks)[0]
        flat  = fit_diffusion_intercept(d_app, ks=ks)[0]
        assert heavy == pytest.approx(0.05, abs=2e-4)
        assert flat == pytest.approx(0.05, abs=2e-4)   # exact ladder: same answer

    def test_a_non_affine_ladder_is_flagged_by_the_residual(self):
        """The slab has no diffusive window; the number should not be trusted."""
        ks = np.arange(1, 7, dtype=float)
        clean = 0.05 + 0.1 / (2.0 * ks)
        bent = clean * np.array([1.0, 0.7, 1.4, 0.6, 1.5, 0.8])
        _, _, r_clean = fit_diffusion_intercept(clean, ks=ks)
        _, _, r_bent = fit_diffusion_intercept(bent, ks=ks)
        assert r_clean == pytest.approx(0.0, abs=1e-12)
        assert r_bent > 0.01

    def test_slabs_without_two_usable_lags_are_nan(self):
        """One point is not an extrapolation."""
        one_lag = np.array([[0.1, np.nan, np.nan]])
        fit, _, resid = fit_diffusion_intercept(one_lag)
        assert np.isnan(fit[0]) and np.isnan(resid[0])

        no_weight = np.array([[0.1, 0.2, 0.3]])
        fit2, _, _ = fit_diffusion_intercept(no_weight, weights=np.zeros((1, 3)))
        assert np.isnan(fit2[0])

    def test_it_fits_a_whole_profile_at_once(self):
        """Shape (n_bins, n_lags) in, (n_bins,) out - the aggregate's layout."""
        ks = np.arange(1, 7, dtype=float)
        d = np.array([0.02, 0.05, 0.20])
        d_app = d[:, None] + 0.1 / (2.0 * ks)[None, :]
        fit, _, _ = fit_diffusion_intercept(d_app, ks=ks)
        assert fit == pytest.approx(d)

    def test_position_dependent_D_survives_the_fit(self):
        """The case that matters: the bias is a constant offset in D_app, so
        it is a LARGER relative error where D is small - which is exactly
        where exp(beta dG) is largest in the resistance integral."""
        rng = np.random.default_rng(11)
        dt, sigma2 = 1.0, 0.04
        n = 400_000
        z = np.empty(n)
        z[0] = 0.0
        for i in range(1, n):
            d = 0.02 if abs(z[i - 1]) < 15.0 else 0.20
            z[i] = z[i - 1] + rng.normal(scale=np.sqrt(2.0 * d * dt))
        obs = z + rng.normal(scale=np.sqrt(sigma2), size=n)

        edges = np.arange(-40.0, 41.0, 5.0)
        centers = 0.5 * (edges[:-1] + edges[1:])
        si = slab_index(obs, edges)
        d_app, counts = lag_scan(obs, si, len(edges) - 1, dt)
        fit, _, _ = fit_diffusion_intercept(d_app, weights=counts)

        core = np.abs(centers) < 12.0
        assert np.nanmean(fit[core]) == pytest.approx(0.02, rel=0.25)
        # The single lag is high by ~sigma2/tau, i.e. by more than D itself.
        assert np.nanmean(d_app[core, 0]) > 2.0 * np.nanmean(fit[core])


class TestLagColumnNames:
    def test_the_first_two_keep_their_historical_names(self):
        """So permeation_process and every saved aggregate keep working."""
        assert lag_column("D_z", 1) == "D_z"
        assert lag_column("D_z", 2) == "D_z_lag2"
        assert lag_column("D_z", 3) == "D_z_lag3"
        assert lag_column("D_xy", 1) == "D_xy"
        assert lag_column("D_xy", 4) == "D_xy_lag4"


class TestChoosingTheFittingRange:
    """Which lags to accumulate and which to FIT are separate decisions.

    Every per-lag column is stored, so scanning wide is nearly free, but the
    range that is actually affine only becomes visible afterwards. Selecting
    it is therefore an aggregation-time choice: a re-fit is a re-aggregation,
    never a recompute.
    """

    AVAILABLE = [1, 2, 3, 4, 5, 6, 7, 8]

    def test_the_default_is_every_scanned_lag(self):
        assert parse_lag_selection(None, self.AVAILABLE) == self.AVAILABLE
        assert parse_lag_selection("  ", self.AVAILABLE) == self.AVAILABLE

    def test_the_spellings(self):
        assert parse_lag_selection("2-6", self.AVAILABLE) == [2, 3, 4, 5, 6]
        assert parse_lag_selection("3-", self.AVAILABLE) == [3, 4, 5, 6, 7, 8]
        assert parse_lag_selection("-4", self.AVAILABLE) == [1, 2, 3, 4]
        assert parse_lag_selection("1,3,5", self.AVAILABLE) == [1, 3, 5]
        assert parse_lag_selection(" 2 , 4 ", self.AVAILABLE) == [2, 4]

    def test_asking_for_lags_that_were_never_scanned_is_not_an_error(self):
        """It intersects; only the result being too small is a problem."""
        assert parse_lag_selection("2-99", [1, 2, 3]) == [2, 3]

    def test_fewer_than_two_lags_is_refused_with_the_reason(self):
        with pytest.raises(ValueError, match="at least two"):
            parse_lag_selection("5", self.AVAILABLE)
        with pytest.raises(ValueError, match="at least two"):
            parse_lag_selection("20-30", self.AVAILABLE)

    def test_an_unreadable_spec_says_what_is_expected(self):
        for bad in ("x-3", "2-y", "a,b"):
            with pytest.raises(ValueError, match="expected a range"):
                parse_lag_selection(bad, self.AVAILABLE)

    def test_dropping_the_smeared_long_lags_restores_the_value(self):
        """The reason the option exists.

        A slab sitting on a gradient in D has windows that straddle it once
        they are long enough, pulling D_app toward the neighbouring region.
        Those lags are still affine-looking individually but bend the ladder,
        and because they sit closest to 1/tau = 0 they have real pull on the
        intercept.
        """
        ks = np.arange(1, 9, dtype=float)
        clean = 0.30 + 0.1 / (2.0 * ks)
        smeared = clean - 0.02 * np.maximum(ks - 3, 0)

        all_lags, _, resid_all = fit_diffusion_intercept(smeared, ks=ks)
        keep = np.array(parse_lag_selection("1-3", list(range(1, 9)))) - 1
        short, _, resid_short = fit_diffusion_intercept(smeared[keep], ks=ks[keep])

        assert all_lags < 0.28                       # dragged down
        assert short == pytest.approx(0.30)          # recovered
        assert resid_all > 10 * max(resid_short, 1e-12)
