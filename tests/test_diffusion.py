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
    build_slab_grid,
    diffusion_from_msd,
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
