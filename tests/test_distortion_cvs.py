"""The membrane-distortion CVs must report distortion, not its absence.

The local-deformation test pins down the flaw these CVs were fixed for: a
permeant that clears the phosphates out of its own 1 nm disc used to read as
an undeformed membrane (exactly 0), because the empty disc fell back to the
leaflet mean. Curvature is checked against a surface whose curvature is known
in closed form, and the water wire against chains built by hand.

Units are nm throughout, as in generate-cvs.
"""

import numpy as np
import pytest

from chiroflux.cv_generation import (
    compute_local_curvature,
    compute_local_deformation,
    compute_water_wire,
)

L = 6.4  # lateral box edge, about the size of the 66-lipid-per-leaflet systems


def _leaflet_grid(z_of_xy, n=8):
    """One leaflet of n x n phosphates on a regular grid, as (1, 3N)."""
    g = (np.arange(n) + 0.5) * L / n
    x, y = np.meshgrid(g, g, indexing="ij")
    x, y = x.ravel(), y.ravel()
    return np.column_stack([x, y, z_of_xy(x, y)]).reshape(1, -1)


def _box(lz=10.0):
    return np.array([[L, L, lz]])


class TestLocalDeformation:
    def test_flat_leaflet_reads_zero(self):
        p = _leaflet_grid(lambda x, y: np.full_like(x, 2.0))
        d, z, deep, n = compute_local_deformation(
            np.array([[L / 2, L / 2]]), p, _box(), z_mid=np.array([0.0])
        )
        assert d[0] == pytest.approx(0.0)
        assert z[0] == pytest.approx(2.0)
        assert deep[0] == pytest.approx(0.0)
        assert n[0] > 0

    def test_empty_disc_is_not_reported_as_undeformed(self):
        """Headgroups pushed out of the 1 nm disc and sagging just beyond it:
        the old fallback returned exactly 0 here."""
        def z(x, y):
            r = np.hypot(x - L / 2, y - L / 2)
            return np.where(r < 1.6, 1.4, 2.0)

        p = _leaflet_grid(z)
        xy = p.reshape(-1, 3)[:, :2]
        p = p.reshape(-1, 3)[np.hypot(*(xy - L / 2).T) >= 1.0].reshape(1, -1)

        d, _, deep, n = compute_local_deformation(
            np.array([[L / 2, L / 2]]), p, _box(), z_mid=np.array([0.0])
        )
        assert n[0] == 0
        assert d[0] < -0.1
        # Measured from the leaflet mean, which the dimple itself lowers.
        assert deep[0] == pytest.approx(1.4 - p.reshape(-1, 3)[:, 2].mean())

    def test_deepest_points_toward_the_midplane_in_both_leaflets(self):
        rng = np.random.default_rng(1)
        jitter = rng.normal(scale=0.05, size=64)
        upper = _leaflet_grid(lambda x, y: 2.0 + jitter)
        lower = _leaflet_grid(lambda x, y: -2.0 - jitter)
        xy, mid = np.array([[L / 2, L / 2]]), np.array([0.0])

        d_u, _, deep_u, _ = compute_local_deformation(xy, upper, _box(), mid)
        d_l, _, deep_l, _ = compute_local_deformation(xy, lower, _box(), mid)
        # The deepest phosphate is at least as far in as the patch mean...
        assert deep_u[0] <= d_u[0]
        assert deep_l[0] >= d_l[0]
        # ...and the mirror image gives the mirror value.
        assert deep_l[0] == pytest.approx(-deep_u[0])

    def test_patch_is_found_across_the_periodic_boundary(self):
        p = _leaflet_grid(lambda x, y: np.where(x < 0.5, 1.5, 2.0))
        d, *_ = compute_local_deformation(
            np.array([[L - 0.05, L / 2]]), p, _box(), z_mid=np.array([0.0])
        )
        # Lifted from +2 toward the dimpled column on the far side of x = 0.
        assert d[0] < 0.0


class TestLocalCurvature:
    def test_flat_leaflet_has_no_curvature(self):
        p = _leaflet_grid(lambda x, y: np.full_like(x, 2.0))
        H = compute_local_curvature(np.array([[1.0, 2.0]]), p, _box())
        assert H[0] == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("x0", [0.0, L / 4, L / 2])
    def test_matches_a_sinusoid_in_closed_form(self, x0):
        A, k = 0.3, 2 * np.pi / L
        p = _leaflet_grid(lambda x, y: 2.0 + A * np.cos(k * x))
        H = compute_local_curvature(np.array([[x0, 1.0]]), p, _box())
        hx = -A * k * np.sin(k * x0)
        hxx = -A * k**2 * np.cos(k * x0)
        assert H[0] == pytest.approx(hxx / (2 * (1 + hx**2) ** 1.5), rel=1e-6, abs=1e-9)

    def test_short_wavelength_protrusions_are_not_curvature(self):
        """A one-lipid checkerboard is below the 2 nm cutoff."""
        def z(x, y):
            i = np.round(x / (L / 8) - 0.5).astype(int)
            j = np.round(y / (L / 8) - 0.5).astype(int)
            return 2.0 + 0.2 * (-1.0) ** (i + j)

        H = compute_local_curvature(np.array([[L / 2, L / 2]]), _leaflet_grid(z), _box())
        assert H[0] == pytest.approx(0.0, abs=1e-8)

    def test_too_few_phosphates_widen_the_cutoff_instead_of_nan(self):
        p = _leaflet_grid(lambda x, y: 2.0 + 0.3 * np.cos(2 * np.pi * x / L), n=3)
        H = compute_local_curvature(np.array([[0.0, 0.0]]), p, _box())
        assert np.isfinite(H[0])
        assert H[0] < 0.0  # on the crest, still curved the right way


class TestWaterWire:
    Z_TOP, Z_BOT = 1.5, -1.5
    BOX = np.array([L, L, 10.0])

    def _chain(self, z_values, x=1.0, y=1.0):
        return np.column_stack([np.full(len(z_values), x),
                                np.full(len(z_values), y), z_values])

    def test_continuous_chain_to_the_interface_is_a_wire(self):
        perm = np.array([[1.0, 1.0, 0.0]])
        water = self._chain(np.arange(0.3, 2.0, 0.28))
        wire, cluster = compute_water_wire(perm, water, self.Z_TOP, self.Z_BOT, self.BOX)
        assert wire == 1.0
        assert cluster == np.sum(np.abs(water[:, 2]) < 1.5)

    def test_broken_chain_is_an_isolated_pocket(self):
        perm = np.array([[1.0, 1.0, 0.0]])
        z = np.r_[0.3, 0.58, 0.86, 1.5, 1.78]  # 0.64 nm gap between 0.86 and 1.5
        water = self._chain(z)
        wire, cluster = compute_water_wire(perm, water, self.Z_TOP, self.Z_BOT, self.BOX)
        assert wire == 0.0
        assert cluster == 3.0

    def test_dry_permeant(self):
        perm = np.array([[1.0, 1.0, 0.0]])
        water = self._chain(np.array([2.0, 2.3]))
        assert compute_water_wire(perm, water, self.Z_TOP, self.Z_BOT, self.BOX) == (0.0, 0.0)

    def test_permeant_outside_the_core_with_water_is_a_wire(self):
        perm = np.array([[1.0, 1.0, 2.0]])
        water = self._chain(np.array([2.3]))
        assert compute_water_wire(perm, water, self.Z_TOP, self.Z_BOT, self.BOX) == (1.0, 0.0)

    def test_chain_links_across_the_lateral_boundary(self):
        perm = np.array([[0.05, 1.0, 0.0]])
        water = np.vstack([
            [[L - 0.2, 1.0, 0.0]],                       # first shell, via PBC
            self._chain(np.arange(0.28, 2.0, 0.28), x=L - 0.2),
        ])
        wire, _ = compute_water_wire(perm, water, self.Z_TOP, self.Z_BOT, self.BOX)
        assert wire == 1.0

    def test_core_water_stored_at_its_periodic_image_is_still_core(self):
        """Raw z = 4.25 in a 4 nm box is 0.25 nm from the midplane."""
        perm = np.array([[1.0, 1.0, 0.0]])
        water = np.array([[1.0, 1.0, 4.25]])
        box = np.array([L, L, 4.0])
        assert compute_water_wire(perm, water, self.Z_TOP, self.Z_BOT, box) == (0.0, 1.0)
