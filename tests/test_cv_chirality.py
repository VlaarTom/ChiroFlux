"""The chirality CVs must actually be chiral, and must resolve what they claim.

Two things are worth pinning down here. First, that the handed quantities are
genuine pseudoscalars: reflecting the whole system has to flip their sign, or
they cannot distinguish an enantiomer from its mirror image at all and the
whole construction is pointless. Second, that the shell decomposition recovers
structure the r-integrated hCN destroys - the motivation for adding it, from
Nandi's result that the chiral preference reverses sign with separation.

Coordinates are laid out in a deliberately huge box so the minimum-image
convention is inert and the geometry under test is the geometry as written.
"""

import numpy as np
import pytest

from chiroflux.cv_generation import (
    HCN_SHELL_EDGES,
    compute_group_orientations,
    compute_handed_cn,
    compute_local_chain_tilt,
)

BOX = 100.0


def _tetra(ca=(0.0, 0.0, 0.0)):
    """[CA, N, C, CB, HA] with n = (N-CA) x (C-CA) pointing along +z."""
    ca = np.asarray(ca, dtype=float)
    n = ca + np.array([1.0, 0.0, 0.0])
    c = ca + np.array([0.0, 1.0, 0.0])
    cb = ca + np.array([0.0, 0.0, 1.0])
    ha = ca - np.array([0.0, 0.0, 1.0])
    return np.concatenate([ca, n, c, cb, ha])[None, :]


def _lipids(*offsets):
    """Lipid positions given as offsets from the origin, as (1, 3N)."""
    return np.concatenate([np.asarray(o, dtype=float) for o in offsets])[None, :]


def _mirror_z(coords):
    """Reflect a flat (n_frames, 3N) coordinate array through the z = 0 plane."""
    out = np.asarray(coords, dtype=float).copy()
    out = out.reshape(out.shape[0], -1, 3)
    out[:, :, 2] *= -1.0
    return out.reshape(coords.shape)


class TestHandedCNIsAPseudoscalar:
    def test_reflection_flips_the_sign(self):
        """The defining property: no sign flip, no enantiomer discrimination."""
        tetra = _tetra()
        lip = _lipids((0.3, 0.1, 0.4), (-0.2, 0.35, -0.5), (0.6, -0.1, 0.2))

        h, ncos, r, shells = compute_handed_cn(
            tetra, lip, BOX, shell_edges=HCN_SHELL_EDGES
        )
        h_m, ncos_m, r_m, shells_m = compute_handed_cn(
            _mirror_z(tetra), _mirror_z(lip), BOX, shell_edges=HCN_SHELL_EDGES
        )

        assert h_m[0] == pytest.approx(-h[0])
        assert ncos_m[0] == pytest.approx(-ncos[0])
        assert np.allclose(shells_m, -shells)
        # Distance is a true scalar and must be untouched by the reflection.
        assert r_m[0] == pytest.approx(r[0])

    def test_an_achiral_arrangement_gives_zero(self):
        """Neighbours placed symmetrically about the n-plane cannot be handed."""
        tetra = _tetra()
        lip = _lipids((0.4, 0.0, 0.3), (0.4, 0.0, -0.3))
        h, _, _, shells = compute_handed_cn(tetra, lip, BOX, shell_edges=HCN_SHELL_EDGES)
        assert h[0] == pytest.approx(0.0, abs=1e-12)
        assert np.allclose(shells, 0.0, atol=1e-12)


class TestSwitchingFunction:
    def test_matches_the_closed_form(self):
        """(1-x^6)/(1-x^12) collapses to 1/(1+x^6); the code must agree."""
        tetra = _tetra()
        d0 = 0.5
        offsets = [(0.0, 0.0, 0.21), (0.0, 0.0, 0.63), (0.0, 0.0, 0.97)]
        lip = _lipids(*offsets)
        h, _, _, _ = compute_handed_cn(tetra, lip, BOX, d0=d0)

        # n is +z and every neighbour is on the +z axis, so every cos is +1.
        expected = sum(1.0 / (1.0 + (abs(o[2]) / d0) ** 6) for o in offsets)
        assert h[0] == pytest.approx(expected, rel=1e-9)

    def test_weight_is_one_half_at_d0(self):
        """The n_exp/m_exp special case must equal the limit, not merely avoid 0/0."""
        tetra = _tetra()
        lip = _lipids((0.0, 0.0, 0.5))
        h, _, _, _ = compute_handed_cn(tetra, lip, BOX, d0=0.5)
        assert h[0] == pytest.approx(0.5)


class TestShellDecomposition:
    def test_shells_partition_the_signed_cosines(self):
        tetra = _tetra()
        lip = _lipids((0.0, 0.0, 0.2), (0.0, 0.0, -0.6), (0.0, 0.0, 1.0))
        _, _, _, shells = compute_handed_cn(tetra, lip, BOX, shell_edges=HCN_SHELL_EDGES)
        # +1 in [0,0.5), -1 in [0.5,0.8), +1 in [0.8,1.2)
        assert shells[0].tolist() == pytest.approx([1.0, -1.0, 1.0])

    def test_recovers_structure_the_integrated_hcn_cancels(self):
        """The reason this decomposition exists.

        One close neighbour of one handedness against six intermediate ones of
        the opposite handedness: the switching function weights them to almost
        exactly zero, so the integrated hCN reports "no chirality here" while
        the shells show a strong, sign-reversing arrangement.
        """
        tetra = _tetra()
        close = [(0.0, 0.0, 0.3)]  # r = 0.3, cos = +1, lands in shell 1
        # Six distinct neighbours at r = 0.65 on a narrow cone about -z, so each
        # contributes cos = -0.9 and they land together in shell 2.
        cos_t, sin_t = 0.9, np.sqrt(1.0 - 0.9**2)
        far = [
            (0.65 * sin_t * np.cos(phi), 0.65 * sin_t * np.sin(phi), -0.65 * cos_t)
            for phi in np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ]
        lip = _lipids(*(close + far))

        h, _, _, shells = compute_handed_cn(tetra, lip, BOX, shell_edges=HCN_SHELL_EDGES)

        assert abs(h[0]) < 0.1, "the integrated hCN should have very nearly cancelled"
        assert shells[0, 0] == pytest.approx(1.0)
        assert shells[0, 1] == pytest.approx(-6 * cos_t)

    def test_empty_shell_is_zero_not_nan(self):
        """NaN here would void the frame for every all-finite analysis."""
        tetra = _tetra()
        lip = _lipids((0.0, 0.0, 0.2))
        _, _, _, shells = compute_handed_cn(tetra, lip, BOX, shell_edges=HCN_SHELL_EDGES)
        assert np.isfinite(shells).all()
        assert shells[0, 1] == 0.0 and shells[0, 2] == 0.0


class TestNearestNeighbour:
    def test_reports_the_closest_neighbour(self):
        tetra = _tetra()
        lip = _lipids((0.0, 0.0, 0.9), (0.0, 0.0, -0.25), (0.0, 0.0, 0.6))
        _, ncos, r, _ = compute_handed_cn(tetra, lip, BOX)
        assert r[0] == pytest.approx(0.25)
        assert ncos[0] == pytest.approx(-1.0)

    def test_no_neighbours_gives_nan(self):
        tetra = _tetra()
        lip = np.zeros((1, 0))
        _, ncos, r, _ = compute_handed_cn(tetra, lip, BOX)
        assert np.isnan(ncos[0]) and np.isnan(r[0])


class TestLocalChainTilt:
    """Upper leaflet: glycerols at z = +1, chains reaching down toward z = 0."""

    Z_MID = np.array([0.0])

    def _patch(self, tilt_dx, n=8, z_c2=1.0, chain=-0.6):
        """n lipids in a ring around the origin, chains offset by tilt_dx in x."""
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        c2 = np.stack([0.3 * np.cos(ang), 0.3 * np.sin(ang), np.full(n, z_c2)], axis=1)
        cc = c2 + np.array([tilt_dx, 0.0, chain])
        return c2.reshape(1, -1), cc.reshape(1, -1)

    def test_untilted_chains_point_along_the_normal(self):
        c2, cc = self._patch(tilt_dx=0.0)
        cos_tilt, count = compute_local_chain_tilt(
            np.zeros((1, 2)), c2, cc, self.Z_MID, BOX
        )
        assert cos_tilt[0] == pytest.approx(-1.0)  # chains point down, toward z_mid
        assert count[0] == 8

    def test_recovers_a_planted_tilt_angle(self):
        c2, cc = self._patch(tilt_dx=0.6, chain=-0.6)  # 45 degrees off -z
        # radius must cover the displaced chain carbons too, or the lateral
        # cutoff clips the far side of the patch and biases the mean.
        cos_tilt, _ = compute_local_chain_tilt(
            np.zeros((1, 2)), c2, cc, self.Z_MID, BOX, radius=3.0
        )
        assert np.degrees(np.arccos(cos_tilt[0])) == pytest.approx(135.0)

    def test_ignores_chain_atoms_from_the_far_leaflet(self):
        """The reason chain atoms are selected by z rather than by .ndx label.

        C210 splits 54/56 in the real index file because chains interdigitate
        through the frame-0 z cutoff, so the pooled input contains the opposite
        leaflet's chains. Those sit outside the midplane-to-glycerol band and
        must not reach the mean.
        """
        c2, cc = self._patch(tilt_dx=0.0)
        # Mirror-image lower leaflet: glycerols at -1, chains reaching up.
        _, cc_far = self._patch(tilt_dx=0.0, z_c2=-1.0, chain=0.6)
        pooled = np.hstack([cc, cc_far])

        clean, _ = compute_local_chain_tilt(np.zeros((1, 2)), c2, cc, self.Z_MID, BOX)
        mixed, _ = compute_local_chain_tilt(np.zeros((1, 2)), c2, pooled, self.Z_MID, BOX)
        assert mixed[0] == pytest.approx(clean[0])

    def test_reflection_flips_the_tilt(self):
        c2, cc = self._patch(tilt_dx=0.4)
        up, _ = compute_local_chain_tilt(np.zeros((1, 2)), c2, cc, self.Z_MID, BOX)
        down, _ = compute_local_chain_tilt(
            np.zeros((1, 2)), _mirror_z(c2), _mirror_z(cc), -self.Z_MID, BOX
        )
        assert down[0] == pytest.approx(-up[0])

    def test_sparse_patch_falls_back_to_the_whole_leaflet(self):
        """A near-empty patch must not void the frame for the whole analysis."""
        c2, cc = self._patch(tilt_dx=0.0, n=8)
        # Permeant far from every lipid laterally: nothing is within 1.0 nm.
        far = np.array([[40.0, 40.0]])
        cos_tilt, count = compute_local_chain_tilt(
            far, c2, cc, self.Z_MID, BOX, radius=1.0
        )
        assert count[0] == 0
        assert np.isfinite(cos_tilt[0])
        assert cos_tilt[0] == pytest.approx(-1.0)  # the global tilt


class TestGroupOrientations:
    def _inputs(self, o_dir, n_dir, ring_z):
        ca = np.zeros(3)
        ref = np.concatenate([ca, np.asarray(o_dir, float), np.asarray(n_dir, float),
                              np.zeros(3), np.zeros(3)])[None, :]
        # ring = CA, CB, CG, CD, N; centroid displaced along z by ring_z
        ring = np.concatenate([
            ca,
            ca + np.array([0.1, 0.0, ring_z]),
            ca + np.array([-0.1, 0.0, ring_z]),
            ca + np.array([0.0, 0.1, ring_z]),
            ca + np.array([0.0, -0.1, ring_z]),
        ])[None, :]
        return ref, ring

    def test_group_vectors_project_onto_the_normal(self):
        ref, ring = self._inputs(o_dir=(0, 0, 1.0), n_dir=(0, 0, -1.0), ring_z=1.0)
        cos_ring, cos_o, cos_n = compute_group_orientations(ref, ring, BOX)
        assert cos_o[0] == pytest.approx(1.0)
        assert cos_n[0] == pytest.approx(-1.0)
        assert cos_ring[0] > 0.99  # centroid sits at +z

    def test_reflection_flips_every_mode_cosine(self):
        ref, ring = self._inputs(o_dir=(0.3, 0.2, 0.9), n_dir=(-0.4, 0.1, -0.8), ring_z=0.7)
        up = compute_group_orientations(ref, ring, BOX)
        down = compute_group_orientations(_mirror_z(ref), _mirror_z(ring), BOX)
        for a, b in zip(up, down):
            assert b[0] == pytest.approx(-a[0])

    def test_ring_centroid_is_unwrapped_across_the_boundary(self):
        """A ring straddling a periodic image must not average to the box centre."""
        box = 4.0
        ca = np.array([0.05, 0.0, 0.0])
        ring_atoms = [ca,
                      ca + np.array([0.1, 0.0, 0.2]),
                      np.array([box - 0.05, 0.0, 0.2]),   # wrapped neighbour
                      ca + np.array([0.0, 0.1, 0.2]),
                      ca + np.array([0.0, -0.1, 0.2])]
        ring = np.concatenate(ring_atoms)[None, :]
        ref = np.concatenate([ca, ca + np.array([0, 0, 1.0]),
                              ca + np.array([0, 0, -1.0]),
                              np.zeros(3), np.zeros(3)])[None, :]
        cos_ring, _, _ = compute_group_orientations(ref, ring, box)
        # All ring atoms sit at z = +0.2 relative to CA, so the centroid is
        # squarely along +z; a wrapped x coordinate must not tilt it.
        assert cos_ring[0] == pytest.approx(1.0)
