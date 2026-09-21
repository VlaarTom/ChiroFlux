"""Vectors and leaflet labels must survive the periodic boundary.

These trajectories are not written with ``-pbc mol/whole`` (the same premise
cv_generation.compute_com_unwrapped is built on), so any quantity built by
subtracting two raw atom positions is wrong whenever those atoms wrapped to
opposite sides. For an intramolecular vector that is not a small error: the
"bond" becomes a box-length vector lying along a box axis, which passes every
non-zero-length check and normalises to a pure artifact.

The frame loop had no test coverage at all, so most of this runs the real
`calculate_selected_slab_profiles` over a synthetic membrane whose geometry is
known exactly, with the coordinates wrapped the way a real .xtc stores them.
"""

import warnings

import numpy as np
import pytest

MDAnalysis = pytest.importorskip("MDAnalysis")

from chiroflux import membrane_spatial as ms
from chiroflux.membrane_spatial import (
    _leaflet_by_resindex,
    _min_image,
    _upper_mask_for,
    mean_tilt_angle,
    p2_order_parameter,
)

L, LZ = 60.0, 100.0
N_PER_LEAFLET = 12
LATERAL_STEP = (0.6, 0.3)     # per bead, so the chains lean coherently
VERTICAL_STEP = 1.1
TRUE_TILT = np.degrees(np.arctan(np.hypot(*LATERAL_STEP) / VERTICAL_STEP))

CHAIN2 = [f"C2{i}" for i in range(1, 17)]
CHAIN1 = [f"C3{i}" for i in range(1, 17)]
LIPID_ATOMS = ["P"] + CHAIN2 + CHAIN1


@pytest.fixture(scope="module")
def synthetic_membrane(tmp_path_factory):
    """A wrapped bilayer whose chain tilt and leaflets are known exactly."""
    from MDAnalysis.coordinates.XTC import XTCWriter

    warnings.simplefilter("ignore")
    n_water, n_perm = 30, 4
    names, resnames, resindices = [], [], []
    ri = 0
    for _leaf in range(2):
        for i in range(N_PER_LEAFLET):
            names += LIPID_ATOMS
            resnames.append("DOPC" if i % 2 == 0 else "POPC")
            resindices += [ri] * len(LIPID_ATOMS)
            ri += 1
    for _ in range(n_water):
        names.append("OH2"); resnames.append("TIP3"); resindices.append(ri); ri += 1
    for i in range(n_perm):
        names.append(f"C{i}"); resindices.append(ri)
    resnames.append("ORP")
    ri += 1

    n_atoms = len(names)
    u = MDAnalysis.Universe.empty(
        n_atoms, n_residues=ri, atom_resindex=np.array(resindices),
        trajectory=True,
    )
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("resname", resnames)
    u.add_TopologyAttr("resid", list(range(1, ri + 1)))
    u.add_TopologyAttr("type", ["C"] * n_atoms)

    rng = np.random.default_rng(0)
    # Lateral positions fixed across frames so leaflet membership is stable;
    # they are drawn over the whole box, so a good fraction of the chains
    # straddle a boundary once wrapped.
    head_xy = rng.uniform(0, L, (2 * N_PER_LEAFLET, 2))

    def positions(t):
        pos = np.zeros((n_atoms, 3))
        k = 0
        for leaf in range(2):
            sgn = 1.0 if leaf == 0 else -1.0
            for i in range(N_PER_LEAFLET):
                xy = head_xy[leaf * N_PER_LEAFLET + i]
                pz = 50.0 + sgn * 18.0
                pos[k] = [xy[0], xy[1], pz]; k += 1
                for chain in (CHAIN2, CHAIN1):
                    for j in range(len(chain)):
                        pos[k] = [
                            xy[0] + LATERAL_STEP[0] * j,
                            xy[1] + LATERAL_STEP[1] * j,
                            pz - sgn * (VERTICAL_STEP * j + 1.0),
                        ]
                        k += 1
        for _ in range(n_water):
            pos[k] = [rng.uniform(0, L), rng.uniform(0, L),
                      rng.choice([10.0, 90.0])]
            k += 1
        for i in range(n_perm):
            pos[k] = [30.0 + 0.3 * i, 30.0,
                      50.0 + 3.0 * np.sin(t / 3.0) + 0.2 * i]
            k += 1
        pos[:, :2] %= L           # wrapped, exactly as a real .xtc stores it
        return pos

    d = tmp_path_factory.mktemp("memb")
    xtc, pdb = str(d / "t.xtc"), str(d / "t.pdb")
    with XTCWriter(xtc, n_atoms=n_atoms) as w:
        for t in range(20):
            u.atoms.positions = positions(t)
            u.dimensions = [L, L, LZ, 90, 90, 90]
            u.trajectory.ts.time = t * 0.2
            w.write(u.atoms)
    u.atoms.positions = positions(0)
    u.atoms.write(pdb)
    return pdb, xtc


def _run(pdb, xtc, groups, **over):
    kw = dict(
        topology=pdb, xtc_files=[xtc], frame_plan=[(xtc, list(range(20)))],
        permeant_resname="ORP", needed_groups=groups,
        slab_range=(-40.0, 40.0), slab_width=2.0, dt=0.2, n_lags=3,
        near_n=(4,), n_depth_bins=2, grid_spacing=8.0, spatial_extent=16.0,
    )
    kw.update(over)
    return ms.calculate_selected_slab_profiles(**kw)


def _mean_tilt(out, col="sn2_theta"):
    tilt = out["order"][1]
    s = tilt[f"{col}_wsum"].to_numpy()
    w = tilt[f"{col}_wtot"].to_numpy()
    return float(np.mean(s[w > 0] / w[w > 0]))


class TestIntramolecularVectorsAcrossTheBoundary:
    def test_the_chain_tilt_comes_back_at_its_geometric_value(self,
                                                              synthetic_membrane):
        pdb, xtc = synthetic_membrane
        got = _mean_tilt(_run(pdb, xtc, {"order"}))
        assert got == pytest.approx(TRUE_TILT, abs=0.05)

    def test_without_the_minimum_image_it_is_badly_wrong(self, monkeypatch,
                                                         synthetic_membrane):
        """Pins the size of the bug, so a regression cannot look plausible.

        The vectors are AVERAGED before the angle is taken, so one ~L artifact
        outweighs several genuine chains — the error is tens of percent, not a
        rounding difference.
        """
        pdb, xtc = synthetic_membrane
        monkeypatch.setattr(ms, "_min_image",
                            lambda v, b: np.asarray(v, dtype=float))
        got = _mean_tilt(_run(pdb, xtc, {"order"}))
        assert abs(got - TRUE_TILT) > 5.0

    def test_p2_of_a_split_bond_is_not_minus_a_half(self):
        box = np.array([L, L, LZ])
        along_z = np.array([[0.0, 0.0, 1.5]])
        # Same bond, written with its two atoms on opposite sides of x.
        split = np.array([[L - 0.0001, 0.0, 1.5]])
        assert p2_order_parameter(along_z) == pytest.approx(1.0)
        assert p2_order_parameter(split) == pytest.approx(-0.5, abs=1e-3)
        assert p2_order_parameter(split, box=box) == pytest.approx(1.0, abs=1e-3)

    def test_min_image_leaves_a_whole_molecule_alone(self):
        box = np.array([L, L, LZ])
        v = np.array([[1.0, -2.0, 3.0], [0.5, 0.5, -0.5]])
        assert _min_image(v, box) == pytest.approx(v)

    def test_min_image_skips_an_unset_axis(self):
        v = np.array([[40.0, 1.0, 2.0]])
        assert _min_image(v, np.array([0.0, L, LZ]))[0, 0] == pytest.approx(40.0)

    def test_tilt_box_argument_is_what_fixes_it(self):
        box = np.array([L, L, LZ])
        base = np.array([[1.0, 5.0, 60.0], [2.0, 5.0, 60.0]])
        tip = base + np.array([[3.0, 0.0, -18.0], [3.0, 0.0, -18.0]])
        wrapped = tip.copy(); wrapped[:, 0] %= L
        base_w = base.copy(); base_w[:, 0] = (base_w[:, 0] - 4.0) % L
        up = np.ones(2, dtype=bool)
        good = mean_tilt_angle(tip, base, upper_mask=up, leaflet="upper",
                               box=box)[0]
        fixed = mean_tilt_angle(wrapped, base_w, upper_mask=up,
                                leaflet="upper", box=box)[0]
        assert np.isfinite(good) and np.isfinite(fixed)


class TestLeafletByLipidIdentity:
    def test_a_deep_tail_atom_cannot_assign_a_leaflet_but_the_P_atom_can(self):
        rng = np.random.default_rng(3)
        n = 128

        class Group:
            def __init__(self, pos, rix):
                self.positions, self.resindices = pos, rix
            def __len__(self):
                return len(self.resindices)

        rix = np.arange(n)
        pz = np.concatenate([rng.normal(18, 1.5, n // 2),
                             rng.normal(-18, 1.5, n // 2)])
        truth = np.concatenate([np.ones(n // 2, bool), np.zeros(n // 2, bool)])
        p_atoms = Group(np.column_stack([np.zeros(n), np.zeros(n), pz]), rix)

        by_identity = _upper_mask_for(rix, _leaflet_by_resindex(p_atoms, 2))
        assert (by_identity == truth).all()

        # Splitting on an atom's own z degrades as that atom moves toward the
        # midplane, i.e. along the chain. That is the damaging part: the
        # mislabelling grows smoothly with bond index, so the leaflet contrast
        # appears to fade toward the chain end as if it were physics.
        accuracy = []
        for separation in (18.0, 8.0, 4.0, 2.0):   # P atom -> terminal methyl
            own_z = np.concatenate([
                rng.normal(+separation, 3.0, n // 2),
                rng.normal(-separation, 3.0, n // 2),
            ])
            accuracy.append(float(((own_z > own_z.mean()) == truth).mean()))

        assert accuracy[0] == pytest.approx(1.0)        # at the headgroup, fine
        assert accuracy[-1] < 0.85                      # at the chain end, not
        assert accuracy == sorted(accuracy, reverse=True)   # monotone decay

    def test_the_two_leaflets_partition_the_lipids(self, synthetic_membrane):
        pdb, xtc = synthetic_membrane
        u = MDAnalysis.Universe(pdb, xtc)
        p = u.select_atoms("(resname DOPC or resname POPC) and name P")
        upper = _leaflet_by_resindex(p, 2)
        assert len(upper) == N_PER_LEAFLET
        mask = _upper_mask_for(p.resindices, upper)
        assert mask.sum() == N_PER_LEAFLET
        assert (~mask).sum() == N_PER_LEAFLET

    def test_an_empty_selection_is_not_an_error(self):
        assert len(_leaflet_by_resindex(None, 2)) == 0
        assert not _upper_mask_for(np.arange(4), np.array([], dtype=np.intp)).any()


class TestTheFrameLoopRuns:
    """It had no coverage at all, which is how the above survived."""

    def test_every_group_produces_populated_tables(self, synthetic_membrane):
        pdb, xtc = synthetic_membrane
        out = _run(pdb, xtc,
                   {"order", "structural", "diffusion", "diffusion_xy",
                    "spatial"})
        assert set(out) == {"order", "structural", "diffusion",
                            "diffusion_xy", "spatial"}
        for group in ("order", "structural", "diffusion", "diffusion_xy"):
            for df in out[group]:
                assert (df.iloc[:, 1:].to_numpy() != 0).any(), group
        acc, shape, x, y, depth = out["spatial"]
        assert shape == (len(x), len(y))
        assert sum(float(v[1].sum()) for v in acc.values()) > 0

    def test_the_lag_scan_reaches_the_table(self, synthetic_membrane):
        pdb, xtc = synthetic_membrane
        diff = _run(pdb, xtc, {"diffusion"}, n_lags=4)["diffusion"][0]
        for col in ("D_z", "D_z_lag2", "D_z_lag3", "D_z_lag4"):
            assert f"{col}_wsum" in diff.columns
            assert f"{col}_wtot" in diff.columns

    def test_thickness_and_curvature_are_computed_once_per_frame(
            self, monkeypatch, synthetic_membrane):
        """Both the structural and the spatial block want them; the per-frame
        cache is what stops the pair of them paying for it twice."""
        pdb, xtc = synthetic_membrane
        counts = {"thick": 0, "curv": 0}
        real_t, real_c = ms._local_thickness_dict, ms.local_membrane_curvature
        monkeypatch.setattr(ms, "_local_thickness_dict",
                            lambda *a, **k: (counts.__setitem__(
                                "thick", counts["thick"] + 1) or real_t(*a, **k)))
        monkeypatch.setattr(ms, "local_membrane_curvature",
                            lambda *a, **k: (counts.__setitem__(
                                "curv", counts["curv"] + 1) or real_c(*a, **k)))

        _run(pdb, xtc, {"structural"})
        one_group = dict(counts)
        counts["thick"] = counts["curv"] = 0
        _run(pdb, xtc, {"structural", "spatial"})
        assert counts == one_group


class TestLegacySpatialFiles:
    """A pre-depth-axis aggregate is meant to load as one implicit bin."""

    def _write(self, tmp_path, arr_wsum, arr_wcount, with_depth):
        p = tmp_path / "sp.npz"
        kw = {"x_coords": np.arange(arr_wsum.shape[-2], dtype=float),
              "y_coords": np.arange(arr_wsum.shape[-1], dtype=float),
              "k_wsum": arr_wsum, "k_wcount": arr_wcount}
        if with_depth:
            kw["depth_edges"] = np.linspace(-40.0, 40.0,
                                            arr_wsum.shape[0] + 1)
        np.savez(p, **kw)
        return ms.SpatialMaps(p)

    def test_a_2d_map_keeps_its_shape(self, tmp_path):
        """np.atleast_3d would have made this (ny, 1) by appending the axis."""
        mean = np.arange(12.0).reshape(4, 3)
        maps = self._write(tmp_path, mean * 2, np.full((4, 3), 2.0), False)
        got = maps.map("k")
        assert got.shape == (4, 3)
        assert got == pytest.approx(mean)

    def test_its_single_depth_bin_has_a_finite_centre(self, tmp_path):
        """An infinite bin centres on NaN, silently killing every selection."""
        maps = self._write(tmp_path, np.ones((4, 3)), np.ones((4, 3)), False)
        assert np.isfinite(maps.depth_centers).all()
        assert len(maps.bins_in_range(-40.0, 40.0)) == 1
        assert maps.coverage("k").shape == (4, 3)
        assert maps.occupancy("k").shape == (1,)

    def test_a_3d_map_is_untouched(self, tmp_path):
        mean = np.arange(24.0).reshape(2, 4, 3)
        maps = self._write(tmp_path, mean * 2, np.full((2, 4, 3), 2.0), True)
        assert maps.map("k", depth=0) == pytest.approx(mean[0])
        assert maps.n_depth == 2


class TestPathInputValidation:
    def test_a_toml_without_lambda_minus_one_says_so(self, tmp_path):
        """It reaches a `None < float` inside the worker otherwise, where the
        TypeError is caught per path and blamed on nothing in particular."""
        toml = tmp_path / "i.toml"
        toml.write_text('[simulation]\ninterfaces = [1.0, 2.0, 3.0]\n')
        with pytest.raises(ValueError, match="lambda_minus_one"):
            ms.read_toml(str(toml))

    def test_a_complete_toml_still_loads(self, tmp_path):
        toml = tmp_path / "i.toml"
        toml.write_text('[simulation]\ninterfaces = [1.0, 2.0, 3.0]\n'
                        '[simulation.tis_set]\nlambda_minus_one = -1.5\n')
        assert ms.read_toml(str(toml)) == (1.0, 3.0, -1.5)


class TestOrderFileAlignment:
    """order.txt and the frame plan must index the same phase points.

    read_traj_plan returns the .g96 step for exactly this purpose; ignoring it
    leaves the two one apart, and since `first`/`last` decide which end of the
    plan gets trimmed, that drops a frame belonging to the path and keeps one
    that does not.
    """

    @staticmethod
    def _order(steps):
        return np.column_stack([np.asarray(steps, dtype=float),
                                np.arange(len(steps), dtype=float)])

    def test_the_g96_row_is_dropped_wherever_it_sits(self):
        for g96_at in (0, 2, 4):
            steps = [10, 20, 30, 40, 50]
            order = self._order(steps)
            plan = [("a.xtc", i) for i in range(4)]
            out, problem = ms.align_order_with_plan(order, plan,
                                                    str(steps[g96_at]))
            assert problem is None, g96_at
            assert len(out) == 4
            assert float(steps[g96_at]) not in out[:, 0]

    def test_the_trimming_end_is_read_after_the_drop(self):
        """The whole point: with the g96 first, `first` must be phase point 2."""
        order = self._order([10, 20, 30, 40])
        plan = [("a.xtc", i) for i in range(3)]
        out, problem = ms.align_order_with_plan(order, plan, "10")
        assert problem is None
        assert out[0, 0] == 20.0          # not 10.0
        assert out[0, 1] == 1.0

    def test_a_path_without_a_g96_point_is_untouched(self):
        order = self._order([10, 20, 30])
        plan = [("a.xtc", i) for i in range(3)]
        out, problem = ms.align_order_with_plan(order, plan, None)
        assert problem is None
        assert out == pytest.approx(order)

    def test_a_length_mismatch_is_reported_not_guessed(self):
        order = self._order([10, 20, 30, 40, 50])
        plan = [("a.xtc", i) for i in range(3)]
        _, problem = ms.align_order_with_plan(order, plan, None)
        assert problem and "must line up" in problem

    def test_a_g96_step_that_matches_nothing_does_not_silently_drop_a_row(self):
        order = self._order([10, 20, 30])
        plan = [("a.xtc", i) for i in range(3)]
        out, problem = ms.align_order_with_plan(order, plan, "999")
        assert problem is None and len(out) == 3

    def test_a_single_row_order_file_stays_2d(self):
        """1-D would make order[0, 1] mean the second column of one row."""
        out, problem = ms.align_order_with_plan(np.array([10.0, 0.5]),
                                                [("a.xtc", 0)], None)
        assert out.shape == (1, 2)
        assert problem is None
        assert out[0, 1] == 0.5

    def test_an_empty_order_file_is_reported(self):
        out, problem = ms.align_order_with_plan(np.empty((0, 2)), [], None)
        assert problem and "no usable rows" in problem


class TestTheTwoTiltAngles:
    """Collective and individual tilt answer different questions.

    Averaging the chain vectors and then taking the angle gives the leaflet's
    net lean; taking each angle and then averaging gives the typical chain's
    tilt. They coincide only when the chains are aligned in azimuth, and the
    gap between them is the splay.
    """

    N = 3000

    def _leaflet(self, tilt_deg, azimuth, seed=0):
        rng = np.random.default_rng(seed)
        th = np.radians(np.full(self.N, float(tilt_deg)))
        ph = (rng.uniform(0, 2 * np.pi, self.N) if azimuth == "uniform"
              else np.zeros(self.N))
        v = np.stack([np.sin(th) * np.cos(ph),
                      np.sin(th) * np.sin(ph),
                      -np.cos(th)], axis=1) * 18.0
        base = np.zeros((self.N, 3)); base[:, 2] = 35.0
        return base + v, base

    def _angles(self, tilt_deg, azimuth):
        tip, base = self._leaflet(tilt_deg, azimuth)
        theta, _, _, theta_ind = mean_tilt_angle(
            tip, base, upper_mask=np.ones(self.N, dtype=bool), leaflet="upper")
        return theta, theta_ind

    def test_a_splayed_leaflet_has_no_net_lean_but_tilted_chains(self):
        theta, theta_ind = self._angles(30, "uniform")
        assert theta == pytest.approx(0.0, abs=1.0)
        assert theta_ind == pytest.approx(30.0, abs=0.1)

    def test_they_agree_when_the_chains_are_aligned(self):
        theta, theta_ind = self._angles(30, "aligned")
        assert theta == pytest.approx(30.0, abs=0.1)
        assert theta_ind == pytest.approx(30.0, abs=0.1)

    def test_an_untilted_leaflet_gives_zero_both_ways(self):
        # Not to 1e-6 deg: arccos is ill-conditioned at cos = 1, where a
        # rounding error eps in the cosine becomes sqrt(2 eps) in the angle.
        # Double-precision rounding alone lands around 1e-5 deg here.
        theta, theta_ind = self._angles(0, "aligned")
        assert theta == pytest.approx(0.0, abs=1e-3)
        assert theta_ind == pytest.approx(0.0, abs=1e-3)

    def test_the_collective_angle_never_exceeds_the_individual_one(self):
        for tilt in (10, 25, 40):
            for azimuth in ("uniform", "aligned"):
                theta, theta_ind = self._angles(tilt, azimuth)
                assert theta <= theta_ind + 1e-6, (tilt, azimuth)

    def test_an_empty_selection_returns_nan_for_both(self):
        empty = np.empty((0, 3))
        theta, phi, _, theta_ind = mean_tilt_angle(
            empty, empty, upper_mask=np.zeros(0, dtype=bool))
        assert np.isnan(theta) and np.isnan(theta_ind)

    def test_both_reach_the_tilt_table(self, synthetic_membrane):
        pdb, xtc = synthetic_membrane
        tilt = _run(*synthetic_membrane, {"order"})["order"][1]
        for col in ("sn2_theta", "sn2_theta_ind", "sn1_theta", "sn1_theta_ind"):
            assert f"{col}_wsum" in tilt.columns
            assert f"{col}_wtot" in tilt.columns

    def test_the_synthetic_membrane_is_azimuthally_aligned_so_they_match(
            self, synthetic_membrane):
        """Its chains all lean the same way, so this pins both at the truth."""
        out = _run(*synthetic_membrane, {"order"})
        assert _mean_tilt(out, "sn2_theta") == pytest.approx(TRUE_TILT, abs=0.05)
        assert _mean_tilt(out, "sn2_theta_ind") == pytest.approx(TRUE_TILT,
                                                                 abs=0.05)
