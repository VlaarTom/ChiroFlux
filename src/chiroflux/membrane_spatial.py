import os

os.environ["OMP_NUM_THREADS"] = "1"
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Optional

import matplotlib
import MDAnalysis as mda
import numpy as np
import pandas as pd
import tomli
import typer
from MDAnalysis.analysis import distances
from scipy.integrate import trapezoid
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

from . import panels

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── PARALLELISM CONFIGURATION ─────────────────────────────────────────────────
ERROR_LOG = "errors_order.log"
ORDER_OUT = "Memb_OP"
PLOT_OUT  = "plots/memb_op"
# ─────────────────────────────────────────────────────────────────────────────

topol_file = '../gromacs_input/topol.tpr'


# ═════════════════════════════════════════════════════════════════════════════
# Chain-bond definitions  (CHARMM36, DOPC / POPC)
# ═════════════════════════════════════════════════════════════════════════════

CHAIN_BONDS = {
    'sn2': [
        ('C21', 'C22'), ('C22', 'C23'), ('C23', 'C24'), ('C24', 'C25'),
        ('C25', 'C26'), ('C26', 'C27'), ('C27', 'C28'), ('C28', 'C29'),
        ('C29', 'C210'),('C210','C211'),('C211','C212'),('C212','C213'),
        ('C213','C214'),('C214','C215'),('C215','C216'),
    ],
    'sn1': [
        ('C31', 'C32'), ('C32', 'C33'), ('C33', 'C34'), ('C34', 'C35'),
        ('C35', 'C36'), ('C36', 'C37'), ('C37', 'C38'), ('C38', 'C39'),
        ('C39', 'C310'),('C310','C311'),('C311','C312'),('C312','C313'),
        ('C313','C314'),('C314','C315'),('C315','C316'),
    ],
}

TILT_ENDPOINTS = {
    'sn2': ('C21', 'C216'),
    'sn1': ('C31', 'C316'),
}

PHOSPHORUS_NAME   = 'P'
WATER_OXYGEN_NAME = 'OH2'
WATER_RESNAME     = 'TIP3'
MEMBRANE_NORMAL   = np.array([0.0, 0.0, 1.0])

# Default window for the water z-density profile, in Å *relative to the
# bilayer midplane* (not absolute box z). Fixed rather than box-derived so
# that every frame, every path and every simulation share one grid — see
# water_defect_profile.
WATER_Z_RANGE     = (-40.0, 40.0)

# ── Leaflet labels used throughout ────────────────────────────────────────────
# 'both'  = all lipids pooled (the original behaviour, kept for backward compat)
# 'upper' = lipids whose P-atom z > midplane
# 'lower' = lipids whose P-atom z < midplane
ALL_LEAFLETS = ('both', 'upper', 'lower')


# ═════════════════════════════════════════════════════════════════════════════
# P2 order parameter
# ═════════════════════════════════════════════════════════════════════════════


def _split_values(text, flag, cast, n):
    """Parse a comma-separated multi-value option.

    The original argparse used ``nargs``, i.e. space-separated values. Typer
    has no direct equivalent for a fixed-arity option, so these are given as
    "a,b" instead; `n` is the required count, or None for "one or more".
    """
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        return list(text)
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if n is not None and len(parts) != n:
        raise typer.BadParameter(
            f"{flag} needs {n} comma-separated values, got {text!r}"
        )
    if not parts:
        raise typer.BadParameter(f"{flag} needs at least one value")
    try:
        return [cast(p) for p in parts]
    except ValueError:
        raise typer.BadParameter(f"{flag}: bad value in {text!r}") from None


def p2_order_parameter(bond_vectors, normal=MEMBRANE_NORMAL):
    """
    Compute the P2 order parameter.

    P2 uses cos²θ so it is insensitive to the sign of the bond vector —
    upper and lower leaflet bonds can safely be mixed.  No leaflet split
    is required here.
    """
    vecs  = np.asarray(bond_vectors, dtype=float)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    valid = (norms[:, 0] > 1e-6)
    if not np.any(valid):
        return np.nan
    v = vecs[valid] / norms[valid]
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    cos_theta = v @ n
    return float(0.5 * (3.0 * np.mean(cos_theta ** 2) - 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# Mean tilt angle  — leaflet-aware
# ═════════════════════════════════════════════════════════════════════════════

def mean_tilt_angle(tip_positions, base_positions, normal=MEMBRANE_NORMAL,
                    leaflet='both', upper_mask=None):
    """
    Compute the mean tilt angle (θ) and azimuthal angle (φ) for a set of
    lipid chains.

    Vector convention
    ------------------
    tip/base follow CHARMM36 acyl-chain numbering (e.g. C21 → C216): the
    base atom sits at the glycerol/headgroup end, near the leaflet surface,
    and the tip atom is the terminal methyl, deep in the hydrophobic core.
    So tip − base points *inward*, toward the bilayer midplane, for BOTH
    leaflets — upper-leaflet chains point roughly in −z, lower-leaflet
    chains roughly in +z.

    Leaflet handling
    ----------------
    Averaging all vectors together causes near-perfect cancellation, giving
    a spurious θ ≈ 90°. Naively comparing every vector against a single +z
    reference is also wrong: it reports θ ≈ 180° − (true tilt) for an
    untilted lipid in either leaflet, since the physical vector points
    inward rather than outward.

    Fix: split by leaflet, compute the mean vector *within each leaflet*
    against that leaflet's own inward reference (−n upper, +n lower), then
    average the resulting angles.

    Parameters
    ----------
    tip_positions  : (N, 3)
    base_positions : (N, 3)
    normal         : (3,) membrane normal unit vector
    leaflet        : 'both' | 'upper' | 'lower'
        'both'  — compute per-leaflet then average (default, corrected)
        'upper' — only upper-leaflet lipids
        'lower' — only lower-leaflet lipids
    upper_mask     : (N,) bool or None
        Ground-truth leaflet membership for each row (True = upper),
        typically sliced from the *full-population* midplane split.
        Callers passing an already-filtered subset (a single leaflet, a
        near-permeant subset, etc.) MUST supply this — re-deriving the
        split from the subset's own mean z is only valid for a genuine
        full, symmetric population and silently mis-splits small or
        single-leaflet subsets. If None, falls back to deriving the split
        from this call's own base_positions (only correct for a full,
        both-leaflet population).
    """
    vecs     = np.asarray(tip_positions,  dtype=float) \
             - np.asarray(base_positions, dtype=float)
    base_pos = np.asarray(base_positions, dtype=float)
    n        = np.asarray(normal, dtype=float)
    n        = n / np.linalg.norm(n)

    if upper_mask is not None:
        upper_sel = np.asarray(upper_mask, dtype=bool)
    else:
        midplane  = base_pos[:, 2].mean()
        upper_sel = base_pos[:, 2] > midplane
    lower_sel = ~upper_sel

    # Determine which leaflet masks to process. Each mask carries its own
    # *inward* reference direction: -n for the upper leaflet, +n for the
    # lower leaflet, matching the physical direction tip-base vectors point
    # in each leaflet (see docstring). Comparing against the wrong sign
    # gives θ ≈ 180° - true_tilt instead of θ ≈ true_tilt for an untilted
    # lipid, and naively using a single shared +n for both leaflets causes
    # near-perfect cancellation on averaging, collapsing to θ ≈ 90°.
    if leaflet == 'upper':
        masks = [(upper_sel, -n)]
    elif leaflet == 'lower':
        masks = [(lower_sel, n)]
    else:                    # 'both' — per-leaflet averages then combined
        masks = [(upper_sel, -n), (lower_sel, n)]

    theta_results = []
    last_mean_vec = vecs.mean(axis=0)   # fallback

    for mask, local_n in masks:
        if not mask.any():
            continue
        v        = vecs[mask]
        mean_vec = v.mean(axis=0)
        v_hat    = mean_vec / (np.linalg.norm(mean_vec) + 1e-12)
        cos_t    = float(np.clip(v_hat @ local_n, -1.0, 1.0))
        theta_results.append(float(np.degrees(np.arccos(cos_t))))
        last_mean_vec = mean_vec

    if not theta_results:
        return np.nan, np.nan, vecs.mean(axis=0)

    theta_deg = float(np.mean(theta_results))

    # φ — use the last (or only) leaflet's mean vector; for 'both' use
    # the orientation-corrected combined mean vector so φ is well-defined.
    if leaflet == 'both':
        # Orient each leaflet's mean vector to point along its own inward
        # reference before averaging. Flipping against a single shared +n
        # (instead of each leaflet's own -n/+n) would treat a physically
        # consistent lateral tilt (both leaflets leaning the same lateral
        # direction) as opposite XY signs, cancelling azimuthal information
        # and leaving φ undefined.
        oriented = []
        for mask, local_n in [(upper_sel, -n), (lower_sel, n)]:
            if not mask.any():
                continue
            mv = vecs[mask].mean(axis=0)
            if (mv @ local_n) < 0:
                mv = -mv
            oriented.append(mv)
        last_mean_vec = np.mean(oriented, axis=0) if oriented else vecs.mean(axis=0)

    v_plane    = last_mean_vec - (last_mean_vec @ n) * n
    plane_norm = np.linalg.norm(v_plane)
    if plane_norm > 1e-6:
        y_axis  = np.array([0.0, 1.0, 0.0])
        cos_phi = float(np.clip(v_plane @ y_axis / plane_norm, -1.0, 1.0))
        phi_deg = float(np.degrees(np.arccos(cos_phi)))
    else:
        phi_deg = float('nan')

    return theta_deg, phi_deg, last_mean_vec


# ═════════════════════════════════════════════════════════════════════════════
# Helper: assign a leaflet mask from base-atom positions
# ═════════════════════════════════════════════════════════════════════════════

def _leaflet_masks(base_positions):
    """
    Return (upper_mask, lower_mask) boolean arrays for N atoms, splitting
    on the mean z-coordinate of base_positions.

    Parameters
    ----------
    base_positions : (N, 3) array

    Returns
    -------
    upper_mask : (N,) bool — True for atoms in the upper leaflet
    lower_mask : (N,) bool
    """
    z       = np.asarray(base_positions, dtype=float)[:, 2]
    mid     = z.mean()
    upper   = z > mid
    return upper, ~upper


def _pbc_minimum_image_2d(vec_xy, box_xy):
    """
    Wrap a set of 2-D displacement vectors into the minimum-image
    convention given lateral box lengths box_xy = (Lx, Ly).
    """
    v = np.asarray(vec_xy, dtype=float).copy()
    for i in range(2):
        L = box_xy[i]
        v[:, i] -= np.round(v[:, i] / L) * L
    return v


# ═════════════════════════════════════════════════════════════════════════════
# Local membrane thickness
# ═════════════════════════════════════════════════════════════════════════════

def local_membrane_thickness(
    universe,
    lipid_resnames      = ('DOPC', 'POPC'),
    phosphorus_name     = PHOSPHORUS_NAME,
    bilayer_normal_axis = 2,
    n_neighbors         = 6,
):
    """
    Local membrane thickness from P–P distances between leaflets.

    Returns
    -------
    upper_xy   : (N_upper, 2) lateral positions of upper-leaflet P atoms
    thickness  : (N_upper,) per-upper-atom thickness values
    mean_thick : float — mean over all upper atoms

    Now also returns per-leaflet mean thickness so callers can accumulate
    them separately.

    Returns (extended)
    ------------------
    upper_xy, thickness, mean_thick,
    upper_xy, thickness_upper, mean_thick_upper,
    lower_xy, thickness_lower, mean_thick_lower

    For simplicity the function returns a dict when called from within the
    main loop so that adding leaflet data does not break existing callers.

    Backward-compatible signature: positional return is unchanged
    (upper_xy, thickness, mean_thick) so old callers keep working.
    The dict is available via the keyword-only `_return_dict` parameter.
    """
    ax   = bilayer_normal_axis
    axes = [i for i in range(3) if i != ax]

    lipid_sel = " or ".join(f"resname {r}" for r in lipid_resnames)
    p_atoms   = universe.select_atoms(
        f"({lipid_sel}) and name {phosphorus_name}"
    )
    if len(p_atoms) == 0:
        return np.empty((0, 2)), np.array([]), np.nan

    p_pos    = p_atoms.positions
    midplane = p_pos[:, ax].mean()

    upper_mask = p_pos[:, ax] > midplane
    lower_mask = ~upper_mask

    upper_pos = p_pos[upper_mask]
    lower_pos = p_pos[lower_mask]

    if len(upper_pos) == 0 or len(lower_pos) == 0:
        return np.empty((0, 2)), np.array([]), np.nan

    box     = universe.dimensions
    box_lat = np.array([box[a] for a in axes])

    lower_xy = lower_pos[:, axes] % box_lat[np.newaxis, :]
    upper_xy = upper_pos[:, axes] % box_lat[np.newaxis, :]

    tree = cKDTree(lower_xy, boxsize=box_lat)
    k    = min(n_neighbors, len(lower_pos))
    dists, idx = tree.query(upper_xy, k=k)

    lower_z_mat = lower_pos[idx, ax]
    if k == 1:
        lower_z_mat = lower_z_mat[:, np.newaxis]

    # Thickness at each upper-leaflet P: |z_upper - mean(z_lower_neighbours)|
    thickness = np.abs(upper_pos[:, ax] - lower_z_mat.mean(axis=1))

    # --- NEW: also compute "thickness" viewed from the lower leaflet --------
    # For each lower P-atom, find nearest upper neighbours and record distance.
    tree_upper = cKDTree(upper_xy, boxsize=box_lat)
    k2         = min(n_neighbors, len(upper_pos))
    _, idx2    = tree_upper.query(lower_xy, k=k2)
    upper_z_mat2 = upper_pos[idx2, ax]
    if k2 == 1:
        upper_z_mat2 = upper_z_mat2[:, np.newaxis]
    thickness_lower = np.abs(lower_pos[:, ax] - upper_z_mat2.mean(axis=1))
    lower_xy_out    = lower_pos[:, axes] % box_lat[np.newaxis, :]

    mean_thick       = float(thickness.mean())
    mean_thick_upper = float(thickness.mean())        # same as upper-viewed
    mean_thick_lower = float(thickness_lower.mean())

    # Backward-compatible positional return; extended info available as attrs
    result = (upper_xy, thickness, mean_thick)
    # Attach leaflet-specific data as extra attributes on the tuple via a
    # small wrapper dict returned alongside; callers that only care about the
    # tuple get the old interface for free.
    result._leaflet = {        # type: ignore[attr-defined]  (not used)
        'upper_xy'        : upper_xy,
        'thickness_upper' : thickness,
        'mean_thick_upper': mean_thick_upper,
        'lower_xy'        : lower_xy_out,
        'thickness_lower' : thickness_lower,
        'mean_thick_lower': mean_thick_lower,
    }
    return result


def _local_thickness_dict(universe, lipid_resnames, phosphorus_name, axis):
    """
    Internal helper: call local_membrane_thickness and return a clean dict
    with per-leaflet arrays, so the frame loop can accumulate upper/lower
    separately without repeating the heavy P–P calculation.
    """
    ax   = axis
    axes = [i for i in range(3) if i != ax]

    lipid_sel = " or ".join(f"resname {r}" for r in lipid_resnames)
    p_atoms   = universe.select_atoms(
        f"({lipid_sel}) and name {phosphorus_name}"
    )
    empty = {
        'upper_xy': np.empty((0, 2)), 'thickness_upper': np.array([]),
        'mean_thick_upper': np.nan,
        'lower_xy': np.empty((0, 2)), 'thickness_lower': np.array([]),
        'mean_thick_lower': np.nan,
        'mean_thick_both': np.nan,
        'upper_resindices': np.array([], dtype=np.intp),
        'lower_resindices': np.array([], dtype=np.intp),
    }
    if len(p_atoms) == 0:
        return empty

    p_pos        = p_atoms.positions
    p_resindices = p_atoms.resindices
    midplane     = p_pos[:, ax].mean()

    upper_mask = p_pos[:, ax] > midplane
    lower_mask = ~upper_mask
    upper_pos  = p_pos[upper_mask]
    lower_pos  = p_pos[lower_mask]
    upper_resindices = p_resindices[upper_mask]
    lower_resindices = p_resindices[lower_mask]

    if len(upper_pos) == 0 or len(lower_pos) == 0:
        return empty

    box     = universe.dimensions
    box_lat = np.array([box[a] for a in axes])
    lower_xy = lower_pos[:, axes] % box_lat[np.newaxis, :]
    upper_xy = upper_pos[:, axes] % box_lat[np.newaxis, :]

    n_neigh = 6
    tree_lo  = cKDTree(lower_xy, boxsize=box_lat)
    k        = min(n_neigh, len(lower_pos))
    _, idx   = tree_lo.query(upper_xy, k=k)
    lo_z_mat = lower_pos[idx, ax]
    if k == 1:
        lo_z_mat = lo_z_mat[:, np.newaxis]
    thick_upper = np.abs(upper_pos[:, ax] - lo_z_mat.mean(axis=1))

    tree_up  = cKDTree(upper_xy, boxsize=box_lat)
    k2       = min(n_neigh, len(upper_pos))
    _, idx2  = tree_up.query(lower_xy, k=k2)
    up_z_mat = upper_pos[idx2, ax]
    if k2 == 1:
        up_z_mat = up_z_mat[:, np.newaxis]
    thick_lower = np.abs(lower_pos[:, ax] - up_z_mat.mean(axis=1))

    mean_both  = float(np.concatenate([thick_upper, thick_lower]).mean())

    return {
        'upper_xy'        : upper_xy,
        'thickness_upper' : thick_upper,
        'mean_thick_upper': float(thick_upper.mean()),
        'lower_xy'        : lower_xy,
        'thickness_lower' : thick_lower,
        'mean_thick_lower': float(thick_lower.mean()),
        'mean_thick_both' : mean_both,
        'upper_resindices': upper_resindices,
        'lower_resindices': lower_resindices,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Radial deformation profile around the permeant
# ═════════════════════════════════════════════════════════════════════════════

def radial_deformation_profile(
    universe,
    permeant_resname    = 'ORP',
    lipid_resnames      = ('DOPC', 'POPC'),
    phosphorus_name     = PHOSPHORUS_NAME,
    bilayer_normal_axis = 2,
    r_max               = 40.0,
    n_bins              = 20,
):
    ax   = bilayer_normal_axis
    axes = [i for i in range(3) if i != ax]

    permeant = universe.select_atoms(f"resname {permeant_resname}")
    if len(permeant) == 0:
        nan_arr = np.full(n_bins, np.nan)
        return np.linspace(0, r_max, n_bins), nan_arr, nan_arr
    perm_lateral = permeant.center_of_geometry()[axes]

    lipid_sel = " or ".join(f"resname {r}" for r in lipid_resnames)
    p_atoms   = universe.select_atoms(
        f"({lipid_sel}) and name {phosphorus_name}"
    )
    if len(p_atoms) == 0:
        nan_arr = np.full(n_bins, np.nan)
        return np.linspace(0, r_max, n_bins), nan_arr, nan_arr

    p_pos    = p_atoms.positions
    midplane = p_pos[:, ax].mean()

    upper_mask = p_pos[:, ax] > midplane
    lower_mask = ~upper_mask

    edges   = np.linspace(0.0, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def _radial_deformation(mask, mean_z):
        pos   = p_pos[mask]
        if len(pos) == 0:
            return np.full(n_bins, np.nan)
        lat   = pos[:, axes]
        r_vec = lat - perm_lateral[np.newaxis, :]
        box   = universe.dimensions[:3]
        for i, a in enumerate(axes):
            L = box[a]
            r_vec[:, i] -= np.round(r_vec[:, i] / L) * L
        r     = np.linalg.norm(r_vec, axis=1)
        deform = pos[:, ax] - mean_z
        result = np.full(n_bins, np.nan)
        for bi in range(n_bins):
            in_bin = (r >= edges[bi]) & (r < edges[bi + 1])
            if in_bin.any():
                result[bi] = deform[in_bin].mean()
        return result

    mean_upper = p_pos[upper_mask, ax].mean() if upper_mask.any() else midplane
    mean_lower = p_pos[lower_mask, ax].mean() if lower_mask.any() else midplane

    def_upper = _radial_deformation(upper_mask, mean_upper)
    def_lower = _radial_deformation(lower_mask, mean_lower)

    return centers, def_upper, def_lower


# ═════════════════════════════════════════════════════════════════════════════
# Water defect profile
# ═════════════════════════════════════════════════════════════════════════════

def water_defect_profile(
    universe,
    permeant_resname    = 'ORP',
    lipid_resnames      = ('DOPC', 'POPC'),
    phosphorus_name     = PHOSPHORUS_NAME,
    water_resname       = WATER_RESNAME,
    water_oxygen_name   = WATER_OXYGEN_NAME,
    bilayer_normal_axis = 2,
    r_max               = 20.0,
    n_radial_bins       = 15,
    z_edges             = None,
):
    """
    Radial water density in the hydrophobic core, plus the water density
    profile along the bilayer normal.

    z-profile conventions
    ---------------------
    The z axis is measured **relative to the bilayer midplane** (the mean
    z of all lipid P atoms), not in absolute box coordinates, and the bins
    are supplied by the caller via `z_edges` rather than derived from the
    box.

    Both of those matter. Deriving the grid from `universe.dimensions`
    rebuilt it every frame under the semiisotropic barostat, so bin *i*
    referred to a different physical height in every frame and to a
    different height again in every path — the accumulated profile was
    smeared within a run and unppoolable across runs. And an absolute-z
    axis moves with membrane COM drift and sits at a different height in
    every system, so two simulations could not be overlaid at all.

    Offsets are wrapped into the minimum image before binning. This
    matters whenever the bilayer is not sitting at the centre of the box:
    with a membrane at z ≈ 30 in a 100 Å box, bulk water at z ≈ 95 is
    physically 35 Å *below* the membrane through the periodic image, and
    an unwrapped offset of +65 would push it outside the window entirely.

    `z_edges` should therefore span at most ±Lz/2; bins beyond that are
    unreachable after wrapping and stay empty.

    Assumption
    ----------
    `midplane` is the plain arithmetic mean of the lipid P-atom z, which
    is only meaningful while the bilayer is not itself split across the z
    boundary — for a straddling membrane that mean lands on the antipode.
    This is the same assumption every other analysis here already makes
    (the leaflet split in _local_thickness_dict, local_membrane_curvature
    and the order-parameter loop all take the same mean), so it holds for
    any trajectory centred on the membrane in the usual way. It is not
    re-derived here, because a water profile that disagreed with the
    leaflet assignment used everywhere else would be worse than one that
    shares its limitation.
    """
    ax   = bilayer_normal_axis
    axes = [i for i in range(3) if i != ax]

    if z_edges is None:
        z_edges = np.linspace(WATER_Z_RANGE[0], WATER_Z_RANGE[1], 51)
    z_edges   = np.asarray(z_edges, dtype=float)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    n_z_bins  = len(z_centers)

    lipid_sel = " or ".join(f"resname {r}" for r in lipid_resnames)
    p_atoms   = universe.select_atoms(
        f"({lipid_sel}) and name {phosphorus_name}"
    )
    water_O   = universe.select_atoms(
        f"resname {water_resname} and name {water_oxygen_name}"
    )

    nan_r = np.full(n_radial_bins, np.nan)
    nan_z = np.full(n_z_bins, np.nan)

    if len(p_atoms) == 0 or len(water_O) == 0:
        r_c = 0.5 * (np.linspace(0.0, r_max, n_radial_bins + 1)[:-1]
                     + np.linspace(0.0, r_max, n_radial_bins + 1)[1:])
        return r_c, nan_r, z_centers, nan_z, 0

    p_pos    = p_atoms.positions
    midplane = p_pos[:, ax].mean()

    upper_z  = p_pos[p_pos[:, ax] > midplane, ax].mean()
    lower_z  = p_pos[p_pos[:, ax] < midplane, ax].mean()
    z_lo, z_hi = min(lower_z, upper_z), max(lower_z, upper_z)

    w_pos    = water_O.positions
    core_mask = (w_pos[:, ax] > z_lo) & (w_pos[:, ax] < z_hi)
    core_pos  = w_pos[core_mask]
    n_defect  = int(core_mask.sum())

    permeant   = universe.select_atoms(f"resname {permeant_resname}")
    if len(permeant) > 0:
        perm_lateral = permeant.center_of_geometry()[axes]
    else:
        perm_lateral = np.array([universe.dimensions[axes[0]],
                                  universe.dimensions[axes[1]]]) / 2.0

    r_edges   = np.linspace(0.0, r_max, n_radial_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    shell_areas = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)

    radial_count = np.zeros(n_radial_bins)
    if n_defect > 0:
        lat   = core_pos[:, axes]
        r_vec = lat - perm_lateral[np.newaxis, :]
        box   = universe.dimensions[:3]
        for i, a in enumerate(axes):
            L = box[a]
            r_vec[:, i] -= np.round(r_vec[:, i] / L) * L
        r = np.linalg.norm(r_vec, axis=1)
        radial_count, _ = np.histogram(r[r <= r_max], bins=r_edges)

    with np.errstate(invalid='ignore', divide='ignore'):
        radial_density = np.where(
            shell_areas > 0,
            radial_count.astype(float) / shell_areas,
            np.nan,
        )

    # Midplane-relative offsets, wrapped into the minimum image so that
    # water on the far side of the periodic boundary is binned by its true
    # distance from the bilayer rather than by its raw box coordinate.
    box_z  = universe.dimensions[ax]
    dz_w   = w_pos[:, ax] - midplane
    dz_w  -= np.round(dz_w / box_z) * box_z

    # Per-bin widths rather than a single dz: the grid is caller-supplied
    # and need not be uniform. box_xy stays per-frame — the lateral area
    # genuinely fluctuates under the barostat, so the density must use it.
    bin_widths   = np.diff(z_edges)
    box_xy       = universe.dimensions[axes[0]] * universe.dimensions[axes[1]]
    z_density, _ = np.histogram(dz_w, bins=z_edges)
    z_density    = z_density.astype(float) / (box_xy * bin_widths)

    return r_centers, radial_density, z_centers, z_density, n_defect


# ═════════════════════════════════════════════════════════════════════════════
# Local membrane curvature
# ═════════════════════════════════════════════════════════════════════════════

def local_membrane_curvature(
    universe,
    lipid_resnames      = ('DOPC', 'POPC'),
    phosphorus_name     = PHOSPHORUS_NAME,
    bilayer_normal_axis = 2,
    grid_spacing        = 5.0,
    rbf_smoothing       = 0.0,
    rbf_kernel          = 'thin_plate_spline',
    permeant_resname    = None,
    r_max_curvature     = 40.0,
    n_radial_bins       = 20,
    ghost_margin        = 20.0,
):
    ax   = bilayer_normal_axis
    lat  = [i for i in range(3) if i != ax]

    lipid_sel = " or ".join(f"resname {r}" for r in lipid_resnames)
    p_atoms   = universe.select_atoms(
        f"({lipid_sel}) and name {phosphorus_name}"
    )

    empty = {
        'upper_H': np.array([]), 'upper_K': np.array([]),
        'lower_H': np.array([]), 'lower_K': np.array([]),
        'upper_xy': np.empty((0, 2)), 'lower_xy': np.empty((0, 2)),
        'mean_H_upper': np.nan, 'mean_H_lower': np.nan,
        'mean_K_upper': np.nan, 'mean_K_lower': np.nan,
        'radial_r': None, 'radial_H_upper': None,
        'upper_resindices': np.array([], dtype=np.intp),
        'lower_resindices': np.array([], dtype=np.intp),
    }

    if len(p_atoms) < 4:
        return empty

    p_pos        = p_atoms.positions
    p_resindices = p_atoms.resindices
    midplane     = p_pos[:, ax].mean()
    upper_mask = p_pos[:, ax] > midplane
    lower_mask = ~upper_mask

    box  = universe.dimensions
    Lx   = box[lat[0]]
    Ly   = box[lat[1]]

    def _fit_and_evaluate_pbc(pos_3d):
        n_real = len(pos_3d)
        if n_real < 4:
            return np.full(n_real, np.nan), np.full(n_real, np.nan)

        xy_real = pos_3d[:, lat].copy()
        z_real  = pos_3d[:, ax]

        xy_real[:, 0] %= Lx
        xy_real[:, 1] %= Ly

        offsets = [(-Lx, -Ly), (-Lx, 0), (-Lx, Ly),
                   (0,   -Ly),            (0,   Ly),
                   (Lx,  -Ly), (Lx,  0), (Lx,  Ly)]

        ghost_xy_list = []
        ghost_z_list  = []
        for dx, dy in offsets:
            gx = xy_real[:, 0] + dx
            gy = xy_real[:, 1] + dy
            in_x = (gx > -ghost_margin) & (gx < Lx + ghost_margin)
            in_y = (gy > -ghost_margin) & (gy < Ly + ghost_margin)
            keep = in_x & in_y
            if keep.any():
                ghost_xy_list.append(np.column_stack([gx[keep], gy[keep]]))
                ghost_z_list.append(z_real[keep])

        if ghost_xy_list:
            all_xy = np.vstack([xy_real] + ghost_xy_list)
            all_z  = np.concatenate([z_real] + ghost_z_list)
        else:
            all_xy = xy_real
            all_z  = z_real

        rbf = RBFInterpolator(all_xy, all_z,
                              kernel=rbf_kernel, smoothing=rbf_smoothing)

        eps = 0.5
        xy  = xy_real

        dzdx = (rbf(xy + [eps, 0]) - rbf(xy - [eps, 0])) / (2 * eps)
        dzdy = (rbf(xy + [0, eps]) - rbf(xy - [0, eps])) / (2 * eps)

        z0       = rbf(xy)
        d2zdx2   = (rbf(xy + [eps, 0]) - 2*z0 + rbf(xy - [eps, 0])) / eps**2
        d2zdy2   = (rbf(xy + [0, eps]) - 2*z0 + rbf(xy - [0, eps])) / eps**2
        d2zdxdy  = (rbf(xy + [eps, eps]) - rbf(xy + [eps, -eps])
                    - rbf(xy - [eps, -eps]) + rbf(xy - [eps,  eps])) / (4*eps**2)

        E = 1.0 + dzdx**2
        F = dzdx * dzdy
        G = 1.0 + dzdy**2
        denom = np.sqrt(E * G - F**2)

        H = (E * d2zdy2 - 2*F*d2zdxdy + G*d2zdx2) / (2 * denom**3)
        K = (d2zdx2 * d2zdy2 - d2zdxdy**2) / denom**4

        return H, K

    upper_pos = p_pos[upper_mask].copy()
    lower_pos = p_pos[lower_mask].copy()
    upper_resindices = p_resindices[upper_mask]
    lower_resindices = p_resindices[lower_mask]

    H_upper, K_upper = _fit_and_evaluate_pbc(upper_pos)
    H_lower, K_lower = _fit_and_evaluate_pbc(lower_pos)

    result = {
        'upper_H'     : H_upper,
        'upper_K'     : K_upper,
        'lower_H'     : H_lower,
        'lower_K'     : K_lower,
        'upper_xy'    : upper_pos[:, lat],
        'lower_xy'    : lower_pos[:, lat],
        'mean_H_upper': float(np.nanmean(np.abs(H_upper))),
        'mean_H_lower': float(np.nanmean(np.abs(H_lower))),
        'mean_K_upper': float(np.nanmean(K_upper)),
        'mean_K_lower': float(np.nanmean(K_lower)),
        'radial_r'         : None,
        'radial_H_upper'   : None,
        'upper_resindices' : upper_resindices,
        'lower_resindices' : lower_resindices,
    }

    if permeant_resname is not None:
        permeant = universe.select_atoms(f"resname {permeant_resname}")
        if len(permeant) > 0 and len(H_upper) > 0:
            perm_lat = permeant.center_of_geometry()[lat]
            r_vec    = upper_pos[:, lat] - perm_lat[np.newaxis, :]
            Lats     = np.array([Lx, Ly])
            r_vec   -= np.round(r_vec / Lats[np.newaxis, :]) * Lats[np.newaxis, :]
            r        = np.linalg.norm(r_vec, axis=1)

            r_edges  = np.linspace(0.0, r_max_curvature, n_radial_bins + 1)
            r_centers= 0.5 * (r_edges[:-1] + r_edges[1:])
            H_radial = np.full(n_radial_bins, np.nan)
            for bi in range(n_radial_bins):
                in_bin = (r >= r_edges[bi]) & (r < r_edges[bi+1])
                if in_bin.any():
                    H_radial[bi] = np.nanmean(H_upper[in_bin])

            result['radial_r']       = r_centers
            result['radial_H_upper'] = H_radial

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 2-D spatial mapping via the MOSAICS stamping algorithm
# ═════════════════════════════════════════════════════════════════════════════

SPATIAL_MAP_META = {
    'p2_sn1_chain' : ('RdBu_r',  'P₂ sn-1 chain'),
    'p2_sn2_chain' : ('RdBu_r',  'P₂ sn-2 chain'),
    'theta_sn1'    : ('plasma',  'Tilt θ sn-1 (°)'),
    'theta_sn2'    : ('plasma',  'Tilt θ sn-2 (°)'),
    'phi_sn1'      : ('twilight','Azimuthal φ sn-1 (°)'),
    'phi_sn2'      : ('twilight','Azimuthal φ sn-2 (°)'),
    'thickness'    : ('coolwarm','Thickness (Å)'),
    'curvature_H'  : ('PiYG',   'Mean curvature |H| (Å⁻¹)'),
}


def build_lateral_grid(box_xy, grid_spacing):
    Lx, Ly   = float(box_xy[0]), float(box_xy[1])
    x_coords = np.arange(0.0, Lx, grid_spacing)
    y_coords = np.arange(0.0, Ly, grid_spacing)
    xx, yy   = np.meshgrid(x_coords, y_coords, indexing='ij')
    grid_xy  = np.column_stack([xx.ravel(), yy.ravel()])
    return grid_xy, (len(x_coords), len(y_coords)), x_coords, y_coords


def build_lateral_grid_centered(extent, grid_spacing):
    """
    Build a lateral grid centered at (0, 0) spanning ±extent, for spatial
    maps expressed relative to the permeant rather than in absolute box
    coordinates (see run_spatial in calculate_selected_slab_profiles).
    """
    coords  = np.arange(-extent, extent + grid_spacing, grid_spacing)
    xx, yy  = np.meshgrid(coords, coords, indexing='ij')
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()])
    return grid_xy, (len(coords), len(coords)), coords, coords


def build_depth_bins(slab_edges, n_depth_bins):
    """
    Coarse permeant-depth bins for the 2-D spatial maps.

    Why the maps need a depth axis
    ------------------------------
    Every spatial map is stamped in a permeant-centered frame and was
    previously accumulated into a single (n_grid,) array, i.e. integrated
    over the permeant's whole z-trajectory and weighted by how much
    (path-weighted) time it spent at each depth. Two simulations with
    different depth-occupancy distributions — which is exactly what you
    get when the barrier sits at a different z, or when the path weights
    differ — then produce maps that differ even if the membrane responds
    identically at every depth. Differencing such maps confounds "the
    membrane deforms differently" with "the permeant was somewhere else".

    Binning by depth and storing the sufficient statistics (wsum, wcount)
    per bin removes the confound: summing the bins back up reproduces the
    old occupancy-weighted map exactly, while supplying a *common* set of
    per-bin weights to both simulations (see SpatialMaps.map) makes the
    comparison depth-matched by construction.

    The bins deliberately span the same range as the 1-D slab grid so a
    map bin can always be related back to a range of `slab_center`.
    """
    n       = max(int(n_depth_bins), 1)
    edges   = np.linspace(float(slab_edges[0]), float(slab_edges[-1]), n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def _stamp_frame(lipid_xy, lipid_values, grid_tree, stamp_radius,
                 wsum_accum, wcount_accum, path_weight,
                 box_xy=None):
    lipid_values = np.asarray(lipid_values, dtype=float)
    lipid_xy     = np.asarray(lipid_xy,     dtype=float)
    valid        = np.isfinite(lipid_values)

    for pos, val, ok in zip(lipid_xy, lipid_values, valid):
        if not ok:
            continue

        candidates = [pos]

        if box_xy is not None:
            Lx, Ly = float(box_xy[0]), float(box_xy[1])
            if pos[0] < stamp_radius:
                candidates.append(np.array([pos[0] + Lx, pos[1]]))
            if pos[0] > Lx - stamp_radius:
                candidates.append(np.array([pos[0] - Lx, pos[1]]))
            if pos[1] < stamp_radius:
                candidates.append(np.array([pos[0], pos[1] + Ly]))
            if pos[1] > Ly - stamp_radius:
                candidates.append(np.array([pos[0], pos[1] - Ly]))
            if pos[0] < stamp_radius and pos[1] < stamp_radius:
                candidates.append(np.array([pos[0] + Lx, pos[1] + Ly]))
            if pos[0] < stamp_radius and pos[1] > Ly - stamp_radius:
                candidates.append(np.array([pos[0] + Lx, pos[1] - Ly]))
            if pos[0] > Lx - stamp_radius and pos[1] < stamp_radius:
                candidates.append(np.array([pos[0] - Lx, pos[1] + Ly]))
            if pos[0] > Lx - stamp_radius and pos[1] > Ly - stamp_radius:
                candidates.append(np.array([pos[0] - Lx, pos[1] - Ly]))

        for cand in candidates:
            neighbours = grid_tree.query_ball_point(cand, stamp_radius)
            if neighbours:
                idx = np.array(neighbours, dtype=np.intp)
                wsum_accum[idx]   += path_weight * val
                wcount_accum[idx] += path_weight


def finalise_stamp_map(wsum_accum, wcount_accum, grid_shape):
    with np.errstate(invalid='ignore', divide='ignore'):
        F_flat = np.where(wcount_accum > 0,
                          wsum_accum / wcount_accum,
                          np.nan)
    return F_flat.reshape(grid_shape)


def _make_spatial_accumulators(M, chain_bonds, lipid_resnames=('DOPC', 'POPC'),
                               n_depth=1):
    """
    Allocate (wsum, wcount) pairs for every spatial observable × species × leaflet.

    Key pattern:
      '<obs>'                    — mixed / both leaflets
      '<obs>_<SPECIES>'          — e.g. 'p2_sn2_chain_DOPC'
      '<obs>_<LEAFLET>'          — e.g. 'p2_sn2_chain_upper'
      '<obs>_<SPECIES>_<LEAFLET>'— e.g. 'p2_sn2_chain_DOPC_upper'

    Each accumulator is (n_depth, M): a permeant-depth axis on top of the
    flattened lateral grid, so a map can be collapsed over depth (exactly
    reproducing the old depth-integrated map) or compared depth-matched
    against another simulation. See build_depth_bins.
    """
    species  = ['mixed'] + list(lipid_resnames)
    leaflets = ['both', 'upper', 'lower']
    acc = {}

    def _add(name):
        acc[name] = [np.zeros((n_depth, M)), np.zeros((n_depth, M))]

    for sp in species:
        sp_sfx = '' if sp == 'mixed' else f'_{sp}'
        for lf in leaflets:
            lf_sfx = '' if lf == 'both' else f'_{lf}'
            sfx = f'{sp_sfx}{lf_sfx}'
            for chain, bonds in chain_bonds.items():
                _add(f'p2_{chain}_chain{sfx}')
                for bi in range(len(bonds)):
                    _add(f'p2_{chain}_bond_{bi}{sfx}')
                _add(f'theta_{chain}{sfx}')
                _add(f'phi_{chain}{sfx}')
            _add(f'thickness{sfx}')
            _add(f'curvature_H{sfx}')
    return acc


def _save_spatial_accumulators(accumulators, x_coords, y_coords, depth_edges,
                               npz_path):
    arrays = {
        'x_coords'   : x_coords,
        'y_coords'   : y_coords,
        'depth_edges': np.asarray(depth_edges, dtype=float),
    }
    for name, (wsum, wcount) in accumulators.items():
        arrays[name]             = wsum      # (n_depth, M)
        arrays[name + '_wcount'] = wcount
    np.savez_compressed(str(npz_path), **arrays)


def _regrid_map(src_x, src_y, src_2d, dst_x, dst_y):
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (src_x, src_y), src_2d,
        method='linear', bounds_error=False, fill_value=np.nan,
    )
    xx, yy = np.meshgrid(dst_x, dst_y, indexing='ij')
    return interp(np.column_stack([xx.ravel(), yy.ravel()])
                  ).reshape(len(dst_x), len(dst_y))


def _depth_edges_of(data):
    """
    Depth-bin edges stored alongside a spatial .npz. Files written before
    the depth axis existed carry a single implicit bin spanning everything.
    """
    if 'depth_edges' in data.files:
        return np.asarray(data['depth_edges'], dtype=float)
    return np.array([-np.inf, np.inf])


def aggregate_spatial_maps(path_start, path_end, weights):
    ref_x     = None
    ref_y     = None
    ref_depth = None
    agg_acc   = None
    obs_keys  = None

    for pn in range(path_start, path_end + 1):
        if pn not in weights:
            continue
        npz_path = Path(ORDER_OUT) / f"{pn}_spatial.npz"
        if not npz_path.exists():
            print(f"  [spatial-agg] {npz_path.name} not found, skipping.")
            continue

        data     = np.load(str(npz_path))
        px       = data['x_coords']
        py       = data['y_coords']
        p_depth  = _depth_edges_of(data)

        if ref_x is None:
            ref_x     = px
            ref_y     = py
            ref_depth = p_depth
            obs_keys  = [k for k in data.files
                         if k not in ('x_coords', 'y_coords', 'depth_edges')
                         and not k.endswith('_wcount')]
            M_ref   = len(ref_x) * len(ref_y)
            D_ref   = len(ref_depth) - 1
            agg_acc = {k: [np.zeros((D_ref, M_ref)), np.zeros((D_ref, M_ref))]
                       for k in obs_keys}

        # Depth bins are derived deterministically from --slab-range and
        # --n-depth-bins, so a mismatch means the path was produced by a
        # different invocation. Pooling those would silently mix depths;
        # skip instead of interpolating along a coarse, physical axis.
        if (len(p_depth) != len(ref_depth)
                or not np.allclose(p_depth, ref_depth, atol=1e-6,
                                   equal_nan=True)):
            print(f"  [spatial-agg] {npz_path.name} has incompatible depth "
                  f"bins ({len(p_depth) - 1} vs {D_ref}), skipping.")
            continue

        same_grid = (
            len(px) == len(ref_x) and len(py) == len(ref_y)
            and np.allclose(px, ref_x, atol=0.01)
            and np.allclose(py, ref_y, atol=0.01)
        )

        for key in obs_keys:
            wk, wck = key, key + '_wcount'
            if wk not in data or wck not in data:
                continue

            src_wsum = np.atleast_2d(data[wk])      # (D, M)
            src_wcnt = np.atleast_2d(data[wck])

            if same_grid:
                agg_acc[key][0] += src_wsum.reshape(D_ref, M_ref)
                agg_acc[key][1] += src_wcnt.reshape(D_ref, M_ref)
            else:
                src_shape = (len(px), len(py))
                for d in range(D_ref):
                    w_d = src_wsum[d].reshape(src_shape)
                    c_d = src_wcnt[d].reshape(src_shape)

                    with np.errstate(invalid='ignore', divide='ignore'):
                        src_avg = np.where(c_d > 0, w_d / c_d, np.nan)

                    dst_avg = _regrid_map(px, py, src_avg, ref_x, ref_y)
                    dst_cnt = _regrid_map(px, py, c_d,     ref_x, ref_y)

                    valid = np.isfinite(dst_avg.ravel()) & (dst_cnt.ravel() > 0)
                    agg_acc[key][0][d][valid] += (dst_avg.ravel()[valid]
                                                  * dst_cnt.ravel()[valid])
                    agg_acc[key][1][d][valid] += dst_cnt.ravel()[valid]

    if agg_acc is None:
        raise RuntimeError("No spatial map .npz files found to aggregate.")

    return (agg_acc, ref_x, ref_y, (len(ref_x), len(ref_y)), ref_depth)


def save_spatial_maps(accumulators, grid_shape, x_coords, y_coords,
                      depth_edges, npz_path):
    """
    Persist the aggregated spatial maps as *sufficient statistics*
    (weighted sum and accumulated weight per depth bin) rather than as a
    single collapsed mean map.

    Storing wsum/wcount instead of the mean buys three things the mean
    alone cannot provide:
      * exact re-collapsing over any subset of depth bins,
      * depth-matched comparison against another simulation (feed both
        the same per-bin weights — see SpatialMaps.map),
      * a real coverage/weight field for masking under-sampled grid
        cells, which the previous mean-only file discarded.

    Use SpatialMaps / load_spatial_maps to read these back; they expose
    the collapsed mean map so existing plotting code is unaffected.
    """
    nx, ny = grid_shape
    arrays = {
        'x_coords'   : x_coords,
        'y_coords'   : y_coords,
        'depth_edges': np.asarray(depth_edges, dtype=float),
    }
    for name, (wsum, wcount) in accumulators.items():
        w2 = np.atleast_2d(wsum)
        c2 = np.atleast_2d(wcount)
        n_depth = w2.shape[0]
        arrays[f'{name}_wsum']   = w2.reshape(n_depth, nx, ny)
        arrays[f'{name}_wcount'] = c2.reshape(n_depth, nx, ny)
    np.savez_compressed(str(npz_path), **arrays)


class SpatialMaps:
    """
    Read-side accessor for an aggregated spatial-map .npz.

    Indexing (`maps[key]`) and membership (`key in maps`) behave like the
    plain dict of 2-D mean maps the old mean-only .npz provided, so the
    plotting helpers work unchanged. The depth axis is reached through
    `map()`, `coverage()` and `occupancy()`.

    Depth handling
    --------------
    depth : None | int | slice | (z_lo, z_hi)
        Which depth bins to include. None = all of them.
    depth_weights : None | 'uniform' | array
        None      — occupancy weighting: sum wsum / sum wcount over the
                    selected bins. This reproduces the old depth-integrated
                    map exactly, and carries the occupancy confound with it.
        'uniform' — average the per-bin means with equal weight, so every
                    depth the permeant visited counts the same regardless
                    of how long it lingered there.
        array     — explicit per-bin weights (length = n selected bins).

    Comparing two simulations
    -------------------------
    Pass the *same* `depth_weights` to both. 'uniform' is the simplest
    such choice; to instead match simulation B onto A's depth profile,
    read `w = A.occupancy(key); w /= w.sum()` and hand that array to both.
    Either way the residual map then reflects membrane response only, not
    a difference in where the permeant spent its time.
    """

    def __init__(self, npz_path, depth=None, depth_weights=None):
        self._data       = np.load(str(npz_path))
        self.path        = str(npz_path)
        self.x_coords    = self._data['x_coords']
        self.y_coords    = self._data['y_coords']
        self.depth_edges = _depth_edges_of(self._data)
        self.depth_centers = 0.5 * (self.depth_edges[:-1]
                                    + self.depth_edges[1:])
        self.keys        = sorted(k[:-5] for k in self._data.files
                                  if k.endswith('_wsum'))
        self.depth         = depth
        self.depth_weights = depth_weights

    # ── dict-like surface ────────────────────────────────────────────────
    def __contains__(self, key):
        return f'{key}_wsum' in self._data

    def __getitem__(self, key):
        return self.map(key)

    @property
    def n_depth(self):
        return max(len(self.depth_edges) - 1, 1)

    def bins_in_range(self, z_lo, z_hi):
        """Indices of depth bins whose centers fall in [z_lo, z_hi]."""
        c = self.depth_centers
        return np.where((c >= z_lo) & (c <= z_hi))[0]

    def _select(self, key, depth):
        wsum = np.atleast_3d(self._data[f'{key}_wsum'])
        wcnt = np.atleast_3d(self._data[f'{key}_wcount'])
        if depth is None:
            return wsum, wcnt
        if isinstance(depth, tuple):
            if len(depth) != 2:
                raise ValueError("depth tuple must be (z_lo, z_hi)")
            idx = self.bins_in_range(depth[0], depth[1])
        elif isinstance(depth, (int, np.integer)):
            idx = [int(depth)]
        else:
            idx = np.arange(self.n_depth)[depth]
        return wsum[idx], wcnt[idx]

    def map(self, key, depth=..., depth_weights=...):
        """2-D mean map for `key`, collapsed over the selected depth bins."""
        if depth is ...:
            depth = self.depth
        if depth_weights is ...:
            depth_weights = self.depth_weights

        wsum, wcnt = self._select(key, depth)

        with np.errstate(invalid='ignore', divide='ignore'):
            if depth_weights is None:
                num = wsum.sum(axis=0)
                den = wcnt.sum(axis=0)
                return np.where(den > 0, num / den, np.nan)

            per_bin = np.where(wcnt > 0, wsum / np.where(wcnt > 0, wcnt, 1.0),
                               np.nan)
            if isinstance(depth_weights, str):
                if depth_weights != 'uniform':
                    raise ValueError(
                        f"unknown depth_weights {depth_weights!r}; "
                        "expected None, 'uniform' or an array"
                    )
                w = np.ones(per_bin.shape[0])
            else:
                w = np.asarray(depth_weights, dtype=float)
                if w.shape[0] != per_bin.shape[0]:
                    raise ValueError(
                        f"depth_weights has length {w.shape[0]} but "
                        f"{per_bin.shape[0]} depth bins are selected"
                    )
            w3  = w[:, None, None]
            # Bins with no data must not dilute the average, so the
            # denominator only counts weights that actually contributed.
            den = (np.isfinite(per_bin) * w3).sum(axis=0)
            num = np.nansum(per_bin * w3, axis=0)
            return np.where(den > 0, num / den, np.nan)

    def coverage(self, key, depth=...):
        """Accumulated weight per grid cell — use it to mask thin cells."""
        if depth is ...:
            depth = self.depth
        _, wcnt = self._select(key, depth)
        return wcnt.sum(axis=0)

    def occupancy(self, key):
        """Total accumulated weight per depth bin, i.e. the depth profile."""
        wcnt = np.atleast_3d(self._data[f'{key}_wcount'])
        return wcnt.sum(axis=(1, 2))


def load_spatial_maps(npz_path, depth=None, depth_weights=None):
    maps = SpatialMaps(npz_path, depth=depth, depth_weights=depth_weights)
    return maps, maps.x_coords, maps.y_coords


# ═════════════════════════════════════════════════════════════════════════════
# Diffusion helpers (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def _short_time_msd(z_series, dt, max_lag_steps):
    N    = len(z_series)
    lags = np.arange(1, max_lag_steps + 1, dtype=float) * dt
    msd  = np.full(max_lag_steps, np.nan)
    for k in range(1, max_lag_steps + 1):
        diffs = z_series[k:] - z_series[:-k]
        if len(diffs) > 0:
            msd[k - 1] = np.mean(diffs ** 2)
    return lags, msd


def _position_autocorrelation_D(z_series, dt):
    if len(z_series) < 4:
        return np.nan
    dz      = z_series - z_series.mean()
    var     = np.mean(dz ** 2)
    if var < 1e-12:
        return np.nan
    N       = len(dz)
    fft_z   = np.fft.rfft(dz, n=2 * N)
    acf_raw = np.fft.irfft(fft_z * np.conj(fft_z))[:N]
    acf_raw /= (np.arange(N, 0, -1))
    C        = acf_raw / acf_raw[0]

    zero_cross = np.where(C <= 0)[0]
    cutoff     = zero_cross[0] if len(zero_cross) > 0 else N
    integral   = trapezoid(C[:cutoff], dx=dt)

    if integral < 1e-12:
        return np.nan
    return float(var / integral)


def _estimate_sojourn_D(z_run, _dt, max_lag_steps):
    D_ein = np.nan
    lag_steps = min(max_lag_steps, len(z_run) - 1)
    if lag_steps >= 1:
        lags, msd = _short_time_msd(z_run, _dt, lag_steps)
        best_D  = np.nan
        best_r2 = -np.inf
        for k in range(1, lag_steps + 1):
            sub_lag = lags[:k]
            sub_msd = msd[:k]
            if np.any(np.isnan(sub_msd)):
                continue
            slope = np.dot(sub_lag, sub_msd) / np.dot(sub_lag, sub_lag)
            resid = sub_msd - slope * sub_lag
            ss_res = np.dot(resid, resid)
            ss_tot = np.dot(sub_msd - sub_msd.mean(),
                            sub_msd - sub_msd.mean()) + 1e-30
            r2 = 1.0 - ss_res / ss_tot
            if r2 > best_r2:
                best_r2 = r2
                best_D  = slope / 2.0
        if np.isfinite(best_D) and best_D > 0:
            D_ein = best_D

    D_hum = _position_autocorrelation_D(z_run, _dt)
    if not (np.isfinite(D_hum) and D_hum > 0):
        D_hum = np.nan

    return D_ein, D_hum


def _unwrap_xy_pbc(raw_xy, prev_xy, jump_xy, box_xy, cutoff=0.5):
    for dim in range(2):
        delta = raw_xy[dim] - prev_xy[dim]
        L     = box_xy[dim]
        if   delta >  cutoff * L:
            jump_xy[dim] -= L
        elif delta < -cutoff * L:
            jump_xy[dim] += L
        prev_xy[dim] = raw_xy[dim]
    return raw_xy + jump_xy


def _msd_xy_from_series(xy_series, _dt, max_lag_steps):
    xy  = np.asarray(xy_series, dtype=float)
    N   = len(xy)
    if N < 4:
        return np.nan, np.nan, np.nan

    lag_steps = min(max_lag_steps, N - 1)
    lags      = np.arange(1, lag_steps + 1, dtype=float) * _dt
    msd_xy    = np.full(lag_steps, np.nan)
    for k in range(1, lag_steps + 1):
        dr  = xy[k:] - xy[:-k]
        msd_xy[k - 1] = np.mean(np.sum(dr ** 2, axis=1))

    best_D  = np.nan
    best_r2 = -np.inf
    for k in range(1, lag_steps + 1):
        sub_lag = lags[:k]
        sub_msd = msd_xy[:k]
        if np.any(np.isnan(sub_msd)):
            continue
        slope  = np.dot(sub_lag, sub_msd) / np.dot(sub_lag, sub_lag)
        resid  = sub_msd - slope * sub_lag
        ss_res = np.dot(resid, resid)
        ss_tot = np.dot(sub_msd - sub_msd.mean(),
                        sub_msd - sub_msd.mean()) + 1e-30
        r2 = 1.0 - ss_res / ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best_D  = slope / 4.0
    D_ein_xy = best_D if (np.isfinite(best_D) and best_D > 0) else np.nan

    D_hum_x = _position_autocorrelation_D(xy[:, 0], _dt)
    D_hum_y = _position_autocorrelation_D(xy[:, 1], _dt)
    if not (np.isfinite(D_hum_x) and D_hum_x > 0):
        D_hum_x = np.nan
    if not (np.isfinite(D_hum_y) and D_hum_y > 0):
        D_hum_y = np.nan

    return D_ein_xy, D_hum_x, D_hum_y


# ═════════════════════════════════════════════════════════════════════════════
# Slab grid construction
# ═════════════════════════════════════════════════════════════════════════════

def build_slab_grid(topology, xtc_files, permeant_resname,
                    bilayer_normal, slab_width, slab_range=None):
    axis = {'x': 0, 'y': 1, 'z': 2}[bilayer_normal.lower()]
    if slab_range is not None:
        z_min, z_max = float(slab_range[0]), float(slab_range[1])
    else:
        all_z = []
        for xtc in xtc_files:
            u = mda.Universe(topology, xtc, refresh_offsets=True)
            perm = u.select_atoms(f"resname {permeant_resname} and not name H*")
            if len(perm) == 0:
                perm = u.select_atoms(f"resname {permeant_resname}")
            for _ts in u.trajectory:
                all_z.append(perm.positions[:, axis].copy())
        all_z = np.concatenate(all_z)
        z_min = all_z.min() - slab_width
        z_max = all_z.max() + slab_width
    edges   = np.arange(z_min, z_max + slab_width, slab_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


# ═════════════════════════════════════════════════════════════════════════════
# Path helpers (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def load_path_weights(weights_file='path_weights.txt'):
    weights = {}
    with open(weights_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            weights[int(parts[0])] = float(parts[1])
    return weights


def read_toml(toml_file):
    if os.path.isfile(toml_file):
        with open(toml_file, mode="rb") as fh:
            config = tomli.load(fh)
    else:
        raise FileNotFoundError(f"TOML not found: {toml_file}")
    interfaces       = config["simulation"]["interfaces"]
    lambda_A         = interfaces[0]
    lambda_B         = interfaces[-1]
    lambda_minus_one = (config["simulation"].get("tis_set") or {}) \
                       .get("lambda_minus_one")
    return lambda_A, lambda_B, lambda_minus_one


def get_reactive_paths(path_number, infretis_data_file, lambda_B):
    first_cols, third_cols = [], []
    with open(infretis_data_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            cols = line.split()
            first_cols.append(int(cols[0]))
            third_cols.append(float(cols[2]))
    first_cols = np.array(first_cols)
    third_cols = np.array(third_cols)
    idx = np.where(first_cols == path_number)[0]
    return bool(third_cols[idx[0]] > lambda_B) if idx.size > 0 else None


def extract_sorted_traj_names(trj_path):
    filenames, directions, seen = [], [], set()
    g96_index = None
    with open(trj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            cols      = line.split()
            step      = cols[0]
            filename  = cols[1]
            direction = cols[3]
            if filename not in seen:
                if filename.endswith('.trr'):
                    filenames.append(filename[:-4] + '.xtc')
                elif filename.endswith('.g96'):
                    g96_index = step
                else:
                    filenames.append(filename)
                seen.add(filename)
                directions.append(direction)
    return filenames, directions, g96_index


def remove_first_last_frames(ensemble, lambda_minus_one, lambda_A, first, last):
    if (ensemble == "plus" and first < lambda_A) or \
       (ensemble == "minus" and (first < lambda_minus_one or first > lambda_A)):
        first_frame = True
    else:
        first_frame = False
    if (ensemble == "plus" and last < lambda_A) or \
       (ensemble == "minus" and (last < lambda_minus_one or last > lambda_A)):
        last_frame = True
    else:
        last_frame = False
    return first_frame, last_frame


# ═════════════════════════════════════════════════════════════════════════════
# CSV path helpers and observable-group definitions
# ═════════════════════════════════════════════════════════════════════════════

OBSERVABLE_GROUPS = {
    'order'        : ('p2', 'tilt'),
    'structural'   : ('thick', 'deform', 'water_z'),
    'diffusion'    : ('diff',),
    'diffusion_xy' : ('diff_xy',),
    'spatial'      : ('spatial',),
}

_CSV_KEY_TO_GROUP = {
    csv_key: group
    for group, keys in OBSERVABLE_GROUPS.items()
    for csv_key in keys
}


def _all_csv_paths(path_number):
    base = Path(ORDER_OUT)
    return {
        'p2'     : base / f"{path_number}_p2.csv",
        'tilt'   : base / f"{path_number}_tilt.csv",
        'thick'  : base / f"{path_number}_thick.csv",
        'deform' : base / f"{path_number}_deform.csv",
        'water_z': base / f"{path_number}_water_z.csv",
        'diff'   : base / f"{path_number}_diffusion.csv",
        'diff_xy': base / f"{path_number}_diffusion_xy.csv",
        'spatial': base / f"{path_number}_spatial.npz",
    }


def _has_near_columns(csvs, group, near_n):
    """
    Peek at an existing CSV's header for the near-permeant columns expected
    for every requested N. Guards against silently skipping recomputation
    when the near-N feature is adopted on top of pre-existing per-path CSVs
    written before that schema existed.
    """
    if group == 'order':
        probe_path  = csvs['p2']
        probe_chain = next(iter(CHAIN_BONDS))
        probe_cols  = [f'near{N}_{probe_chain}_p2_chain_wsum' for N in near_n]
    else:  # 'structural'
        probe_path = csvs['thick']
        probe_cols = [f'near{N}_thickness_wsum' for N in near_n]
    try:
        header = pd.read_csv(probe_path, comment='#', nrows=0).columns
    except Exception:
        return False
    return all(c in header for c in probe_cols)


def _needed_groups(csvs, overwrite, near_n=()):
    if overwrite:
        return set(OBSERVABLE_GROUPS.keys())
    needed = set()
    for group, keys in OBSERVABLE_GROUPS.items():
        if not all(csvs[k].exists() for k in keys):
            needed.add(group)
            continue
        if near_n and group in ('order', 'structural') and \
           not _has_near_columns(csvs, group, near_n):
            needed.add(group)
    return needed


# ═════════════════════════════════════════════════════════════════════════════
# Selective single-pass accumulator  — now with leaflet dimension
# ═════════════════════════════════════════════════════════════════════════════

def calculate_selected_slab_profiles(
    topology,
    xtc_files,
    xtc_directions,
    permeant_resname,
    needed_groups,
    weight              = 1.0,
    first_frame         = False,
    last_frame          = False,
    lipid_resnames      = ('DOPC', 'POPC'),
    slab_width          = 1.0,
    bilayer_normal      = 'z',
    slab_range          = None,
    chain_bonds         = CHAIN_BONDS,
    tilt_endpoints      = TILT_ENDPOINTS,
    phosphorus_name     = PHOSPHORUS_NAME,
    water_resname       = WATER_RESNAME,
    water_oxygen_name   = WATER_OXYGEN_NAME,
    r_max               = 40.0,
    n_radial_bins       = 20,
    n_z_bins            = 50,
    water_z_range       = WATER_Z_RANGE,
    curvature_smoothing = 0.0,
    dt                  = None,
    max_lag_ps          = 500.0,
    min_slab_points     = 5,
    grid_spacing        = 5.0,
    stamp_radius        = None,
    near_n              = (),
    spatial_extent      = None,
    n_depth_bins        = 10,
):
    """
    Single-pass accumulator.

    Accumulator dimensions
    ----------------------
    All order/structural accumulators now carry a *leaflet* dimension in
    addition to the existing *species* dimension:

      leaflet ∈ {'both', 'upper', 'lower'}
      species ∈ {'mixed', 'DOPC', 'POPC'}

    CSV column naming convention
    ----------------------------
    <leaflet_prefix><species_prefix><chain>_<quantity>_wsum
    where leaflet_prefix = '' for 'both', 'upper_' for 'upper', etc.
    and   species_prefix = '' for 'mixed', 'DOPC_' for 'DOPC', etc.

    e.g.  'upper_DOPC_sn2_p2_chain_wsum'
          'lower_sn1_theta_wsum'
          'sn2_p2_bond_0_wsum'   (both leaflets, mixed species — backward compat)
    """
    run_order         = 'order'        in needed_groups
    run_structural    = 'structural'   in needed_groups
    run_diffusion     = 'diffusion'    in needed_groups
    run_diffusion_xy  = 'diffusion_xy' in needed_groups
    run_spatial       = 'spatial'      in needed_groups

    axis      = {'x': 0, 'y': 1, 'z': 2}[bilayer_normal.lower()]
    normal    = MEMBRANE_NORMAL
    _lat_axes = [a for a in range(3) if a != axis]

    edges, centers = build_slab_grid(
        topology, xtc_files, permeant_resname,
        bilayer_normal, slab_width, slab_range,
    )
    n_slabs = len(centers)

    r_edges   = np.linspace(0.0, r_max, n_radial_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    # ── Species and leaflet label sets ────────────────────────────────────────
    _ORDER_SPECIES  = ['mixed'] + list(lipid_resnames)
    _STRUCT_SPECIES = ['mixed'] + list(lipid_resnames)
    _LEAFLETS       = ['both', 'upper', 'lower']

    # ── ORDER accumulators ────────────────────────────────────────────────────
    if run_order:
        p2_wsum  = {}
        p2_wtot  = {}
        for sp in _ORDER_SPECIES:
            for lf in _LEAFLETS:
                for chain, bonds in chain_bonds.items():
                    for bi in range(len(bonds)):
                        p2_wsum[(sp, lf, chain, bi)] = np.zeros(n_slabs)
                        p2_wtot[(sp, lf, chain, bi)] = np.zeros(n_slabs)
                    p2_wsum[(sp, lf, chain, 'chain')] = np.zeros(n_slabs)
                    p2_wtot[(sp, lf, chain, 'chain')] = np.zeros(n_slabs)

        theta_wsum = {(sp, lf, ch): np.zeros(n_slabs)
                      for sp in _ORDER_SPECIES
                      for lf in _LEAFLETS
                      for ch in tilt_endpoints}
        theta_wtot = {(sp, lf, ch): np.zeros(n_slabs)
                      for sp in _ORDER_SPECIES
                      for lf in _LEAFLETS
                      for ch in tilt_endpoints}
        phi_wsum   = {(sp, lf, ch): np.zeros(n_slabs)
                      for sp in _ORDER_SPECIES
                      for lf in _LEAFLETS
                      for ch in tilt_endpoints}
        phi_wtot   = {(sp, lf, ch): np.zeros(n_slabs)
                      for sp in _ORDER_SPECIES
                      for lf in _LEAFLETS
                      for ch in tilt_endpoints}

        # Near-permeant accumulators: species-split but NOT leaflet-split —
        # a sample of N=5-10 lipids is too thin to split further, and the
        # near-set is already whichever leaflet(s) are actually adjacent to
        # the permeant. Compared against the existing ('mixed','both') bulk
        # average to see whether the permeant locally perturbs its neighbours.
        near_p2_wsum = {}
        near_p2_wtot = {}
        for N in near_n:
            for sp in _ORDER_SPECIES:
                for chain, bonds in chain_bonds.items():
                    for bi in range(len(bonds)):
                        near_p2_wsum[(N, sp, chain, bi)] = np.zeros(n_slabs)
                        near_p2_wtot[(N, sp, chain, bi)] = np.zeros(n_slabs)
                    near_p2_wsum[(N, sp, chain, 'chain')] = np.zeros(n_slabs)
                    near_p2_wtot[(N, sp, chain, 'chain')] = np.zeros(n_slabs)

        near_theta_wsum = {(N, sp, ch): np.zeros(n_slabs)
                           for N in near_n
                           for sp in _ORDER_SPECIES
                           for ch in tilt_endpoints}
        near_theta_wtot = {(N, sp, ch): np.zeros(n_slabs)
                           for N in near_n
                           for sp in _ORDER_SPECIES
                           for ch in tilt_endpoints}
        near_phi_wsum   = {(N, sp, ch): np.zeros(n_slabs)
                           for N in near_n
                           for sp in _ORDER_SPECIES
                           for ch in tilt_endpoints}
        near_phi_wtot   = {(N, sp, ch): np.zeros(n_slabs)
                           for N in near_n
                           for sp in _ORDER_SPECIES
                           for ch in tilt_endpoints}

    # ── STRUCTURAL accumulators ───────────────────────────────────────────────
    if run_structural:
        # Thickness: per species × per leaflet (a genuine 3-way split — each
        # leaflet's own P atoms give a distinct "local thickness" reading).
        thick_wsum   = {(sp, lf): np.zeros(n_slabs)
                        for sp in _STRUCT_SPECIES for lf in _LEAFLETS}
        thick_wtot   = {(sp, lf): np.zeros(n_slabs)
                        for sp in _STRUCT_SPECIES for lf in _LEAFLETS}

        # Curvature: per species only. H_upper/H_lower are already the
        # leaflet split (the upper- and lower-surface RBF fits) — there is
        # no separate "species filtered to leaflet X" curvature value, so
        # this must NOT also carry an `lf` dimension (that previously
        # produced NaN or duplicated columns and nonsensical plot legends
        # like "lower leaflet" paired with an "upper" surface value).
        H_upper_wsum = {sp: np.zeros(n_slabs) for sp in _STRUCT_SPECIES}
        H_upper_wtot = {sp: np.zeros(n_slabs) for sp in _STRUCT_SPECIES}
        K_upper_wsum = {sp: np.zeros(n_slabs) for sp in _STRUCT_SPECIES}
        K_upper_wtot = {sp: np.zeros(n_slabs) for sp in _STRUCT_SPECIES}
        H_lower_wsum = {sp: np.zeros(n_slabs) for sp in _STRUCT_SPECIES}
        H_lower_wtot = {sp: np.zeros(n_slabs) for sp in _STRUCT_SPECIES}

        # Near-permeant thickness/curvature: species-split, not leaflet-split
        # (combines whichever leaflet each near lipid actually belongs to
        # into one "local membrane near the permeant" value).
        near_thick_wsum = {(N, sp): np.zeros(n_slabs)
                           for N in near_n for sp in _STRUCT_SPECIES}
        near_thick_wtot = {(N, sp): np.zeros(n_slabs)
                           for N in near_n for sp in _STRUCT_SPECIES}
        near_H_wsum     = {(N, sp): np.zeros(n_slabs)
                           for N in near_n for sp in _STRUCT_SPECIES}
        near_H_wtot     = {(N, sp): np.zeros(n_slabs)
                           for N in near_n for sp in _STRUCT_SPECIES}

        # Deformation, water: mixed only (whole-leaflet quantities)
        def_upper_wsum    = np.zeros(n_radial_bins)
        def_upper_wtot    = np.zeros(n_radial_bins)
        def_lower_wsum    = np.zeros(n_radial_bins)
        def_lower_wtot    = np.zeros(n_radial_bins)
        water_radial_wsum = np.zeros(n_radial_bins)
        water_radial_wtot = np.zeros(n_radial_bins)
        n_defect_wsum     = np.zeros(n_slabs)
        n_defect_wtot     = np.zeros(n_slabs)
        H_radial_wsum     = np.zeros(n_radial_bins)
        H_radial_wtot     = np.zeros(n_radial_bins)

        # Fixed, midplane-relative water grid. Previously this opened a
        # second Universe purely to read the first frame's box height and
        # built the grid from it, which made the axis differ between paths
        # (and disagree with the per-frame grid water_defect_profile was
        # actually binning into). The grid is now a pure function of the
        # CLI arguments, so every path and every simulation shares it.
        z_edges   = np.linspace(float(water_z_range[0]),
                                float(water_z_range[1]), n_z_bins + 1)
        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
        water_z_wsum = np.zeros(n_z_bins)
        water_z_wtot = np.zeros(n_z_bins)
        _water_z_range_checked = False

    # ── Diffusion accumulators (unchanged) ────────────────────────────────────
    if run_diffusion:
        D_ein_wsum  = np.zeros(n_slabs)
        D_ein_wtot  = np.zeros(n_slabs)
        D_hum_wsum  = np.zeros(n_slabs)
        D_hum_wtot  = np.zeros(n_slabs)
        nsoj_wsum   = np.zeros(n_slabs)
        nsoj_wtot   = np.zeros(n_slabs)
        z_full        = []
        slab_idx_full = []

    if run_diffusion_xy:
        D_xy_ein_wsum  = np.zeros(n_slabs)
        D_xy_ein_wtot  = np.zeros(n_slabs)
        D_xy_hx_wsum   = np.zeros(n_slabs)
        D_xy_hx_wtot   = np.zeros(n_slabs)
        D_xy_hy_wsum   = np.zeros(n_slabs)
        D_xy_hy_wtot   = np.zeros(n_slabs)
        nsoj_xy_wsum   = np.zeros(n_slabs)
        nsoj_xy_wtot   = np.zeros(n_slabs)
        xy_full           = []
        slab_idx_xy_full  = []

    if run_spatial:
        # The map is centered on the permeant's lateral position every
        # frame (see the SPATIAL block in the frame loop below), so its
        # extent no longer depends on box size and the grid can be built
        # once up front rather than lazily on the first frame.
        _stamp_r_initialised = stamp_radius is not None
        _stamp_r    = stamp_radius
        _spatial_half_extent = spatial_extent if spatial_extent is not None else r_max
        _grid_xy, _grid_shape, _x_coords, _y_coords = \
            build_lateral_grid_centered(_spatial_half_extent, grid_spacing)
        _grid_tree   = cKDTree(_grid_xy)
        # Depth axis: maps are accumulated per permeant-depth bin instead
        # of being integrated over the whole trajectory, so that a map can
        # later be collapsed over any depth range or compared depth-matched
        # against another simulation (see build_depth_bins / SpatialMaps).
        _depth_edges, _depth_centers = build_depth_bins(edges, n_depth_bins)
        _n_depth     = len(_depth_centers)
        _spatial_acc = _make_spatial_accumulators(
            len(_grid_xy), chain_bonds, lipid_resnames, n_depth=_n_depth
        )

    lipid_sel  = " or ".join(f"resname {r}" for r in lipid_resnames)
    _dt        = dt
    _max_lag_steps = None

    # ═════════════════════════════════════════════════════════════════════════
    # Frame loop
    # ═════════════════════════════════════════════════════════════════════════
    for i, (xtc, direction) in enumerate(zip(xtc_files, xtc_directions)):
        u          = mda.Universe(topology, xtc, refresh_offsets=True)
        permeant   = u.select_atoms(f"resname {permeant_resname}")
        perm_heavy = permeant.select_atoms("not name H*")
        if len(perm_heavy) == 0:
            perm_heavy = permeant
        membrane   = u.select_atoms("resname DOPC POPC")

        if _dt is None:
            u.trajectory[0]
            _dt = float(u.trajectory.dt)
        if (run_diffusion or run_diffusion_xy) and _max_lag_steps is None:
            _max_lag_steps = max(1, int(max_lag_ps / _dt))

        if run_diffusion_xy:
            _pbc_jump   = np.zeros(2)
            _pbc_prev   = None
            _pbc_cutoff = 0.5

        # ── Pre-build atom-index arrays (once per XTC) ────────────────────────
        if run_order or run_spatial:
            all_lipids = u.select_atoms(lipid_sel)

            bond_idx     = {}
            bond_residx  = {}
            species_mask = {}

            for chain, bonds in chain_bonds.items():
                for bi, (a1_name, a2_name) in enumerate(bonds):
                    idx_a1, idx_a2, resnames = [], [], []
                    for res in all_lipids.residues:
                        s1 = res.atoms.select_atoms(f"name {a1_name}")
                        s2 = res.atoms.select_atoms(f"name {a2_name}")
                        if len(s1) == 1 and len(s2) == 1:
                            idx_a1.append(s1.indices[0])
                            idx_a2.append(s2.indices[0])
                            resnames.append(res.resname)
                    if idx_a1:
                        rn_arr  = np.array(resnames)
                        ia1_arr = np.array(idx_a1, dtype=np.intp)
                        bond_idx[(chain, bi)] = (
                            ia1_arr,
                            np.array(idx_a2, dtype=np.intp),
                        )
                        bond_residx[(chain, bi)] = u.atoms.resindices[ia1_arr]
                        species_mask[(chain, bi)] = {
                            sp: (rn_arr == sp) for sp in lipid_resnames
                        }

            tilt_idx     = {}
            tilt_residx  = {}
            tilt_sp_mask = {}
            for chain, (base_name, tip_name) in tilt_endpoints.items():
                idx_base, idx_tip, resnames = [], [], []
                for res in all_lipids.residues:
                    sb = res.atoms.select_atoms(f"name {base_name}")
                    st = res.atoms.select_atoms(f"name {tip_name}")
                    if len(sb) == 1 and len(st) == 1:
                        idx_base.append(sb.indices[0])
                        idx_tip.append(st.indices[0])
                        resnames.append(res.resname)
                if idx_base:
                    rn_arr   = np.array(resnames)
                    ibas_arr = np.array(idx_base, dtype=np.intp)
                    tilt_idx[chain] = (
                        ibas_arr,
                        np.array(idx_tip,  dtype=np.intp),
                    )
                    tilt_residx[chain] = u.atoms.resindices[ibas_arr]
                    tilt_sp_mask[chain] = {
                        sp: (rn_arr == sp) for sp in lipid_resnames
                    }

        n_frames      = len(u.trajectory)
        frame_indices = list(range(n_frames))
        if direction == '-1':
            frame_indices = list(reversed(frame_indices))
        if first_frame and i == 0:
            frame_indices = frame_indices[1:]
        if last_frame and i == len(xtc_files) - 1:
            frame_indices = frame_indices[:-1]
        if i != 0 and i != len(xtc_files) - 1:
            frame_indices = frame_indices[1:]

        for fi in frame_indices:
            u.trajectory[fi]

            membrane_z = membrane.positions[:, axis].mean()
            perm_z     = membrane_z - perm_heavy.positions[:, axis].mean()
            slab_idx   = int(np.clip(
                np.searchsorted(edges, perm_z, side='right') - 1,
                0, n_slabs - 1
            ))

            # ── Near-permeant proximity ranking ─────────────────────────────────
            # Resindices of the N lipids (by P-atom, 3-D PBC-aware distance)
            # closest to the permeant this frame, for each requested N.
            near_sets = {}
            if near_n and (run_order or run_structural) and len(permeant) > 0:
                p_all_near = u.select_atoms(
                    f"({lipid_sel}) and name {phosphorus_name}"
                )
                if len(p_all_near) > 0:
                    perm_com_near = permeant.center_of_geometry()
                    d_near = distances.distance_array(
                        perm_com_near[np.newaxis, :], p_all_near.positions,
                        box=u.dimensions,
                    )[0]
                    order_near = np.argsort(d_near)
                    resix_near = p_all_near.resindices
                    for N in near_n:
                        k = min(N, len(order_near))
                        near_sets[N] = resix_near[order_near[:k]]

            # ── ORDER group ───────────────────────────────────────────────────
            if run_order:
                all_pos = u.atoms.positions

                for chain, bonds in chain_bonds.items():
                    bond_p2_per_lip_all = {}

                    for bi in range(len(bonds)):
                        key = (chain, bi)
                        if key not in bond_idx:
                            continue
                        ia1, ia2 = bond_idx[key]
                        vecs  = all_pos[ia2] - all_pos[ia1]
                        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                        ok    = norms[:, 0] > 1e-6
                        p2_per_lip = np.full(len(ia1), np.nan)
                        if ok.any():
                            v_hat = vecs[ok] / norms[ok]
                            cos2  = (v_hat @ normal) ** 2
                            p2_per_lip[ok] = 0.5 * (3.0 * cos2 - 1.0)
                        bond_p2_per_lip_all[bi] = p2_per_lip

                        # Leaflet assignment from base-atom (ia1) z-position
                        base_z  = all_pos[ia1, axis]
                        mid_z   = base_z.mean()
                        lf_masks = {
                            'both' : np.ones(len(ia1), dtype=bool),
                            'upper': base_z > mid_z,
                            'lower': base_z <= mid_z,
                        }

                        for sp in _ORDER_SPECIES:
                            sp_mask = (np.ones(len(ia1), dtype=bool)
                                       if sp == 'mixed'
                                       else species_mask.get(key, {}).get(
                                           sp, np.zeros(len(ia1), dtype=bool)))
                            for lf in _LEAFLETS:
                                combined = sp_mask & lf_masks[lf]
                                vals = p2_per_lip[combined]
                                if len(vals) and np.any(np.isfinite(vals)):
                                    p2_val = float(np.nanmean(vals))
                                    if np.isfinite(p2_val):
                                        p2_wsum[(sp, lf, chain, bi)][slab_idx] += weight * p2_val
                                        p2_wtot[(sp, lf, chain, bi)][slab_idx] += weight

                            if near_sets and key in bond_residx:
                                near_sp_vals = p2_per_lip[sp_mask]
                                residx_bond  = bond_residx[key][sp_mask]
                                for N in near_n:
                                    if N not in near_sets:
                                        continue
                                    near_mask = np.isin(residx_bond, near_sets[N])
                                    vals_n = near_sp_vals[near_mask]
                                    if len(vals_n) and np.any(np.isfinite(vals_n)):
                                        p2_val_n = float(np.nanmean(vals_n))
                                        if np.isfinite(p2_val_n):
                                            near_p2_wsum[(N, sp, chain, bi)][slab_idx] += weight * p2_val_n
                                            near_p2_wtot[(N, sp, chain, bi)][slab_idx] += weight

                    # Chain averages per species × leaflet
                    if bond_p2_per_lip_all:
                        mat = np.column_stack(list(bond_p2_per_lip_all.values()))
                        # Leaflet masks from the base atom of bond 0
                        first_bi = next(iter(bond_p2_per_lip_all))
                        ia1_0    = bond_idx[(chain, first_bi)][0]
                        base_z0  = all_pos[ia1_0, axis]
                        mid_z0   = base_z0.mean()
                        lf_masks_ch = {
                            'both' : np.ones(mat.shape[0], dtype=bool),
                            'upper': base_z0 > mid_z0,
                            'lower': base_z0 <= mid_z0,
                        }
                        for sp in _ORDER_SPECIES:
                            sp_mask = (np.ones(mat.shape[0], dtype=bool)
                                       if sp == 'mixed'
                                       else species_mask.get(
                                           (chain, first_bi), {}
                                       ).get(sp, np.zeros(mat.shape[0],
                                                          dtype=bool)))
                            for lf in _LEAFLETS:
                                combined = sp_mask & lf_masks_ch[lf]
                                vals = mat[combined]
                                if len(vals):
                                    chain_p2 = float(np.nanmean(vals))
                                    if np.isfinite(chain_p2):
                                        p2_wsum[(sp, lf, chain, 'chain')][slab_idx] += weight * chain_p2
                                        p2_wtot[(sp, lf, chain, 'chain')][slab_idx] += weight

                            if near_sets and (chain, first_bi) in bond_residx:
                                residx_ch  = bond_residx[(chain, first_bi)][sp_mask]
                                mat_sp     = mat[sp_mask]
                                for N in near_n:
                                    if N not in near_sets:
                                        continue
                                    near_mask = np.isin(residx_ch, near_sets[N])
                                    vals_n = mat_sp[near_mask]
                                    if len(vals_n):
                                        chain_p2_n = float(np.nanmean(vals_n))
                                        if np.isfinite(chain_p2_n):
                                            near_p2_wsum[(N, sp, chain, 'chain')][slab_idx] += weight * chain_p2_n
                                            near_p2_wtot[(N, sp, chain, 'chain')][slab_idx] += weight

                # Tilt angles per species × leaflet
                for chain in tilt_endpoints:
                    if chain not in tilt_idx:
                        continue
                    ib, it   = tilt_idx[chain]
                    base_z   = all_pos[ib, axis]
                    mid_z    = base_z.mean()
                    lf_masks_tilt = {
                        'both' : np.ones(len(ib), dtype=bool),
                        'upper': base_z > mid_z,
                        'lower': base_z <= mid_z,
                    }
                    for sp in _ORDER_SPECIES:
                        sp_mask = (np.ones(len(ib), dtype=bool)
                                   if sp == 'mixed'
                                   else tilt_sp_mask.get(chain, {}).get(
                                       sp, np.zeros(len(ib), dtype=bool)))
                        for lf in _LEAFLETS:
                            combined = sp_mask & lf_masks_tilt[lf]
                            if not combined.any():
                                continue
                            # Pass the ground-truth (full-population) leaflet
                            # split for exactly this subset — mean_tilt_angle
                            # must not re-derive it from `combined`'s own
                            # mean z, which mis-splits filtered subsets.
                            theta, phi, _ = mean_tilt_angle(
                                all_pos[it[combined]],
                                all_pos[ib[combined]],
                                normal,
                                leaflet=lf,
                                upper_mask=lf_masks_tilt['upper'][combined],
                            )
                            if np.isfinite(theta):
                                theta_wsum[(sp, lf, chain)][slab_idx] += weight * theta
                                theta_wtot[(sp, lf, chain)][slab_idx] += weight
                            if np.isfinite(phi):
                                phi_wsum[(sp, lf, chain)][slab_idx]   += weight * phi
                                phi_wtot[(sp, lf, chain)][slab_idx]   += weight

                        if near_sets and chain in tilt_residx:
                            residx_tilt = tilt_residx[chain]
                            for N in near_n:
                                if N not in near_sets:
                                    continue
                                near_combined = sp_mask & np.isin(residx_tilt, near_sets[N])
                                if not near_combined.any():
                                    continue
                                # Near-N sets are small and often single-
                                # leaflet; splitting by their own mean z
                                # (the default) mis-assigns members to the
                                # wrong pseudo-leaflet. Use the real,
                                # full-population split instead.
                                theta_n, phi_n, _ = mean_tilt_angle(
                                    all_pos[it[near_combined]],
                                    all_pos[ib[near_combined]],
                                    normal,
                                    leaflet='both',
                                    upper_mask=lf_masks_tilt['upper'][near_combined],
                                )
                                if np.isfinite(theta_n):
                                    near_theta_wsum[(N, sp, chain)][slab_idx] += weight * theta_n
                                    near_theta_wtot[(N, sp, chain)][slab_idx] += weight
                                if np.isfinite(phi_n):
                                    near_phi_wsum[(N, sp, chain)][slab_idx]   += weight * phi_n
                                    near_phi_wtot[(N, sp, chain)][slab_idx]   += weight

            # ── STRUCTURAL group ──────────────────────────────────────────────
            if run_structural:
                for sp in _STRUCT_SPECIES:
                    rn_tup = tuple(lipid_resnames) if sp == 'mixed' else (sp,)
                    td = _local_thickness_dict(u, rn_tup, phosphorus_name, axis)

                    # 'both' = combined mean (backward-compatible)
                    if np.isfinite(td['mean_thick_both']):
                        thick_wsum[(sp, 'both')][slab_idx] += weight * td['mean_thick_both']
                        thick_wtot[(sp, 'both')][slab_idx] += weight
                    # 'upper'
                    if np.isfinite(td['mean_thick_upper']):
                        thick_wsum[(sp, 'upper')][slab_idx] += weight * td['mean_thick_upper']
                        thick_wtot[(sp, 'upper')][slab_idx] += weight
                    # 'lower'
                    if np.isfinite(td['mean_thick_lower']):
                        thick_wsum[(sp, 'lower')][slab_idx] += weight * td['mean_thick_lower']
                        thick_wtot[(sp, 'lower')][slab_idx] += weight

                    # Near-permeant thickness: pool P atoms from whichever
                    # leaflet(s) the near lipids actually belong to.
                    if near_sets:
                        residx_th = np.concatenate(
                            [td['upper_resindices'], td['lower_resindices']]
                        )
                        thick_th = np.concatenate(
                            [td['thickness_upper'], td['thickness_lower']]
                        )
                        for N in near_n:
                            if N not in near_sets:
                                continue
                            near_mask = np.isin(residx_th, near_sets[N])
                            vals_n = thick_th[near_mask]
                            if len(vals_n) and np.any(np.isfinite(vals_n)):
                                thick_val_n = float(np.nanmean(vals_n))
                                if np.isfinite(thick_val_n):
                                    near_thick_wsum[(N, sp)][slab_idx] += weight * thick_val_n
                                    near_thick_wtot[(N, sp)][slab_idx] += weight

                # Deformation and water: mixed / whole-bilayer only
                _, def_u, def_l = radial_deformation_profile(
                    u,
                    permeant_resname    = permeant_resname,
                    lipid_resnames      = lipid_resnames,
                    phosphorus_name     = phosphorus_name,
                    bilayer_normal_axis = axis,
                    r_max               = r_max,
                    n_bins              = n_radial_bins,
                )
                for bi in range(n_radial_bins):
                    if np.isfinite(def_u[bi]):
                        def_upper_wsum[bi] += weight * def_u[bi]
                        def_upper_wtot[bi] += weight
                    if np.isfinite(def_l[bi]):
                        def_lower_wsum[bi] += weight * def_l[bi]
                        def_lower_wtot[bi] += weight

                # z_edges is handed in, so the returned density is binned on
                # exactly the grid whose centers are written to the CSV. The
                # old call let the function build its own per-frame grid and
                # then discarded the centers it came back with.
                if not _water_z_range_checked:
                    _box_z_now = float(u.dimensions[axis])
                    if (z_edges[-1] - z_edges[0]) > _box_z_now:
                        print(f"  [worker] WARNING: --water-z-range spans "
                              f"{z_edges[-1] - z_edges[0]:.1f} Å but the box "
                              f"is only {_box_z_now:.1f} Å tall; bins beyond "
                              f"±{_box_z_now / 2:.1f} Å are unreachable after "
                              f"minimum-image wrapping and will stay empty.")
                    _water_z_range_checked = True

                _, rad_cnt, _, z_dens, n_def = water_defect_profile(
                    u,
                    permeant_resname    = permeant_resname,
                    lipid_resnames      = lipid_resnames,
                    phosphorus_name     = phosphorus_name,
                    water_resname       = water_resname,
                    water_oxygen_name   = water_oxygen_name,
                    bilayer_normal_axis = axis,
                    r_max               = r_max,
                    n_radial_bins       = n_radial_bins,
                    z_edges             = z_edges,
                )
                water_radial_wsum += weight * rad_cnt
                water_radial_wtot += weight
                water_z_wsum      += weight * z_dens
                water_z_wtot      += weight
                n_defect_wsum[slab_idx] += weight * n_def
                n_defect_wtot[slab_idx] += weight

                # Curvature per species. local_membrane_curvature already
                # returns the upper- and lower-surface fits separately, so
                # there is no additional leaflet dimension to loop over here
                # (see the accumulator-init comment above).
                for sp in _STRUCT_SPECIES:
                    rn_tup = tuple(lipid_resnames) if sp == 'mixed' else (sp,)
                    curv = local_membrane_curvature(
                        u,
                        lipid_resnames      = rn_tup,
                        phosphorus_name     = phosphorus_name,
                        bilayer_normal_axis = axis,
                        rbf_smoothing       = curvature_smoothing,
                        permeant_resname    = permeant_resname,
                        r_max_curvature     = r_max,
                        n_radial_bins       = n_radial_bins,
                    )
                    if np.isfinite(curv['mean_H_upper']):
                        H_upper_wsum[sp][slab_idx] += weight * curv['mean_H_upper']
                        H_upper_wtot[sp][slab_idx] += weight
                    if np.isfinite(curv['mean_H_lower']):
                        H_lower_wsum[sp][slab_idx] += weight * curv['mean_H_lower']
                        H_lower_wtot[sp][slab_idx] += weight
                    if np.isfinite(curv['mean_K_upper']):
                        K_upper_wsum[sp][slab_idx] += weight * curv['mean_K_upper']
                        K_upper_wtot[sp][slab_idx] += weight
                    if sp == 'mixed' and curv['radial_H_upper'] is not None:
                        for bi in range(n_radial_bins):
                            v = curv['radial_H_upper'][bi]
                            if np.isfinite(v):
                                H_radial_wsum[bi] += weight * v
                                H_radial_wtot[bi] += weight

                    # Near-permeant curvature: pool |H| from whichever
                    # leaflet(s) the near lipids actually belong to.
                    if near_sets:
                        residx_cv = np.concatenate(
                            [curv['upper_resindices'], curv['lower_resindices']]
                        )
                        H_cv = np.abs(np.concatenate(
                            [curv['upper_H'], curv['lower_H']]
                        ))
                        for N in near_n:
                            if N not in near_sets:
                                continue
                            near_mask = np.isin(residx_cv, near_sets[N])
                            vals_n = H_cv[near_mask]
                            if len(vals_n) and np.any(np.isfinite(vals_n)):
                                H_val_n = float(np.nanmean(vals_n))
                                if np.isfinite(H_val_n):
                                    near_H_wsum[(N, sp)][slab_idx] += weight * H_val_n
                                    near_H_wtot[(N, sp)][slab_idx] += weight

            # ── DIFFUSION (unchanged) ─────────────────────────────────────────
            if run_diffusion:
                z_full.append(perm_z)
                slab_idx_full.append(slab_idx)

            if run_diffusion_xy:
                raw_xy  = perm_heavy.positions[:, _lat_axes].mean(axis=0)
                box_xy  = u.dimensions[_lat_axes]
                if _pbc_prev is None:
                    _pbc_prev = raw_xy.copy()
                    _pbc_jump = np.zeros(2)
                    unwrapped = raw_xy.copy()
                else:
                    unwrapped = _unwrap_xy_pbc(
                        raw_xy, _pbc_prev, _pbc_jump, box_xy, _pbc_cutoff
                    )
                xy_full.append(unwrapped.copy())
                slab_idx_xy_full.append(slab_idx)

            # ── SPATIAL group ─────────────────────────────────────────────────
            if run_spatial and len(permeant) > 0:
                box_xy_frame = u.dimensions[_lat_axes]

                if not _stamp_r_initialised:
                    if _stamp_r is None:
                        lipid_sel_sp = " or ".join(
                            f"resname {r}" for r in lipid_resnames
                        )
                        p_sp   = u.select_atoms(
                            f"({lipid_sel_sp}) and name {phosphorus_name}"
                        )
                        n_lip  = max(len(p_sp) // 2, 1)
                        A_lip  = (box_xy_frame[0] * box_xy_frame[1]) / n_lip
                        _stamp_r = float(np.sqrt(A_lip / np.pi))
                    _stamp_r_initialised = True

                all_pos_sp  = u.atoms.positions
                perm_lat_sp = permeant.center_of_geometry()[_lat_axes]

                # Which depth bin this frame's stamps belong to. Uses the
                # same perm_z as the 1-D slab assignment, so map bin d
                # always corresponds to a known slab_center range.
                _depth_idx = int(np.clip(
                    np.searchsorted(_depth_edges, perm_z, side='right') - 1,
                    0, _n_depth - 1
                ))

                def _stamp_depth(acc_key, xy, vals, _d=_depth_idx):
                    wsum, wcnt = _spatial_acc[acc_key]
                    _stamp_frame(xy, vals, _grid_tree, _stamp_r,
                                 wsum[_d], wcnt[_d], weight)

                for chain, bonds in chain_bonds.items():
                    first_bond_key = (chain, 0)
                    if first_bond_key not in bond_idx:
                        continue
                    base_idx_sp = bond_idx[first_bond_key][0]
                    lip_xy_sp   = _pbc_minimum_image_2d(
                        all_pos_sp[base_idx_sp][:, _lat_axes]
                        - perm_lat_sp[np.newaxis, :],
                        box_xy_frame,
                    )

                    # Leaflet masks from base-atom z
                    base_z_sp = all_pos_sp[base_idx_sp, axis]
                    mid_z_sp  = base_z_sp.mean()
                    lf_masks_sp = {
                        'both' : np.ones(len(base_idx_sp), dtype=bool),
                        'upper': base_z_sp > mid_z_sp,
                        'lower': base_z_sp <= mid_z_sp,
                    }

                    bond_p2_per_lip = []
                    for bi in range(len(bonds)):
                        key = (chain, bi)
                        if key not in bond_idx:
                            continue
                        ia1, ia2 = bond_idx[key]
                        vecs  = all_pos_sp[ia2] - all_pos_sp[ia1]
                        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                        ok    = norms[:, 0] > 1e-6
                        p2_per_lip = np.full(len(ia1), np.nan)
                        if ok.any():
                            v_hat = vecs[ok] / norms[ok]
                            cos2  = (v_hat @ normal) ** 2
                            p2_per_lip[ok] = 0.5 * (3.0 * cos2 - 1.0)
                        bond_p2_per_lip.append(p2_per_lip)

                        for sp in ['mixed'] + list(lipid_resnames):
                            sp_sfx  = '' if sp == 'mixed' else f'_{sp}'
                            sp_mask = (np.ones(len(ia1), dtype=bool)
                                       if sp == 'mixed'
                                       else species_mask.get(key, {}).get(
                                           sp, np.zeros(len(ia1), dtype=bool)))
                            for lf in _LEAFLETS:
                                lf_sfx   = '' if lf == 'both' else f'_{lf}'
                                combined = sp_mask & lf_masks_sp[lf]
                                sfx_full = f'{sp_sfx}{lf_sfx}'
                                acc_key  = f'p2_{chain}_bond_{bi}{sfx_full}'
                                if acc_key not in _spatial_acc:
                                    continue
                                _stamp_depth(acc_key,
                                             lip_xy_sp[combined],
                                             p2_per_lip[combined])

                    if bond_p2_per_lip:
                        mat = np.column_stack(bond_p2_per_lip)
                        for sp in ['mixed'] + list(lipid_resnames):
                            sp_sfx  = '' if sp == 'mixed' else f'_{sp}'
                            sp_mask = (np.ones(mat.shape[0], dtype=bool)
                                       if sp == 'mixed'
                                       else species_mask.get(
                                           (chain, 0), {}
                                       ).get(sp, np.zeros(mat.shape[0],
                                                          dtype=bool)))
                            for lf in _LEAFLETS:
                                lf_sfx   = '' if lf == 'both' else f'_{lf}'
                                combined = sp_mask & lf_masks_sp[lf]
                                sfx_full = f'{sp_sfx}{lf_sfx}'
                                acc_key  = f'p2_{chain}_chain{sfx_full}'
                                if acc_key not in _spatial_acc:
                                    continue
                                chain_p2_sp = (np.nanmean(mat[combined], axis=1)
                                               if combined.any()
                                               else np.full(combined.sum(), np.nan))
                                _stamp_depth(acc_key,
                                             lip_xy_sp[combined],
                                             chain_p2_sp)

                    # Tilt angles per species × leaflet
                    if chain in tilt_idx:
                        ib, it  = tilt_idx[chain]

                        # Leaflet masks for tilt (based on base-atom z).
                        # Computed up front because θ needs a per-lipid
                        # sign correction (see below).
                        base_z_tilt = all_pos_sp[ib, axis]
                        mid_z_tilt  = base_z_tilt.mean()
                        lf_masks_tilt = {
                            'both' : np.ones(len(ib), dtype=bool),
                            'upper': base_z_tilt > mid_z_tilt,
                            'lower': base_z_tilt <= mid_z_tilt,
                        }

                        vt      = all_pos_sp[it] - all_pos_sp[ib]
                        vt_norm = np.linalg.norm(vt, axis=1, keepdims=True)
                        ok_t    = vt_norm[:, 0] > 1e-6
                        theta_lip = np.full(len(ib), np.nan)
                        phi_lip   = np.full(len(ib), np.nan)
                        if ok_t.any():
                            vh    = vt[ok_t] / vt_norm[ok_t]
                            # tip-base points inward (toward the bilayer
                            # core) for both leaflets — see mean_tilt_angle
                            # docstring. Compare against each lipid's own
                            # inward reference (-normal upper, +normal
                            # lower) so an untilted lipid reports θ≈0,
                            # not θ≈180.
                            sign  = np.where(lf_masks_tilt['upper'][ok_t],
                                             -1.0, 1.0)
                            cos_t = np.clip(sign * (vh @ normal), -1.0, 1.0)
                            theta_lip[ok_t] = np.degrees(np.arccos(cos_t))
                            vpl   = (vt[ok_t]
                                     - (vt[ok_t] @ normal)[:, None] * normal)
                            pn    = np.linalg.norm(vpl, axis=1)
                            y_ax  = np.array([0.0, 1.0, 0.0])
                            phi_vals = np.full(ok_t.sum(), np.nan)
                            mask_phi = pn > 1e-6
                            if mask_phi.any():
                                cos_phi = np.clip(
                                    (vpl[mask_phi] / pn[mask_phi, None]) @ y_ax,
                                    -1.0, 1.0
                                )
                                phi_vals[mask_phi] = np.degrees(
                                    np.arccos(cos_phi)
                                )
                            phi_lip[ok_t] = phi_vals

                        for sp in ['mixed'] + list(lipid_resnames):
                            sp_sfx  = '' if sp == 'mixed' else f'_{sp}'
                            sp_mask = (np.ones(len(ib), dtype=bool)
                                       if sp == 'mixed'
                                       else tilt_sp_mask.get(chain, {}).get(
                                           sp, np.zeros(len(ib), dtype=bool)))
                            for lf in _LEAFLETS:
                                lf_sfx   = '' if lf == 'both' else f'_{lf}'
                                combined = sp_mask & lf_masks_tilt[lf]
                                sfx_full = f'{sp_sfx}{lf_sfx}'
                                for angle, vals_arr in (('theta', theta_lip),
                                                        ('phi',   phi_lip)):
                                    acc_key = f'{angle}_{chain}{sfx_full}'
                                    if acc_key not in _spatial_acc:
                                        continue
                                    _stamp_depth(acc_key,
                                                 lip_xy_sp[combined],
                                                 vals_arr[combined])

                # Thickness maps per species × leaflet
                for sp in ['mixed'] + list(lipid_resnames):
                    sp_sfx = '' if sp == 'mixed' else f'_{sp}'
                    rn_tup = tuple(lipid_resnames) if sp == 'mixed' else (sp,)
                    td = _local_thickness_dict(u, rn_tup, phosphorus_name, axis)

                    # 'both' must pool BOTH leaflets' P atoms into the same
                    # map (two stamp calls into the same acc_key), not just
                    # reuse the upper-leaflet data — otherwise the 'both'
                    # map is identical to the 'upper' map.
                    for lf, xy_key, vals_key in (
                        ('both',  'upper_xy', 'thickness_upper'),
                        ('both',  'lower_xy', 'thickness_lower'),
                        ('upper', 'upper_xy', 'thickness_upper'),
                        ('lower', 'lower_xy', 'thickness_lower'),
                    ):
                        lf_sfx  = '' if lf == 'both' else f'_{lf}'
                        acc_key = f'thickness{sp_sfx}{lf_sfx}'
                        if acc_key not in _spatial_acc:
                            continue
                        xy_sp = td[xy_key]
                        vs    = td[vals_key]
                        if len(xy_sp) > 0:
                            lip_xy_t = _pbc_minimum_image_2d(
                                xy_sp - perm_lat_sp[np.newaxis, :],
                                box_xy_frame,
                            )
                            _stamp_depth(acc_key, lip_xy_t, vs)

                # Curvature maps per species × leaflet
                for sp in ['mixed'] + list(lipid_resnames):
                    sp_sfx  = '' if sp == 'mixed' else f'_{sp}'
                    rn_tup  = tuple(lipid_resnames) if sp == 'mixed' else (sp,)
                    curv_sp = local_membrane_curvature(
                        u,
                        lipid_resnames      = rn_tup,
                        phosphorus_name     = phosphorus_name,
                        bilayer_normal_axis = axis,
                        rbf_smoothing       = curvature_smoothing,
                    )
                    # 'both' must pool BOTH leaflets' curvature into the
                    # same map (two stamp calls into the same acc_key), not
                    # just reuse the upper-leaflet data — otherwise the
                    # 'both' map is identical to the 'upper' map (e.g.
                    # "mean curvature DOPC" == "mean curvature DOPC (upper
                    # leaflet)").
                    for lf, xy_key, h_key in (
                        ('both',  'upper_xy', 'upper_H'),
                        ('both',  'lower_xy', 'lower_H'),
                        ('upper', 'upper_xy', 'upper_H'),
                        ('lower', 'lower_xy', 'lower_H'),
                    ):
                        lf_sfx  = '' if lf == 'both' else f'_{lf}'
                        acc_key = f'curvature_H{sp_sfx}{lf_sfx}'
                        if acc_key not in _spatial_acc:
                            continue
                        xy_c = curv_sp[xy_key]
                        hc   = curv_sp[h_key]
                        if len(hc) > 0:
                            lip_xy_c = _pbc_minimum_image_2d(
                                xy_c - perm_lat_sp[np.newaxis, :],
                                box_xy_frame,
                            )
                            _stamp_depth(acc_key, lip_xy_c, np.abs(hc))

    # ── Diffusion post-processing (unchanged) ─────────────────────────────────
    if run_diffusion and len(z_full) >= 2 and _dt is not None:
        z_arr  = np.array(z_full)
        si_arr = np.array(slab_idx_full, dtype=int)

        sojourn_D_ein = [[] for _ in range(n_slabs)]
        sojourn_D_hum = [[] for _ in range(n_slabs)]

        run_start = 0
        for f in range(1, len(si_arr) + 1):
            if f == len(si_arr) or si_arr[f] != si_arr[run_start]:
                run_len = f - run_start
                si      = si_arr[run_start]
                if run_len >= min_slab_points:
                    z_run = z_arr[run_start:f]
                    D_ein, D_hum = _estimate_sojourn_D(
                        z_run, _dt, _max_lag_steps
                    )
                    if np.isfinite(D_ein):
                        sojourn_D_ein[si].append(D_ein)
                    if np.isfinite(D_hum):
                        sojourn_D_hum[si].append(D_hum)
                run_start = f

        for si in range(n_slabs):
            if sojourn_D_ein[si]:
                D_ein_wsum[si] += weight * float(np.mean(sojourn_D_ein[si]))
                D_ein_wtot[si] += weight
            if sojourn_D_hum[si]:
                D_hum_wsum[si] += weight * float(np.mean(sojourn_D_hum[si]))
                D_hum_wtot[si] += weight
            n_soj = len(sojourn_D_ein[si])
            if n_soj > 0:
                nsoj_wsum[si] += weight * n_soj
                nsoj_wtot[si] += weight

    if run_diffusion_xy and len(xy_full) >= 2 and _dt is not None:
        xy_arr  = np.array(xy_full)
        si_arr  = np.array(slab_idx_xy_full, dtype=int)

        sojourn_D_xy_ein = [[] for _ in range(n_slabs)]
        sojourn_D_xy_hx  = [[] for _ in range(n_slabs)]
        sojourn_D_xy_hy  = [[] for _ in range(n_slabs)]

        run_start = 0
        for f in range(1, len(si_arr) + 1):
            if f == len(si_arr) or si_arr[f] != si_arr[run_start]:
                run_len = f - run_start
                si      = si_arr[run_start]
                if run_len >= min_slab_points:
                    xy_run = xy_arr[run_start:f]
                    D_ein_xy, D_hx, D_hy = _msd_xy_from_series(
                        xy_run, _dt, _max_lag_steps
                    )
                    if np.isfinite(D_ein_xy):
                        sojourn_D_xy_ein[si].append(D_ein_xy)
                    if np.isfinite(D_hx):
                        sojourn_D_xy_hx[si].append(D_hx)
                    if np.isfinite(D_hy):
                        sojourn_D_xy_hy[si].append(D_hy)
                run_start = f

        for si in range(n_slabs):
            if sojourn_D_xy_ein[si]:
                D_xy_ein_wsum[si] += weight * float(np.mean(sojourn_D_xy_ein[si]))
                D_xy_ein_wtot[si] += weight
            if sojourn_D_xy_hx[si]:
                D_xy_hx_wsum[si] += weight * float(np.mean(sojourn_D_xy_hx[si]))
                D_xy_hx_wtot[si] += weight
            if sojourn_D_xy_hy[si]:
                D_xy_hy_wsum[si] += weight * float(np.mean(sojourn_D_xy_hy[si]))
                D_xy_hy_wtot[si] += weight
            n_soj_xy = len(sojourn_D_xy_ein[si])
            if n_soj_xy > 0:
                nsoj_xy_wsum[si] += weight * n_soj_xy
                nsoj_xy_wtot[si] += weight

    # ── Assemble DataFrames ───────────────────────────────────────────────────
    output = {}

    if run_order:
        p2_data   = {'slab_center': centers}
        tilt_data = {'slab_center': centers}

        for sp in _ORDER_SPECIES:
            sp_pfx = '' if sp == 'mixed' else f'{sp}_'
            for lf in _LEAFLETS:
                lf_pfx = '' if lf == 'both' else f'{lf}_'
                pfx = f'{lf_pfx}{sp_pfx}'
                for chain, bonds in chain_bonds.items():
                    for bi in range(len(bonds)):
                        p2_data[f'{pfx}{chain}_p2_bond_{bi}_wsum'] = p2_wsum[(sp, lf, chain, bi)]
                        p2_data[f'{pfx}{chain}_p2_bond_{bi}_wtot'] = p2_wtot[(sp, lf, chain, bi)]
                    p2_data[f'{pfx}{chain}_p2_chain_wsum'] = p2_wsum[(sp, lf, chain, 'chain')]
                    p2_data[f'{pfx}{chain}_p2_chain_wtot'] = p2_wtot[(sp, lf, chain, 'chain')]

                for chain in tilt_endpoints:
                    tilt_data[f'{pfx}{chain}_theta_wsum'] = theta_wsum[(sp, lf, chain)]
                    tilt_data[f'{pfx}{chain}_theta_wtot'] = theta_wtot[(sp, lf, chain)]
                    tilt_data[f'{pfx}{chain}_phi_wsum']   = phi_wsum[(sp, lf, chain)]
                    tilt_data[f'{pfx}{chain}_phi_wtot']   = phi_wtot[(sp, lf, chain)]

        # Near-permeant columns: not leaflet-split, prefixed 'near{N}_'.
        for N in near_n:
            for sp in _ORDER_SPECIES:
                sp_pfx = '' if sp == 'mixed' else f'{sp}_'
                pfx = f'near{N}_{sp_pfx}'
                for chain, bonds in chain_bonds.items():
                    for bi in range(len(bonds)):
                        p2_data[f'{pfx}{chain}_p2_bond_{bi}_wsum'] = near_p2_wsum[(N, sp, chain, bi)]
                        p2_data[f'{pfx}{chain}_p2_bond_{bi}_wtot'] = near_p2_wtot[(N, sp, chain, bi)]
                    p2_data[f'{pfx}{chain}_p2_chain_wsum'] = near_p2_wsum[(N, sp, chain, 'chain')]
                    p2_data[f'{pfx}{chain}_p2_chain_wtot'] = near_p2_wtot[(N, sp, chain, 'chain')]

                for chain in tilt_endpoints:
                    tilt_data[f'{pfx}{chain}_theta_wsum'] = near_theta_wsum[(N, sp, chain)]
                    tilt_data[f'{pfx}{chain}_theta_wtot'] = near_theta_wtot[(N, sp, chain)]
                    tilt_data[f'{pfx}{chain}_phi_wsum']   = near_phi_wsum[(N, sp, chain)]
                    tilt_data[f'{pfx}{chain}_phi_wtot']   = near_phi_wtot[(N, sp, chain)]

        output['order'] = (pd.DataFrame(p2_data), pd.DataFrame(tilt_data))

    if run_structural:
        thick_dict = {'slab_center': centers}
        for sp in _STRUCT_SPECIES:
            sp_pfx = '' if sp == 'mixed' else f'{sp}_'
            for lf in _LEAFLETS:
                lf_pfx = '' if lf == 'both' else f'{lf}_'
                pfx = f'{lf_pfx}{sp_pfx}'
                thick_dict[f'{pfx}thickness_wsum'] = thick_wsum[(sp, lf)]
                thick_dict[f'{pfx}thickness_wtot'] = thick_wtot[(sp, lf)]

            # Curvature has no leaflet dimension of its own to loop over —
            # H_upper/H_lower/K_upper already ARE the upper-/lower-surface
            # split (see accumulator-init comment above).
            sp_only_pfx = sp_pfx
            thick_dict[f'{sp_only_pfx}H_upper_wsum'] = H_upper_wsum[sp]
            thick_dict[f'{sp_only_pfx}H_upper_wtot'] = H_upper_wtot[sp]
            thick_dict[f'{sp_only_pfx}K_upper_wsum'] = K_upper_wsum[sp]
            thick_dict[f'{sp_only_pfx}K_upper_wtot'] = K_upper_wtot[sp]
            thick_dict[f'{sp_only_pfx}H_lower_wsum'] = H_lower_wsum[sp]
            thick_dict[f'{sp_only_pfx}H_lower_wtot'] = H_lower_wtot[sp]
        thick_dict['n_defect_wsum'] = n_defect_wsum
        thick_dict['n_defect_wtot'] = n_defect_wtot

        # Near-permeant columns: not leaflet-split, prefixed 'near{N}_'.
        for N in near_n:
            for sp in _STRUCT_SPECIES:
                sp_pfx = '' if sp == 'mixed' else f'{sp}_'
                pfx = f'near{N}_{sp_pfx}'
                thick_dict[f'{pfx}thickness_wsum'] = near_thick_wsum[(N, sp)]
                thick_dict[f'{pfx}thickness_wtot'] = near_thick_wtot[(N, sp)]
                thick_dict[f'{pfx}H_wsum']         = near_H_wsum[(N, sp)]
                thick_dict[f'{pfx}H_wtot']         = near_H_wtot[(N, sp)]

        output['structural'] = (
            pd.DataFrame(thick_dict),
            pd.DataFrame({
                'r_center'         : r_centers,
                'def_upper_wsum'   : def_upper_wsum,
                'def_upper_wtot'   : def_upper_wtot,
                'def_lower_wsum'   : def_lower_wsum,
                'def_lower_wtot'   : def_lower_wtot,
                'H_radial_wsum'    : H_radial_wsum,
                'H_radial_wtot'    : H_radial_wtot,
                'water_radial_wsum': water_radial_wsum,
                'water_radial_wtot': water_radial_wtot,
            }),
            pd.DataFrame({
                'z_center'     : z_centers,
                'water_z_wsum' : water_z_wsum,
                'water_z_wtot' : water_z_wtot,
            }),
        )

    if run_diffusion:
        output['diffusion'] = (
            pd.DataFrame({
                'slab_center'    : centers,
                'D_einstein_wsum': D_ein_wsum,
                'D_einstein_wtot': D_ein_wtot,
                'D_hummer_wsum'  : D_hum_wsum,
                'D_hummer_wtot'  : D_hum_wtot,
                'n_sojourns_wsum': nsoj_wsum,
                'n_sojourns_wtot': nsoj_wtot,
            }),
        )

    if run_diffusion_xy:
        output['diffusion_xy'] = (
            pd.DataFrame({
                'slab_center'       : centers,
                'D_xy_ein_wsum'     : D_xy_ein_wsum,
                'D_xy_ein_wtot'     : D_xy_ein_wtot,
                'D_xy_hummer_x_wsum': D_xy_hx_wsum,
                'D_xy_hummer_x_wtot': D_xy_hx_wtot,
                'D_xy_hummer_y_wsum': D_xy_hy_wsum,
                'D_xy_hummer_y_wtot': D_xy_hy_wtot,
                'n_sojourns_wsum'   : nsoj_xy_wsum,
                'n_sojourns_wtot'   : nsoj_xy_wtot,
            }),
        )

    if run_spatial:
        output['spatial'] = (_spatial_acc, _grid_shape, _x_coords, _y_coords,
                             _depth_edges)

    return output


# ═════════════════════════════════════════════════════════════════════════════
# Per-path worker (unchanged except function calls are compatible)
# ═════════════════════════════════════════════════════════════════════════════

_GROUP_CSV_KEYS = {
    'order'        : [(0, 'p2'),    (1, 'tilt')],
    'structural'   : [(0, 'thick'), (1, 'deform'), (2, 'water_z')],
    'diffusion'    : [(0, 'diff')],
    'diffusion_xy' : [(0, 'diff_xy')],
}


def process_single_path(
    path_number,
    overwrite,
    weights,
    lambda_A,
    lambda_B,
    lambda_minus_one,
    infretis_data_file,
    permeant_resname    = 'ORP',
    lipid_resnames      = ('DOPC', 'POPC'),
    slab_width          = 1.0,
    bilayer_normal      = 'z',
    slab_range          = None,
    phosphorus_name     = PHOSPHORUS_NAME,
    water_resname       = WATER_RESNAME,
    water_oxygen_name   = WATER_OXYGEN_NAME,
    r_max               = 40.0,
    n_radial_bins       = 20,
    n_z_bins            = 50,
    water_z_range       = WATER_Z_RANGE,
    curvature_smoothing = 0.0,
    dt                  = None,
    max_lag_ps          = 500.0,
    min_slab_points     = 5,
    grid_spacing        = 5.0,
    stamp_radius        = None,
    near_n              = (),
    spatial_extent      = None,
    n_depth_bins        = 10,
):
    csvs   = _all_csv_paths(path_number)
    needed = _needed_groups(csvs, overwrite, near_n)

    if not needed:
        return (path_number, 'skipped', 'All output files already exist.')

    path_folder = f"../load/{path_number}/accepted/"
    if not os.path.exists(path_folder):
        return (path_number, 'skipped',
                f'Path folder does not exist: {path_folder}')

    try:
        order = np.loadtxt(f"../load/{path_number}/order.txt",
                           comments=('#', '@'))
        traj  = np.loadtxt(f"../load/{path_number}/traj.txt",
                           comments=('#', '@'), usecols=(2,))
        op    = np.column_stack((order, traj))
        first, last = op[0, 1], op[-1, 1]

        ensemble = "plus" \
            if (lambda_minus_one < first < lambda_A) or \
               (lambda_minus_one < last  < lambda_A) \
            else "minus"

        reactive = get_reactive_paths(path_number, infretis_data_file, lambda_B)
        if reactive is None:
            return (path_number, 'skipped', 'Path not found in infretis_data.txt')

        weight = weights.get(path_number, None)
        if weight is None:
            print(f"  [worker] WARNING: path {path_number} not in weights — defaulting to 0.0")
            weight = 0.0

        first_frame, last_frame = remove_first_last_frames(
            ensemble, lambda_minus_one, lambda_A, first, last
        )

        xtc_names, directions, _ = extract_sorted_traj_names(
            f"../load/{path_number}/traj.txt"
        )
        xtc_files = [os.path.join(path_folder, f) for f in xtc_names]

        if not xtc_files:
            return (path_number, 'skipped', 'No xtc files found.')

        skipped_groups = set(OBSERVABLE_GROUPS.keys()) - needed
        print(
            f"[worker] Path {path_number} (weight={weight:.6e}): "
            f"{len(xtc_files)} xtc file(s) — computing {sorted(needed)}"
            + (f", skipping {sorted(skipped_groups)}" if skipped_groups else "")
        )

        results = calculate_selected_slab_profiles(
            topology            = topol_file,
            xtc_files           = xtc_files,
            xtc_directions      = directions,
            permeant_resname    = permeant_resname,
            needed_groups       = needed,
            weight              = weight,
            first_frame         = first_frame,
            last_frame          = last_frame,
            lipid_resnames      = lipid_resnames,
            slab_width          = slab_width,
            bilayer_normal      = bilayer_normal,
            slab_range          = slab_range,
            chain_bonds         = CHAIN_BONDS,
            tilt_endpoints      = TILT_ENDPOINTS,
            phosphorus_name     = phosphorus_name,
            water_resname       = water_resname,
            water_oxygen_name   = water_oxygen_name,
            r_max               = r_max,
            n_radial_bins       = n_radial_bins,
            n_z_bins            = n_z_bins,
            water_z_range       = water_z_range,
            curvature_smoothing = curvature_smoothing,
            dt                  = dt,
            max_lag_ps          = max_lag_ps,
            min_slab_points     = min_slab_points,
            grid_spacing        = grid_spacing,
            stamp_radius        = stamp_radius,
            near_n              = near_n,
            spatial_extent      = spatial_extent,
            n_depth_bins        = n_depth_bins,
        )

        header = (f"# {'reactive' if reactive else 'non-reactive'}\n"
                  f"# {ensemble} ensemble\n")

        written = []

        for group, df_tuple in results.items():
            if group == 'spatial':
                continue
            for df_idx, csv_key in _GROUP_CSV_KEYS[group]:
                csv_path = csvs[csv_key]
                df       = df_tuple[df_idx]
                with open(csv_path, 'w') as fh:
                    fh.write(header)
                df.to_csv(csv_path, mode='a', index=False, float_format='%.6e')
                written.append(csv_path.name)

        if 'spatial' in results:
            sp_acc, sp_shape, sp_x, sp_y, sp_depth = results['spatial']
            _save_spatial_accumulators(sp_acc, sp_x, sp_y, sp_depth,
                                       csvs['spatial'])
            written.append(csvs['spatial'].name)

        print(f"  [worker] Path {path_number} -> {written}")
        return (path_number, 'ok')

    except Exception:
        return (path_number, 'error', traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
# Parallel dispatcher (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def loop_over_paths_parallel(
    path_start,
    path_end,
    overwrite           = False,
    weights_file        = 'path_weights.txt',
    data_file           = None,
    permeant_resname    = 'ORP',
    lipid_resnames      = ('DOPC', 'POPC'),
    slab_width          = 1.0,
    bilayer_normal      = 'z',
    slab_range          = None,
    phosphorus_name     = PHOSPHORUS_NAME,
    water_resname       = WATER_RESNAME,
    water_oxygen_name   = WATER_OXYGEN_NAME,
    r_max               = 40.0,
    n_radial_bins       = 20,
    n_z_bins            = 50,
    water_z_range       = WATER_Z_RANGE,
    curvature_smoothing = 0.0,
    dt                  = None,
    max_lag_ps          = 500.0,
    min_slab_points     = 5,
    grid_spacing        = 5.0,
    stamp_radius        = None,
    near_n              = (),
    spatial_extent      = None,
    n_depth_bins        = 10,
):
    Path(ORDER_OUT).mkdir(parents=True, exist_ok=True)

    lambda_A, lambda_B, lambda_minus_one = read_toml('../infretis.toml')
    weights      = load_path_weights(weights_file)
    path_numbers = list(range(path_start, (path_end or path_start) + 1))

    print(f"Merged analysis: paths {path_start}–{path_end}, "
          f"{N_WORKERS} workers (single pass per path).")
    print(f"Weights loaded from '{weights_file}': {len(weights)} entries.")
    print("=" * 80)

    results = {'ok': [], 'skipped': [], 'error': []}

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        future_to_path = {
            # Keywords, not positionals: this call used to be ~20 bare
            # positional arguments, so inserting a parameter anywhere in
            # process_single_path's signature silently shifted every
            # argument after it onto the wrong name.
            executor.submit(
                process_single_path,
                pn, overwrite, weights, lambda_A, lambda_B, lambda_minus_one,
                data_file,
                permeant_resname    = permeant_resname,
                lipid_resnames      = lipid_resnames,
                slab_width          = slab_width,
                bilayer_normal      = bilayer_normal,
                slab_range          = slab_range,
                phosphorus_name     = phosphorus_name,
                water_resname       = water_resname,
                water_oxygen_name   = water_oxygen_name,
                r_max               = r_max,
                n_radial_bins       = n_radial_bins,
                n_z_bins            = n_z_bins,
                water_z_range       = water_z_range,
                curvature_smoothing = curvature_smoothing,
                dt                  = dt,
                max_lag_ps          = max_lag_ps,
                min_slab_points     = min_slab_points,
                grid_spacing        = grid_spacing,
                stamp_radius        = stamp_radius,
                near_n              = near_n,
                spatial_extent      = spatial_extent,
                n_depth_bins        = n_depth_bins,
            ): pn
            for pn in path_numbers
        }
        for future in as_completed(future_to_path):
            pn = future_to_path[future]
            try:
                result = future.result()
            except Exception:
                result = (pn, 'error', traceback.format_exc())
            status = result[1]
            results[status].append(pn)
            if status == 'ok':
                print(f"  Path {pn} completed.")
            elif status == 'skipped':
                print(f"  [–] Path {pn} skipped: {result[2]}")
            elif status == 'error':
                print(f"  Path {pn} FAILED — see {ERROR_LOG}")
                with open(ERROR_LOG, 'a') as ef:
                    ef.write(f"\n{'='*60}\nPATH {pn} FAILED\n{'='*60}\n")
                    ef.write(result[2])

    print("\n" + "=" * 80)
    print("MERGED ANALYSIS SUMMARY")
    print(f"  Completed : {len(results['ok'])}")
    print(f"  Skipped   : {len(results['skipped'])}")
    print(f"  Failed    : {len(results['error'])}")
    if results['error']:
        print(f"  Error details written to: {ERROR_LOG}")
    print("=" * 80)


# ═════════════════════════════════════════════════════════════════════════════
# Aggregation helpers
# ═════════════════════════════════════════════════════════════════════════════

def _assign_blocks(path_numbers, n_blocks):
    """
    Split paths into contiguous blocks of MC time for a block jackknife.

    Path numbers increase monotonically along the infretis Monte-Carlo
    chain, so contiguous ranges of path number are contiguous stretches of
    MC time. Successive accepted paths are correlated (a shooting move
    reuses most of its parent), which means treating individual paths as
    independent samples underestimates the error. Deleting whole blocks
    instead absorbs that correlation, provided each block is longer than
    the path-decorrelation time.

    Returns
    -------
    block_of : (P,) int array — block index for each entry of path_numbers
    n_used   : int            — number of blocks actually created
    """
    pn      = np.asarray(path_numbers)
    n_paths = len(pn)
    if n_paths == 0:
        return np.zeros(0, dtype=int), 0
    K = int(min(max(int(n_blocks), 1), n_paths))
    order    = np.argsort(pn, kind='stable')
    block_of = np.empty(n_paths, dtype=int)
    block_of[order] = (np.arange(n_paths) * K) // n_paths
    return block_of, K


def _pool_weighted_means(all_dfs, wsum_cols, wtot_cols,
                         path_numbers=None, n_blocks=20):
    """
    Pool per-path (wsum, wtot) accumulators into weighted means, and keep
    the information needed to judge whether those means mean anything.

    In addition to the mean, each observable now gets three companion
    columns:

      '<obs>_wtot' — total accumulated weight in the bin. Zero-ish weight
                     is what makes the tails of the slab range look like
                     structure when they are noise.
      '<obs>_neff' — Kish effective number of contributing paths,
                     (Σw)² / Σw², computed over paths. Path weights here
                     are strongly skewed, so a bin fed by one dominant
                     path has n_eff ≈ 1 no matter how large its raw wtot.
      '<obs>_err'  — block-jackknife standard error of the mean.

    The estimator is a ratio of sums over paths, μ = Σ wsum_p / Σ wtot_p,
    so the delete-one-block jackknife applies directly and needs a single
    vectorised pass rather than a resampling loop. It is also
    deterministic, which a bootstrap is not — the same inputs give the
    same error bars on every rerun, which matters once these end up in a
    figure.
    """
    all_centers = np.unique(np.round(
        np.concatenate([df.iloc[:, 0].values for df in all_dfs]), 6
    ))
    n = len(all_centers)
    C = len(wsum_cols)

    if path_numbers is None:
        path_numbers = np.arange(len(all_dfs))
    block_of, K = _assign_blocks(path_numbers, n_blocks)

    # (K, n_bins, n_obs) block-wise accumulators, plus Σw² per bin for n_eff.
    block_wsum = np.zeros((K, n, C))
    block_wtot = np.zeros((K, n, C))
    wtot_sq    = np.zeros((n, C))

    for df, k in zip(all_dfs, block_of):
        idx_map = np.searchsorted(
            all_centers, np.round(df.iloc[:, 0].values, 6)
        )
        ws_p = np.zeros((n, C))
        wt_p = np.zeros((n, C))
        for ci, (ws, wt) in enumerate(zip(wsum_cols, wtot_cols)):
            if ws in df.columns and wt in df.columns:
                np.add.at(ws_p[:, ci], idx_map, df[ws].values)
                np.add.at(wt_p[:, ci], idx_map, df[wt].values)
        block_wsum[k] += ws_p
        block_wtot[k] += wt_p
        wtot_sq       += wt_p ** 2

    A = block_wsum.sum(axis=0)
    B = block_wtot.sum(axis=0)

    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(B > 0, A / np.where(B > 0, B, 1.0), np.nan)
        neff = np.where(wtot_sq > 0, B ** 2 / np.where(wtot_sq > 0, wtot_sq, 1.0), 0.0)

        # Delete-one-block jackknife over the ratio estimator. A block that
        # empties a bin leaves the leave-one-out estimate undefined there;
        # drop it from the spread and from the (K-1)/K factor rather than
        # letting it poison the whole bin.
        A_k   = A[None, :, :] - block_wsum
        B_k   = B[None, :, :] - block_wtot
        valid = B_k > 0
        mu_k  = np.where(valid, A_k / np.where(valid, B_k, 1.0), np.nan)

        # nansum/n_valid rather than nanmean: bins with no data at all are
        # routine here (the tails of the slab range), and nanmean would
        # emit an "empty slice" warning for every one of them.
        n_valid = valid.sum(axis=0)
        mu_bar  = np.where(n_valid > 0,
                           np.nansum(mu_k, axis=0) / np.maximum(n_valid, 1),
                           np.nan)
        ss      = np.nansum((mu_k - mu_bar[None, :, :]) ** 2, axis=0)
        err     = np.where(
            n_valid > 1,
            np.sqrt((n_valid - 1) / np.maximum(n_valid, 1) * ss),
            np.nan,
        )

    result = {all_dfs[0].columns[0]: all_centers}
    for ci, ws in enumerate(wsum_cols):
        col_name = ws.replace('_wsum', '')
        result[col_name]             = mean[:, ci]
        result[f'{col_name}_wtot']   = B[:, ci]
        result[f'{col_name}_neff']   = neff[:, ci]
        result[f'{col_name}_err']    = err[:, ci]

    return pd.DataFrame(result)


def contains_key(d, key):
    return key in d


def _wsum_wtot_cols(dfs):
    """
    Union the '_wsum' columns across ALL supplied DataFrames, not just the
    first — a single stale/pre-schema-change CSV in the list must not
    silently drop columns (e.g. new near-permeant columns) from the pool.
    """
    all_cols = set()
    for df in dfs:
        all_cols.update(df.columns)
    wsum = sorted(c for c in all_cols if c.endswith('_wsum'))
    wtot = [c.replace('_wsum', '_wtot') for c in wsum]
    return wsum, wtot


def _collect_csvs(path_start, path_end, weights, kind):
    """Return (path_numbers, dfs) — the path numbers are needed to build
    contiguous MC-time blocks for the jackknife in _pool_weighted_means."""
    pns, dfs = [], []
    for pn in range(path_start, path_end + 1):
        csvs = _all_csv_paths(pn)
        csv  = csvs.get(kind)
        if csv and csv.exists() and contains_key(weights, pn):
            dfs.append(pd.read_csv(csv, comment='#'))
            pns.append(pn)
        else:
            print(f"  [aggregate] Path {pn} '{kind}' CSV not found or "
                  f"no weight available, skipping.")
    return pns, dfs


def aggregate_all_results(path_start, path_end, weights, n_blocks=20):
    results = {}
    for kind in ('p2', 'tilt', 'thick', 'deform', 'water_z', 'diff', 'diff_xy'):
        pns, dfs = _collect_csvs(path_start, path_end, weights, kind)
        if not dfs:
            raise RuntimeError(f"No '{kind}' CSVs found to aggregate.")
        ws, wt = _wsum_wtot_cols(dfs)
        results[kind] = _pool_weighted_means(dfs, ws, wt,
                                             path_numbers=pns,
                                             n_blocks=n_blocks)
    return (results['p2'], results['tilt'], results['thick'],
            results['deform'], results['water_z'], results['diff'],
            results['diff_xy'])


# ═════════════════════════════════════════════════════════════════════════════
# Plot style helpers
# ═════════════════════════════════════════════════════════════════════════════

# Species: label, color, linestyle
_SPECIES_STYLE = {
    'mixed': ('All lipids', '#333333', '-'),
    'DOPC' : ('DOPC',       '#1f77b4', '--'),
    'POPC' : ('POPC',       '#d62728', ':'),
}

# Leaflet: label, line-dash modifier (applied on top of species linestyle)
_LEAFLET_STYLE = {
    'both' : ('both leaflets', 1.0,  ''),        # no change to linestyle
    'upper': ('upper leaflet', 0.85, (5, 2)),     # dashed overlay marker
    'lower': ('lower leaflet', 0.55, (2, 2)),     # dotted overlay marker
}

def _col_prefix(sp, lf):
    """
    Return the CSV column prefix for a given (species, leaflet) combination.
    Convention: '{lf_}_{sp_}' where both are '' for the default ('both'/'mixed').
    e.g. ('mixed','both') -> ''
         ('DOPC','upper') -> 'upper_DOPC_'
         ('mixed','lower') -> 'lower_'
    """
    lf_pfx = '' if lf == 'both'  else f'{lf}_'
    sp_pfx = '' if sp == 'mixed' else f'{sp}_'
    return f'{lf_pfx}{sp_pfx}'


# Bins below this effective number of contributing paths are masked out of
# the plots. n_eff counts *paths*, not frames: with the skewed infretis
# path weights a bin can carry a large raw weight and still rest on one or
# two paths, which is not something to draw a line through.
MIN_NEFF = 2.0

# Companion columns _pool_weighted_means emits next to every observable.
# Anything globbing aggregate columns by name must exclude these.
_STAT_SUFFIXES = ('_wtot', '_neff', '_err')


def _masked_series(agg, col, min_neff=MIN_NEFF):
    """
    Return (values, err) for `col`, with bins whose effective path count
    falls below `min_neff` blanked to NaN. `err` is None when the
    aggregate predates the jackknife columns.
    """
    if col not in agg.columns:
        return None, None
    vals = agg[col].values.astype(float).copy()
    err  = (agg[f'{col}_err'].values.astype(float)
            if f'{col}_err' in agg.columns else None)

    neff_col = f'{col}_neff'
    if neff_col in agg.columns and min_neff and min_neff > 0:
        thin = agg[neff_col].values < min_neff
        vals[thin] = np.nan
        if err is not None:
            err = err.copy()
            err[thin] = np.nan
    return vals, err


def _plot_series(ax, x, agg, col, min_neff=MIN_NEFF, band=True, **kwargs):
    """
    Plot one observable with its block-jackknife uncertainty band.

    The band is the thing that makes a between-simulation difference
    readable: two curves that separate by less than the sum of their bands
    are not separated at all.
    """
    vals, err = _masked_series(agg, col, min_neff)
    if vals is None:
        return None
    line, = ax.plot(x, vals, **kwargs)
    if band and err is not None and np.any(np.isfinite(err)):
        ax.fill_between(x, vals - err, vals + err,
                        color=line.get_color(),
                        alpha=0.18 * kwargs.get('alpha', 1.0),
                        lw=0, zorder=line.get_zorder() - 0.1)
    return line


# ═════════════════════════════════════════════════════════════════════════════
# 1-D profile plot helpers  (leaflet-aware)
# ═════════════════════════════════════════════════════════════════════════════

def plot_p2_profile(p2_agg, chain='sn2',
                    lipid_species=('mixed', 'DOPC', 'POPC'),
                    leaflets=('both', 'upper', 'lower'),
                    min_neff=MIN_NEFF, output_path=None):
    """
    Plot P₂ order-parameter profiles.

    Parameters
    ----------
    leaflets : tuple of str
        Which leaflets to overlay.  'both' = pooled (original behaviour).
        Showing all three gives six lines per chain (3 species × 2 leaflets
        plus the pooled curve).
    """
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap    = cm.viridis

    # Per-bond curves for mixed / both only (keep plot readable).
    # _STAT_SUFFIXES must be excluded or the jackknife/weight companion
    # columns get drawn as if they were extra bonds.
    bond_cols = sorted(
        [c for c in p2_agg.columns
         if c.startswith(f'{chain}_p2_bond_') and not c.startswith(('upper_', 'lower_'))
         and not any(c.startswith(f'{sp}_') for sp in ('DOPC', 'POPC'))
         and not c.endswith(_STAT_SUFFIXES)],
        key=lambda x: int(x.split('_bond_')[1].split('_')[0])
    )
    n_bonds = len(bond_cols)
    for bi, col in enumerate(bond_cols):
        # Per-bond curves are context, not the message — no band on these.
        _plot_series(ax, p2_agg['slab_center'], p2_agg, col,
                     min_neff=min_neff, band=False,
                     color=cmap(bi / max(n_bonds - 1, 1)),
                     lw=0.8, alpha=0.40,
                     label=f'bond {bi}' if bi == 0 else '_')

    # Chain-averaged lines per species × leaflet
    for sp in lipid_species:
        s_label, color, s_ls = _SPECIES_STYLE.get(sp, (sp, 'grey', '-'))
        for lf in leaflets:
            lf_label, alpha, _ = _LEAFLET_STYLE[lf]
            pfx = _col_prefix(sp, lf)
            col = f'{pfx}{chain}_p2_chain'
            if col not in p2_agg.columns:
                continue
            # Distinguish leaflets by line width; 'both' is bold
            lw = 2.5 if lf == 'both' else 1.4
            ls = s_ls if lf == 'both' else ('--' if lf == 'upper' else ':')
            label = f'{s_label} {lf_label}'
            _plot_series(ax, p2_agg['slab_center'], p2_agg, col,
                         min_neff=min_neff,
                         color=color, lw=lw, ls=ls, alpha=alpha, label=label)

    ax.axhline(0,    color='grey', lw=0.8, ls=':')
    ax.axhline(-0.5, color='grey', lw=0.8, ls=':')
    ax.set_xlabel('z-displacement (Å)', fontsize=13)
    ax.set_ylabel('P₂ order parameter', fontsize=13)
    ax.set_ylim(-0.6, 1.05)
    ax.legend(fontsize=7, ncol=3, frameon=False)
    ax.set_title(f'P₂ order parameter — chain {chain}', fontsize=13)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    return fig


def plot_tilt_profile(tilt_agg,
                      lipid_species=('mixed', 'DOPC', 'POPC'),
                      leaflets=('both', 'upper', 'lower'),
                      min_neff=MIN_NEFF, output_path=None):
    """Plot tilt θ and azimuthal φ profiles per species × leaflet."""

    # Detect chains from the 'both/mixed' columns only — excludes leaflet-
    # and species-prefixed columns as well as 'near{N}_...' near-permeant
    # columns, which also end in '_theta' but are not real chain names.
    chains = sorted({c.split('_theta')[0]
                     for c in tilt_agg.columns
                     if c.endswith('_theta')
                     and not any(c.startswith(p) for p in ('upper_', 'lower_'))
                     and not any(c.startswith(f'{sp}_') for sp in ('DOPC', 'POPC'))
                     and not c.startswith('near')})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, angle, ylabel in zip(
        axes, ['theta', 'phi'], ['Mean tilt θ (°)', 'Azimuthal φ (°)']
    ):
        for chain in chains:
            for sp in lipid_species:
                s_label, color, s_ls = _SPECIES_STYLE.get(sp, (sp, 'grey', '-'))
                for lf in leaflets:
                    lf_label, alpha, _ = _LEAFLET_STYLE[lf]
                    pfx = _col_prefix(sp, lf)
                    col = f'{pfx}{chain}_{angle}'
                    if col not in tilt_agg.columns:
                        continue
                    lw = 2.0 if lf == 'both' else 1.2
                    ls = s_ls if lf == 'both' else ('--' if lf == 'upper' else ':')
                    _plot_series(ax, tilt_agg['slab_center'], tilt_agg, col,
                                 min_neff=min_neff,
                                 color=color, lw=lw, ls=ls, alpha=alpha,
                                 label=f'{s_label} {chain} {lf_label}')
        ax.set_xlabel('z-displacement (Å)', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.legend(fontsize=7, frameon=False, ncol=2)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    return fig


def plot_thickness_profile(thick_agg,
                           lipid_species=('mixed', 'DOPC', 'POPC'),
                           leaflets=('both', 'upper', 'lower'),
                           min_neff=MIN_NEFF, output_path=None):
    """
    Plot local membrane thickness vs permeant z-displacement per species × leaflet.

    For the 'upper' and 'lower' leaflets the thickness still represents the
    P–P distance between the two leaflets, but now computed from the
    perspective of each individual leaflet's P-atoms.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    for sp in lipid_species:
        s_label, color, s_ls = _SPECIES_STYLE.get(sp, (sp, 'grey', '-'))
        for lf in leaflets:
            lf_label, alpha, _ = _LEAFLET_STYLE[lf]
            pfx = _col_prefix(sp, lf)
            col = f'{pfx}thickness'
            if col not in thick_agg.columns:
                continue
            lw = 2.0 if lf == 'both' else 1.2
            ls = s_ls if lf == 'both' else ('--' if lf == 'upper' else ':')
            _plot_series(ax, thick_agg['slab_center'], thick_agg, col,
                         min_neff=min_neff,
                         color=color, lw=lw, ls=ls, alpha=alpha,
                         label=f'{s_label} {lf_label}')
    ax.set_xlabel('z-displacement (Å)', fontsize=13)
    ax.set_ylabel('Local thickness (Å)', fontsize=13)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    return fig


def plot_deformation_profile(deform_agg, min_neff=MIN_NEFF, output_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, label, color in (
        ('def_upper', 'Upper leaflet', '#1f77b4'),
        ('def_lower', 'Lower leaflet', '#ff7f0e'),
    ):
        _plot_series(ax, deform_agg['r_center'], deform_agg, col,
                     min_neff=min_neff, lw=2, label=label, color=color)
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    ax.set_xlabel('Lateral distance from permeant (Å)', fontsize=13)
    ax.set_ylabel('Mean z-deformation (Å)', fontsize=13)
    ax.legend(frameon=False)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    return fig


def plot_water_defect(deform_agg, water_z_agg, min_neff=MIN_NEFF,
                      output_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    if 'water_radial' in deform_agg.columns:
        width = (deform_agg['r_center'].iloc[1]
                 - deform_agg['r_center'].iloc[0]) * 0.9
        ax.bar(deform_agg['r_center'], deform_agg['water_radial'],
               width=width, color='steelblue', alpha=0.8)
    ax.set_xlabel('Lateral distance from permeant (Å)', fontsize=13)
    ax.set_ylabel('Water density in core (atoms Å⁻²)', fontsize=13)
    ax.set_title('Radial water density', fontsize=13)

    ax = axes[1]
    _plot_series(ax, water_z_agg['z_center'], water_z_agg, 'water_z',
                 min_neff=min_neff, color='steelblue', lw=2)
    ax.axvline(0, color='grey', lw=0.8, ls='--')
    ax.set_xlabel('z − z$_{midplane}$ (Å)', fontsize=13)
    ax.set_ylabel('Water density (atoms Å⁻³)', fontsize=13)
    ax.set_title('Water z-density profile', fontsize=13)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    return fig


def plot_curvature_profile(thick_agg, deform_agg,
                           lipid_species=('mixed', 'DOPC', 'POPC'),
                           min_neff=MIN_NEFF, output_path=None):
    """
    Plot mean |H| per species vs permeant position, and radial H.

    Curvature has no separate "leaflet-filtered population" axis: each
    species' P atoms are fit as one continuous upper surface and one
    continuous lower surface, so 'H_upper' and 'H_lower' already ARE the
    full leaflet split. (Earlier versions also looped over a `leaflets`
    filter here, which produced meaningless labels like "lower leaflet"
    attached to the upper-surface H value, and all-NaN columns for the
    mismatched combinations.)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    for sp in lipid_species:
        s_label, color, s_ls = _SPECIES_STYLE.get(sp, (sp, 'grey', '-'))
        sp_pfx = '' if sp == 'mixed' else f'{sp}_'
        col_u = f'{sp_pfx}H_upper'
        col_l = f'{sp_pfx}H_lower'
        _plot_series(ax, thick_agg['slab_center'], thick_agg, col_u,
                     min_neff=min_neff, color=color, lw=2.0, ls=s_ls,
                     label=f'{s_label} upper leaflet')
        _plot_series(ax, thick_agg['slab_center'], thick_agg, col_l,
                     min_neff=min_neff, color=color, lw=1.4, ls=s_ls,
                     alpha=0.6, label=f'{s_label} lower leaflet')

    ax.set_xlabel('z-displacement (Å)', fontsize=13)
    ax.set_ylabel('Mean |H| (Å⁻¹)', fontsize=13)
    ax.set_title('Mean curvature vs permeant position', fontsize=13)
    ax.legend(fontsize=7, frameon=False, ncol=2)

    ax = axes[1]
    _plot_series(ax, deform_agg['r_center'], deform_agg, 'H_radial',
                 min_neff=min_neff, lw=2, color='#2ca02c',
                 label='Upper leaflet (mixed)')
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    ax.set_xlabel('Lateral distance from permeant (Å)', fontsize=13)
    ax.set_ylabel('Mean curvature H (Å⁻¹)', fontsize=13)
    ax.set_title('Radial curvature around permeant', fontsize=13)
    ax.legend(frameon=False)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Near-permeant vs. bulk comparison plots
# ═════════════════════════════════════════════════════════════════════════════

def _plot_near_vs_bulk(agg_df, bulk_col, near_n, near_col_fn,
                       xlabel, ylabel, title,
                       bulk_label='Bulk (all lipids)',
                       min_neff=MIN_NEFF, output_path=None):
    """
    Generic near-permeant-vs-bulk comparison: one bold bulk curve plus one
    line per requested N, all vs. permeant z-displacement (insertion depth).
    A gap between the bulk curve and the near-N curves at a given depth is
    the signature of the permeant locally perturbing its nearest neighbours.

    The near-N curves rest on 5-10 lipids per frame, so they are far
    noisier than the bulk curve — reading a gap off them without the
    jackknife bands drawn is how you talk yourself into a perturbation
    that is not there.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _plot_series(ax, agg_df['slab_center'], agg_df, bulk_col,
                 min_neff=min_neff, color='#333333', lw=2.5, label=bulk_label)
    cmap = plt.get_cmap('viridis')
    for i, N in enumerate(near_n):
        col = near_col_fn(N)
        _plot_series(ax, agg_df['slab_center'], agg_df, col,
                     min_neff=min_neff,
                     color=cmap(i / max(len(near_n) - 1, 1)),
                     lw=2.0, ls='--', label=f'Nearest {N} lipids')
    ax.axhline(0, color='grey', lw=0.8, ls=':')
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    return fig


def plot_near_vs_bulk_p2(p2_agg, chain='sn2', near_n=(5, 10),
                         min_neff=MIN_NEFF, output_path=None):
    return _plot_near_vs_bulk(
        p2_agg, f'{chain}_p2_chain', near_n,
        lambda N: f'near{N}_{chain}_p2_chain',
        'z-displacement (Å)', 'P₂ order parameter',
        f'Near-permeant vs. bulk P₂ order — chain {chain}',
        min_neff=min_neff, output_path=output_path,
    )


def plot_near_vs_bulk_tilt(tilt_agg, chain='sn2', near_n=(5, 10),
                           min_neff=MIN_NEFF, output_path=None):
    return _plot_near_vs_bulk(
        tilt_agg, f'{chain}_theta', near_n,
        lambda N: f'near{N}_{chain}_theta',
        'z-displacement (Å)', 'Mean tilt θ (°)',
        f'Near-permeant vs. bulk tilt angle — chain {chain}',
        min_neff=min_neff, output_path=output_path,
    )


def plot_near_vs_bulk_thickness(thick_agg, near_n=(5, 10),
                                min_neff=MIN_NEFF, output_path=None):
    return _plot_near_vs_bulk(
        thick_agg, 'thickness', near_n,
        lambda N: f'near{N}_thickness',
        'z-displacement (Å)', 'Local thickness (Å)',
        'Near-permeant vs. bulk membrane thickness',
        min_neff=min_neff, output_path=output_path,
    )


def plot_near_vs_bulk_curvature(thick_agg, near_n=(5, 10),
                                min_neff=MIN_NEFF, output_path=None):
    # Bulk reference is the ('mixed','both') UPPER-leaflet RBF fit — the
    # near-N curve pools whichever leaflet(s) the near lipids actually
    # belong to, so this comparison is only approximate near the bilayer
    # midplane where leaflet membership is ambiguous.
    return _plot_near_vs_bulk(
        thick_agg, 'H_upper', near_n,
        lambda N: f'near{N}_H',
        'z-displacement (Å)', 'Mean |H| (Å⁻¹)',
        'Near-permeant vs. bulk curvature',
        bulk_label='Bulk (upper leaflet, mixed)',
        min_neff=min_neff, output_path=output_path,
    )


def plot_diffusion_profile(diff_agg, min_neff=MIN_NEFF, output_path=None):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    colors = {'D_einstein': '#1f77b4', 'D_hummer': '#d62728'}
    labels = {'D_einstein': 'Einstein (short-time MSD)',
              'D_hummer'  : 'Hummer (pos. autocorr.)'}
    for col, color in colors.items():
        _plot_series(ax1, diff_agg['slab_center'], diff_agg, col,
                     min_neff=min_neff, lw=2, color=color, label=labels[col])
    ax1.set_ylabel('D(z)  (Å² ps⁻¹)', fontsize=13)
    ax1.set_yscale('log')
    ax1.legend(frameon=False, fontsize=11)
    ax1.set_title('Local diffusion coefficient along bilayer normal', fontsize=13)
    if 'n_sojourns' in diff_agg.columns:
        width = (diff_agg['slab_center'].iloc[1]
                 - diff_agg['slab_center'].iloc[0]) * 0.85
        ax2.bar(diff_agg['slab_center'], diff_agg['n_sojourns'],
                width=width, color='steelblue', alpha=0.7)
    ax2.set_xlabel('z-displacement (Å)', fontsize=13)
    ax2.set_ylabel('# sojourns', fontsize=12)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    return fig


def plot_diffusion_xy_profile(diff_xy_agg, min_neff=MIN_NEFF, output_path=None):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    series = {
        'D_xy_ein'      : ('#1f77b4', 'Einstein 2-D MSD'),
        'D_xy_hummer_x' : ('#d62728', 'Hummer ACF  D_x'),
        'D_xy_hummer_y' : ('#2ca02c', 'Hummer ACF  D_y'),
    }
    for col, (color, label) in series.items():
        _plot_series(ax1, diff_xy_agg['slab_center'], diff_xy_agg, col,
                     min_neff=min_neff, lw=2, color=color, label=label)
    ax1.set_ylabel('D_xy(z)  (Å² ps⁻¹)', fontsize=13)
    ax1.set_yscale('log')
    ax1.legend(frameon=False, fontsize=11)
    ax1.set_title('Lateral diffusion coefficient along bilayer normal', fontsize=13)
    if 'n_sojourns' in diff_xy_agg.columns:
        width = (diff_xy_agg['slab_center'].iloc[1]
                 - diff_xy_agg['slab_center'].iloc[0]) * 0.85
        ax2.bar(diff_xy_agg['slab_center'], diff_xy_agg['n_sojourns'],
                width=width, color='steelblue', alpha=0.7)
    ax2.set_xlabel('z-displacement (Å)', fontsize=13)
    ax2.set_ylabel('# sojourns', fontsize=12)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Spatial map plotting helpers  (leaflet-aware)
# ═════════════════════════════════════════════════════════════════════════════

def _map_panel(ax, data_2d, x_coords, y_coords, title, cmap,
               vmin=None, vmax=None, permeant_xy=(0.0, 0.0)):
    # Maps are stamped in a permeant-centered coordinate frame (see the
    # SPATIAL block in calculate_selected_slab_profiles), so the permeant
    # sits at the origin by construction — the marker defaults there rather
    # than requiring a per-path position that no longer applies once maps
    # are pooled across many independent trajectories.
    im = ax.pcolormesh(
        x_coords, y_coords, data_2d.T,
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading='auto', rasterized=True,
    )
    ax.set_aspect('equal')
    ax.set_xlabel('x − x$_{permeant}$ (Å)', fontsize=11)
    ax.set_ylabel('y − y$_{permeant}$ (Å)', fontsize=11)
    ax.set_title(title, fontsize=12)
    if permeant_xy is not None:
        ax.plot(permeant_xy[0], permeant_xy[1],
                'o', ms=8, mfc='white', mec='black', mew=1.5, zorder=5)
    return im


def plot_spatial_map(
    data_2d,
    x_coords,
    y_coords,
    title       = '',
    cmap        = 'RdBu_r',
    vmin        = None,
    vmax        = None,
    permeant_xy = (0.0, 0.0),
    output_path = None,
):
    if vmin is None and vmax is None:
        finite = data_2d[np.isfinite(data_2d)]
        if len(finite):
            absmax = np.percentile(np.abs(finite), 98)
            vmin, vmax = -absmax, absmax

    fig, ax = plt.subplots(figsize=(7, 6))
    im = _map_panel(ax, data_2d, x_coords, y_coords, title, cmap,
                    vmin=vmin, vmax=vmax, permeant_xy=permeant_xy)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    return fig


def _depth_tag(maps, depth, depth_weights=None):
    """
    Filename/title fragment identifying a depth selection and weighting.

    The weighting has to appear in the filename: an occupancy-weighted and
    a depth-uniform map of the same observable are different quantities,
    and without the tag the second run silently overwrites the first with
    a file that looks identical but no longer means the same thing.
    """
    if depth is None:
        d_sfx, d_title = '', ''
    elif isinstance(depth, tuple):
        d_sfx   = f'_z{depth[0]:g}to{depth[1]:g}'
        d_title = f'  [z {depth[0]:g}…{depth[1]:g} Å]'
    else:
        lo = maps.depth_edges[int(depth)]
        hi = maps.depth_edges[int(depth) + 1]
        d_sfx   = f'_zbin{int(depth)}'
        d_title = f'  [z {lo:.1f}…{hi:.1f} Å]'

    if depth_weights is not None:
        tag = depth_weights if isinstance(depth_weights, str) else 'wdepth'
        d_sfx   += f'_{tag}'
        d_title += f'  [{tag} depth weighting]'
    return d_sfx, d_title


def plot_all_spatial_maps(
    npz_path,
    chain_bonds    = CHAIN_BONDS,
    lipid_species  = ('mixed', 'DOPC', 'POPC'),
    leaflets       = ('both', 'upper', 'lower'),
    permeant_xy    = (0.0, 0.0),
    output_dir     = None,
    plot_bonds     = False,
    depth          = None,
    depth_weights  = None,
):
    """
    Load a finalised spatial map .npz and produce one PNG per observable ×
    species × leaflet.

    File naming: <obs>_<chain>[_<species>][_<leaflet>][_<depth>]_map.png
    e.g.  p2_sn2_chain_DOPC_upper_map.png
          theta_sn1_lower_map.png
          thickness_map.png   (mixed / both, all depths — original name)

    depth / depth_weights select and weight the permeant-depth bins; see
    SpatialMaps. The defaults (all bins, occupancy weighting) reproduce
    the original depth-integrated maps exactly.
    """
    data, x_coords, y_coords = load_spatial_maps(
        npz_path, depth=depth, depth_weights=depth_weights
    )
    out_dir = Path(output_dir) if output_dir else Path(npz_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    d_sfx, d_title = _depth_tag(data, depth, depth_weights)
    figs = {}

    def _save(fig, name):
        path = out_dir / name
        fig.savefig(str(path), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> {path.name}")
        return fig

    def _sp_lf_key(base_key, sp, lf):
        sp_sfx = '' if sp == 'mixed' else f'_{sp}'
        lf_sfx = '' if lf == 'both'  else f'_{lf}'
        return f'{base_key}{sp_sfx}{lf_sfx}'

    def _sp_lf_suffix(sp, lf):
        sp_sfx = '' if sp == 'mixed' else f'_{sp}'
        lf_sfx = '' if lf == 'both'  else f'_{lf}'
        return f'{sp_sfx}{lf_sfx}'

    for sp in lipid_species:
        for lf in leaflets:
            # d_sfx / d_title fold the depth selection into every filename
            # and title, so per-depth runs never overwrite each other.
            sfx      = _sp_lf_suffix(sp, lf) + d_sfx
            sp_label = '' if sp == 'mixed' else f' ({sp})'
            lf_label = '' if lf == 'both'  else f' {lf} leaflet'
            combo    = sp_label + lf_label + d_title

            # Chain-averaged P₂ maps
            for chain in chain_bonds:
                key = _sp_lf_key(f'p2_{chain}_chain', sp, lf)
                if key not in data:
                    continue
                cmap_name, label = SPATIAL_MAP_META.get(
                    f'p2_{chain}_chain', ('RdBu_r', f'P₂ {chain} chain')
                )
                fig = plot_spatial_map(
                    data[key], x_coords, y_coords,
                    title=label + combo, cmap=cmap_name,
                    permeant_xy=permeant_xy,
                )
                figs[key] = _save(fig, f'p2_{chain}_chain{sfx}_map.png')

            # Per-bond P₂ maps (optional)
            if plot_bonds:
                for chain, bonds in chain_bonds.items():
                    for bi in range(len(bonds)):
                        key = _sp_lf_key(f'p2_{chain}_bond_{bi}', sp, lf)
                        if key not in data:
                            continue
                        fig = plot_spatial_map(
                            data[key], x_coords, y_coords,
                            title=f'P₂ {chain} bond {bi}{combo}',
                            cmap='RdBu_r', permeant_xy=permeant_xy,
                        )
                        figs[key] = _save(fig, f'p2_{chain}_bond_{bi}{sfx}_map.png')

            # Tilt angle maps
            for chain in chain_bonds:
                for angle in ('theta', 'phi'):
                    key      = _sp_lf_key(f'{angle}_{chain}', sp, lf)
                    meta_key = f'{angle}_{chain}'
                    if key not in data:
                        continue
                    cmap_name, label = SPATIAL_MAP_META.get(
                        meta_key, ('plasma', f'{angle} {chain}')
                    )
                    finite = data[key][np.isfinite(data[key])]
                    vmin_t = float(np.nanpercentile(finite,  2)) if len(finite) else None
                    vmax_t = float(np.nanpercentile(finite, 98)) if len(finite) else None
                    fig = plot_spatial_map(
                        data[key], x_coords, y_coords,
                        title=label + combo, cmap=cmap_name,
                        vmin=vmin_t, vmax=vmax_t,
                        permeant_xy=permeant_xy,
                    )
                    figs[key] = _save(fig, f'{angle}_{chain}{sfx}_map.png')

            # Thickness map
            key = _sp_lf_key('thickness', sp, lf)
            if key in data:
                cmap_name, label = SPATIAL_MAP_META['thickness']
                finite = data[key][np.isfinite(data[key])]
                vmin_t = float(np.nanpercentile(finite,  2)) if len(finite) else None
                vmax_t = float(np.nanpercentile(finite, 98)) if len(finite) else None
                fig = plot_spatial_map(
                    data[key], x_coords, y_coords,
                    title=label + combo, cmap=cmap_name,
                    vmin=vmin_t, vmax=vmax_t, permeant_xy=permeant_xy,
                )
                figs[key] = _save(fig, f'thickness{sfx}_map.png')

            # Curvature |H| map
            key = _sp_lf_key('curvature_H', sp, lf)
            if key in data:
                cmap_name, label = SPATIAL_MAP_META['curvature_H']
                finite = data[key][np.isfinite(data[key])]
                vmax_c = float(np.nanpercentile(finite, 98)) if len(finite) else None
                fig = plot_spatial_map(
                    data[key], x_coords, y_coords,
                    title=label + combo, cmap=cmap_name,
                    vmin=0.0, vmax=vmax_c, permeant_xy=permeant_xy,
                )
                figs[key] = _save(fig, f'curvature_H{sfx}_map.png')

    return figs


def plot_spatial_overview(
    npz_path,
    chain          = 'sn2',
    lipid_species  = ('mixed', 'DOPC', 'POPC'),
    leaflets       = ('both', 'upper', 'lower'),
    permeant_xy    = (0.0, 0.0),
    output_path    = None,
    depth          = None,
    depth_weights  = None,
):
    """
    Multi-panel overview: rows = species × leaflet combinations,
    columns = observables (P₂, θ, φ, thickness, curvature, coverage).

    depth / depth_weights select and weight the permeant-depth bins
    (see SpatialMaps); the defaults reproduce the depth-integrated maps.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    data, x_coords, y_coords = load_spatial_maps(
        npz_path, depth=depth, depth_weights=depth_weights
    )
    _, d_title = _depth_tag(data, depth, depth_weights)

    PANEL_DEFS = [
        (f'p2_{chain}_chain',  SPATIAL_MAP_META.get(f'p2_{chain}_chain',
         ('RdBu_r',  f'P₂ {chain} chain'))),
        (f'theta_{chain}',     SPATIAL_MAP_META.get(f'theta_{chain}',
         ('plasma',  f'Tilt θ {chain} (°)'))),
        (f'phi_{chain}',       SPATIAL_MAP_META.get(f'phi_{chain}',
         ('twilight', f'Azimuthal φ {chain} (°)'))),
        ('thickness',          SPATIAL_MAP_META.get('thickness',
         ('coolwarm', 'Thickness (Å)'))),
        ('curvature_H',        SPATIAL_MAP_META.get('curvature_H',
         ('PiYG',    'Mean curvature |H| (Å⁻¹)'))),
    ]

    n_cols = 6  # 5 observables + 1 coverage
    rows   = [(sp, lf) for sp in lipid_species for lf in leaflets]
    n_rows = len(rows)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.0 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    for row_idx, (sp, lf) in enumerate(rows):
        sp_sfx   = '' if sp == 'mixed' else f'_{sp}'
        lf_sfx   = '' if lf == 'both'  else f'_{lf}'
        sfx      = f'{sp_sfx}{lf_sfx}'
        sp_label = 'All lipids' if sp == 'mixed' else sp
        lf_label = 'both leaflets' if lf == 'both' else f'{lf} leaflet'

        for col_idx, (base_key, (cmap_name, label)) in enumerate(PANEL_DEFS):
            ax  = axes[row_idx][col_idx]
            key = f'{base_key}{sfx}'
            if key not in data:
                ax.set_visible(False)
                continue

            arr    = data[key]
            finite = arr[np.isfinite(arr)]
            if base_key.startswith('p2'):
                absmax = np.percentile(np.abs(finite), 98) if len(finite) else 1
                vmin_p, vmax_p = -absmax, absmax
            elif base_key == 'curvature_H':
                vmin_p, vmax_p = 0.0, (float(np.percentile(finite, 98))
                                        if len(finite) else None)
            else:
                vmin_p = float(np.percentile(finite,  2)) if len(finite) else None
                vmax_p = float(np.percentile(finite, 98)) if len(finite) else None

            title = f'{label}\n({sp_label}, {lf_label})'
            im    = _map_panel(ax, arr, x_coords, y_coords, title,
                               cmap_name, vmin=vmin_p, vmax=vmax_p,
                               permeant_xy=permeant_xy)
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', size='4%', pad=0.05)
            plt.colorbar(im, cax=cax)

        # Coverage panel (col 5). This is now the real accumulated weight
        # per grid cell, normalised to its own maximum, rather than a
        # binary "is this cell not NaN" — a cell can be non-NaN and still
        # rest on a single stamp, which is exactly the case you need to
        # see before believing a feature in the panels to its left.
        ax_cov  = axes[row_idx][5]
        key_cov = f'p2_{chain}_chain{sfx}'
        if key_cov in data:
            cov_w    = data.coverage(key_cov)
            cov_max  = np.nanmax(cov_w) if np.any(cov_w > 0) else 1.0
            coverage = cov_w / cov_max
            im_cov   = _map_panel(ax_cov, coverage, x_coords, y_coords,
                                  f'Relative coverage\n({sp_label}, {lf_label})',
                                  'Greens', vmin=0, vmax=1,
                                  permeant_xy=permeant_xy)
            div = make_axes_locatable(ax_cov)
            cax = div.append_axes('right', size='4%', pad=0.05)
            plt.colorbar(im_cov, cax=cax)
        else:
            ax_cov.set_visible(False)

    fig.suptitle(f'Spatial membrane maps — chain {chain}{d_title}',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

def membrane_spatial(
    # ── Input data ────────────────────────────────────────────
    weights: Annotated[str, typer.Option("-weights", "--weights", help="", rich_help_panel=panels.INPUT)] = 'path_weights.txt',
    data: Annotated[str, typer.Option("-data", "--data", help="Path to infretis_data.txt.", rich_help_panel=panels.INPUT)] = ...,
    permeant: Annotated[str, typer.Option("-permeant", "--permeant", help="", rich_help_panel=panels.INPUT)] = 'ORP',
    leaflets_opt: Annotated[str, typer.Option("-leaflets", "--leaflets", help="Comma-separated leaflets to include in plots, e.g. 'both,upper,lower'.", rich_help_panel=panels.INPUT)] = "both,upper,lower",

    # ── Dataset construction ──────────────────────────────────
start: Annotated[int, typer.Option("-start", "--start", help="", rich_help_panel=panels.DATASET)] = ...,
    end: Annotated[Optional[int], typer.Option("-end", "--end", help="", rich_help_panel=panels.DATASET)] = None,
    max_lag_ps: Annotated[float, typer.Option("-max-lag-ps", "--max-lag-ps", help="", rich_help_panel=panels.DATASET)] = 500.0,
    dt: Annotated[Optional[float], typer.Option("-dt", "--dt", help="", rich_help_panel=panels.DATASET)] = None,

    # ── CV corrections: representation ────────────────────────
    slab_width: Annotated[float, typer.Option("-slab-width", "--slab-width", help="", rich_help_panel=panels.REPR)] = 1.0,
    bilayer_normal: Annotated[str, typer.Option("-bilayer-normal", "--bilayer-normal", help="", rich_help_panel=panels.REPR)] = 'z',
    slab_range_opt: Annotated[Optional[str], typer.Option("-slab-range", "--slab-range", help="Fixed slab range in Angstrom as 'min,max', e.g. '-40,40'.", rich_help_panel=panels.REPR)] = None,
    r_max: Annotated[float, typer.Option("-r-max", "--r-max", help="", rich_help_panel=panels.REPR)] = 30.0,
    n_radial_bins: Annotated[int, typer.Option("-n-radial-bins", "--n-radial-bins", help="", rich_help_panel=panels.REPR)] = 15,
    n_z_bins: Annotated[int, typer.Option("-n-z-bins", "--n-z-bins", help="Number of bins for the water z-density profile, spanning --water-z-range.", rich_help_panel=panels.REPR)] = 60,
    water_z_range_opt: Annotated[Optional[str], typer.Option("-water-z-range", "--water-z-range", help="Window in Angstrom as 'min,max' for the water z-density profile, measured relative to the bilayer midplane. Fixed rather than box-derived so the profile is poolable across paths and comparable between simulations. Must not exceed the box height; ±40 Å suits a ~100 Å box.", rich_help_panel=panels.REPR)] = None,
    curvature_smoothing: Annotated[float, typer.Option("-curvature-smoothing", "--curvature-smoothing", help="", rich_help_panel=panels.REPR)] = 0.0,
    min_slab_points: Annotated[int, typer.Option("-min-slab-points", "--min-slab-points", help="", rich_help_panel=panels.REPR)] = 5,
    grid_spacing: Annotated[float, typer.Option("-grid-spacing", "--grid-spacing", help="", rich_help_panel=panels.REPR)] = 5.0,
    stamp_radius: Annotated[Optional[float], typer.Option("-stamp-radius", "--stamp-radius", help="", rich_help_panel=panels.REPR)] = None,
    near_n_opt: Annotated[str, typer.Option("-near-n", "--near-n", help="Comma-separated N values for near-permeant local membrane metrics (nearest-N-lipid average vs bulk), e.g. '5,10'. Pass an empty string to disable.", rich_help_panel=panels.REPR)] = "5,10",
    spatial_extent: Annotated[Optional[float], typer.Option("-spatial-extent", "--spatial-extent", help="Half-width (Å) of the permeant-centered 2-D spatial map window. Defaults to --r-max.", rich_help_panel=panels.REPR)] = None,
    n_depth_bins: Annotated[int, typer.Option("-n-depth-bins", "--n-depth-bins", help="Number of permeant-depth bins for the 2-D spatial maps, spanning --slab-range. Maps are stored per depth bin so they can be collapsed or compared depth-matched; 1 restores the old fully depth-integrated behaviour.", rich_help_panel=panels.REPR)] = 10,
    spatial_depth_weighting: Annotated[str, typer.Option("-spatial-depth-weighting", "--spatial-depth-weighting", help="How to combine depth bins into the plotted maps. 'occupancy' weights each bin by the time the permeant spent there (the original, depth-integrated map). 'uniform' gives every depth equal weight — use it when comparing simulations whose depth sampling differs.", rich_help_panel=panels.REPR)] = 'occupancy',
    spatial_depth_plots: Annotated[bool, typer.Option("-spatial-depth-plots", "--spatial-depth-plots", help="Also emit one spatial overview figure per depth bin.", rich_help_panel=panels.REPR)] = False,

    # ── Model and training ────────────────────────────────────
    workers: Annotated[int, typer.Option("-workers", "--workers", help="", rich_help_panel=panels.MODEL)] = 32,
    n_blocks: Annotated[int, typer.Option("-n-blocks", "--n-blocks", help="Number of contiguous MC-time blocks for the block-jackknife error bars on the 1-D profiles. Blocks must be longer than the path decorrelation time; fewer, longer blocks give a more conservative error.", rich_help_panel=panels.MODEL)] = 20,
    min_neff: Annotated[float, typer.Option("-min-neff", "--min-neff", help="Blank out profile bins supported by fewer than this effective number of paths (Kish n_eff). 0 disables masking.", rich_help_panel=panels.MODEL)] = 'MIN_NEFF',

    # ── Output ────────────────────────────────────────────────
    overwrite: Annotated[bool, typer.Option("-overwrite", "--overwrite", help="", rich_help_panel=panels.OUTPUT)] = False,
    plot: Annotated[bool, typer.Option("-plot", "--plot", help="", rich_help_panel=panels.OUTPUT)] = False,
    plot_bonds: Annotated[bool, typer.Option("-plot-bonds", "--plot-bonds", help="", rich_help_panel=panels.OUTPUT)] = False,
):
    """Spatial membrane analysis: radial/z maps, curvature and local structure around the permeant."""
    slab_range_opt = _split_values(slab_range_opt, "-slab-range", float, 2)
    water_z_range_opt = _split_values(water_z_range_opt, "-water-z-range", float, 2)
    # an empty -near-n disables the near-permeant metrics, as the original
    # `--near-n` with no values did
    near_n_opt = _split_values(near_n_opt, "-near-n", int, None) if near_n_opt.strip() else []
    leaflets_opt = _split_values(leaflets_opt, "-leaflets", str, None)
    # These were module-level assignments in the original __main__ block, so
    # helper functions below still read them as globals. Declaring them here
    # keeps that working now that the block lives inside a function.
    global N_WORKERS, leaflets, near_n, slab_range, water_z_range

    args = SimpleNamespace(
        start=start,
        end=end,
        overwrite=overwrite,
        workers=workers,
        weights=weights,
        data=data,
        permeant=permeant,
        slab_width=slab_width,
        bilayer_normal=bilayer_normal,
        slab_range=slab_range_opt,
        r_max=r_max,
        n_radial_bins=n_radial_bins,
        n_z_bins=n_z_bins,
        water_z_range=water_z_range_opt,
        curvature_smoothing=curvature_smoothing,
        plot=plot,
        max_lag_ps=max_lag_ps,
        min_slab_points=min_slab_points,
        dt=dt,
        grid_spacing=grid_spacing,
        stamp_radius=stamp_radius,
        near_n=near_n_opt,
        spatial_extent=spatial_extent,
        n_depth_bins=n_depth_bins,
        spatial_depth_weighting=spatial_depth_weighting,
        spatial_depth_plots=spatial_depth_plots,
        n_blocks=n_blocks,
        min_neff=min_neff,
        plot_bonds=plot_bonds,
        leaflets=leaflets_opt,
    )

    import argparse

    parser = argparse.ArgumentParser(
        description='Parallel weighted lipid order-parameter / structural / '
                    'spatial analysis (single-pass merged loop, leaflet-aware).'
    )
    parser.add_argument('-s', '--start',         type=int,   required=True)
    parser.add_argument('-e', '--end',           type=int,   default=None)
    parser.add_argument('--overwrite',           action='store_true', default=False)
    parser.add_argument('-w', '--workers',       type=int,   default=32)
    parser.add_argument('--weights',             type=str,   default='path_weights.txt')
    parser.add_argument('--data',                type=str,   required=True, help='Path to infretis_data.txt.')
    parser.add_argument('--permeant',            type=str,   default='ORP')
    parser.add_argument('--slab-width',          type=float, default=1.0)
    parser.add_argument('--bilayer-normal',      type=str,   default='z')
    parser.add_argument('--slab-range',          type=float, nargs=2, metavar=('Z_MIN', 'Z_MAX'), default=(-25.0, 25.0))
    parser.add_argument('--r-max',               type=float, default=30.0)
    parser.add_argument('--n-radial-bins',       type=int,   default=15)
    parser.add_argument('--n-z-bins',            type=int,   default=60,
                        help='Number of bins for the water z-density '
                             'profile, spanning --water-z-range.')
    parser.add_argument('--water-z-range',       type=float, nargs=2,
                        metavar=('Z_MIN', 'Z_MAX'), default=WATER_Z_RANGE,
                        help='Window (Å) for the water z-density profile, '
                             'measured relative to the bilayer midplane. '
                             'Fixed rather than box-derived so the profile '
                             'is poolable across paths and comparable '
                             'between simulations. Must not exceed the box '
                             'height; ±40 Å suits a ~100 Å box.')
    parser.add_argument('--curvature-smoothing', type=float, default=0.0)
    parser.add_argument('--plot',                action='store_true', default=True)
    parser.add_argument('--max-lag-ps',          type=float, default=500.0)
    parser.add_argument('--min-slab-points',     type=int,   default=5)
    parser.add_argument('--dt',                  type=float, default=None)
    parser.add_argument('--grid-spacing',        type=float, default=5.0)
    parser.add_argument('--stamp-radius',        type=float, default=None)
    parser.add_argument('--near-n',              type=int,   nargs='*',
                        default=[5, 10],
                        help='N values for near-permeant local membrane '
                             'metrics (nearest-N-lipid average vs. bulk). '
                             'Pass --near-n with no values to disable.')
    parser.add_argument('--spatial-extent',      type=float, default=None,
                        help='Half-width (Å) of the permeant-centered 2-D '
                             'spatial map window. Defaults to --r-max.')
    parser.add_argument('--n-depth-bins',        type=int,   default=10,
                        help='Number of permeant-depth bins for the 2-D '
                             'spatial maps, spanning --slab-range. Maps are '
                             'stored per depth bin so they can be collapsed '
                             'or compared depth-matched; 1 restores the old '
                             'fully depth-integrated behaviour.')
    parser.add_argument('--spatial-depth-weighting',
                        choices=['occupancy', 'uniform'], default='occupancy',
                        help="How to combine depth bins into the plotted "
                             "maps. 'occupancy' weights each bin by the time "
                             "the permeant spent there (the original, "
                             "depth-integrated map). 'uniform' gives every "
                             "depth equal weight — use it when comparing "
                             "simulations whose depth sampling differs.")
    parser.add_argument('--spatial-depth-plots', action='store_true', default=False,
                        help='Also emit one spatial overview figure per '
                             'depth bin.')
    parser.add_argument('--n-blocks',            type=int,   default=20,
                        help='Number of contiguous MC-time blocks for the '
                             'block-jackknife error bars on the 1-D '
                             'profiles. Blocks must be longer than the path '
                             'decorrelation time; fewer, longer blocks give '
                             'a more conservative error.')
    parser.add_argument('--min-neff',            type=float, default=MIN_NEFF,
                        help='Blank out profile bins supported by fewer than '
                             'this effective number of paths (Kish n_eff). '
                             '0 disables masking.')
    parser.add_argument('--plot-bonds',          action='store_true', default=False)
    parser.add_argument('--leaflets',            type=str,   nargs='+',
                        default=['both', 'upper', 'lower'],
                        choices=['both', 'upper', 'lower'],
                        help='Which leaflets to include in plots.')
    args = parser.parse_args()

    slab_range = tuple(args.slab_range) if args.slab_range else None
    end        = args.end if args.end is not None else args.start
    leaflets   = tuple(args.leaflets)
    _species   = ('mixed', 'DOPC', 'POPC')
    N_WORKERS = args.workers

    loop_over_paths_parallel(
        path_start          = args.start,
        path_end            = end,
        overwrite           = args.overwrite,
        weights_file        = args.weights,
        data_file           = args.data,
        permeant_resname    = args.permeant,
        lipid_resnames      = ('DOPC', 'POPC'),
        slab_width          = args.slab_width,
        bilayer_normal      = args.bilayer_normal,
        slab_range          = slab_range,
        r_max               = args.r_max,
        n_radial_bins       = args.n_radial_bins,
        n_z_bins            = args.n_z_bins,
        water_z_range       = tuple(args.water_z_range),
        curvature_smoothing = args.curvature_smoothing,
        dt                  = args.dt,
        max_lag_ps          = args.max_lag_ps,
        min_slab_points     = args.min_slab_points,
        grid_spacing        = args.grid_spacing,
        stamp_radius        = args.stamp_radius,
        near_n              = tuple(args.near_n),
        spatial_extent      = args.spatial_extent,
        n_depth_bins        = args.n_depth_bins,
    )

    if args.plot:
        os.makedirs(PLOT_OUT, exist_ok=True)
        print("Aggregating results...")
        weights = load_path_weights(args.weights)

        p2_agg, tilt_agg, thick_agg, deform_agg, water_z_agg, diff_agg, \
            diff_xy_agg = aggregate_all_results(args.start, end, weights,
                                                n_blocks=args.n_blocks)

        for agg, name in (
            (p2_agg,       "p2_aggregate.csv"),
            (tilt_agg,     "tilt_aggregate.csv"),
            (thick_agg,    "thickness_aggregate.csv"),
            (deform_agg,   "deformation_aggregate.csv"),
            (water_z_agg,  "water_z_aggregate.csv"),
            (diff_agg,     "diffusion_aggregate.csv"),
            (diff_xy_agg,  "diffusion_xy_aggregate.csv"),
        ):
            agg.to_csv(os.path.join(PLOT_OUT, name),
                       index=False, float_format='%.6e')
            print(f"  -> {name}")

        min_neff = args.min_neff
        for chain in CHAIN_BONDS:
            plot_p2_profile(
                p2_agg, chain=chain,
                lipid_species=_species, leaflets=leaflets, min_neff=min_neff,
                output_path=os.path.join(PLOT_OUT, f"p2_{chain}.png"))
        plot_tilt_profile(
            tilt_agg, lipid_species=_species, leaflets=leaflets,
            min_neff=min_neff,
            output_path=os.path.join(PLOT_OUT, "tilt_angles.png"))
        plot_thickness_profile(
            thick_agg, lipid_species=_species, leaflets=leaflets,
            min_neff=min_neff,
            output_path=os.path.join(PLOT_OUT, "thickness.png"))
        plot_deformation_profile(
            deform_agg, min_neff=min_neff,
            output_path=os.path.join(PLOT_OUT, "deformation.png"))
        plot_water_defect(
            deform_agg, water_z_agg, min_neff=min_neff,
            output_path=os.path.join(PLOT_OUT, "water_defect.png"))
        plot_curvature_profile(
            thick_agg, deform_agg, lipid_species=_species, min_neff=min_neff,
            output_path=os.path.join(PLOT_OUT, "curvature.png"))

        if args.near_n:
            near_n = tuple(args.near_n)
            for chain in CHAIN_BONDS:
                plot_near_vs_bulk_p2(
                    p2_agg, chain=chain, near_n=near_n, min_neff=min_neff,
                    output_path=os.path.join(PLOT_OUT, f"near_vs_bulk_p2_{chain}.png"))
                plot_near_vs_bulk_tilt(
                    tilt_agg, chain=chain, near_n=near_n, min_neff=min_neff,
                    output_path=os.path.join(PLOT_OUT, f"near_vs_bulk_tilt_{chain}.png"))
            plot_near_vs_bulk_thickness(
                thick_agg, near_n=near_n, min_neff=min_neff,
                output_path=os.path.join(PLOT_OUT, "near_vs_bulk_thickness.png"))
            plot_near_vs_bulk_curvature(
                thick_agg, near_n=near_n, min_neff=min_neff,
                output_path=os.path.join(PLOT_OUT, "near_vs_bulk_curvature.png"))

        plot_diffusion_profile(
            diff_agg, min_neff=min_neff,
            output_path=os.path.join(PLOT_OUT, "diffusion.png"))
        plot_diffusion_xy_profile(
            diff_xy_agg, min_neff=min_neff,
            output_path=os.path.join(PLOT_OUT, "diffusion_xy.png"))

        # Spatial maps
        print("Aggregating spatial maps...")
        try:
            sp_agg, sp_x, sp_y, sp_shape, sp_depth = aggregate_spatial_maps(
                args.start, end, weights
            )
            agg_npz = os.path.join(PLOT_OUT, "spatial_aggregate.npz")
            save_spatial_maps(sp_agg, sp_shape, sp_x, sp_y, sp_depth, agg_npz)
            print("  -> spatial_aggregate.npz")

            # 'occupancy' is the historical depth-integrated view; 'uniform'
            # strips out how long the permeant lingered at each depth, which
            # is what makes maps comparable between simulations.
            depth_w = (None if args.spatial_depth_weighting == 'occupancy'
                       else 'uniform')
            ov_sfx  = '' if depth_w is None else f'_{depth_w}'

            plot_all_spatial_maps(
                agg_npz,
                chain_bonds   = CHAIN_BONDS,
                lipid_species = _species,
                leaflets      = leaflets,
                output_dir    = PLOT_OUT,
                plot_bonds    = args.plot_bonds,
                depth_weights = depth_w,
            )

            for chain in CHAIN_BONDS:
                plot_spatial_overview(
                    agg_npz,
                    chain         = chain,
                    lipid_species = _species,
                    leaflets      = leaflets,
                    depth_weights = depth_w,
                    output_path   = os.path.join(
                        PLOT_OUT, f"spatial_overview_{chain}{ov_sfx}.png"
                    ),
                )

            if args.spatial_depth_plots:
                sp_maps = SpatialMaps(agg_npz)
                for d in range(sp_maps.n_depth):
                    lo, hi = sp_maps.depth_edges[d], sp_maps.depth_edges[d + 1]
                    for chain in CHAIN_BONDS:
                        plot_spatial_overview(
                            agg_npz,
                            chain         = chain,
                            lipid_species = _species,
                            leaflets      = leaflets,
                            depth         = d,
                            output_path   = os.path.join(
                                PLOT_OUT,
                                f"spatial_overview_{chain}_zbin{d}.png"
                            ),
                        )
                    print(f"  -> depth bin {d}: z {lo:.1f}…{hi:.1f} Å")

        except RuntimeError as exc:
            print(f"  [spatial] Skipping spatial plots: {exc}")
