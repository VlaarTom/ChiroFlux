import os

os.environ["OMP_NUM_THREADS"] = "1"
import traceback
import warnings
import zipfile
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
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

from . import panels
from .pathdata import read_traj_plan, remove_offset_files, traj_plan_segments

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


def p2_order_parameter(bond_vectors, normal=MEMBRANE_NORMAL, box=None):
    """
    Compute the P2 order parameter.

    P2 uses cos²θ so it is insensitive to the sign of the bond vector —
    upper and lower leaflet bonds can safely be mixed.  No leaflet split
    is required here.

    `box` (3 box lengths) minimum-images the vectors first, which is required
    whenever they were built from raw, independently-wrapped coordinates —
    see _min_image. A bond split across a lateral boundary otherwise reads as
    a box-length vector lying in the membrane plane, i.e. P2 = -0.5, and the
    error is one-directional because lipids do not wrap along the normal.
    """
    vecs  = np.asarray(bond_vectors, dtype=float)
    if box is not None:
        vecs = _min_image(vecs, box)
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
# Collective tilt angle  — leaflet-aware
# ═════════════════════════════════════════════════════════════════════════════

def mean_tilt_angle(tip_positions, base_positions, normal=MEMBRANE_NORMAL,
                    leaflet='both', upper_mask=None, box=None):
    """
    Collective tilt angle (θ) and azimuthal angle (φ) for a set of lipid
    chains.

    Returns
    -------
    theta_collective, phi, mean_vector, theta_individual

    Two different tilt angles, both wanted
    --------------------------------------
    `theta_collective` averages the chain VECTORS and takes the angle of the
    mean. That is a nematic-like order parameter: the direction the leaflet
    leans as a whole. Because the random part of each chain's orientation
    cancels in the vector average, it is systematically SMALLER than the
    typical chain's tilt — a leaflet with every chain at 30° to the normal but
    uniformly spread in azimuth gives θ_collective ≈ 0.

    `theta_individual` takes the angle of each chain first and averages those.
    That is "the average lipid is tilted by X", and for the same splayed
    leaflet it gives 30°.

    Neither is a substitute for the other, which is why both are returned:
    θ_collective answers "is the leaflet leaning, and which way", θ_individual
    answers "how far from the normal is a typical chain". Their difference
    measures the splay. Both are built the same way otherwise — per leaflet
    against that leaflet's own inward reference, then combined — so they
    differ only in whether the averaging happens before or after the arccos.

    Periodic images
    ---------------
    `box` (3 box lengths) minimum-images each tip-base vector. Supply it.
    The end-to-end vector of an acyl chain is long — ~18 Å for C21→C216 —
    so in a ~70 Å box on the order of 15% of chains straddle a lateral
    boundary in any frame, and an unwrapped one becomes a ~L vector lying in
    the membrane plane. Because the vectors are averaged before the angle is
    taken, one such artifact outweighs about four genuine chains: an
    untilted leaflet measured this way reads several degrees tilted.

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
    if box is not None:
        vecs = _min_image(vecs, box)
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
        return np.nan, np.nan, vecs.mean(axis=0), np.nan

    theta_deg = float(np.mean(theta_results))

    # The mean of the INDIVIDUAL chain angles, reported alongside. Each chain
    # is measured against its own leaflet's inward reference and the angles
    # are averaged per leaflet before the leaflets are combined, matching how
    # theta_deg is built — so the two differ only in whether the averaging
    # happens before or after the arccos, which is exactly the quantity of
    # interest: theta_ind - theta_deg is a measure of how splayed the leaflet
    # is, and theta_ind alone is what "the average lipid is tilted by X" means.
    ind_results = []
    for mask, local_n in masks:
        if not mask.any():
            continue
        v      = vecs[mask]
        norms  = np.linalg.norm(v, axis=1)
        usable = norms > 1e-12
        if not usable.any():
            continue
        cos_i = np.clip((v[usable] / norms[usable, None]) @ local_n, -1.0, 1.0)
        ind_results.append(float(np.degrees(np.arccos(cos_i)).mean()))
    theta_ind = float(np.mean(ind_results)) if ind_results else np.nan

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

    return theta_deg, phi_deg, last_mean_vec, theta_ind


# ═════════════════════════════════════════════════════════════════════════════
# Helper: leaflet membership
# ═════════════════════════════════════════════════════════════════════════════

def _leaflet_by_resindex(p_atoms, axis=2):
    """Sorted resindices whose lipid sits in the upper leaflet.

    Leaflet membership is a property of the LIPID, taken once from its
    phosphorus, and every leaflet-resolved observable should look it up here
    rather than re-derive a split from whatever atoms it happens to hold. A
    bond deep in the acyl chain has base atoms on both sides of the midplane
    with strongly overlapping distributions, so splitting on their own mean z
    assigns a large fraction of them to the wrong leaflet — and does so worse
    the deeper the bond, which reads as a physical trend and is not one.

    Returned sorted, as an array rather than a set, so the lookup below stays
    vectorised: it runs once per bond per chain per frame.
    """
    if p_atoms is None or len(p_atoms) == 0:
        return np.array([], dtype=np.intp)
    z = p_atoms.positions[:, axis]
    return np.sort(p_atoms.resindices[z > z.mean()])


def _upper_mask_for(resindices, upper_resix):
    """Boolean upper-leaflet mask for rows carrying these resindices."""
    if len(upper_resix) == 0:
        return np.zeros(len(resindices), dtype=bool)
    return np.isin(resindices, upper_resix, assume_unique=False)


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


def _min_image(vectors, box_lengths):
    """Wrap displacement vectors into the minimum image, per axis.

    For INTRAMOLECULAR vectors — a C-C bond, a chain's end-to-end vector —
    this is not a refinement but a correctness requirement, because these
    trajectories are not written with ``-pbc mol/whole``: an atom pair
    straddling a boundary differs by ~L in the raw coordinates, and the
    resulting "bond" is a box-length vector pointing along a box axis. It
    passes any non-zero-length check and normalises to a direction that is
    pure artifact, so it corrupts an order parameter silently rather than
    loudly.

    An axis with a zero or unset box length is left alone rather than
    dividing by zero.
    """
    v = np.array(vectors, dtype=float, copy=True)
    L = np.asarray(box_lengths, dtype=float)
    if v.ndim == 1:
        good = L > 0
        v[good] -= np.round(v[good] / L[good]) * L[good]
        return v
    good = L > 0
    v[:, good] -= np.round(v[:, good] / L[good]) * L[good]
    return v


def _whole_centroid(positions, box_lengths):
    """Geometric centre of ONE molecule, made whole across the PBC first.

    These trajectories are not pre-processed with ``-pbc mol/whole`` — the
    same assumption cv_generation.compute_com_unwrapped is built around — so a
    molecule's own atoms can be wrapped to opposite sides of the box. A plain
    mean of those raw positions then lands near mid-box rather than on the
    molecule, off by up to ~L/2. Every permeant-centred quantity here (the
    lateral MSD, the depth CV, and every radial or stamped spatial map) is
    measured *from* that point, so the error moves the whole reference frame,
    not one lipid's value.

    Every atom is min-imaged against the group's first atom before averaging.
    That is valid only for a single bonded molecule, where there is one anchor
    the rest of the group is genuinely near — do not use it for a centroid
    built from many unrelated molecules (one marker atom per lipid, say),
    which has no such anchor. It is an exact no-op when the molecule is
    already whole, so it only changes the frames that were wrong.

    The returned point is NOT re-wrapped into the primary cell: it tracks the
    anchor atom rigidly, which is what makes consecutive frames differ by a
    clean multiple of L and keeps `_unwrap_xy_pbc`'s 0.5*L cutoff reliable.
    Callers that difference it against other positions must min-image the
    difference, as they already do.
    """
    pos = np.asarray(positions, dtype=float)
    if len(pos) == 0:
        return np.full(pos.shape[1] if pos.ndim > 1 else 3, np.nan)
    if len(pos) == 1:
        return pos[0].copy()

    L   = np.asarray(box_lengths, dtype=float)
    ref = pos[0]
    rel = pos - ref[np.newaxis, :]
    # An unset or zero box length means "no periodicity along this axis"
    # rather than a division by zero.
    good = L > 0
    rel[:, good] -= np.round(rel[:, good] / L[good]) * L[good]
    return ref + rel.mean(axis=0)


# ═════════════════════════════════════════════════════════════════════════════
# Local membrane thickness
# ═════════════════════════════════════════════════════════════════════════════

def _local_thickness_dict(universe, lipid_resnames, phosphorus_name, axis,
                          p_atoms=None):
    """
    Local membrane thickness from P-P distances between the leaflets, as a
    dict of per-leaflet arrays so the frame loop can accumulate upper and
    lower separately without repeating the P-P calculation.

    `p_atoms` lets the caller pass a selection made once per trajectory
    instead of paying for a selection-string parse on every frame of every
    species; it must be the same selection this would have made.
    """
    ax   = axis
    axes = [i for i in range(3) if i != ax]

    if p_atoms is None:
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
    perm_lateral = _whole_centroid(permeant.positions,
                                   universe.dimensions[:3])[axes]

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
        perm_lateral = _whole_centroid(permeant.positions,
                                       universe.dimensions[:3])[axes]
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

CURVATURE_METHODS  = ('rbf', 'fourier')
CURVATURE_SURFACES = ('species', 'all')


def check_curvature_options(method, surface, wavelength):
    """Reject a bad curvature setting before any path is processed."""
    if method not in CURVATURE_METHODS:
        raise ValueError(f"-curvature-method must be one of {CURVATURE_METHODS}, "
                         f"not '{method}'.")
    if surface not in CURVATURE_SURFACES:
        raise ValueError(f"-curvature-surface must be one of "
                         f"{CURVATURE_SURFACES}, not '{surface}'.")
    if not (np.isfinite(wavelength) and wavelength > 0):
        raise ValueError(f"-curvature-wavelength must be > 0, got {wavelength}.")


def _monge_curvature(hx, hy, hxx, hyy, hxy):
    """H and K of a height field z = h(x, y) from its derivatives."""
    E = 1.0 + hx**2
    F = hx * hy
    G = 1.0 + hy**2
    denom = np.sqrt(E * G - F**2)
    H = (E * hyy - 2*F*hxy + G*hxx) / (2 * denom**3)
    K = (hxx * hyy - hxy**2) / denom**4
    return H, K


def _fourier_modes(Lx, Ly, wavelength):
    """Wave vectors (M, 2) of the kept modes, one per +/-q pair.

    A mode is kept when its wavelength is at least `wavelength`, i.e.
    |q| <= 2 pi / wavelength. Only one of q and -q is listed: with a cosine
    and a sine per mode that already spans the real series.
    """
    m_max = int(np.floor(Lx / wavelength))
    n_max = int(np.floor(Ly / wavelength))
    modes = [(m, n)
             for n in range(0, n_max + 1)
             for m in range(-m_max, m_max + 1)
             if (n > 0 or m > 0)
             and (m / Lx) ** 2 + (n / Ly) ** 2 <= 1.0 / wavelength ** 2]
    if not modes:
        return np.empty((0, 2))
    mn = np.asarray(modes, dtype=float)
    return 2 * np.pi * mn / np.array([Lx, Ly])


def _fourier_surface_curvature(xy, z, Lx, Ly, wavelength):
    """H, K at each point of a low-pass periodic Fourier fit of z(x, y).

    Least squares, not interpolation: the fitted surface does not pass through
    each phosphate, so single-lipid protrusions shorter than `wavelength` are
    filtered out instead of turning into curvature, and two phosphates at the
    same (x, y) are just two samples of the surface there. The series is
    periodic in the box, so no ghost images are needed and positions need not
    be wrapped. Derivatives are analytic.

    Returns NaNs (with a warning) when the leaflet has fewer phosphates than
    the series has coefficients: the fit would be underdetermined and the
    minimum-norm answer is as noise-dominated as exact interpolation.
    """
    n = len(z)
    q = _fourier_modes(Lx, Ly, wavelength)
    n_coef = 1 + 2 * len(q)
    if n < n_coef:
        warnings.warn(
            f"Fourier curvature: {n} phosphates cannot determine {n_coef} "
            f"coefficients (box {Lx:.1f} x {Ly:.1f} A, cutoff {wavelength} A). "
            "Raise -curvature-wavelength, or fit on all lipids with "
            "-curvature-surface all. Curvature for this leaflet is NaN.",
            stacklevel=3,
        )
        return np.full(n, np.nan), np.full(n, np.nan)

    xy    = np.asarray(xy, dtype=np.float64)
    phase = xy @ q.T                       # (n, M)
    cos, sin = np.cos(phase), np.sin(phase)
    A = np.hstack([np.ones((n, 1)), cos, sin])
    coef, *_ = np.linalg.lstsq(A, np.asarray(z, dtype=np.float64), rcond=None)
    a, b = coef[1:1 + len(q)], coef[1 + len(q):]

    qx, qy = q[:, 0], q[:, 1]
    d1 = -sin * a + cos * b                # d/d(phase)
    d2 = -cos * a - sin * b                # d^2/d(phase)^2
    hx,  hy  = d1 @ qx, d1 @ qy
    hxx, hyy = d2 @ qx**2, d2 @ qy**2
    hxy      = d2 @ (qx * qy)
    return _monge_curvature(hx, hy, hxx, hyy, hxy)


def _curvature_on_species(curv_all, species_resindices):
    """One species' curvature, read off the surface fitted to ALL lipids.

    The membrane has one surface per leaflet; a species does not have its
    own. Fitting a surface through only the POPC phosphates (11 per leaflet
    at 5:1) interpolates across gaps of ~20 A between them, so its curvature
    describes the sparse sampling rather than the membrane. Here H and K come
    from the all-lipid fit and are evaluated at this species' phosphates, so
    the species-resolved numbers differ only in WHERE the surface is sampled.
    Leaflet membership comes along from the all-lipid split as well.

    Returns a dict with the same keys as local_membrane_curvature. The radial
    profile is left out (None): it is only ever taken from the mixed result.
    """
    out = {'radial_r': None, 'radial_H_upper': None}
    for leaf in ('upper', 'lower'):
        keep = np.isin(curv_all[f'{leaf}_resindices'], species_resindices)
        for field in ('H', 'K', 'xy', 'resindices'):
            out[f'{leaf}_{field}'] = curv_all[f'{leaf}_{field}'][keep]
        H, K = out[f'{leaf}_H'], out[f'{leaf}_K']
        out[f'mean_H_{leaf}'] = (float(np.nanmean(np.abs(H)))
                                 if np.isfinite(H).any() else np.nan)
        out[f'mean_K_{leaf}'] = (float(np.nanmean(K))
                                 if np.isfinite(K).any() else np.nan)
    return out


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
    p_atoms             = None,
    method              = 'rbf',
    wavelength          = 20.0,
):
    """Mean (H) and Gaussian (K) curvature of each leaflet's phosphate surface.

    `method` picks how the surface z(x, y) is built from the phosphates:

    'rbf'      an RBF through every phosphate (exactly, at rbf_smoothing=0),
               with ghost images for the periodic boundary. With exact
               interpolation each phosphate's thermal protrusion becomes
               curvature, so the values are dominated by lipid-scale noise.
    'fourier'  a least-squares fit of a periodic Fourier series holding only
               the modes of wavelength >= `wavelength` (Angstrom), with the
               derivatives taken analytically. Periodic by construction, and
               the cutoff is an explicit length scale for what counts as
               membrane shape rather than single-lipid motion.
    """
    if method not in CURVATURE_METHODS:
        raise ValueError(f"Unknown curvature method '{method}'; "
                         f"expected one of {CURVATURE_METHODS}.")
    ax   = bilayer_normal_axis
    lat  = [i for i in range(3) if i != ax]

    # `p_atoms` may be supplied by a caller that selected once per trajectory;
    # the RBF fit below is the most expensive thing done per frame, so the
    # selection parse on top of it is worth avoiding.
    if p_atoms is None:
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
        # float32 `%` can round a tiny negative coordinate up to exactly L;
        # that point and its -L ghost would then sit on both edges of the
        # cell, and its ghost could land exactly on a real point at 0.
        xy_real[xy_real[:, 0] >= Lx, 0] -= Lx
        xy_real[xy_real[:, 1] >= Ly, 1] -= Ly

        # An interpolating RBF cannot pass through two different heights at
        # the same (x, y): the system gets two identical rows and is singular.
        # It happens for real - two phosphates of one leaflet stacked
        # vertically (one lipid pushed out of the plane) that also land on the
        # same 0.01 A xtc grid point. The leaflet genuinely has two heights
        # there, so fit one point at their mean z and still evaluate H and K
        # at every real phosphate below.
        fit_xy, inverse = np.unique(xy_real, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        fit_z   = (np.bincount(inverse, weights=z_real)
                   / np.bincount(inverse))
        if len(fit_xy) < 4:
            return np.full(n_real, np.nan), np.full(n_real, np.nan)

        offsets = [(-Lx, -Ly), (-Lx, 0), (-Lx, Ly),
                   (0,   -Ly),            (0,   Ly),
                   (Lx,  -Ly), (Lx,  0), (Lx,  Ly)]

        ghost_xy_list = []
        ghost_z_list  = []
        for dx, dy in offsets:
            gx = fit_xy[:, 0] + dx
            gy = fit_xy[:, 1] + dy
            in_x = (gx > -ghost_margin) & (gx < Lx + ghost_margin)
            in_y = (gy > -ghost_margin) & (gy < Ly + ghost_margin)
            keep = in_x & in_y
            if keep.any():
                ghost_xy_list.append(np.column_stack([gx[keep], gy[keep]]))
                ghost_z_list.append(fit_z[keep])

        if ghost_xy_list:
            all_xy = np.vstack([fit_xy] + ghost_xy_list)
            all_z  = np.concatenate([fit_z] + ghost_z_list)
        else:
            all_xy = fit_xy
            all_z  = fit_z

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

        return _monge_curvature(dzdx, dzdy, d2zdx2, d2zdy2, d2zdxdy)

    def _fit_and_evaluate(pos_3d):
        if method == 'fourier':
            if len(pos_3d) < 4:
                return (np.full(len(pos_3d), np.nan),
                        np.full(len(pos_3d), np.nan))
            return _fourier_surface_curvature(pos_3d[:, lat], pos_3d[:, ax],
                                              Lx, Ly, wavelength)
        return _fit_and_evaluate_pbc(pos_3d)

    upper_pos = p_pos[upper_mask].copy()
    lower_pos = p_pos[lower_mask].copy()
    upper_resindices = p_resindices[upper_mask]
    lower_resindices = p_resindices[lower_mask]

    H_upper, K_upper = _fit_and_evaluate(upper_pos)
    H_lower, K_lower = _fit_and_evaluate(lower_pos)

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
            perm_lat = _whole_centroid(permeant.positions,
                                       universe.dimensions[:3])[lat]
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
    # These maps are stamped from PER-LIPID angles, so they are the
    # individual tilt (mean_tilt_angle's theta_individual), not the collective
    # one the 1-D profile's `theta` column carries. A collective average has
    # no per-lipid value to stamp anywhere.
    'theta_sn1'    : ('plasma',  'Chain tilt θ sn-1 (°)'),
    'theta_sn2'    : ('plasma',  'Chain tilt θ sn-2 (°)'),
    'phi_sn1'      : ('twilight','Azimuthal φ sn-1 (°)'),
    'phi_sn2'      : ('twilight','Azimuthal φ sn-2 (°)'),
    'thickness'    : ('coolwarm','Thickness (Å)'),
    'curvature_H'  : ('PiYG',   'Mean curvature |H| (Å⁻¹)'),
}


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
    """Spread each lipid's value onto the grid cells within `stamp_radius`.

    `box_xy` adds the periodic images of a lipid near a box edge, and is for
    a grid laid out in ABSOLUTE box coordinates. The frame loop does not use
    it and must not: its grid is permeant-centred (build_lateral_grid_centered)
    and the offsets handed in here are already minimum-imaged, so each lipid
    has exactly one correct position relative to the permeant and adding
    images would double-count it. The parameter is kept for callers building
    a box-frame map.
    """
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

    That bin is given finite edges rather than (-inf, +inf): the centre of an
    infinite bin is NaN, which silently disables every depth selection built
    on `depth_centers`. A single bin wide enough to hold any permeant depth
    behaves the same way for selection and keeps the centre finite.
    """
    if 'depth_edges' in data.files:
        return np.asarray(data['depth_edges'], dtype=float)
    return np.array([-1e30, 1e30])


def _as_depth_stack(arr):
    """A stored map as (n_depth, nx, ny).

    NOT np.atleast_3d: that appends the new axis at the END, so a legacy
    2-D (nx, ny) map becomes (nx, ny, 1) and every `sum(axis=0)` below
    collapses the x axis instead of the depth axis — silently returning a
    transposed, wrongly-shaped map rather than failing.
    """
    a = np.asarray(arr)
    return a if a.ndim == 3 else a[np.newaxis, ...]


def aggregate_spatial_maps(path_start, path_end, weights):
    """Pool every path's spatial accumulators onto the first path's grid.

    Each path's stored (unweighted) sums are multiplied by its current weight
    here; paths without a positive weight are left out.

    The accumulators are stacked (n_observables, n_depth, M) in the per-path
    file, so paths on the reference grid are added in one operation each;
    only a path on a different lateral grid falls back to regridding the
    observables one depth bin at a time.
    """
    ref = None
    agg_wsum = agg_wcnt = None
    no_file = bad_depth = 0

    for pn in range(path_start, path_end + 1):
        weight = weights.get(pn, 0.0)
        if not weight > 0.0:
            continue
        sp = read_path_spatial(path_output_file(pn))
        if sp is None:
            no_file += 1
            continue
        # Stored unweighted; the current weight goes on here.
        sp['wsum'] = sp['wsum'] * weight
        sp['wcount'] = sp['wcount'] * weight

        if ref is None:
            ref = sp
            ref_index = {name: i for i, name in enumerate(sp['names'])}
            agg_wsum = np.zeros_like(sp['wsum'])
            agg_wcnt = np.zeros_like(sp['wcount'])

        # Depth bins are derived deterministically from --slab-range and
        # --n-depth-bins, so a mismatch means the path was produced by a
        # different invocation. Pooling those would silently mix depths;
        # skip instead of interpolating along a coarse, physical axis.
        p_depth, r_depth = sp['depth_edges'], ref['depth_edges']
        if (len(p_depth) != len(r_depth)
                or not np.allclose(p_depth, r_depth, atol=1e-6, equal_nan=True)):
            bad_depth += 1
            continue

        if sp['names'] == ref['names']:
            src_idx = dst_idx = slice(None)
        else:
            common = [n for n in sp['names'] if n in ref_index]
            lookup = {name: i for i, name in enumerate(sp['names'])}
            src_idx = [lookup[n] for n in common]
            dst_idx = [ref_index[n] for n in common]

        px, py = sp['x_coords'], sp['y_coords']
        rx, ry = ref['x_coords'], ref['y_coords']
        same_grid = (
            len(px) == len(rx) and len(py) == len(ry)
            and np.allclose(px, rx, atol=0.01)
            and np.allclose(py, ry, atol=0.01)
        )

        if same_grid:
            agg_wsum[dst_idx] += sp['wsum'][src_idx]
            agg_wcnt[dst_idx] += sp['wcount'][src_idx]
            continue

        src_shape = (len(px), len(py))
        src_list = (range(len(sp['names'])) if isinstance(src_idx, slice)
                    else src_idx)
        dst_list = (range(len(sp['names'])) if isinstance(dst_idx, slice)
                    else dst_idx)
        for si, di in zip(src_list, dst_list):
            for d in range(agg_wsum.shape[1]):
                w_d = sp['wsum'][si, d].reshape(src_shape)
                c_d = sp['wcount'][si, d].reshape(src_shape)
                with np.errstate(invalid='ignore', divide='ignore'):
                    src_avg = np.where(c_d > 0, w_d / c_d, np.nan)
                dst_avg = _regrid_map(px, py, src_avg, rx, ry).ravel()
                dst_cnt = _regrid_map(px, py, c_d, rx, ry).ravel()
                valid = np.isfinite(dst_avg) & (dst_cnt > 0)
                agg_wsum[di, d][valid] += dst_avg[valid] * dst_cnt[valid]
                agg_wcnt[di, d][valid] += dst_cnt[valid]

    if ref is None:
        raise RuntimeError("No spatial maps found to aggregate.")
    if no_file or bad_depth:
        print(f"  [spatial-agg] skipped {no_file} weighted path(s) without current spatial maps "
              f"and {bad_depth} with incompatible depth bins.")

    agg_acc = {name: [agg_wsum[i], agg_wcnt[i]]
               for i, name in enumerate(ref['names'])}
    rx, ry = ref['x_coords'], ref['y_coords']
    return (agg_acc, rx, ry, (len(rx), len(ry)), ref['depth_edges'])


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
        wsum = _as_depth_stack(self._data[f'{key}_wsum'])
        wcnt = _as_depth_stack(self._data[f'{key}_wcount'])
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
        wcnt = _as_depth_stack(self._data[f'{key}_wcount'])
        return wcnt.sum(axis=(1, 2))


def load_spatial_maps(npz_path, depth=None, depth_weights=None):
    maps = SpatialMaps(npz_path, depth=depth, depth_weights=depth_weights)
    return maps, maps.x_coords, maps.y_coords


# ═════════════════════════════════════════════════════════════════════════════
# Local diffusion along the bilayer normal
# ═════════════════════════════════════════════════════════════════════════════
#
# D(z) is the short-time limit of the mean squared displacement of the
# permeant, taken among the frames that *start* in a slab:
#
#     D(z) = < [z(t+tau) - z(t)]^2 >_{z(t) in slab} / (2 tau)
#
# The binning is by the starting position and nothing else. An earlier version
# instead cut the trajectory into "sojourns" - runs of consecutive frames in
# the same slab - and estimated D from each. That conditions the sample on the
# permeant *not leaving*, which keeps the slowest excursions and throws the
# rest away, and it does so hardest where the motion is fastest. On a Brownian
# walk of known D with 1 A slabs it recovered 31% of the true D in a fast
# region and 66% in a slow one, flattening a true 6:1 contrast between them to
# 2:1. The start-binned windows below recover both to 100% on the same test.
#
# `lag` is in frames, and one lag alone is not enough to answer the question.
# D_app(tau) = MSD(tau) / 2 tau is only equal to D inside the *diffusive
# window*, and that window is bounded from both sides:
#
#   * too short - the permeant is still rattling inside a transient cage of
#     lipid tails (and, below ~0.1 ps, still flying ballistically). MSD then
#     grows slower than linearly, so D_app DECAYS toward the plateau and a
#     shorter lag reads HIGH. Taking tau -> 0 makes this worse, not better.
#   * too long - the window wanders sqrt(2 D tau) while the value is charged
#     to the slab it STARTED in, so D is smeared over neighbouring slabs.
#     Staying local needs sqrt(2 D tau) <~ slab width, which at 1 A slabs and
#     bulk-water D is already only a couple of ps.
#
# Neither bound is knowable in advance, so instead of picking one lag this
# accumulates a short scan, k = 1 .. n_lags times the base lag, and fits
#
#     MSD(tau) = c + 2 n_dim D tau
#
# per slab. The intercept c absorbs the fast *bounded* component (vibration,
# ballistic flight, intra-cage rattling); the slope is the diffusive D. In the
# D_app domain that fit is a straight line against 1/tau,
#
#     D_app(tau) = D + c / (2 n_dim tau),
#
# whose intercept as 1/tau -> 0 is D. That form is what `fit_diffusion_intercept`
# uses, because it needs only the per-lag D_app values the aggregate already
# holds: the abscissa is 1/k up to a constant factor dt, and a constant factor
# on the abscissa rescales the slope while leaving the intercept alone. So the
# fitted D needs no knowledge of dt at all (only `c`, reported for diagnosis,
# does).
#
# The path-ensemble conditioning enters this picture only weakly. Conditioning
# a diffusion on its future is a Doob h-transform: it changes the drift, not
# the diffusion coefficient, and a drift contributes v^2 tau^2 to the MSD,
# i.e. O(tau) in D_app, against the O(1) error from being off the plateau. The
# dominant bias at a single short lag is the non-diffusive one, and it is not
# specific to path sampling.
#
# Column convention: k = 1 stays `D_z` and k = 2 stays `D_z_lag2`, so every
# existing consumer keeps working; k >= 3 are `D_z_lag3` and up, and the fitted
# intercept is `D_z_fit`.


def slab_binned_msd(series, slab_idx, n_slabs, lag):
    """(sum of squared displacement, window count) per starting slab.

    `series` is (N,) for the normal or (N, 2) for the lateral plane; the
    squared displacement sums over the trailing axis either way. Windows are
    every consecutive pair `lag` frames apart, binned by where they started.
    """
    series = np.asarray(series, dtype=float)
    n_frames = series.shape[0]
    total = np.zeros(n_slabs)
    count = np.zeros(n_slabs)
    if lag < 1 or n_frames <= lag:
        return total, count

    delta = series[lag:] - series[:-lag]
    squared = delta ** 2 if delta.ndim == 1 else np.sum(delta ** 2, axis=1)
    start = np.asarray(slab_idx[:-lag], dtype=int)

    np.add.at(total, start, squared)
    np.add.at(count, start, 1.0)
    return total, count


def lag_column(prefix, k):
    """Name of the per-lag D_app column for lag k*lag_frames.

    k = 1 keeps the bare prefix and k = 2 keeps the historical `_lag2`, so the
    column names that existed before the lag scan still mean exactly what they
    meant (see the section comment above).
    """
    return prefix if k == 1 else f'{prefix}_lag{k}'


def fit_diffusion_intercept(d_app, weights=None, ks=None):
    """Extrapolate D_app(tau) to 1/tau -> 0, per slab.

    Parameters
    ----------
    d_app : (..., n_lags) array
        D_app at lags k = 1..n_lags times the base lag, trailing axis last.
    weights : (..., n_lags) array or None
        Per-lag weights, normally the window counts. Equal weights if None.
    ks : (n_lags,) array or None
        The lag multipliers, defaulting to 1..n_lags. Only their ratios
        matter.

    Returns
    -------
    d_fit : (...,) intercept, i.e. D with the bounded fast component removed
    slope : (...,) the fitted slope against 1/k, = c / (2 n_dim lag dt).
        Multiply by 2 n_dim lag dt to read c in A^2.
    resid : (...,) weighted RMS residual of the fit, in the units of d_app.
        Large values mean MSD is not affine in tau over the scanned range, so
        the slab has no diffusive window at this frame spacing.

    Slabs with fewer than two usable lags come back NaN: an intercept through
    one point is not an extrapolation. With exactly two it reduces to the
    two-point extrapolation 2*D_app(2 tau) - D_app(tau) — note the order: the
    line is straight in 1/tau, not in tau, so it is the LONGER lag that gets
    the positive weight. The tau-space form 2*D_app(tau) - D_app(2 tau) that
    a Richardson reflex suggests extrapolates the wrong way entirely.
    """
    y = np.asarray(d_app, dtype=float)
    n_lags = y.shape[-1]
    ks = (np.arange(1, n_lags + 1, dtype=float) if ks is None
          else np.asarray(ks, dtype=float))
    x = 1.0 / ks                                    # abscissa, up to 1/(lag*dt)
    x = np.broadcast_to(x, y.shape)

    w = (np.ones_like(y) if weights is None
         else np.asarray(weights, dtype=float).astype(float))
    w = np.where(np.isfinite(y) & np.isfinite(w) & (w > 0), w, 0.0)
    y = np.where(w > 0, y, 0.0)

    # Weighted least squares for y = a + b x, done in closed form so the whole
    # profile is fitted in one vectorised pass (this runs inside every
    # jackknife replicate).
    sw   = w.sum(axis=-1)
    swx  = (w * x).sum(axis=-1)
    swy  = (w * y).sum(axis=-1)
    swxx = (w * x * x).sum(axis=-1)
    swxy = (w * x * y).sum(axis=-1)

    det = sw * swxx - swx ** 2
    n_used = (w > 0).sum(axis=-1)
    # det <= 0 also catches every lag sharing one x, which no real scan does
    # but a degenerate `ks` would.
    ok = (n_used >= 2) & (det > 0)

    with np.errstate(invalid='ignore', divide='ignore'):
        safe_det = np.where(ok, det, 1.0)
        slope     = np.where(ok, (sw * swxy - swx * swy) / safe_det, np.nan)
        intercept = np.where(ok, (swxx * swy - swx * swxy) / safe_det, np.nan)

        pred = intercept[..., np.newaxis] + slope[..., np.newaxis] * x
        ss   = (w * (y - pred) ** 2).sum(axis=-1)
        # n_used - 2 degrees of freedom; exactly two lags fit perfectly, so
        # report 0 rather than 0/0.
        resid = np.where(ok & (n_used > 2),
                         np.sqrt(ss / np.where(sw > 0, sw, 1.0)),
                         np.where(ok, 0.0, np.nan))

    return intercept, slope, resid


def diffusion_from_msd(total, count, lag, dt, n_dim=1):
    """D per slab from accumulated squared displacements, in A^2/ps.

    MSD = 2 n_dim D t, so D = <dz^2> / (2 n_dim tau). Empty slabs come back
    NaN rather than zero - no windows is not a measurement of zero mobility.
    """
    denominator = 2.0 * n_dim * lag * dt
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(count > 0, total / (denominator * np.maximum(count, 1.0)),
                        np.nan)


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


# ═════════════════════════════════════════════════════════════════════════════
# Slab grid construction
# ═════════════════════════════════════════════════════════════════════════════

def build_slab_grid(topology, xtc_files, permeant_resname,
                    bilayer_normal, slab_width, slab_range=None,
                    lipid_resnames=('DOPC', 'POPC')):
    """Slab edges and centres for the permeant-depth axis.

    The axis is the same one the frame loop bins on,
    ``membrane_z - permeant_z`` - the permeant's depth relative to the bilayer
    centre, running the same way as the order parameter, not lab-frame z.

    `slab_range` should normally be given and is what the CLI passes: the grid
    has to be identical for every path, because the per-path CSVs are pooled
    bin by bin afterwards, and a grid derived from one path's own sampling
    would differ from the next path's. Deriving it is therefore a fallback
    only, and it warns.
    """
    axis = {'x': 0, 'y': 1, 'z': 2}[bilayer_normal.lower()]
    if slab_range is not None:
        z_min, z_max = float(slab_range[0]), float(slab_range[1])
    else:
        warnings.warn(
            "build_slab_grid: no slab_range given, so the grid is derived from "
            "this path's own sampling. Different paths then get different "
            "grids, which cannot be pooled bin by bin - pass -slab-range.",
            stacklevel=2,
        )
        lipid_sel = " or ".join(f"resname {r}" for r in lipid_resnames)
        depths = []
        for xtc in xtc_files:
            u = mda.Universe(topology, xtc, refresh_offsets=True)
            perm = u.select_atoms(f"resname {permeant_resname} and not name H*")
            if len(perm) == 0:
                perm = u.select_atoms(f"resname {permeant_resname}")
            membrane = u.select_atoms(lipid_sel)
            for _ts in u.trajectory:
                depths.append(membrane.positions[:, axis].mean()
                              - perm.positions[:, axis].mean())
        depths = np.asarray(depths, dtype=float)
        z_min = depths.min() - slab_width
        z_max = depths.max() + slab_width
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
    # Every path is classified plus/minus by comparing against this, so a
    # missing key is not an optional setting here: left as None it reaches a
    # `None < float` comparison inside the per-path worker, where the
    # resulting TypeError is caught per path and reported as a traceback that
    # says nothing about the .toml. Fail once, at the top, naming the key.
    if lambda_minus_one is None:
        raise ValueError(
            f"{toml_file}: [simulation.tis_set] lambda_minus_one is not set. "
            "It is needed to tell a minus-ensemble path from a plus-ensemble "
            "one, which decides whether the first and last phase points "
            "belong to the path."
        )
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


def align_order_with_plan(order, plan, g96_index):
    """Drop order.txt's .g96 row so it lines up with the frame plan.

    Returns (order, problem); `problem` is None when they line up, otherwise a
    message for the caller to skip the path with.

    A .g96 phase point has no .xtc frame, so `read_traj_plan` leaves it out of
    the plan and returns its step instead — precisely so a caller can drop the
    matching order-parameter row. Skipping that step leaves the two one apart,
    and the consequence is not a shifted lookup somewhere harmless: `first`
    and `last` are read off order.txt to decide which END of the plan is
    trimmed, so an off-by-one there removes a frame that belongs to the path
    and keeps one that does not.

    The row is matched on its step (column 0) rather than by position, because
    the .g96 point is not always at an end. If that match does not remove
    exactly one row the order file is not what it is assumed to be, and the
    caller is told rather than left to line the two up by luck.
    """
    order = np.atleast_2d(order)
    if g96_index is not None and len(order):
        keep = ~np.isclose(order[:, 0], float(g96_index))
        if keep.sum() == len(order) - 1:
            order = order[keep]

    if len(order) == 0:
        return order, 'order.txt has no usable rows.'
    if len(order) != len(plan):
        return order, (
            f'order.txt has {len(order)} phase point(s) but traj.txt lists '
            f'{len(plan)} with a trajectory frame'
            + (f' (after dropping the .g96 point at step {g96_index})'
               if g96_index is not None else '')
            + '; they must line up for the order parameter to index the frames.'
        )
    return order, None


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

# ═════════════════════════════════════════════════════════════════════════════
# Per-path output file
# ═════════════════════════════════════════════════════════════════════════════
#
# Everything one path produces goes into a single compressed Memb_OP/<n>.npz.
# It used to be seven CSVs and a spatial .npz per path, and at the path counts
# RETIS reaches that was the dominant cost of the analysis: ~1.4 MB and eight
# files per path, 87% of the bytes being `0.000000e+00` written out as text,
# since one path only visits a fraction of the membrane. The same content is
# ~0.19 MB in one file and reads back ~10x faster at aggregation.
#
# Layout. Each table is stored as three members rather than one per column -
# the zip overhead of ~1200 small members made a column-wise file *slower* to
# read than the CSVs it replaced:
#
#     <key>__columns   names, the axis column first
#     <key>__axis      the axis (slab_center, r_center, z_center), float64
#     <key>__data      every other column, float32
#
# and the spatial accumulators are stacked the same way into
# spatial__wsum / spatial__wcount of shape (n_observables, n_depth, M), with
# spatial__names giving the order. float32 is not a precision loss against
# what was there before: the CSVs were written with %.6e, 7 significant
# digits, which is what float32 holds.
#
# Per-path granularity is kept on purpose. Workers write concurrently, each to
# its own file, and a missing or partial group is recomputed on its own and
# merged into the file.
#
# The sums are stored UNWEIGHTED (every path accumulated with weight 1) and the
# WHAM path weight is applied in `_collect_tables` / `aggregate_spatial_maps`.
# The weights renormalise every time the simulations are extended; with the
# weight baked in, as version 1 did, that meant recomputing every path, and a
# path with no weight yet held nothing that could ever be rescaled. Every
# accumulator is linear in the weight (+= w*x, += w), so applying it afterwards
# is exact.

#: 6: the tilt tables carry both angles - `theta` (collective, the leaflet's
#:    net lean) and `theta_ind` (the mean individual chain tilt).
#: 5: intramolecular vectors (P2 bonds, chain tilt) minimum-imaged, and
#:    leaflet membership taken from the lipid's own P atom instead of from
#:    each observable's own atoms' z, in BOTH the 1-D and the spatial block.
#:    Both change the order-parameter and tilt tables; tilt changes a lot.
#: 4: the diffusion tables carry a lag scan (D_z, D_z_lag2, D_z_lag3 ...)
#:    instead of a single lag plus one check, so the aggregate can fit
#:    D_app(tau) -> D. A version-3 file holds only two lags; it still pools,
#:    and its fit degenerates to the two-point extrapolation.
#: 3: permeant centroid made whole across the PBC before it is used as the
#:    origin of anything (lateral MSD, stamped maps, radial profiles, the
#:    near-N ranking), and the lateral unwrap state carried across segment
#:    junctions instead of being reset per .xtc.
#: 2: unweighted sums. 1: weighted at accumulation.
#: None of these are interchangeable, so an older file is treated as absent
#: and recomputed. A version-2 file is not merely less accurate - pooling it
#: with a version-3 one mixes two different map origins in the same average.
PATH_FILE_VERSION = 6

_TABLE_KEYS = ('p2', 'tilt', 'thick', 'deform', 'water_z', 'diff', 'diff_xy')


def path_output_file(path_number, out_dir=None):
    """The one file a path's results live in."""
    return Path(out_dir if out_dir is not None else ORDER_OUT) / f"{path_number}.npz"


def _pack_table(key, df):
    return {
        f'{key}__columns': np.array([str(c) for c in df.columns]),
        f'{key}__axis'   : df.iloc[:, 0].to_numpy(dtype=np.float64),
        f'{key}__data'   : df.iloc[:, 1:].to_numpy(dtype=np.float32),
    }


def _pack_spatial(accumulators, x_coords, y_coords, depth_edges):
    names = list(accumulators)
    if not names:
        raise ValueError("No spatial accumulators to write.")
    return {
        'spatial__names'      : np.array(names),
        'spatial__wsum'       : np.stack([np.atleast_2d(accumulators[n][0])
                                          for n in names]).astype(np.float32),
        'spatial__wcount'     : np.stack([np.atleast_2d(accumulators[n][1])
                                          for n in names]).astype(np.float32),
        'spatial__x_coords'   : np.asarray(x_coords, dtype=np.float64),
        'spatial__y_coords'   : np.asarray(y_coords, dtype=np.float64),
        'spatial__depth_edges': np.asarray(depth_edges, dtype=np.float64),
    }


def _open_path_output(path):
    """np.load the file, or None if it is missing or unreadable.

    Unreadable covers a file left behind by something other than
    `write_path_output` - that writes through a temporary file and a rename,
    so a killed worker leaves the previous file intact rather than a truncated
    one.
    """
    try:
        return np.load(str(path))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        warnings.warn(f"{path}: unreadable ({exc}); treating it as absent.",
                      stacklevel=3)
        return None


def _file_version(data):
    return int(data['meta__version']) if 'meta__version' in data.files else 0


def path_output_version(path):
    """The file's format version, or None if there is no readable file."""
    data = _open_path_output(path)
    if data is None:
        return None
    with data:
        return _file_version(data)


def _open_current(path):
    """np.load the file only if it is in the current format, else None.

    An older file is not merely stale: version 1 stored sums already
    multiplied by the path weight, and reading it as unweighted would apply the
    weight twice. So it counts as absent - recomputed by the workers, skipped
    by the aggregation.
    """
    data = _open_path_output(path)
    if data is None:
        return None
    if _file_version(data) != PATH_FILE_VERSION:
        data.close()
        return None
    return data


def path_output_contents(path):
    """The table keys, plus 'spatial', that the file holds."""
    data = _open_current(path)
    if data is None:
        return set()
    with data:
        files = set(data.files)
    present = {key for key in _TABLE_KEYS if f'{key}__data' in files}
    if 'spatial__wsum' in files:
        present.add('spatial')
    return present


def write_path_output(path, tables=None, spatial=None, meta=None):
    """Write a path's tables and spatial maps, merging into what is there.

    `tables` maps table keys to DataFrames whose first column is the axis;
    `spatial` is (accumulators, x_coords, y_coords, depth_edges); `meta` is a
    dict of scalars. Groups already in the file and not supplied here are
    carried over, so a run that recomputes only the diffusion keeps the order
    parameters from an earlier one.

    The file is written to a temporary name and renamed into place, so readers
    never see a half-written file.
    """
    path = Path(path)
    tables = dict(tables or {})
    meta = dict(meta or {})
    replaced = set(tables) | ({'spatial'} if spatial is not None else set())

    arrays = {}
    old = _open_current(path) if path.exists() else None
    if old is not None:
        with old:
            for name in old.files:
                group, _, field = name.partition('__')
                if group in replaced:
                    continue
                if group == 'meta' and (field in meta or field == 'version'):
                    continue
                arrays[name] = old[name]

    for key, df in tables.items():
        if key not in _TABLE_KEYS:
            raise ValueError(f"Unknown table '{key}'; expected one of {_TABLE_KEYS}.")
        arrays.update(_pack_table(key, df))
    if spatial is not None:
        arrays.update(_pack_spatial(*spatial))

    arrays['meta__version'] = np.array(PATH_FILE_VERSION)
    for field, value in meta.items():
        arrays[f'meta__{field}'] = np.array(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    # A file object, not a name: np.savez would append '.npz' to 'x.npz.tmp'.
    with open(tmp, 'wb') as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def _unpack_table(data, key):
    columns = data[f'{key}__columns'].tolist()
    df = pd.DataFrame(data[f'{key}__data'].astype(np.float64),
                      columns=columns[1:])
    df.insert(0, columns[0], data[f'{key}__axis'])
    return df


def read_path_tables(path, keys=_TABLE_KEYS):
    """{key: DataFrame} for whichever of `keys` the file holds, unweighted."""
    data = _open_current(path)
    if data is None:
        return {}
    with data:
        files = set(data.files)
        return {key: _unpack_table(data, key)
                for key in keys if f'{key}__data' in files}


def read_path_columns(path, key):
    """Column names of one table, without reading its data."""
    data = _open_current(path)
    if data is None:
        return []
    with data:
        name = f'{key}__columns'
        return data[name].tolist() if name in data.files else []


def read_path_spatial(path):
    """The stacked spatial accumulators, or None if the file has none.

    Returns a dict with 'names', 'wsum' and 'wcount' (n_obs, n_depth, M) in
    float64, unweighted, and the grid's 'x_coords', 'y_coords' and
    'depth_edges'.
    """
    data = _open_current(path)
    if data is None:
        return None
    with data:
        if 'spatial__wsum' not in data.files:
            return None
        return {
            'names'      : data['spatial__names'].tolist(),
            'wsum'       : data['spatial__wsum'].astype(np.float64),
            'wcount'     : data['spatial__wcount'].astype(np.float64),
            'x_coords'   : data['spatial__x_coords'],
            'y_coords'   : data['spatial__y_coords'],
            'depth_edges': data['spatial__depth_edges'],
        }


def _has_near_columns(path, group, near_n):
    """
    Whether an existing file already carries the near-permeant columns for
    every requested N. Guards against silently skipping recomputation when
    the near-N feature is adopted on top of files written before that schema
    existed.
    """
    if group == 'order':
        key, probe_chain = 'p2', next(iter(CHAIN_BONDS))
        probe_cols = [f'near{N}_{probe_chain}_p2_chain_wsum' for N in near_n]
    else:  # 'structural'
        key = 'thick'
        probe_cols = [f'near{N}_thickness_wsum' for N in near_n]
    columns = set(read_path_columns(path, key))
    return all(c in columns for c in probe_cols)


def _needed_groups(path, overwrite, near_n=()):
    if overwrite:
        return set(OBSERVABLE_GROUPS.keys())
    present = path_output_contents(path)
    needed = set()
    for group, keys in OBSERVABLE_GROUPS.items():
        if not all(k in present for k in keys):
            needed.add(group)
            continue
        if near_n and group in ('order', 'structural') and \
           not _has_near_columns(path, group, near_n):
            needed.add(group)
    return needed


# ═════════════════════════════════════════════════════════════════════════════
# Selective single-pass accumulator  — now with leaflet dimension
# ═════════════════════════════════════════════════════════════════════════════

def calculate_selected_slab_profiles(
    topology,
    xtc_files,
    frame_plan,
    permeant_resname,
    needed_groups,
    weight              = 1.0,
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
    curvature_method    = 'rbf',
    curvature_wavelength = 20.0,
    curvature_surface   = 'species',
    dt                  = None,
    lag_frames          = 1,
    n_lags              = 6,
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
    check_curvature_options(curvature_method, curvature_surface,
                            curvature_wavelength)
    run_structural    = 'structural'   in needed_groups
    run_diffusion     = 'diffusion'    in needed_groups
    run_diffusion_xy  = 'diffusion_xy' in needed_groups
    run_spatial       = 'spatial'      in needed_groups

    axis      = {'x': 0, 'y': 1, 'z': 2}[bilayer_normal.lower()]
    normal    = MEMBRANE_NORMAL
    _lat_axes = [a for a in range(3) if a != axis]

    edges, centers = build_slab_grid(
        topology, xtc_files, permeant_resname,
        bilayer_normal, slab_width, slab_range, lipid_resnames,
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
        # theta_ind: the mean of the INDIVIDUAL chain angles, beside the
        # collective theta above. See mean_tilt_angle - they answer different
        # questions and their difference measures leaflet splay.
        theta_ind_wsum = {(sp, lf, ch): np.zeros(n_slabs)
                          for sp in _ORDER_SPECIES
                          for lf in _LEAFLETS
                          for ch in tilt_endpoints}
        theta_ind_wtot = {(sp, lf, ch): np.zeros(n_slabs)
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
        near_theta_ind_wsum = {(N, sp, ch): np.zeros(n_slabs)
                               for N in near_n
                               for sp in _ORDER_SPECIES
                               for ch in tilt_endpoints}
        near_theta_ind_wtot = {(N, sp, ch): np.zeros(n_slabs)
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

    # ── Diffusion accumulators ────────────────────────────────────────────────
    # wsum holds summed squared displacement over 2*n_dim*tau and wtot the
    # window count, so the aggregate's ratio is the pooled MSD estimate.
    # One (wsum, wtot) pair per lag multiplier k = 1..n_lags, so the aggregate
    # can fit D_app against 1/tau instead of trusting a single lag. See the
    # section comment above fit_diffusion_intercept.
    _lag_ks = list(range(1, max(int(n_lags), 2) + 1))

    if run_diffusion:
        D_z_wsum      = {k: np.zeros(n_slabs) for k in _lag_ks}
        D_z_wtot      = {k: np.zeros(n_slabs) for k in _lag_ks}
        nwin_wsum     = np.zeros(n_slabs)
        nwin_wtot     = np.zeros(n_slabs)
        z_full        = []
        slab_idx_full = []

    if run_diffusion_xy:
        D_xy_wsum      = {k: np.zeros(n_slabs) for k in _lag_ks}
        D_xy_wtot      = {k: np.zeros(n_slabs) for k in _lag_ks}
        nwin_xy_wsum   = np.zeros(n_slabs)
        nwin_xy_wtot   = np.zeros(n_slabs)
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
    n_frames_seen   = 0
    n_out_of_range  = 0
    _dt_warned      = False

    # Lateral unwrapping state for the permeant. It belongs to the *path*, not
    # to the .xtc file: frame_plan's segments are consecutive phase points, so
    # the last frame of one segment and the first of the next are one dt apart
    # and the accumulated jump has to carry across the junction. Resetting it
    # per file discarded that jump, so a permeant that had crossed a lateral
    # boundary earlier in the path produced one spurious ~L displacement at
    # every junction, i.e. a ~L^2 window binned into whatever slab it landed
    # in.
    if run_diffusion_xy:
        _pbc_jump   = np.zeros(2)
        _pbc_prev   = None
        _pbc_cutoff = 0.5

    # ═════════════════════════════════════════════════════════════════════════
    # Frame loop
    # ═════════════════════════════════════════════════════════════════════════
    for xtc, frame_indices in frame_plan:
        u          = mda.Universe(topology, xtc, refresh_offsets=True)
        permeant   = u.select_atoms(f"resname {permeant_resname}")
        perm_heavy = permeant.select_atoms("not name H*")
        if len(perm_heavy) == 0:
            perm_heavy = permeant
        membrane   = u.select_atoms("resname DOPC POPC")

        # A single-frame .xtc reports dt = 0 — there is no frame pair to take
        # the difference of — and dt is a divisor of every D, so latching onto
        # that zero turns a whole path's diffusion columns into inf and poisons
        # every slab it touches once they are pooled. Keep asking later
        # segments instead, and if none of them can say, skip the group rather
        # than emit a number (see the guard on the post-processing below).
        # Passing -dt explicitly avoids the question entirely and is worth
        # doing: the value is a property of the run, not of whichever segment
        # happened to be read first.
        if (run_diffusion or run_diffusion_xy) and dt is None:
            u.trajectory[0]
            _candidate = float(u.trajectory.dt)
            if _candidate > 0.0:
                if _dt is None:
                    _dt = _candidate
                elif not np.isclose(_candidate, _dt, rtol=1e-3):
                    # Segments written at different output frequencies cannot
                    # share one tau: the series is treated as evenly spaced,
                    # and it is not. Say so rather than scale part of it
                    # wrongly.
                    if not _dt_warned:
                        print(f"  [worker] WARNING: frame spacing changes "
                              f"between this path's segments ({_dt} vs "
                              f"{_candidate} ps). The MSD assumes one spacing; "
                              f"pass -dt to fix it explicitly.")
                        _dt_warned = True

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

        # P atoms, selected once per .xtc rather than per frame per species.
        # They serve two purposes below: the leaflet identity of every lipid
        # (see _leaflet_by_resindex) and the thickness/curvature surfaces.
        _p_atoms = {}
        if run_order or run_structural or run_spatial:
            for sp in (['mixed'] + list(lipid_resnames)):
                sel = lipid_sel if sp == 'mixed' else f"resname {sp}"
                _p_atoms[sp] = u.select_atoms(
                    f"({sel}) and name {phosphorus_name}"
                )

        # Frames come from traj.txt via frame_plan: exactly this path's phase
        # points, in path order, already trimmed of the first/last one where
        # the ensemble requires it. This matters more here than elsewhere -
        # z_series feeds MSD and autocorrelation estimates, so a frame that was
        # never on the path inserts a spurious jump and corrupts the effective
        # time step. The rule this replaces dropped one frame per junction
        # where the .xtc files carry 2*(n_segments-1)+1 frames outside the path.

        for fi in frame_indices:
            u.trajectory[fi]
            n_frames_seen += 1
            box_3 = u.dimensions[:3]

            # Leaflet membership by lipid IDENTITY, from the P atom, resolved
            # once per frame. Every leaflet-resolved observable below keys off
            # this rather than off its own atoms' z. Splitting a *bond* on the
            # mean z of its own base atoms works for bond 0, whose base (C21)
            # sits at the glycerol and is well separated between leaflets, but
            # degrades to noise for deep-tail bonds, whose base atoms cluster
            # around the midplane with heavily overlapping distributions. That
            # made the leaflet contrast appear to vanish toward the chain end
            # as an artifact of the assignment rather than as physics.
            _upper_resix = (_leaflet_by_resindex(_p_atoms.get('mixed'), axis)
                            if (run_order or run_spatial)
                            else np.array([], dtype=np.intp))
            _upper_resix_sp = _upper_resix

            # Thickness and curvature are the two expensive per-frame
            # computations (a cKDTree pair and an RBF fit with ghost images,
            # each per species), and the structural and spatial blocks both
            # want them. Memoise per frame so they are computed once, not
            # twice - when both groups run this halves the frame cost.
            _frame_cache = {}

            def _thickness_of(sp, _cache=_frame_cache):
                if ('thick', sp) not in _cache:
                    _cache[('thick', sp)] = _local_thickness_dict(
                        u, tuple(lipid_resnames) if sp == 'mixed' else (sp,),
                        phosphorus_name, axis, p_atoms=_p_atoms.get(sp),
                    )
                return _cache[('thick', sp)]

            def _curvature_of(sp, _cache=_frame_cache):
                # The radial add-on is always computed: it is a distance array
                # and a binning against an RBF fit that has already been paid
                # for, so making it optional would only split the cache in two
                # and fit twice. With curvature_surface='all' a species is
                # sampled from the mixed fit instead of getting its own.
                if ('curv', sp) in _cache:
                    return _cache[('curv', sp)]
                if curvature_surface == 'all' and sp != 'mixed':
                    _cache[('curv', sp)] = _curvature_on_species(
                        _curvature_of('mixed'), _p_atoms[sp].resindices,
                    )
                else:
                    _cache[('curv', sp)] = local_membrane_curvature(
                        u,
                        lipid_resnames      = (tuple(lipid_resnames)
                                               if sp == 'mixed' else (sp,)),
                        phosphorus_name     = phosphorus_name,
                        bilayer_normal_axis = axis,
                        rbf_smoothing       = curvature_smoothing,
                        permeant_resname    = permeant_resname,
                        r_max_curvature     = r_max,
                        n_radial_bins       = n_radial_bins,
                        p_atoms             = _p_atoms.get(sp),
                        method              = curvature_method,
                        wavelength          = curvature_wavelength,
                    )
                return _cache[('curv', sp)]

            membrane_z = membrane.positions[:, axis].mean()
            perm_z     = membrane_z - perm_heavy.positions[:, axis].mean()
            slab_idx   = int(np.clip(
                np.searchsorted(edges, perm_z, side='right') - 1,
                0, n_slabs - 1
            ))
            # The clip folds an out-of-range depth into the first or last slab
            # rather than dropping it. With --slab-range set wider than the box
            # (the usual case) nothing can land outside, so this never fires;
            # count it rather than test it per observable, because if the range
            # is ever set too narrow the contamination lands silently in the
            # edge bins — and those are the bulk-water bins that set the
            # reference scale for D. See the summary printed after the loop.
            if not edges[0] <= perm_z < edges[-1]:
                n_out_of_range += 1

            # ── Near-permeant proximity ranking ─────────────────────────────────
            # Resindices of the N lipids (by P-atom, 3-D PBC-aware distance)
            # closest to the permeant this frame, for each requested N.
            near_sets = {}
            if near_n and (run_order or run_structural) and len(permeant) > 0:
                p_all_near = u.select_atoms(
                    f"({lipid_sel}) and name {phosphorus_name}"
                )
                if len(p_all_near) > 0:
                    perm_com_near = _whole_centroid(permeant.positions,
                                                    u.dimensions[:3])
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
                        # Minimum image: the trajectories are not -pbc mol, so
                        # a bond whose two atoms wrapped to opposite sides is
                        # otherwise a box-length vector in the membrane plane,
                        # i.e. a spurious P2 of -0.5 (see _min_image).
                        vecs  = _min_image(all_pos[ia2] - all_pos[ia1], box_3)
                        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                        ok    = norms[:, 0] > 1e-6
                        p2_per_lip = np.full(len(ia1), np.nan)
                        if ok.any():
                            v_hat = vecs[ok] / norms[ok]
                            cos2  = (v_hat @ normal) ** 2
                            p2_per_lip[ok] = 0.5 * (3.0 * cos2 - 1.0)
                        bond_p2_per_lip_all[bi] = p2_per_lip

                        # Leaflet by lipid identity (its P atom), not by
                        # this bond's own base-atom z — see _leaflet_by_resindex.
                        up_b = _upper_mask_for(bond_residx[key], _upper_resix)
                        lf_masks = {
                            'both' : np.ones(len(ia1), dtype=bool),
                            'upper': up_b,
                            'lower': ~up_b,
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
                        # Leaflet by lipid identity, as for the per-bond masks.
                        first_bi = next(iter(bond_p2_per_lip_all))
                        up_ch = _upper_mask_for(
                            bond_residx[(chain, first_bi)], _upper_resix)
                        lf_masks_ch = {
                            'both' : np.ones(mat.shape[0], dtype=bool),
                            'upper': up_ch,
                            'lower': ~up_ch,
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
                    up_t = _upper_mask_for(tilt_residx[chain], _upper_resix)
                    lf_masks_tilt = {
                        'both' : np.ones(len(ib), dtype=bool),
                        'upper': up_t,
                        'lower': ~up_t,
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
                            theta, phi, _, theta_ind = mean_tilt_angle(
                                all_pos[it[combined]],
                                all_pos[ib[combined]],
                                normal,
                                leaflet=lf,
                                upper_mask=lf_masks_tilt['upper'][combined],
                                box=box_3,
                            )
                            if np.isfinite(theta):
                                theta_wsum[(sp, lf, chain)][slab_idx] += weight * theta
                                theta_wtot[(sp, lf, chain)][slab_idx] += weight
                            if np.isfinite(theta_ind):
                                theta_ind_wsum[(sp, lf, chain)][slab_idx] += weight * theta_ind
                                theta_ind_wtot[(sp, lf, chain)][slab_idx] += weight
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
                                theta_n, phi_n, _, theta_ind_n = mean_tilt_angle(
                                    all_pos[it[near_combined]],
                                    all_pos[ib[near_combined]],
                                    normal,
                                    leaflet='both',
                                    upper_mask=lf_masks_tilt['upper'][near_combined],
                                    box=box_3,
                                )
                                if np.isfinite(theta_n):
                                    near_theta_wsum[(N, sp, chain)][slab_idx] += weight * theta_n
                                    near_theta_wtot[(N, sp, chain)][slab_idx] += weight
                                if np.isfinite(theta_ind_n):
                                    near_theta_ind_wsum[(N, sp, chain)][slab_idx] += weight * theta_ind_n
                                    near_theta_ind_wtot[(N, sp, chain)][slab_idx] += weight
                                if np.isfinite(phi_n):
                                    near_phi_wsum[(N, sp, chain)][slab_idx]   += weight * phi_n
                                    near_phi_wtot[(N, sp, chain)][slab_idx]   += weight

            # ── STRUCTURAL group ──────────────────────────────────────────────
            if run_structural:
                for sp in _STRUCT_SPECIES:
                    td = _thickness_of(sp)

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
                    curv = _curvature_of(sp)
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
                box_xy  = u.dimensions[_lat_axes]
                raw_xy  = _whole_centroid(
                    perm_heavy.positions, u.dimensions[:3]
                )[_lat_axes]
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
                # Every map is stamped in the frame centred on this point, so
                # a permeant split across a lateral boundary would shift the
                # whole map, not one lipid's contribution — see _whole_centroid.
                perm_lat_sp = _whole_centroid(permeant.positions,
                                              u.dimensions[:3])[_lat_axes]

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

                    # Leaflet by lipid identity, exactly as the 1-D order
                    # block does it — not by this chain's own base-atom z.
                    up_sp = _upper_mask_for(
                        u.atoms.resindices[base_idx_sp], _upper_resix_sp)
                    lf_masks_sp = {
                        'both' : np.ones(len(base_idx_sp), dtype=bool),
                        'upper': up_sp,
                        'lower': ~up_sp,
                    }

                    bond_p2_per_lip = []
                    for bi in range(len(bonds)):
                        key = (chain, bi)
                        if key not in bond_idx:
                            continue
                        ia1, ia2 = bond_idx[key]
                        vecs  = _min_image(
                            all_pos_sp[ia2] - all_pos_sp[ia1], box_3)
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

                        # Leaflet by lipid identity; θ needs a per-lipid
                        # sign correction from it (see below).
                        up_tl = _upper_mask_for(tilt_residx[chain],
                                                _upper_resix_sp)
                        lf_masks_tilt = {
                            'both' : np.ones(len(ib), dtype=bool),
                            'upper': up_tl,
                            'lower': ~up_tl,
                        }

                        vt      = _min_image(
                            all_pos_sp[it] - all_pos_sp[ib], box_3)
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
                    td = _thickness_of(sp)

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
                    curv_sp = _curvature_of(sp)
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

    if n_out_of_range:
        print(f"  [worker] WARNING: {n_out_of_range} of {n_frames_seen} frames "
              f"had a permeant depth outside --slab-range "
              f"[{edges[0]:.1f}, {edges[-1]:.1f}] Å and were folded into the "
              f"edge slabs. Widen --slab-range: those frames contaminate the "
              f"first/last bin of every profile, D included.")

    # ── Diffusion post-processing ─────────────────────────────────────────────
    # Squared displacements are pooled here, not per-slab diffusion constants:
    # the sums and the window counts go into the weighted accumulators
    # separately, so the aggregate's wsum/wtot ratio is the pooled
    # <dz^2>/(2 n_dim tau) over every window of every path rather than a mean
    # of per-path means, and a path that visited a slab twice no longer speaks
    # as loudly as one that visited it a thousand times.
    # A short path supports the first few lags and not the rest; every k is
    # tried on its own, and one that does not fit simply contributes nothing
    # to its own columns. The fit then runs on whichever lags a slab has.
    if (run_diffusion or run_diffusion_xy) and not (_dt is not None and _dt > 0.0):
        print("  [worker] WARNING: no usable frame spacing could be read from "
              "this path's trajectories (a single-frame .xtc reports dt = 0), "
              "so its diffusion columns are left empty. Pass -dt to set it "
              "explicitly.")

    if run_diffusion and len(z_full) > lag_frames and _dt:
        z_arr  = np.asarray(z_full, dtype=float)
        si_arr = np.asarray(slab_idx_full, dtype=int)

        for k in _lag_ks:
            lag = k * lag_frames
            total, count = slab_binned_msd(z_arr, si_arr, n_slabs, lag)
            D_z_wsum[k] += weight * total / (2.0 * lag * _dt)
            D_z_wtot[k] += weight * count
            if k == 1:
                nwin_wsum += weight * count
                nwin_wtot += weight * (count > 0)

    if run_diffusion_xy and len(xy_full) > lag_frames and _dt:
        xy_arr = np.asarray(xy_full, dtype=float)
        si_arr = np.asarray(slab_idx_xy_full, dtype=int)

        for k in _lag_ks:
            lag = k * lag_frames
            total, count = slab_binned_msd(xy_arr, si_arr, n_slabs, lag)
            D_xy_wsum[k] += weight * total / (4.0 * lag * _dt)
            D_xy_wtot[k] += weight * count
            if k == 1:
                nwin_xy_wsum += weight * count
                nwin_xy_wtot += weight * (count > 0)

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
                    tilt_data[f'{pfx}{chain}_theta_ind_wsum'] = theta_ind_wsum[(sp, lf, chain)]
                    tilt_data[f'{pfx}{chain}_theta_ind_wtot'] = theta_ind_wtot[(sp, lf, chain)]
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
                    tilt_data[f'{pfx}{chain}_theta_ind_wsum'] = near_theta_ind_wsum[(N, sp, chain)]
                    tilt_data[f'{pfx}{chain}_theta_ind_wtot'] = near_theta_ind_wtot[(N, sp, chain)]
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
        diff_data = {'slab_center': centers}
        for k in _lag_ks:
            col = lag_column('D_z', k)
            diff_data[f'{col}_wsum'] = D_z_wsum[k]
            diff_data[f'{col}_wtot'] = D_z_wtot[k]
        diff_data['n_windows_wsum'] = nwin_wsum
        diff_data['n_windows_wtot'] = nwin_wtot
        output['diffusion'] = (pd.DataFrame(diff_data),)

    if run_diffusion_xy:
        diff_xy_data = {'slab_center': centers}
        for k in _lag_ks:
            col = lag_column('D_xy', k)
            diff_xy_data[f'{col}_wsum'] = D_xy_wsum[k]
            diff_xy_data[f'{col}_wtot'] = D_xy_wtot[k]
        diff_xy_data['n_windows_wsum'] = nwin_xy_wsum
        diff_xy_data['n_windows_wtot'] = nwin_xy_wtot
        output['diffusion_xy'] = (pd.DataFrame(diff_xy_data),)

    if run_spatial:
        output['spatial'] = (_spatial_acc, _grid_shape, _x_coords, _y_coords,
                             _depth_edges)

    return output


# ═════════════════════════════════════════════════════════════════════════════
# Per-path worker (unchanged except function calls are compatible)
# ═════════════════════════════════════════════════════════════════════════════

_GROUP_TABLE_KEYS = {
    'order'        : [(0, 'p2'),    (1, 'tilt')],
    'structural'   : [(0, 'thick'), (1, 'deform'), (2, 'water_z')],
    'diffusion'    : [(0, 'diff')],
    'diffusion_xy' : [(0, 'diff_xy')],
}


def process_single_path(
    path_number,
    overwrite,
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
    curvature_method    = 'rbf',
    curvature_wavelength = 20.0,
    curvature_surface   = 'species',
    dt                  = None,
    lag_frames          = 1,
    n_lags              = 6,
    grid_spacing        = 5.0,
    stamp_radius        = None,
    near_n              = (),
    spatial_extent      = None,
    n_depth_bins        = 10,
):
    out_file = path_output_file(path_number)
    needed   = _needed_groups(out_file, overwrite, near_n)

    if not needed:
        return (path_number, 'skipped', 'All output files already exist.')

    path_folder = f"../load/{path_number}/accepted/"
    if not os.path.exists(path_folder):
        return (path_number, 'skipped',
                f'Path folder does not exist: {path_folder}')

    try:
        # align_order_with_plan applies atleast_2d: a one-row order.txt
        # otherwise loads as 1-D, and order[0, 1] would then silently mean the
        # second *column* of the only row.
        order = np.loadtxt(f"../load/{path_number}/order.txt",
                           comments=("#", "@"))

        plan, g96_index = read_traj_plan(f"../load/{path_number}/traj.txt")
        order, problem = align_order_with_plan(order, plan, g96_index)
        if problem:
            return (path_number, 'skipped', problem)

        first, last = order[0, 1], order[-1, 1]

        ensemble = "plus" \
            if (lambda_minus_one < first < lambda_A) or \
               (lambda_minus_one < last  < lambda_A) \
            else "minus"

        reactive = get_reactive_paths(path_number, infretis_data_file, lambda_B)
        if reactive is None:
            return (path_number, 'skipped', 'Path not found in infretis_data.txt')

        first_frame, last_frame = remove_first_last_frames(
            ensemble, lambda_minus_one, lambda_A, first, last
        )

        if first_frame:
            plan = plan[1:]
        if last_frame:
            plan = plan[:-1]
        frame_plan = [(os.path.join(path_folder, name), indices)
                      for name, indices in traj_plan_segments(plan)]
        xtc_files = [xtc for xtc, _ in frame_plan]

        if not xtc_files:
            return (path_number, 'skipped', 'No xtc files found.')

        skipped_groups = set(OBSERVABLE_GROUPS.keys()) - needed
        print(
            f"[worker] Path {path_number}: "
            f"{len(xtc_files)} xtc file(s) — computing {sorted(needed)}"
            + (f", skipping {sorted(skipped_groups)}" if skipped_groups else "")
        )

        try:
            results = calculate_selected_slab_profiles(
                topology            = topol_file,
                xtc_files           = xtc_files,
                frame_plan          = frame_plan,
                permeant_resname    = permeant_resname,
                needed_groups       = needed,
                # Unweighted on purpose: the path weight is applied when the
                # paths are aggregated, so a renormalisation of the weights after
                # the simulations are extended needs a re-aggregation, not a rerun.
                # Every accumulator is linear in it (+= w*x, += w), so that is exact.
                weight              = 1.0,
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
                curvature_method    = curvature_method,
                curvature_wavelength = curvature_wavelength,
                curvature_surface   = curvature_surface,
                dt                  = dt,
                lag_frames          = lag_frames,
                n_lags              = n_lags,
                grid_spacing        = grid_spacing,
                stamp_radius        = stamp_radius,
                near_n              = near_n,
                spatial_extent      = spatial_extent,
                n_depth_bins        = n_depth_bins,
            )
        finally:
            # Done with this path's xtc files: drop the offset index MDAnalysis
            # saved next to each of them (see remove_offset_files).
            remove_offset_files(xtc_files)

        tables = {}
        for group, df_tuple in results.items():
            if group == 'spatial':
                continue
            for df_idx, table_key in _GROUP_TABLE_KEYS[group]:
                tables[table_key] = df_tuple[df_idx]

        spatial = None
        if 'spatial' in results:
            sp_acc, _sp_shape, sp_x, sp_y, sp_depth = results['spatial']
            spatial = (sp_acc, sp_x, sp_y, sp_depth)

        write_path_output(
            out_file, tables=tables, spatial=spatial,
            meta={'reactive': bool(reactive), 'ensemble': ensemble},
        )
        written = sorted(tables) + (['spatial'] if spatial is not None else [])
        print(f"  [worker] Path {path_number} -> {out_file.name} {written}")
        return (path_number, 'ok')

    except Exception:
        return (path_number, 'error', traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
# Parallel dispatcher (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def _note_legacy_outputs(out_dir):
    """Say so if the folder still holds the old per-path CSV layout.

    Those files are not read any more - every path is recomputed into its
    single .npz - so they are dead weight, and without a note it looks as
    though the earlier results were silently thrown away.
    """
    legacy = [f for pattern in ('*_p2.csv', '*_diffusion.csv', '*_spatial.npz')
              for f in Path(out_dir).glob(pattern)]
    if legacy:
        print(f"Note: {out_dir}/ holds {len(legacy)} file(s) in the old "
              "per-path CSV layout (e.g. {}). They are no longer read; every "
              "path is written to a single <n>.npz, and the old files can be "
              "deleted.".format(legacy[0].name))


def loop_over_paths_parallel(
    path_start,
    path_end,
    overwrite           = False,
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
    curvature_method    = 'rbf',
    curvature_wavelength = 20.0,
    curvature_surface   = 'species',
    dt                  = None,
    lag_frames          = 1,
    n_lags              = 6,
    grid_spacing        = 5.0,
    stamp_radius        = None,
    near_n              = (),
    spatial_extent      = None,
    n_depth_bins        = 10,
):
    check_curvature_options(curvature_method, curvature_surface,
                            curvature_wavelength)
    Path(ORDER_OUT).mkdir(parents=True, exist_ok=True)
    _note_legacy_outputs(ORDER_OUT)

    lambda_A, lambda_B, lambda_minus_one = read_toml('../infretis.toml')
    path_numbers = list(range(path_start, (path_end or path_start) + 1))

    print(f"Merged analysis: paths {path_start}–{path_end}, "
          f"{N_WORKERS} workers (single pass per path). Per-path results are "
          "stored unweighted; the path weights are applied at aggregation.")
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
                pn, overwrite, lambda_A, lambda_B, lambda_minus_one,
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
                curvature_method    = curvature_method,
                curvature_wavelength = curvature_wavelength,
                curvature_surface   = curvature_surface,
                dt                  = dt,
                lag_frames          = lag_frames,
                n_lags              = n_lags,
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
                         path_numbers=None, n_blocks=20, derived=None):
    """
    Pool per-path (wsum, wtot) accumulators into weighted means, and keep
    the information needed to judge whether those means mean anything.

    `derived` adds quantities that are *functions of several pooled means*
    rather than means themselves — the diffusion intercept fit is the case
    this exists for. Each entry is (name, fn), where fn takes
    (means, weights, index) with `means` and `weights` shaped
    (..., n_bins, n_obs) and `index` mapping a column name to its position,
    and returns either one (..., n_bins) array or a dict of them. It is
    applied to the full estimate and, identically, inside every
    leave-one-block-out replicate, so the jackknife error it produces is the
    error *of the derived quantity* — which is what is wanted, and what
    propagating the per-lag error bars afterwards could not give, since the
    lags share windows and are strongly correlated.

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
    axis_name = all_dfs[0].columns[0]
    return _pool_streamed(
        lambda: iter(all_dfs), len(all_dfs), all_centers, axis_name,
        wsum_cols, wtot_cols, path_numbers=path_numbers,
        n_blocks=n_blocks, derived=derived,
    )


def _pool_streamed(open_tables, n_tables, all_centers, axis_name,
                   wsum_cols, wtot_cols, path_numbers=None,
                   n_blocks=20, derived=None):
    """The pooling above, over an iterator instead of a list of DataFrames.

    Memory is what forces this shape. The estimator only ever needs each
    path's contribution *added into* a block accumulator, so holding every
    path's tables at once - which is what building the list costs - is pure
    overhead: at ~0.75 MB of float64 tables per path, 5000 paths is nearly
    4 GB before any arithmetic happens. Streaming holds one path at a time
    and the (K, n_bins, n_obs) blocks, which is tens of MB regardless of how
    many paths there are.

    `open_tables` is called to produce a fresh iterator of DataFrames in the
    same order as `path_numbers`; the axis grid and column list have to be
    known up front, which is what the cheap metadata pass in _scan_tables is
    for.
    """
    n = len(all_centers)
    C = len(wsum_cols)

    if path_numbers is None:
        path_numbers = np.arange(n_tables)
    block_of, K = _assign_blocks(path_numbers, n_blocks)

    # (K, n_bins, n_obs) block-wise accumulators, plus Σw² per bin for n_eff.
    block_wsum = np.zeros((K, n, C))
    block_wtot = np.zeros((K, n, C))
    wtot_sq    = np.zeros((n, C))

    for df, k in zip(open_tables(), block_of):
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

    result = {axis_name: all_centers}
    for ci, ws in enumerate(wsum_cols):
        col_name = ws.replace('_wsum', '')
        result[col_name]             = mean[:, ci]
        result[f'{col_name}_wtot']   = B[:, ci]
        result[f'{col_name}_neff']   = neff[:, ci]
        result[f'{col_name}_err']    = err[:, ci]

    # Derived quantities: computed from the pooled means, then jackknifed by
    # recomputing them on every replicate. B_k is passed as the per-replicate
    # weight so a fit weighted by window count stays consistent with the
    # means it is fitted to.
    index = {ws.replace('_wsum', ''): ci for ci, ws in enumerate(wsum_cols)}
    for name, fn in (derived or []):
        full = fn(mean, B, index)
        reps = fn(mu_k, B_k, index)
        if not isinstance(full, dict):
            full, reps = {name: full}, {name: reps}
        for sub_name, full_vals in full.items():
            rep_vals = reps[sub_name]
            with np.errstate(invalid='ignore', divide='ignore'):
                ok      = np.isfinite(rep_vals)
                n_ok    = ok.sum(axis=0)
                rep_bar = np.where(n_ok > 0,
                                   np.nansum(rep_vals, axis=0)
                                   / np.maximum(n_ok, 1),
                                   np.nan)
                ss_d    = np.nansum((rep_vals - rep_bar[None, :]) ** 2, axis=0)
                err_d   = np.where(
                    n_ok > 1,
                    np.sqrt((n_ok - 1) / np.maximum(n_ok, 1) * ss_d),
                    np.nan,
                )
            result[sub_name]           = full_vals
            result[f'{sub_name}_err']  = err_d

    return pd.DataFrame(result)


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


def _weighted(df, weight):
    """The table with every accumulator column multiplied by the path weight."""
    df = df.copy()
    columns = [c for c in df.columns[1:] if c.endswith(('_wsum', '_wtot'))]
    df[columns] = df[columns] * weight
    return df


def read_path_axis(path, key):
    """The axis column of one table, without decompressing its data block.

    np.load on an .npz is lazy, so naming one member reads only that member.
    That is what makes the metadata pass in _scan_tables cheap enough to be
    worth doing separately from the accumulation pass.
    """
    data = _open_current(path)
    if data is None:
        return None
    with data:
        name = f'{key}__axis'
        return np.asarray(data[name]) if name in data.files else None


def _scan_tables(path_start, path_end, weights, kinds=_TABLE_KEYS):
    """Metadata pass: which paths contribute, on what axis, with what columns.

    Reads only each file's column names and axis, never its data, so the
    accumulation pass that follows can stream one path at a time instead of
    holding them all. Returns {kind: (path_numbers, centers, wsum, wtot)},
    with the same skip accounting _collect_tables printed.
    """
    pns = {kind: [] for kind in kinds}
    cols = {kind: set() for kind in kinds}
    centers = {kind: set() for kind in kinds}
    axis_name = {kind: None for kind in kinds}
    no_weight = no_file = stale = 0
    missing = {kind: 0 for kind in kinds}

    for pn in range(path_start, path_end + 1):
        weight = weights.get(pn, 0.0)
        if not weight > 0.0:
            no_weight += 1
            continue
        path = path_output_file(pn)
        present = path_output_contents(path)
        if not present:
            version = path_output_version(path)
            if version is not None and version != PATH_FILE_VERSION:
                stale += 1
            else:
                no_file += 1
            continue
        for kind in kinds:
            if kind not in present:
                missing[kind] += 1
                continue
            names = read_path_columns(path, kind)
            axis = read_path_axis(path, kind)
            if not names or axis is None:
                missing[kind] += 1
                continue
            pns[kind].append(pn)
            cols[kind].update(names)
            centers[kind].update(np.round(axis, 6).tolist())
            axis_name[kind] = names[0]

    if no_weight or no_file:
        print(f"  [aggregate] skipped {no_weight} path(s) without a positive "
              f"weight and {no_file} without an output file.")
    if stale:
        print(f"  [aggregate] skipped {stale} path file(s) in an older format "
              "(their sums carry the weights they were computed with). A run "
              "recomputes them automatically, provided the path's trajectories "
              "are still in ../load.")
    partial = {kind: n for kind, n in missing.items() if n}
    if partial:
        print(f"  [aggregate] paths missing individual tables: {partial}")

    out = {}
    for kind in kinds:
        wsum = sorted(c for c in cols[kind] if c.endswith('_wsum'))
        wtot = [c.replace('_wsum', '_wtot') for c in wsum]
        out[kind] = (pns[kind], np.array(sorted(centers[kind])),
                     axis_name[kind], wsum, wtot)
    return out


def _stream_tables(path_numbers, kind, weights):
    """Yield one weighted table at a time, in `path_numbers` order."""
    for pn in path_numbers:
        tables = read_path_tables(path_output_file(pn), (kind,))
        yield _weighted(tables[kind], weights.get(pn, 0.0))


def _collect_tables(path_start, path_end, weights, kinds=_TABLE_KEYS):
    """Every path's tables with its current weight applied, one open per path.

    Returns ({kind: path_numbers}, {kind: DataFrames}); the path numbers are
    needed to build contiguous MC-time blocks for the jackknife in
    _pool_weighted_means. Paths without a positive weight contribute nothing
    and are left out rather than pooled as zeros.
    """
    pns = {kind: [] for kind in kinds}
    dfs = {kind: [] for kind in kinds}
    no_weight = no_file = stale = 0
    missing = {kind: 0 for kind in kinds}

    for pn in range(path_start, path_end + 1):
        weight = weights.get(pn, 0.0)
        if not weight > 0.0:
            no_weight += 1
            continue
        path = path_output_file(pn)
        tables = read_path_tables(path, kinds)
        if not tables:
            version = path_output_version(path)
            if version is not None and version != PATH_FILE_VERSION:
                stale += 1
            else:
                no_file += 1
            continue
        for kind in kinds:
            if kind in tables:
                dfs[kind].append(_weighted(tables[kind], weight))
                pns[kind].append(pn)
            else:
                missing[kind] += 1

    # One summary rather than a line per path: at 10^4 paths the per-path
    # messages drowned everything else the aggregation printed.
    if no_weight or no_file:
        print(f"  [aggregate] skipped {no_weight} path(s) without a positive "
              f"weight and {no_file} without an output file.")
    if stale:
        print(f"  [aggregate] skipped {stale} path file(s) in an older format "
              "(their sums carry the weights they were computed with). A run "
              "recomputes them automatically, provided the path's trajectories "
              "are still in ../load.")
    partial = {kind: n for kind, n in missing.items() if n}
    if partial:
        print(f"  [aggregate] paths missing individual tables: {partial}")
    return pns, dfs


def _lag_columns_present(columns, prefix):
    """The lag multipliers k whose D_app column is in `columns`, ascending."""
    ks = []
    for k in range(1, 1000):
        name = lag_column(prefix, k)
        if f'{name}_wsum' in columns or name in columns:
            ks.append(k)
        elif k > 2:
            break
    return ks


def parse_lag_selection(text, available):
    """Which lag multipliers enter the fit, from a "2-6" / "3-" / "1,3,5" spec.

    Choosing the fitting range is a separate decision from choosing what to
    accumulate, and it is made here rather than at compute time because every
    per-lag column is stored: scanning generously costs almost nothing, and
    the range that is actually affine only becomes visible afterwards, from
    the ladder and from `<prefix>_fit_resid`. Re-fitting is a re-aggregation,
    not a recompute.

    `available` is the lags the table holds; the selection is intersected with
    it, so asking for lags that were never accumulated is not an error.
    """
    avail = sorted(available)
    if text is None or not str(text).strip():
        return avail

    spec = str(text).strip()
    lo_hi = spec.split('-')
    try:
        if len(lo_hi) == 2 and ',' not in spec:
            lo_s, hi_s = (part.strip() for part in lo_hi)
            lo = int(lo_s) if lo_s else (min(avail) if avail else 1)
            hi = int(hi_s) if hi_s else (max(avail) if avail else 1)
            chosen = [k for k in avail if lo <= k <= hi]
        else:
            wanted = {int(p) for p in spec.replace(',', ' ').split()}
            chosen = [k for k in avail if k in wanted]
    except ValueError:
        raise ValueError(
            f"-fit-lags: cannot read {text!r}; expected a range like '2-6', "
            "an open range like '3-', or a list like '1,3,5'."
        ) from None

    if len(chosen) < 2:
        raise ValueError(
            f"-fit-lags {text!r} leaves {len(chosen)} of the available lags "
            f"{avail}, and an intercept needs at least two. The fit is an "
            "extrapolation to 1/tau -> 0, not a reading at one lag."
        )
    return chosen


def _diffusion_fit_spec(columns, prefix, fit_lags=None):
    """The `derived` entry that fits D_app(tau) -> D for one diffusion table.

    Returns [] when the table carries fewer than two lags, i.e. was written
    before the lag scan existed — those files still aggregate, just without
    the fitted column.
    """
    ks = _lag_columns_present(columns, prefix)
    if len(ks) < 2:
        return []
    ks = parse_lag_selection(fit_lags, ks)
    names = [lag_column(prefix, k) for k in ks]

    def _fit(means, wts, index, _ks=ks, _names=names, _prefix=prefix):
        cols   = [index[n] for n in _names if n in index]
        used_k = [k for k, n in zip(_ks, _names) if n in index]
        if len(cols) < 2:
            nan = np.full(means.shape[:-1], np.nan)
            return {f'{_prefix}_fit': nan, f'{_prefix}_fit_c': nan.copy(),
                    f'{_prefix}_fit_resid': nan.copy()}
        d_app = means[..., cols]
        # Window counts as the per-lag weight: the long lags have fewer
        # windows and deserve less pull on the extrapolation.
        w     = wts[..., cols]
        d_fit, slope, resid = fit_diffusion_intercept(
            d_app, weights=w, ks=np.asarray(used_k, dtype=float)
        )
        # `slope` is c / (2 n_dim * lag * dt); reported as-is because dt is
        # not available here. It is a diagnostic, not an input to anything.
        return {f'{_prefix}_fit': d_fit,
                f'{_prefix}_fit_c': slope,
                f'{_prefix}_fit_resid': resid}

    return [(f'{prefix}_fit', _fit)]


def _attach_fit_support(df, prefix, fit_lags=None):
    """Give the fitted column the wtot/neff companions the maskers expect.

    The fit is no better supported than its worst-supported lag, so n_eff is
    the minimum over the lags that entered it; wtot is taken from k = 1, the
    lag with the most windows, purely so the column exists.
    """
    ks = _lag_columns_present(df.columns, prefix)
    fit = f'{prefix}_fit'
    if fit not in df.columns or len(ks) < 2:
        return df
    ks = parse_lag_selection(fit_lags, ks)
    neff_cols = [f'{lag_column(prefix, k)}_neff' for k in ks]
    neff_cols = [c for c in neff_cols if c in df.columns]
    if neff_cols:
        df[f'{fit}_neff'] = df[neff_cols].min(axis=1)
    if f'{prefix}_wtot' in df.columns:
        df[f'{fit}_wtot'] = df[f'{prefix}_wtot']
    return df


def aggregate_all_results(path_start, path_end, weights, n_blocks=20,
                          fit_lags=None):
    # Metadata first, then one streaming pass per table kind: the tables are
    # never all in memory at once. See _pool_streamed.
    scan = _scan_tables(path_start, path_end, weights)
    _FIT_PREFIX = {'diff': 'D_z', 'diff_xy': 'D_xy'}
    results = {}
    reported = False
    for kind in _TABLE_KEYS:
        pn_list, centers, axis_name, ws, wt = scan[kind]
        if not pn_list:
            raise RuntimeError(f"No '{kind}' tables found to aggregate.")
        derived = (_diffusion_fit_spec(set(ws), _FIT_PREFIX[kind], fit_lags)
                   if kind in _FIT_PREFIX else [])
        if derived and not reported:
            used = parse_lag_selection(
                fit_lags, _lag_columns_present(set(ws), _FIT_PREFIX[kind]))
            avail = _lag_columns_present(set(ws), _FIT_PREFIX[kind])
            note = '' if len(used) == len(avail) else f" of {len(avail)} scanned"
            print(f"  [aggregate] D fitted over lags k = "
                  f"{', '.join(str(k) for k in used)}{note}"
                  f" (times -lag-frames)")
            reported = True
        results[kind] = _pool_streamed(
            lambda pl=pn_list, k=kind: _stream_tables(pl, k, weights),
            len(pn_list), centers, axis_name, ws, wt,
            path_numbers=pn_list, n_blocks=n_blocks, derived=derived,
        )
        if kind in _FIT_PREFIX:
            results[kind] = _attach_fit_support(results[kind],
                                                _FIT_PREFIX[kind], fit_lags)
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

    # Three panels, because the two tilt angles are different quantities and
    # overlaying them on one axis invites reading the gap as noise: the
    # collective angle is the leaflet's net lean, the individual one is the
    # typical chain's tilt, and the difference between them is the splay.
    fig, axes = plt.subplots(1, 3, figsize=(19, 5), sharey=False)

    for ax, angle, ylabel in zip(
        axes, ['theta', 'theta_ind', 'phi'],
        ['Collective tilt θ (°)', 'Mean chain tilt θ (°)', 'Azimuthal φ (°)']
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
        'z-displacement (Å)', 'Collective tilt θ (°)',
        f'Near-permeant vs. bulk collective tilt — chain {chain}',
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


def _plot_diffusion(agg, prefix, msd_label, title, ylabel,
                    min_neff=MIN_NEFF, output_path=None):
    """Shared body of the two diffusion-profile plots.

    The lag ladder is drawn as thin lines shading from light to dark, and the
    fitted intercept over it in black. Reading it: the ladder collapsing onto
    the fit is a slab whose motion is diffusive at this frame spacing; a
    ladder that keeps descending with lag is one that is not, and there the
    single-lag D is an overestimate and only the fit is meaningful.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    x  = agg['slab_center']
    ks = _lag_columns_present(agg.columns, prefix)

    shades = plt.cm.Blues(np.linspace(0.35, 0.9, max(len(ks), 1)))
    for k, color in zip(ks, shades):
        col = lag_column(prefix, k)
        label = (f'{msd_label} at τ' if k == 1 else f'at {k}τ')
        _plot_series(ax1, x, agg, col, min_neff=min_neff,
                     band=False, lw=1.2, color=color, label=label)

    fit = f'{prefix}_fit'
    if fit in agg.columns:
        _plot_series(ax1, x, agg, fit, min_neff=min_neff, lw=2.4,
                     color='black',
                     label='fit: MSD = c + 2nDτ, extrapolated to 1/τ → 0')

    ax1.set_ylabel(ylabel, fontsize=13)
    ax1.set_yscale('log')
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_title(title, fontsize=13)

    if 'n_windows' in agg.columns:
        width = (x.iloc[1] - x.iloc[0]) * 0.85
        ax2.bar(x, agg['n_windows'], width=width,
                color='steelblue', alpha=0.7)
    ax2.set_xlabel('z-displacement (Å)', fontsize=13)
    ax2.set_ylabel('# windows / path', fontsize=12)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    return fig


def plot_diffusion_profile(diff_agg, min_neff=MIN_NEFF, output_path=None):
    return _plot_diffusion(
        diff_agg, 'D_z', '⟨Δz²⟩/2τ',
        'Local diffusion coefficient along bilayer normal',
        'D(z)  (Å² ps⁻¹)',
        min_neff=min_neff, output_path=output_path,
    )


def plot_diffusion_xy_profile(diff_xy_agg, min_neff=MIN_NEFF, output_path=None):
    return _plot_diffusion(
        diff_xy_agg, 'D_xy', '⟨Δr²⟩/4τ',
        'Lateral diffusion coefficient along bilayer normal',
        'D_xy(z)  (Å² ps⁻¹)',
        min_neff=min_neff, output_path=output_path,
    )


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
         ('plasma',  f'Chain tilt θ {chain} (°)'))),
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
    weights: Annotated[str, typer.Option("-weights", "--weights", help="WHAM path weights, applied when the paths are aggregated (-plot). Computing the per-path files does not need them: those store unweighted sums, so after the weights change a rerun with -plot re-aggregates without recomputing any path", rich_help_panel=panels.INPUT)] = 'path_weights.txt',
    data: Annotated[str, typer.Option("-data", "--data", help="Path to infretis_data.txt.", rich_help_panel=panels.INPUT)] = ...,
    permeant: Annotated[str, typer.Option("-permeant", "--permeant", help="", rich_help_panel=panels.INPUT)] = 'ORP',
    leaflets_opt: Annotated[str, typer.Option("-leaflets", "--leaflets", help="Comma-separated leaflets to include in plots, e.g. 'both,upper,lower'.", rich_help_panel=panels.INPUT)] = "both,upper,lower",

    # ── Dataset construction ──────────────────────────────────
    start: Annotated[int, typer.Option("-start", "--start", help="", rich_help_panel=panels.DATASET)] = ...,
    end: Annotated[Optional[int], typer.Option("-end", "--end", help="", rich_help_panel=panels.DATASET)] = None,
    lag_frames: Annotated[int, typer.Option("-lag-frames", "--lag-frames", help="Base lag in frames for the local diffusion estimate, D = <dz^2>/(2 tau) over the windows starting in each slab. The scan runs at 1..N times this, so 1 is almost always right; raise it only if consecutive frames are so close together that even the longest lag in the scan is still pre-diffusive.", rich_help_panel=panels.DATASET)] = 1,
    fit_lags: Annotated[Optional[str], typer.Option("-fit-lags", "--fit-lags", help="Which of the scanned lags the D_z_fit / D_xy_fit extrapolation uses: a range '2-6', an open range '3-', or a list '1,3,5'. Default is all of them. This is an aggregation-time choice, so changing it re-aggregates but does not recompute. Narrow the top end where <obs>_fit_resid is large at steep D gradients (long windows straddling them smear the contrast); raise the bottom end where the residual is large even in flat regions (the shortest lags are not yet diffusive).", rich_help_panel=panels.DATASET)] = None,
    n_lags: Annotated[int, typer.Option("-n-lags", "--n-lags", help="How many lags to scan, k = 1..N times -lag-frames. D_app(tau) equals D only inside the diffusive window, so instead of trusting one lag the aggregate fits MSD = c + 2nD*tau across the scan and reports the slope as D_z_fit / D_xy_fit; the intercept c absorbs cage rattling. Per-lag values stay available as D_z, D_z_lag2, D_z_lag3 ... The upper lag must still keep sqrt(2 D tau) below the slab width, so more is not always better. 2 reduces the fit to the old two-point check.", rich_help_panel=panels.DATASET)] = 6,
    dt: Annotated[Optional[float], typer.Option("-dt", "--dt", help="Time between consecutive frames, in ps. Only the diffusion uses it, as the divisor in D = <dz^2>/(2 n tau), so it scales D and hence P linearly. Left unset it is read from the trajectory, which is right when the segments are long but reports 0 for a single-frame .xtc; setting it explicitly is recommended.", rich_help_panel=panels.DATASET)] = None,

    # ── CV corrections: representation ────────────────────────
    slab_width: Annotated[float, typer.Option("-slab-width", "--slab-width", help="", rich_help_panel=panels.REPR)] = 1.0,
    bilayer_normal: Annotated[str, typer.Option("-bilayer-normal", "--bilayer-normal", help="", rich_help_panel=panels.REPR)] = 'z',
    slab_range_opt: Annotated[str, typer.Option("-slab-range", "--slab-range", help="Permeant-depth range in Angstrom as 'min,max', on the membrane_z - permeant_z axis (the order-parameter axis, not lab z). Fixed rather than derived so every path and every simulation share one grid - the per-path CSVs are pooled bin by bin.", rich_help_panel=panels.REPR)] = '-40,40',
    r_max: Annotated[float, typer.Option("-r-max", "--r-max", help="", rich_help_panel=panels.REPR)] = 30.0,
    n_radial_bins: Annotated[int, typer.Option("-n-radial-bins", "--n-radial-bins", help="", rich_help_panel=panels.REPR)] = 15,
    n_z_bins: Annotated[int, typer.Option("-n-z-bins", "--n-z-bins", help="Number of bins for the water z-density profile, spanning --water-z-range.", rich_help_panel=panels.REPR)] = 60,
    water_z_range_opt: Annotated[str, typer.Option("-water-z-range", "--water-z-range", help="Window in Angstrom as 'min,max' for the water z-density profile, measured relative to the bilayer midplane. Fixed rather than box-derived so the profile is poolable across paths and comparable between simulations. Must not exceed the box height; ±40 Å suits a ~100 Å box.", rich_help_panel=panels.REPR)] = '-40,40',
    curvature_smoothing: Annotated[float, typer.Option("-curvature-smoothing", "--curvature-smoothing", help="RBF smoothing for -curvature-method rbf. At 0 the surface passes exactly through every phosphate, so single-lipid protrusions dominate H and K. Ignored by the fourier method.", rich_help_panel=panels.REPR)] = 0.0,
    curvature_method: Annotated[str, typer.Option("-curvature-method", "--curvature-method", help="How each leaflet's phosphate surface is built for H and K. 'rbf': an interpolating RBF through every phosphate (the original; noise-dominated at -curvature-smoothing 0). 'fourier': a least-squares periodic Fourier fit keeping only wavelengths >= -curvature-wavelength, with analytic derivatives. Changing it changes the stored curvature, so rerun the structural and spatial groups with -overwrite rather than mixing methods across paths.", rich_help_panel=panels.REPR)] = 'rbf',
    curvature_wavelength: Annotated[float, typer.Option("-curvature-wavelength", "--curvature-wavelength", help="Shortest wavelength (A) the fourier curvature keeps; shorter undulations count as single-lipid motion and are filtered out. Needs at least 1 + 2M phosphates per leaflet for M kept modes (~37 coefficients at 20 A in a 68 A box). Ignored by the rbf method.", rich_help_panel=panels.REPR)] = 20.0,
    curvature_surface: Annotated[str, typer.Option("-curvature-surface", "--curvature-surface", help="Which surface the per-species (DOPC, POPC) curvature is read from. 'species': a separate fit through only that species' phosphates (the original; sparse for the minority lipid). 'all': the one surface fitted to all lipids, evaluated at that species' phosphates. The mixed result is the same either way.", rich_help_panel=panels.REPR)] = 'species',
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
    min_neff: Annotated[float, typer.Option("-min-neff", "--min-neff", help="Blank out profile bins supported by fewer than this effective number of paths (Kish n_eff). 0 disables masking.", rich_help_panel=panels.MODEL)] = MIN_NEFF,

    # ── Output ────────────────────────────────────────────────
    overwrite: Annotated[bool, typer.Option("-O", "-overwrite", "--overwrite", help="", rich_help_panel=panels.OUTPUT)] = False,
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
    global N_WORKERS, leaflets, near_n, slab_range

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
        curvature_method=curvature_method,
        curvature_wavelength=curvature_wavelength,
        curvature_surface=curvature_surface,
        plot=plot,
        lag_frames=lag_frames,
        n_lags=n_lags,
        fit_lags=fit_lags,
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

    # The options above are the interface; there is no second argparse layer.
    # (There used to be: this function built `args` from the Typer options and
    # then overwrote it with `parser.parse_args()` of sys.argv, so every option
    # was silently discarded and the command could not run at all.)


    slab_range = tuple(args.slab_range) if args.slab_range else None
    end        = args.end if args.end is not None else args.start
    leaflets   = tuple(args.leaflets)
    _species   = ('mixed', 'DOPC', 'POPC')
    N_WORKERS = args.workers

    loop_over_paths_parallel(
        path_start          = args.start,
        path_end            = end,
        overwrite           = args.overwrite,
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
        curvature_method    = args.curvature_method,
        curvature_wavelength = args.curvature_wavelength,
        curvature_surface   = args.curvature_surface,
        dt                  = args.dt,
        lag_frames          = args.lag_frames,
        n_lags              = args.n_lags,
        # fit_lags is not passed to the workers: it selects among columns that
        # are already in the per-path files, so it is consumed at aggregation.
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
                                                n_blocks=args.n_blocks,
                                                fit_lags=args.fit_lags)

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
