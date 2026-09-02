import os

os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import re
import traceback
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import tomli
import typer

from . import panels

# MDAnalysis is a required dependency, but only this module uses it and it costs
# ~0.5s to import, so it is bound on first use rather than at import time. That
# keeps `chiroflux --help` and every command that does not touch trajectories
# free of it. _require_mdanalysis() populates the three globals below.
mda = None
capped_distance = None
mda_rmsd = None

_MDANALYSIS_HINT = (
    "chiroflux generate-cvs needs MDAnalysis, which is a required dependency "
    "of chiroflux - if it is missing, the install is incomplete. Reinstall "
    "with:\n\n    pip install -e .\n"
)


def _require_mdanalysis():
    """Bind MDAnalysis into module scope on first use."""
    global mda, capped_distance, mda_rmsd
    if mda is not None:
        return
    try:
        import MDAnalysis as _mda
        from MDAnalysis.analysis.rms import rmsd as _rmsd
        from MDAnalysis.lib.distances import capped_distance as _cd
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(f"{exc}\n\n{_MDANALYSIS_HINT}") from exc
    mda, capped_distance, mda_rmsd = _mda, _cd, _rmsd

#######################################################################################
### UGENT
### Received from SS on 20-02-2026
### Adapted to match proline system by TV
###
### MDAnalysis rewrite (this file)
### -------------------------------
### The original `parallel_system_file_gen_archive.py` computes every CV by shelling
### out to a GROMACS CLI tool (gmx traj / rms / gyrate / angle / select / hbond /
### pairdist) once PER FEATURE PER TRAJECTORY SEGMENT, writing an intermediate .xvg
### text file that is then re-read with np.loadtxt.  For ~90 features and several
### xtc segments per path that is ~90x more subprocess launches and full-trajectory
### re-reads than necessary, plus ~90 intermediate files per path that then have to
### be archived into a per-path tar and (for the bulk coordinate dumps) purged again
### -- all of which exists purely to work around gmx's file-in/file-out interface.
###
### This version loads each xtc segment with MDAnalysis ONCE, walks its frames a
### single time, and computes every raw quantity (coordinates, distances, box) it
### needs directly in memory. There are no intermediate .xvg files, no per-path Files/
### staging directory, and no tar archive -- so the entire cache-invalidation /
### archive-rewrite machinery from the original (delete_from_tar, tar_member_names,
### archive_path, migrate_flat_xvg_files, purge_dumps, PURGED_TAGS, ...) is gone
### along with the disk and inode footprint it existed to manage.  The only cache
### left is coarse: `ML/{path}.txt` existing already skips a path unless --overwrite
### is given.  --recalculate / --only / --keep-dumps are gone with it: they existed
### to make per-feature recomputation cheap against a tar archive, which no longer
### exists -- everything for a path is now computed together in one (fast) pass.
###
### All the pure-numpy CV math (compute_angles, compute_ring_plane_angle*, the
### chirality-sensitive pseudoscalars, ACSF, tetrahedrality, Cremer-Pople, ...) is
### copied verbatim from the original: it only ever operated on coordinate arrays,
### never on gmx itself, so it needs no changes.
###
### VALIDATION CAVEAT: atom groups were cross-checked against the real topol.tpr /
### .ndx files (atom names, indices, the "PRO_HB.ndx" donor/acceptor chemistry) but
### there is no local ../load/<path>/accepted/ trajectory data to run this against
### the original gmx-based script and diff the resulting ML/*.txt column-by-column.
### Before pointing production at this file, run both scripts on a handful of the
### same paths and compare outputs -- pay particular attention to PRO_PO_hb /
### PRO_CO_hb / PRO_N_hb (hand-rolled replica of gmx hbond's donor/acceptor distance
### + angle criterion) and PRO_dih_chiral / PRO_dih_OH (hand-rolled dihedral --
### standard IUPAC sign convention, should match gmx angle -type dihedral, but was
### not checked bit-for-bit against a real gmx run).
#######################################################################################

ML = "ML_mda"
ERROR_LOG = "errors.log"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analysis script for the PRO system (MDAnalysis backend)."
    )
    parser.add_argument("-s", "--start", type=int, default=None,
                        help="Start from this path number.")
    parser.add_argument("-e", "--end", type=int, default=None,
                        help="End with this path number.")
    parser.add_argument("-d", "--data", type=str, required=True,
                        help="Path to the infretis_data.txt file")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Recompute a path even if ML/{path}.txt already exists.")
    parser.add_argument("-w", "--workers", type=int, default=64)
    return parser.parse_args()


def extract_sorted_traj_names(trj_path):
    filenames = []
    directions = []
    seen = set()
    g96_index = None

    with open(trj_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            columns = line.split()
            step = columns[0]
            filename = columns[1]
            direction = columns[3]

            if filename not in seen:
                if filename.endswith(".trr"):
                    filenames.append(filename[:-4] + ".xtc")
                elif filename.endswith(".g96"):
                    g96_index = step
                else:
                    filenames.append(filename)
                seen.add(filename)
                directions.append(direction)

    print(f"g96_index: {g96_index}")
    return filenames, directions, g96_index


def get_reactive_paths(path_number, infretis_data_file, lambda_B):
    first_cols = []
    third_cols = []

    with open(infretis_data_file, "r") as file:
        for line in file:
            if line.startswith("#") or not line.strip():
                continue
            columns = line.split()
            first_cols.append(int(columns[0]))
            third_cols.append(float(columns[2]))

    first_cols = np.array(first_cols)
    third_cols = np.array(third_cols)
    match_indices = np.where(first_cols == path_number)[0]

    if match_indices.size > 0:
        return third_cols[match_indices[0]] > lambda_B
    else:
        return None


def read_toml(toml_file):
    if os.path.isfile(toml_file):
        with open(toml_file, mode="rb") as read:
            config = tomli.load(read)
    else:
        print("No toml file, exit.")
        return
    interfaces = config["simulation"]["interfaces"]
    lambda_A = interfaces[0]
    lambda_B = interfaces[-1]
    if "tis_set" in config["simulation"]:
        lambda_minus_one = config["simulation"]["tis_set"].get("lambda_minus_one")
    else:
        lambda_minus_one = None
    return lambda_A, lambda_B, lambda_minus_one


# ── .ndx parsing ──────────────────────────────────────────────────────────────
# The leaflet-split lipid marker groups (upper/lower by a z-cutoff frozen at the
# time these files were generated -- see the "z_>_4.06" / "z_<_4.06" group names)
# cannot be reconstructed from a live selection without knowing that historical
# cutoff, so they are read verbatim from the same .ndx files the original script
# used, in the same file order gmx numbers them (0-based here, gmx is 1-based).

def parse_ndx(path):
    """Parse a GROMACS .ndx file into an OrderedDict of name -> 0-based int array."""
    groups = OrderedDict()
    cur = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"\[\s*(.+?)\s*\]", line)
            if m:
                cur = m.group(1)
                groups[cur] = []
            elif cur is not None:
                groups[cur].extend(int(tok) - 1 for tok in line.split())
    return OrderedDict((k, np.asarray(v, dtype=int)) for k, v in groups.items())


def ndx_group_list(path):
    """Groups in file order -- i.e. gmx's positional group numbering (0, 1, 2, ...)."""
    return list(parse_ndx(path).values())


# ── Pure-numpy CV math (verbatim from the original script; unchanged) ────────

def compute_angles(coords, box):
    """
    Compute angle of vector between two atoms with PBC correction.

    Parameters
    ----------
    coords : (n_frames, 6) or (n_frames, 2, 3)
        Flattened or structured coordinates of 2 atoms per frame.
    box : scalar, (3,), or (n_frames, 3)
        Box lengths in x, y, z.

    Returns
    -------
    angles in degrees
    """
    coords = np.asarray(coords)
    n_frames = coords.shape[0]
    coords = coords.reshape(n_frames, 2, 3)
    vecs = coords[:, 1, :] - coords[:, 0, :]

    box = np.asarray(box)
    if box.ndim == 0:
        box = np.full((n_frames, 3), box)
    elif box.ndim == 1:
        if box.shape[0] == 3:
            box = np.tile(box, (n_frames, 1))
        else:
            raise ValueError("box must be scalar or shape (3,) or (n_frames, 3)")
    elif box.shape != (n_frames, 3):
        raise ValueError("box must have shape (n_frames, 3)")

    vecs = vecs - box * np.round(vecs / box)

    norms = np.linalg.norm(vecs, axis=1)
    safe = norms > 1e-12
    vecs[safe] /= norms[safe, None]

    cos_theta = np.clip(vecs[:, 2], -1.0, 1.0)
    angles = np.zeros(n_frames)
    angles[safe] = np.degrees(np.arccos(cos_theta[safe]))

    return angles


def compute_ring_plane_angle_continuous(coords, box):
    """
    Ring plane angle with:
    - full 3D PBC
    - consistent unwrapping
    - PCA-based normal
    - continuous sign tracking
    """
    coords = np.asarray(coords)
    n_frames = coords.shape[0]
    ring = coords.reshape(n_frames, 5, 3)

    box = np.asarray(box)
    if box.ndim == 0:
        box = np.full((n_frames, 3), box)
    elif box.ndim == 1:
        if box.shape[0] == 3:
            box = np.tile(box, (n_frames, 1))
        else:
            raise ValueError("box must be scalar or shape (3,) or (n_frames, 3)")
    elif box.shape != (n_frames, 3):
        raise ValueError("box must have shape (n_frames, 3)")

    def min_image(vec):
        return vec - box[:, None, :] * np.round(vec / box[:, None, :])

    ref = ring[:, 0:1, :]
    ring_unwrapped = ref + min_image(ring - ref)

    centroid = np.mean(ring_unwrapped, axis=1)

    ring_centered = ring_unwrapped - centroid[:, None, :]
    cov = np.einsum('fni,fnj->fij', ring_centered, ring_centered) / 5.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0]

    norms = np.linalg.norm(normals, axis=1)
    safe = norms > 1e-12
    normals[safe] /= norms[safe, None]

    for i in range(1, n_frames):
        if np.dot(normals[i], normals[i - 1]) < 0:
            normals[i] *= -1.0

    cos_theta = np.clip(normals[:, 2], -1.0, 1.0)
    angles = np.zeros(n_frames)
    angles[safe] = np.degrees(np.arccos(cos_theta[safe]))

    return angles


def compute_ring_plane_angle(coords, box):
    """
    Ring plane angle with full 3D PBC.

    Parameters
    ----------
    coords : (n_frames, 15) or (n_frames, 5, 3)
    box : scalar, (3,), or (n_frames, 3)
    """
    coords = np.asarray(coords)
    n_frames = coords.shape[0]
    ring = coords.reshape(n_frames, 5, 3)

    box = np.asarray(box)
    if box.ndim == 0:
        box = np.full((n_frames, 3), box)
    elif box.ndim == 1:
        if box.shape[0] == 3:
            box = np.tile(box, (n_frames, 1))
        else:
            raise ValueError("box must be scalar or shape (3,) or (n_frames, 3)")
    elif box.shape != (n_frames, 3):
        raise ValueError("box must have shape (n_frames, 3)")

    def min_image(vec):
        return vec - box[:, None, :] * np.round(vec / box[:, None, :])

    ref = ring[:, 0:1, :]
    ring_unwrapped = ref + min_image(ring - ref)

    centroid = np.mean(ring_unwrapped, axis=1)

    ring_centered = ring_unwrapped - centroid[:, None, :]
    cov = np.einsum('fni,fnj->fij', ring_centered, ring_centered) / 5.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0]

    norms = np.linalg.norm(normals, axis=1)
    safe = norms > 1e-12
    normals[safe] /= norms[safe, None]

    cos_theta = np.clip(normals[:, 2], -1.0, 1.0)
    angles = np.zeros(n_frames)
    angles[safe] = np.degrees(np.arccos(cos_theta[safe]))

    return angles


def compute_ring_plane_angle_chiral(coords_6, box):
    """
    Compute the angle between a 5-membered ring's normal vector and the
    membrane normal (z-axis), with the normal oriented consistently using
    a chiral substituent as a reference.

    Parameters
    ----------
    coords_6 : np.ndarray, shape (n_frames, 18)
        Flattened XYZ coordinates of 6 atoms per frame.
        Atom order: [ substituent | ring_1 | ring_2 | ring_3 | ring_4 | ring_5 ]
    box : float or array-like

    Returns
    -------
    np.ndarray, shape (n_frames,)
        Angle in degrees in [0, 180].
    """
    coords_6 = np.asarray(coords_6)
    n_frames = coords_6.shape[0]

    substituent = coords_6[:, 0:3]
    ring = coords_6[:, 3:18].reshape(n_frames, 5, 3)

    box = np.asarray(box)
    if box.ndim == 0:
        box = np.full((n_frames, 3), box)
    elif box.ndim == 1:
        if box.shape[0] == 3:
            box = np.tile(box, (n_frames, 1))
        else:
            raise ValueError("box must be scalar, (3,), or (n_frames, 3)")
    elif box.shape != (n_frames, 3):
        raise ValueError("box must have shape (n_frames, 3)")

    def min_image(vec):
        return vec - box[:, None, :] * np.round(vec / box[:, None, :])

    ref = ring[:, 0:1, :]

    ring_unwrapped = ref + min_image(ring - ref)
    sub_unwrapped = ref[:, 0, :] + (
        substituent - ref[:, 0, :]
        - box * np.round((substituent - ref[:, 0, :]) / box)
    )

    centroid = np.mean(ring_unwrapped, axis=1)

    ring_centered = ring_unwrapped - centroid[:, None, :]
    cov = np.einsum('fni,fnj->fij', ring_centered, ring_centered) / 5.0

    eigvals, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0]

    norms = np.linalg.norm(normals, axis=1)
    safe = norms > 1e-12
    normals[safe] /= norms[safe, None]

    to_sub = sub_unwrapped - centroid
    sub_norm = np.linalg.norm(to_sub, axis=1)
    safe2 = sub_norm > 1e-12
    to_sub[safe2] /= sub_norm[safe2, None]
    flip = np.einsum('fi,fi->f', normals, to_sub) < 0
    normals[flip] *= -1

    cos_theta = np.clip(normals[:, 2], -1.0, 1.0)

    angles = np.zeros(n_frames)
    angles[safe] = np.degrees(np.arccos(cos_theta[safe]))

    return angles


def compute_coordination_number(distances):
    d0, n, m = 0.5, 6, 12
    r_ratio = distances / d0
    r_over_d0_n = r_ratio ** n
    r_over_d0_m = r_ratio ** m
    numerator = 1 - r_over_d0_n
    denominator = 1 - r_over_d0_m
    singular = np.abs(denominator) < 1e-10
    with np.errstate(divide='ignore', invalid='ignore'):
        cn_values = np.where(singular, n / m, numerator / denominator)
    return np.sum(cn_values, axis=1)


def compute_vec_cent_angles(coords, box, pbc=(True, True, True)):
    """
    Computes angle between (center -> substituent centroid) vector and z-axis.

    Parameters
    ----------
    coords : (n_frames, 12) array
        Flattened coordinates: 1 center atom + 3 substituents
    box : array-like
    pbc : tuple of bool
    """
    coords = np.asarray(coords)
    n_frames = coords.shape[0]
    center = coords[:, 0:3]
    subs = coords[:, 3:12].reshape(n_frames, 3, 3)

    box = np.asarray(box)
    if box.ndim == 0:
        box = np.full((n_frames, 3), box)
    elif box.ndim == 1:
        if box.shape[0] == 3:
            box = np.tile(box, (n_frames, 1))
        else:
            raise ValueError("box must be scalar, (3,), or (n_frames, 3)")
    elif box.shape != (n_frames, 3):
        raise ValueError("box must have shape (n_frames, 3)")

    pbc = np.asarray(pbc, dtype=bool)
    if pbc.shape != (3,):
        raise ValueError("pbc must be a tuple of length 3")

    # Unwrap each (bonded) substituent relative to the center atom BEFORE
    # averaging -- these trajectories are not pre-processed with -pbc mol, so
    # a naive mean of raw, possibly independently-wrapped positions would be
    # wrong exactly when the molecule straddles a periodic boundary.
    rel = subs - center[:, None, :]
    rel_shift = np.zeros_like(rel)
    rel_shift[:, :, pbc] = box[:, None, pbc] * np.round(rel[:, :, pbc] / box[:, None, pbc])
    subs_unwrapped = center[:, None, :] + (rel - rel_shift)
    centroid = subs_unwrapped.mean(axis=1)
    vecs = centroid - center

    shift = np.zeros_like(vecs)
    shift[:, pbc] = box[:, pbc] * np.round(vecs[:, pbc] / box[:, pbc])
    vecs -= shift

    norms = np.linalg.norm(vecs, axis=1)
    safe = norms > 1e-12
    vecs[safe] /= norms[safe, None]

    cos_theta = np.clip(vecs[:, 2], -1.0, 1.0)
    angles = np.zeros(n_frames)
    angles[safe] = np.degrees(np.arccos(cos_theta[safe]))

    return angles


def compute_acsf(coordinates, box, center_index=0, Rc=6.0, eta=1.0, Rs=0.0, zeta=1.0, lambd=1.0):
    """
    Atom centered symmetry functions (ACSF) (for ML potentials).
    Jorg Behler, J. Chem. Phys. 134, 074106 (2011). doi:10.1063/1.3553717.
    """
    coordinates = np.asarray(coordinates)
    box = np.asarray(box)
    n_frames = coordinates.shape[0]
    n_atoms = coordinates.shape[1] // 3
    coords = coordinates.reshape(n_frames, n_atoms, 3)

    if box.ndim == 0:
        box = np.full((n_frames, 3), box)
    elif box.ndim == 1:
        if box.shape[0] == 3:
            box = np.tile(box, (n_frames, 1))
        else:
            raise ValueError("box must be scalar or shape (3,)")
    elif box.shape != (n_frames, 3):
        raise ValueError(f"box must have shape (n_frames, 3), got {box.shape}")

    def min_image_vec(vec):
        return vec - box[:, None, :] * np.round(vec / box[:, None, :])

    def min_image_pair(vec):
        return vec - box[:, None, None, :] * np.round(vec / box[:, None, None, :])

    rij_vecs = coords - coords[:, center_index:center_index+1, :]
    rij_vecs = min_image_vec(rij_vecs)
    rij = np.linalg.norm(rij_vecs, axis=2)
    fc = np.where(rij <= Rc, 0.5 * (np.cos(np.pi * rij / Rc) + 1.0), 0.0)

    rij[:, center_index] = 1.0
    fc[:, center_index] = 0.0
    G2 = np.sum(np.exp(-eta * (rij - Rs)**2) * fc, axis=1)

    rij_i = rij[:, :, None]
    rij_k = rij[:, None, :]
    vec_j = rij_vecs[:, :, None, :]
    vec_k = rij_vecs[:, None, :, :]
    dot = np.sum(vec_j * vec_k, axis=-1)
    denom = rij_i * rij_k
    denom = np.where(denom < 1e-12, 1e-12, denom)
    cos_theta = np.clip(dot / denom, -1.0, 1.0)

    rjk_vec = coords[:, :, None, :] - coords[:, None, :, :]
    rjk_vec = min_image_pair(rjk_vec)
    rjk = np.linalg.norm(rjk_vec, axis=-1)
    fc_jk = np.where(rjk <= Rc, 0.5 * (np.cos(np.pi * rjk / Rc) + 1.0), 0.0)

    fc_ij = fc[:, :, None]
    fc_ik = fc[:, None, :]
    mask = (fc_ij > 0) & (fc_ik > 0) & (fc_jk > 0)

    idx = np.arange(n_atoms)
    not_center = idx != center_index
    mask &= not_center[None, :, None]
    mask &= not_center[None, None, :]
    mask &= (idx[None, :, None] != idx[None, None, :])

    G4_term = (
        2**(1 - zeta)
        * (1 + lambd * cos_theta)**zeta
        * np.exp(-eta * (rij_i**2 + rij_k**2 + rjk**2))
        * fc_ij * fc_ik * fc_jk
    )
    G4_term *= mask
    G4 = 0.5 * np.sum(G4_term, axis=(1, 2))

    return G2, G4


def compute_signed_volume_normalized(coords, box):
    """
    Compute normalized signed volume (scalar triple product) with 3D PBC.

    Parameters
    ----------
    coords : (n_frames, 15)  1 central atom + 4 surrounding atoms
    box : scalar, (3,), or (n_frames, 3)
    """
    coords = np.asarray(coords)
    n_frames = coords.shape[0]
    if coords.shape[1] != 15:
        raise ValueError("Each frame must have 15 numbers (1 center + 4 atoms)")
    coords = coords.reshape(n_frames, 5, 3)

    box = np.asarray(box)
    if box.ndim == 0:
        box = np.full((n_frames, 3), box)
    elif box.ndim == 1:
        if box.shape[0] == 3:
            box = np.tile(box, (n_frames, 1))
        else:
            raise ValueError("box must be scalar or shape (3,) or (n_frames, 3)")
    elif box.shape != (n_frames, 3):
        raise ValueError("box must have shape (n_frames, 3)")

    def min_image(vec):
        return vec - box[:, None, :] * np.round(vec / box[:, None, :])

    r0 = coords[:, 0:1, :]

    coords_unwrapped = r0 + min_image(coords - r0)

    r1 = coords_unwrapped[:, 1, :] - coords_unwrapped[:, 0, :]
    r2 = coords_unwrapped[:, 2, :] - coords_unwrapped[:, 0, :]
    r3 = coords_unwrapped[:, 3, :] - coords_unwrapped[:, 0, :]

    numerator = np.einsum('ij,ij->i', r1, np.cross(r2, r3))

    denom = (
        np.linalg.norm(r1, axis=1)
        * np.linalg.norm(r2, axis=1)
        * np.linalg.norm(r3, axis=1)
    )
    volumes = np.zeros(n_frames)
    mask = denom > 1e-12
    volumes[mask] = numerator[mask] / denom[mask]

    return volumes


def compute_tetrahedrality(coords, box):
    """
    Compute tetrahedral distortion parameter with 3D PBC.

    Parameters
    ----------
    coords : (n_frames, 15)  1 central atom + 4 surrounding atoms
    box : scalar, (3,), or (n_frames, 3)

    Returns
    -------
    q_values : (n_frames,)  q = 0 is ideal tetrahedral geometry
    """
    coords = np.asarray(coords)
    n_frames = coords.shape[0]

    if coords.shape[1] != 15:
        raise ValueError("Each frame must have 15 numbers (1 center + 4 atoms)")

    coords = coords.reshape(n_frames, 5, 3)
    box = np.asarray(box)
    if box.ndim == 0:
        box = np.full((n_frames, 3), box)
    elif box.ndim == 1:
        if box.shape[0] == 3:
            box = np.tile(box, (n_frames, 1))
        else:
            raise ValueError("box must be scalar or shape (3,) or (n_frames,3)")
    elif box.shape != (n_frames, 3):
        raise ValueError("box must have shape (n_frames,3)")

    def min_image(vec):
        return vec - box[:, None, :] * np.round(vec / box[:, None, :])

    r0 = coords[:, 0:1, :]
    coords_unwrapped = r0 + min_image(coords - r0)

    vectors = coords_unwrapped[:, 1:, :] - coords_unwrapped[:, 0:1, :]

    norms = np.linalg.norm(vectors, axis=2, keepdims=True)
    valid = np.all(norms > 1e-12, axis=(1, 2))
    unit_vectors = np.zeros_like(vectors)
    unit_vectors[valid] = vectors[valid] / norms[valid]

    cos_matrix = np.einsum('fik,fjk->fij', unit_vectors, unit_vectors)

    target = -1.0 / 3.0
    q_values = np.zeros(n_frames)

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    for i, j in pairs:
        q_values += (cos_matrix[:, i, j] - target)**2

    q_values *= 3.0 / 8.0
    q_values[~valid] = np.nan

    return q_values


def _normalize_box(box, n_frames):
    """Broadcast a scalar / (3,) / (n_frames, 3) box to (n_frames, 3)."""
    box = np.asarray(box, dtype=float)
    if box.ndim == 0:
        return np.full((n_frames, 3), box)
    if box.ndim == 1:
        if box.shape[0] != 3:
            raise ValueError("box must be scalar, (3,), or (n_frames, 3)")
        return np.tile(box, (n_frames, 1))
    if box.shape != (n_frames, 3):
        raise ValueError(f"box must have shape ({n_frames}, 3), got {box.shape}")
    return box


def _min_image(vec, box):
    """
    Minimum-image convention. `vec` has shape (n_frames, 3) or (n_frames, n, 3);
    `box` is (n_frames, 3) and is broadcast along the leading frame axis.
    """
    shape = [box.shape[0]] + [1] * (vec.ndim - 2) + [3]
    b = box.reshape(shape)
    return vec - b * np.round(vec / b)


def _unit(vecs):
    """Normalise along the last axis, leaving near-zero vectors at zero."""
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    out = np.zeros_like(vecs)
    np.divide(vecs, norms, out=out, where=norms > 1e-12)
    return out


def compute_body_frame(coords_tetra, box):
    """
    Right-handed body frame anchored on the proline stereocenter.

    Parameters
    ----------
    coords_tetra : (n_frames, 15)  [CA, N, C, CB, HA]
    box : scalar, (3,), or (n_frames, 3)

    Returns
    -------
    ca : (n_frames, 3)      position of CA, the stereocenter
    normal : (n_frames, 3)  unit PSEUDOVECTOR n = (N - CA) x (C - CA)
    axis : (n_frames, 3)    unit molecular axis (C - N), a true vector
    """
    coords = np.asarray(coords_tetra, dtype=float)
    n_frames = coords.shape[0]
    if coords.shape[1] != 15:
        raise ValueError("Each frame must have 15 numbers (CA, N, C, CB, HA)")
    coords = coords.reshape(n_frames, 5, 3)
    box = _normalize_box(box, n_frames)

    ca = coords[:, 0, :]
    rel = _min_image(coords - ca[:, None, :], box)

    e1 = rel[:, 1, :]
    e2 = rel[:, 2, :]
    normal = _unit(np.cross(e1, e2))
    axis = _unit(e2 - e1)
    return ca, normal, axis


def compute_handed_cn(coords_tetra, coords_lipid, box, d0=0.5, n_exp=6, m_exp=12):
    """
    Handed (pseudoscalar) coordination number.

        hCN = sum_j  w(r_j) * (n . d_j) / r_j
    """
    ca, normal, _ = compute_body_frame(coords_tetra, box)

    lip = np.asarray(coords_lipid, dtype=float)
    n_frames = lip.shape[0]
    lip = lip.reshape(n_frames, -1, 3)
    box = _normalize_box(box, n_frames)

    d = _min_image(lip - ca[:, None, :], box)
    r = np.linalg.norm(d, axis=-1)

    ratio = r / d0
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        numerator = 1.0 - ratio ** n_exp
        denominator = 1.0 - ratio ** m_exp
        w = np.where(np.abs(denominator) < 1e-10, n_exp / m_exp, numerator / denominator)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    safe = r > 1e-12
    cos_signed = np.zeros_like(r)
    np.divide(np.einsum('fi,fji->fj', normal, d), r, out=cos_signed, where=safe)

    handed = np.sum(w * cos_signed, axis=1)

    nearest = np.argmin(np.where(safe, r, np.inf), axis=1)
    nearest_cos = cos_signed[np.arange(n_frames), nearest]

    return handed, nearest_cos


def compute_signed_azimuth(coords_tetra, coords_lipid, box):
    """
    Signed azimuthal angle about the membrane normal between the proline
    molecular axis (N -> C) and the vector from the stereocenter to the
    nearest lipid phosphorus, both projected into the membrane plane.
    """
    ca, _, axis = compute_body_frame(coords_tetra, box)

    lip = np.asarray(coords_lipid, dtype=float)
    n_frames = lip.shape[0]
    lip = lip.reshape(n_frames, -1, 3)
    box = _normalize_box(box, n_frames)

    d = _min_image(lip - ca[:, None, :], box)
    r = np.linalg.norm(d, axis=-1)
    nearest = np.argmin(np.where(r > 1e-12, r, np.inf), axis=1)
    lvec = d[np.arange(n_frames), nearest]

    px, py = axis[:, 0], axis[:, 1]
    lx, ly = lvec[:, 0], lvec[:, 1]
    cross_z = px * ly - py * lx
    dot = px * lx + py * ly

    return np.degrees(np.arctan2(cross_z, dot))


def compute_cremer_pople(coords_ring, box):
    """
    Cremer-Pople puckering coordinates of the 5-membered pyrrolidine ring.
    Reference: Cremer, D.; Pople, J. A. J. Am. Chem. Soc. 1975, 97, 1354.
    """
    coords = np.asarray(coords_ring, dtype=float)
    n_frames = coords.shape[0]
    if coords.shape[1] != 15:
        raise ValueError("Each frame must have 15 numbers (5 ring atoms)")
    coords = coords.reshape(n_frames, 5, 3)
    box = _normalize_box(box, n_frames)

    ref = coords[:, 0:1, :]
    ring = ref + _min_image(coords - ref, box)
    centroid = ring.mean(axis=1)
    rel = ring - centroid[:, None, :]

    j = np.arange(5)
    r1 = np.einsum('fjc,j->fc', rel, np.sin(2.0 * np.pi * j / 5.0))
    r2 = np.einsum('fjc,j->fc', rel, np.cos(2.0 * np.pi * j / 5.0))
    normal = _unit(np.cross(r1, r2))

    z = np.einsum('fjc,fc->fj', rel, normal)

    scale = np.sqrt(2.0 / 5.0)
    a = scale * np.sum(z * np.cos(4.0 * np.pi * j / 5.0), axis=1)
    b = -scale * np.sum(z * np.sin(4.0 * np.pi * j / 5.0), axis=1)

    q2 = np.hypot(a, b)
    phi2 = np.degrees(np.arctan2(b, a))
    return q2, phi2


def compute_local_deformation(pro_xy, coords_p, box, radius=1.0):
    """
    Local membrane deformation around the permeant.
    """
    p = np.asarray(coords_p, dtype=float)
    n_frames = p.shape[0]
    p = p.reshape(n_frames, -1, 3)
    box = _normalize_box(box, n_frames)

    pro_xy = np.asarray(pro_xy, dtype=float).reshape(n_frames, -1)[:, :2]

    dxy = p[:, :, :2] - pro_xy[:, None, :]
    bxy = box[:, None, :2]
    dxy = dxy - bxy * np.round(dxy / bxy)
    r_xy = np.linalg.norm(dxy, axis=-1)

    mask = r_xy < radius
    count = mask.sum(axis=1)
    global_z = p[:, :, 2].mean(axis=1)
    local_sum = np.where(mask, p[:, :, 2], 0.0).sum(axis=1)
    local_z = np.where(count > 0, local_sum / np.maximum(count, 1), global_z)

    return local_z - global_z, local_z, count.astype(float)


# ── New helpers replacing the gmx-specific machinery ──────────────────────────

def compute_com_unwrapped(coords, masses, box):
    """
    Mass-weighted center of mass of a single, bonded (small) group of atoms,
    unwrapping every atom relative to the group's own first atom before
    averaging. These trajectories are not pre-processed with -pbc mol/whole,
    so a naive mean of raw positions is wrong exactly when the group
    straddles a periodic boundary -- rare for a group this size, but not
    impossible, and unwrapping first costs nothing.

    Only meaningful for atoms belonging to ONE molecule (there is a single
    "first atom" to anchor to); do not use this for a COM built from many
    unrelated molecules (e.g. one marker atom per lipid) -- see the note in
    gather_path_arrays' zcom handling for why that case has no equivalent
    fix.
    """
    n_frames = coords.shape[0]
    box = _normalize_box(box, n_frames)
    ref = coords[:, 0:1, :]
    unwrapped = ref + _min_image(coords - ref, box)
    w = masses / masses.sum()
    com = np.einsum('a,fac->fc', w, unwrapped)
    return com, unwrapped


def compute_rg(coords, masses, box):
    """Mass-weighted radius of gyration, matching gmx gyrate's default."""
    com, unwrapped = compute_com_unwrapped(coords, masses, box)
    disp = unwrapped - com[:, None, :]
    rg2 = np.einsum('a,fa->f', masses, np.sum(disp**2, axis=-1)) / masses.sum()
    return np.sqrt(rg2)


def compute_rmsd_series(coords, ref_coords, box):
    """
    Least-squares-superposition RMSD per frame, matching gmx rms's default.

    Unwraps the (few, bonded) moving atoms relative to their own first atom
    first: gmx rms itself does no such reconstruction either, so this is a
    deliberate hardening over the original rather than parity with it --
    ORP's 3-atom backbone is small enough that a PBC split is rare, but not
    impossible over a long trajectory, and the fix is nearly free here.
    """
    n = coords.shape[0]
    b = _normalize_box(box, n)
    ref0 = coords[:, 0:1, :]
    coords_unwrapped = ref0 + _min_image(coords - ref0, b)
    return np.array([mda_rmsd(coords_unwrapped[i], ref_coords, superposition=True)
                      for i in range(n)])


def compute_bonded_angle(coords, box):
    """Angle a-b-c (degrees) for 3 explicit atoms, PBC-corrected bond vectors."""
    n = coords.shape[0]
    xyz = coords.reshape(n, 3, 3)
    b = _normalize_box(box, n)
    v1 = _min_image(xyz[:, 0, :] - xyz[:, 1, :], b)
    v2 = _min_image(xyz[:, 2, :] - xyz[:, 1, :], b)
    cos = (np.einsum('fi,fi->f', v1, v2)
           / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def compute_dihedral_series(coords, box):
    """Standard (IUPAC-convention) dihedral a-b-c-d in degrees, PBC-corrected."""
    n = coords.shape[0]
    xyz = coords.reshape(n, 4, 3)
    b = _normalize_box(box, n)
    b0 = _min_image(xyz[:, 0, :] - xyz[:, 1, :], b)
    b1 = _min_image(xyz[:, 2, :] - xyz[:, 1, :], b)
    b2 = _min_image(xyz[:, 3, :] - xyz[:, 2, :], b)
    b1n = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
    v = b0 - np.einsum('fi,fi->f', b0, b1n)[:, None] * b1n
    w = b2 - np.einsum('fi,fi->f', b2, b1n)[:, None] * b1n
    x = np.einsum('fi,fi->f', v, w)
    y = np.einsum('fi,fi->f', np.cross(b1n, v), w)
    return np.degrees(np.arctan2(y, x))


def group_distances(ref_pos, sel_pos, box):
    """Per-frame distances from one reference atom to each of several selection
    atoms, PBC-corrected -- the same array shape gmx pairdist's -o used to give."""
    diff = sel_pos - ref_pos[:, None, :]
    diff = _min_image(diff, _normalize_box(box, diff.shape[0]))
    return np.linalg.norm(diff, axis=-1)


def count_atoms_within(orp_pos, target_pos, box6, radius):
    """
    Number of target atoms with at least one ORP atom within `radius` --
    matches gmx select's atom-based (not residue/COM-based) "within R of SEL".
    """
    pairs = capped_distance(orp_pos, target_pos, max_cutoff=radius, box=box6,
                             return_distances=False)
    if pairs.shape[0] == 0:
        return 0
    return len(np.unique(pairs[:, 1]))


def _hbond_angle_deg(h_pos, d_pos, a_pos, box3):
    """Hydrogen-Donor-Acceptor angle (degrees) for one donor/H and several
    candidate acceptors, orthorhombic minimum image (consistent with the rest
    of this file's PBC handling)."""
    v1 = h_pos - d_pos
    v1 = v1 / np.linalg.norm(v1)
    v2 = a_pos - d_pos[None, :]
    v2 = v2 - box3 * np.round(v2 / box3)
    v2n = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
    cos = np.clip(v2n @ v1, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def count_hbonds(h_pos, d_pos, acc_pos, box6, cutoff, angle_cutoff_deg=30.0):
    """
    Replica of gmx hbond's default criterion: donor-acceptor heavy-atom
    distance <= cutoff AND Hydrogen-Donor-Acceptor angle <= 30 degrees.
    """
    pairs = capped_distance(d_pos[None, :], acc_pos, max_cutoff=cutoff, box=box6,
                             return_distances=False)
    if pairs.shape[0] == 0:
        return 0
    acc_idx = pairs[:, 1]
    ang = _hbond_angle_deg(h_pos, d_pos, acc_pos[acc_idx], box6[:3])
    return int(np.count_nonzero(ang <= angle_cutoff_deg))


# ── System context, resolved on first use ─────────────────────────────────────
# The original script resolved all of the below at IMPORT time, which made the
# module unimportable unless the process happened to be sitting in a simulation
# directory containing ../gromacs_input/. As a package module it has to be
# importable anywhere - `chiroflux generate-cvs --help` must not need a .tpr -
# so the identical work is deferred into this initialiser and run once, on the
# first call that actually needs the atom indices.
#
# The body is the original block verbatim; only the hardcoded '../gromacs_input'
# prefix became a parameter, and the assignments became globals so every
# function below still reads them exactly as it did before.

_SYSTEM_READY = False


def _init_system(gromacs_input="../gromacs_input"):
    """Resolve topology, .ndx groups and atom indices. Idempotent."""
    global _SYSTEM_READY
    global topol_file, conf_gro_file, rmsd_index_file, rg_index_file, angle_index_file
    global angle_OH_index_file, ring_plane_index_file, ring_plane_chiral_index_file, dihedral_index_file, dihedral_OH_index_file
    global tetrahedral_index_file, CN_DOPC_index_file, CN_POPC_index_file, ZCOM_index_file, lipid_per_leaflet
    global atom_per_lipid_dopc, atom_per_lipid_popc, IDX_BACKBONE, IDX_HEAVY, IDX_C_CG
    global IDX_ANG_OH, IDX_RING5, IDX_RING6, IDX_DIH_CHIRAL, IDX_DIH_OH
    global IDX_TETRA5, CN_DOPC_GROUPS, CN_POPC_GROUPS, IDX_CN_REF, IDX_CN_REF_COMBINED
    global ZCOM_GROUPS, ZCOM_SCALAR_GROUPS, _ref_u, IDX_ORP_ALL, IDX_ORP_N
    global IDX_ORP_H01, IDX_ORP_O01, IDX_ORP_H02, IDX_PHOSPHATE, IDX_CARBONYL
    global IDX_NITROGEN, IDX_WATER, IDX_DOPC_ALL, IDX_POPC_ALL, HEAVY_MASSES
    global ORP_MASSES, ZCOM_MASSES, REF_BACKBONE_POS, COORD_NUM_SPECS_MAIN, COORD_NUM_SPECS_EXTRA
    global COORD_NUM_SPECS, _SEL_IDX_DOPC, _SEL_IDX_POPC
    if _SYSTEM_READY:
        return
    _require_mdanalysis()

    # ── Static config ─────────────────────────────────────────────────────────────
    topol_file                   = f'{gromacs_input}/topol.tpr'
    conf_gro_file                 = f'{gromacs_input}/conf.gro'
    rmsd_index_file               = f"{gromacs_input}/PRO_backbone.ndx"
    rg_index_file                 = f"{gromacs_input}/PRO_Non_H.ndx"
    angle_index_file              = f"{gromacs_input}/PRO_C_CG.ndx"
    angle_OH_index_file           = f"{gromacs_input}/PRO_C_O_H.ndx"
    ring_plane_index_file         = f"{gromacs_input}/PRO_ring_plane.ndx"
    ring_plane_chiral_index_file  = f"{gromacs_input}/PRO_ring_plane_chiral.ndx"
    dihedral_index_file           = f"{gromacs_input}/PRO_dihedral.ndx"
    dihedral_OH_index_file        = f"{gromacs_input}/PRO_dihedral_OH.ndx"
    tetrahedral_index_file        = f"{gromacs_input}/PRO_tetrahedral.ndx"
    CN_DOPC_index_file            = f"{gromacs_input}/CN_DOPC.ndx"
    CN_POPC_index_file            = f"{gromacs_input}/CN_POPC.ndx"
    ZCOM_index_file               = f'{gromacs_input}/ZCOM_index.ndx'
    lipid_per_leaflet     = 66
    atom_per_lipid_dopc   = 138
    atom_per_lipid_popc   = 134

    # ── Atom-group indices (0-based), resolved once at import time ───────────────
    # Small/frozen intramolecular groups come straight from the .ndx files gmx already
    # used (identical semantics, no reinterpretation needed).
    IDX_BACKBONE   = ndx_group_list(rmsd_index_file)[0]                 # N, CA, C
    IDX_HEAVY      = ndx_group_list(rg_index_file)[0]                   # 8 heavy atoms
    IDX_C_CG       = ndx_group_list(angle_index_file)[0]                # C, CG
    IDX_ANG_OH     = ndx_group_list(angle_OH_index_file)[0]             # C, O, H02
    IDX_RING5      = ndx_group_list(ring_plane_index_file)[0]           # CA,CB,CG,CD,N
    IDX_RING6      = ndx_group_list(ring_plane_chiral_index_file)[0]    # C,CA,CB,CG,CD,N
    IDX_DIH_CHIRAL = ndx_group_list(dihedral_index_file)[0]             # CB,CA,C,O
    IDX_DIH_OH     = ndx_group_list(dihedral_OH_index_file)[0]          # O,C,O01,H02
    IDX_TETRA5     = ndx_group_list(tetrahedral_index_file)[0]          # CA,N,C,CB,HA

    CN_DOPC_GROUPS = ndx_group_list(CN_DOPC_index_file)  # 19 groups, 0..18
    CN_POPC_GROUPS = ndx_group_list(CN_POPC_index_file)  # 19 groups, 0..18
    IDX_CN_REF     = CN_DOPC_GROUPS[0:5]                 # CA, O01, N, HA, CD (same in both files)
    IDX_CN_REF_COMBINED = np.concatenate(IDX_CN_REF)     # same 5 atoms as one group, in order

    ZCOM_GROUPS = ndx_group_list(ZCOM_index_file)  # 14 groups, 0..13
    # 0=ORP 1=N_top 2=N_bot 3=P_top 4=P_bot 5=O2_top 6=O2_bot 7=O3_top 8=O3_bot
    # 9=C2_top 10=C2_bot 11=C3_top 12=C3_bot 13=whole membrane
    ZCOM_SCALAR_GROUPS = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # stored as scalar COM-z only
    # groups 0 (ORP), 3 (P_top), 4 (P_bot) are stored as full per-atom positions:
    # ORP's own COM needs x,y too (pro_xy), and P_top/P_bot feed the chirality CVs.

    # ORP donor/acceptor atoms and the resname-based bulk selections are resolved by
    # NAME rather than by .ndx group, since none of gmx's leaflet-splitting applies to
    # them -- a plain "resname ORP and name X" is unambiguous and self-documenting.
    # ORP is the neutral (COOH + secondary-amine) form of proline permeating the
    # bilayer: N1-H01 and O01-H02 are its only two polar-hydrogen donor sites; the
    # membrane-side acceptor sets (phosphate O's, ester O's, choline N) were
    # identified from the real topol.tpr (see PRO_HB.ndx / atom-name lookup).
    #
    # A bare mda.Universe(tpr) (no trajectory reader attached) is only used here
    # for topology (atom selections, masses) -- that never needs coordinates and
    # works everywhere. Getting the tpr's own embedded reference COORDINATES back
    # out reliably is a separate, harder problem: locally this returned real
    # positions (oddly already in nm, unlike every real trajectory reader, which
    # gives Angstrom), but on another machine/MDAnalysis version the exact same
    # call raised "NoDataError: This Universe has no coordinates". That
    # inconsistency isn't something to build on, so the reference structure's
    # coordinates are instead read from conf_gro_file -- the same initial
    # structure, in a format (.gro) MDAnalysis reads through a real, consistent
    # Reader -- rather than trusted to a bare Universe(tpr).
    _ref_u = mda.Universe(topol_file)
    IDX_ORP_ALL  = _ref_u.select_atoms("resname ORP").indices
    IDX_ORP_N    = _ref_u.select_atoms("resname ORP and name N").indices[0]
    IDX_ORP_H01  = _ref_u.select_atoms("resname ORP and name H01").indices[0]
    IDX_ORP_O01  = _ref_u.select_atoms("resname ORP and name O01").indices[0]
    IDX_ORP_H02  = _ref_u.select_atoms("resname ORP and name H02").indices[0]
    IDX_PHOSPHATE = _ref_u.select_atoms("resname DOPC POPC and name O11 O12 O13 O14").indices
    IDX_CARBONYL  = _ref_u.select_atoms("resname DOPC POPC and name O21 O22 O31 O32").indices
    IDX_NITROGEN  = _ref_u.select_atoms("resname DOPC POPC and name N").indices
    IDX_WATER    = _ref_u.select_atoms("resname TIP3").indices
    IDX_DOPC_ALL = _ref_u.select_atoms("resname DOPC").indices
    IDX_POPC_ALL = _ref_u.select_atoms("resname POPC").indices
    HEAVY_MASSES = _ref_u.atoms[IDX_HEAVY].masses.copy()
    ORP_MASSES = _ref_u.atoms[IDX_ORP_ALL].masses.copy()
    ZCOM_MASSES = [_ref_u.atoms[g].masses.copy() for g in ZCOM_GROUPS]
    del _ref_u


    _ref_u = mda.Universe(topol_file, conf_gro_file)
    # A real Reader (GRO here) always converts to MDAnalysis's internal Angstrom,
    # same as every trajectory frame read elsewhere in this file -- verified
    # directly: reading this exact atom through the GRO reader gives (34.99,
    # 61.84, 25.64), matching the XTC-read value to rounding, both Angstrom.
    REF_BACKBONE_POS = _ref_u.atoms[IDX_BACKBONE].positions.copy() * 0.1  # Angstrom -> nm
    del _ref_u

    # ── Coordination-number (ref, sel) pairs, transcribed verbatim from the
    #    original process_single_path's `params` OrderedDict (lines ~2075-2140). ──
    # ref indexes IDX_CN_REF (0=CA, 1=O01, 2=N, 3=HA, 4=CD); sel indexes the 19
    # CN_DOPC_GROUPS / CN_POPC_GROUPS (5..18 are lipid leaflet markers).
    # Split in two exactly where the original put 'PRO_tetra' (right after
    # CD_O32_l_POPC, before the "Additional CVs" block) so the output column order
    # matches the original ML.txt schema exactly, not just its column names.
    COORD_NUM_SPECS_MAIN = [
        ('CA_C2_u_DOPC', 'DOPC', 0, 5),   ('CA_C2_l_DOPC', 'DOPC', 0, 6),
        ('CA_P_u_DOPC',  'DOPC', 0, 7),   ('CA_P_l_DOPC',  'DOPC', 0, 8),
        ('O_N_u_DOPC',   'DOPC', 1, 9),   ('O_N_l_DOPC',   'DOPC', 1, 10),
        ('N_P_u_DOPC',   'DOPC', 2, 7),   ('N_P_l_DOPC',   'DOPC', 2, 8),
        ('CA_CC2_u_DOPC','DOPC', 0, 11),  ('CA_CC2_l_DOPC','DOPC', 0, 12),
        ('CA_CC3_u_DOPC','DOPC', 0, 13),  ('CA_CC3_l_DOPC','DOPC', 0, 14),
        ('CA_C2_u_POPC', 'POPC', 0, 5),   ('CA_C2_l_POPC', 'POPC', 0, 6),
        ('CA_P_u_POPC',  'POPC', 0, 7),   ('CA_P_l_POPC',  'POPC', 0, 8),
        ('O_N_u_POPC',   'POPC', 1, 9),   ('O_N_l_POPC',   'POPC', 1, 10),
        ('N_P_u_POPC',   'POPC', 2, 7),   ('N_P_l_POPC',   'POPC', 2, 8),
        ('CA_CC2_u_POPC','POPC', 0, 11),  ('CA_CC2_l_POPC','POPC', 0, 12),
        ('CA_CC3_u_POPC','POPC', 0, 13),  ('CA_CC3_l_POPC','POPC', 0, 14),
        ('HA_P_u_DOPC',  'DOPC', 3, 7),   ('HA_P_l_DOPC',  'DOPC', 3, 8),
        ('HA_O22_u_DOPC','DOPC', 3, 15),  ('HA_O22_l_DOPC','DOPC', 3, 16),
        ('HA_O32_u_DOPC','DOPC', 3, 17),  ('HA_O32_l_DOPC','DOPC', 3, 18),
        ('CD_P_u_DOPC',  'DOPC', 4, 7),   ('CD_P_l_DOPC',  'DOPC', 4, 8),
        ('CD_O22_u_DOPC','DOPC', 4, 15),  ('CD_O22_l_DOPC','DOPC', 4, 16),
        ('CD_O32_u_DOPC','DOPC', 4, 17),  ('CD_O32_l_DOPC','DOPC', 4, 18),
        ('HA_P_u_POPC',  'POPC', 3, 7),   ('HA_P_l_POPC',  'POPC', 3, 8),
        ('HA_O22_u_POPC','POPC', 3, 15),  ('HA_O22_l_POPC','POPC', 3, 16),
        ('HA_O32_u_POPC','POPC', 3, 17),  ('HA_O32_l_POPC','POPC', 3, 18),
        ('CD_P_u_POPC',  'POPC', 4, 7),   ('CD_P_l_POPC',  'POPC', 4, 8),
        ('CD_O22_u_POPC','POPC', 4, 15),  ('CD_O22_l_POPC','POPC', 4, 16),
        ('CD_O32_u_POPC','POPC', 4, 17),  ('CD_O32_l_POPC','POPC', 4, 18),
    ]
    COORD_NUM_SPECS_EXTRA = [
        ('N_CC2_u_DOPC', 'DOPC', 1, 11),  ('N_CC2_l_DOPC', 'DOPC', 1, 12),
        ('N_CC3_u_DOPC', 'DOPC', 1, 13),  ('N_CC3_l_DOPC', 'DOPC', 1, 14),
        ('O_CC2_u_DOPC', 'DOPC', 2, 11),  ('O_CC2_l_DOPC', 'DOPC', 2, 12),
        ('O_CC3_u_DOPC', 'DOPC', 2, 13),  ('O_CC3_l_DOPC', 'DOPC', 2, 14),
        ('N_CC2_u_POPC', 'POPC', 1, 11),  ('N_CC2_l_POPC', 'POPC', 1, 12),
        ('N_CC3_u_POPC', 'POPC', 1, 13),  ('N_CC3_l_POPC', 'POPC', 1, 14),
        ('O_CC2_u_POPC', 'POPC', 2, 11),  ('O_CC2_l_POPC', 'POPC', 2, 12),
        ('O_CC3_u_POPC', 'POPC', 2, 13),  ('O_CC3_l_POPC', 'POPC', 2, 14),
    ]
    COORD_NUM_SPECS = COORD_NUM_SPECS_MAIN + COORD_NUM_SPECS_EXTRA

    # Every lipid-side sel group index actually referenced above, per species.
    _SEL_IDX_DOPC = sorted({sel for _, sp, _, sel in COORD_NUM_SPECS if sp == 'DOPC'})
    _SEL_IDX_POPC = sorted({sel for _, sp, _, sel in COORD_NUM_SPECS if sp == 'POPC'})


    # ── One-pass-per-segment trajectory reader ────────────────────────────────────


    _SYSTEM_READY = True


def gather_path_arrays(xtc_files, directions):
    """
    Read every xtc segment of a path exactly once with MDAnalysis, in the order
    (and per-segment reversal) the original script used, and return every raw
    quantity the CVs below need. Small atom-group positions are kept as full
    (n_frames, n_atoms, 3) arrays (cheap); bulk quantities (water/lipid contact
    counts, hydrogen bonds, whole-membrane COM) are reduced to a scalar
    immediately, frame by frame, so the large atom sets are never stored.
    """
    small_keys = (['backbone', 'heavy', 'c_cg', 'ang_oh', 'ring5', 'ring6',
                   'dih_chiral', 'dih_oh', 'tetra5', 'orp_all', 'lipPu', 'lipPl',
                   'cn_ref']
                  + [f'dopc_{i}' for i in _SEL_IDX_DOPC]
                  + [f'popc_{i}' for i in _SEL_IDX_POPC])

    small_idx = {
        'backbone': IDX_BACKBONE, 'heavy': IDX_HEAVY, 'c_cg': IDX_C_CG,
        'ang_oh': IDX_ANG_OH, 'ring5': IDX_RING5, 'ring6': IDX_RING6,
        'dih_chiral': IDX_DIH_CHIRAL, 'dih_oh': IDX_DIH_OH, 'tetra5': IDX_TETRA5,
        'orp_all': IDX_ORP_ALL, 'lipPu': ZCOM_GROUPS[3], 'lipPl': ZCOM_GROUPS[4],
        'cn_ref': IDX_CN_REF_COMBINED,
    }
    for i in _SEL_IDX_DOPC:
        small_idx[f'dopc_{i}'] = CN_DOPC_GROUPS[i]
    for i in _SEL_IDX_POPC:
        small_idx[f'popc_{i}'] = CN_POPC_GROUPS[i]

    scalar_keys = (['water_03', 'dopc_05', 'popc_05', 'water_06', 'water_10',
                    'hb_po', 'hb_co', 'hb_n']
                   + [f'zcom_{gi}' for gi in ZCOM_SCALAR_GROUPS])

    box_segs = []
    small_segs = {k: [] for k in small_keys}
    scalar_segs = {k: [] for k in scalar_keys}

    for xtc_file, direction in zip(xtc_files, directions):
        u = mda.Universe(topol_file, xtc_file)
        n = len(u.trajectory)
        order = range(n - 1, -1, -1) if direction == "-1" else range(n)

        seg_box = np.empty((n, 3))
        seg_small = {k: np.empty((n, len(small_idx[k]), 3)) for k in small_keys}
        seg_scalar = {k: np.empty(n) for k in scalar_keys}

        for out_i, frame_i in enumerate(order):
            ts = u.trajectory[frame_i]
            # MDAnalysis's native length unit is Angstrom regardless of the
            # source file (GROMACS itself is nm-native) -- every distance
            # constant in this file (0.3/0.5/0.6/1.0/0.35 nm cutoffs, d0=0.5
            # in compute_coordination_number, Rc=6.0 in compute_acsf, ...) was
            # transcribed from the nm-native gmx original, so convert to nm
            # here, once, rather than rescale every constant individually.
            pos = ts.positions * 0.1
            dims = ts.dimensions.copy()
            dims[:3] *= 0.1                 # lengths only; dims[3:6] are angles
            box3 = dims[:3]
            seg_box[out_i] = box3

            for k in small_keys:
                seg_small[k][out_i] = pos[small_idx[k]]

            orp_pos = pos[IDX_ORP_ALL]
            seg_scalar['water_03'][out_i] = count_atoms_within(orp_pos, pos[IDX_WATER], dims, 0.3)
            seg_scalar['dopc_05'][out_i]  = count_atoms_within(orp_pos, pos[IDX_DOPC_ALL], dims, 0.5)
            seg_scalar['popc_05'][out_i]  = count_atoms_within(orp_pos, pos[IDX_POPC_ALL], dims, 0.5)
            seg_scalar['water_06'][out_i] = count_atoms_within(orp_pos, pos[IDX_WATER], dims, 0.6)
            seg_scalar['water_10'][out_i] = count_atoms_within(orp_pos, pos[IDX_WATER], dims, 1.0)

            h01, n1, o01, h02 = pos[IDX_ORP_H01], pos[IDX_ORP_N], pos[IDX_ORP_O01], pos[IDX_ORP_H02]
            seg_scalar['hb_po'][out_i] = (
                count_hbonds(h01, n1, pos[IDX_PHOSPHATE], dims, 0.35)
                + count_hbonds(h02, o01, pos[IDX_PHOSPHATE], dims, 0.35)
            )
            seg_scalar['hb_co'][out_i] = (
                count_hbonds(h01, n1, pos[IDX_CARBONYL], dims, 0.35)
                + count_hbonds(h02, o01, pos[IDX_CARBONYL], dims, 0.35)
            )
            seg_scalar['hb_n'][out_i] = (
                count_hbonds(h01, n1, pos[IDX_NITROGEN], dims, 0.35)
                + count_hbonds(h02, o01, pos[IDX_NITROGEN], dims, 0.35)
            )

            # NOTE on PBC here: unlike ORP (one bonded molecule -- see
            # compute_com_unwrapped), each of these zcom groups is one marker
            # atom per lipid, drawn from up to 66 DIFFERENT molecules with no
            # bond connecting them, so there is no single reference atom to
            # unwrap the rest relative to before averaging. A raw z-mean is
            # only safe because these are leaflet phosphate/choline/whole-
            # membrane atoms, which sit in a narrow z-band with a solvent
            # buffer on both sides in a normal bilayer setup -- they aren't
            # expected to be near the z-periodic boundary. gmx traj -com
            # (what the original script used here) makes the same assumption:
            # it does not reconstruct a group spanning unrelated molecules
            # either without an explicit -pbc flag, which the original never
            # passed. If the membrane genuinely drifts to the box edge in z
            # this would need trajectory-level (nojump-style) unwrapping,
            # which is a materially bigger change than a per-frame fix.
            for gi, masses in zip(ZCOM_SCALAR_GROUPS, [ZCOM_MASSES[g] for g in ZCOM_SCALAR_GROUPS]):
                idx = ZCOM_GROUPS[gi]
                seg_scalar[f'zcom_{gi}'][out_i] = np.average(pos[idx, 2], weights=masses)

        box_segs.append(seg_box)
        for k in small_keys:
            small_segs[k].append(seg_small[k])
        for k in scalar_keys:
            scalar_segs[k].append(seg_scalar[k])

    box = np.vstack(box_segs)
    small = {k: np.vstack(v) for k, v in small_segs.items()}
    scalars = {k: np.concatenate(v) for k, v in scalar_segs.items()}
    return box, small, scalars


def write_ML_txt(path_number, reactive, ensemble, lambda_A, lambda_minus_one, **arrays):
    print("=" * 40)
    print("Writing the final file started ...")
    names = list(arrays.keys())
    arrays = list(arrays.values())

    print("Array shapes before stacking:")
    for i, arr in enumerate(arrays):
        print(f"{names[i]:7s}: {arr.shape}")

    data = np.column_stack(arrays[1:])
    n_op = arrays[0].shape[0]

    if data.shape[0] > 1:
        adjacent_duplicate = np.all(
            np.isclose(data[1:], data[:-1], equal_nan=True), axis=1)
        keep = np.concatenate(([True], ~adjacent_duplicate))
        n_seam = int((~keep).sum())
        if n_seam:
            print(f"Seam frames removed at positions: {np.where(~keep)[0]}")
        data = data[keep]
    else:
        n_seam = 0
    print(f"Total {n_seam} seam frame(s) removed.")

    size_difference = data.shape[0] - n_op
    if size_difference > 0:
        print(f"[!] WARNING: {size_difference} more feature row(s) than order "
              f"parameters after seam removal - trimming features from the end.")
        data = data[:n_op]
    elif size_difference < 0:
        print(f"[!] WARNING: {-size_difference} fewer feature row(s) than order "
              f"parameters - trimming the order parameter from the end.")
        arrays[0] = arrays[0][:data.shape[0]]

    data = np.column_stack((arrays[0], data))
    deleted = []
    first, last = arrays[0][0], arrays[0][-1]

    if (ensemble == "plus" and first < lambda_A) or \
       (ensemble == "minus" and (first < lambda_minus_one or first > lambda_A)):
        data = np.delete(data, 0, axis=0)
        print(f"The first phase point was removed: {first}")
        deleted.append("first")

    if (ensemble == "plus" and last < lambda_A) or \
       (ensemble == "minus" and (last < lambda_minus_one or last > lambda_A)):
        data = np.delete(data, -1, axis=0)
        print(f"The last phase point was removed: {last}")
        deleted.append("last")

    print(f"Deleted points: {', '.join(deleted) if deleted else 'none'}")

    df = pd.DataFrame(data, columns=names)
    output_path = f"{ML}/{path_number}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# {'reactive' if reactive else 'non-reactive'}\n")
        f.write(f"# {ensemble} ensemble\n")
        if size_difference > 0:
            trimmed = f"{size_difference} feature row(s) trimmed from the end"
        elif size_difference < 0:
            trimmed = f"{-size_difference} order-parameter row(s) trimmed from the end"
        else:
            trimmed = "no trimming needed"
        f.write(
            f"# {n_seam} seam frame(s) removed, {trimmed}, "
            f"deleted phase points: {', '.join(deleted) if deleted else 'none'}\n"
        )
        df.to_csv(f, sep=' ', index=False, float_format='%.5f')

    print(f"Arrays written to '{output_path}' successfully.")
    print("=" * 80)


def process_single_path(path_number, overwrite, lambda_A, lambda_B, lambda_minus_one):
    """
    Process a single path number -- the unit of parallelism, one call per path,
    runs in its own worker process. Everything is computed in one MDAnalysis
    pass over the path's trajectory segments; there is no per-feature cache, so
    a re-run either fully recomputes a path or (without --overwrite) skips it
    outright if ML/{path}.txt already exists.

    Returns:
        (path_number, "ok")
        (path_number, "skipped", msg)
        (path_number, "error", msg)
    """
    output_path = f"{ML}/{path_number}.txt"
    if not overwrite and os.path.isfile(output_path):
        return (path_number, "skipped", f"{output_path} already exists.")

    path_folder = f"../load/{path_number}/accepted/"
    if not os.path.exists(path_folder):
        return (path_number, "skipped", f"Path folder does not exist: {path_folder}")

    try:
        order = np.loadtxt(f"../load/{path_number}/order.txt", comments=('#', '@'))
        traj = np.loadtxt(f"../load/{path_number}/traj.txt", comments=('#', '@'), usecols=(2,))
        order_parameter = np.column_stack((order, traj))

        if ((lambda_minus_one < order_parameter[0, 1] < lambda_A) or
                (lambda_minus_one < order_parameter[-1, 1] < lambda_A)):
            ensemble = "plus"
        else:
            ensemble = "minus"

        reactive = get_reactive_paths(path_number, infretis_data_file, lambda_B)
        if reactive is None:
            return (path_number, "skipped", "Path not found in infretis_data.txt")

        status = 'reactive' if reactive else 'non-reactive'
        print("=" * 80)
        print(f"[Worker] Path: {path_number}, {status}")

        xtc_files_sorted, xtc_files_direction, g96_index = extract_sorted_traj_names(
            f"../load/{path_number}/traj.txt"
        )
        xtc_files = [os.path.join(path_folder, f) for f in xtc_files_sorted]
        print(f"{len(xtc_files)} xtc files found: {xtc_files_sorted}")

        if g96_index is not None:
            order_parameter = np.delete(order_parameter, int(g96_index), axis=0)

        if len(xtc_files) == 0:
            return (path_number, "skipped", "No xtc files found")

        # ── One pass over every frame of every segment ────────────────────────
        box, small, scalars = gather_path_arrays(xtc_files, xtc_files_direction)
        n_frames = box.shape[0]

        lx, ly, lz = box[:, 0], box[:, 1], box[:, 2]
        # Kept as (Lx, Lx, Lz) -- i.e. the same semi-isotropic-NPT approximation
        # (box X == box Y) the original script used for Mem_APL/BoxXY, so this
        # column stays numerically comparable to historical ML.txt output. The
        # true Ly is available in `ly` above if that approximation is ever
        # dropped in favour of Mem_APL = Lx*Ly / lipid_per_leaflet.
        pbc_box = np.column_stack([lx, lx, lz])
        Mem_APL = (lx ** 2) / lipid_per_leaflet

        tetra_coords = small['tetra5'].reshape(n_frames, -1)
        ring_coords = small['ring5'].reshape(n_frames, -1)
        lip_Pu = small['lipPu'].reshape(n_frames, -1)
        lip_Pl = small['lipPl'].reshape(n_frames, -1)
        lip_C2u_DOPC = small['dopc_5'].reshape(n_frames, -1)
        lip_C2l_DOPC = small['dopc_6'].reshape(n_frames, -1)
        lip_O22u_DOPC = small['dopc_15'].reshape(n_frames, -1)
        lip_O22l_DOPC = small['dopc_16'].reshape(n_frames, -1)
        lip_C2u_POPC = small['popc_5'].reshape(n_frames, -1)
        lip_C2l_POPC = small['popc_6'].reshape(n_frames, -1)

        hcn_C2u_DOPC, ncos_C2u_DOPC = compute_handed_cn(tetra_coords, lip_C2u_DOPC, pbc_box)
        hcn_C2l_DOPC, ncos_C2l_DOPC = compute_handed_cn(tetra_coords, lip_C2l_DOPC, pbc_box)
        hcn_O22u_DOPC, _ = compute_handed_cn(tetra_coords, lip_O22u_DOPC, pbc_box)
        hcn_O22l_DOPC, _ = compute_handed_cn(tetra_coords, lip_O22l_DOPC, pbc_box)
        hcn_C2u_POPC, _ = compute_handed_cn(tetra_coords, lip_C2u_POPC, pbc_box)
        hcn_C2l_POPC, _ = compute_handed_cn(tetra_coords, lip_C2l_POPC, pbc_box)
        hcn_Pu, _ = compute_handed_cn(tetra_coords, lip_Pu, pbc_box)
        hcn_Pl, _ = compute_handed_cn(tetra_coords, lip_Pl, pbc_box)

        azim_P = compute_signed_azimuth(tetra_coords, np.hstack([lip_Pu, lip_Pl]), pbc_box)
        cp_q2, cp_phi2 = compute_cremer_pople(ring_coords, pbc_box)

        # ORP is a single, small, bonded molecule, so -- unlike the lipid
        # leaflet-marker groups below -- there is a well-defined reference
        # atom to unwrap the other 16 relative to before averaging.
        orp_com, _ = compute_com_unwrapped(small['orp_all'], ORP_MASSES, pbc_box)
        pro_xy = orp_com[:, :2]
        z_PRO = orp_com[:, 2]

        def_u, locz_u, _ = compute_local_deformation(pro_xy, lip_Pu, pbc_box, radius=1.0)
        def_l, locz_l, _ = compute_local_deformation(pro_xy, lip_Pl, pbc_box, radius=1.0)

        rmsd_vals = compute_rmsd_series(small['backbone'], REF_BACKBONE_POS, pbc_box)
        rg_vals = compute_rg(small['heavy'], HEAVY_MASSES, pbc_box)

        # z-COM of the P_top/P_bot marker groups: these two zcom groups are
        # stored as full positions (lip_Pu/lip_Pl, reused by the chirality CVs
        # above) rather than reduced to a scalar in gather_path_arrays, so their
        # mass-weighted COM-z is computed here instead of read from `scalars`.
        z_PTop = np.average(lip_Pu.reshape(n_frames, -1, 3)[:, :, 2], axis=1, weights=ZCOM_MASSES[3])
        z_PBot = np.average(lip_Pl.reshape(n_frames, -1, 3)[:, :, 2], axis=1, weights=ZCOM_MASSES[4])

        params = OrderedDict([
            ('OP_Lamb', np.squeeze(order_parameter[:, 1])),
            ('Mem_APL', np.squeeze(Mem_APL)),
            ('PRO_RMS', np.squeeze(rmsd_vals)),
            ('Water', np.squeeze(scalars['water_03'] / 3)),
            ('DOPC', np.squeeze(scalars['dopc_05'] / atom_per_lipid_dopc)),
            ('POPC', np.squeeze(scalars['popc_05'] / atom_per_lipid_popc)),
            ('BoxXY', np.squeeze(lx)),
            ('BoxZ', np.squeeze(lz)),
            ('z_PRO', np.squeeze(z_PRO)),
            ('z_NTop', np.squeeze(scalars['zcom_1'])),
            ('z_NBot', np.squeeze(scalars['zcom_2'])),
            ('z_PTop', np.squeeze(z_PTop)),
            ('z_PBot', np.squeeze(z_PBot)),
            ('z_O2_T', np.squeeze(scalars['zcom_5'])),
            ('z_O2_B', np.squeeze(scalars['zcom_6'])),
            ('z_O3_T', np.squeeze(scalars['zcom_7'])),
            ('z_O3_B', np.squeeze(scalars['zcom_8'])),
            ('z_C2_T', np.squeeze(scalars['zcom_9'])),
            ('z_C2_B', np.squeeze(scalars['zcom_10'])),
            ('z_C3_T', np.squeeze(scalars['zcom_11'])),
            ('z_C3_B', np.squeeze(scalars['zcom_12'])),
            ('z_Memb', np.squeeze(scalars['zcom_13'])),
            ('PRO_rg', np.squeeze(rg_vals)),
            ('PRO_PO_hb', np.squeeze(scalars['hb_po'])),
            ('PRO_CO_hb', np.squeeze(scalars['hb_co'])),
            ('PRO_N_hb', np.squeeze(scalars['hb_n'])),
            ('PRO_dih_chiral', np.squeeze(compute_dihedral_series(small['dih_chiral'].reshape(n_frames, -1), pbc_box))),
            ('PRO_dih_OH', np.squeeze(compute_dihedral_series(small['dih_oh'].reshape(n_frames, -1), pbc_box))),
            ('PRO_ang_OH', np.squeeze(compute_bonded_angle(small['ang_oh'].reshape(n_frames, -1), pbc_box))),
            ('PRO_ang_C_CG', np.squeeze(compute_angles(small['c_cg'].reshape(n_frames, -1), pbc_box))),
            ('PRO_r_plane', np.squeeze(compute_ring_plane_angle_continuous(ring_coords, pbc_box))),
            ('PRO_r_plane_arccos', np.squeeze(compute_ring_plane_angle(ring_coords, pbc_box))),
            ('PRO_r_plane_chiral', np.squeeze(compute_ring_plane_angle_chiral(small['ring6'].reshape(n_frames, -1), pbc_box))),
            ('PRO_cen_vec', np.squeeze(compute_vec_cent_angles(tetra_coords, pbc_box))),
            ('PRO_sign_vol', np.squeeze(compute_signed_volume_normalized(tetra_coords, pbc_box))),
            ('G2_ACSF', None),   # filled in below (needs both G2 and G4 at once)
            ('G4_ACSF', None),
        ])

        G2, G4 = compute_acsf(tetra_coords, pbc_box)
        params['G2_ACSF'] = np.squeeze(G2)
        params['G4_ACSF'] = np.squeeze(G4)

        for name, species, ref_i, sel_i in COORD_NUM_SPECS_MAIN:
            sel_key = f"{'dopc' if species == 'DOPC' else 'popc'}_{sel_i}"
            ref_pos = small['cn_ref'][:, ref_i, :]
            distances = group_distances(ref_pos, small[sel_key], pbc_box)
            params[name] = np.squeeze(compute_coordination_number(distances))

        # 'PRO_tetra' sits here in the column order to match the original
        # script's params dict (it inserted it right before the "Additional
        # CVs" coord_num block).
        params['PRO_tetra'] = np.squeeze(compute_tetrahedrality(tetra_coords, pbc_box))

        for name, species, ref_i, sel_i in COORD_NUM_SPECS_EXTRA:
            sel_key = f"{'dopc' if species == 'DOPC' else 'popc'}_{sel_i}"
            ref_pos = small['cn_ref'][:, ref_i, :]
            distances = group_distances(ref_pos, small[sel_key], pbc_box)
            params[name] = np.squeeze(compute_coordination_number(distances))

        params['PRO_hCN_C2u_DOPC'] = np.squeeze(hcn_C2u_DOPC)
        params['PRO_hCN_C2l_DOPC'] = np.squeeze(hcn_C2l_DOPC)
        params['PRO_hCN_O22u_DOPC'] = np.squeeze(hcn_O22u_DOPC)
        params['PRO_hCN_O22l_DOPC'] = np.squeeze(hcn_O22l_DOPC)
        params['PRO_hCN_C2u_POPC'] = np.squeeze(hcn_C2u_POPC)
        params['PRO_hCN_C2l_POPC'] = np.squeeze(hcn_C2l_POPC)
        params['PRO_hCN_Pu'] = np.squeeze(hcn_Pu)
        params['PRO_hCN_Pl'] = np.squeeze(hcn_Pl)
        params['PRO_nCos_C2u_DOPC'] = np.squeeze(ncos_C2u_DOPC)
        params['PRO_nCos_C2l_DOPC'] = np.squeeze(ncos_C2l_DOPC)
        params['PRO_azim_P'] = np.squeeze(azim_P)
        params['PRO_CP_phi2'] = np.squeeze(cp_phi2)
        params['PRO_CP_q2'] = np.squeeze(cp_q2)
        params['Mem_def_u'] = np.squeeze(def_u)
        params['Mem_def_l'] = np.squeeze(def_l)
        params['Mem_thick_loc'] = np.squeeze(locz_u - locz_l)
        params['Water_06'] = np.squeeze(scalars['water_06'] / 3)
        params['Water_10'] = np.squeeze(scalars['water_10'] / 3)

        write_ML_txt(path_number, reactive, ensemble, lambda_A, lambda_minus_one, **params)
        return (path_number, "ok")

    except Exception:
        tb = traceback.format_exc()
        return (path_number, "error", tb)


def loop_over_paths_parallel(path_start, path_end, overwrite, lambda_A, lambda_B, lambda_minus_one):
    """
    Dispatch path processing across N_WORKERS parallel processes.
    """
    os.makedirs(ML, exist_ok=True)

    end = path_start + 1 if path_end is None else path_end + 1
    path_numbers = list(range(path_start, end))

    print(f"Processing paths {path_start} - {path_end} with {N_WORKERS} workers.")
    print("=" * 80)

    results = {"ok": [], "skipped": [], "error": []}

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        future_to_path = {
            executor.submit(
                process_single_path, pn, overwrite, lambda_A, lambda_B, lambda_minus_one
            ): pn
            for pn in path_numbers
        }

        for future in as_completed(future_to_path):
            pn = future_to_path[future]
            try:
                result = future.result()
            except Exception:
                tb = traceback.format_exc()
                result = (pn, "error", tb)

            status = result[1]
            results[status].append(pn)

            if status == "ok":
                print(f"  Path {pn} completed successfully.")
            elif status == "skipped":
                print(f"  [-] Path {pn} skipped: {result[2]}")
            elif status == "error":
                msg = result[2]
                print(f"  Path {pn} FAILED - see {ERROR_LOG}")
                with open(ERROR_LOG, "a") as ef:
                    ef.write(f"\n{'='*60}\n")
                    ef.write(f"PATH {pn} FAILED\n")
                    ef.write(f"{'='*60}\n")
                    ef.write(msg)
                    ef.write("\n")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"  Completed : {len(results['ok'])}   {results['ok']}")
    print(f"  Skipped   : {len(results['skipped'])}   {results['skipped']}")
    print(f"  Failed    : {len(results['error'])}   {results['error']}")
    if results["error"]:
        print(f"  Error details written to: {ERROR_LOG}")
    print("=" * 80)


def generate_cvs(
    # ── Input data ────────────────────────────────────────────────────────
    data: Annotated[str, typer.Option("-data", help="The infretis_data.txt file listing the paths to process.", rich_help_panel=panels.INPUT)] = "infretis_data.txt",
    toml: Annotated[str, typer.Option("-toml", help="The infretis .toml config, read for the interface positions.", rich_help_panel=panels.INPUT)] = "../infretis.toml",
    gromacs_input: Annotated[str, typer.Option("-gromacs-input", help="Directory holding topol.tpr, conf.gro and the .ndx index files that define the atom groups.", rich_help_panel=panels.INPUT)] = "../gromacs_input",

    # ── Dataset construction ──────────────────────────────────────────────
    start: Annotated[Optional[int], typer.Option("-start", help="First path number to process; default = from the beginning.", rich_help_panel=panels.DATASET)] = None,
    end: Annotated[Optional[int], typer.Option("-end", help="Last path number to process; default = to the end.", rich_help_panel=panels.DATASET)] = None,

    # ── Model and training (parallelism) ──────────────────────────────────
    workers: Annotated[int, typer.Option("-workers", help="Worker processes; each handles one path at a time.", rich_help_panel=panels.MODEL)] = 8,

    # ── Output ────────────────────────────────────────────────────────────
    out_dir: Annotated[str, typer.Option("-out-dir", help="Directory for the per-path <path_nr>.txt CV files.", rich_help_panel=panels.OUTPUT)] = ML,
    overwrite: Annotated[bool, typer.Option("-O", help="Recompute paths whose output file already exists.", rich_help_panel=panels.OUTPUT)] = False,
):
    """Compute per-frame CVs from MD trajectories into per-path .txt files.

    Reads each path's .xtc segments once with MDAnalysis and writes the
    per-frame CV table that every other chiroflux command consumes. Needs the
    optional `generate` extra (MDAnalysis) plus the trajectories, topology and
    .ndx files - it is the only command that touches simulation data rather
    than the .txt files produced from it.
    """
    global N_WORKERS, infretis_data_file, ML

    _require_mdanalysis()
    _init_system(gromacs_input)

    N_WORKERS = workers
    infretis_data_file = data
    ML = out_dir
    os.makedirs(ML, exist_ok=True)

    lambda_A, lambda_B, lambda_minus_one = read_toml(toml)
    loop_over_paths_parallel(start, end, overwrite,
                             lambda_A, lambda_B, lambda_minus_one)
