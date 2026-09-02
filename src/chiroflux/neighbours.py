import os
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Optional

import MDAnalysis as mda
import numpy as np
import pandas as pd
import tomli
import typer
from MDAnalysis.analysis import distances

from . import panels

# ── PARALLELISM CONFIGURATION ─────────────────────────────────────────────────
ERROR_LOG  = "errors_neighbour.log"
NEIGH_OUT  = "Neighbour"
PLOT_OUT   = "plots/neighbour"
# ─────────────────────────────────────────────────────────────────────────────

topol_file = '../gromacs_input/topol.tpr'

# ── Bulk membrane composition (5:1 DOPC:POPC) ────────────────────────────────
# These are the expected fractions under the null hypothesis of no preference.
# For n_neighbors=3 nearest lipids drawn independently from the bulk:
#   P(all 3 = DOPC) = (5/6)^3 ≈ 0.5787
#   P(all 3 = POPC) = (1/6)^3 ≈ 0.0046
#   P(Mix)          = 1 - P(3-DOPC) - P(3-POPC) ≈ 0.4167
BULK_FRACS = {
    'DOPC': 5 / 6,
    'POPC': 1 / 6,
}
N_NEIGHBORS_DEFAULT = 3



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


def _null_class_probs(n_neighbors=N_NEIGHBORS_DEFAULT, bulk_fracs=None):
    """
    Compute the expected probabilities of each classification under the null
    hypothesis that neighbours are drawn independently from the bulk composition.

    Returns
    -------
    dict[str, float]  — keys '3-DOPC', '3-POPC', 'Mix'
    """
    if bulk_fracs is None:
        bulk_fracs = BULK_FRACS
    p_dopc = bulk_fracs['DOPC']
    p_popc = bulk_fracs['POPC']
    p_3dopc = p_dopc ** n_neighbors
    p_3popc = p_popc ** n_neighbors
    return {
        '3-DOPC': p_3dopc,
        '3-POPC': p_3popc,
        'Mix':    1.0 - p_3dopc - p_3popc,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Chain-carbon selection (CHARMM36, DOPC / POPC)
# ═════════════════════════════════════════════════════════════════════════════

def get_chain_carbons(lipid_residue):
    return lipid_residue.atoms.select_atoms(
        "name C2* C3*"
    )

# ═════════════════════════════════════════════════════════════════════════════
# Neighbour classification
# ═════════════════════════════════════════════════════════════════════════════

def classify_neighbors(neighbor_resnames):
    counts = Counter(neighbor_resnames)
    if counts.get('DOPC', 0) == 3:
        return '3-DOPC'
    elif counts.get('POPC', 0) == 3:
        return '3-POPC'
    else:
        return 'Mix'


# ═════════════════════════════════════════════════════════════════════════════
# Path helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_path_weights(weights_file='path_weights.txt'):
    """
    Load per-path weights from a two-column whitespace-delimited file:
        path_number (int)   weight (float)
    Returns
    -------
    dict[int, float]
    """
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


def extract_sorted_traj_names(trj_path):
    filenames  = []
    directions = []
    seen       = set()
    g96_index  = None

    with open(trj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            columns   = line.split()
            step      = columns[0]
            filename  = columns[1]
            direction = columns[3]
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


def contains_key(d: dict[int, float], key: int) -> bool:
    # Only include paths in histogram that have weights
    return key in d

# ═════════════════════════════════════════════════════════════════════════════
# Slab grid construction
# ═════════════════════════════════════════════════════════════════════════════

def build_slab_grid(
    topology,
    xtc_files,
    permeant_resname,
    bilayer_normal,
    slab_width,
    slab_range=None,
):
    """
    Construct the slab-edge and slab-center arrays.

    If *slab_range* is provided as (z_min, z_max) those values are used
    directly as the grid boundaries — no trajectory scan is performed.
    If *slab_range* is None the boundaries are determined by scanning all
    frames of every xtc file and adding one slab_width of padding on each side.

    Parameters
    ----------
    topology         : str
    xtc_files        : list of str
    permeant_resname : str
    bilayer_normal   : str   — 'x', 'y', or 'z'
    slab_width       : float — Å
    slab_range       : tuple (z_min, z_max) in Å, or None

    Returns
    -------
    edges   : np.ndarray  — slab boundary positions (length n_slabs + 1)
    centers : np.ndarray  — slab midpoints          (length n_slabs)
    """
    axis = {'x': 0, 'y': 1, 'z': 2}[bilayer_normal.lower()]

    if slab_range is not None:
        z_min, z_max = float(slab_range[0]), float(slab_range[1])
        print(f"  [slab-grid] Using manual range: z_min={z_min:.3f}, "
              f"z_max={z_max:.3f} Å")
    else:
        all_z = []
        for xtc in xtc_files:
            u          = mda.Universe(topology, xtc, refresh_offsets=True)
            perm_heavy = u.select_atoms(
                f"resname {permeant_resname} and not name H*"
            )
            if len(perm_heavy) == 0:
                perm_heavy = u.select_atoms(f"resname {permeant_resname}")
            for _ts in u.trajectory:           # iterate directly, not list()
                all_z.append(perm_heavy.positions[:, axis].copy())
        all_z = np.concatenate(all_z)
        z_min = all_z.min() - slab_width
        z_max = all_z.max() + slab_width
        print(f"  [slab-grid] Auto range: z_min={z_min:.3f}, z_max={z_max:.3f} Å")

    edges   = np.arange(z_min, z_max + slab_width, slab_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


# ═════════════════════════════════════════════════════════════════════════════
# Core MDAnalysis neighbour analysis  (single path)
# ═════════════════════════════════════════════════════════════════════════════

def _precompute_chain_groups(lipid_residues):
    groups = {}
    for res in lipid_residues:
        cc = get_chain_carbons(res)
        if len(cc) > 0:
            groups[res.resindex] = cc
    return groups


def calculate_neighbour_slab_counts(
    topology,
    xtc_files,
    xtc_directions,
    permeant_resname,
    weight           = 1.0,
    first_frame      = False,
    last_frame       = False,
    lipid_resnames   = ('DOPC', 'POPC'),
    slab_width       = 1.0,
    bilayer_normal   = 'z',
    n_neighbors      = 3,
    slab_range       = None,
):
    """
    Accumulate weighted slab counts for one path. No normalisation is applied;
    the raw weighted counts are returned so they can be summed across all paths
    before a single final normalisation in aggregate_neighbour_results().
    Directions are not neccessarily needed as counts are accumulated but are
    passed in anyway.

    In addition to the raw (dwell-time-weighted) counts, this also returns
    path-length-normalised counts (norm_count_*), where every path's total
    contribution sums to `weight` regardless of how many frames it has. This
    gives each path equal say in the pooled statistics, instead of paths
    that linger longer in a slab dominating it.

    Parameters
    ----------
    topology         : str
    xtc_files        : list of str
    xtc_directions   : list of str  ('1' or '-1')
    permeant_resname : str
    weight           : float — path weight applied to every observation
    first_point      : bool  — whether to exclude the first frame (to avoid double-counting)
    last_point       : bool  — whether to exclude the last frame (to avoid double-counting)
    lipid_resnames   : tuple
    slab_width       : float — Å
    bilayer_normal   : str
    n_neighbors      : int
    slab_range       : tuple (z_min, z_max) in Å, or None

    Returns
    -------
    counts_df   : pd.DataFrame
        columns: slab_center,
                 weighted_count_3-DOPC, weighted_count_3-POPC, weighted_count_Mix,
                 norm_count_3-DOPC, norm_count_3-POPC, norm_count_Mix
    raw_records : list of dict
    """
    axis   = {'x': 0, 'y': 1, 'z': 2}[bilayer_normal.lower()]
    labels = ('3-DOPC', '3-POPC', 'Mix')

    # ── Build slab grid ───────────────────────────────────────────────────────
    edges, centers = build_slab_grid(
        topology         = topology,
        xtc_files        = xtc_files,
        permeant_resname = permeant_resname,
        bilayer_normal   = bilayer_normal,
        slab_width       = slab_width,
        slab_range       = slab_range,
    )
    n_slabs     = len(centers)
    slab_counts = {lbl: np.zeros(n_slabs, dtype=float) for lbl in labels}
    raw_records = []
    n_events    = 0   # total classified frames for this path — used to
                       # length-normalise so every path contributes `weight`
                       # in total, regardless of how many frames it has

    # ── Classify neighbours frame by frame ────────────────────────────────────
    for i, (xtc, direction) in enumerate(zip(xtc_files, xtc_directions)):
        u = mda.Universe(topology, xtc, refresh_offsets=True)
        membrane   = u.select_atoms("resname DOPC POPC")
        permeant   = u.select_atoms(f"resname {permeant_resname}")
        perm_heavy = permeant.select_atoms("not name H*")
        if len(perm_heavy) == 0:
            perm_heavy = permeant

        lipid_sel  = " or ".join(f"resname {r}" for r in lipid_resnames)
        all_lipids = u.select_atoms(lipid_sel)
        if len(all_lipids) == 0:
            raise ValueError(
                f"No lipid atoms found for resnames {lipid_resnames} in {xtc}"
            )

        chain_groups          = _precompute_chain_groups(all_lipids.residues)
        lipid_res_with_chains = [
            r for r in all_lipids.residues if r.resindex in chain_groups
        ]

        # Collect frame indices, reverse if needed, then seek explicitly
        n_frames    = len(u.trajectory)
        frame_indices = list(range(n_frames))
        if direction == '-1':
            frame_indices = list(reversed(frame_indices))

        # Adjust for first and last frame if lower than lambda_A or higher than lambda_B
        if first_frame:
            if i == 0:
                frame_indices = frame_indices[1:]
        if last_frame:
            if i == len(xtc_files) - 1:
                frame_indices = frame_indices[:-1]
        # Always remove first frame for "inner" trajectories to avoid overlap
        if i != 0 and i != len(xtc_files) - 1:
            frame_indices = frame_indices[1:]

        for fi in frame_indices:
            u.trajectory[fi]          # seek to this frame — updates all positions

            for perm_res in permeant.residues:
                ph = perm_res.atoms.select_atoms("not name H*")
                if len(ph) == 0:
                    ph = perm_res.atoms

                membrane_z = membrane.positions[:, axis].mean()
                perm_z   =  membrane_z - ph.positions[:, axis].mean()

                min_dists = []
                for res in lipid_res_with_chains:
                    chain_pos = chain_groups[res.resindex].positions
                    dist_mat  = distances.distance_array(
                        ph.positions, chain_pos, box=u.trajectory.ts.dimensions
                    )
                    min_dists.append((dist_mat.min(), res))

                min_dists.sort(key=lambda x: x[0])
                nearest           = min_dists[:n_neighbors]
                neighbor_resnames = [res.resname.upper() for _, res in nearest]
                classification    = classify_neighbors(neighbor_resnames)

                slab_idx = int(np.searchsorted(edges, perm_z, side='right') - 1)
                slab_idx = int(np.clip(slab_idx, 0, n_slabs - 1))
                slab_counts[classification][slab_idx] += weight
                n_events += 1

                raw_records.append({
                    'frame':             fi,
                    'direction':         direction,
                    'xtc':               os.path.basename(xtc),
                    'z_position':        perm_z,
                    'classification':    classification,
                    'neighbor_resnames': neighbor_resnames,
                    'min_distances':     [d for d, _ in nearest],
                    'weight':            weight,
                })

    # ── Path-length-normalised counts ─────────────────────────────────────────
    # Divides out the number of frames this path contributed, so this path's
    # columns sum to `weight` in total — every path counts equally, instead
    # of paths with more frames (longer dwell time) dominating the sum.
    norm_counts = {
        lbl: (slab_counts[lbl] / n_events if n_events > 0 else slab_counts[lbl] * 0.0)
        for lbl in labels
    }

    # ── Return raw weighted counts (no normalisation) + normalised counts ────
    counts_df = pd.DataFrame({
        'slab_center':              centers,
        'weighted_count_3-DOPC':   slab_counts['3-DOPC'],
        'weighted_count_3-POPC':   slab_counts['3-POPC'],
        'weighted_count_Mix':      slab_counts['Mix'],
        'norm_count_3-DOPC':       norm_counts['3-DOPC'],
        'norm_count_3-POPC':       norm_counts['3-POPC'],
        'norm_count_Mix':          norm_counts['Mix'],
    })
    return counts_df, raw_records


# ═════════════════════════════════════════════════════════════════════════════
# Per-path worker
# ═════════════════════════════════════════════════════════════════════════════

def _csv_path(path_number):
    return Path(NEIGH_OUT) / f"{path_number}_neighbour.csv"


def remove_first_last_frames(ensemble, lambda_minus_one, lambda_A, first, last):
    """
    Remove the first and last frame from the trajectory data to avoid double-counting
    due to frame overlap between consecutive xtc files.
    """
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


def classify_files(path_start, path_end, weights):
    plus_reactive = []
    plus_non_reactive = []
    minus = []

    for pn in range(path_start, path_end + 1):
        csv = _csv_path(pn)
        if csv.exists() and contains_key(weights, pn):
            with open(csv, 'r') as f:
                header1 = f.readline().strip()
                header2 = f.readline().strip()
            if "non-reactive" in header1:
                reactivity = "non-reactive"
            elif "reactive" in header1:
                reactivity = "reactive"
            else:
                reactivity = None

            if "plus" in header2:
                ensemble = "plus"
            elif "minus" in header2:
                ensemble = "minus"
            else:
                ensemble = None

            if reactivity == "non-reactive" and ensemble == "plus":
                plus_non_reactive.append(csv)
            elif reactivity == "reactive" and ensemble == "plus":
                plus_reactive.append(csv)
            elif ensemble == "minus":
                minus.append(csv)
    return plus_reactive, plus_non_reactive, minus


def process_single_path_neighbour(
    path_number,
    overwrite,
    weights,
    lambda_A,
    lambda_B,
    lambda_minus_one,
    infretis_data_file,
    permeant_resname = 'ORP',
    lipid_resnames   = ('DOPC', 'POPC'),
    slab_width       = 1.0,
    bilayer_normal   = 'z',
    slab_range       = None,
):
    """
    Process neighbour analysis for one path. Writes a CSV containing only
    raw weighted counts (no normalisation).

    Parameters
    ----------
    path_number      : int
    overwrite        : bool
    weights          : dict[int, float]
    lambda_A         : float
    lambda_B         : float
    lambda_minus_one : float or None
    infretis_data_file : str
    permeant_resname : str
    lipid_resnames   : tuple
    slab_width       : float
    bilayer_normal   : str
    slab_range       : tuple (z_min, z_max) or None
    """
    out_csv = _csv_path(path_number)

    if not overwrite and out_csv.exists():
        return (path_number, 'skipped', 'Output CSV already exists.')

    path_folder = f"../load/{path_number}/accepted/"
    if not os.path.exists(path_folder):
        return (path_number, 'skipped',
                f'Path folder does not exist: {path_folder}')

    try:
        order = np.loadtxt(f"../load/{path_number}/order.txt", comments=('#', '@'))
        traj  = np.loadtxt(f"../load/{path_number}/traj.txt",  comments=('#', '@'), usecols=(2,))
        order_parameter = np.column_stack((order, traj))
        first = order_parameter[0, 1]
        last = order_parameter[-1, 1]
        if ((lambda_minus_one < first < lambda_A) or
                (lambda_minus_one < last < lambda_A)):
            ensemble = "plus"
        else:
            ensemble = "minus"

        reactive = get_reactive_paths(path_number, infretis_data_file, lambda_B)
        if reactive is None:
            return (path_number, "skipped", "Path not found in infretis_data.txt")

        weight = weights.get(path_number, None)
        if weight is None:
            print(f"  [neigh-worker] WARNING: path {path_number} not found in "
                  f"weights file — defaulting to weight=1.0")
            weight = 1.0

        #Check to see if first or last frame should be deleted
        first_frame, last_frame = remove_first_last_frames(ensemble, lambda_minus_one, lambda_A, first, last)

        xtc_names, directions, _ = extract_sorted_traj_names(
            f"../load/{path_number}/traj.txt"
        )
        xtc_files = [os.path.join(path_folder, f) for f in xtc_names]

        if len(xtc_files) == 0:
            return (path_number, 'skipped', 'No xtc files found.')

        print(f"[neigh-worker] Path {path_number} (weight={weight:.6e}): "
              f"{len(xtc_files)} xtc file(s)"
              + (f", slab_range={slab_range}" if slab_range else ", slab_range=auto"))

        counts_df, _ = calculate_neighbour_slab_counts(
            topology         = topol_file,
            xtc_files        = xtc_files,
            xtc_directions   = directions,
            permeant_resname = permeant_resname,
            weight           = weight,
            first_frame      = first_frame,
            last_frame       = last_frame,
            lipid_resnames   = lipid_resnames,
            slab_width       = slab_width,
            bilayer_normal   = bilayer_normal,
            slab_range       = slab_range,
        )

        with open(out_csv, 'w') as f:
            f.write(f"# {'reactive' if reactive else 'non-reactive'}\n")
            f.write(f"# {ensemble} ensemble\n")

        counts_df.to_csv(out_csv, mode='a', index=False, float_format='%.6e')
        print(f"  [neigh-worker] Path {path_number} -> {out_csv}")

        return (path_number, 'ok')

    except Exception:
        return (path_number, 'error', traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
# Parallel dispatcher
# ═════════════════════════════════════════════════════════════════════════════

def loop_over_paths_neighbour_parallel(
    path_start,
    path_end,
    overwrite          = False,
    weights_file       = 'path_weights.txt',
    data_file          = None,
    n_workers          = 32,
    permeant_resname   = 'ORP',
    lipid_resnames     = ('DOPC', 'POPC'),
    slab_width         = 1.0,
    bilayer_normal     = 'z',
    slab_range         = None,
):
    """
    Dispatch neighbour analysis across N_WORKERS parallel processes.

    Parameters
    ----------
    slab_range : tuple (z_min, z_max) in Å, or None
        When provided every worker uses the same fixed grid.  Strongly
        recommended for correct aggregation — a shared grid means per-path
        CSVs are directly column-summable with no interpolation needed.
    """
    Path(NEIGH_OUT).mkdir(parents=True, exist_ok=True)

    lambda_A, lambda_B, lambda_minus_one = read_toml('../infretis.toml')

    weights      = load_path_weights(weights_file)
    end          = path_start + 1 if path_end is None else path_end + 1
    path_numbers = list(range(path_start, end))

    print(f"Neighbour analysis: paths {path_start}–{path_end}, {n_workers} workers.")
    print(f"Weights loaded from '{weights_file}': {len(weights)} entries.")
    if slab_range:
        print(f"Manual slab range: {slab_range[0]} – {slab_range[1]} Å")
    else:
        print("Slab range: auto (determined per path from trajectory data)")
    print("=" * 80)

    results = {'ok': [], 'skipped': [], 'error': []}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_path = {
            executor.submit(
                process_single_path_neighbour,
                pn, overwrite, weights, lambda_A, lambda_B, lambda_minus_one,
                data_file, permeant_resname, lipid_resnames,
                slab_width, bilayer_normal, slab_range,
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
                    ef.write('\n')

    print("\n" + "=" * 80)
    print("NEIGHBOUR ANALYSIS SUMMARY")
    print(f"  Completed : {len(results['ok'])}   {results['ok']}")
    print(f"  Skipped   : {len(results['skipped'])}   {results['skipped']}")
    print(f"  Failed    : {len(results['error'])}   {results['error']}")
    if results['error']:
        print(f"  Error details written to: {ERROR_LOG}")
    print("=" * 80)


# ═════════════════════════════════════════════════════════════════════════════
# Aggregation across all paths  (single normalisation happens here)
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_neighbour_results(path_start, path_end, weights, count_prefix='weighted_count_'):
    """
    Sum the raw weighted counts from every per-path CSV onto a common slab
    grid, then apply a single normalisation so that 3-DOPC + 3-POPC + Mix
    sums to 1.0 in every slab.

    count_prefix : str
        'weighted_count_' (default) — dwell-time-weighted, paths that linger
        in a slab dominate it.
        'norm_count_' — path-length-normalised, every path contributes
        equally regardless of how many frames it has.

    Returns
    -------
    pd.DataFrame  columns: slab_center, 3-DOPC, 3-POPC, Mix
    """
    labels  = ('3-DOPC', '3-POPC', 'Mix')
    all_dfs = []

    for pn in range(path_start, path_end + 1):
        csv = _csv_path(pn)
        if csv.exists() and contains_key(weights, pn):
            all_dfs.append(pd.read_csv(csv, comment='#'))
        else:
            print(f"  [aggregate] Path {pn}: CSV not found or no weight available, skipping.")

    if not all_dfs:
        raise RuntimeError("No neighbour CSVs found to aggregate.")
    return pool_results(all_dfs, labels, count_prefix=count_prefix)


def pool_results(all_dfs, labels, count_prefix='weighted_count_'):
    # ── Build a common slab grid from the union of all slab centers ───────────
    all_centers = np.unique(
        np.round(np.concatenate([df['slab_center'].values for df in all_dfs]), 6)
    )

    # ── Sum weighted counts across paths ──────────────────────────────────────
    pooled_counts = {lbl: np.zeros(len(all_centers), dtype=float) for lbl in labels}

    for df in all_dfs:
        centers_rounded = np.round(df['slab_center'].values, 6)
        idx_map         = np.searchsorted(all_centers, centers_rounded)
        for lbl in labels:
            col = f'{count_prefix}{lbl}'
            if col in df.columns:
                np.add.at(pooled_counts[lbl], idx_map, df[col].values)

    # ── Single normalisation over the fully pooled counts ─────────────────────
    total   = sum(pooled_counts[lbl] for lbl in labels)
    nonzero = total > 0

    result = {'slab_center': all_centers}
    for lbl in labels:
        prob          = np.zeros(len(all_centers))
        prob[nonzero] = pooled_counts[lbl][nonzero] / total[nonzero]
        result[lbl]   = prob

    return pd.DataFrame(result)


def aggregate_ensemble_reactivity_specific_results(path_start, path_end, weights):
    """
    Similar to aggregate_neighbour_results() but only sums paths of a specific ensemble
    (e.g. "plus" or "minus") and reactive/non-reactive status.

    Returns
    -------
    pd.DataFrame  columns: slab_center, 3-DOPC, 3-POPC, Mix
    """
    labels  = ('3-DOPC', '3-POPC', 'Mix')
    plus_reactive_data = []
    plus_non_reactive_data = []
    minus_data = []

    plus_reactive, plus_non_reactive, minus = classify_files(path_start, path_end, weights)
    for csv in plus_reactive:
        df = pd.read_csv(csv, comment='#')
        plus_reactive_data.append(df)
    if plus_reactive_data:
        results = pool_results(plus_reactive_data, labels)
        results.to_csv(str(Path(PLOT_OUT) / "neighbours_plus_reactive.csv"), index=False, float_format='%.6e')
        plot_neighbour_probabilities(results, output_path=str(
                Path(PLOT_OUT) / "neighbours_plus_reactive.png"
            ))

    for csv in plus_non_reactive:
        df = pd.read_csv(csv, comment='#')
        plus_non_reactive_data.append(df)
    if plus_non_reactive_data:
        results = pool_results(plus_non_reactive_data, labels)
        results.to_csv(str(Path(PLOT_OUT) / "neighbours_plus_non_reactive.csv"), index=False, float_format='%.6e')
        plot_neighbour_probabilities(results, output_path=str(
                Path(PLOT_OUT) / "neighbours_plus_non_reactive.png"
            ))

    for csv in minus:
        df = pd.read_csv(csv, comment='#')
        minus_data.append(df)
    if minus_data:
        results = pool_results(minus_data, labels)
        results.to_csv(str(Path(PLOT_OUT) / "neighbours_minus.csv"), index=False, float_format='%.6e')
        plot_neighbour_probabilities(results, output_path=str(
                Path(PLOT_OUT) / "neighbours_minus.png"
            ))


# ═════════════════════════════════════════════════════════════════════════════
# Statistical enrichment analysis
# ═════════════════════════════════════════════════════════════════════════════

def compute_enrichment_statistics(
    path_start,
    path_end,
    weights,
    n_neighbors    = N_NEIGHBORS_DEFAULT,
    bulk_fracs     = None,
    n_bootstrap    = 2000,
    min_count      = 5.0,
    alpha          = 0.05,
    label_subset   = None,
    count_prefix   = 'weighted_count_',
    path_dfs       = None,
):
    """
    Compute per-slab enrichment ratios and significance for each classification
    label relative to the null hypothesis derived from bulk membrane composition.

    Strategy
    --------
    The fundamental challenge is that paths are *correlated* (path sampling
    produces a Markov chain of paths).  Treating individual frames as independent
    observations massively inflates the effective sample size and produces
    spuriously low p-values.  Instead we treat each **path** as one independent
    observation and use **path-level block bootstrap** to estimate uncertainty.

    For each slab s and label l:
      1. Compute the observed weighted probability  p_obs[s,l].
      2. Compute the null probability  p_null[l] from the bulk composition.
      3. Compute the enrichment ratio  E[s,l] = p_obs[s,l] / p_null[l].
      4. Bootstrap by resampling paths with replacement (weights retained),
         recomputing the full pool_results() for each resample, to obtain
         E_boot[s,l,b] for b = 1 … n_bootstrap.
      5. Report 95 % CI (percentile method) and a two-sided p-value
         (fraction of bootstrap samples with |E_boot - 1| >= |E_obs - 1|).

    Parameters
    ----------
    path_start   : int
    path_end     : int
    weights      : dict[int, float]
    n_neighbors  : int   — must match the value used during trajectory analysis
    bulk_fracs   : dict  — {'DOPC': float, 'POPC': float}; defaults to 5:1
    n_bootstrap  : int   — number of bootstrap resamples (>=1000 recommended)
    min_count    : float — minimum total weighted count per slab to report
                   (scale depends on count_prefix — see below)
    alpha        : float — significance level for the confidence interval
    label_subset : list  — restrict output to these labels; None = all three
    count_prefix : str   — 'weighted_count_' (default) for dwell-time-weighted
                   statistics (paths that linger in a slab dominate it), or
                   'norm_count_' for path-length-normalised statistics (every
                   path contributes equally regardless of dwell time). The
                   two live on very different scales, so min_count should be
                   chosen accordingly (norm_count totals sum to ~n_paths at
                   most, vs. weighted_count totals which scale with dwell time).
    path_dfs     : list of (path_number, df), optional — reuse already-loaded
                   path CSVs (e.g. to compute both count_prefix variants
                   without reading every CSV twice) instead of loading them
                   from disk again.

    Returns
    -------
    stats_df : pd.DataFrame
        One row per (slab_center, label) with columns:
          slab_center, label,
          p_obs, p_null, enrichment,
          ci_lo, ci_hi,
          p_value, significant,
          significant_ci — standard percentile-CI test (ci excludes E=1);
              prefer this over `significant` for well-sampled slabs — see
              the note above the `significant_ci` computation for why
              `significant` is biased toward p≈0.5 and underpowered there.
          total_weighted_count
    """
    labels = ('3-DOPC', '3-POPC', 'Mix')
    if label_subset is not None:
        labels = tuple(l for l in labels if l in label_subset)

    null_probs = _null_class_probs(n_neighbors=n_neighbors, bulk_fracs=bulk_fracs)

    # ── Load all path DataFrames and build a common slab grid ─────────────────
    if path_dfs is None:
        path_dfs = []   # list of (path_number, df)
        for pn in range(path_start, path_end + 1):
            csv = _csv_path(pn)
            if csv.exists() and contains_key(weights, pn):
                path_dfs.append((pn, pd.read_csv(csv, comment='#')))
            else:
                print(f"  [stats] Path {pn}: CSV not found or no weight — skipping.")

    if not path_dfs:
        raise RuntimeError("No neighbour CSVs found for statistical analysis.")

    n_paths = len(path_dfs)
    print(f"  [stats] {n_paths} paths loaded for bootstrap analysis "
          f"(count_prefix='{count_prefix}').")

    all_dfs     = [df for _, df in path_dfs]
    obs_pooled  = pool_results(all_dfs, labels, count_prefix=count_prefix)
    all_centers = obs_pooled['slab_center'].values

    # ── Total weighted counts per slab (for the min_count filter) ────────────
    # Recompute from the raw counts so we have unscaled totals.
    count_cols = [f'{count_prefix}{lbl}' for lbl in ('3-DOPC', '3-POPC', 'Mix')]
    total_weighted = np.zeros(len(all_centers))
    for df in all_dfs:
        centers_r = np.round(df['slab_center'].values, 6)
        idx_map   = np.searchsorted(np.round(all_centers, 6), centers_r)
        for col in count_cols:
            if col in df.columns:
                np.add.at(total_weighted, idx_map, df[col].values)

    # ── Bootstrap over paths ──────────────────────────────────────────────────
    rng          = np.random.default_rng(seed=42)
    boot_results = {lbl: np.full((n_bootstrap, len(all_centers)), np.nan)
                    for lbl in labels}

    for b in range(n_bootstrap):
        boot_idx   = rng.integers(0, n_paths, size=n_paths)
        boot_dfs   = [all_dfs[i] for i in boot_idx]
        boot_pool  = pool_results(boot_dfs, labels, count_prefix=count_prefix)

        # Align to the common grid (some slabs may be missing in a resample)
        boot_centers_r = np.round(boot_pool['slab_center'].values, 6)
        idx_map        = np.searchsorted(np.round(all_centers, 6), boot_centers_r)
        for lbl in labels:
            boot_results[lbl][b, idx_map] = boot_pool[lbl].values

    # ── Assemble output table ─────────────────────────────────────────────────
    records = []
    ci_lo_q = alpha / 2
    ci_hi_q = 1.0 - alpha / 2

    for s, z in enumerate(all_centers):
        if total_weighted[s] < min_count:
            continue
        for lbl in labels:
            p_obs  = obs_pooled[lbl].values[s]
            p_null = null_probs[lbl]

            # Avoid division by zero for labels with p_null = 0
            if p_null == 0:
                enrichment = np.inf if p_obs > 0 else np.nan
            else:
                enrichment = p_obs / p_null

            boot_e   = boot_results[lbl][:, s] / p_null if p_null > 0 else np.full(n_bootstrap, np.nan)
            valid    = ~np.isnan(boot_e)
            n_valid  = valid.sum()

            if n_valid < 10:
                ci_lo = ci_hi = p_val = np.nan
            else:
                ci_lo = float(np.nanpercentile(boot_e, ci_lo_q * 100))
                ci_hi = float(np.nanpercentile(boot_e, ci_hi_q * 100))
                # Two-sided p-value: fraction of bootstrap enrichments at least
                # as extreme as observed, under H0: E = 1
                obs_dev  = abs(enrichment - 1.0)
                boot_dev = np.abs(boot_e[valid] - 1.0)
                p_val    = float((boot_dev >= obs_dev).mean())
                # Clamp to [1/n_bootstrap, 1] to avoid exact zeros
                p_val    = max(p_val, 1.0 / n_bootstrap)

            # significant_ci: standard nonparametric bootstrap test — the 95%
            # percentile CI excludes the null value (E=1). Kept alongside the
            # original `significant` (deviation-vs-1 p-value test) rather than
            # replacing it: that p-value compares the observed deviation from 1
            # to the bootstrap resamples' OWN deviation from 1, but those
            # resamples are centered on the observed estimate itself, not on
            # the null — so roughly half of them land on the far side of the
            # estimate from 1 by construction, regardless of how far the
            # estimate truly is from 1. That makes `significant` biased toward
            # p≈0.5 and largely powerless once a slab is reasonably well
            # sampled. `significant_ci` doesn't have that issue.
            significant_ci = (
                (ci_hi < 1.0) or (ci_lo > 1.0)
                if not (np.isnan(ci_lo) or np.isnan(ci_hi))
                else False
            )

            records.append({
                'slab_center':          z,
                'label':                lbl,
                'p_obs':                p_obs,
                'p_null':               p_null,
                'enrichment':           enrichment,
                'ci_lo':                ci_lo,
                'ci_hi':                ci_hi,
                'p_value':              p_val,
                'significant':          (p_val < alpha) if not np.isnan(p_val) else False,
                'significant_ci':       significant_ci,
                'total_weighted_count': total_weighted[s],
            })

    stats_df = pd.DataFrame(records)
    print(f"  [stats] Bootstrap complete ({n_bootstrap} resamples, {n_paths} paths).")
    return stats_df


def print_enrichment_summary(stats_df, alpha=0.05):
    """
    Print a readable summary of the enrichment statistics,
    collapsed across slabs (global test) and per slab.
    """
    print("\n" + "=" * 70)
    print("ENRICHMENT SUMMARY  (E > 1: enriched near permeant, E < 1: depleted)")
    print(f"Significance level: α = {alpha}")
    print("=" * 70)

    for lbl in stats_df['label'].unique():
        sub = stats_df[stats_df['label'] == lbl].copy()
        sig = sub[sub['significant_ci']]
        print(f"\n  {lbl}")
        print(f"    Mean enrichment (all slabs)  : {sub['enrichment'].mean():.3f}")
        print(f"    Median enrichment            : {sub['enrichment'].median():.3f}")
        print(f"    Slabs with significant E     : {len(sig)} / {len(sub)}")
        if len(sig) > 0:
            print(f"    Significant slab z-positions : "
                  f"{sig['slab_center'].values.tolist()}")
            print(f"    Enrichment range in sig slabs: "
                  f"{sig['enrichment'].min():.3f} – {sig['enrichment'].max():.3f}")

    print("\n" + "=" * 70)


# ═════════════════════════════════════════════════════════════════════════════
# Optional plot helper
# ═════════════════════════════════════════════════════════════════════════════

def plot_neighbour_probabilities(df, output_path=None):
    import matplotlib.pyplot as plt

    colors = {'3-DOPC': '#1f77b4', '3-POPC': '#ff7f0e', 'Mix': '#2ca02c'}
    fig, ax = plt.subplots(figsize=(8, 4))
    for lbl, color in colors.items():
        ax.plot(df['slab_center'], df[lbl], label=lbl, color=color, lw=2)

    ax.set_xlabel('z-displacement (Å)')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    return fig


def plot_enrichment(stats_df, output_path=None, alpha=0.05, title_suffix=''):
    """
    Plot per-slab enrichment ratios with bootstrap confidence intervals.
    A horizontal dashed line at E=1 marks the null expectation.
    Slabs with significant enrichment/depletion are marked with filled symbols.

    Parameters
    ----------
    stats_df     : pd.DataFrame — output of compute_enrichment_statistics()
    output_path  : str or None
    alpha        : float — significance level used to label the plot
    title_suffix : str — appended to each subplot title, e.g. to distinguish
                   dwell-time-weighted vs. path-length-normalised statistics
    """
    import matplotlib.pyplot as plt

    colors = {'3-DOPC': '#1f77b4', '3-POPC': '#ff7f0e', 'Mix': '#2ca02c'}
    null_probs = _null_class_probs()

    fig, axes = plt.subplots(
        1, len(stats_df['label'].unique()),
        figsize=(5 * len(stats_df['label'].unique()), 4),
        sharey=False,
    )
    if len(stats_df['label'].unique()) == 1:
        axes = [axes]

    for ax, lbl in zip(axes, stats_df['label'].unique()):
        sub  = stats_df[stats_df['label'] == lbl].sort_values('slab_center')
        z    = sub['slab_center'].values
        E    = sub['enrichment'].values
        cilo = sub['ci_lo'].values
        cihi = sub['ci_hi'].values
        sig  = sub['significant_ci'].values
        color = colors.get(lbl, 'grey')

        # Confidence band
        ax.fill_between(z, cilo, cihi, color=color, alpha=0.20, label='95% CI')
        # Enrichment line
        ax.plot(z, E, color=color, lw=2, label=lbl)
        # Null expectation
        ax.axhline(1.0, color='black', lw=1, ls='--', label='Null (bulk ratio)')
        # Significant points
        if sig.any():
            ax.scatter(z[sig], E[sig], color=color, s=40, zorder=5,
                       marker='*', label=f'p < {alpha}')

        ax.set_xlabel('z-displacement (Å)')
        ax.set_ylabel('Enrichment ratio E')
        ax.set_title(f'{lbl}{title_suffix}')
        ax.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"  [plot] Enrichment plot saved to {output_path}")
    else:
        plt.show()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

def neighbours(
    # ── Input data ────────────────────────────────────────────
    weights: Annotated[str, typer.Option("-weights", "--weights", help="", rich_help_panel=panels.INPUT)] = 'path_weights.txt',
    data: Annotated[str, typer.Option("-data", "--data", help="", rich_help_panel=panels.INPUT)] = '../infretis_data_17.txt',
    permeant: Annotated[str, typer.Option("-permeant", "--permeant", help="", rich_help_panel=panels.INPUT)] = 'ORP',

    # ── Dataset construction ──────────────────────────────────
start: Annotated[int, typer.Option("-start", "--start", help="", rich_help_panel=panels.DATASET)] = ...,
    end: Annotated[Optional[int], typer.Option("-end", "--end", help="", rich_help_panel=panels.DATASET)] = None,

    # ── CV corrections: representation ────────────────────────
    slab_width: Annotated[float, typer.Option("-slab-width", "--slab-width", help="", rich_help_panel=panels.REPR)] = 1.0,
    bilayer_normal: Annotated[str, typer.Option("-bilayer-normal", "--bilayer-normal", help="", rich_help_panel=panels.REPR)] = 'z',
    slab_range_opt: Annotated[Optional[str], typer.Option("-slab-range", "--slab-range", help="Fixed slab range in Angstrom as 'min,max', e.g. '-40,40'.", rich_help_panel=panels.REPR)] = None,

    # ── Model and training ────────────────────────────────────
    n_workers: Annotated[int, typer.Option("-n-workers", "--n-workers", help="", rich_help_panel=panels.MODEL)] = 20,
    n_bootstrap: Annotated[int, typer.Option("-n-bootstrap", "--n-bootstrap", help="Number of path-level bootstrap resamples (default: 5000).", rich_help_panel=panels.MODEL)] = 5000,
    alpha: Annotated[float, typer.Option("-alpha", "--alpha", help="Significance level for bootstrap test (default: 0.05).", rich_help_panel=panels.MODEL)] = 0.05,
    min_count: Annotated[float, typer.Option("-min-count", "--min-count", help="Minimum total weighted count per slab to include (default: 0.0005).", rich_help_panel=panels.MODEL)] = 0.0005,
    min_count_equal_path: Annotated[float, typer.Option("-min-count-equal-path", "--min-count-equal-path", help="Minimum total path-length-normalised count per slab to include for the equal-path-weighted statistics (default: 0.0). This is on a totally different scale from --min-count, since it is not inflated by dwell time — normalised path contributions sum to at most their path weight.", rich_help_panel=panels.MODEL)] = 0.0,
    bulk_dopc: Annotated[float, typer.Option("-bulk-dopc", "--bulk-dopc", help="Bulk DOPC mole fraction for null hypothesis (default: 5/6).", rich_help_panel=panels.MODEL)] = '5 / 6',

    # ── Output ────────────────────────────────────────────────
    overwrite: Annotated[bool, typer.Option("-overwrite", "--overwrite", help="", rich_help_panel=panels.OUTPUT)] = False,
    plot: Annotated[bool, typer.Option("-plot", "--plot", help="", rich_help_panel=panels.OUTPUT)] = False,
    stats: Annotated[bool, typer.Option("-stats", "--stats", help="Compute enrichment statistics and bootstrap CI after aggregation.", rich_help_panel=panels.OUTPUT)] = False,
):
    """Lipid neighbour composition and enrichment around the permeant, per membrane slab."""
    slab_range_opt = _split_values(slab_range_opt, "-slab-range", float, 2)
    # These were module-level assignments in the original __main__ block, so
    # helper functions below still read them as globals. Declaring them here
    # keeps that working now that the block lives inside a function.
    global slab_range

    args = SimpleNamespace(
        start=start,
        end=end,
        n_workers=n_workers,
        overwrite=overwrite,
        weights=weights,
        data=data,
        permeant=permeant,
        slab_width=slab_width,
        bilayer_normal=bilayer_normal,
        slab_range=slab_range_opt,
        plot=plot,
        stats=stats,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        min_count=min_count,
        min_count_equal_path=min_count_equal_path,
        bulk_dopc=bulk_dopc,
    )

    import argparse

    parser = argparse.ArgumentParser(
        description='Parallel weighted nearest-neighbour lipid analysis.'
    )
    parser.add_argument('-s', '--start',         type=int,   required=True)
    parser.add_argument('-e', '--end',           type=int,   default=None)
    parser.add_argument('-n', '--n-workers',    type=int,   default=20)
    parser.add_argument('--overwrite',           action='store_true', default=False)
    parser.add_argument('--weights',             type=str,   default='path_weights.txt')
    parser.add_argument('--data',                type=str,   default='../infretis_data_17.txt')
    parser.add_argument('--permeant',            type=str,   default='ORP')
    parser.add_argument('--slab-width',          type=float, default=1.0)
    parser.add_argument('--bilayer-normal',      type=str,   default='z')
    parser.add_argument('--slab-range',          type=float, nargs=2,
                        metavar=('Z_MIN', 'Z_MAX'), default=(-25,25),
                        help='Fixed slab range in Å, e.g. --slab-range -40 40.')
    parser.add_argument('--plot',                action='store_true', default=False)
    # ── new statistics flags ──────────────────────────────────────────────────
    parser.add_argument('--stats',               action='store_true', default=False,
                        help='Compute enrichment statistics and bootstrap CI after aggregation.')
    parser.add_argument('--n-bootstrap',         type=int,   default=5000,
                        help='Number of path-level bootstrap resamples (default: 5000).')
    parser.add_argument('--alpha',               type=float, default=0.05,
                        help='Significance level for bootstrap test (default: 0.05).')
    parser.add_argument('--min-count',           type=float, default=0.0005,
                        help='Minimum total weighted count per slab to include (default: 0.0005).')
    parser.add_argument('--min-count-equal-path', type=float, default=0.0,
                        help='Minimum total path-length-normalised count per slab to include '
                             'for the equal-path-weighted statistics (default: 0.0). This is '
                             'on a totally different scale from --min-count, since it is not '
                             'inflated by dwell time — normalised path contributions sum to '
                             'at most their path weight.')
    parser.add_argument('--bulk-dopc',           type=float, default=5/6,
                        help='Bulk DOPC mole fraction for null hypothesis (default: 5/6).')
    args = parser.parse_args()

    slab_range = tuple(args.slab_range) if args.slab_range is not None else None

    loop_over_paths_neighbour_parallel(
        path_start       = args.start,
        path_end         = args.end,
        overwrite        = args.overwrite,
        weights_file     = args.weights,
        data_file        = args.data,
        n_workers        = args.n_workers,
        permeant_resname = args.permeant,
        lipid_resnames   = ('DOPC', 'POPC'),
        slab_width       = args.slab_width,
        bilayer_normal   = args.bilayer_normal,
        slab_range       = slab_range,
    )

    if args.plot or args.stats:
        os.makedirs(PLOT_OUT, exist_ok=True)
        end     = args.end if args.end is not None else args.start
        weights = load_path_weights(args.weights)

    if args.plot:
        print("Aggregating results (dwell-time-weighted) ...")
        agg = aggregate_neighbour_results(args.start, end, weights)
        out = os.path.join(PLOT_OUT, "neighbour_aggregate.csv")
        agg.to_csv(out, index=False, float_format='%.6e')
        print(f"Aggregate profile written to {out}")
        plot_neighbour_probabilities(agg, output_path=str(
                os.path.join(PLOT_OUT, "neighbours.png")
            ))

        print("Aggregating results (equal-path-weighted, dwell-time removed) ...")
        agg_eq = aggregate_neighbour_results(args.start, end, weights,
                                              count_prefix='norm_count_')
        out_eq = os.path.join(PLOT_OUT, "neighbour_aggregate_equal_path.csv")
        agg_eq.to_csv(out_eq, index=False, float_format='%.6e')
        print(f"Equal-path aggregate profile written to {out_eq}")
        plot_neighbour_probabilities(agg_eq, output_path=str(
                os.path.join(PLOT_OUT, "neighbours_equal_path.png")
            ))

        aggregate_ensemble_reactivity_specific_results(args.start, end, weights)

    if args.stats:
        print("\nRunning enrichment statistics ...")
        bulk_fracs = {
            'DOPC': args.bulk_dopc,
            'POPC': 1.0 - args.bulk_dopc,
        }

        # Load every path CSV once and reuse it for both weighting schemes —
        # avoids reading (potentially thousands of) CSVs from disk twice.
        path_dfs = []
        for pn in range(args.start, end + 1):
            csv = _csv_path(pn)
            if csv.exists() and contains_key(weights, pn):
                path_dfs.append((pn, pd.read_csv(csv, comment='#')))
            else:
                print(f"  [stats] Path {pn}: CSV not found or no weight — skipping.")

        stats_df = compute_enrichment_statistics(
            path_start   = args.start,
            path_end     = end,
            weights      = weights,
            n_bootstrap  = args.n_bootstrap,
            bulk_fracs   = bulk_fracs,
            min_count    = args.min_count,
            alpha        = args.alpha,
            count_prefix = 'weighted_count_',
            path_dfs     = path_dfs,
        )
        stats_out = os.path.join(PLOT_OUT, "enrichment_statistics.csv")
        stats_df.to_csv(stats_out, index=False, float_format='%.6e')
        print(f"Enrichment statistics written to {stats_out}")

        print_enrichment_summary(stats_df, alpha=args.alpha)

        enrich_plot = os.path.join(PLOT_OUT, "enrichment_ratios.png")
        plot_enrichment(stats_df, output_path=enrich_plot, alpha=args.alpha,
                         title_suffix=' (dwell-time-weighted)')

        # ── Equal-path-weighted variant ────────────────────────────────────
        # Every path contributes equally regardless of how many frames it
        # spent in a given slab, instead of long-dwelling paths dominating.
        print("\nRunning enrichment statistics (equal-path-weighted) ...")
        stats_df_eq = compute_enrichment_statistics(
            path_start   = args.start,
            path_end     = end,
            weights      = weights,
            n_bootstrap  = args.n_bootstrap,
            bulk_fracs   = bulk_fracs,
            min_count    = args.min_count_equal_path,
            alpha        = args.alpha,
            count_prefix = 'norm_count_',
            path_dfs     = path_dfs,
        )
        stats_out_eq = os.path.join(PLOT_OUT, "enrichment_statistics_equal_path.csv")
        stats_df_eq.to_csv(stats_out_eq, index=False, float_format='%.6e')
        print(f"Equal-path enrichment statistics written to {stats_out_eq}")

        print_enrichment_summary(stats_df_eq, alpha=args.alpha)

        enrich_plot_eq = os.path.join(PLOT_OUT, "enrichment_ratios_equal_path.png")
        plot_enrichment(stats_df_eq, output_path=enrich_plot_eq, alpha=args.alpha,
                         title_suffix=' (equal-path-weighted)')
