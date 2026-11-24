"""Compare the membrane around the permeant between two path classes.

``chiroflux membrane-spatial`` writes one file per path with unweighted sums
binned by permeant depth, and stores the path's class (reactive, unreactive,
[0-]) next to them. ``-paths`` there aggregates one class at a time; this
command puts two classes side by side - by default the reactive against the
unreactive plus paths - and asks where, and in what, they differ.

Why depth-matched only:

* Reactive paths reach deeper than unreactive ones, so any average over depth
  differs between the classes for that reason alone. Every comparison here is
  made bin by bin on the permeant-depth axis, and only where both classes
  have enough independent paths (``-min-neff``).
* The radial phosphate deformation and the water z-density are therefore
  compared in their depth-binned form (``deform_depth``, ``water_z_depth``:
  one row per permeant-depth bin of the spatial maps, one column per radius
  or height), never in the ``deform`` / ``water_z`` tables, which sum every
  frame whatever the permeant depth.

Why a joint block jackknife:

Path numbers run along the Monte-Carlo chain, and reactive and unreactive
paths are interleaved in it, so the two class means are not independent.
Both are built from the same contiguous MC-time blocks, and each
leave-one-block-out replicate drops a block from both at once. The spread of
the replicated *difference* is then the error of the difference, covariance
included - which two separate error bars cannot give. It is the estimator
``membrane-spatial`` already uses for its profiles, applied to the
difference.

The many bins and observables make some bins significant by chance: about
``alpha`` of the usable bins, which the ranking reports beside each count.
Neighbouring bins are correlated, so an isolated significant bin means less
than a run of them.

Two simulations (``-run-b``):

The same comparison between two runs - the L and the D enantiomer - instead
of two classes of one run. Each run is given by its run directory
(``-run-a``, ``-run-b``), and ``-dir`` and ``-weights`` name the per-path
folder and the weights file inside every run, so they are given once. Each
run keeps its own weights, path numbers and MC-time blocks. The runs share no Monte-Carlo chain, so their means are
independent and the error of the difference is sqrt(err_A² + err_B²), each
from its own run's block jackknife. The t threshold uses min(K_A, K_B) - 1
degrees of freedom, the conservative end of the Welch range.

The depth axis and the leaflet names are in the lab frame, so a run that
entered from the other face covers the opposite depths with its leaflets
named the other way round. ``-rotate-b`` puts B in A's frame by the 180°
rotation about x, (x, y, z) -> (x, -y, -z). A rotation and not the mirror
z -> -z, because a mirror would turn the chiral lipids into their
enantiomers; the rotation only relabels coordinates. Under it

* the permeant depth, and the water height in ``water_z_depth``, change sign,
* upper and lower swap, in every observable that names a leaflet,
* the phosphate z-displacements (``def_*``) change sign,
* the azimuth φ, measured from +y, becomes 180° - φ,
* the maps' depth bins and y axis reverse,

and everything else - P2, tilt θ against each leaflet's own inward normal,
thicknesses, |H|, diffusion, water counts - is unchanged. The Gaussian
curvature is stored for the upper leaflet only, so B's rotated K has no
counterpart in A and is left out. See ``rotated_observable``.
"""

import fnmatch
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Optional

import matplotlib
import numpy as np
import pandas as pd
import typer
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import membrane_spatial as ms
from . import panels
from .pathdata import _check_overwrite

#: The tables on the 1 Å slab axis of the permeant depth.
SLAB_TABLES = ("p2", "tilt", "thick", "diff", "diff_xy")

#: The radial / water-z profiles per coarse permeant-depth bin.
DEPTH_BINNED_TABLES = ms._DEPTH_BINNED_KEYS

#: Every table whose axis is the permeant depth, and so can be compared.
DEPTH_TABLES = SLAB_TABLES + DEPTH_BINNED_TABLES

#: Not distortion: how long the permeant lingered, which differs between the
#: classes trivially. Kept in the per-table CSVs, left out of the ranking.
_NOT_RANKED = ("n_windows",)

#: Spatial observables compared map by map, per leaflet ('' = both).
_MAP_LEAFLETS = ("", "_upper", "_lower")

Z_LABEL = "z-displacement (Å)"

_COLOR_A = "#1a5c8a"
_COLOR_B = "#8a1a0d"


# ═════════════════════════════════════════════════════════════════════════════
# Where a path's data comes from, and in which frame
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PathFile:
    """One per-path file, and whether it is read rotated into the other run's
    frame (-rotate-b)."""
    path: str
    rotate: bool = False


def _source(key, where):
    """The PathFile of path `key`.

    `where` is either the folder of a single run, with `key` its path number,
    or a {key: PathFile} map over two runs, whose keys are (label, number).
    """
    if isinstance(where, Mapping):
        return where[key]
    return PathFile(str(ms.path_output_file(key, where)))


def _path_number(key):
    return key[1] if isinstance(key, tuple) else key


#: Leaflet tokens the rotation swaps.
_LEAFLET_SWAP = {"upper": "lower", "lower": "upper"}


def rotated_observable(name):
    """Where observable `name` of run B lands in A's frame under -rotate-b.

    Returns (new name, offset, scale), the value becoming offset + scale *
    value, or None when the rotated quantity has no counterpart in A's files.
    The rotation is (x, y, z) -> (x, -y, -z); the module docstring says why.
    Names are '_'-separated tokens, and a leaflet is always its own token
    ('upper_sn2_p2_chain', 'H_upper', 'def_upper_r3.00', 'thickness_DOPC_lower').
    """
    parts = ms.split_bin_column(name)
    if parts and parts[0] == "water_z" and parts[1] == "z":
        # height above the midplane: the column of -z holds this column's data
        return ms.bin_column("water_z", "z", -parts[2]), 0.0, 1.0
    tokens = name.split("_")
    leaflet = any(t in _LEAFLET_SWAP for t in tokens)
    # K exists for the upper leaflet only; H_radial is the upper leaflet's
    # signed H. Rotated, both describe the lower leaflet, which A lacks.
    if ("K" in tokens and leaflet) or tokens[:2] == ["H", "radial"]:
        return None
    new = "_".join(_LEAFLET_SWAP.get(t, t) for t in tokens)
    if tokens[0] == "def":
        return new, 0.0, -1.0       # lab-frame z-displacement of the phosphates
    if "phi" in tokens:
        return new, 180.0, -1.0     # arccos of the in-plane vector against +y
    return new, 0.0, 1.0


def _rotate_table(df):
    """A per-path table of run B in A's frame: depth negated, observables
    renamed and transformed as rotated_observable says."""
    axis = df.columns[0]
    out = {axis: -df[axis].to_numpy(dtype=float) + 0.0}
    for col in df.columns[1:]:
        if not col.endswith("_wsum") or f"{col[:-5]}_wtot" not in df:
            continue
        rot = rotated_observable(col[:-5])
        if rot is None:
            continue
        new, off, scale = rot
        ws = df[col].to_numpy(dtype=float)
        wt = df[f"{col[:-5]}_wtot"].to_numpy(dtype=float)
        # the stored sums are Σ w·v and Σ w, so v -> off + scale·v is linear
        out[f"{new}_wsum"] = off * wt + scale * ws
        out[f"{new}_wtot"] = wt
    return pd.DataFrame(out)


def _is_mirror_symmetric(values, atol):
    v = np.asarray(values, dtype=float)
    return np.allclose(v, -v[::-1], atol=atol)


def _rotate_spatial(sp):
    """A path's maps in the other run's frame: depth bins and the y axis
    reversed, observables renamed and transformed as rotated_observable says.

    Needs a depth grid and a y grid symmetric about zero, so that a reversed
    bin is again a bin; membrane-spatial's defaults are.
    """
    if not (_is_mirror_symmetric(sp["depth_edges"], 1e-6)
            and _is_mirror_symmetric(sp["y_coords"], 0.01)):
        raise ValueError(
            "-rotate-b needs the maps' depth bins and lateral grid to be "
            "symmetric about zero; these are depth edges "
            f"{np.round(sp['depth_edges'], 2).tolist()} and y "
            f"{np.round(sp['y_coords'], 2).tolist()}.")
    nx, ny = len(sp["x_coords"]), len(sp["y_coords"])
    names, ws, wc = [], [], []
    for i, n in enumerate(sp["names"]):
        rot = rotated_observable(n)
        if rot is None:
            continue
        new, off, scale = rot
        count = np.asarray(sp["wcount"][i], dtype=float)
        names.append(new)
        ws.append(off * count + scale * np.asarray(sp["wsum"][i], dtype=float))
        wc.append(count)

    def flip(stack):
        a = np.asarray(stack)
        n_obs, n_depth, m = a.shape
        # cells are x-major (indexing 'ij'), so y is the last grid axis
        return (a.reshape(n_obs, n_depth, nx, ny)[:, ::-1, :, ::-1]
                .reshape(n_obs, n_depth, m))

    return {**sp, "names": names, "wsum": flip(ws), "wcount": flip(wc)}


def _read_table(key, where, kind):
    src = _source(key, where)
    df = ms.read_path_tables(src.path, (kind,)).get(kind)
    return _rotate_table(df) if df is not None and src.rotate else df


def _read_spatial(key, where):
    src = _source(key, where)
    sp = ms.read_path_spatial(src.path)
    return _rotate_spatial(sp) if sp is not None and src.rotate else sp


# ═════════════════════════════════════════════════════════════════════════════
# Which paths
# ═════════════════════════════════════════════════════════════════════════════

#: Where membrane-spatial's per-path files and the WHAM path weights sit
#: inside a run directory (membrane-spatial runs from <run>/analysis).
DEFAULT_DATA_DIR = os.path.join("analysis", ms.ORDER_OUT)
DEFAULT_WEIGHTS = os.path.join("wham", "path_weights.txt")


def resolve_run(run, data_dir, weights):
    """(data folder, weights file) inside run directory `run`.

    -dir and -weights name one folder and one file relative to every run, so
    two runs laid out alike need them once. An absolute path is refused: it
    would point both runs at the same data.
    """
    for flag, rel in (("-dir", data_dir), ("-weights", weights)):
        if os.path.isabs(rel):
            raise typer.BadParameter(
                f"{flag} {rel!r}: give it relative to the run directory "
                "(-run-a, -run-b), not as an absolute path.")
    if not os.path.isdir(run):
        raise typer.BadParameter(f"{run}: no such run directory.")
    folder, wfile = os.path.join(run, data_dir), os.path.join(run, weights)
    if not os.path.isfile(wfile):
        raise typer.BadParameter(
            f"{wfile}: no such weights file (-weights {weights!r} in {run}).")
    return folder, wfile


def path_files_in(data_dir):
    """Sorted numbers of the per-path files <n>.npz in `data_dir`.

    Only that name counts: a .tmp of a write in progress, and the old
    <n>_spatial.npz layout that membrane-spatial no longer reads, are not
    path files.
    """
    if not os.path.isdir(data_dir):
        raise typer.BadParameter(f"{data_dir}: no such folder of "
                                 "membrane-spatial path files.")
    pns = []
    for entry in os.scandir(data_dir):
        stem, ext = os.path.splitext(entry.name)
        if ext == ".npz" and stem.isdigit() and entry.is_file():
            pns.append(int(stem))
    return sorted(pns)


def classify_paths(weights, classes, out_dir, minus=None):
    """{path: class} for every path file in `out_dir` with a positive weight
    and a class in `classes`, and a summary of what was left out.

    Every file in the folder is considered: there is no path range. Same rule
    as the aggregation: a path without a positive weight (burn-in, or not in
    the weights file) is not in the ensemble. A path whose file has no class
    (an older membrane-spatial) is counted, not guessed. Weighted paths that
    have no file are counted too, since they are data the comparison lacks.

    `minus`: WHAM's [0-] path numbers (ms.load_minus_paths), which decide
    the ensemble; None uses the one stored in each file. The summary counts
    the paths whose class this changes ("relabelled").
    """
    files = path_files_in(out_dir)
    cls_of = {}
    no_weight = unknown = relabelled = 0
    for pn in files:
        if not weights.get(pn, 0.0) > 0.0:
            no_weight += 1
            continue
        meta = ms.read_path_meta(ms.path_output_file(pn, out_dir))
        cls = ms.path_class_of(meta, None if minus is None else pn in minus)
        if minus is not None and cls != ms.path_class_of(meta):
            relabelled += 1
        if cls is None:
            unknown += 1
        elif cls in classes:
            cls_of[pn] = cls
    have = set(files)
    no_file = sum(1 for pn, w in weights.items() if w > 0.0 and pn not in have)
    return cls_of, {"files": len(files), "no_weight": no_weight,
                    "unknown": unknown, "no_file": no_file,
                    "relabelled": relabelled}


def _report_selection(head, info):
    """Print what classify_paths took and left out."""
    print(f"{head}  ({info['files']} path files; {info['no_weight']} without a "
          "positive weight left out)")
    if info.get("relabelled"):
        print(f"  {info['relabelled']} weighted path(s) classed differently "
              "from their file: WHAM's [0-] list decides (the file's stored "
              "ensemble is wrong for swapped paths written before 2026-11-18)")
    if info["unknown"]:
        print(f"  {info['unknown']} weighted path(s) have no class in their "
              "file (not written by the current membrane-spatial); left out")
    if info["no_file"]:
        print(f"  {info['no_file']} weighted path(s) have no file in the "
              "folder (not yet run through membrane-spatial)")


def collect_run(label, data_dir, weights, classes, rotate=False, minus=None):
    """One run of a two-run comparison, every path of it labelled `label`.

    The weighted paths of `classes` among the files in `data_dir`, keyed
    (label, path number) because the two runs' path numbers overlap; `minus`
    as in classify_paths. Returns {key: label}, {key: weight},
    {key: PathFile} and a summary dict.
    """
    cls_of, info = classify_paths(weights, classes, data_dir, minus)
    keys = [(label, pn) for pn in sorted(cls_of)]
    info["per_class"] = {c: sum(1 for v in cls_of.values() if v == c)
                         for c in classes}
    return ({k: label for k in keys},
            {k: weights[k[1]] for k in keys},
            {k: PathFile(str(ms.path_output_file(k[1], data_dir)), rotate)
             for k in keys},
            info)


# ═════════════════════════════════════════════════════════════════════════════
# 1-D profiles
# ═════════════════════════════════════════════════════════════════════════════

def _scan(pns, kind, out_dir):
    """Axis, axis name and observables of one table over `pns`.

    Reads names and axes only, like ms._scan_tables. The column set is the
    union over paths, so one file from before a schema change cannot drop
    columns from the comparison. `out_dir` is a folder or a {key: PathFile}
    map (see _source); a rotated file is reported in the other run's frame.
    """
    centers, cols, axis_name = set(), set(), None
    for pn in pns:
        src = _source(pn, out_dir)
        names = ms.read_path_columns(src.path, kind)
        axis = ms.read_path_axis(src.path, kind)
        if not names or axis is None:
            continue
        axis_name = names[0]
        obs = [c[:-len("_wsum")] for c in names if c.endswith("_wsum")]
        if src.rotate:
            axis = -np.asarray(axis, dtype=float) + 0.0
            obs = [r[0] for r in map(rotated_observable, obs) if r is not None]
        cols.update(obs)
        centers.update(np.round(axis, 6).tolist())
    return np.array(sorted(centers)), axis_name, sorted(cols)


def _blocks_per_class(K, labels):
    """K as {class: blocks}: one shared count, or each run's own."""
    return dict(K) if isinstance(K, Mapping) else {c: K for c in labels}


def _block_sums(pns, block_of, K, cls_of, w_of, labels, kind, centers, obs,
                out_dir):
    """Per class: (K, n_bins, n_obs) block sums of wsum and wtot, and Σwtot².

    Each path's stored sums are multiplied by its weight here, as in the
    aggregation. Σ(w·wtot)² over paths is what the Kish n_eff needs. K is one
    count for both classes, or {class: K} when each has its own blocks.
    """
    n, C = len(centers), len(obs)
    Ks = _blocks_per_class(K, labels)
    S = {c: np.zeros((Ks[c], n, C)) for c in labels}
    T = {c: np.zeros((Ks[c], n, C)) for c in labels}
    Q = {c: np.zeros((n, C)) for c in labels}
    ws_cols = [f"{o}_wsum" for o in obs]
    wt_cols = [f"{o}_wtot" for o in obs]
    for pn, k in zip(pns, block_of):
        df = _read_table(pn, out_dir, kind)
        if df is None:
            continue
        rows = np.searchsorted(centers, np.round(df.iloc[:, 0].to_numpy(), 6))
        ws = np.nan_to_num(df.reindex(columns=ws_cols).to_numpy(dtype=float))
        wt = np.nan_to_num(df.reindex(columns=wt_cols).to_numpy(dtype=float))
        ws_p, wt_p = np.zeros((n, C)), np.zeros((n, C))
        np.add.at(ws_p, rows, ws)
        np.add.at(wt_p, rows, wt)
        c, w = cls_of[pn], w_of[pn]
        S[c][k] += w * ws_p
        T[c][k] += w * wt_p
        Q[c] += (w * wt_p) ** 2
    return S, T, Q


def _ratio(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _jackknife_err(reps):
    """Delete-one-block jackknife error over axis 0, ignoring undefined replicates.

    A replicate is undefined where leaving its block out empties the bin; it
    is dropped from the spread and from the (K-1)/K factor, as in
    ms._pool_streamed.
    """
    ok = np.isfinite(reps)
    n_ok = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        bar = np.where(n_ok > 0, np.nansum(reps, axis=0) / np.maximum(n_ok, 1),
                       np.nan)
        ss = np.nansum(np.where(ok, reps - bar[None], 0.0) ** 2, axis=0)
        return np.where(n_ok > 1, np.sqrt((n_ok - 1) / np.maximum(n_ok, 1) * ss),
                        np.nan)


def compare_sums(S, T, Q, a, b, min_neff, t_crit, independent=False):
    """Class means, their difference b - a, and the jackknife error on it.

    Works on any trailing shape: (K, n_bins, n_obs) for the profiles,
    (K, n_obs, n_depth, M) for the maps. Returns a dict of arrays of that
    shape without the K axis.

    Default: a and b are classes of one MC chain, blocked alike, and the
    error is the joint jackknife of the difference, covariance included.
    `independent`: a and b are separate runs, each with its own blocks (K may
    differ between them); the means are independent and the error is
    sqrt(err_a² + err_b²).
    """
    out = {}
    reps = {}
    for c in (a, b):
        A, B = S[c].sum(axis=0), T[c].sum(axis=0)
        mean = _ratio(A, B)
        neff = np.where(Q[c] > 0, B ** 2 / np.where(Q[c] > 0, Q[c], 1.0), 0.0)
        reps[c] = _ratio(A[None] - S[c], B[None] - T[c])
        thin = neff < min_neff
        out[f"mean_{c}"] = np.where(thin, np.nan, mean)
        out[f"err_{c}"] = np.where(thin, np.nan, _jackknife_err(reps[c]))
        out[f"neff_{c}"] = neff
        # blocks this class has data in: the jackknife spread comes from
        # these alone, so one block means no error at all
        out[f"blocks_{c}"] = (T[c] > 0).sum(axis=0)

    usable = np.isfinite(out[f"mean_{a}"]) & np.isfinite(out[f"mean_{b}"])
    delta = np.where(usable, out[f"mean_{b}"] - out[f"mean_{a}"], np.nan)
    if independent:
        err = np.hypot(out[f"err_{a}"], out[f"err_{b}"])
    else:
        err = _jackknife_err(reps[b] - reps[a])
    enough = (out[f"blocks_{a}"] > 1) & (out[f"blocks_{b}"] > 1)
    err = np.where(usable & enough, err, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(err > 0, delta / np.where(err > 0, err, 1.0), np.nan)
    out["delta"] = delta
    out["delta_err"] = err
    out["t"] = t
    with np.errstate(invalid="ignore"):
        out["significant"] = np.isfinite(t) & (np.abs(t) > t_crit)
    return out


def profile_table(axis_name, centers, obs, res, a, b):
    """The comparison of one table as a wide DataFrame, one row per depth bin."""
    data = {axis_name: centers}
    for j, o in enumerate(obs):
        for c in (a, b):
            data[f"{o}_{c}"] = res[f"mean_{c}"][:, j]
            data[f"{o}_{c}_err"] = res[f"err_{c}"][:, j]
            data[f"{o}_{c}_neff"] = res[f"neff_{c}"][:, j]
        data[f"{o}_delta"] = res["delta"][:, j]
        data[f"{o}_delta_err"] = res["delta_err"][:, j]
        data[f"{o}_t"] = res["t"][:, j]
        data[f"{o}_significant"] = res["significant"][:, j]
    return pd.DataFrame(data)


def ranking_rows(kind, centers, obs, res, alpha):
    """One row per observable: how much of the depth axis differs, and where."""
    rows = []
    for j, o in enumerate(obs):
        if o.startswith(_NOT_RANKED):
            continue
        t = res["t"][:, j]
        finite = np.isfinite(t)
        if not finite.any():
            continue
        sig = res["significant"][:, j]
        i = int(np.nanargmax(np.abs(t)))
        rows.append({
            "table": kind,
            "observable": o,
            "usable_bins": int(finite.sum()),
            "significant_bins": int(sig.sum()),
            "expected_by_chance": round(alpha * int(finite.sum()), 2),
            "max_abs_t": float(abs(t[i])),
            "depth_at_max": float(centers[i]),
            "delta_at_max": float(res["delta"][i, j]),
            "delta_err_at_max": float(res["delta_err"][i, j]),
            "significant_depth_min": float(centers[sig].min()) if sig.any() else np.nan,
            "significant_depth_max": float(centers[sig].max()) if sig.any() else np.nan,
        })
    return rows


def key_profiles(available):
    """(table, observable, label) for the profiles worth a figure each.

    `available` maps table -> observables present, so a run without near-N
    columns or without a diffusion table just has fewer figures.
    """
    want = []
    for chain in ms.CHAIN_BONDS:
        want.append(("p2", f"{chain}_p2_chain", f"$P_2$ {chain}"))
        want.append(("tilt", f"{chain}_theta", f"tilt θ {chain} (°)"))
    want += [
        ("thick", "thickness", "local thickness (Å)"),
        ("thick", "upper_thickness", "local thickness, upper (Å)"),
        ("thick", "lower_thickness", "local thickness, lower (Å)"),
        ("thick", "H_upper", "mean curvature H, upper (1/Å)"),
        ("thick", "H_lower", "mean curvature H, lower (1/Å)"),
        ("thick", "K_upper", "Gaussian curvature K, upper (1/Å²)"),
        ("thick", "n_defect", "water molecules in the core"),
    ]
    near = sorted({int(o[4:].split("_", 1)[0])
                   for o in available.get("thick", ())
                   if o.startswith("near") and o.endswith("_thickness")
                   and o[4:].split("_", 1)[0].isdigit()})
    for N in near:
        for chain in ms.CHAIN_BONDS:
            want.append(("p2", f"near{N}_{chain}_p2_chain",
                         f"$P_2$ {chain}, nearest {N}"))
        want.append(("thick", f"near{N}_thickness", f"thickness, nearest {N} (Å)"))
        want.append(("thick", f"near{N}_H", f"curvature H, nearest {N} (1/Å)"))
    want += [
        ("diff", "D_z", "$D_z$ permeant, lag 1"),
        ("diff_xy", "D_xy", "$D_{xy}$ permeant, lag 1"),
    ]
    return [(k, o, lab) for k, o, lab in want if o in available.get(k, ())]


def plot_profile(x, res, j, label, a, b, t_crit, out_path, overw):
    """Both classes with their error bands above, the difference below."""
    _check_overwrite(out_path, overw)
    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    for c, color in ((a, _COLOR_A), (b, _COLOR_B)):
        m, e = res[f"mean_{c}"][:, j], res[f"err_{c}"][:, j]
        top.plot(x, m, color=color, lw=1.6, label=c)
        top.fill_between(x, m - e, m + e, color=color, alpha=0.2, lw=0)
    top.set_ylabel(label)
    top.legend(fontsize=9, frameon=False)

    d, e = res["delta"][:, j], res["delta_err"][:, j]
    bot.axhline(0.0, color="0.4", lw=0.8)
    bot.plot(x, d, color="black", lw=1.2)
    bot.fill_between(x, d - t_crit * e, d + t_crit * e, color="0.6",
                     alpha=0.4, lw=0)
    sig = res["significant"][:, j]
    if sig.any():
        bot.plot(x[sig], d[sig], "o", ms=3, color=_COLOR_B,
                 label="interval excludes 0")
        bot.legend(fontsize=8, frameon=False)
    bot.set_ylabel(f"Δ ({b} − {a})")
    bot.set_xlabel(Z_LABEL)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_overview(x, rows, out_path, overw, t_max=5.0):
    """The t of every key profile against depth, in one figure.

    rows: (label, t array). Read across a row for where an observable
    differs, down a column for what differs at one depth.
    """
    _check_overwrite(out_path, overw)
    T = np.array([t for _, t in rows])
    edges = np.concatenate([[x[0] - (x[1] - x[0]) / 2 if len(x) > 1 else x[0] - 0.5],
                            (x[:-1] + x[1:]) / 2,
                            [x[-1] + (x[-1] - x[-2]) / 2 if len(x) > 1 else x[0] + 0.5]])
    cmap = plt.get_cmap("RdBu_r").with_extremes(bad="0.88")
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(rows) + 1.6))
    im = ax.pcolormesh(edges, np.arange(len(rows) + 1),
                       np.ma.masked_invalid(np.clip(T, -t_max, t_max)),
                       cmap=cmap, vmin=-t_max, vmax=t_max, shading="flat")
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([lab for lab, _ in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(Z_LABEL)
    fig.colorbar(im, ax=ax, label="t = Δ / σ(Δ)", extend="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Spatial maps
# ═════════════════════════════════════════════════════════════════════════════

#: What -map-obs default compares: the chain order and tilt of both chains,
#: the thickness and the curvature, all lipids together, per leaflet.
DEFAULT_MAP_OBS = tuple(
    f"{base}{lf}"
    for base in ([f"p2_{chain}_chain" for chain in ms.CHAIN_BONDS]
                 + [f"theta_{chain}" for chain in ms.CHAIN_BONDS]
                 + ["thickness", "curvature_H"])
    for lf in _MAP_LEAFLETS
)


def spatial_names(pns, out_dir):
    """The map observables the files hold and their (n_depth, M) shape.

    Read from the first path that has maps; ([], None) if none has. Over two
    runs, the first path of each: the names both runs have, in A's frame.
    """
    found = {}
    for pn in pns:
        run = pn[0] if isinstance(pn, tuple) else None
        if run in found:
            continue
        sp = _read_spatial(pn, out_dir)
        if sp is not None:
            found[run] = (sp["names"], sp["wsum"].shape[1:])
    if not found:
        return [], None
    first, shape = next(iter(found.values()))
    common = [n for n in first if all(n in names for names, _ in found.values())]
    return common, shape


def select_map_names(spec, available):
    """The map observables -map-obs picks, in the order the files hold them.

    'default' is DEFAULT_MAP_OBS and 'all' every observable there is;
    anything else is comma-separated shell-style patterns, e.g.
    'thickness*,p2_sn2_bond_*_DOPC' (names: <obs>[_<SPECIES>][_<leaflet>]).
    """
    spec = spec.strip()
    if spec == "default":
        wanted = set(DEFAULT_MAP_OBS)
        return [n for n in available if n in wanted]
    if spec == "all":
        return list(available)
    patterns = [p.strip() for p in spec.split(",") if p.strip()]
    return [n for n in available
            if any(fnmatch.fnmatchcase(n, p) for p in patterns)]


def map_chunks(names, shape, K, memory_gb):
    """`names` split so one chunk's comparison fits in about `memory_gb`.

    Per observable the block sums hold S and T for two classes over K blocks
    (4K maps of shape (n_depth, M)), and compare_sums adds the two classes'
    leave-one-out replicates, their difference and the jackknife's
    temporaries - about 9K + 6 such maps in all, in float64. Each further
    chunk is one more pass over the path files.
    """
    per_obs = (9 * K + 6) * int(np.prod(shape)) * 8
    size = max(1, int(memory_gb * 1e9 // per_obs))
    return [names[i:i + size] for i in range(0, len(names), size)]


def map_block_sums(pns, block_of, K, cls_of, w_of, labels, names, out_dir):
    """Per class: (K, n_obs, n_depth, M) block sums of the spatial maps.

    The grid and depth bins are the first readable path's. A path on another
    grid was made by a run with other settings; it is counted and left out
    rather than regridded, because mixing grids inside one class but not the
    other would be a difference of its own.
    """
    ref = S = T = Q = None
    skipped = 0
    Ks = _blocks_per_class(K, labels)
    for pn, k in zip(pns, block_of):
        sp = _read_spatial(pn, out_dir)
        if sp is None:
            continue
        if ref is None:
            ref = sp
            present = [n for n in names if n in sp["names"]]
            shape = (len(present),) + sp["wsum"].shape[1:]
            S = {c: np.zeros((Ks[c],) + shape) for c in labels}
            T = {c: np.zeros((Ks[c],) + shape) for c in labels}
            Q = {c: np.zeros(shape) for c in labels}
        same = (sp["wsum"].shape[1:] == ref["wsum"].shape[1:]
                and np.allclose(sp["x_coords"], ref["x_coords"], atol=0.01)
                and np.allclose(sp["y_coords"], ref["y_coords"], atol=0.01)
                and np.allclose(sp["depth_edges"], ref["depth_edges"], atol=1e-6)
                and all(n in sp["names"] for n in present))
        if not same:
            skipped += 1
            continue
        idx = [sp["names"].index(n) for n in present]
        c, w = cls_of[pn], w_of[pn]
        S[c][k] += w * sp["wsum"][idx]
        T[c][k] += w * sp["wcount"][idx]
        Q[c] += (w * sp["wcount"][idx]) ** 2
    if ref is None:
        return None
    return {"S": S, "T": T, "Q": Q, "names": present, "skipped": skipped,
            "x": ref["x_coords"], "y": ref["y_coords"],
            "depth_edges": ref["depth_edges"]}


def plot_map_comparison(maps, res, i, a, b, out_path, overw):
    """One row per depth bin that both classes reach: a, b and their difference.

    Cells whose difference interval excludes zero carry a dot.
    """
    nx, ny = len(maps["x"]), len(maps["y"])
    grid = lambda arr: arr.reshape(arr.shape[0], nx, ny)
    ma, mb = grid(res[f"mean_{a}"][i]), grid(res[f"mean_{b}"][i])
    d, sig = grid(res["delta"][i]), grid(res["significant"][i])
    depths = [k for k in range(d.shape[0]) if np.isfinite(d[k]).any()]
    if not depths:
        return False
    _check_overwrite(out_path, overw)

    both = np.concatenate([ma[depths][np.isfinite(ma[depths])],
                           mb[depths][np.isfinite(mb[depths])]])
    vmin, vmax = np.percentile(both, [2, 98]) if both.size else (None, None)
    finite_d = np.abs(d[depths][np.isfinite(d[depths])])
    dmax = np.percentile(finite_d, 98) if finite_d.size else None
    X, Y = np.meshgrid(maps["x"], maps["y"], indexing="ij")
    edges = maps["depth_edges"]

    fig, axes = plt.subplots(len(depths), 3, figsize=(11, 3.4 * len(depths)),
                             squeeze=False)
    for row, k in enumerate(depths):
        tag = f"z {edges[k]:.1f}…{edges[k + 1]:.1f} Å"
        for col, (arr, title, cmap, lo, hi) in enumerate((
            (ma[k], f"{a}, {tag}", "viridis", vmin, vmax),
            (mb[k], f"{b}, {tag}", "viridis", vmin, vmax),
            (d[k], f"{b} − {a}", "RdBu_r",
             -dmax if dmax else None, dmax),
        )):
            ax = axes[row, col]
            im = ms._map_panel(ax, arr, maps["x"], maps["y"], title, cmap,
                               vmin=lo, vmax=hi)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if col == 2 and sig[k].any():
                ax.scatter(X[sig[k]], Y[sig[k]], s=5, c="black", zorder=6)
    fig.suptitle(maps["names"][i])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def map_ranking_row(name, res, i, depth_edges, alpha):
    """Where one map observable differs most, or None without a usable cell."""
    t = res["t"][i]
    finite = np.isfinite(t)
    if not finite.any():
        return None
    sig = res["significant"][i]
    d, cell = np.unravel_index(int(np.nanargmax(np.abs(t))), t.shape)
    return {
        "observable": name,
        "usable_cells": int(finite.sum()),
        "significant_cells": int(sig.sum()),
        "expected_by_chance": round(alpha * int(finite.sum()), 2),
        "max_abs_t": float(abs(t[d, cell])),
        "depth_bin_at_max": f"{depth_edges[d]:.1f}…{depth_edges[d + 1]:.1f}",
        "delta_at_max": float(res["delta"][i][d, cell]),
    }


def plot_depth_binned_compare(depth, obs, res, name, label, xlabel, cmap,
                              a, b, out_path, overw):
    """A radial or water-z profile against the permeant depth, per class and
    as their difference, with the cells whose interval excludes zero dotted.
    """
    found = sorted((parts[2], j) for j, o in enumerate(obs)
                   for parts in [ms.split_bin_column(o)]
                   if parts and parts[0] == name)
    if not found:
        return False
    _check_overwrite(out_path, overw)
    x = np.array([c for c, _ in found])
    cols = [j for _, j in found]
    ma, mb = res[f"mean_{a}"][:, cols], res[f"mean_{b}"][:, cols]
    d, sig = res["delta"][:, cols], res["significant"][:, cols]

    both = np.concatenate([ma[np.isfinite(ma)], mb[np.isfinite(mb)]])
    if cmap == "RdBu_r" and both.size:
        hi = float(np.percentile(np.abs(both), 98))
        lo = -hi
    elif both.size:
        lo, hi = np.percentile(both, [2, 98])
    else:
        lo = hi = None
    finite_d = np.abs(d[np.isfinite(d)])
    dmax = float(np.percentile(finite_d, 98)) if finite_d.size else None
    X, Y = np.meshgrid(x, depth)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, arr, title, cm, vlo, vhi in (
        (axes[0], ma, a, cmap, lo, hi),
        (axes[1], mb, b, cmap, lo, hi),
        (axes[2], d, f"{b} − {a}", "RdBu_r",
         -dmax if dmax else None, dmax),
    ):
        im = ax.pcolormesh(x, depth, np.ma.masked_invalid(arr), cmap=cm,
                           vmin=vlo, vmax=vhi, shading="nearest")
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    if sig.any():
        axes[2].scatter(X[sig], Y[sig], s=8, c="black", zorder=5)
    axes[0].set_ylabel("Permeant " + Z_LABEL)
    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


# ═════════════════════════════════════════════════════════════════════════════
# One weighting, start to finish
# ═════════════════════════════════════════════════════════════════════════════

def assign_blocks(pns, cls_of, a, b, n_blocks, independent):
    """MC-time blocks: (block_of aligned with pns, K, degrees of freedom).

    One chain: blocks shared by both classes, K an int. Two runs: each run
    blocked along its own path numbers, K a {class: K} map, and the
    conservative min(K_a, K_b) - 1 degrees of freedom.
    """
    if not independent:
        block_of, K = ms._assign_blocks(pns, n_blocks)
        return block_of, K, K - 1
    block_of, K = np.zeros(len(pns), dtype=int), {}
    for c in (a, b):
        idx = [i for i, p in enumerate(pns) if cls_of[p] == c]
        blk, K[c] = ms._assign_blocks([_path_number(pns[i]) for i in idx],
                                      n_blocks)
        block_of[idx] = blk
    return block_of, K, min(K.values()) - 1


def _depth_reached(centers, res, c):
    """The depth range where class `c` has any data, as text."""
    rows = (res[f"neff_{c}"] > 0).any(axis=tuple(range(1, res[f"neff_{c}"].ndim)))
    if not rows.any():
        return "nowhere"
    return f"{centers[rows].min():.1f}…{centers[rows].max():.1f} Å"


def run_comparison(cls_of, w_of, a, b, n_blocks, min_neff, alpha, map_obs,
                   map_memory, data_dir, out_dir, overw, independent=False):
    """Every table, the ranking and the maps for one set of path weights.

    `map_obs` is the list of map observables to compare (empty: no maps).
    `independent`: a and b are two runs (see compare_sums), and `data_dir`
    the {key: PathFile} map over both.
    """
    os.makedirs(out_dir, exist_ok=True)
    pns = sorted(cls_of)
    block_of, K, dof = assign_blocks(pns, cls_of, a, b, n_blocks, independent)
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, max(dof, 1)))
    blocks = (f"{K[a]} MC-time blocks in {a}, {K[b]} in {b}" if independent
              else f"{K} MC-time blocks")
    print(f"  {blocks}; |t| > {t_crit:.2f} counts as a difference "
          f"(two-sided alpha = {alpha:g})")

    paths_csv = os.path.join(out_dir, "paths.csv")
    _check_overwrite(paths_csv, overw)
    listing = {"path": [_path_number(p) for p in pns],
               "class": [cls_of[p] for p in pns],
               "weight": [w_of[p] for p in pns],
               "block": block_of}
    if isinstance(data_dir, Mapping):
        listing["file"] = [data_dir[p].path for p in pns]
        listing["rotated"] = [data_dir[p].rotate for p in pns]
    pd.DataFrame(listing).to_csv(paths_csv, index=False)

    ranking, results, available = [], {}, {}
    for kind in DEPTH_TABLES:
        centers, axis_name, obs = _scan(pns, kind, data_dir)
        if not obs:
            print(f"  [{kind}] no paths hold this table; skipped")
            continue
        S, T, Q = _block_sums(pns, block_of, K, cls_of, w_of, (a, b), kind,
                              centers, obs, data_dir)
        res = compare_sums(S, T, Q, a, b, min_neff, t_crit, independent)
        if independent and kind == SLAB_TABLES[0]:
            # Two runs need not overlap at all; say where each one is before
            # an empty ranking leaves the reader guessing.
            print(f"  depth reached: {a} {_depth_reached(centers, res, a)}, "
                  f"{b} {_depth_reached(centers, res, b)}")
        results[kind] = (centers, obs, res)
        available[kind] = set(obs)
        out = os.path.join(out_dir, f"{kind}_compare.csv")
        _check_overwrite(out, overw)
        profile_table(axis_name, centers, obs, res, a, b).to_csv(
            out, index=False, float_format="%.6e")
        ranking += ranking_rows(kind, centers, obs, res, alpha)
        print(f"  -> {kind}_compare.csv")

    if not results:
        raise RuntimeError("No depth-resolved tables found to compare.")

    rank = pd.DataFrame(ranking)
    if not rank.empty:
        rank = rank.sort_values(["significant_bins", "max_abs_t"],
                                ascending=False, ignore_index=True)
    out = os.path.join(out_dir, "distortion_ranking.csv")
    _check_overwrite(out, overw)
    rank.to_csv(out, index=False, float_format="%.6g")
    print("  -> distortion_ranking.csv")

    keys = key_profiles(available)
    overview = []
    for kind, o, label in keys:
        centers, obs, res = results[kind]
        j = obs.index(o)
        plot_profile(centers, res, j, label, a, b, t_crit,
                     os.path.join(out_dir, f"compare_{o}.png"), overw)
        overview.append((label, res["t"][:, j]))
    # the key profiles are all on the slab grid, so one x axis serves all rows
    if overview:
        plot_overview(results[keys[0][0]][0], overview,
                      os.path.join(out_dir, "overview_t.png"), overw)
        print(f"  -> {len(overview)} profile figures and overview_t.png")

    drawn = 0
    for kind, name, label, xlabel, cmap in ms.DEPTH_BINNED_PLOTS:
        if kind in results:
            centers, obs, res = results[kind]
            drawn += plot_depth_binned_compare(
                centers, obs, res, name, label, xlabel, cmap, a, b,
                os.path.join(out_dir, f"compare_{name}_vs_depth.png"), overw)
    if drawn:
        print(f"  -> {drawn} depth-binned deformation/water figures")

    if map_obs:
        _compare_maps(pns, block_of, K, cls_of, w_of, a, b, min_neff, t_crit,
                      alpha, map_obs, map_memory, data_dir, out_dir, overw,
                      independent)
    return rank


def _compare_maps(pns, block_of, K, cls_of, w_of, a, b, min_neff, t_crit,
                  alpha, names, memory_gb, data_dir, out_dir, overw,
                  independent=False):
    """The selected map observables, a memory-sized chunk at a time."""
    _, shape = spatial_names(pns, data_dir)
    if shape is None:
        print("  [maps] no spatial maps in the path files; skipped")
        return
    K_max = max(K.values()) if isinstance(K, Mapping) else K
    chunks = map_chunks(names, shape, K_max, memory_gb)
    if len(chunks) > 1:
        print(f"  [maps] {len(names)} observables in {len(chunks)} passes "
              f"over the path files (-map-memory {memory_gb:g} GB)")

    arrays, rows, drawn, skipped = {}, [], 0, 0
    for chunk in chunks:
        sums = map_block_sums(pns, block_of, K, cls_of, w_of, (a, b), chunk,
                              data_dir)
        if sums is None or not sums["names"]:
            continue
        skipped = sums["skipped"]
        res = compare_sums(sums["S"], sums["T"], sums["Q"], a, b, min_neff,
                           t_crit, independent)
        nx, ny = len(sums["x"]), len(sums["y"])
        arrays.update({"x_coords": sums["x"], "y_coords": sums["y"],
                       "depth_edges": sums["depth_edges"]})
        for i, name in enumerate(sums["names"]):
            for key in (f"mean_{a}", f"mean_{b}", "delta", "delta_err", "t"):
                arr = res[key][i]
                arrays[f"{name}__{key}"] = arr.reshape(arr.shape[0], nx, ny)
            row = map_ranking_row(name, res, i, sums["depth_edges"], alpha)
            if row is not None:
                rows.append(row)
            drawn += plot_map_comparison(
                sums, res, i, a, b,
                os.path.join(out_dir, f"map_compare_{name}.png"), overw)
        del sums, res
    if not arrays:
        print("  [maps] none of the selected observables is in the files")
        return
    if skipped:
        print(f"  [maps] left out {skipped} path(s) on another grid "
              "or other depth bins than the first")

    out = os.path.join(out_dir, "spatial_compare.npz")
    _check_overwrite(out, overw)
    np.savez_compressed(out, **arrays)
    rank = pd.DataFrame(rows)
    if not rank.empty:
        rank = rank.sort_values(["significant_cells", "max_abs_t"],
                                ascending=False, ignore_index=True)
    out = os.path.join(out_dir, "map_ranking.csv")
    _check_overwrite(out, overw)
    rank.to_csv(out, index=False, float_format="%.6g")
    print(f"  -> spatial_compare.npz, map_ranking.csv, {drawn} map figures")


def _print_top(rank, two_runs=False, n=12):
    if rank.empty:
        print("  nothing has a usable bin in both classes")
        if two_runs:
            print("  (two runs that entered from opposite faces share no depth "
                  "bins until one is rotated: see -rotate-b)")
        return
    print(f"  {'observable':<34} {'table':<13} {'sig/usable':>11} "
          f"{'max|t|':>7} {'at z':>7} {'delta':>11}")
    for _, r in rank.head(n).iterrows():
        print(f"  {r['observable']:<34} {r['table']:<13} "
              f"{r['significant_bins']:>4}/{r['usable_bins']:<6} "
              f"{r['max_abs_t']:>7.3g} {r['depth_at_max']:>7.1f} "
              f"{r['delta_at_max']:>11.4g}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _one_run(weights, data_dir, a, b):
    """Two classes of one run, from every path file in `data_dir`:
    ({pn: class}, {pn: weight}, a, b)."""
    a = a or "unreactive"
    b = b or "reactive"
    for flag, cls in (("-a", a), ("-b", b)):
        if cls not in ms.PATH_CLASSES:
            raise typer.BadParameter(
                f"{flag}: {cls!r} is not one of {', '.join(ms.PATH_CLASSES)}.")
    if a == b:
        raise typer.BadParameter("-a and -b name the same class.")

    w_all = ms.load_path_weights(weights)
    minus = ms.load_minus_paths(weights)
    print(ms.describe_minus_source(weights, minus))
    cls_of, info = classify_paths(w_all, (a, b), data_dir, minus)
    counts = {c: sum(1 for v in cls_of.values() if v == c) for c in (a, b)}
    _report_selection(f"{data_dir}: "
                      + ", ".join(f"{c} {n}" for c, n in counts.items()), info)
    for c, n in counts.items():
        if n == 0:
            raise typer.BadParameter(
                f"No weighted {c} paths with a membrane-spatial file in "
                f"{data_dir}.")
    return cls_of, {p: w_all[p] for p in cls_of}, a, b


def _two_runs(runs, paths):
    """Two runs, every selected path of each labelled with its run.

    runs: (label, folder, weights file, rotate) for A and B; every path file
    in the folder is considered. Returns ({key: label}, {key: weight},
    {key: PathFile}).
    """
    subsets = ms.parse_path_subsets(paths)
    classes = ms.PATH_CLASSES if "all" in subsets else subsets
    cls_of, w_of, where = {}, {}, {}
    for label, folder, wfile, rotate in runs:
        w = ms.load_path_weights(wfile)
        if not any(v > 0.0 for v in w.values()):
            raise typer.BadParameter(f"{wfile}: no path has a positive weight.")
        minus = ms.load_minus_paths(wfile)
        print(f"{label}: " + ms.describe_minus_source(wfile, minus))
        c, ww, src, info = collect_run(label, folder, w, classes, rotate, minus)
        _report_selection(
            f"{label}: {folder}"
            + (" (rotated into A's frame)" if rotate else "") + ": "
            + ", ".join(f"{k} {n}" for k, n in info["per_class"].items()),
            info)
        if not c:
            raise typer.BadParameter(
                f"{label}: no weighted {'/'.join(classes)} paths with a "
                f"membrane-spatial file in {folder}.")
        cls_of.update(c)
        w_of.update(ww)
        where.update(src)
    return cls_of, w_of, where


def membrane_compare(
    # ── Input data ────────────────────────────────────────────────────────
    run_a: Annotated[str, typer.Option("-run-a", help="REQUIRED. Run directory of one stage, e.g. L_PRO_neutral/infinit_entry; -dir and -weights are looked up inside it.", rich_help_panel=panels.INPUT)] = ...,
    run_b: Annotated[Optional[str], typer.Option("-run-b", help="Run directory of the same stage of a second simulation, e.g. the other enantiomer, laid out like -run-a. Compares run B against run A instead of two classes of one run; the runs are independent, so the error of the difference is sqrt(err_A² + err_B²) from each run's own block jackknife. Compare like with like: the same stage (entry, internal, escape) of both.", rich_help_panel=panels.INPUT)] = None,
    data_dir: Annotated[str, typer.Option("-dir", help="Folder of the per-path files that membrane-spatial wrote, relative to the run directory (each run's, with -run-b). Every <n>.npz in it whose path has a positive weight is used; there is no path range.", rich_help_panel=panels.INPUT)] = DEFAULT_DATA_DIR,
    weights: Annotated[str, typer.Option("-weights", help="WHAM path weights, the file membrane-spatial -plot uses, relative to the run directory (each run's, with -run-b).", rich_help_panel=panels.INPUT)] = DEFAULT_WEIGHTS,
    a: Annotated[Optional[str], typer.Option("-a", help="Reference path class: reactive, unreactive or minus (default unreactive). One run only; with -run-b, -paths picks the classes.", rich_help_panel=panels.INPUT)] = None,
    b: Annotated[Optional[str], typer.Option("-b", help="Path class compared against -a (default reactive). The difference reported is B minus A.", rich_help_panel=panels.INPUT)] = None,
    label_a: Annotated[str, typer.Option("-label-a", help="Name of run A in tables and figures, e.g. L (with -run-b).", rich_help_panel=panels.INPUT)] = "A",
    label_b: Annotated[str, typer.Option("-label-b", help="Name of run B in tables and figures, e.g. D (with -run-b).", rich_help_panel=panels.INPUT)] = "B",

    # ── Dataset construction ──────────────────────────────────────────────
    paths: Annotated[str, typer.Option("-paths", help="With -run-b: the path classes pooled in each run, comma-separated from all, reactive, unreactive, minus. 'all' is every classified weighted path, as membrane-spatial -paths all.", rich_help_panel=panels.DATASET)] = "all",

    # ── Symmetry ──────────────────────────────────────────────────────────
    rotate_b: Annotated[bool, typer.Option("-rotate-b", help="Run B entered from the other face: read it rotated 180° about x, (x, y, z) -> (x, -y, -z). Its depth and water heights change sign, upper and lower leaflet swap, phosphate z-displacements change sign, phi becomes 180 - phi, and the maps' depth bins and y axis reverse. Without it, two runs that entered from opposite faces share no depth bins.", rich_help_panel=panels.SYMMETRY)] = False,

    # ── Model and training ────────────────────────────────────────────────
    n_blocks: Annotated[int, typer.Option("-n-blocks", help="Contiguous MC-time blocks for the jackknife: shared by both classes of one run, or per run with -run-b. Blocks must be longer than the path decorrelation time; a class present in only a few blocks gets a noisy error, and in fewer than two none at all.", rich_help_panel=panels.MODEL)] = 20,
    min_neff: Annotated[float, typer.Option("-min-neff", help="Leave out depth bins (and map cells) where either class rests on fewer than this effective number of paths (Kish n_eff).", rich_help_panel=panels.MODEL)] = ms.MIN_NEFF,
    alpha: Annotated[float, typer.Option("-alpha", help="Two-sided significance level per bin; the threshold is Student's t with n_blocks - 1 degrees of freedom (the smaller run's, with -run-b).", rich_help_panel=panels.MODEL)] = 0.05,
    unweighted: Annotated[bool, typer.Option("-unweighted", help="Also run the comparison with every path weighted equally, into <out-dir>/unweighted. Same paths; a difference that appears only with the WHAM weights rests on a few heavy paths.", rich_help_panel=panels.MODEL)] = False,
    map_memory: Annotated[float, typer.Option("-map-memory", help="Roughly how many GB the map comparison may hold at once. A selection that needs more is compared in chunks of observables, each one more pass over the path files - slower, not less exact.", rich_help_panel=panels.MODEL)] = 2.0,

    # ── Output ────────────────────────────────────────────────────────────
    out_dir: Annotated[Optional[str], typer.Option("-out-dir", help="Output folder (default: <membrane-spatial plot folder>/compare_<b>_vs_<a>, with the run labels for -run-b).", rich_help_panel=panels.OUTPUT)] = None,
    map_obs: Annotated[str, typer.Option("-map-obs", help="Which permeant-centred 2-D maps to compare per depth bin. 'default': chain order (P2) and tilt of both chains, thickness and curvature H, all lipids, each leaflet and both. 'all': every map the files hold, i.e. also per bond, per species and phi. Otherwise comma-separated shell patterns on the map names <obs>[_<SPECIES>][_<leaflet>], e.g. 'thickness*,p2_sn2_bond_*_DOPC'. 'none' skips the maps. map_ranking.csv ranks whatever is selected.", rich_help_panel=panels.OUTPUT)] = "default",
    overw: Annotated[bool, typer.Option("-O", help="Overwrite existing files.", rich_help_panel=panels.OUTPUT)] = False,
):
    """Depth-matched membrane distortion of two path classes, or of two
    simulations (-run-b), and where they differ.

    Reads the per-path files membrane-spatial wrote, so no trajectory is
    re-read: every <n>.npz in <run>/-dir whose path has a positive weight in
    <run>/-weights, with no path range to set. Every observable of the depth-resolved tables (order, tilt,
    thickness, curvature, core water, permeant diffusion, and the radial
    phosphate deformation and water profiles per depth bin) is compared bin
    by bin on the permeant-depth axis, with a block-jackknife error on the
    difference itself. distortion_ranking.csv lists what differs most;
    overview_t.png shows it against depth.
    """
    two_runs = run_b is not None
    if two_runs:
        if a is not None or b is not None:
            raise typer.BadParameter(
                "-a/-b pick two classes of one run; with -run-b the runs are "
                "compared, and -paths picks the classes pooled in each.")
        if not label_a or not label_b or label_a == label_b:
            raise typer.BadParameter("-label-a and -label-b must be two "
                                     "different, non-empty names.")
        if any(os.sep in lab for lab in (label_a, label_b)):
            raise typer.BadParameter("-label-a/-label-b name files; leave "
                                     f"out {os.sep!r}.")
        a, b = label_a, label_b
        dir_a, w_a = resolve_run(run_a, data_dir, weights)
        dir_b, w_b = resolve_run(run_b, data_dir, weights)
        cls_of, w_of, where = _two_runs(
            [(a, dir_a, w_a, False), (b, dir_b, w_b, rotate_b)], paths)
    else:
        given = [flag for flag, v in (("-rotate-b", rotate_b or None),
                                      ("-paths", None if paths == "all" else paths))
                 if v is not None]
        if given:
            raise typer.BadParameter(f"{', '.join(given)} only apply with "
                                     "-run-b, a second simulation.")
        dir_a, w_a = resolve_run(run_a, data_dir, weights)
        cls_of, w_of, a, b = _one_run(w_a, dir_a, a, b)
        where = dir_a
    out_dir = out_dir or os.path.join(ms.PLOT_OUT, f"compare_{b}_vs_{a}")

    # Resolved before any work, so a pattern that matches nothing fails here
    # rather than after the profiles have been written.
    names = []
    if map_obs.strip() != "none":
        available, _ = spatial_names(sorted(cls_of), where)
        names = select_map_names(map_obs, available)
        if available and not names:
            raise typer.BadParameter(
                f"-map-obs {map_obs!r} matches none of the {len(available)} "
                f"map observables, e.g. {', '.join(available[:4])}.")
        if names:
            print(f"maps: {len(names)} of {len(available)} observables")

    print("\n══ WHAM-weighted ══")
    rank = run_comparison(cls_of, w_of, a, b, n_blocks, min_neff, alpha, names,
                          map_memory, where, out_dir, overw, two_runs)
    print("\nlargest differences:")
    _print_top(rank, two_runs)

    if unweighted:
        sub = os.path.join(out_dir, "unweighted")
        print(f"\n══ unweighted (every path equal; written to {sub}) ══")
        rank_u = run_comparison(cls_of, {p: 1.0 for p in cls_of}, a, b,
                                n_blocks, min_neff, alpha, names, map_memory,
                                where, sub, overw, two_runs)
        print("\nlargest differences, unweighted:")
        _print_top(rank_u, two_runs)
    print("\nDone!")
