"""
Analyze simulation text files (parallel version, all columns):
- Classify by reactive/non-reactive (line 1) and plus/minus ensemble (line 2)
- For each of the 3 groups: produce a summary stats CSV and per-column histogram PNGs

Memory-efficient intermediate binning:
- Bin centers are fixed BEFORE any data is accumulated via COLUMN_RANGES config.
- Every FLUSH_EVERY files, raw values are histogrammed using update_histogram()
  and saved as .npz files in INTERMEDIATE_DIR.
- After parsing, intermediates are summed into final bin counts and plotted.
- If SKIP_PARSING=True, existing intermediates are reused.

Histogramming:
- Uses update_histogram() with np.add.at for correctness.
- factor = weight (scalar per path), applied to all rows of that path.
- Supports both 1D and 2D histograms.

Angular correction (spherical Jacobian):
- Columns listed in SPHERICAL_ANGLE_COLS are divided by sin(theta) after
  accumulation so that the plotted distribution reflects true solid-angle
  density rather than raw angle counts. Division is done only where
  sin(theta) > SINE_CLIP_THRESHOLD to avoid divide-by-zero at 0 / 180 deg.
- PRO_r_plane uses cos(phi) as its Jacobian factor (elevation angle).
- Both the raw and corrected histogram are saved side-by-side.

2D histograms vs OP_Lamb:
- For every column (except OP_Lamb itself) a joint 2D histogram is produced
  with OP_Lamb on the X-axis. Outputs go in histograms2d_<tag>/ folders.
- Angular columns also get a sine-corrected 2D version.

Group-specific column ranges:
- COLUMN_RANGES is the default used by ("non-reactive", "plus") and
  ("reactive", "plus").
- COLUMN_RANGES_NR_MINUS overrides ranges for ("non-reactive", "minus"),
  whose paths end where the plus ensemble begins and therefore span a
  narrower part of the OP_Lamb axis. Tighter ranges give clearer histograms.
- get_bin_info(group_key) returns the correct bin_info dict for each group.

Statistical analysis of DOPC/POPC preference:
- For each (col_dopc, col_popc) pair and each OP_Lamb bin, a bootstrap
  resampling over paths (not frames) estimates the uncertainty in the
  enrichment ratio E = frac_obs / frac_stoich.
- The null hypothesis is that E = 1 (no preference beyond bulk stoichiometry).
- A two-sided p-value is reported as the fraction of bootstrap enrichments
  at least as extreme as observed.
- Results are saved to preference_stats_<tag>.csv and overlaid on the
  preference plots as shaded confidence bands.

Output layout:
    <OUTPUT_DIR>/
      intermediates/
      histograms_<tag>/          1-D histograms (raw + sine-corrected for angular)
      histograms2d_<tag>/        2-D histograms vs OP_Lamb
      preference_dopc_popc_<tag>/
        <label>_preference.png          (fractional contact + enrichment + CI band)
        <label>_cn_per_lipid.png        (per-molecule CN, normalised)
        preference_stats_<tag>.csv      (enrichment, CI, p-value per OP_Lamb bin)
      stats_*.csv
      file_classification.txt
"""

import csv
import glob
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import numpy as np
import tomli
import typer

from . import panels

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ── 1. CONFIGURATION ─────────────────────────────────────────────────────────
INPUT_DIR         = "ML/"
FILE_PATTERN      = ".txt"
OUTPUT_DIR        = "analysis_output_merged"
PATH_WEIGHTS_FILE = "path_weights.txt"
N_WORKERS         = 8
HIST_DPI          = 300
FLUSH_EVERY       = 2000

SKIP_PARSING = False

TESTING    = False
test_value = 1

INTERMEDIATE_DIR = os.path.join(OUTPUT_DIR, "intermediates")

# ── Default column ranges (used by non-reactive plus and reactive plus) ───────
# ── Histogram binning, loaded from the -ranges file ──────────────────────────
# These used to be two large literal dicts here, which meant every simulation
# needed its own edited copy of this file - the reason four divergent copies of
# the original script existed. They are now read from a TOML file given with
# -ranges (see examples/column_ranges.toml) and populated by _load_column_ranges
# before the run starts. The helper functions below still read them as globals,
# exactly as before.
COLUMN_RANGES = {}
COLUMN_RANGES_NR_MINUS = {}


def _load_column_ranges(path):
    """Read the [ranges] / [ranges_nr_minus] tables from a TOML file.

    Returns two dicts of ``{column: (min, max, n_bins)}``. Raises rather than
    falling back to a default: the binning determines every histogram in the
    output, so a silently wrong table would be worse than no run at all.
    """
    fpath = Path(path)
    if not fpath.is_file():
        raise typer.BadParameter(
            f"-ranges: no such file {path!r}. This file defines the histogram "
            "binning for every CV; see examples/column_ranges.toml for the "
            "format and copy it per simulation."
        )
    with open(fpath, "rb") as fh:
        cfg = tomli.load(fh)

    if "ranges" not in cfg:
        raise typer.BadParameter(
            f"-ranges: {path!r} has no [ranges] section."
        )

    def _table(section):
        out = {}
        for col, val in cfg.get(section, {}).items():
            if len(val) != 3:
                raise typer.BadParameter(
                    f"-ranges: [{section}] {col!r} must be [min, max, n_bins], "
                    f"got {val!r}"
                )
            lo, hi, nbins = val
            if not hi > lo:
                raise typer.BadParameter(
                    f"-ranges: [{section}] {col!r} has max <= min ({lo}, {hi})."
                )
            if int(nbins) < 1:
                raise typer.BadParameter(
                    f"-ranges: [{section}] {col!r} needs at least 1 bin, got {nbins}."
                )
            out[col] = (float(lo), float(hi), int(nbins))
        return out

    return _table("ranges"), _table("ranges_nr_minus")


def get_column_ranges(group_key):
    if group_key != ("non-reactive", "minus"):
        return COLUMN_RANGES
    merged = dict(COLUMN_RANGES)
    merged.update(COLUMN_RANGES_NR_MINUS)
    return merged


# ── Angular / spherical-surface correction ────────────────────────────────────
SPHERICAL_ANGLE_COLS = {"PRO_ang_C_CG", "PRO_r_plane", "PRO_r_plane_arccos", "PRO_r_plane_chiral", "PRO_cen_vec"}#, "PRO_dih_OH", "PRO_ang_OH", "PRO_dih_chiral"}
SINE_CLIP_THRESHOLD  = 1e-6
N_COSINE_BINS = 20

REF_COL = "OP_Lamb"

# ── 1b. CROSS-SIMULATION OP_Lamb MERGING ─────────────────────────────────────
# Path sampling in this folder only produces paths that traverse the OP_Lamb
# window in ONE direction (A -> B).  The complementary direction is sampled by the
# other or inverse infRETIS run.  Its plus-ensemble phase points
# can be added into the OP_Lamb histogram so that -log(p) is taken on the
# combined statistics instead of on a one-directional subset.
#
# Only OP_Lamb is merged.  The two runs do not share a common feature set (their
# ML/*.txt headers differ), and every other column would additionally need a
# leaflet/handedness transform under mirroring, so nothing else is combined.
#
# MERGE_OTHER_SIM   : master switch for folding in the other run.
# OTHER_SIM_ML_DIR  : that run's ML/ folder (holds <path_num>.txt).
# OTHER_SIM_WEIGHTS : that run's WHAM path_weights.txt.
#                     Both are resolved against the CWD and a few parent
#                     prefixes, so the script works from either analysis folder.
# SYMMETRIC_OP      : the OP axis is mirror-symmetric about 0.  The other run's
#                     phase points are folded in reversed, OP -> -OP.  Use this
#                     when the other run samples the opposite branch (entry at
#                     negative OP_Lamb vs escape at positive OP_Lamb).  Set to
#                     False if both runs already share one OP convention.
# SELF_MIRROR       : additionally fold THIS run's own phase points in reversed
#                     (OP -> -OP).  Only useful when the configured OP_Lamb
#                     range spans both branches; with a one-sided range the
#                     mirrored copy falls outside every bin and is discarded.
#                     Only useful for the "internal" simulations.
# MERGE_ENSEMBLES   : which ensembles contribute.  Plus only, as required.
# MERGE_NORM        : how the two histograms are put on a common scale before
#                     being added.  This matters: each run's weighted histogram
#                     piles up near its OWN lambda_A, so the two are NOT on a
#                     common scale to begin with.
#                       "correction" - scale the escape run by the externally
#                                 supplied correction factor (default; this is
#                                 the physically referenced choice, see below)
#                       "match" - one multiplicative factor, fitted in log space
#                                 over the overlap window
#                       "area"  - normalise each to unit area first
#                       "raw"   - add as-is (each plus weight set sums to 1)
# MERGE_MATCH_RANGE : (lo, hi) OP window used to fit the "match" factor, in this
#                     run's OP convention.  None = every overlapping bin.
# MERGE_MIN_FRAC    : bins below this fraction of a histogram's peak are ignored
#                     when fitting the "match" factor (they are the noisy tails).
MERGE_OTHER_SIM   = True
OTHER_SIM_ML_DIR  = "../../analysis_esc/analysis/ML"
OTHER_SIM_WEIGHTS = "../../analysis_esc/analysis/path_weights.txt"
SYMMETRIC_OP      = True
SELF_MIRROR       = False
MERGE_ENSEMBLES   = ("plus",)
MERGE_NORM        = "correction"
MERGE_MATCH_RANGE = None
MERGE_MIN_FRAC    = 1e-3

MERGE_TAG = "merged_op"

# ── Escape-run correction factor ─────────────────────────────────────────────
# The escape run's weighted histogram is referenced to its own state A, so it
# has to be rescaled before it can be added to the other run.  The factor is
#
#     correction_factor = (N_A * (P_A / P_C_min) * (P_C / P_D_min)) / N_D
#
# with N_A = N_D = 1 and every P a TOTAL crossing probability.  These are
# entered by hand for now.  Each one is the last row of the corresponding run's
#   wham/Pcross.txt , column 2 (P-wham)
# e.g. escape -> 2.868490156674135e-04, entry -> 2.624258248023158e-03.
#
# CORRECTION_APPLY_TO : which histogram the factor multiplies.
#                         "local" - this run (use when running from the escape
#                                   folder; the default)
#                         "other" - the other run (use when running from the
#                                   entry folder, so the escape data being
#                                   folded in is the one corrected)
# Leave any P at None to signal "not filled in yet": the merge then reports
# exactly what is missing and falls back to the fitted "match" factor, which is
# NOT physically referenced.
CORRECTION_APPLY_TO = "other"
N_A     = 1.0
N_D     = 1.0
P_A = 0.002624258248023157
P_C = 2.7752649339303137e-06
P_D = 0.00028684901566741386
P_C_MIN = P_D
P_D_MIN = P_C

HIST_COLOR = {
    ("non-reactive", "minus"): "#2a7fbf",
    ("non-reactive", "plus"):  "#1a5c8a",
    ("reactive",     "plus"):  "#8a1a0d",
}

ALL_GROUPS = [
    ("non-reactive", "minus"),
    ("non-reactive", "plus"),
    ("reactive",     "plus"),
]

# ── Stoichiometric fractions from the 5:1 DOPC:POPC ratio ────────────────────
F_DOPC = 5.0 / 6.0
F_POPC = 1.0 / 6.0

# PREFERENCE_MIN_COUNTS = 10.0 # not used

# ── Bootstrap configuration ───────────────────────────────────────────────────
# N_BOOTSTRAP : number of path-level resamples for CI and p-value estimation.
# ALPHA       : significance level (two-sided). CI = [ALPHA/2, 1-ALPHA/2].
# MIN_PEAK_FRACTION : OP_Lamb bins whose total weight is below this fraction of
#               the per-pair peak are masked as NaN (too noisy to test).
N_BOOTSTRAP        = 2000
ALPHA              = 0.05
MIN_PEAK_FRACTION  = 0.01   # 1 % of peak — consistent with original masking


# ── 2. PATH WEIGHTS LOADER ───────────────────────────────────────────────────

def load_path_weights(filepath):
    raw        = {}
    skip_below = None

    with open(filepath, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# skip="):
                try:
                    skip_below = int(line.split("=", 1)[1])
                except ValueError:
                    raise ValueError(f"Could not parse skip value from: {line!r}")
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    raw[int(parts[0])] = float(parts[1])
                except ValueError:
                    print(f"  WARNING: skipping malformed line: {line!r}")

    if skip_below is None:
        print("  WARNING: no '# skip=<value>' header found")

    ordered_paths = sorted([(pn, w) for pn, w in raw.items()])

    print(f"  Loaded {len(raw)} entries from {filepath!r}")
    print(f"  Skip below: {skip_below if skip_below is not None else 'NONE'}")
    if ordered_paths:
        print(f"  Path range: {ordered_paths[0][0]} - {ordered_paths[-1][0]}")
    else:
        print("  No paths after filtering.")

    return ordered_paths


# ── 3. FILE PARSER ────────────────────────────────────────────────────────────

def classify_file(filepath):
    try:
        with open(filepath, "r") as fh:
            lines = fh.readlines()

        if len(lines) < 4:
            return (filepath, None, None, None, None)

        line1 = lines[0].strip().lstrip("#").strip()
        if "non-reactive" in line1:
            reactivity = "non-reactive"
        elif "reactive" in line1:
            reactivity = "reactive"
        else:
            return (filepath, None, None, None, None)

        line2 = lines[1].strip().lstrip("#").strip()
        if "plus" in line2:
            ensemble = "plus"
        elif "minus" in line2:
            ensemble = "minus"
        else:
            return (filepath, None, None, None, None)

        headers = lines[3].strip().lstrip("#").strip().split()
        n_cols  = len(headers)

        rows = []
        for line in lines[4:-1]:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            row = []
            for i in range(n_cols):
                try:
                    row.append(float(parts[i]) if i < len(parts) else np.nan)
                except ValueError:
                    row.append(np.nan)
            rows.append(row)

        data = np.array(rows, dtype=np.float64) if rows else np.empty((0, n_cols))
        return (filepath, reactivity, ensemble, headers, data)

    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        return (filepath, None, None, None, None)


# ── 4. BIN CENTER BUILDER ────────────────────────────────────────────────────

def build_bin_centers(column_ranges):
    bin_info = {}
    for col, (vmin, vmax, n_bins) in column_ranges.items():
        dx      = (vmax - vmin) / n_bins
        centers = vmin + np.arange(n_bins) * dx + 0.5 * dx
        bin_info[col] = (centers, vmin, dx)
    return bin_info


def build_all_bin_infos():
    print("\nBuilding global bin centers from COLUMN_RANGES config:")
    for col, (vmin, vmax, n_bins) in COLUMN_RANGES.items():
        dx      = (vmax - vmin) / n_bins
        centers_0 = vmin + 0.5 * dx
        print(f"  {col:25s}: {n_bins:3d} bins  [{vmin:.4g}, {vmax:.4g}]  "
              f"dx={dx:.4g}  centers[0]={centers_0:.4g}")

    if COLUMN_RANGES_NR_MINUS:
        print("\n  Overrides for (non-reactive, minus):")
        for col, (vmin, vmax, n_bins) in COLUMN_RANGES_NR_MINUS.items():
            dx      = (vmax - vmin) / n_bins
            centers_0 = vmin + 0.5 * dx
            print(f"    {col:25s}: {n_bins:3d} bins  [{vmin:.4g}, {vmax:.4g}]  "
                  f"dx={dx:.4g}  centers[0]={centers_0:.4g}")

    all_bin_infos = {}
    for key in ALL_GROUPS:
        all_bin_infos[key] = build_bin_centers(get_column_ranges(key))
    return all_bin_infos


# ── 5. HISTOGRAMMING ─────────────────────────────────────────────────────────

def update_histogram(data, factor, histogram, dx, Minx, Miny=None, dy=None):
    if Miny is not None and dy is not None:
        x  = data[:, 0]
        y  = data[:, 1]
        ix = ((x - Minx) / dx).astype(int)
        iy = ((y - Miny) / dy).astype(int)
        n_x, n_y = histogram.shape
        mask = (ix >= 0) & (ix < n_x) & (iy >= 0) & (iy < n_y)
        linear = ix[mask] * n_y + iy[mask]
        histogram += np.bincount(linear, weights=np.full(linear.size, factor),
                                 minlength=n_x * n_y).reshape(n_x, n_y)
    else:
        x  = data if data.ndim == 1 else data[:, 0]
        ix = ((x - Minx) / dx).astype(int)
        n_x = histogram.shape[0]
        mask = (ix >= 0) & (ix < n_x)
        histogram += np.bincount(ix[mask], weights=np.full(mask.sum(), factor),
                                 minlength=n_x)
    return histogram


# ── 6. SPHERICAL-SURFACE CORRECTION ─────────────────────────────────────────

def sine_correction_weights(col, centers_deg):
    rad = np.deg2rad(centers_deg)
    divisors = np.sin(rad)
    divisors = np.where(np.abs(divisors) < SINE_CLIP_THRESHOLD, np.nan, divisors)
    return divisors


def apply_sine_correction(col, centers, counts):
    divisors  = sine_correction_weights(col, centers)
    corrected = counts.astype(np.float64).copy()
    corrected /= divisors
    return corrected


# ── COSINE-TRANSFORM HISTOGRAM ───────────────────────────────────────────────

def build_cosine_bin_info(col, n_bins=180):
    vmin, vmax = (-1.0, 1.0)
    dx = (vmax - vmin) / n_bins
    centers = vmin + np.arange(n_bins) * dx + 0.5 * dx
    return centers, vmin, dx


def make_cosine_transform_histogram(centers, counts, col_name, group_label,
                                    out_path, color):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    finite = counts[np.isfinite(counts)]
    if finite.size == 0 or finite.sum() == 0:
        print(f"    SKIPPED (empty cosine-transform): {os.path.basename(out_path)}")
        return

    rad = np.deg2rad(centers)
    # if col_name == "PRO_r_plane":
    #     u_vals   = np.sin(rad)
    #     u_label  = f"sin({col_name}) [elevation]"
    # else:
    #     u_vals   = np.cos(rad)
    #     u_label  = f"cos({col_name})"
    u_vals   = np.cos(rad)
    u_label  = f"cos({col_name})"

    n_bins_out = N_COSINE_BINS
    u_centers_out, u_min, du = build_cosine_bin_info(col_name, n_bins=n_bins_out)
    hist_u = np.zeros(n_bins_out, dtype=np.float64)

    ix_arr = ((u_vals - u_min) / du).astype(int)
    valid  = np.isfinite(u_vals) & np.isfinite(counts) & (ix_arr >= 0) & (ix_arr < n_bins_out)
    hist_u = np.bincount(ix_arr[valid], weights=counts[valid], minlength=n_bins_out).astype(np.float64)

    finite_u = hist_u[np.isfinite(hist_u)]
    if finite_u.size == 0 or finite_u.sum() == 0:
        print(f"    SKIPPED (empty after rebinning): {os.path.basename(out_path)}")
        return

    total_u = np.nansum(hist_u)
    if total_u > 0:
        density_u = hist_u / (total_u * du)
    else:
        density_u = hist_u

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(u_centers_out, density_u, width=du,
           color=color, edgecolor="white", linewidth=0.4, alpha=0.9)

    ax.set_xlabel(u_label, fontsize=15)
    ax.set_ylabel("Probability density", fontsize=15)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:,.4f}")
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=HIST_DPI)
    plt.close(fig)


# ── 7. STATS FROM BINS ───────────────────────────────────────────────────────

STATS_FIELDS = ["n", "mean", "std", "min", "p25", "median", "p75", "max"]

def column_stats_from_bins(centers, counts):
    finite_mask = np.isfinite(counts)
    total = counts[finite_mask].sum()
    if total == 0:
        return {k: np.nan for k in STATS_FIELDS}

    c = centers[finite_mask]
    w = counts[finite_mask]

    half_widths = np.diff(centers) / 2
    edges = np.concatenate((
        [centers[0] - half_widths[0]],
        centers[:-1] + half_widths,
        [centers[-1] + half_widths[-1]]
    ))

    mean   = float(np.sum(c * w) / total)
    var    = float(np.sum(w * (c - mean) ** 2) / total)
    std    = float(np.sqrt(var))
    vmin   = float(edges[np.argmax(counts > 0)])
    vmax   = float(edges[np.argmax(np.cumsum(counts > 0) == (counts > 0).sum()) + 1])
    cdf    = np.cumsum(np.where(finite_mask, counts, 0)) / total
    p25    = float(centers[np.searchsorted(cdf, 0.25)])
    median = float(centers[np.searchsorted(cdf, 0.50)])
    p75    = float(centers[np.searchsorted(cdf, 0.75)])

    return {
        "n": int(total), "mean": mean, "std": std,
        "min": vmin, "p25": p25, "median": median, "p75": p75, "max": vmax,
    }


def write_stats_csv(col_stats_dict, filepath):
    with open(filepath, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["column"] + STATS_FIELDS)
        for col, stats in col_stats_dict.items():
            writer.writerow([col] + [stats[k] for k in STATS_FIELDS])
    print(f"  Saved stats : {os.path.relpath(filepath)}")


# ── 8. INTERMEDIATE FILE HELPERS ─────────────────────────────────────────────

def intermediate_path(group_key, col, chunk_idx, kind="1d"):
    react, ens = group_key
    tag = f"{react.replace('-', '')}_{ens}"
    safe_col = col.replace("/", "_").replace("\\", "_")
    return os.path.join(
        INTERMEDIATE_DIR,
        f"{chunk_idx:04d}__{tag}__{kind}__{safe_col}.npz"
    )


def flush_to_intermediate(buffer_data, buffer_weights, all_bin_infos, chunk_idx):
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    out_of_range_totals = defaultdict(int)

    for group_key in ALL_GROUPS:
        bin_info = all_bin_infos[group_key]
        col_dict = buffer_data[group_key]

        ref_centers, ref_Minx, ref_dx = bin_info[REF_COL]
        ref_n  = len(ref_centers)
        ref_list = col_dict.get(REF_COL, [])

        col_ranges_for_group = get_column_ranges(group_key)

        for col, (centers, Minx, dx) in bin_info.items():
            n_bins = len(centers)

            hist_1d   = np.zeros(n_bins, dtype=np.float64)
            vals_list = col_dict.get(col, [])

            for vals_arr, weight in vals_list:
                clean = vals_arr[~np.isnan(vals_arr)]
                if len(clean) == 0:
                    continue
                vmax_edge = Minx + n_bins * dx
                n_out = int(np.sum((clean < Minx) | (clean >= vmax_edge)))
                if n_out > 0:
                    out_of_range_totals[col] += n_out
                update_histogram(clean, weight, hist_1d, dx, Minx)

            np.savez_compressed(
                intermediate_path(group_key, col, chunk_idx, "1d"),
                counts=hist_1d, centers=centers
            )

            if col == REF_COL:
                continue

            hist_2d  = np.zeros((n_bins, ref_n), dtype=np.float64)
            col_list = col_dict.get(col, [])
            n_pairs  = min(len(ref_list), len(col_list))

            for i in range(n_pairs):
                ref_arr, _      = ref_list[i]
                col_arr, weight = col_list[i]
                valid = ~np.isnan(ref_arr) & ~np.isnan(col_arr)
                if not np.any(valid):
                    continue
                paired = np.column_stack((col_arr[valid], ref_arr[valid]))
                update_histogram(
                    paired, weight, hist_2d,
                    dx, Minx,
                    Miny=ref_Minx, dy=ref_dx
                )

            np.savez_compressed(
                intermediate_path(group_key, col, chunk_idx, "2d"),
                counts=hist_2d,
                centers_x=ref_centers,
                centers_y=centers
            )

    if out_of_range_totals:
        print(f"\n  [chunk {chunk_idx}] Out-of-range value counts "
              f"(check COLUMN_RANGES / COLUMN_RANGES_NR_MINUS):")
        for col, n in sorted(out_of_range_totals.items(), key=lambda x: -x[1]):
            vmin_cfg, vmax_cfg, _ = COLUMN_RANGES[col]
            print(f"    {col:25s}: {n:>8,} values outside "
                  f"[{vmin_cfg:.4g}, {vmax_cfg:.4g}]")

    for group_key in ALL_GROUPS:
        buffer_data[group_key].clear()
        buffer_weights[group_key].clear()


def load_and_sum_intermediates(group_key, col, kind="1d"):
    react, ens = group_key
    tag      = f"{react.replace('-', '')}_{ens}"
    safe_col = col.replace("/", "_").replace("\\", "_")

    pattern = os.path.join(
    INTERMEDIATE_DIR, f"[0-9][0-9][0-9][0-9]__{tag}__{kind}__{safe_col}.npz"
    )
    npz_files = sorted(glob.glob(pattern))

    if not npz_files:
        return (None, None) if kind == "1d" else (None, None, None)

    total_counts = None
    centers = centers_x = centers_y = None

    for npz_path in npz_files:
        d = np.load(npz_path)
        if kind == "1d":
            c, e = d["counts"], d["centers"]
            if total_counts is None:
                total_counts, centers = c.copy(), e
            else:
                if not np.allclose(centers, e):
                    print(f"  WARNING: mismatched centers in "
                          f"{os.path.basename(npz_path)}")
                    continue
                total_counts += c
        else:
            c, ex, ey = d["counts"], d["centers_x"], d["centers_y"]
            if total_counts is None:
                total_counts, centers_x, centers_y = c.copy(), ex, ey
            else:
                if not (np.allclose(centers_x, ex) and
                        np.allclose(centers_y, ey)):
                    print(f"  WARNING: mismatched centers in "
                          f"{os.path.basename(npz_path)}")
                    continue
                total_counts += c

    return (centers, total_counts) if kind == "1d" \
        else (centers_x, centers_y, total_counts)


def load_per_chunk_2d(group_key, col):
    """
    Load each chunk's 2D intermediate separately and return a list of
    (counts_2d, centers_x, centers_y) — one entry per chunk.

    This is used by the bootstrap to reconstruct per-path contributions.
    Because FLUSH_EVERY paths are merged into each chunk, we treat each
    chunk as a single independent block observation. This is a conservative
    approximation: within a chunk paths are summed, so variance is slightly
    underestimated. With FLUSH_EVERY=3000 and many chunks the approximation
    is accurate enough.
    """
    react, ens = group_key
    tag      = f"{react.replace('-', '')}_{ens}"
    safe_col = col.replace("/", "_").replace("\\", "_")
    pattern  = os.path.join(
        INTERMEDIATE_DIR,
        f"[0-9][0-9][0-9][0-9]__{tag}__2d__{safe_col}.npz"
    )
    npz_files = sorted(glob.glob(pattern))

    chunks = []
    for npz_path in npz_files:
        d = np.load(npz_path)
        chunks.append((d["counts"].copy(), d["centers_x"], d["centers_y"]))
    return chunks


# ── 9. WEIGHT VALIDATION ─────────────────────────────────────────────────────

def validate_weights(ensemble_path_weights):
    print("\n" + "=" * 60)
    print("WEIGHT VALIDATION")
    print("=" * 60)

    ensemble_totals = defaultdict(float)
    ensemble_counts = defaultdict(int)

    for key in ALL_GROUPS:
        react, ens = key
        pw        = ensemble_path_weights[key]
        group_sum = sum(pw.values())
        n_paths   = len(pw)
        print(f"  {react:15s} | {ens:5s}: {n_paths:5d} paths, "
              f"weight sum = {group_sum:.10f}")
        ensemble_totals[ens] += group_sum
        ensemble_counts[ens] += n_paths

    print()
    for ens in sorted(ensemble_totals):
        total     = ensemble_totals[ens]
        n_paths   = ensemble_counts[ens]
        deviation = abs(total - 1.0)
        status    = "OK" if deviation < 1e-6 else f"WARNING (off by {deviation:.6e})"
        print(f"  ALL {ens.upper():5s} paths combined : {n_paths:5d} paths, "
              f"weight sum = {total:.10f}  [{status}]")

    print("=" * 60)


# ── 10. PLOTTING ─────────────────────────────────────────────────────────────

def make_histogram_from_bins(centers, counts, col_name, group_label,
                             out_path, color, corrected=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    corrected = False
    finite = counts[np.isfinite(counts)]
    if finite.size == 0 or finite.sum() == 0:
        print(f"    SKIPPED (empty): {os.path.basename(out_path)}")
        return

    dx  = centers[1] - centers[0]
    total = np.nansum(counts)
    if total > 0:
        counts = counts / (total * dx)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(centers, counts, width=dx,
           color=color, edgecolor="white", linewidth=0.4, alpha=0.9)

    ax.set_xlabel(col_name, fontsize=15)
    ax.set_ylabel("Probability density / sin(\u03b8)" if corrected
                  else "Probability density", fontsize=15)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:,.4f}")
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=HIST_DPI)
    plt.close(fig)


def make_2d_histogram(centers_x, centers_y, counts_2d,
                      col_x, col_y, group_label, out_path,
                      corrected=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    corrected = False

    data = counts_2d.copy().astype(np.float64)
    data[~np.isfinite(data) | (data <= 0)] = np.nan

    if not np.any(np.isfinite(data)):
        print(f"    SKIPPED (empty 2D): {os.path.basename(out_path)}")
        return

    dx = centers_x[1] - centers_x[0]
    dy = centers_y[1] - centers_y[0]
    edges_x = np.concatenate(([centers_x[0] - dx / 2], centers_x + dx / 2))
    edges_y = np.concatenate(([centers_y[0] - dy / 2], centers_y + dy / 2))

    total = np.nansum(data)
    if total > 0:
        data = data / (total * dx * dy)

    pos_data = data[data > 0]
    if pos_data.size == 0:
        print(f"    SKIPPED (no positive 2D values): {os.path.basename(out_path)}")
        plt.close("all")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    vmin = float(np.nanmin(pos_data))
    vmax = float(np.nanmax(data))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    mesh = ax.pcolormesh(edges_x, edges_y, data,
                         norm=norm, cmap="viridis", shading="flat")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Probability density", fontsize=10)

    ax.set_xlabel(col_x, fontsize=15)
    ax.set_ylabel(col_y, fontsize=15)
    ax.tick_params(labelsize=13)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=HIST_DPI)
    plt.close(fig)


def make_2d_histogram_conditional(
    centers_x, centers_y, counts_2d,
    col_x, col_y, group_label, out_path,
    mode="y_given_x"
    ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = counts_2d.astype(np.float64).copy()
    data[~np.isfinite(data) | (data <= 0)] = 0.0

    if col_y in SPHERICAL_ANGLE_COLS:
        result = cosine_transform_2d(centers_x, centers_y, data)
        if result is None:
            return
        centers_y, data = result

    dx = centers_x[1] - centers_x[0]
    dy = centers_y[1] - centers_y[0]

    edges_x = np.concatenate(([centers_x[0] - dx / 2], centers_x + dx / 2))
    edges_y = np.concatenate(([centers_y[0] - dy / 2], centers_y + dy / 2))

    with np.errstate(divide="ignore", invalid="ignore"):
        if mode == "y_given_x":
            norm = np.nanmax(data, axis=1)[:, np.newaxis]
            prob = np.where(norm > 0, data / norm, np.nan)
            cbar_label = "FE"
        elif mode == "x_given_y":
            norm = np.nanmax(data, axis=0)[np.newaxis, :]
            prob = np.where(norm > 0, data / norm, np.nan)
            cbar_label = "FE"
        else:
            raise ValueError("mode must be 'y_given_x' or 'x_given_y'")

    with np.errstate(divide="ignore", invalid="ignore"):
        data = -np.log(prob)

    data -= np.nanmin(data)

    finite = data[np.isfinite(data) & (data > 0)]
    if finite.size == 0:
        return

    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))

    fig, ax = plt.subplots(figsize=(9, 6))

    mesh = ax.pcolormesh(
        edges_x, edges_y, data,
        cmap="RdYlBu_r", shading="flat"
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)

    ax.set_xlabel(f"{col_x}", fontsize=15)
    ax.set_ylabel(col_y, fontsize=15)
    ax.tick_params(labelsize=13)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=HIST_DPI)
    plt.close(fig)


def make_2d_histogram_free_energy(centers_x, centers_y, counts_2d,
                                   col_x, col_y, group_label, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = counts_2d.copy().astype(np.float64)
    data[~np.isfinite(data) | (data <= 0)] = 0.0

    if not np.any(np.isfinite(data)):
        print(f"    SKIPPED (empty FES): {os.path.basename(out_path)}")
        return

    if col_y in SPHERICAL_ANGLE_COLS:
        result = cosine_transform_2d(centers_x, centers_y, data)
        if result is None:
            return
        centers_y, data = result

    dx = centers_x[1] - centers_x[0]
    dy = centers_y[1] - centers_y[0]

    norm = np.nanmax(data)
    prob = data / norm

    with np.errstate(divide="ignore", invalid="ignore"):
        fes = -np.log(prob)

    finite_mask = np.isfinite(fes)
    if not finite_mask.any():
        print(f"    SKIPPED (all-inf FES): {os.path.basename(out_path)}")
        return
    fes -= np.nanmin(fes)

    edges_x = np.concatenate(([centers_x[0] - dx / 2], centers_x + dx / 2))
    edges_y = np.concatenate(([centers_y[0] - dy / 2], centers_y + dy / 2))

    fig, ax = plt.subplots(figsize=(9, 6))

    vmax = np.nanpercentile(fes[finite_mask], 95)
    mesh = ax.pcolormesh(edges_x, edges_y, fes,
                         cmap="RdYlBu_r", shading="flat")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(r"$-\ln\,p$ (kT)", fontsize=10)

    ax.set_xlabel(col_x, fontsize=15)
    ax.set_ylabel(col_y, fontsize=15)
    ax.tick_params(labelsize=13)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=HIST_DPI)
    plt.close(fig)


# ── 10b. DOPC vs POPC PREFERENCE: PAIRS + STOICHIOMETRY ──────────────────────

DOPC_POPC_PAIRS = [
    ("CA_C2_u_DOPC",    "CA_C2_u_POPC",     "CA_C2_upper"),
    ("CA_C2_l_DOPC",    "CA_C2_l_POPC",     "CA_C2_lower"),
    ("CA_P_u_DOPC",     "CA_P_u_POPC",      "CA_P_upper"),
    ("CA_P_l_DOPC",     "CA_P_l_POPC",      "CA_P_lower"),
    ("O_N_u_DOPC",      "O_N_u_POPC",       "O_N_upper"),
    ("O_N_l_DOPC",      "O_N_l_POPC",       "O_N_lower"),
    ("N_P_u_DOPC",      "N_P_u_POPC",       "N_P_upper"),
    ("N_P_l_DOPC",      "N_P_l_POPC",       "N_P_lower"),
    ("CA_CC2_u_DOPC",   "CA_CC2_u_POPC",    "CA_CC2_upper"),
    ("CA_CC2_l_DOPC",   "CA_CC2_l_POPC",    "CA_CC2_lower"),
    ("CA_CC3_u_DOPC",   "CA_CC3_u_POPC",    "CA_CC3_upper"),
    ("CA_CC3_l_DOPC",   "CA_CC3_l_POPC",    "CA_CC2_lower"),
    ("HA_P_u_DOPC",     "HA_P_u_POPC",      "HA_P_u_upper"),
    ("HA_P_l_DOPC",     "HA_P_l_POPC",      "HA_P_l_lower"),
    ("HA_O22_u_DOPC",   "HA_O22_u_POPC",    "HA_O22_u_upper"),
    ("HA_O22_l_DOPC",   "HA_O22_l_POPC",    "HA_O22_l_lower"),
    ("HA_O32_u_DOPC",   "HA_O32_u_POPC",    "HA_O32_u_upper"),
    ("HA_O32_l_DOPC",   "HA_O32_l_POPC",    "HA_O32_l_lower"),
    ("CD_P_u_DOPC",     "CD_P_u_POPC",      "CD_P_u_upper"),
    ("CD_P_l_DOPC",     "CD_P_l_POPC",      "CD_P_l_lower"),
    ("CD_O22_u_DOPC",   "CD_O22_u_POPC",    "CD_O22_u_upper"),
    ("CD_O22_l_DOPC",   "CD_O22_l_POPC",    "CD_O22_l_lower"),
    ("CD_O32_u_DOPC",   "CD_O32_u_POPC",    "CD_O32_u_upper"),
    ("CD_O32_l_DOPC",   "CD_O32_l_POPC",    "CD_O32_l_lower"),
    ("N_CC2_u_DOPC",    "N_CC2_u_POPC",     "N_CC2_u_upper"),
    ("N_CC2_l_DOPC",    "N_CC2_l_POPC",     "N_CC2_l_lower"),
    ("N_CC3_u_DOPC",    "N_CC3_u_POPC",     "N_CC3_u_upper"),
    ("N_CC3_l_DOPC",    "N_CC3_l_POPC",     "N_CC3_l_lower"),
    ("O_CC2_u_DOPC",    "O_CC2_u_POPC",     "O_CC2_u_upper"),
    ("O_CC2_l_DOPC",    "O_CC2_l_POPC",     "O_CC2_l_lower"),
    ("O_CC3_u_DOPC",    "O_CC3_u_POPC",     "O_CC3_u_upper"),
    ("O_CC3_l_DOPC",    "O_CC3_l_POPC",     "O_CC3_l_lower"),
]


# ═════════════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS: DOPC/POPC PREFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def _compute_enrichment_from_chunks(chunks_dopc, chunks_popc, lamb_centers):
    """
    Given a list of per-chunk 2D arrays (counts_2d, shape (n_cn, n_lamb))
    for DOPC and POPC, compute the observed DOPC enrichment ratio
    E_DOPC[i] = frac_DOPC[i] / F_DOPC at each OP_Lamb bin i.

    Returns
    -------
    dopc_counts : np.ndarray (n_lamb,)  — marginal DOPC weighted count
    popc_counts : np.ndarray (n_lamb,)  — marginal POPC weighted count
    frac_dopc   : np.ndarray (n_lamb,)  — DOPC fraction (NaN where masked)
    enrich_dopc : np.ndarray (n_lamb,)  — DOPC enrichment (NaN where masked)
    mask        : np.ndarray (n_lamb,)  bool — True where data is too sparse
    """
    n_lamb = len(lamb_centers)
    dopc_counts = np.zeros(n_lamb, dtype=np.float64)
    popc_counts = np.zeros(n_lamb, dtype=np.float64)

    for c2d, _, _ in chunks_dopc:
        dopc_counts += np.nansum(c2d, axis=0)
    for c2d, _, _ in chunks_popc:
        popc_counts += np.nansum(c2d, axis=0)

    total  = dopc_counts + popc_counts
    peak   = np.nanmax(total)
    mask   = (peak == 0) | (total < MIN_PEAK_FRACTION * peak)

    with np.errstate(invalid="ignore", divide="ignore"):
        frac_dopc   = np.where(~mask, dopc_counts / total, np.nan)
        enrich_dopc = np.where(~mask, frac_dopc / F_DOPC,  np.nan)

    return dopc_counts, popc_counts, frac_dopc, enrich_dopc, mask


def compute_dopc_popc_preference_statistics(group_key, col_dopc, col_popc,
                                             lamb_centers,
                                             n_bootstrap=N_BOOTSTRAP,
                                             alpha=ALPHA,
                                             rng=None):
    """
    Bootstrap (chunk-level) significance test for DOPC vs POPC contact
    preference along the OP_Lamb coordinate.

    The sampling unit is a **chunk** (a block of FLUSH_EVERY paths).  Chunks
    are the coarsest independent units available from the pre-binned
    intermediates.  Resampling chunks with replacement gives a distribution
    of enrichment ratios E_DOPC whose spread reflects path-level variability
    without the massive overcount that frame-level statistics would produce.

    Null hypothesis
    ---------------
    H0: E_DOPC = 1  (no preference beyond the 5:1 bulk stoichiometry)
    Two-sided alternative: H1: E_DOPC ≠ 1

    For each OP_Lamb bin the p-value is the fraction of bootstrap enrichments
    at least as extreme as observed (|E_boot - 1| >= |E_obs - 1|), clamped
    to [1/n_bootstrap, 1].

    Parameters
    ----------
    group_key   : tuple  e.g. ("non-reactive", "plus")
    col_dopc    : str    column name for DOPC contact number
    col_popc    : str    column name for POPC contact number
    lamb_centers: np.ndarray  OP_Lamb bin centers from the loaded intermediates
    n_bootstrap : int    number of chunk-level bootstrap resamples
    alpha       : float  significance level for the CI (two-sided)
    rng         : np.random.Generator or None

    Returns
    -------
    dict with keys:
        lamb_centers  : np.ndarray
        enrich_dopc   : np.ndarray (observed)
        enrich_popc   : np.ndarray (observed; = frac_POPC / F_POPC)
        frac_dopc     : np.ndarray (observed)
        frac_popc     : np.ndarray (observed)
        dopc_counts   : np.ndarray
        popc_counts   : np.ndarray
        ci_lo_dopc    : np.ndarray  (ALPHA/2 percentile of bootstrap E_DOPC)
        ci_hi_dopc    : np.ndarray  (1-ALPHA/2 percentile)
        ci_lo_popc    : np.ndarray
        ci_hi_popc    : np.ndarray
        p_value_dopc  : np.ndarray  (two-sided, vs H0: E=1)
        p_value_popc  : np.ndarray
        significant_dopc : np.ndarray (bool)
        significant_popc : np.ndarray (bool)
        mask          : np.ndarray (bool; True = too sparse, excluded)
        n_chunks      : int
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    # ── Load per-chunk 2D intermediates ───────────────────────────────────────
    chunks_dopc = load_per_chunk_2d(group_key, col_dopc)
    chunks_popc = load_per_chunk_2d(group_key, col_popc)

    if not chunks_dopc or not chunks_popc:
        return None

    n_chunks = min(len(chunks_dopc), len(chunks_popc))
    chunks_dopc = chunks_dopc[:n_chunks]
    chunks_popc = chunks_popc[:n_chunks]

    n_lamb = len(lamb_centers)

    # ── Observed enrichment ───────────────────────────────────────────────────
    (dopc_counts_obs, popc_counts_obs,
     frac_dopc_obs, enrich_dopc_obs, mask) = _compute_enrichment_from_chunks(
        chunks_dopc, chunks_popc, lamb_centers
    )
    frac_popc_obs   = np.where(~mask, 1.0 - frac_dopc_obs, np.nan)
    enrich_popc_obs = np.where(~mask, frac_popc_obs / F_POPC, np.nan)

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    boot_enrich_dopc = np.full((n_bootstrap, n_lamb), np.nan)
    boot_enrich_popc = np.full((n_bootstrap, n_lamb), np.nan)

    idx_pool = np.arange(n_chunks)

    for b in range(n_bootstrap):
        boot_idx        = rng.integers(0, n_chunks, size=n_chunks)
        boot_cd         = [chunks_dopc[i] for i in boot_idx]
        boot_cp         = [chunks_popc[i] for i in boot_idx]
        _, _, _, be_d, bmask = _compute_enrichment_from_chunks(
            boot_cd, boot_cp, lamb_centers
        )
        bf_d = np.where(~bmask, be_d * F_DOPC, np.nan)   # recover frac
        bf_p = np.where(~bmask, 1.0 - bf_d, np.nan)
        be_p = np.where(~bmask, bf_p / F_POPC, np.nan)

        boot_enrich_dopc[b] = be_d
        boot_enrich_popc[b] = be_p

    # ── Confidence intervals (percentile method) ───────────────────────────────
    ci_lo_q = alpha / 2 * 100
    ci_hi_q = (1.0 - alpha / 2) * 100

    ci_lo_dopc = np.nanpercentile(boot_enrich_dopc, ci_lo_q, axis=0)
    ci_hi_dopc = np.nanpercentile(boot_enrich_dopc, ci_hi_q, axis=0)
    ci_lo_popc = np.nanpercentile(boot_enrich_popc, ci_lo_q, axis=0)
    ci_hi_popc = np.nanpercentile(boot_enrich_popc, ci_hi_q, axis=0)

    # ── Two-sided p-values (H0: E = 1) ────────────────────────────────────────
    obs_dev_d   = np.abs(enrich_dopc_obs - 1.0)           # (n_lamb,)
    obs_dev_p   = np.abs(enrich_popc_obs - 1.0)
    valid_d     = ~np.isnan(boot_enrich_dopc)              # (n_bootstrap, n_lamb)
    valid_p     = ~np.isnan(boot_enrich_popc)
    n_valid_d   = valid_d.sum(axis=0)                      # (n_lamb,)
    n_valid_p   = valid_p.sum(axis=0)
    extreme_d   = (np.abs(boot_enrich_dopc - 1.0) >= obs_dev_d[np.newaxis, :]).sum(axis=0).astype(float)
    extreme_p   = (np.abs(boot_enrich_popc - 1.0) >= obs_dev_p[np.newaxis, :]).sum(axis=0).astype(float)
    safe_n_d    = np.where(n_valid_d > 0, n_valid_d, 1)
    safe_n_p    = np.where(n_valid_p > 0, n_valid_p, 1)

    p_value_dopc = np.where(
        (~mask) & (n_valid_d >= 10),
        np.maximum(extreme_d / safe_n_d, 1.0 / n_bootstrap),
        np.nan,
    )
    p_value_popc = np.where(
        (~mask) & (n_valid_p >= 10),
        np.maximum(extreme_p / safe_n_p, 1.0 / n_bootstrap),
        np.nan,
    )

    significant_dopc = (~np.isnan(p_value_dopc)) & (p_value_dopc < alpha)
    significant_popc = (~np.isnan(p_value_popc)) & (p_value_popc < alpha)

    return {
        "lamb_centers":      lamb_centers,
        "enrich_dopc":       enrich_dopc_obs,
        "enrich_popc":       enrich_popc_obs,
        "frac_dopc":         frac_dopc_obs,
        "frac_popc":         frac_popc_obs,
        "dopc_counts":       dopc_counts_obs,
        "popc_counts":       popc_counts_obs,
        "ci_lo_dopc":        ci_lo_dopc,
        "ci_hi_dopc":        ci_hi_dopc,
        "ci_lo_popc":        ci_lo_popc,
        "ci_hi_popc":        ci_hi_popc,
        "p_value_dopc":      p_value_dopc,
        "p_value_popc":      p_value_popc,
        "significant_dopc":  significant_dopc,
        "significant_popc":  significant_popc,
        "mask":              mask,
        "n_chunks":          n_chunks,
    }


def write_preference_stats_csv(all_pair_stats, out_path):
    """
    Write per-pair, per-OP_Lamb-bin statistics to a CSV file.

    Parameters
    ----------
    all_pair_stats : dict  { label: stats_dict }
        stats_dict is the return value of compute_dopc_popc_preference_statistics()
    out_path : str
    """
    fieldnames = [
        "pair_label", "OP_Lamb",
        "frac_dopc", "frac_popc",
        "enrich_dopc", "enrich_popc",
        "ci_lo_dopc", "ci_hi_dopc",
        "ci_lo_popc", "ci_hi_popc",
        "p_value_dopc", "p_value_popc",
        "significant_dopc", "significant_popc",
        "dopc_counts", "popc_counts",
    ]

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for label, st in all_pair_stats.items():
            if st is None:
                continue
            n = len(st["lamb_centers"])
            for i in range(n):
                if st["mask"][i]:
                    continue
                writer.writerow({
                    "pair_label":       label,
                    "OP_Lamb":          f"{st['lamb_centers'][i]:.6g}",
                    "frac_dopc":        f"{st['frac_dopc'][i]:.6g}",
                    "frac_popc":        f"{st['frac_popc'][i]:.6g}",
                    "enrich_dopc":      f"{st['enrich_dopc'][i]:.6g}",
                    "enrich_popc":      f"{st['enrich_popc'][i]:.6g}",
                    "ci_lo_dopc":       f"{st['ci_lo_dopc'][i]:.6g}",
                    "ci_hi_dopc":       f"{st['ci_hi_dopc'][i]:.6g}",
                    "ci_lo_popc":       f"{st['ci_lo_popc'][i]:.6g}",
                    "ci_hi_popc":       f"{st['ci_hi_popc'][i]:.6g}",
                    "p_value_dopc":     f"{st['p_value_dopc'][i]:.6g}",
                    "p_value_popc":     f"{st['p_value_popc'][i]:.6g}",
                    "significant_dopc": st["significant_dopc"][i],
                    "significant_popc": st["significant_popc"][i],
                    "dopc_counts":      f"{st['dopc_counts'][i]:.6g}",
                    "popc_counts":      f"{st['popc_counts'][i]:.6g}",
                })
    print(f"  Preference stats -> {os.path.relpath(out_path)}")


# ═════════════════════════════════════════════════════════════════════════════
# PREFERENCE PLOT (updated to include CI bands and significance markers)
# ═════════════════════════════════════════════════════════════════════════════

def make_dopc_popc_preference_plot(group_key, out_dir,
                                   n_bootstrap=N_BOOTSTRAP, alpha=ALPHA):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    react, ens  = group_key
    group_label = f"{react} | {ens} ensemble"

    os.makedirs(out_dir, exist_ok=True)

    all_pair_stats = {}
    rng = np.random.default_rng(seed=42)

    for col_dopc, col_popc, label in DOPC_POPC_PAIRS:
        # ── Load 2D intermediates ──────────────────────────────────────────
        cx_d, cy_d, c2d_dopc = load_and_sum_intermediates(
            group_key, col_dopc, kind="2d"
        )
        cx_p, cy_p, c2d_popc = load_and_sum_intermediates(
            group_key, col_popc, kind="2d"
        )

        if cx_d is None or cx_p is None:
            print(f"    [{group_label}] SKIPPED (missing intermediates): {label}")
            all_pair_stats[label] = None
            continue

        if not np.allclose(cx_d, cx_p):
            print(f"    [{group_label}] SKIPPED (OP_Lamb axes differ): {label}")
            all_pair_stats[label] = None
            continue

        lamb_centers = cx_d

        # ── Compute bootstrap statistics ───────────────────────────────────
        print(f"    [{group_label}] Bootstrap for {label} "
              f"({n_bootstrap} resamples)...")
        stats = compute_dopc_popc_preference_statistics(
            group_key, col_dopc, col_popc,
            lamb_centers, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng
        )

        if stats is None:
            print(f"    [{group_label}] SKIPPED (no chunk data): {label}")
            all_pair_stats[label] = None
            continue

        all_pair_stats[label] = stats

        # Convenience aliases
        frac_dopc   = stats["frac_dopc"]
        frac_popc   = stats["frac_popc"]
        enrich_dopc = stats["enrich_dopc"]
        enrich_popc = stats["enrich_popc"]
        ci_lo_d     = stats["ci_lo_dopc"]   # CI on enrichment
        ci_hi_d     = stats["ci_hi_dopc"]
        ci_lo_p     = stats["ci_lo_popc"]
        ci_hi_p     = stats["ci_hi_popc"]
        sig_dopc    = stats["significant_dopc"]
        sig_popc    = stats["significant_popc"]

        # Convert enrichment CI back to fractional-contact CI for plotting
        frac_ci_lo_d = ci_lo_d * F_DOPC
        frac_ci_hi_d = ci_hi_d * F_DOPC
        frac_ci_lo_p = ci_lo_p * F_POPC
        frac_ci_hi_p = ci_hi_p * F_POPC

        # ── Per-molecule CN (stoich-normalised) — from full 2D histograms ──
        dopc_counts = np.nansum(c2d_dopc, axis=0).astype(np.float64)
        popc_counts = np.nansum(c2d_popc, axis=0).astype(np.float64)
        total_counts = dopc_counts + popc_counts
        peak = np.nanmax(total_counts)
        mask = stats["mask"]

        with np.errstate(invalid="ignore", divide="ignore"):
            dopc_mean_cn = (
                np.nansum(c2d_dopc * cy_d[:, np.newaxis], axis=0) / dopc_counts
            )
            popc_mean_cn = (
                np.nansum(c2d_popc * cy_p[:, np.newaxis], axis=0) / popc_counts
            )

        dopc_mean_cn[mask] = np.nan
        popc_mean_cn[mask] = np.nan

        cn_per_dopc = dopc_mean_cn / F_DOPC
        cn_per_popc = popc_mean_cn / F_POPC
        denom_cn    = cn_per_dopc + cn_per_popc
        with np.errstate(invalid="ignore", divide="ignore"):
            cn_per_dopc_norm = cn_per_dopc / denom_cn
            cn_per_popc_norm = cn_per_popc / denom_cn

        # ── PLOT 1: fractional contact + enrichment + CI bands ─────────────
        fig, ax1 = plt.subplots(figsize=(9, 5))

        ax1.plot(lamb_centers, frac_popc, color="#e05c2a", lw=2.0,
                 label="POPC fraction")
        ax1.plot(lamb_centers, frac_dopc, color="#2a7fbf", lw=2.0,
                 label="DOPC fraction")

        # Bootstrap 95 % CI bands (on fractional contact)
        ax1.fill_between(lamb_centers, frac_ci_lo_d, frac_ci_hi_d,
                         color="#2a7fbf", alpha=0.20,
                         label=f"DOPC {int((1-alpha)*100)}% CI")
        ax1.fill_between(lamb_centers, frac_ci_lo_p, frac_ci_hi_p,
                         color="#e05c2a", alpha=0.20,
                         label=f"POPC {int((1-alpha)*100)}% CI")

        ax1.axhline(F_POPC, color="#e05c2a", lw=1.0, ls="--",
                    label=f"POPC stoich. ({F_POPC:.3f})")
        ax1.axhline(F_DOPC, color="#2a7fbf", lw=1.0, ls="--",
                    label=f"DOPC stoich. ({F_DOPC:.3f})")

        # Significance markers at the bottom of the plot
        sig_y_d = np.full(len(lamb_centers), np.nan)
        sig_y_p = np.full(len(lamb_centers), np.nan)
        sig_y_d[sig_dopc] = 0.02
        sig_y_p[sig_popc] = 0.05
        ax1.scatter(lamb_centers[sig_dopc], sig_y_d[sig_dopc],
                    marker="^", color="#2a7fbf", s=30, zorder=5,
                    label=f"DOPC sig. (p<{alpha})")
        ax1.scatter(lamb_centers[sig_popc], sig_y_p[sig_popc],
                    marker="v", color="#e05c2a", s=30, zorder=5,
                    label=f"POPC sig. (p<{alpha})")

        ax1.set_xlabel("OP_Lamb", fontsize=14)
        ax1.set_ylabel("Fractional contact", fontsize=14)
        ax1.set_ylim(0.0, 1.0)
        ax1.tick_params(labelsize=12)
        ax1.spines[["top", "right"]].set_visible(False)

        ax2 = ax1.twinx()
        ax2.plot(lamb_centers, enrich_popc, color="#e05c2a",
                 lw=1.5, ls=":", alpha=0.7, label="POPC enrichment")
        ax2.fill_between(lamb_centers, ci_lo_p, ci_hi_p,
                         color="#e05c2a", alpha=0.10)
        ax2.axhline(1.0, color="grey", lw=0.8, ls=":")
        ax2.set_ylabel("Enrichment (frac / stoich)", fontsize=11, color="grey")
        ax2.tick_params(labelsize=10, colors="grey")
        ax2.spines["right"].set_edgecolor("grey")

        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2,
                   fontsize=8, loc="upper left",
                   framealpha=0.85, edgecolor="#cccccc")

        fig.suptitle(f"{label}  —  {group_label}", fontsize=11, color="#444444")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{label}_preference.png"), dpi=HIST_DPI)
        plt.close(fig)

        # ── PLOT 2: per-molecule CN (stoich-normalised) ───────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(lamb_centers, cn_per_dopc_norm, color="#2a7fbf", lw=2.0,
                label="DOPC (per-molecule, normalised)")
        ax.plot(lamb_centers, cn_per_popc_norm, color="#e05c2a", lw=2.0,
                label="POPC (per-molecule, normalised)")
        ax.axhline(0.5, color="grey", lw=1.0, ls="--",
                   label="Equal per-molecule contact (0.5)")

        ax.set_xlabel("OP_Lamb", fontsize=14)
        ax.set_ylabel("Normalised CN per lipid molecule", fontsize=13)
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(labelsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=10, framealpha=0.85, edgecolor="#cccccc")

        fig.suptitle(f"{label}  —  {group_label}", fontsize=11, color="#444444")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{label}_cn_per_lipid.png"), dpi=HIST_DPI)
        plt.close(fig)

        # ── PLOT 3: enrichment ratio with CI band ─────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.fill_between(lamb_centers, ci_lo_d, ci_hi_d,
                        color="#2a7fbf", alpha=0.25, label=f"DOPC {int((1-alpha)*100)}% CI")
        ax.fill_between(lamb_centers, ci_lo_p, ci_hi_p,
                        color="#e05c2a", alpha=0.25, label=f"POPC {int((1-alpha)*100)}% CI")
        ax.plot(lamb_centers, enrich_dopc, color="#2a7fbf", lw=2.0,
                label="DOPC enrichment")
        ax.plot(lamb_centers, enrich_popc, color="#e05c2a", lw=2.0,
                label="POPC enrichment")
        ax.axhline(1.0, color="black", lw=1.0, ls="--", label="Null (bulk ratio)")
        ax.scatter(lamb_centers[sig_dopc], enrich_dopc[sig_dopc],
                   marker="*", color="#2a7fbf", s=60, zorder=5,
                   label=f"DOPC sig. (p<{alpha})")
        ax.scatter(lamb_centers[sig_popc], enrich_popc[sig_popc],
                   marker="*", color="#e05c2a", s=60, zorder=5,
                   label=f"POPC sig. (p<{alpha})")

        ax.set_xlabel("OP_Lamb", fontsize=14)
        ax.set_ylabel("Enrichment ratio E = frac / stoich", fontsize=13)
        ax.tick_params(labelsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, framealpha=0.85, edgecolor="#cccccc")

        fig.suptitle(f"{label}  —  {group_label}\n"
                     f"E > 1: enriched near permeant   |   "
                     f"E < 1: depleted",
                     fontsize=10, color="#444444")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{label}_enrichment.png"), dpi=HIST_DPI)
        plt.close(fig)

    # ── Write statistics CSV ───────────────────────────────────────────────
    react_tag = react.replace("-", "")
    stats_csv = os.path.join(out_dir, f"preference_stats_{react_tag}_{ens}.csv")
    write_preference_stats_csv(all_pair_stats, stats_csv)

    print(f"  [{group_label}] DOPC/POPC preference plots + stats -> "
          f"{os.path.relpath(out_dir)}/")


def cosine_transform_2d(centers_x, centers_y, counts_2d):
    finite = counts_2d[np.isfinite(counts_2d)]
    if finite.size == 0 or finite.sum() == 0:
        return None

    rad_y = np.deg2rad(centers_y)
    u_vals = np.cos(rad_y)

    n_ref = len(centers_x)
    n_u   = N_COSINE_BINS
    u_min, u_max = -1.0, 1.0
    du    = (u_max - u_min) / n_u
    u_centers = u_min + np.arange(n_u) * du + 0.5 * du

    hist_u = np.zeros((n_u, n_ref), dtype=np.float64)

    iu_arr = ((u_vals - u_min) / du).astype(int)
    valid  = np.isfinite(u_vals) & (iu_arr >= 0) & (iu_arr < n_u)
    np.add.at(hist_u, iu_arr[valid], counts_2d[valid, :])

    data = hist_u
    data[~np.isfinite(data) | (data <= 0)] = np.nan
    return u_centers, data


def make_cosine_transform_2d_histogram(centers_x, centers_y, counts_2d,
                                        col_x, col_y, group_label, out_path):
    finite = counts_2d[np.isfinite(counts_2d)]
    if finite.size == 0 or finite.sum() == 0:
        print(f"    SKIPPED (empty 2D cosine-transform): {os.path.basename(out_path)}")
        return

    rad_y = np.deg2rad(centers_y)
    if col_y == "PRO_r_plane":
        u_vals  = np.cos(rad_y) # Test to see if this works better than sin for the r-plane angle
        u_label = f"cos({col_y})"# [elevation]"
    else:
        u_vals  = np.cos(rad_y)
        u_label = f"cos({col_y})"

    n_ref = len(centers_x)
    n_u   = N_COSINE_BINS
    u_min, u_max = -1.0, 1.0
    du    = (u_max - u_min) / n_u
    u_centers = u_min + np.arange(n_u) * du + 0.5 * du

    hist_u = np.zeros((n_u, n_ref), dtype=np.float64)

    iu_arr = ((u_vals - u_min) / du).astype(int)
    valid  = np.isfinite(u_vals) & (iu_arr >= 0) & (iu_arr < n_u)
    np.add.at(hist_u, iu_arr[valid], counts_2d[valid, :])

    data = hist_u
    data[~np.isfinite(data) | (data <= 0)] = np.nan

    if not np.any(np.isfinite(data)):
        print(f"    SKIPPED (empty after rebinning): {os.path.basename(out_path)}")
        return

    dx = centers_x[1] - centers_x[0]
    edges_x = np.concatenate(([centers_x[0] - dx / 2], centers_x + dx / 2))
    edges_u = np.concatenate(([u_min], u_centers + du / 2))

    total = np.nansum(data)
    if total > 0:
        data = data / (total * dx * du)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=(9, 6))
    vmin = float(np.nanmin(data[data > 0]))
    vmax = float(np.nanmax(data))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    mesh = ax.pcolormesh(edges_x, edges_u, data,
                         norm=norm, cmap="viridis", shading="flat")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Probability density", fontsize=10)
    ax.set_xlabel(col_x, fontsize=15)
    ax.set_ylabel(u_label, fontsize=15)
    ax.tick_params(labelsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=HIST_DPI)
    plt.close(fig)


# ── 10c. CROSS-SIMULATION OP_Lamb MERGE ──────────────────────────────────────

def resolve_data_path(path, must_be_dir=False):
    """
    Resolve a configured path to the other run against the CWD and a couple of
    parent prefixes, so the script runs from either analysis folder without the
    config having to be edited.  Returns None if nothing matches.
    """
    if os.path.isabs(path):
        cands = [path]
    else:
        cands = [path,
                 os.path.join("..", path),
                 os.path.join("..", "..", path),
                 os.path.join("..", "..", "..", path)]
    for c in cands:
        ok = os.path.isdir(c) if must_be_dir else os.path.isfile(c)
        if ok:
            return os.path.normpath(c)
    return None


def parse_op_column(args):
    """
    Read a single ML/<path_num>.txt and return only the REF_COL (OP_Lamb)
    values, already multiplied by `sign`.  Returns None when the file belongs to
    an ensemble we are not merging, or holds no usable rows.

    Row selection mirrors classify_file(): header on line 4, data from line 5
    up to but excluding the final line.
    """
    filepath, sign = args
    try:
        with open(filepath, "r") as fh:
            lines = fh.readlines()

        if len(lines) < 5:
            return None

        line2 = lines[1].strip().lstrip("#").strip()
        if "plus" in line2:
            ensemble = "plus"
        elif "minus" in line2:
            ensemble = "minus"
        else:
            return None
        if ensemble not in MERGE_ENSEMBLES:
            return None

        headers = lines[3].strip().lstrip("#").strip().split()
        if REF_COL not in headers:
            return None
        ci = headers.index(REF_COL)

        vals = []
        for line in lines[4:-1]:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if ci >= len(parts):
                continue
            try:
                vals.append(float(parts[ci]))
            except ValueError:
                continue

        if not vals:
            return None

        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None

        return os.path.basename(filepath), sign * arr

    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        return None


def build_op_histogram(ml_dir, weights_file, centers, Minx, dx, sign, label):
    """
    Build a path-weight-weighted OP_Lamb histogram from an ML/ folder, on the
    bin grid given by (centers, Minx, dx).  `sign` is +1 to take the values as
    they are, or -1 to fold them in reversed (mirror symmetry).

    Returns (hist, meta) where meta carries the bookkeeping used for reporting.
    """
    n_bins = len(centers)
    hist   = np.zeros(n_bins, dtype=np.float64)

    print(f"\n  Reading {label} from {ml_dir!r}")
    ordered = load_path_weights(weights_file)
    lookup  = {pn: w for pn, w in ordered}

    files, missing = [], []
    for pn, _ in ordered:
        fp = os.path.join(ml_dir, f"{pn}{FILE_PATTERN}")
        if os.path.isfile(fp):
            files.append((fp, float(sign)))
        else:
            missing.append(pn)

    if missing:
        print(f"  WARNING: {len(missing)} weighted paths have no file in {ml_dir!r} "
              f"(their weight is dropped)")

    if not files:
        print(f"  ERROR: no files found for {label}.")
        return hist, {"n_paths": 0, "w_used": 0.0, "n_frames": 0,
                      "n_missing": len(missing), "n_out": 0,
                      "op_min": np.nan, "op_max": np.nan}

    vmax_edge = Minx + n_bins * dx
    n_paths = n_frames = n_out = 0
    w_used  = 0.0
    op_min, op_max = np.inf, -np.inf

    with Pool(processes=N_WORKERS) as pool:
        it = pool.imap_unordered(parse_op_column, files, chunksize=64)
        if HAS_TQDM:
            it = tqdm(it, total=len(files), desc=f"  {label}", unit="file")
        for res in it:
            if res is None:
                continue
            name, vals = res
            pn = int(os.path.splitext(name)[0])
            w  = lookup[pn]

            op_min = min(op_min, float(vals.min()))
            op_max = max(op_max, float(vals.max()))
            n_out += int(np.sum((vals < Minx) | (vals >= vmax_edge)))

            update_histogram(vals, w, hist, dx, Minx)

            n_paths  += 1
            w_used   += w
            n_frames += vals.size

    meta = {"n_paths": n_paths, "w_used": w_used, "n_frames": n_frames,
            "n_missing": len(missing), "n_out": n_out,
            "op_min": op_min, "op_max": op_max}

    print(f"    {label}: {n_paths} paths, weight sum = {w_used:.6f}, "
          f"{n_frames:,} phase points")
    print(f"    OP_Lamb range after sign={sign:+.0f}: "
          f"[{op_min:.3f}, {op_max:.3f}]  (bins cover [{Minx:.3f}, {vmax_edge:.3f}])")
    if n_out:
        print(f"    WARNING: {n_out:,} phase points fell outside the bin range "
              f"and were discarded")
    if n_paths and n_out == n_frames:
        print(f"    WARNING: {label} contributed nothing — check SYMMETRIC_OP "
              f"and the OP_Lamb entry in COLUMN_RANGES.")

    return hist, meta


def fit_match_factor(centers, h_a, h_b, match_range=None, min_frac=MERGE_MIN_FRAC):
    """
    Fit the single multiplicative factor s that best maps h_b onto h_a, by
    averaging log(h_a / h_b) over the bins where both histograms carry
    meaningful weight.  A constant factor in probability is a constant offset in
    free energy, which is the only thing that is undetermined between two
    independently normalised path ensembles.

    Returns (s, n_overlap, rms_kT) where rms_kT is the residual spread of
    log(h_a) - log(s * h_b) over the fit window, in kT.  A large rms means the
    two runs disagree in SHAPE, not just in normalisation, and no single factor
    can reconcile them.
    """
    ok = (h_a > 0) & (h_b > 0) & np.isfinite(h_a) & np.isfinite(h_b)
    if h_a.max() > 0:
        ok &= h_a >= min_frac * h_a.max()
    if h_b.max() > 0:
        ok &= h_b >= min_frac * h_b.max()
    if match_range is not None:
        lo, hi = match_range
        ok &= (centers >= lo) & (centers <= hi)

    n = int(ok.sum())
    if n == 0:
        return np.nan, 0, np.nan

    logratio = np.log(h_a[ok]) - np.log(h_b[ok])
    s   = float(np.exp(np.mean(logratio)))
    rms = float(np.sqrt(np.mean((logratio - np.mean(logratio)) ** 2)))
    return s, n, rms


def escape_correction_factor():
    """
    correction_factor = (N_A * (P_A / P_C_min) * (P_C / P_D_min)) / N_D

    Returns (factor, missing) where `missing` lists the names of the crossing
    probabilities that have not been filled in.  When anything is missing, or
    any value is non-positive, the factor comes back as NaN.
    """
    supplied = {"P_A": P_A, "P_C": P_C, "P_C_MIN": P_C_MIN, "P_D_MIN": P_D_MIN}
    missing  = [name for name, val in supplied.items() if val is None]
    if missing:
        return np.nan, missing

    vals = {name: float(val) for name, val in supplied.items() if val is not None}
    bad  = [name for name, val in vals.items() if not val > 0.0]
    if bad:
        print(f"  ERROR: crossing probabilities must be positive, got "
              f"{ {n: vals[n] for n in bad} }")
        return np.nan, bad

    factor = ((float(N_A) * (vals["P_A"] / vals["P_C_MIN"])
                          * (vals["P_C"] / vals["P_D_MIN"])) / float(N_D))
    return float(factor), []


def merge_op_histograms(centers, h_local, h_other,
                        mode=MERGE_NORM,
                        match_range=MERGE_MATCH_RANGE):
    """
    Put the two OP_Lamb histograms on a common scale and add them.

    Each mode determines the relative scale of the two runs on its own; the two
    are then pooled directly, with no free mixing weight on top.

    Returns (merged, info).  `merged` is rescaled so its total weight equals
    that of h_local, which keeps the downstream density normalisation and the
    stats CSV on the same footing as the un-merged run.
    """
    info = {"mode": mode, "scale": 1.0,
            "n_overlap": 0, "rms_kT": np.nan, "fitted_scale": np.nan,
            "mul_local": 1.0, "mul_other": 1.0}

    tot_l = float(np.nansum(h_local))
    tot_o = float(np.nansum(h_other))
    if tot_o <= 0:
        print("  Other run contributed no weight — merge skipped.")
        return h_local.copy(), info

    if mode == "correction":
        factor, missing = escape_correction_factor()
        if missing:
            print(f"  WARNING: the escape correction factor is not usable — "
                  f"missing/invalid: {', '.join(missing)}.")
            print("           Fill these in at the top of the script (last "
                  "row of each run's wham/Pcross.txt, column P-wham).")
            print("           Falling back to the fitted 'match' factor, "
                  "which is NOT physically referenced.")
            return merge_op_histograms(centers, h_local, h_other,
                                       mode="match",
                                       match_range=match_range)

        # Report how the physical factor compares with what the data implies.
        fitted, n_ov, rms = fit_match_factor(centers, h_local, h_other,
                                             match_range)
        info.update(scale=factor, n_overlap=n_ov, rms_kT=rms,
                    fitted_scale=fitted)

        # The factor fixes the relative scale physically; the two are then
        # simply pooled.
        if CORRECTION_APPLY_TO == "local":
            mul_l, mul_o = factor, 1.0
        elif CORRECTION_APPLY_TO == "other":
            mul_l, mul_o = 1.0, factor
        elif CORRECTION_APPLY_TO == "none":
            mul_l, mul_o = 1.0, 1.0
        else:
            raise ValueError(f"CORRECTION_APPLY_TO must be 'local', 'other' or "
                             f"'none', got {CORRECTION_APPLY_TO!r}")

    elif mode == "raw":
        mul_l, mul_o = 1.0, 1.0

    elif mode == "area":
        mul_l = 1.0 / tot_l if tot_l > 0 else 1.0
        mul_o = 1.0 / tot_o

    elif mode == "match":
        s, n_ov, rms = fit_match_factor(centers, h_local, h_other, match_range)
        if not np.isfinite(s):
            print("  WARNING: no overlapping bins to match on — falling back "
                  "to 'area'.")
            return merge_op_histograms(centers, h_local, h_other, mode="area")
        info.update(scale=s, n_overlap=n_ov, rms_kT=rms)
        mul_l, mul_o = 1.0, s

    else:
        raise ValueError(f"MERGE_NORM must be 'correction', 'match', 'area' or "
                         f"'raw', got {mode!r}")

    info.update(mul_local=float(mul_l), mul_other=float(mul_o))
    merged = mul_l * h_local + mul_o * h_other
    tot_m  = float(np.nansum(merged))
    if tot_l > 0 and tot_m > 0:
        merged *= tot_l / tot_m
    return merged, info


def plot_op_merge(centers, h_local, h_other, merged, info,
                  out_density, out_fes, local_label, other_label):
    """
    Two-panel diagnostic: the weighted densities on a log axis, and the
    corresponding free energies -ln p (each shifted to its own minimum).  The
    other run is drawn with the scale factor applied, so a shape disagreement is
    immediately visible as non-parallel curves.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dx = centers[1] - centers[0]

    def density(h):
        t = np.nansum(h)
        return h / (t * dx) if t > 0 else h

    def fes(h):
        p = density(h)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = -np.log(np.where(p > 0, p, np.nan))
        if np.any(np.isfinite(f)):
            f = f - np.nanmin(f)
        return f

    # Draw each run with the multiplier the merge actually applied, so a shape
    # disagreement shows up as non-parallel curves rather than as an offset.
    mul_l = info.get("mul_local", 1.0)
    mul_o = info.get("mul_other", 1.0)
    scale = mul_o / mul_l if mul_l else 1.0

    d_loc = density(h_local)
    d_oth = density(h_other * scale)
    d_mrg = density(merged)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(centers, d_loc, lw=1.6, color="#1a5c8a", label=local_label)
    ax.plot(centers, d_oth, lw=1.6, color="#8a1a0d", label=other_label)
    ax.plot(centers, d_mrg, lw=2.2, color="#4a1a6b", label="merged")
    ax.set_yscale("log")
    ax.set_xlabel(REF_COL, fontsize=15)
    ax.set_ylabel("Probability density", fontsize=15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=13)
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_density, dpi=HIST_DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(centers, fes(h_local), lw=1.6, color="#1a5c8a", label=local_label)
    ax.plot(centers, fes(h_other), lw=1.6, color="#8a1a0d", label=other_label)
    ax.plot(centers, fes(merged),  lw=2.2, color="#4a1a6b", label="merged")
    ax.set_xlabel(REF_COL, fontsize=15)
    ax.set_ylabel(r"$-\ln\,p$ (kT)", fontsize=15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=13)
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_fes, dpi=HIST_DPI)
    plt.close(fig)


def write_op_merge_txt(centers, h_local, h_other, merged, out_path):
    """Dump the merged OP_Lamb histogram and its free energy as plain text."""
    dx = centers[1] - centers[0]

    def fes(h):
        t = np.nansum(h)
        if t <= 0:
            return np.full_like(h, np.nan, dtype=np.float64)
        p = h / (t * dx)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = -np.log(np.where(p > 0, p, np.nan))
        return f - np.nanmin(f) if np.any(np.isfinite(f)) else f

    f_loc, f_oth, f_mrg = fes(h_local), fes(h_other), fes(merged)

    with open(out_path, "w") as fh:
        fh.write(f"# {REF_COL} histogram merged across simulations\n")
        fh.write("# columns: center  counts_local  counts_other  counts_merged"
                 "  FE_local(kT)  FE_other(kT)  FE_merged(kT)\n")
        for i, c in enumerate(centers):
            fh.write(f"{c:.6f} {h_local[i]:.10e} {h_other[i]:.10e} "
                     f"{merged[i]:.10e} {f_loc[i]:.6f} {f_oth[i]:.6f} "
                     f"{f_mrg[i]:.6f}\n")
    print(f"  Saved merged data : {os.path.relpath(out_path)}")


def apply_merged_op_marginal(counts_2d, centers_x, merged_centers, merged_counts):
    """
    Rescale a 2-D histogram column-by-column along OP_Lamb so that its OP_Lamb
    marginal equals the merged 1-D histogram.

    counts_2d has shape (n_col_bins, n_lamb_bins) with OP_Lamb along axis 1,
    which is what flush_to_intermediate() writes and what the 2-D plotters
    expect.

    Only the OP_Lamb marginal comes from the merge.  The conditional
    distribution P(col | OP_Lamb) within each OP_Lamb column is left untouched,
    because the other run does not share these feature columns — it can only
    tell us how much weight belongs at each OP_Lamb, not how it is distributed
    over the column.

    Returns (rescaled, n_scaled, n_gap) where n_gap counts OP_Lamb bins that the
    merge gives weight to but this run never sampled, so nothing can be placed
    there.
    """
    if merged_counts is None or counts_2d is None:
        return counts_2d, 0, 0

    if counts_2d.shape[1] != len(merged_counts):
        print(f"  WARNING: merged marginal has {len(merged_counts)} bins but the "
              f"2-D histogram has {counts_2d.shape[1]} — not rescaled.")
        return counts_2d, 0, 0

    if not np.allclose(centers_x, merged_centers):
        print("  WARNING: OP_Lamb centers differ between the merged histogram "
              "and the 2-D histogram — not rescaled.")
        return counts_2d, 0, 0

    marg = np.nansum(counts_2d, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(marg > 0, merged_counts / marg, 0.0)
    scale = np.where(np.isfinite(scale), scale, 0.0)

    rescaled = counts_2d.astype(np.float64) * scale[np.newaxis, :]
    n_scaled = int(np.sum((marg > 0) & (merged_counts > 0)))
    n_gap    = int(np.sum((marg <= 0) & (merged_counts > 0)))
    return rescaled, n_scaled, n_gap


def run_op_merge(all_bin_infos):
    """
    Fold the other run's plus-ensemble OP_Lamb phase points (and optionally this
    run's own, reversed) into the combined-plus OP_Lamb histogram, then write
    the merged histogram, free energy and diagnostics.

    Returns (centers, merged_counts) or None when the merge did not run.
    """
    print("\n" + "=" * 60)
    print("CROSS-SIMULATION OP_Lamb MERGE")
    print("=" * 60)

    bin_info = all_bin_infos[("non-reactive", "plus")]
    if REF_COL not in bin_info:
        print(f"  {REF_COL} has no entry in COLUMN_RANGES — merge skipped.")
        return None
    centers, Minx, dx = bin_info[REF_COL]

    # ── This run's combined plus-ensemble histogram, from the intermediates ──
    h_local = None
    for key in [("non-reactive", "plus"), ("reactive", "plus")]:
        c, counts = load_and_sum_intermediates(key, REF_COL, kind="1d")
        if c is None:
            continue
        if h_local is None:
            h_local = counts.copy()
        elif np.allclose(centers, c):
            h_local += counts
        else:
            print(f"  WARNING: bin centers mismatch for {key} — not merged.")
    if h_local is None:
        print("  No local plus-ensemble intermediates found — merge skipped.")
        return None

    print(f"\n  Local plus-ensemble histogram: total weight = "
          f"{np.nansum(h_local):.6e}")

    h_other = np.zeros_like(h_local)
    parts   = []

    # ── The other run ────────────────────────────────────────────────────────
    ml_dir  = resolve_data_path(OTHER_SIM_ML_DIR,  must_be_dir=True)
    weights = resolve_data_path(OTHER_SIM_WEIGHTS, must_be_dir=False)

    if ml_dir is None or weights is None:
        print("\n  ERROR: could not locate the other run.")
        print(f"    OTHER_SIM_ML_DIR  = {OTHER_SIM_ML_DIR!r} -> "
              f"{ml_dir or 'NOT FOUND'}")
        print(f"    OTHER_SIM_WEIGHTS = {OTHER_SIM_WEIGHTS!r} -> "
              f"{weights or 'NOT FOUND'}")
        print(f"    (searched relative to {os.getcwd()!r} and up to three "
              f"parent levels)")
        return None

    sign = -1.0 if SYMMETRIC_OP else 1.0
    h_o, meta_o = build_op_histogram(
        ml_dir, weights, centers, Minx, dx, sign,
        label=f"other run ({'mirrored' if SYMMETRIC_OP else 'direct'})"
    )
    h_other += h_o
    parts.append(("other run", meta_o))

    # ── Optionally this run's own data, reversed ─────────────────────────────
    if SELF_MIRROR:
        if not SYMMETRIC_OP:
            print("\n  NOTE: SELF_MIRROR requires SYMMETRIC_OP — skipped.")
        else:
            own_ml      = resolve_data_path(INPUT_DIR, must_be_dir=True)
            own_weights = resolve_data_path(PATH_WEIGHTS_FILE, must_be_dir=False)
            if own_ml is None or own_weights is None:
                print("\n  WARNING: could not locate this run's own ML/ or "
                      "path_weights.txt for SELF_MIRROR — skipped.")
            else:
                h_s, meta_s = build_op_histogram(
                    own_ml, own_weights, centers, Minx, dx, -1.0,
                    label="this run (mirrored)"
                )
                h_other += h_s
                parts.append(("this run mirrored", meta_s))

    if np.nansum(h_other) <= 0:
        print("\n  Nothing was added (all contributions fell outside the bins). "
              "Check SYMMETRIC_OP and COLUMN_RANGES[REF_COL].")
        return None

    # ── Merge ────────────────────────────────────────────────────────────────
    merged, info = merge_op_histograms(centers, h_local, h_other)

    print(f"\n  Merge mode        : {info['mode']}")
    if info["mode"] == "correction":
        print(f"  Correction factor : {info['scale']:.6e}  applied to "
              f"the {CORRECTION_APPLY_TO} histogram")
        print("    = (N_A * (P_A / P_C_min) * (P_C / P_D_min)) / N_D")
        print(f"      N_A={N_A:g}  P_A={P_A:g}  P_C_min={P_C_MIN:g}  "
              f"P_C={P_C:g}  P_D_min={P_D_MIN:g}  N_D={N_D:g}")
        if np.isfinite(info["fitted_scale"]):
            print(f"  Data-implied factor: {info['fitted_scale']:.6e} "
                  f"(fitted on {info['n_overlap']} overlapping bins)")
            ratio = info["fitted_scale"] / info["scale"] if info["scale"] else np.nan
            print(f"    ratio fitted/correction = {ratio:.4g} "
                  f"({abs(np.log(ratio)):.3f} kT) — these should agree if the "
                  f"correction is consistent with the sampling")
    if info["mode"] == "match":
        print(f"  Scale factor      : {info['scale']:.6e} "
              f"(fitted on {info['n_overlap']} overlapping bins)")
    if info["mode"] in ("correction", "match"):
        print(f"  Shape residual    : {info['rms_kT']:.3f} kT rms")
        if np.isfinite(info["rms_kT"]) and info["rms_kT"] > 1.0:
            print("  WARNING: the two runs differ in SHAPE by more than 1 kT "
                  "rms, so no single scale factor reconciles them.")
            print("           Inspect the diagnostic plots before trusting the "
                  "merged free energy.")
    print(f"  Applied scale     : local x {info['mul_local']:.6e}, "
          f"other x {info['mul_other']:.6e}")

    n_local_only  = int(np.sum((h_local > 0) & (h_other <= 0)))
    n_other_only  = int(np.sum((h_other > 0) & (h_local <= 0)))
    n_both        = int(np.sum((h_local > 0) & (h_other > 0)))
    print(f"  Bin coverage      : {n_both} shared, {n_local_only} local-only, "
          f"{n_other_only} gained from the other run")

    # ── Output ───────────────────────────────────────────────────────────────
    out_dir = os.path.join(OUTPUT_DIR, f"histograms_{MERGE_TAG}")
    os.makedirs(out_dir, exist_ok=True)
    safe = REF_COL.replace("/", "_").replace("\\", "_")

    make_histogram_from_bins(
        centers, merged, REF_COL, "merged plus ensembles",
        os.path.join(out_dir, f"{safe}_merged.png"), "#4a1a6b"
    )
    plot_op_merge(
        centers, h_local, h_other, merged, info,
        os.path.join(out_dir, f"{safe}_merge_density.png"),
        os.path.join(out_dir, f"{safe}_merge_freeenergy.png"),
        local_label="this run", other_label="other run"
        + (" (mirrored)" if SYMMETRIC_OP else ""),
    )
    write_op_merge_txt(centers, h_local, h_other, merged,
                       os.path.join(out_dir, f"{safe}_merged.txt"))

    write_stats_csv({f"{REF_COL}_local":  column_stats_from_bins(centers, h_local),
                     f"{REF_COL}_other":  column_stats_from_bins(centers, h_other),
                     f"{REF_COL}_merged": column_stats_from_bins(centers, merged)},
                    os.path.join(OUTPUT_DIR, f"stats_{MERGE_TAG}.csv"))

    print(f"  Merged histograms -> {os.path.relpath(out_dir)}/")
    print("=" * 60)
    return centers, merged


# ── 11. MAIN ─────────────────────────────────────────────────────────────────

def histograms(
    # ── Input data ────────────────────────────────────────────────────────
    cv_dir: Annotated[str, typer.Option("-cv-dir", help="Folder of per-path CV .txt files.", rich_help_panel=panels.INPUT)] = INPUT_DIR,
    weights: Annotated[str, typer.Option("-weights", help="WHAM path-weights file listing which paths to include.", rich_help_panel=panels.INPUT)] = PATH_WEIGHTS_FILE,
    ranges: Annotated[str, typer.Option("-ranges", help="REQUIRED. TOML file giving the histogram binning per CV as [min, max, n_bins], in a [ranges] table plus an optional [ranges_nr_minus] override. Per-simulation, so it lives outside the code - see examples/column_ranges.toml.", rich_help_panel=panels.INPUT)] = ...,

    # ── Dataset construction ──────────────────────────────────────────────
    merge_other: Annotated[bool, typer.Option("-merge-other/-no-merge-other", help="Fold a second simulation's histograms into this one.", rich_help_panel=panels.DATASET)] = MERGE_OTHER_SIM,
    other_cv_dir: Annotated[str, typer.Option("-other-cv-dir", help="The other simulation's CV folder (used when -merge-other).", rich_help_panel=panels.DATASET)] = OTHER_SIM_ML_DIR,
    other_weights: Annotated[str, typer.Option("-other-weights", help="The other simulation's path-weights file.", rich_help_panel=panels.DATASET)] = OTHER_SIM_WEIGHTS,
    correction_apply_to: Annotated[str, typer.Option("-correction-apply-to", help="Which run the escape/entry correction factor multiplies: 'local' or 'other'.", rich_help_panel=panels.DATASET)] = CORRECTION_APPLY_TO,
    symmetric_op: Annotated[bool, typer.Option("-symmetric-op/-no-symmetric-op", help="Treat the OP as symmetric when merging the two runs.", rich_help_panel=panels.DATASET)] = SYMMETRIC_OP,

    # ── Model and training (parallelism) ──────────────────────────────────
    workers: Annotated[int, typer.Option("-workers", help="Worker processes for the parsing pass.", rich_help_panel=panels.MODEL)] = N_WORKERS,
    n_bootstrap: Annotated[int, typer.Option("-n-bootstrap", help="Path-level bootstrap resamples for confidence intervals.", rich_help_panel=panels.MODEL)] = N_BOOTSTRAP,
    skip_parsing: Annotated[bool, typer.Option("-skip-parsing", help="Reuse the intermediates from a previous run instead of re-reading the CV files.", rich_help_panel=panels.MODEL)] = SKIP_PARSING,

    # ── Output ────────────────────────────────────────────────────────────
    out_dir: Annotated[str, typer.Option("-out-dir", help="Directory for histograms, stats CSVs and intermediates.", rich_help_panel=panels.OUTPUT)] = OUTPUT_DIR,
):
    """Weighted CV histograms, statistics and 2D maps over a path ensemble.

    Bins every CV over all paths using their WHAM weights, split by
    reactive/non-reactive and plus/minus ensemble, and can fold in a second
    simulation (e.g. the entry run alongside the escape run) on a common OP
    axis.

    Configuration note: this analysis is driven by module-level constants that
    its ~37 helper functions read as globals - COLUMN_RANGES in particular is a
    large per-system binning table. Rather than rewrite all of them, the
    options above rebind the handful of globals that actually differed between
    the copies of this script found in the tree. Everything else stays a module
    constant, edited in cv_histograms.py.
    """
    global INPUT_DIR, PATH_WEIGHTS_FILE, OUTPUT_DIR, INTERMEDIATE_DIR, N_WORKERS
    global MERGE_OTHER_SIM, OTHER_SIM_ML_DIR, OTHER_SIM_WEIGHTS
    global CORRECTION_APPLY_TO, SYMMETRIC_OP, N_BOOTSTRAP, SKIP_PARSING

    INPUT_DIR = cv_dir
    PATH_WEIGHTS_FILE = weights
    OUTPUT_DIR = out_dir
    INTERMEDIATE_DIR = os.path.join(OUTPUT_DIR, "intermediates")
    N_WORKERS = workers
    MERGE_OTHER_SIM = merge_other
    OTHER_SIM_ML_DIR = other_cv_dir
    OTHER_SIM_WEIGHTS = other_weights
    CORRECTION_APPLY_TO = correction_apply_to
    SYMMETRIC_OP = symmetric_op
    N_BOOTSTRAP = n_bootstrap
    SKIP_PARSING = skip_parsing

    global COLUMN_RANGES, COLUMN_RANGES_NR_MINUS
    COLUMN_RANGES, COLUMN_RANGES_NR_MINUS = _load_column_ranges(ranges)
    print(f"Loaded {len(COLUMN_RANGES)} column ranges from {ranges!r}"
          f" (+{len(COLUMN_RANGES_NR_MINUS)} non-reactive/minus override(s))")
    if REF_COL not in COLUMN_RANGES:
        raise typer.BadParameter(
            f"-ranges: {ranges!r} has no entry for the reference column "
            f"{REF_COL!r}; every histogram is binned against it."
        )


    os.makedirs(OUTPUT_DIR,       exist_ok=True)
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

    print(f"\nLoading path weights from: {PATH_WEIGHTS_FILE!r}")
    ordered_paths      = load_path_weights(PATH_WEIGHTS_FILE)
    path_weight_lookup = {pn: w for pn, w in ordered_paths}

    if TESTING:
        ordered_paths = ordered_paths[:test_value]
        print(f"TESTING: limited to first {test_value} paths "
              f"({ordered_paths[0][0]} - {ordered_paths[-1][0]})")

    all_bin_infos = build_all_bin_infos()
    all_headers   = list(COLUMN_RANGES.keys())

    group_files           = {k: [] for k in ALL_GROUPS}
    not_found             = []
    unclassified          = []
    ensemble_path_weights = {k: {} for k in ALL_GROUPS}

    if SKIP_PARSING:
        print("\nSKIP_PARSING=True — loading intermediates directly.")
        existing = glob.glob(os.path.join(INTERMEDIATE_DIR, "*.npz"))
        if not existing:
            print("ERROR: No intermediates found. Set SKIP_PARSING=False.")
            raise SystemExit(1)
        print(f"  Found {len(existing)} intermediate files.")

    else:
        files = []
        for path_num, _ in ordered_paths:
            fp = os.path.join(INPUT_DIR, f"{path_num}{FILE_PATTERN}")
            if os.path.isfile(fp):
                files.append(fp)
            else:
                not_found.append(path_num)

        if not_found:
            print(f"  WARNING: {len(not_found)} path numbers from weights file "
                  f"have no corresponding file in {INPUT_DIR!r}:")
            print(f"    {not_found[:10]}{'...' if len(not_found) > 10 else ''}")

        print(f"\n{len(files)} files to process "
              f"(from {len(ordered_paths)} listed in weights file)")

        if not files:
            print("ERROR: No files to process.")
            raise SystemExit(1)

        print("Testing file parser on test file...")
        test = classify_file(files[1])
        print(f"  {os.path.basename(files[1])} -> "
              f"reactivity={test[1]}, ensemble={test[2]}, "
              f"headers={'ok' if test[3] else 'NONE'}, "
              f"rows={test[4].shape[0] if test[4] is not None else 'NONE'}")

        if test[3] and test[4] is not None and test[4].shape[0] > 0:
            print("\n  Data ranges in test file (vs configured COLUMN_RANGES):")
            print(f"  {'Column':25s}  {'Data min':>12}  {'Data max':>12}  "
                  f"{'Cfg min':>10}  {'Cfg max':>10}  {'In range?':>10}")
            print("  " + "-" * 85)
            for ci, col in enumerate(test[3]):
                if ci >= test[4].shape[1]:
                    continue
                vals = test[4][:, ci]
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    continue
                d_min, d_max = vals.min(), vals.max()
                if col in COLUMN_RANGES:
                    c_min, c_max, _ = COLUMN_RANGES[col]
                    in_range = "OK" if d_min >= c_min and d_max <= c_max else "MISMATCH"
                else:
                    c_min = c_max = float("nan")
                    in_range = "NOT IN CFG"
                print(f"  {col:25s}  {d_min:>12.4g}  {d_max:>12.4g}  "
                      f"{c_min:>10.4g}  {c_max:>10.4g}  {in_range:>10}")
            print()

        not_in_cfg = [col for col in test[3] if col not in COLUMN_RANGES]
        if not_in_cfg:
            print(f"\n  WARNING: {len(not_in_cfg)} columns in file have NO entry in "
                f"COLUMN_RANGES and will be silently skipped:")
            for col in not_in_cfg:
                print(f"    {col!r}")
            print("  Check for name mismatches between file headers and COLUMN_RANGES keys.\n")

        print(f"Parser test passed. Using {N_WORKERS} worker processes...")

        buffer_data       = {k: defaultdict(list) for k in ALL_GROUPS}
        buffer_weights    = {k: defaultdict(list) for k in ALL_GROUPS}
        files_accumulated = 0
        chunk_idx         = 0

        with Pool(processes=N_WORKERS) as pool:
            iterator = pool.imap_unordered(classify_file, files, chunksize=32)
            if HAS_TQDM:
                iterator = tqdm(iterator, total=len(files),
                                desc="Parsing files", unit="file")

            for result in iterator:
                try:
                    fp, reactivity, ensemble, headers, data = result
                except (BrokenPipeError, EOFError, OSError) as e:
                    print(f"  WARNING: IPC error: {e}, skipping.")
                    continue

                if reactivity is None:
                    unclassified.append(fp)
                    continue

                key = (reactivity, ensemble)
                if key not in group_files:
                    unclassified.append(fp)
                    continue

                path_num = int(os.path.splitext(os.path.basename(fp))[0])
                weight   = path_weight_lookup[path_num]

                group_files[key].append(fp)
                ensemble_path_weights[key][path_num] = weight

                group_bin_info = all_bin_infos[key]

                if data.shape[0] > 0 and headers:
                    for ci, col in enumerate(headers):
                        if ci < data.shape[1] and col in group_bin_info:
                            buffer_data[key][col].append(
                                (data[:, ci].copy(), weight)
                            )

                files_accumulated += 1

                if files_accumulated % FLUSH_EVERY == 0:
                    print(f"\n  Flushing chunk {chunk_idx} "
                          f"after {files_accumulated} files...")
                    flush_to_intermediate(buffer_data, buffer_weights,
                                          all_bin_infos, chunk_idx)
                    chunk_idx += 1

        has_data = any(
            len(v) > 0
            for d in buffer_data.values()
            for v in d.values()
        )
        if has_data:
            n_rem = files_accumulated % FLUSH_EVERY or FLUSH_EVERY
            print(f"\n  Flushing final chunk {chunk_idx} ({n_rem} files)...")
            flush_to_intermediate(buffer_data, buffer_weights,
                                  all_bin_infos, chunk_idx)

        print(f"\nColumns: {', '.join(all_headers)}")

        class_path = os.path.join(OUTPUT_DIR, "file_classification.txt")
        with open(class_path, "w") as out:
            for key in ALL_GROUPS:
                react, ens = key
                lst = group_files[key]
                out.write(f"\n{'='*60}\n")
                out.write(f"  {react.upper()} | {ens.upper()} ENSEMBLE"
                          f"  ({len(lst)} files)\n")
                out.write(f"{'='*60}\n")
                for f in lst:
                    out.write(f"  {os.path.basename(f)}\n")
            if unclassified:
                out.write(f"\n{'='*60}\n")
                out.write(f"  UNCLASSIFIED  ({len(unclassified)} files)\n")
                out.write(f"{'='*60}\n")
                for f in unclassified:
                    out.write(f"  {os.path.basename(f)}\n")
            if not_found:
                out.write(f"\n{'='*60}\n")
                out.write(f"  NOT FOUND ON DISK  ({len(not_found)} paths)\n")
                out.write(f"{'='*60}\n")
                for pn in not_found:
                    out.write(f"  {pn}\n")

        print(f"Classification written to: {class_path}")
        for key in ALL_GROUPS:
            r, e = key
            print(f"  {r:15s} | {e:5s} ensemble: "
                  f"{len(group_files[key]):5d} files")
        if unclassified:
            print(f"  {'unclassified':15s}            : "
                  f"{len(unclassified):5d} files")
        if not_found:
            print(f"  {'not found on disk':24s}: {len(not_found):5d} paths")

        validate_weights(ensemble_path_weights)

    # ── 11d. Sum intermediates → stats + 1-D + 2-D histograms ────────────
    print("\nGenerating stats and histograms from intermediates...")

    PLUS_GROUPS = [("non-reactive", "plus"), ("reactive", "plus")]

    for key in ALL_GROUPS:
        react, ens  = key
        tag         = f"{react.replace('-', '')}_{ens}"
        color       = HIST_COLOR[key]
        group_label = f"{react} | {ens} ensemble"

        group_headers = list(get_column_ranges(key).keys())

        pattern_1d = os.path.join(INTERMEDIATE_DIR, f"[0-9][0-9][0-9][0-9]__{tag}__1d__*.npz")
        if not glob.glob(pattern_1d):
            print(f"  [{group_label}] - no 1-D intermediates found, skipping.")
            continue

        stats_path     = os.path.join(OUTPUT_DIR, f"stats_{tag}.csv")
        col_stats_dict = {}
        for col in group_headers:
            centers, counts = load_and_sum_intermediates(key, col, kind="1d")
            if centers is not None:
                col_stats_dict[col] = column_stats_from_bins(centers, counts)
        write_stats_csv(col_stats_dict, stats_path)

        hist_dir_1d = os.path.join(OUTPUT_DIR, f"histograms_{tag}")
        os.makedirs(hist_dir_1d, exist_ok=True)

        hist_iter = group_headers
        if HAS_TQDM:
            hist_iter = tqdm(hist_iter,
                             desc=f"  1-D histograms [{group_label}]",
                             unit="col")

        for col in hist_iter:
            centers, counts = load_and_sum_intermediates(key, col, kind="1d")
            if centers is None:
                continue
            safe_name = col.replace("/", "_").replace("\\", "_")

            make_histogram_from_bins(
                centers, counts, col, group_label,
                os.path.join(hist_dir_1d, f"{safe_name}.png"),
                color, corrected=False
            )

            if col in SPHERICAL_ANGLE_COLS:
                corrected = apply_sine_correction(col, centers, counts)
                make_histogram_from_bins(
                    centers, corrected, col, group_label,
                    os.path.join(hist_dir_1d, f"{safe_name}_sinecorrected.png"),
                    color, corrected=True
                )
                make_cosine_transform_histogram(
                    centers, counts, col, group_label,
                    os.path.join(hist_dir_1d, f"{safe_name}_costransform.png"),
                    color
                )
        print(f"  [{group_label}] 1-D histograms -> {os.path.relpath(hist_dir_1d)}/")

        hist_dir_2d = os.path.join(OUTPUT_DIR, f"histograms2d_{tag}")
        os.makedirs(hist_dir_2d, exist_ok=True)

        cols_2d = [c for c in group_headers if c != REF_COL]
        iter_2d = cols_2d
        if HAS_TQDM:
            iter_2d = tqdm(cols_2d,
                           desc=f"  2-D histograms [{group_label}]",
                           unit="col")

        for col in iter_2d:
            cx, cy, counts_2d = load_and_sum_intermediates(key, col, kind="2d")
            if cx is None:
                continue
            safe_name = col.replace("/", "_").replace("\\", "_")

            make_2d_histogram(
                cx, cy, counts_2d,
                REF_COL, col, group_label,
                os.path.join(hist_dir_2d, f"{REF_COL}_vs_{safe_name}.png"),
            )

            if col in SPHERICAL_ANGLE_COLS:
                divisors = sine_correction_weights(col, cy)
                corr_2d  = counts_2d.astype(np.float64) / divisors[:, np.newaxis]
                make_2d_histogram(
                    cx, cy, corr_2d,
                    REF_COL, col, group_label,
                    os.path.join(hist_dir_2d,
                                 f"{REF_COL}_vs_{safe_name}_sinecorrected.png"),
                )
                make_cosine_transform_2d_histogram(
                    cx, cy, counts_2d,
                    REF_COL, col, group_label,
                    os.path.join(hist_dir_2d,
                                 f"{REF_COL}_vs_{safe_name}_costransform.png"),
                )
        print(f"  [{group_label}] 2-D histograms -> {os.path.relpath(hist_dir_2d)}/")

    # ── 11d-bis. Cross-simulation OP_Lamb merge ───────────────────────────────
    # Runs before the combined-plus section so the merged OP_Lamb counts can be
    # folded into the combined 2-D histograms below.
    merged_op = run_op_merge(all_bin_infos) if MERGE_OTHER_SIM else None

    # ── 11d-ter. Combined PLUS ensemble ───────────────────────────────────────
    print("\nGenerating combined plus-ensemble histograms...")

    combined_tag     = "combined_plus"
    combined_label   = "non-reactive + reactive | plus ensemble"
    combined_color   = "#4a1a6b"
    combined_headers = list(COLUMN_RANGES.keys())

    combined_dir_1d = os.path.join(OUTPUT_DIR, f"histograms_{combined_tag}")
    combined_dir_2d = os.path.join(OUTPUT_DIR, f"histograms2d_{combined_tag}")
    os.makedirs(combined_dir_1d, exist_ok=True)
    os.makedirs(combined_dir_2d, exist_ok=True)

    combined_stats_path = os.path.join(OUTPUT_DIR, f"stats_{combined_tag}.csv")
    combined_col_stats  = {}

    hist_iter = combined_headers
    if HAS_TQDM:
        hist_iter = tqdm(combined_headers,
                         desc=f"  1-D histograms [{combined_label}]",
                         unit="col")

    for col in hist_iter:
        combined_counts_1d = None
        combined_centers   = None
        for key in PLUS_GROUPS:
            centers, counts = load_and_sum_intermediates(key, col, kind="1d")
            if centers is None:
                continue
            if combined_counts_1d is None:
                combined_counts_1d = counts.copy()
                combined_centers   = centers
            else:
                if np.allclose(combined_centers, centers):
                    combined_counts_1d += counts
                else:
                    print(f"  WARNING: combined plus 1D centers mismatch for {col}")

        if combined_counts_1d is None:
            continue

        safe_name = col.replace("/", "_").replace("\\", "_")
        combined_col_stats[col] = column_stats_from_bins(combined_centers,
                                                          combined_counts_1d)

        make_histogram_from_bins(
            combined_centers, combined_counts_1d, col, combined_label,
            os.path.join(combined_dir_1d, f"{safe_name}.png"),
            combined_color, corrected=False
        )

        if col in SPHERICAL_ANGLE_COLS:
            corrected = apply_sine_correction(col, combined_centers,
                                              combined_counts_1d)
            make_histogram_from_bins(
                combined_centers, corrected, col, combined_label,
                os.path.join(combined_dir_1d, f"{safe_name}_sinecorrected.png"),
                combined_color, corrected=True
            )
            make_cosine_transform_histogram(
                combined_centers, combined_counts_1d, col, combined_label,
                os.path.join(combined_dir_1d, f"{safe_name}_costransform.png"),
                combined_color
            )

    write_stats_csv(combined_col_stats, combined_stats_path)
    print(f"  [{combined_label}] 1-D histograms -> {os.path.relpath(combined_dir_1d)}/")

    cols_2d_combined = [c for c in combined_headers if c != REF_COL]
    iter_2d = cols_2d_combined
    if HAS_TQDM:
        iter_2d = tqdm(cols_2d_combined,
                       desc=f"  2-D histograms [{combined_label}]",
                       unit="col")

    n_rescaled_cols = 0
    merge_gap_bins  = 0

    for col in iter_2d:
        combined_counts_2d = None
        combined_cx = combined_cy = None

        for key in PLUS_GROUPS:
            cx, cy, counts_2d = load_and_sum_intermediates(key, col, kind="2d")
            if cx is None:
                continue
            if combined_counts_2d is None:
                combined_counts_2d = counts_2d.copy()
                combined_cx, combined_cy = cx, cy
            else:
                if np.allclose(combined_cx, cx) and np.allclose(combined_cy, cy):
                    combined_counts_2d += counts_2d
                else:
                    print(f"  WARNING: combined plus 2D centers mismatch for {col}")

        if combined_counts_2d is None:
            continue

        # Impose the merged OP_Lamb marginal on the joint histogram, so every
        # plot below is drawn from the combined statistics of both runs.
        if merged_op is not None:
            combined_counts_2d, n_sc, n_gap = apply_merged_op_marginal(
                combined_counts_2d, combined_cx, merged_op[0], merged_op[1]
            )
            if n_sc:
                n_rescaled_cols += 1
                merge_gap_bins = max(merge_gap_bins, n_gap)

        safe_name = col.replace("/", "_").replace("\\", "_")

        make_2d_histogram(
            combined_cx, combined_cy, combined_counts_2d,
            REF_COL, col, combined_label,
            os.path.join(combined_dir_2d, f"{REF_COL}_vs_{safe_name}.png"),
        )

        make_2d_histogram_conditional(
            combined_cx, combined_cy, combined_counts_2d,
            REF_COL, col, combined_label,
            os.path.join(combined_dir_2d, f"{REF_COL}_vs_{safe_name}_normcol.png"), "x_given_y",
        )

        make_2d_histogram_conditional(
            combined_cx, combined_cy, combined_counts_2d,
            REF_COL, col, combined_label,
            os.path.join(combined_dir_2d, f"{REF_COL}_vs_{safe_name}_normrow.png"), "y_given_x",
        )

        make_2d_histogram_free_energy(
            combined_cx, combined_cy, combined_counts_2d,
            REF_COL, col, combined_label,
            os.path.join(combined_dir_2d, f"{REF_COL}_vs_{safe_name}_fes.png"),
        )

        if col in SPHERICAL_ANGLE_COLS:
            divisors = sine_correction_weights(col, combined_cy)
            corr_2d  = combined_counts_2d.astype(np.float64) / divisors[:, np.newaxis]
            make_2d_histogram(
                combined_cx, combined_cy, corr_2d,
                REF_COL, col, combined_label,
                os.path.join(combined_dir_2d,
                             f"{REF_COL}_vs_{safe_name}_sinecorrected.png"),
            )
            make_cosine_transform_2d_histogram(
                combined_cx, combined_cy, combined_counts_2d,
                REF_COL, col, combined_label,
                os.path.join(combined_dir_2d,
                             f"{REF_COL}_vs_{safe_name}_costransform.png"),
            )

    if merged_op is not None:
        print(f"  [{combined_label}] merged OP_Lamb marginal applied to "
              f"{n_rescaled_cols} of {len(cols_2d_combined)} 2-D histograms")
        if merge_gap_bins:
            print(f"    NOTE: {merge_gap_bins} OP_Lamb bins carry merged weight "
                  f"but were never sampled by this run, so they stay empty in "
                  f"the 2-D plots.")

    print(f"  [{combined_label}] 2-D histograms -> {os.path.relpath(combined_dir_2d)}/")

    # ── 11e. DOPC vs POPC preference analysis (with bootstrap statistics) ──────
    print(f"\nGenerating DOPC/POPC lipid preference plots "
          f"(bootstrap n={N_BOOTSTRAP}, α={ALPHA})...")
    for key in ALL_GROUPS:
        react, ens = key
        tag        = f"{react.replace('-', '')}_{ens}"
        pref_dir   = os.path.join(OUTPUT_DIR, f"preference_dopc_popc_{tag}")
        make_dopc_popc_preference_plot(
            key, pref_dir,
            n_bootstrap=N_BOOTSTRAP,
            alpha=ALPHA,
        )

    print("\nDone!")
