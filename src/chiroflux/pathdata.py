"""Reading an infretis simulation: paths, trajectories and their weights.

Everything here turns the on-disk output of a TIS/RETIS run - the ``.toml``
config, the ``infretis_data.txt`` path table and the per-path CV trajectory
files - into arrays the analyses can work on, together with the WHAM path
weights that unbias those arrays across the TIS ensembles.

Trajectory file layout (per path, e.g. ``ML/<path_nr>.txt``):
  line 1: "# reactive" / "# non-reactive"
  line 2: "# <ensemble info>"
  line 3: "# <duplicate/edit info>"
  line 4: column names (no leading "#")
  line 5+: data

``_compute_path_weights`` is pure arithmetic rather than I/O, but it is kept
here because it is never useful on its own: every caller reads the path table
and immediately weights it.

This module depends only on numpy, tomli and the standard library. It must
stay that way - it is what lets the analyses that need no machine learning
(statistics, PCA, DeepTDA data prep) avoid importing shap, scikit-learn and
lightgbm entirely.
"""

import warnings
from pathlib import Path

import numpy as np
import tomli

# Which paths an analysis should consider, by reactive/non-reactive outcome.
_PATHS_CHOICES = ("all", "reactive", "nonreactive")


def _check_overwrite(path, overw):
    """Refuse to clobber an existing output file unless -O was given.

    An empty `path` means "not written", and is always allowed.
    """
    if not overw and path and Path(path).exists():
        raise ValueError(f"Output file {path} already exists!")


def _load_path_table(data, nskip, M):
    """Load infretis_data.txt, filtered to paths that actually contribute.

    M is the number of TIS interfaces (len(interfaces) from the .toml file);
    the data file has M-1 ensemble columns for path_f and M-1 for path_w.

    Returns
    -------
    pnr    : (N_paths,) int path numbers
    maxop  : (N_paths,) float max order-parameter value reached
    path_f : (N_paths, M-1) fractional path-count contribution per ensemble
    path_w : (N_paths, M-1) associated weight per ensemble
    """
    raw = np.loadtxt(data, dtype=str)
    raw = raw[nskip:]

    # Column 3 is the minus-ensemble's path_f value, not a generic "no
    # contribution" marker - a row counts if it contributes to *any*
    # plus-ensemble column (the M-1 columns sliced into path_f below).
    non_zero = np.any(raw[:, 4 : 3 + M] != "----", axis=1)
    raw[raw == "----"] = "0.0"

    pnr = raw[non_zero, 0].astype(int)
    maxop = raw[non_zero, 2].astype(float)
    path_f = raw[non_zero, 4 : 3 + M].astype(float)
    path_w = raw[non_zero, 4 + M : 3 + 2 * M].astype(float)

    return pnr, maxop, path_f, path_w

def _compute_path_weights(maxop, path_f, path_w, interfaces):
    """WHAM per-path statistical weight (unbiased across TIS ensembles).

    Each path's raw ensemble weight is rescaled by Q_{K(lambda_max)}, where
    K(lambda) is the highest TIS interface index <= lambda.
    """
    w = np.where(path_w != 0, path_f / path_w, 0.0)
    col_sum = np.sum(w, axis=0)
    frac_sum = np.sum(path_f, axis=0)
    scale = np.where(col_sum != 0, frac_sum / col_sum, 0.0)
    w = w * scale
    wsum = np.sum(w, axis=0)

    N_ens = w.shape[1]
    ploc = np.ones(N_ens + 1)
    for i in range(1, N_ens + 1):
        crosses_i = maxop >= interfaces[i]
        num = np.sum(crosses_i[:, None] * w[:, :i])
        den = np.sum(wsum[:i] / ploc[:i])
        ploc[i] = num / den if den > 0 else 0.0

    cumden = np.cumsum(wsum / ploc[:N_ens])
    Q = 1.0 / cumden

    K_per_path = np.searchsorted(interfaces, maxop, side="right") - 1
    K_per_path = np.clip(K_per_path, 0, N_ens - 1)

    return Q[K_per_path] * np.sum(w, axis=1)

def _matches_exclude(name, exclude):
    """True if name equals or contains any of the exclude substrings."""
    return bool(exclude) and any(pattern in name for pattern in exclude)

def _discover_columns(cv_dir, pnr_expected, op_col, cv_cols, encoding, exclude=None):
    """Return (cv_names, op_col_index, cv_col_indices) from the first
    available file's column-name line (line 4, see module docstring).

    exclude : list of substrings; any CV whose name fully or partially
              matches one of them is dropped (only applied when cv_cols
              is None, i.e. no explicit CV list was requested).
    """
    header_line: str | None = None
    for pnr in pnr_expected:
        fpath = cv_dir / f"{pnr}.txt"
        if fpath.exists():
            with fpath.open(encoding=encoding) as f:
                lines = [f.readline() for _ in range(4)]
            line = lines[3].strip()
            if line:
                header_line = line
                break

    if header_line is None:
        raise FileNotFoundError(
            f"No trajectory .txt files found in {cv_dir} for the given path numbers."
        )

    all_cols = header_line.split()

    if op_col not in all_cols:
        raise ValueError(
            f"Order-parameter column '{op_col}' not found in header.\n"
            f"Available columns: {all_cols}"
        )
    op_idx = all_cols.index(op_col)

    if cv_cols is None:
        cv_cols = [
            c for c in all_cols
            if c != op_col and not _matches_exclude(c, exclude)
        ]
        if exclude and not cv_cols:
            raise ValueError(f"Excluding {exclude} leaves no CV columns to train on.")
    else:
        missing_cols = [c for c in cv_cols if c not in all_cols]
        if missing_cols:
            raise ValueError(
                f"Requested CV columns not found in header: {missing_cols}\n"
                f"Available: {all_cols}"
            )

    cv_idxs = [all_cols.index(c) for c in cv_cols]
    return cv_cols, op_idx, cv_idxs

def _load_trajectory(fpath, encoding):
    """Load a trajectory file (skip the 4 header lines).
    Returns float64 array of shape (T, N_cols)."""
    data = np.loadtxt(
        fpath, dtype=float, comments="#", skiprows=4, encoding=encoding
    )
    if data.ndim == 1:
        data = data[np.newaxis, :]  # single-frame trajectory edge case
    return data

def _first_crossing_idx(op_vals, threshold):
    """Index of the first frame where op_vals >= threshold, or None."""
    meets = op_vals >= threshold
    if not np.any(meets):
        return None
    return int(np.argmax(meets))  # argmax on bool gives first True index

def _read_path_header(fpath, encoding):
    """Read the reactive label (line 1) and ensemble type (line 2).

    "Minus ensemble" paths only sample below lambda_0 and never reach the
    actual TIS interfaces, so they must be excluded from CV-crossing/SHAP
    analysis; only "plus ensemble" paths are relevant there.
    """
    with fpath.open(encoding=encoding) as f:
        label_line = f.readline().strip().lstrip("#").strip().lower()
        ens_line = f.readline().strip().lstrip("#").strip().lower()
    if label_line not in ("reactive", "non-reactive"):
        raise ValueError(
            f"{fpath}: expected 'reactive' or 'non-reactive' on line 1, "
            f"got {label_line!r}"
        )
    if ens_line not in ("plus ensemble", "minus ensemble"):
        raise ValueError(
            f"{fpath}: expected 'plus ensemble' or 'minus ensemble' on "
            f"line 2, got {ens_line!r}"
        )
    return label_line == "reactive", ens_line == "plus ensemble"

def _extract_cv_crossings(
    cv_dir,
    pnr_expected,
    subgrid,
    op_col,
    cv_cols,
    encoding="utf-8",
    exclude=None,
):
    """Load per-trajectory .txt files and extract (first) crossing CV values.

    Parameters
    ----------
    cv_dir       : directory containing 1.txt, 2.txt, ... trajectory files.
    pnr_expected : 1-D int array of path numbers, in order.
    subgrid      : lambda values to check crossing for, shape (N_sub,).
                   Pass the TIS interfaces themselves to get the CV value at
                   each path's first crossing of each interface.
    op_col       : name of the order-parameter column.
    cv_cols      : CV column names to extract; None -> all except op_col.
    encoding     : text encoding of the .txt files.
    exclude      : list of substrings; CVs whose name fully or partially
                   matches one of them are dropped. Only applied when
                   cv_cols is None.

    Returns
    -------
    cv_array   : (N_paths, N_cvs, N_sub) float64, NaN where not reached.
    cv_names   : list of CV column names, length N_cvs.
    pnr_loaded : copy of pnr_expected (rows for missing files are all-NaN).
    """
    cv_dir = Path(cv_dir)
    N_paths = len(pnr_expected)
    N_sub = len(subgrid)

    cv_names, op_idx, cv_idxs = _discover_columns(
        cv_dir, pnr_expected, op_col, cv_cols, encoding, exclude=exclude
    )
    N_cvs = len(cv_names)

    cv_array = np.full((N_paths, N_cvs, N_sub), np.nan, dtype=np.float64)

    missing = 0
    for row_j, pnr in enumerate(pnr_expected):
        fpath = cv_dir / f"{pnr}.txt"
        if not fpath.exists():
            missing += 1
            continue

        try:
            frames = _load_trajectory(fpath, encoding)
        except Exception as exc:
            warnings.warn(f"Could not parse {fpath}: {exc}", stacklevel=2)
            continue

        max_col = max(op_idx, max(cv_idxs, default=-1))
        if frames.shape[1] <= max_col:
            warnings.warn(
                f"{fpath}: expected >= {max_col + 1} columns, "
                f"got {frames.shape[1]} - skipping.",
                stacklevel=2,
            )
            continue

        op_vals = frames[:, op_idx]

        for alpha in range(N_sub):
            lam_alpha = subgrid[alpha]
            first_idx = _first_crossing_idx(op_vals, lam_alpha)
            if first_idx is None:
                continue

            for k, ci in enumerate(cv_idxs):
                cv_array[row_j, k, alpha] = frames[first_idx, ci]

    if missing:
        warnings.warn(
            f"{missing}/{N_paths} trajectory files not found in {cv_dir}. "
            "Corresponding rows are all-NaN.",
            stacklevel=2,
        )

    return cv_array, cv_names, pnr_expected.copy()

def _extract_path_metadata(cv_dir, pnr_expected, encoding="utf-8"):
    """Read the reactive label and ensemble type for each path number.

    Returns
    -------
    labels  : (N_paths,) float array; 1.0 = reactive, 0.0 = non-reactive,
              NaN where the file is missing.
    is_plus : (N_paths,) bool array; True if the path is a "plus ensemble"
              path (the only ones that actually cross the TIS interfaces).
              False for "minus ensemble" paths and for missing files.
    """
    cv_dir = Path(cv_dir)
    labels = np.full(len(pnr_expected), np.nan)
    is_plus = np.zeros(len(pnr_expected), dtype=bool)

    missing = 0
    for j, pnr in enumerate(pnr_expected):
        fpath = cv_dir / f"{pnr}.txt"
        if not fpath.exists():
            missing += 1
            continue
        reactive, plus = _read_path_header(fpath, encoding)
        labels[j] = float(reactive)
        is_plus[j] = plus

    if missing:
        warnings.warn(
            f"{missing}/{len(pnr_expected)} trajectory files not found in "
            f"{cv_dir} for label/ensemble extraction.",
            stacklevel=2,
        )
    return labels, is_plus

def _scan_trajectory(fpath, max_col, encoding):
    """Cheaply count a trajectory's data rows and check its column width,
    without parsing any floats (used to size the output arrays up front)."""
    n_rows = 0
    first_n_cols = None
    with fpath.open(encoding=encoding) as f:
        for i, line in enumerate(f):
            if i < 4:
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if first_n_cols is None:
                first_n_cols = len(stripped.split())
            n_rows += 1
    if n_rows == 0 or first_n_cols is None or first_n_cols <= max_col:
        return None
    return n_rows

def _load_interfaces_from_toml(toml_path):
    """Read only the interface list from a simulation toml (no trajectory I/O)."""
    with open(toml_path, "rb") as f:
        cfg = tomli.load(f)
    return np.asarray(cfg["simulation"]["interfaces"], dtype=float)
