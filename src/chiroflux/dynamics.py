"""Time-resolved (dynamical) CV features in a window around each interface crossing.

Every other analysis in ChiroFlux reduces a path to a *single frame* per TIS
interface - the first crossing of lambda_i - and throws the rest of the time
series away. This module keeps a fixed-length window around that crossing and
asks what the CV was *doing*, not just what it was:

  var      fluctuation amplitude of the detrended window (local softness)
  tau      autocorrelation time of the detrended window (local friction /
           how fast the CV decorrelates), same estimator convention as
           ``membrane_spatial._position_autocorrelation_D``
  lagOP    lead/lag of the CV against the order parameter: the shift at which
           the CV window best correlates with the OP window. Positive means
           the CV *follows* the OP; negative means it *leads* it.
  rOP      the (signed) correlation at that best lag - how tightly coupled to
           the progress coordinate the CV is at all.

``-spectral`` adds the four frequency-domain descriptors used in the nanopore
literature (spectral energy, main frequency, spectral centroid, spectral
entropy), obtained by FFT of the same window - as in Zhang et al., *Angew.
Chem. Int. Ed.* 2025, doi:10.1002/anie.202515531, which classifies amino-acid
enantiomers from ionic-current traces. That setting is close to the ideal case
for these features (uniform acquisition, stationary blockade events, one
otherwise featureless channel); a TIS path is none of those things, and the CVs
here are already interpretable, so the same descriptors have much less to add.
They are provided mainly so the
redundancy question can be settled with data rather than argument: for an
overdamped CV, spectral energy *is* the variance (Parseval) and the centroid
and entropy are largely reparameterisations of tau. ``dynamics_redundancy.txt``
and ``feature_redundancy.png`` report the measured rank correlation between
every pair of feature kinds, so a spectral feature that turns out to be a noisy
restatement of ``var``/``tau`` shows up as such immediately.

Two preprocessing choices matter and are deliberate:

* **Detrending.** var/tau/spectra are computed on a linearly detrended,
  Hann-windowed segment. A path crossing an interface has a monotone drift by
  construction, and an undetrended segment puts all its power in the lowest
  bins, which swamps the centroid and inflates the variance.
* **No detrending for lead/lag.** ``lagOP``/``rOP`` use the mean-removed but
  *not* detrended window: there the ramp is the signal, and removing it would
  destroy exactly the ordering information being measured.

Window comparability is enforced rather than hoped for: unless ``-min-frames``
is raised, only paths that supply the *full* window contribute, so every sample
shares one window length and one frequency resolution. Spectral entropy in
particular is biased by segment length, so mixing lengths silently invents
differences between long and short paths.

Choosing ``-half-window`` is the one real tuning decision, and it is a trade:
``tau`` can only resolve timescales a few times shorter than the window, so a
window that is too short flattens exactly the differences the analysis is for,
while a longer window disqualifies every path that cannot supply it. On a
synthetic test with a planted 30-vs-12-frame timescale difference, an 81-frame
window put ``tau`` at rank 13 (Cohen's d = 0.27) and a 301-frame window put it
at rank 2 (d = 1.05), at the cost of ~20% of the paths. Start long, then
shorten until the path count stops being acceptable.

One thing to expect in the redundancy report: ``specE`` against ``var`` lands
high but not at 1.0 (~0.75 on the synthetic test). The two are identical in
expectation, but ``specE`` is measured through a Hann taper that weights the
middle of the window, so on a single short, strongly correlated segment it is
the *noisier* of the two estimates of the same quantity - which is the point.

Note on leakage: ``-window-mode centered`` includes frames *after* the crossing,
which are part of the path's future and therefore partly encode the
reactive/non-reactive label. That is fine when the features are read as a
description of the local environment, and not fine when they are read as a
prediction. Use ``-window-mode pre`` for the causal version.
"""

import datetime
import os
import warnings
from pathlib import Path
from typing import Annotated, Optional

import matplotlib
import numpy as np
import tomli
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import panels
from .pathdata import (
    _check_overwrite,
    _compute_path_weights,
    _discover_columns,
    _extract_path_metadata,
    _first_crossing_idx,
    _load_path_table,
    _load_trajectory,
)
from .plotting import (
    _fig_inches,
    _plot_importance_bar,
    _plot_interface_heatmap,
    _top_n_indices,
    _truncation_note,
)
from .statistical_analysis import (
    _weighted_cohens_d,
    _weighted_ks,
    _weighted_spearman,
)

#: Feature kinds always computed, in the order they appear in the feature axis.
CORE_KINDS = ("var", "tau", "lagOP", "rOP")

#: Frequency-domain kinds, appended when -spectral is given.
SPEC_KINDS = ("specE", "fmain", "speccent", "specent")

_KIND_HELP = {
    "var": "fluctuation variance (detrended window)",
    "tau": "autocorrelation time (detrended window)",
    "lagOP": "lead/lag vs the order parameter (+ = CV follows OP)",
    "rOP": "correlation with the OP at the best lag",
    "specE": "spectral energy (= variance, by Parseval)",
    "fmain": "frequency of the largest PSD bin",
    "speccent": "spectral centroid, sum(f * p(f))",
    "specent": "normalised spectral entropy of the PSD",
}


# ---------------------------------------------------------------------------
# Single-window signal processing
# ---------------------------------------------------------------------------

def _detrend(x):
    """Remove the mean and the best-fit linear trend from a 1-D window."""
    n = len(x)
    t = np.arange(n, dtype=float)
    t -= t.mean()
    xm = x - x.mean()
    denom = float(np.dot(t, t))
    if denom > 0.0:
        xm = xm - (float(np.dot(t, xm)) / denom) * t
    return xm


def _autocorr_time(x, dt):
    """Integrated autocorrelation time of a mean-zero window.

    The normalised ACF is obtained by FFT (Wiener-Khinchin) with the unbiased
    1/(N-k) correction, then integrated up to its first zero crossing - the
    same truncation ``membrane_spatial._position_autocorrelation_D`` uses, so
    the two numbers are directly comparable.
    """
    n = len(x)
    if n < 4:
        return np.nan
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f), n=2 * n)[:n]
    acf /= np.arange(n, 0, -1)
    if not np.isfinite(acf[0]) or acf[0] <= 0.0:
        return np.nan
    C = acf / acf[0]

    zero_cross = np.where(C <= 0.0)[0]
    cutoff = int(zero_cross[0]) if len(zero_cross) > 0 else n
    if cutoff < 2:
        # Decorrelates within one frame: below the sampling resolution.
        return 0.0
    # Trapezoid without a scipy dependency.
    seg = C[:cutoff]
    return float(dt * (seg.sum() - 0.5 * (seg[0] + seg[-1])))


def _lead_lag(cv_win, op_win, max_lag, dt):
    """Lag (in time units) at which `cv_win` best correlates with `op_win`.

    Both windows are used mean-removed but *not* detrended: the ramp across
    the crossing is the feature being aligned. Each candidate lag is scored by
    a true Pearson r over the overlapping segment only, so short overlaps at
    large lags are not flattered by a shrinking numerator.

    Sign convention follows ``cv[t + L] ~ op[t]``: L > 0 means the CV
    reproduces the OP's behaviour L frames later, i.e. the CV *follows*.
    """
    n = len(cv_win)
    max_lag = int(min(max_lag, n - 3))
    if max_lag < 1:
        return np.nan, np.nan

    best_r = np.nan
    best_lag = np.nan
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = cv_win[lag:], op_win[: n - lag]
        else:
            a, b = cv_win[: n + lag], op_win[-lag:]
        if len(a) < 3:
            continue
        a = a - a.mean()
        b = b - b.mean()
        na = float(np.dot(a, a))
        nb = float(np.dot(b, b))
        if na <= 0.0 or nb <= 0.0:
            continue
        r = float(np.dot(a, b)) / np.sqrt(na * nb)
        if not np.isfinite(best_r) or abs(r) > abs(best_r):
            best_r = r
            best_lag = lag * dt
    return best_lag, best_r


def _spectral_features(x, dt):
    """(energy, main frequency, centroid, entropy) of a detrended window.

    The one-sided periodogram of the Hann-windowed segment is folded back to
    two-sided power so that ``energy`` is an actual variance estimate rather
    than an arbitrary multiple of one - which is what makes its redundancy
    with ``var`` legible instead of merely proportional.

    The DC bin is dropped before the shape descriptors: on a detrended,
    windowed segment it carries only spectral leakage, and leaving it in makes
    the centroid a function of the leakage rather than of the dynamics.
    """
    n = len(x)
    if n < 8:
        return np.nan, np.nan, np.nan, np.nan

    w = np.hanning(n)
    X = np.fft.rfft(x * w)
    P = np.abs(X) ** 2
    # Fold the negative frequencies back in (Nyquist bin is not duplicated).
    if n % 2 == 0:
        P[1:-1] *= 2.0
    else:
        P[1:] *= 2.0

    wss = float(np.sum(w**2))
    energy = float(P.sum() / (n * wss)) if wss > 0 else np.nan

    freqs = np.fft.rfftfreq(n, d=dt)[1:]
    Pp = P[1:]
    total = float(Pp.sum())
    if total <= 0.0 or len(Pp) < 2:
        return energy, np.nan, np.nan, np.nan

    p = Pp / total
    fmain = float(freqs[int(np.argmax(Pp))])
    centroid = float(np.dot(freqs, p))
    nz = p[p > 0.0]
    entropy = float(-np.sum(nz * np.log(nz)) / np.log(len(p)))
    return energy, fmain, centroid, entropy


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------

def _window_bounds(crossing, n_frames, width, mode, min_frames):
    """Half-open [lo, hi) index range of the window around `crossing`.

    Returns None when the path cannot supply an acceptable window. With
    `min_frames` at 0 the full `width` is required, which is what keeps every
    sample on one window length and one frequency grid.
    """
    if mode == "centered":
        half = width // 2
        lo, hi = crossing - half, crossing + half + 1
    else:  # "pre": the window ends on the crossing frame
        lo, hi = crossing - width + 1, crossing + 1

    if min_frames <= 0:
        return (lo, hi) if (lo >= 0 and hi <= n_frames) else None

    lo, hi = max(0, lo), min(n_frames, hi)
    return (lo, hi) if (hi - lo) >= min_frames else None


def _feature_names(cv_names, kinds):
    """Flat feature names, CV-major so a CV's kinds stay adjacent."""
    return [f"{cv}_{kind}" for cv in cv_names for kind in kinds]


def _apply_angle_series(col, transform):
    """cos / cos2 of a degree-valued time series (avoids wrap-around in var)."""
    if transform == "cos":
        return np.cos(np.deg2rad(col))
    return np.cos(np.deg2rad(col)) ** 2


def _extract_dynamical_features(
    cv_dir,
    pnr_expected,
    interfaces,
    op_col,
    cv_cols,
    kinds,
    half_window,
    window_mode,
    min_frames,
    max_lag,
    dt,
    angle_cols=None,
    sym_angle_cols=None,
    encoding="utf-8",
    exclude=None,
):
    """Compute per-path, per-CV, per-interface window features in one pass.

    Windows are never materialised across paths: at ~1e4 paths x ~30 CVs x
    ~20 interfaces x ~100 frames a stored window array runs to terabytes, so
    each window is reduced to its features as soon as it is read.

    Returns
    -------
    feats      : (N_paths, N_cvs * len(kinds), N_interfaces) float64, NaN where
                 the path supplied no acceptable window.
    feat_names : flat feature names matching axis 1.
    cv_names   : the CV names behind them (with angle renames applied).
    n_windows  : (N_interfaces,) count of paths that contributed a window.
    """
    cv_dir = Path(cv_dir)
    width = 2 * half_window + 1
    n_paths = len(pnr_expected)
    n_int = len(interfaces)

    cv_names, op_idx, cv_idxs = _discover_columns(
        cv_dir, pnr_expected, op_col, cv_cols, encoding, exclude=exclude
    )

    # Angle handling has to happen on the series, before any variance is taken.
    transforms = {}
    display_names = list(cv_names)
    for col in angle_cols or []:
        if col in cv_names:
            transforms[cv_names.index(col)] = "cos"
            display_names[cv_names.index(col)] = f"cos({col})"
            print(f"  Transformed '{col}' → cos({col})  [asymmetric angle]")
    for col in sym_angle_cols or []:
        if col in cv_names:
            transforms[cv_names.index(col)] = "cos2"
            display_names[cv_names.index(col)] = f"cos2({col})"
            print(f"  Transformed '{col}' → cos²({col})  [symmetric angle]")

    n_cvs = len(cv_names)
    n_kinds = len(kinds)
    want_spec = any(k in SPEC_KINDS for k in kinds)
    feats = np.full((n_paths, n_cvs * n_kinds, n_int), np.nan, dtype=np.float64)
    n_windows = np.zeros(n_int, dtype=int)

    missing = 0
    for row_j, pnr in enumerate(pnr_expected):
        fpath = cv_dir / f"{pnr}.txt"
        if not fpath.exists():
            missing += 1
            continue
        try:
            frames = _load_trajectory(fpath, encoding)
        except Exception as exc:
            print(f"  [warn] could not parse {fpath}: {exc}")
            continue

        max_col = max(op_idx, max(cv_idxs, default=-1))
        if frames.shape[1] <= max_col:
            continue

        op_vals = frames[:, op_idx]
        n_frames = len(op_vals)

        for alpha in range(n_int):
            crossing = _first_crossing_idx(op_vals, interfaces[alpha])
            if crossing is None:
                continue
            bounds = _window_bounds(crossing, n_frames, width, window_mode, min_frames)
            if bounds is None:
                continue
            lo, hi = bounds

            op_win = op_vals[lo:hi]
            if not np.all(np.isfinite(op_win)):
                continue
            n_windows[alpha] += 1

            for k, ci in enumerate(cv_idxs):
                raw = frames[lo:hi, ci]
                if not np.all(np.isfinite(raw)):
                    continue
                if k in transforms:
                    raw = _apply_angle_series(raw, transforms[k])

                base = k * n_kinds
                detr = _detrend(raw)

                values = {}
                values["var"] = float(np.dot(detr, detr) / len(detr))
                values["tau"] = _autocorr_time(detr, dt)
                lag, rlag = _lead_lag(raw, op_win, max_lag, dt)
                values["lagOP"] = lag
                values["rOP"] = rlag
                if want_spec:
                    e, fm, cent, ent = _spectral_features(detr, dt)
                    values["specE"] = e
                    values["fmain"] = fm
                    values["speccent"] = cent
                    values["specent"] = ent

                for ki, kind in enumerate(kinds):
                    feats[row_j, base + ki, alpha] = values[kind]

    if missing:
        print(f"  [warn] {missing}/{n_paths} trajectory files not found in {cv_dir}.")

    return feats, _feature_names(display_names, kinds), display_names, n_windows


# ---------------------------------------------------------------------------
# Effect sizes and redundancy
# ---------------------------------------------------------------------------

def _interface_metrics(X, y, weights, feature_names, top_n_print=20):
    """Weighted Cohen's d / Spearman / KS per feature, printing the top rows.

    Same estimators as ``chiroflux statistics`` - imported rather than
    reimplemented - but the printed table is truncated, because a dynamical
    run has N_cvs x N_kinds features and printing all of them per interface
    buries the result.
    """
    F = len(feature_names)
    cohens_d = np.full(F, np.nan)
    spearman = np.full(F, np.nan)
    ks_dist = np.full(F, np.nan)

    for k in range(F):
        col = X[:, k]
        finite = np.isfinite(col) & np.isfinite(y) & np.isfinite(weights)
        if np.sum(finite) < 4:
            continue
        xf, yf, wf = col[finite], y[finite], weights[finite]
        if np.sum(yf == 1) == 0 or np.sum(yf == 0) == 0:
            continue
        cohens_d[k] = _weighted_cohens_d(xf, yf, wf)
        spearman[k] = _weighted_spearman(xf, yf, wf)
        ks_dist[k] = _weighted_ks(xf, yf, wf)

    order = np.argsort(np.abs(np.nan_to_num(cohens_d)))[::-1]
    shown = min(top_n_print, F)
    print(f"  {'rank':>4}  {'feature':<32}  {'Cohen d':>9}  {'Spearman r':>10}  {'KS':>7}")
    for rank, idx in enumerate(order[:shown], 1):
        print(
            f"  {rank:>4}  {feature_names[idx]:<32}  "
            f"{cohens_d[idx]:9.4f}  {spearman[idx]:10.4f}  {ks_dist[idx]:7.4f}"
        )
    if F > shown:
        print(f"  ... {F - shown} further features in the output file.")

    return {"cohens_d": cohens_d, "spearman": spearman, "ks": ks_dist}


def _spearman_pair(a, b):
    """Unweighted Spearman between two vectors, on their common finite rows."""
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 8:
        return np.nan
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    da = float(np.dot(ra, ra))
    db = float(np.dot(rb, rb))
    if da <= 0.0 or db <= 0.0:
        return np.nan
    return float(np.dot(ra, rb) / np.sqrt(da * db))


def _kind_redundancy(feats, kinds, n_cvs):
    """Mean |Spearman| between every pair of feature kinds, averaged over CVs.

    This is the redundancy check the module exists to make cheap: if
    ``specE`` is variance under another name it lands at ~1.0 against ``var``,
    and a spectral feature that is genuinely independent does not.

    Paths and interfaces are pooled per CV, since the question is whether two
    kinds measure the same thing at all, not whether they do so at one lambda.
    """
    n_kinds = len(kinds)
    acc = np.full((n_kinds, n_kinds, n_cvs), np.nan)
    for cv in range(n_cvs):
        base = cv * n_kinds
        cols = [feats[:, base + ki, :].reshape(-1) for ki in range(n_kinds)]
        for i in range(n_kinds):
            for j in range(i, n_kinds):
                r = _spearman_pair(cols[i], cols[j])
                acc[i, j, cv] = r
                acc[j, i, cv] = r
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.abs(acc), axis=2)


def _weighted_class_mean(values, labels, weights, cls):
    """Weighted mean of `values` over the finite rows of one class."""
    m = np.isfinite(values) & np.isfinite(weights) & (labels == cls)
    if not np.any(m) or weights[m].sum() <= 0:
        return np.nan
    return float(np.average(values[m], weights=weights[m]))


# ---------------------------------------------------------------------------
# Pairwise lead/lag (second pass)
# ---------------------------------------------------------------------------

def _pairwise_lag_matrices(
    cv_dir, pnr_expected, interfaces, op_col, sel_cv_cols, labels, weights,
    half_window, window_mode, min_frames, max_lag, dt,
    angle_cols=None, sym_angle_cols=None, encoding="utf-8",
):
    """Class-resolved pairwise lead/lag between a small set of CVs.

    Restricted to `sel_cv_cols` on purpose: the pair count is quadratic, and
    at full CV count this dominates the whole run. The reactive minus
    non-reactive difference is what carries mechanism - whether the *ordering*
    of events, not their magnitude, differs by outcome.

    Returns (diff, names) with diff of shape (N_interfaces, K, K); entry
    [i, a, b] is the weighted mean lag of CV a behind CV b at interface i,
    reactive minus non-reactive.
    """
    cv_dir = Path(cv_dir)
    width = 2 * half_window + 1
    n_int = len(interfaces)

    cv_names, op_idx, cv_idxs = _discover_columns(
        cv_dir, pnr_expected, op_col, list(sel_cv_cols), encoding, exclude=None
    )
    transforms = {}
    display = list(cv_names)
    for col in angle_cols or []:
        if col in cv_names:
            transforms[cv_names.index(col)] = "cos"
            display[cv_names.index(col)] = f"cos({col})"
    for col in sym_angle_cols or []:
        if col in cv_names:
            transforms[cv_names.index(col)] = "cos2"
            display[cv_names.index(col)] = f"cos2({col})"

    K = len(cv_idxs)
    # Per class: weighted sums and weights, per interface and pair.
    wsum = np.zeros((2, n_int, K, K))
    wtot = np.zeros((2, n_int, K, K))

    for row_j, pnr in enumerate(pnr_expected):
        lab = labels[row_j]
        w = weights[row_j]
        if not np.isfinite(lab) or not np.isfinite(w) or w <= 0:
            continue
        cls = int(lab)
        fpath = cv_dir / f"{pnr}.txt"
        if not fpath.exists():
            continue
        try:
            frames = _load_trajectory(fpath, encoding)
        except Exception:
            continue
        if frames.shape[1] <= max(op_idx, max(cv_idxs, default=-1)):
            continue

        op_vals = frames[:, op_idx]
        for alpha in range(n_int):
            crossing = _first_crossing_idx(op_vals, interfaces[alpha])
            if crossing is None:
                continue
            bounds = _window_bounds(crossing, len(op_vals), width, window_mode, min_frames)
            if bounds is None:
                continue
            lo, hi = bounds

            wins = []
            for k, ci in enumerate(cv_idxs):
                raw = frames[lo:hi, ci]
                if not np.all(np.isfinite(raw)):
                    wins.append(None)
                    continue
                wins.append(_apply_angle_series(raw, transforms[k]) if k in transforms else raw)

            for a in range(K):
                if wins[a] is None:
                    continue
                for b in range(a + 1, K):
                    if wins[b] is None:
                        continue
                    lag, _ = _lead_lag(wins[a], wins[b], max_lag, dt)
                    if not np.isfinite(lag):
                        continue
                    wsum[cls, alpha, a, b] += w * lag
                    wtot[cls, alpha, a, b] += w
                    wsum[cls, alpha, b, a] += w * (-lag)
                    wtot[cls, alpha, b, a] += w

    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(wtot > 0, wsum / np.where(wtot > 0, wtot, 1.0), np.nan)
    return means[1] - means[0], display


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_redundancy(matrix, kinds, out_path, overw=False):
    """Heatmap of mean |Spearman| between feature kinds."""
    _check_overwrite(out_path, overw)
    n = len(kinds)
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * n, 0.9 + 0.65 * n))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma")
    plt.colorbar(im, ax=ax, label="mean |Spearman ρ| across CVs")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(kinds, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(kinds, fontsize=8)
    for i in range(n):
        for j in range(n):
            if np.isfinite(matrix[i, j]):
                ax.text(
                    j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if matrix[i, j] < 0.65 else "black",
                )
    ax.set_title("Feature-kind redundancy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Redundancy heatmap saved to {out_path}")


def _plot_lag_profile(lag_diff, cv_names, interfaces, out_path, unit,
                      overw=False, top_n=None):
    """Reactive minus non-reactive mean lead/lag against the OP, per interface.

    A CV whose column is systematically negative reaches its transition earlier
    in reactive paths than in unsuccessful ones - an ordering statement no
    single-frame feature can make.

    `top_n` keeps the CVs with the largest |Δ lag| at any interface. Besides
    keeping the figure inside matplotlib's size limit, it stops the colour
    scale - which is symmetric about the largest |Δ| in the whole array - from
    being set by one extreme CV and washing out everything else.
    """
    _check_overwrite(out_path, overw)
    finite_rows = np.any(np.isfinite(lag_diff), axis=1)
    if not np.any(finite_rows):
        return

    with warnings.catch_warnings():
        # An all-NaN column is normal: that CV reached no usable window.
        warnings.simplefilter("ignore", RuntimeWarning)
        per_cv = np.nanmax(np.abs(lag_diff), axis=0)
    keep = _top_n_indices(per_cv, top_n)
    shown_names = [cv_names[i] for i in keep]
    lag_diff = lag_diff[:, keep]

    vmax = np.nanmax(np.abs(lag_diff))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig, ax = plt.subplots(
        figsize=(_fig_inches(0.9, len(shown_names), 6),
                 _fig_inches(0.5, len(interfaces), 4))
    )
    im = ax.imshow(lag_diff, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label=f"Δ lag vs OP, reactive − non-reactive [{unit}]")
    ax.set_xticks(np.arange(len(shown_names)))
    ax.set_xticklabels(shown_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(interfaces)))
    ax.set_yticklabels([f"λ={v:.4f}" for v in interfaces], fontsize=8)
    ax.set_xlabel("Collective Variable")
    ax.set_ylabel("Interface")
    ax.set_title("Event ordering: does the CV move earlier in reactive paths?"
                 + _truncation_note(len(shown_names), len(cv_names)))
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Lead/lag profile saved to {out_path}")


def _plot_pair_lag(diff_i, names, out_path, unit, lam, overw=False):
    """Pairwise lead/lag difference matrix at one interface."""
    _check_overwrite(out_path, overw)
    if not np.any(np.isfinite(diff_i)):
        return
    vmax = np.nanmax(np.abs(diff_i))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.5 + 0.8 * n, 1.2 + 0.7 * n))
    im = ax.imshow(diff_i, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label=f"Δ lag of row behind column [{unit}]")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title(f"Pairwise event ordering, reactive − non-reactive  (λ={lam:.4f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Pairwise lag matrix saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def dynamics(
    toml: Annotated[str, typer.Option("-toml", help="The .toml file", rich_help_panel=panels.INPUT)] = "infretis.toml",
    data: Annotated[str, typer.Option("-data", help="The infretis_data.txt file", rich_help_panel=panels.INPUT)] = "infretis_data.txt",
    cv_dir: Annotated[str, typer.Option("-cv-dir", help="Path data folder with CV values in .txt files", rich_help_panel=panels.INPUT)] = "ML",
    op_col: Annotated[str, typer.Option("-op-col", help="Order-parameter column name", rich_help_panel=panels.INPUT)] = "OP_Lamb",
    nskip: Annotated[int, typer.Option("-nskip", help="Skip the first nskip rows of infretis_data.txt", rich_help_panel=panels.DATASET)] = 1000,
    half_window: Annotated[int, typer.Option("-half-window", help="Frames on each side of the crossing; the window is 2N+1 frames. Must be several times the tau you want to resolve, but longer windows disqualify more paths", rich_help_panel=panels.DATASET)] = 50,
    window_mode: Annotated[str, typer.Option("-window-mode", help="'centered' on the crossing, or 'pre' for frames up to it only (no outcome leakage)", rich_help_panel=panels.DATASET)] = "centered",
    min_frames: Annotated[int, typer.Option("-min-frames", help="Accept clipped windows down to this length; 0 (default) requires the full window so every sample shares one length", rich_help_panel=panels.DATASET)] = 0,
    min_class: Annotated[int, typer.Option("-min-class", help="Minimum number of paths in each class for an interface to be analysed", rich_help_panel=panels.DATASET)] = 100,
    dt: Annotated[float, typer.Option("-dt", help="Time between stored frames. Default 1.0 = report tau in frames and frequencies in 1/frame", rich_help_panel=panels.DATASET)] = 1.0,
    time_unit: Annotated[str, typer.Option("-time-unit", help="Name of the -dt unit, used in axis labels only", rich_help_panel=panels.DATASET)] = "frames",
    max_lag: Annotated[int, typer.Option("-max-lag", help="Largest lead/lag searched, in frames", rich_help_panel=panels.DATASET)] = 25,
    cv_cols: Annotated[Optional[str], typer.Option("-cv-cols", help="Comma-separated CV columns to use; default = all except -op-col", rich_help_panel=panels.SELECT)] = None,
    exclude: Annotated[Optional[str], typer.Option("-exclude", help="Comma-separated substrings; CVs whose name matches one are dropped (only when -cv-cols is unset)", rich_help_panel=panels.SELECT)] = None,
    angle_cols: Annotated[Optional[str], typer.Option("-angle-cols", help="Comma-separated CV columns in degrees to convert to cos(θ) (asymmetric molecules)", rich_help_panel=panels.REPR)] = None,
    sym_angle_cols: Annotated[Optional[str], typer.Option("-sym-angle-cols", help="Comma-separated CV columns in degrees to convert to cos²(θ) (symmetric molecules)", rich_help_panel=panels.REPR)] = None,
    spectral: Annotated[bool, typer.Option("-spectral", help="Also compute the FFT features (spectral energy, main frequency, centroid, entropy)", rich_help_panel=panels.MODEL)] = False,
    pair_lags: Annotated[int, typer.Option("-pair-lags", help="Pairwise lead/lag matrices for the top-K CVs (second pass over the files); 0 disables", rich_help_panel=panels.MODEL)] = 8,
    plot_top: Annotated[int, typer.Option("-plot-top", help="Draw only this many features per figure, chosen by effect size; 0 draws all. A dynamics run has N_cvs x N_kinds features, and an uncapped figure exceeds matplotlib's size limit", rich_help_panel=panels.OUTPUT)] = 40,
    plot_dir: Annotated[str, typer.Option("-plot-dir", help="Root directory for output plots", rich_help_panel=panels.OUTPUT)] = "dynamics_plots",
    out: Annotated[str, typer.Option("-out", help="Output file for the feature rankings", rich_help_panel=panels.OUTPUT)] = "dynamics_ranking.txt",
    redundancy_file: Annotated[str, typer.Option("-redundancy-file", help="Output file for the feature-kind redundancy table", rich_help_panel=panels.OUTPUT)] = "dynamics_redundancy.txt",
    overw: Annotated[bool, typer.Option("-O", help="Force overwriting of existing files", rich_help_panel=panels.OUTPUT)] = False,
):
    """Per-interface CV *dynamics* — fluctuation amplitude, timescale and event ordering.

    Where 'statistics' and 'shap-ml' read one frame per interface crossing,
    this reads a fixed-length window around it and asks what the CV was doing:

      <CV>_var     fluctuation amplitude (local softness)
      <CV>_tau     autocorrelation time (local friction)
      <CV>_lagOP   lead/lag against the order parameter (+ = CV follows)
      <CV>_rOP     correlation with the OP at that lag

    With -spectral, the four FFT descriptors used in nanopore work are added
    (<CV>_specE, _fmain, _speccent, _specent). They are mostly there to be
    tested rather than trusted: feature_redundancy.png and the redundancy
    table measure how much they restate var and tau, which for an overdamped
    CV is nearly all of it (spectral energy is the variance, by Parseval).

    Features are ranked per interface by weighted |Cohen's d| using the same
    estimators as 'statistics'. Output (written to -plot-dir):
      interface_NNN_effect_bar.png  — |Cohen's d| over the dynamical features
      interface_NNN_pairlag.png     — pairwise event ordering (with -pair-lags)
      lag_profile.png               — Δ lead/lag vs OP, reactive − non-reactive
      feature_redundancy.png        — mean |ρ| between feature kinds
      interface_heatmap.png         — |Cohen's d| across all interfaces

    The first three draw only the -plot-top strongest features (40 by default,
    0 for all) and say so in the title. This is not only legibility: these
    figures size themselves per feature, and a run with N_cvs x N_kinds of them
    asks for a canvas past matplotlib's limit, which used to fail at save time
    after the whole analysis had run. Every feature is always in -out.

    Leave -dt at 1.0 and tau is in frames and frequencies in 1/frame; pass the
    real frame spacing to get physical units. Note that -window-mode centered
    includes post-crossing frames, which partly encode the outcome — use
    'pre' when the features are meant to be predictive rather than descriptive.
    """
    if window_mode not in ("centered", "pre"):
        raise ValueError(f"-window-mode must be 'centered' or 'pre', got {window_mode!r}")
    if half_window < 4:
        raise ValueError("-half-window must be at least 4 for the window features to mean anything.")

    kinds = CORE_KINDS + SPEC_KINDS if spectral else CORE_KINDS

    with open(toml, "rb") as f:
        cfg = tomli.load(f)
    interfaces = np.asarray(cfg["simulation"]["interfaces"], dtype=float)
    M = len(interfaces)

    pnr, maxop, path_f, path_w = _load_path_table(data, nskip, M)
    path_weights = _compute_path_weights(maxop, path_f, path_w, interfaces)

    ranking_out = os.path.join(plot_dir, out)
    redundancy_out = os.path.join(plot_dir, redundancy_file)
    _check_overwrite(ranking_out, overw)
    _check_overwrite(redundancy_out, overw)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    width = 2 * half_window + 1
    print(
        f"Window: {width} frames, mode '{window_mode}', dt = {dt} {time_unit}"
        f"{' (full window required)' if min_frames <= 0 else f' (clipped to >= {min_frames} allowed)'}"
    )
    if window_mode == "centered":
        print(
            "  NOTE: centered windows include post-crossing frames, which are part of\n"
            "        the path's future and so partly encode the reactive label. Use\n"
            "        -window-mode pre for a leakage-free, predictive reading."
        )
    if min_frames > 0:
        print(
            "  NOTE: variable window lengths make spectral entropy and frequency\n"
            "        resolution length-dependent; prefer -min-frames 0 where possible."
        )

    feats, feat_names, cv_names, n_windows = _extract_dynamical_features(
        cv_dir=cv_dir,
        pnr_expected=pnr,
        interfaces=interfaces,
        op_col=op_col,
        cv_cols=cv_cols.split(",") if cv_cols else None,
        kinds=kinds,
        half_window=half_window,
        window_mode=window_mode,
        min_frames=min_frames,
        max_lag=max_lag,
        dt=dt,
        angle_cols=angle_cols.split(",") if angle_cols else None,
        sym_angle_cols=sym_angle_cols.split(",") if sym_angle_cols else None,
        exclude=exclude.split(",") if exclude else None,
    )

    labels, is_plus = _extract_path_metadata(cv_dir, pnr)
    feats = feats[is_plus]
    labels = labels[is_plus]
    path_weights = path_weights[is_plus]

    n_kinds = len(kinds)
    n_cvs = len(cv_names)
    print(
        f"{int(np.sum(is_plus))} plus-ensemble paths  |  {n_cvs} CVs x "
        f"{n_kinds} kinds = {len(feat_names)} features  |  {M} interfaces."
    )

    # --- per-interface effect sizes -------------------------------------------
    results = []
    lag_diff = np.full((M, n_cvs), np.nan)
    lag_kind_idx = kinds.index("lagOP")

    for i in range(M):
        start = datetime.datetime.now()
        X_i = feats[:, :, i]

        usable = np.any(np.isfinite(X_i), axis=1) & np.isfinite(labels)
        n_pos = int(np.sum(labels[usable] == 1))
        n_neg = int(np.sum(labels[usable] == 0))
        print(
            f"\n[interface {i + 1:3d}/{M}] lambda = {interfaces[i]:.4f}  "
            f"({n_pos} reactive, {n_neg} non-reactive, {n_windows[i]} full windows)"
        )

        if min(n_pos, n_neg) < min_class:
            print(f"  SKIP: fewer than {min_class} paths in one class with a usable window.")
            results.append({"interface": i, "lambda": interfaces[i], "ranking": None})
            continue

        metrics = _interface_metrics(X_i, labels, path_weights, feat_names)

        for cv in range(n_cvs):
            col = X_i[:, cv * n_kinds + lag_kind_idx]
            m1 = _weighted_class_mean(col, labels, path_weights, 1)
            m0 = _weighted_class_mean(col, labels, path_weights, 0)
            lag_diff[i, cv] = m1 - m0

        prefix = f"interface_{i:03d}_"
        _plot_importance_bar(
            np.abs(np.nan_to_num(metrics["cohens_d"])), feat_names,
            str(Path(plot_dir) / f"{prefix}effect_bar.png"), overw=overw,
            xlabel="|Cohen's d|",
            title="Dynamical feature importance (|Cohen's d|)",
            message="  Effect-size bar plot saved to",
            top_n=plot_top,
        )

        abs_d = np.abs(np.nan_to_num(metrics["cohens_d"]))
        order = np.argsort(abs_d)[::-1]
        results.append({
            "interface": i,
            "lambda": interfaces[i],
            "ranking": [(feat_names[idx], float(abs_d[idx])) for idx in order],
            "metrics": metrics,
        })
        print(f"Interface {i + 1}/{M} done in {datetime.datetime.now() - start}.")

    # --- cross-interface summaries -------------------------------------------
    _plot_interface_heatmap(
        results, feat_names,
        str(Path(plot_dir) / "interface_heatmap.png"), overw=overw,
        value_label="|Cohen's d|",
        title="|Cohen's d| of dynamical features across interfaces",
        top_n=plot_top,
    )
    _plot_lag_profile(
        lag_diff, cv_names, interfaces,
        str(Path(plot_dir) / "lag_profile.png"), unit=time_unit, overw=overw,
        top_n=plot_top,
    )

    red = _kind_redundancy(feats, kinds, n_cvs)
    _plot_redundancy(red, list(kinds), str(Path(plot_dir) / "feature_redundancy.png"), overw=overw)

    with open(redundancy_out, "w") as f:
        f.write("# mean |Spearman rho| between dynamical feature kinds, averaged over CVs\n")
        f.write("# a pair at ~1.0 means the two kinds carry the same information\n")
        f.write("# kind_a\tkind_b\tmean_abs_rho\n")
        for i, ka in enumerate(kinds):
            for j, kb in enumerate(kinds):
                if j <= i:
                    continue
                f.write(f"{ka}\t{kb}\t{red[i, j]:.6f}\n")
    print(f"Redundancy table saved to {redundancy_out}.")

    print("\nFeature-kind redundancy (mean |ρ| over CVs):")
    for i, ka in enumerate(kinds):
        for j, kb in enumerate(kinds):
            if j <= i:
                continue
            flag = "  <-- redundant" if np.isfinite(red[i, j]) and red[i, j] > 0.9 else ""
            print(f"  {ka:>9} vs {kb:<9} {red[i, j]:6.3f}{flag}")

    # --- pairwise event ordering (second pass) -------------------------------
    if pair_lags > 0:
        mean_abs_d = np.full(n_cvs, np.nan)
        for cv in range(n_cvs):
            vals = []
            for r in results:
                if r["ranking"] is None:
                    continue
                d = np.abs(r["metrics"]["cohens_d"][cv * n_kinds:(cv + 1) * n_kinds])
                if np.any(np.isfinite(d)):
                    vals.append(np.nanmax(d))
            if vals:
                mean_abs_d[cv] = float(np.mean(vals))

        if np.any(np.isfinite(mean_abs_d)):
            K = int(min(pair_lags, n_cvs))
            top = np.argsort(np.nan_to_num(mean_abs_d))[::-1][:K]
            # _discover_columns matches the file's own column names, so ask for
            # the pre-transform names rather than the cos()-renamed ones.
            raw_names, _, _ = _discover_columns(
                Path(cv_dir), pnr, op_col,
                cv_cols.split(",") if cv_cols else None, "utf-8",
                exclude=exclude.split(",") if exclude else None,
            )
            sel = [raw_names[i] for i in sorted(top)]
            print(f"\nPairwise lead/lag for the top {K} CVs: {', '.join(sel)}")
            print("  (second pass over the trajectory files)")

            diff, disp = _pairwise_lag_matrices(
                cv_dir, pnr[is_plus], interfaces, op_col, sel, labels, path_weights,
                half_window, window_mode, min_frames, max_lag, dt,
                angle_cols=angle_cols.split(",") if angle_cols else None,
                sym_angle_cols=sym_angle_cols.split(",") if sym_angle_cols else None,
            )
            for i in range(M):
                _plot_pair_lag(
                    diff[i], disp,
                    str(Path(plot_dir) / f"interface_{i:03d}_pairlag.png"),
                    unit=time_unit, lam=interfaces[i], overw=overw,
                )

    # --- rankings ------------------------------------------------------------
    with open(ranking_out, "w") as f:
        f.write(f"# window = {width} frames, mode = {window_mode}, dt = {dt} {time_unit}\n")
        f.write("# interface\tlambda\trank\tfeature\tabs_cohens_d\tcohens_d\tspearman\tks\n")
        for r in results:
            if r["ranking"] is None:
                continue
            m = r["metrics"]
            for rank, (name, abs_d_val) in enumerate(r["ranking"], 1):
                k = feat_names.index(name)
                f.write(
                    f"{r['interface']}\t{r['lambda']:.6f}\t{rank}\t{name}\t"
                    f"{abs_d_val:.6f}\t{m['cohens_d'][k]:.6f}\t"
                    f"{m['spearman'][k]:.6f}\t{m['ks'][k]:.6f}\n"
                )
    print(f"\nRankings saved to {ranking_out}.")
