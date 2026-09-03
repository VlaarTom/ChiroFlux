"""Compare two SASA profiles - typically the L and D enantiomers.

``chiroflux sasa`` combines the runs listed in its ``-runs`` file (entry,
internal, escape) into one profile spanning the membrane. This command takes
two such profiles and reports where they differ.

Why a separate command rather than a flag on ``sasa``:

* ``sasa`` reads trajectories; this reads cached per-path arrays. Folding both
  into one invocation would re-parse everything each time a plot or a threshold
  is adjusted, which is the cost ``-skip-parsing`` already exists to avoid.
* Two profiles means two output directories, and ``sasa``'s paths are
  single-valued (one ``-out-dir``, one ``INTERMEDIATE_DIR``).
* It generalises for free: entry vs escape, or either against a control, with
  no new flags.

The difference is bootstrapped *jointly* rather than compared as two separate
confidence bands. Non-overlapping intervals do imply a difference, but
overlapping ones do not imply the absence of one, so reading two bands against
each other systematically understates significance. Each replicate here
resamples paths within A and within B and takes the difference of the two
resampled means, which gives a confidence interval on the difference itself.
"""

import json
import os
import warnings
from typing import Annotated, Optional

import matplotlib
import numpy as np
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import panels
from . import sasa as _sasa
from .cv_histograms import HIST_DPI
from .pathdata import _check_overwrite
from .sasa import (
    ALPHA,
    COMBINED_PLUS,
    MIN_WEIGHT_FRACTION,
    N_BOOTSTRAP,
    Z_BIN_WIDTH,
    Z_LABEL,
    Z_RANGE,
    _parse_range,
    build_sasa_bin_info,
    merge_groups,
)

#: What the profile is made of, in the order the outputs report it.
_QUANTITIES = ("tot", "pol", "apo", "exposure")

_QUANTITY_LABEL = {
    "tot": "total SASA [$\\mathrm{\\AA}^2$]",
    "pol": "polar SASA [$\\mathrm{\\AA}^2$]",
    "apo": "apolar SASA [$\\mathrm{\\AA}^2$]",
    "exposure": "exposed fraction",
}

#: Written by `sasa` next to its intermediates so a comparison can check that
#: two profiles share a z axis.
META_NAME = "sasa_meta.json"


def _load_side(out_dir, label):
    """Load the per-path arrays a `sasa` run left in `out_dir`.

    Reuses ``sasa.merge_groups``, which reads the module-level
    ``INTERMEDIATE_DIR``; that global is swapped for the duration rather than
    duplicating the loader.
    """
    intm = os.path.join(out_dir, "intermediates")
    if not os.path.isdir(intm):
        raise typer.BadParameter(
            f"-{label}: {out_dir!r} has no intermediates/ directory. Run "
            "`chiroflux sasa` for that simulation first; its intermediates are "
            "what this command compares."
        )
    prev = _sasa.INTERMEDIATE_DIR
    _sasa.INTERMEDIATE_DIR = intm
    try:
        data = merge_groups(COMBINED_PLUS)
    finally:
        _sasa.INTERMEDIATE_DIR = prev

    if data is None:
        raise typer.BadParameter(
            f"-{label}: no plus-ensemble intermediates found in {intm!r}."
        )
    return data


def _require_symmetric_z(z_rng):
    """Reversing the bin axis is only z -> -z on a range centred on zero."""
    lo, hi = z_rng
    if abs(lo + hi) > 1e-9:
        raise typer.BadParameter(
            f"-mirror-b needs a z axis symmetric about the membrane centre, but "
            f"the range is [{lo:g}, {hi:g}]. Reversing the bins equals z -> -z "
            "only when it is symmetric, so mirroring an asymmetric range would "
            "shift the profile instead of reflecting it."
        )


def _mirror_z(data):
    """Reflect a profile through the membrane centre: z -> -z.

    L and D can traverse the membrane in opposite directions, so one side's
    depth axis runs the other way and the two are not comparable bin for bin
    until one is reflected. Reversing the bin axis maps bin i to bin n-1-i,
    which is exactly z -> -z on a symmetric range (checked by the caller).

    The phosphate planes both swap and negate: reflecting turns the upper
    leaflet into the lower one.
    """
    out = dict(data)
    for key in ("w", "tot", "pol", "apo", "free", "tot2"):
        if key in data:
            out[key] = np.ascontiguousarray(data[key][:, ::-1])
    if data.get("hist2d") is not None:
        # (n_sasa, n_z): only the z axis reflects
        out["hist2d"] = np.ascontiguousarray(data["hist2d"][:, ::-1])
    if "zp_up" in data and "zp_lo" in data:
        out["zp_up"] = -np.asarray(data["zp_lo"], dtype=float)
        out["zp_lo"] = -np.asarray(data["zp_up"], dtype=float)
    return out


def _read_meta(out_dir):
    """The binning `sasa` recorded, or None for a run predating the metadata."""
    path = os.path.join(out_dir, META_NAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _check_comparable(dir_a, dir_b, data_a, data_b):
    """Refuse to difference two profiles that are not on the same z axis.

    A silent mismatch here would produce a difference profile that looks
    perfectly reasonable and means nothing.
    """
    n_a, n_b = data_a["w"].shape[1], data_b["w"].shape[1]
    if n_a != n_b:
        raise typer.BadParameter(
            f"-a has {n_a} z bins but -b has {n_b}. The two runs used different "
            "-z-range/-z-bin-width settings and cannot be differenced."
        )

    meta_a, meta_b = _read_meta(dir_a), _read_meta(dir_b)
    if meta_a is None or meta_b is None:
        warnings.warn(
            f"No {META_NAME} in one or both directories (they predate it), so "
            "only the bin count could be checked. Confirm both runs used the "
            "same -z-range and -z-bin-width.",
            stacklevel=2,
        )
        return

    for key in ("z_range", "z_bin_width", "fold_symmetric",
                "probe_radius", "occlude_with_water"):
        if meta_a.get(key) != meta_b.get(key):
            raise typer.BadParameter(
                f"-a and -b disagree on {key}: {meta_a.get(key)!r} vs "
                f"{meta_b.get(key)!r}. The profiles are not comparable."
            )


def _weighted_means(data, counts=None):
    """Weighted mean per z bin for each quantity, optionally over a resample.

    `counts` is a (n_replicates, n_paths) multinomial draw; None means use every
    path once, i.e. the point estimate.
    """
    W = data["w"]
    if counts is None:
        w = W.sum(axis=0)
        num = {k: data[k].sum(axis=0) for k in ("tot", "pol", "apo", "free")}
    else:
        w = counts @ W
        num = {k: counts @ data[k] for k in ("tot", "pol", "apo", "free")}

    with np.errstate(invalid="ignore", divide="ignore"):
        out = {k: np.where(w > 0, num[k] / w, np.nan)
               for k in ("tot", "pol", "apo", "free")}
        out["exposure"] = np.where(out["free"] > 0, out["tot"] / out["free"], np.nan)
    return out, w


def _bin_mask(data):
    """True where a bin holds too little weight to report, as `sasa` does."""
    w_tot = data["w"].sum(axis=0)
    peak = w_tot.max() if w_tot.size else 0.0
    return (peak <= 0) | (w_tot < MIN_WEIGHT_FRACTION * peak)


def _bootstrap_difference(data_a, data_b, n_bootstrap, alpha, seed=42):
    """Delta(z) = B - A per quantity, with a joint path-level bootstrap.

    Each replicate resamples paths within A and within B independently and
    differences the two resampled means, so the interval describes the
    difference rather than either profile. The sampling unit is the path, for
    the same reason as in ``sasa.weighted_profile``: frames within a path are
    consecutive points of one trajectory and resampling them would understate
    the uncertainty by orders of magnitude.
    """
    rng = np.random.default_rng(seed)
    mask = _bin_mask(data_a) | _bin_mask(data_b)

    mean_a, _ = _weighted_means(data_a)
    mean_b, _ = _weighted_means(data_b)
    out = {"mask": mask,
           "n_paths_a": data_a["w"].shape[0],
           "n_paths_b": data_b["w"].shape[0]}
    for q in _QUANTITIES:
        out[f"a_{q}"] = np.where(~mask, mean_a[q], np.nan)
        out[f"b_{q}"] = np.where(~mask, mean_b[q], np.nan)
        out[f"delta_{q}"] = out[f"b_{q}"] - out[f"a_{q}"]

    n_a, n_b = data_a["w"].shape[0], data_b["w"].shape[0]
    if min(n_a, n_b) < 2 or n_bootstrap < 1:
        for q in _QUANTITIES:
            out[f"lo_{q}"] = np.full(mask.shape, np.nan)
            out[f"hi_{q}"] = np.full(mask.shape, np.nan)
        return out

    lo_q, hi_q = alpha / 2 * 100, (1.0 - alpha / 2) * 100
    block = 200
    boots = {q: [] for q in _QUANTITIES}
    for start in range(0, n_bootstrap, block):
        n_rep = min(block, n_bootstrap - start)
        ca = rng.multinomial(n_a, np.full(n_a, 1.0 / n_a), size=n_rep).astype(float)
        cb = rng.multinomial(n_b, np.full(n_b, 1.0 / n_b), size=n_rep).astype(float)
        ma, _ = _weighted_means(data_a, ca)
        mb, _ = _weighted_means(data_b, cb)
        for q in _QUANTITIES:
            boots[q].append(mb[q] - ma[q])

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for q, chunks in boots.items():
            stack = np.concatenate(chunks, axis=0)
            out[f"lo_{q}"] = np.where(~mask, np.nanpercentile(stack, lo_q, axis=0), np.nan)
            out[f"hi_{q}"] = np.where(~mask, np.nanpercentile(stack, hi_q, axis=0), np.nan)
    return out


def _significant(res, q):
    """Bins whose difference interval excludes zero."""
    lo, hi = res[f"lo_{q}"], res[f"hi_{q}"]
    with np.errstate(invalid="ignore"):
        return np.isfinite(lo) & np.isfinite(hi) & ((lo > 0) | (hi < 0))


def _plot_comparison(z, res, q, label_a, label_b, out_path, overw=False):
    """Two profiles above, their difference with a confidence band below."""
    _check_overwrite(out_path, overw)
    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    top.plot(z, res[f"a_{q}"], color="#1a5c8a", label=label_a)
    top.plot(z, res[f"b_{q}"], color="#8a1a0d", label=label_b)
    top.set_ylabel(_QUANTITY_LABEL[q])
    top.legend(fontsize=9)
    top.set_title(f"{label_b} vs {label_a}: {q}")

    bot.axhline(0.0, color="0.4", lw=0.8)
    bot.plot(z, res[f"delta_{q}"], color="black", lw=1.2)
    bot.fill_between(z, res[f"lo_{q}"], res[f"hi_{q}"],
                     color="0.6", alpha=0.4, linewidth=0)
    sig = _significant(res, q)
    if sig.any():
        bot.plot(z[sig], res[f"delta_{q}"][sig], "o", ms=3, color="#8a1a0d",
                 label="interval excludes 0")
        bot.legend(fontsize=8)
    bot.set_ylabel(f"$\\Delta$ {q}\n({label_b} - {label_a})")
    bot.set_xlabel(Z_LABEL)

    plt.tight_layout()
    plt.savefig(out_path, dpi=HIST_DPI, bbox_inches="tight")
    plt.close()
    print(f"  {q}: {out_path}")


def sasa_compare(
    # ── Input data ────────────────────────────────────────────────────────
    a: Annotated[str, typer.Option("-a", help="REQUIRED. Output directory of the first `chiroflux sasa` run (the reference, e.g. L).", rich_help_panel=panels.INPUT)] = ...,
    b: Annotated[str, typer.Option("-b", help="REQUIRED. Output directory of the second run (e.g. D). The difference reported is B minus A.", rich_help_panel=panels.INPUT)] = ...,
    label_a: Annotated[str, typer.Option("-label-a", help="Legend label for -a.", rich_help_panel=panels.INPUT)] = "A",
    label_b: Annotated[str, typer.Option("-label-b", help="Legend label for -b.", rich_help_panel=panels.INPUT)] = "B",

    # ── Dataset construction ──────────────────────────────────────────────
    z_range: Annotated[Optional[str], typer.Option("-z-range", help="z axis as 'min,max' in Angstrom. Must match what both runs used; only needed if they did not use the default.", rich_help_panel=panels.DATASET)] = None,
    z_bin_width: Annotated[float, typer.Option("-z-bin-width", help="z bin width in Angstrom, matching both runs.", rich_help_panel=panels.DATASET)] = Z_BIN_WIDTH,

    # ── CV corrections (symmetry) ─────────────────────────────────────────
    mirror_b: Annotated[bool, typer.Option("-mirror-b/-no-mirror-b", help="Reflect the -b profile through the membrane centre (z -> -z) before differencing. Use when the two permeants traverse the membrane in opposite directions, so their depth axes run opposite ways. Needs a z range symmetric about zero, and must not be combined with a run-level mirror_z that already did it.", rich_help_panel=panels.SYMMETRY)] = False,

    # ── Model and training ────────────────────────────────────────────────
    n_bootstrap: Annotated[int, typer.Option("-n-bootstrap", help="Path-level resamples for the confidence band on the difference.", rich_help_panel=panels.MODEL)] = N_BOOTSTRAP,
    alpha: Annotated[float, typer.Option("-alpha", help="Two-sided significance level; the band is the [alpha/2, 1-alpha/2] interval.", rich_help_panel=panels.MODEL)] = ALPHA,
    seed: Annotated[int, typer.Option("-seed", help="Bootstrap seed.", rich_help_panel=panels.MODEL)] = 42,

    # ── Output ────────────────────────────────────────────────────────────
    out_dir: Annotated[str, typer.Option("-out-dir", help="Directory for the comparison plots and CSV.", rich_help_panel=panels.OUTPUT)] = "sasa_comparison",
    overw: Annotated[bool, typer.Option("-O", help="Overwrite existing files.", rich_help_panel=panels.OUTPUT)] = False,
):
    """Compare two SASA profiles and report where they differ.

    Takes the output directories of two `chiroflux sasa` runs - typically the
    L and D enantiomers, each already combining its own entry/internal/escape
    runs - and recomputes both profiles from their cached per-path arrays, so
    no trajectory is re-read.

    The difference is bootstrapped jointly rather than by comparing two
    separate confidence bands, which would understate significance. Bins whose
    interval excludes zero are marked in the plots and flagged in the CSV.
    """
    global Z_RANGE
    z_rng = _parse_range(z_range, "-z-range", 2) if z_range else Z_RANGE

    data_a = _load_side(a, "a")
    data_b = _load_side(b, "b")
    _check_comparable(a, b, data_a, data_b)

    prev_range, prev_width = _sasa.Z_RANGE, _sasa.Z_BIN_WIDTH
    _sasa.Z_RANGE, _sasa.Z_BIN_WIDTH = z_rng, z_bin_width
    try:
        z = build_sasa_bin_info()["z"][0]
    finally:
        _sasa.Z_RANGE, _sasa.Z_BIN_WIDTH = prev_range, prev_width

    if len(z) != data_a["w"].shape[1]:
        raise typer.BadParameter(
            f"-z-range/-z-bin-width give {len(z)} bins but the runs have "
            f"{data_a['w'].shape[1]}. Pass the values the runs were made with."
        )

    if mirror_b:
        _require_symmetric_z(z_rng)
        meta_b = _read_meta(b)
        if meta_b and any(meta_b.get("runs_mirror_z") or []):
            warnings.warn(
                f"-mirror-b was given, but {b!r} was built with mirror_z set on "
                f"run(s) {[n for n, m in zip(meta_b.get('runs', []), meta_b.get('runs_mirror_z', [])) if m]}. "
                "Reflecting again undoes that. Mirror either at build time or "
                "here, not both.",
                stacklevel=2,
            )
        data_b = _mirror_z(data_b)

    print(f"{label_a}: {data_a['w'].shape[0]} paths from {a}")
    print(f"{label_b}: {data_b['w'].shape[0]} paths from {b}"
          + ("  [mirrored: z -> -z]" if mirror_b else ""))
    print(f"bootstrapping the difference ({n_bootstrap} resamples)...")
    res = _bootstrap_difference(data_a, data_b, n_bootstrap, alpha, seed)

    os.makedirs(out_dir, exist_ok=True)
    print("\nplots:")
    for q in _QUANTITIES:
        _plot_comparison(z, res, q, label_a, label_b,
                         os.path.join(out_dir, f"sasa_compare_{q}.png"), overw)

    csv_path = os.path.join(out_dir, "sasa_comparison.csv")
    _check_overwrite(csv_path, overw)
    cols = ["z"]
    for q in _QUANTITIES:
        cols += [f"{label_a}_{q}", f"{label_b}_{q}", f"delta_{q}",
                 f"delta_{q}_lo", f"delta_{q}_hi", f"delta_{q}_significant"]
    with open(csv_path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for i, zi in enumerate(z):
            row = [f"{zi:.4f}"]
            for q in _QUANTITIES:
                sig = _significant(res, q)[i]
                row += [f"{res[f'a_{q}'][i]:.6g}", f"{res[f'b_{q}'][i]:.6g}",
                        f"{res[f'delta_{q}'][i]:.6g}", f"{res[f'lo_{q}'][i]:.6g}",
                        f"{res[f'hi_{q}'][i]:.6g}", str(bool(sig))]
            fh.write(",".join(row) + "\n")
    print(f"\nper-bin table: {csv_path}")

    print("\nbins whose difference interval excludes zero:")
    for q in _QUANTITIES:
        sig = _significant(res, q)
        usable = int(np.sum(~res["mask"]))
        if sig.any():
            zs = z[sig]
            print(f"  {q:<9} {int(sig.sum()):>4}/{usable:<4} bins, "
                  f"z in [{zs.min():.1f}, {zs.max():.1f}] A, "
                  f"largest |delta| = {np.nanmax(np.abs(res[f'delta_{q}'][sig])):.3g}")
        else:
            print(f"  {q:<9}    0/{usable:<4} bins - no significant difference")
    print("\nDone!")
