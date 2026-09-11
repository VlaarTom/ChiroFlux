"""Compare the DOPC/POPC contact preference between two simulations.

The single-simulation preference plot has one degree of freedom. Since
``frac_POPC = 1 - frac_DOPC`` and ``E = frac / F``, the two enrichment curves
are locked together as ``E_POPC = 6 - 5 E_DOPC``: they are the same number
drawn twice, on lever arms that differ fivefold. Differencing the enrichments
between two runs therefore inherits that distortion — the minority species'
curve exaggerates and the majority species' curve compresses the very same
difference. This module differences ``frac_DOPC`` instead, which is the one
free quantity and is on an undistorted scale.

Two things about a *pair* of runs that a naive comparison gets wrong:

**Leaflet.** A permeant entering from below only ever contacts the lower
leaflet, so its ``*_u_*`` coordination numbers are empty, and vice versa. In
this project L enters the lower leaflet and D the upper (``CA_C2_l_DOPC`` is
0.349 in L against 0.00016 for ``CA_C2_u_DOPC``; D is the mirror). Comparing
like-named labels across the two runs therefore pits real data against an empty
column. ``-leaflet-l``/``-leaflet-d`` say which leaflet each run samples, and
the pairs are matched on the leaflet-free stem: L's ``CA_C2_l_DOPC`` against
D's ``CA_C2_u_DOPC``.

**Degenerate bins.** An empty column does not come out blank. All of its weight
sits in one coordination-number bin, the first moment collapses to (bin centre)
x (frame weight), the frame weights cancel between the species, and the result
is ``frac_DOPC = 0.5`` exactly in every bin — a clean-looking flat line, not
missing data. Two such columns differenced give exactly zero, which reads as
perfect agreement. ``_compute_enrichment_from_chunks`` masks those bins, and
they are excluded here rather than counted as agreement.

The uncertainty comes from resampling *chunks in both runs inside one loop*, so
the difference is bootstrapped directly rather than assembled from two
independently-fitted confidence intervals.
"""

import csv
import os
import warnings
from pathlib import Path
from typing import Annotated

import matplotlib
import numpy as np
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import panels
from .cv_histograms import (
    DOPC_POPC_PAIRS,
    _compute_enrichment_from_chunks,
    load_per_chunk_2d,
)
from .pathdata import _check_overwrite

#: Subdirectory of a run's histogram output that holds the per-chunk .npz files.
INTERMEDIATES_SUBDIR = "intermediates"


def _leaflet_key(leaflet):
    if leaflet not in ("lower", "upper"):
        raise ValueError(
            f"-leaflet-* must be 'lower' or 'upper', got {leaflet!r}"
        )
    return "_l_" if leaflet == "lower" else "_u_"


#: Name for the pair built from the whole-lipid counts, which are called
#: plainly "DOPC" and "POPC" and so have no contact type to strip out.
WHOLE_LIPID_NAME = "whole_lipid"


def _stem(col):
    """Contact-type name, with the leaflet marker *and* the species dropped.

    Every pair is a DOPC-against-POPC comparison, so a species in the name is
    misleading: ``CA_C2_u_DOPC`` and ``CA_C2_u_POPC`` are two halves of one
    comparison, and that comparison is called ``CA_C2``. The name reaches the
    output file names and the overview's row labels, where carrying "DOPC"
    would suggest a DOPC-only quantity.
    """
    s = col.replace("_l_", "_").replace("_u_", "_")
    for species in ("DOPC", "POPC"):
        if s == species:
            return WHOLE_LIPID_NAME
        if s.endswith("_" + species):
            return s[: -(len(species) + 1)]
    return s


def _pairs_for_leaflet(leaflet):
    """{contact name: (dopc_col, popc_col)} for the pairs on one leaflet.

    Pairs with no leaflet marker belong to *both* leaflets: they count every
    atom of a species near the permeant, and a permeant that only reaches one
    leaflet makes that the near-leaflet count, so no mapping is needed.
    """
    key = _leaflet_key(leaflet)
    return {
        _stem(col_d): (col_d, col_p)
        for col_d, col_p, _ in DOPC_POPC_PAIRS
        if key in col_d or not ("_l_" in col_d or "_u_" in col_d)
    }


def _frac_and_axis(root, group_key, col_dopc, col_popc):
    """(chunks_dopc, chunks_popc, lamb_centers) for one run and one pair."""
    cd = load_per_chunk_2d(group_key, col_dopc, root=root)
    cp = load_per_chunk_2d(group_key, col_popc, root=root)
    if not cd or not cp:
        return None
    n = min(len(cd), len(cp))
    return cd[:n], cp[:n], np.asarray(cd[0][1], dtype=float)


def _observed_frac(chunks_d, chunks_p, lamb):
    r = _compute_enrichment_from_chunks(chunks_d, chunks_p, lamb)
    return r["frac_dopc"], r["mask"]


def _mean_cn(chunks_d, chunks_p, lamb):
    """Contacts per frame at each OP bin, both species pooled.

    Reported because the hard-degeneracy mask only catches bins with *no*
    weight above the lowest CN bin. A bin where nearly every frame is in that
    bin survives the mask but its fraction is still dominated by the bin's
    0.05 centre rather than by contacts, which pulls it toward 0.5. This is
    the number that says how much of the bin is real; `-min-mean-cn` filters
    on it.
    """
    r = _compute_enrichment_from_chunks(chunks_d, chunks_p, lamb)
    frames = r["frames_dopc"] + r["frames_popc"]
    contacts = r["contacts_dopc"] + r["contacts_popc"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(frames > 0, contacts / np.where(frames > 0, frames, 1),
                        np.nan)


def compare_one_pair(l_data, d_data, lamb, n_bootstrap, alpha, rng,
                     min_mean_cn=0.0):
    """Bootstrap the difference in DOPC contact fraction between two runs.

    Both runs' chunks are resampled inside the same iteration, so the spread
    that comes back is the spread of the *difference* and needs no assumption
    that the two runs' intervals can be combined after the fact.

    The p-value is pivoted on the observed difference — the fraction of
    resamples that move as far from it as it sits from zero — because the
    resamples are centred on the observation, not on the null.
    """
    (lcd, lcp), (dcd, dcp) = l_data, d_data

    frac_l, mask_l = _observed_frac(lcd, lcp, lamb)
    frac_d, mask_d = _observed_frac(dcd, dcp, lamb)
    mean_cn_l = _mean_cn(lcd, lcp, lamb)
    mean_cn_d = _mean_cn(dcd, dcp, lamb)
    mask = mask_l | mask_d
    if min_mean_cn > 0:
        with np.errstate(invalid="ignore"):
            mask = mask | ~(mean_cn_l >= min_mean_cn) | ~(mean_cn_d >= min_mean_cn)
    diff_obs = np.where(~mask, frac_l - frac_d, np.nan)

    n_l, n_d = len(lcd), len(dcd)
    boot = np.full((n_bootstrap, len(lamb)), np.nan)
    for b in range(n_bootstrap):
        il = rng.integers(0, n_l, size=n_l)
        idd = rng.integers(0, n_d, size=n_d)
        fl, ml = _observed_frac([lcd[i] for i in il], [lcp[i] for i in il], lamb)
        fd, md = _observed_frac([dcd[i] for i in idd], [dcp[i] for i in idd], lamb)
        boot[b] = np.where(~(ml | md), fl - fd, np.nan)

    with warnings.catch_warnings():
        # A bin masked in every resample gives an all-NaN column here. That is
        # the expected outcome for a bin neither run samples, not a problem.
        warnings.simplefilter("ignore", RuntimeWarning)
        ci_lo = np.nanpercentile(boot, alpha / 2 * 100, axis=0)
        ci_hi = np.nanpercentile(boot, (1 - alpha / 2) * 100, axis=0)

    n_valid = np.sum(~np.isnan(boot), axis=0)
    dev = np.abs(boot - diff_obs[np.newaxis, :])
    extreme = np.nansum(dev >= np.abs(diff_obs)[np.newaxis, :], axis=0).astype(float)
    p_value = np.where(
        (~mask) & (n_valid >= 10),
        np.maximum(extreme / np.where(n_valid > 0, n_valid, 1), 1.0 / n_bootstrap),
        np.nan,
    )

    return {
        "lamb": lamb,
        "frac_l": np.where(~mask, frac_l, np.nan),
        "frac_d": np.where(~mask, frac_d, np.nan),
        "diff": diff_obs,
        "ci_lo": np.where(~mask, ci_lo, np.nan),
        "ci_hi": np.where(~mask, ci_hi, np.nan),
        "p_value": p_value,
        "significant": (~np.isnan(p_value)) & (p_value < alpha),
        "mean_cn_l": mean_cn_l,
        "mean_cn_d": mean_cn_d,
        "mask": mask,
        "n_chunks_l": n_l,
        "n_chunks_d": n_d,
    }


def _plot_pair(stem, res, label_l, label_d, out_path, alpha, overw):
    _check_overwrite(out_path, overw)
    lamb, diff = res["lamb"], res["diff"]
    if not np.any(np.isfinite(diff)):
        return False

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.3]},
    )

    ax1.plot(lamb, res["frac_l"], color="#2a7fbf", lw=2.0, label=label_l)
    ax1.plot(lamb, res["frac_d"], color="#d1701f", lw=2.0, label=label_d)
    ax1.axhline(5.0 / 6.0, color="grey", lw=1.0, ls="--",
                label="bulk stoichiometry (5:6)")
    ax1.set_ylabel("DOPC contact fraction", fontsize=12)
    ax1.legend(fontsize=8, framealpha=0.85, edgecolor="#cccccc")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.fill_between(lamb, res["ci_lo"], res["ci_hi"], color="#7b3294",
                     alpha=0.22, label=f"{int((1 - alpha) * 100)}% CI")
    ax2.plot(lamb, diff, color="#7b3294", lw=2.0,
             label=f"{label_l} − {label_d}")
    ax2.axhline(0.0, color="grey", lw=1.0, ls=":")
    sig = res["significant"]
    if np.any(sig):
        ax2.scatter(lamb[sig], diff[sig], marker="o", s=18, zorder=5,
                    color="#7b3294", label=f"p < {alpha}")
    ax2.set_xlabel("OP_Lamb", fontsize=12)
    ax2.set_ylabel("Δ DOPC contact fraction", fontsize=12)
    ax2.legend(fontsize=8, framealpha=0.85, edgecolor="#cccccc")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{stem}   ({label_l} vs {label_d})", fontsize=11, color="#444444")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def _plot_overview(results, label_l, label_d, out_path, overw):
    """One row per pair: Δ fraction against OP_Lamb, on a shared scale."""
    _check_overwrite(out_path, overw)
    stems = [s for s, r in results.items() if np.any(np.isfinite(r["diff"]))]
    if not stems:
        return
    lamb = results[stems[0]]["lamb"]
    mat = np.vstack([results[s]["diff"] for s in stems])
    vmax = np.nanmax(np.abs(mat))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(max(7, 0.09 * len(lamb)),
                                    max(3.5, 0.32 * len(stems))))
    im = ax.imshow(mat, aspect="auto", cmap="PuOr_r", vmin=-vmax, vmax=vmax,
                   extent=[lamb[0], lamb[-1], len(stems) - 0.5, -0.5])
    plt.colorbar(im, ax=ax, label=f"Δ DOPC fraction ({label_l} − {label_d})")
    ax.set_yticks(np.arange(len(stems)))
    ax.set_yticklabels(stems, fontsize=8)
    ax.set_xlabel("OP_Lamb")
    ax.set_title(f"DOPC/POPC preference: {label_l} vs {label_d}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Overview -> {out_path}")


def _write_csv(results, out_path, label_l, label_d, overw):
    _check_overwrite(out_path, overw)
    fields = ["pair", "OP_Lamb", f"frac_dopc_{label_l}", f"frac_dopc_{label_d}",
              "delta_frac_dopc", "ci_lo", "ci_hi", "p_value", "significant",
              "mean_cn_l", "mean_cn_d", "n_chunks_l", "n_chunks_d"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for stem, r in results.items():
            for i in range(len(r["lamb"])):
                if r["mask"][i]:
                    continue
                w.writerow({
                    "pair": stem,
                    "OP_Lamb": f"{r['lamb'][i]:.6g}",
                    f"frac_dopc_{label_l}": f"{r['frac_l'][i]:.6g}",
                    f"frac_dopc_{label_d}": f"{r['frac_d'][i]:.6g}",
                    "delta_frac_dopc": f"{r['diff'][i]:.6g}",
                    "ci_lo": f"{r['ci_lo'][i]:.6g}",
                    "ci_hi": f"{r['ci_hi'][i]:.6g}",
                    "p_value": f"{r['p_value'][i]:.6g}",
                    "significant": r["significant"][i],
                    "mean_cn_l": f"{r['mean_cn_l'][i]:.6g}",
                    "mean_cn_d": f"{r['mean_cn_d'][i]:.6g}",
                    "n_chunks_l": r["n_chunks_l"],
                    "n_chunks_d": r["n_chunks_d"],
                })
    print(f"  Table -> {out_path}")


def preference_compare(
    dir_l: Annotated[str, typer.Option("-dir-l", help="First run's histogram output directory (the one holding intermediates/)", rich_help_panel=panels.INPUT)] = "L/analysis_output",
    dir_d: Annotated[str, typer.Option("-dir-d", help="Second run's histogram output directory", rich_help_panel=panels.INPUT)] = "D/analysis_output",
    label_l: Annotated[str, typer.Option("-label-l", help="Name for the first run in plots and column headers", rich_help_panel=panels.INPUT)] = "L",
    label_d: Annotated[str, typer.Option("-label-d", help="Name for the second run", rich_help_panel=panels.INPUT)] = "D",
    paths: Annotated[str, typer.Option("-paths", help="Which paths to compare: 'reactive' or 'nonreactive'", rich_help_panel=panels.DATASET)] = "reactive",
    ensemble: Annotated[str, typer.Option("-ensemble", help="Which ensemble: 'plus' or 'minus'", rich_help_panel=panels.DATASET)] = "plus",
    leaflet_l: Annotated[str, typer.Option("-leaflet-l", help="Leaflet the first run actually contacts: 'lower' or 'upper'. A run entering from below never touches the upper leaflet, so its upper columns are empty", rich_help_panel=panels.DATASET)] = "lower",
    leaflet_d: Annotated[str, typer.Option("-leaflet-d", help="Leaflet the second run actually contacts", rich_help_panel=panels.DATASET)] = "upper",
    min_mean_cn: Annotated[float, typer.Option("-min-mean-cn", help="Drop OP bins whose mean contacts-per-frame is below this in either run. 0 keeps all. Bins near the lowest CN bin's centre (0.05) are floor-dominated and sit close to 0.5 regardless of the data", rich_help_panel=panels.DATASET)] = 0.0,
    n_bootstrap: Annotated[int, typer.Option("-n-bootstrap", help="Chunk-level resamples used for the difference's confidence interval", rich_help_panel=panels.MODEL)] = 2000,
    alpha: Annotated[float, typer.Option("-alpha", help="Two-sided significance level", rich_help_panel=panels.MODEL)] = 0.05,
    seed: Annotated[int, typer.Option("-seed", help="Random seed for the bootstrap", rich_help_panel=panels.MODEL)] = 42,
    out_dir: Annotated[str, typer.Option("-out-dir", help="Directory for the plots and the table", rich_help_panel=panels.OUTPUT)] = "preference_compare",
    overw: Annotated[bool, typer.Option("-O", help="Force overwriting of existing files", rich_help_panel=panels.OUTPUT)] = False,
):
    """Difference the DOPC/POPC contact preference of two simulations.

    Compares `frac_DOPC(OP_Lamb)` rather than either enrichment curve. The two
    enrichments are locked together (`E_POPC = 6 - 5 E_DOPC`), so differencing
    them inherits the fivefold asymmetry of the 5:1 stoichiometry and makes the
    same difference look large for POPC and negligible for DOPC. The contact
    fraction is the one free quantity and is on an undistorted scale.

    Pairs are matched across runs on the leaflet-free column stem, so
    `-leaflet-l lower -leaflet-d upper` compares L's `CA_C2_l_DOPC` against D's
    `CA_C2_u_DOPC`. Getting this wrong is not obvious from the output: the
    leaflet a run never touches yields `frac_DOPC = 0.5` in every bin, which
    looks like a result rather than an empty column. Those bins are masked.

    Both runs must already have been processed by `chiroflux histograms`, since
    this reads their per-chunk `intermediates/`. Uncertainty comes from
    resampling chunks in both runs inside one loop, so the interval is on the
    difference itself.

    Writes to -out-dir:
      <stem>_compare.png   fractions for both runs, and their difference with CI
      overview.png         Δ fraction for every pair against OP_Lamb
      preference_compare.csv
    """
    for name, d in (("-dir-l", dir_l), ("-dir-d", dir_d)):
        if not os.path.isdir(d):
            raise ValueError(f"{name}: {d} is not a directory.")

    react = "non-reactive" if paths.startswith("non") else "reactive"
    if ensemble not in ("plus", "minus"):
        raise ValueError(f"-ensemble must be 'plus' or 'minus', got {ensemble!r}")
    group_key = (react, ensemble)

    root_l = os.path.join(dir_l, INTERMEDIATES_SUBDIR)
    root_d = os.path.join(dir_d, INTERMEDIATES_SUBDIR)
    for name, r in (("-dir-l", root_l), ("-dir-d", root_d)):
        if not os.path.isdir(r):
            raise ValueError(
                f"{name}: no '{INTERMEDIATES_SUBDIR}/' under it. Point at the "
                "histogram output directory, and run `chiroflux histograms` "
                "for that simulation first."
            )

    pairs_l = _pairs_for_leaflet(leaflet_l)
    pairs_d = _pairs_for_leaflet(leaflet_d)
    stems = sorted(set(pairs_l) & set(pairs_d))
    if not stems:
        raise ValueError(
            f"No pairs in common between {leaflet_l} ({len(pairs_l)}) and "
            f"{leaflet_d} ({len(pairs_d)})."
        )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    print(f"{label_l} ({leaflet_l} leaflet) vs {label_d} ({leaflet_d} leaflet)  "
          f"| {react}/{ensemble} | {len(stems)} pairs | {n_bootstrap} resamples")

    results = {}
    for stem in stems:
        got_l = _frac_and_axis(root_l, group_key, *pairs_l[stem])
        got_d = _frac_and_axis(root_d, group_key, *pairs_d[stem])
        if got_l is None or got_d is None:
            print(f"  {stem:<22} SKIPPED (missing intermediates)")
            continue
        lcd, lcp, lamb_l = got_l
        dcd, dcp, lamb_d = got_d
        if not np.allclose(lamb_l, lamb_d):
            print(f"  {stem:<22} SKIPPED (OP_Lamb axes differ between runs)")
            continue

        res = compare_one_pair((lcd, lcp), (dcd, dcp), lamb_l,
                               n_bootstrap, alpha, rng,
                               min_mean_cn=min_mean_cn)
        results[stem] = res

        usable = int(np.sum(~res["mask"]))
        n_sig = int(np.sum(res["significant"]))
        mean_d = np.nanmean(res["diff"])
        print(f"  {stem:<22} bins {usable:>4}/{len(lamb_l):<4} "
              f"mean Δ {mean_d:+.4f}  significant {n_sig:>4}")

        _plot_pair(stem, res, label_l, label_d,
                   os.path.join(out_dir, f"{stem}_compare.png"), alpha, overw)

    if not results:
        print("\nNothing comparable was found.")
        return

    _plot_overview(results, label_l, label_d,
                   os.path.join(out_dir, "overview.png"), overw)
    _write_csv(results, os.path.join(out_dir, "preference_compare.csv"),
               label_l, label_d, overw)

    all_diff = np.concatenate([r["diff"] for r in results.values()])
    all_sig = np.concatenate([r["significant"] for r in results.values()])
    finite = np.isfinite(all_diff)
    print(f"\n{int(finite.sum())} usable bins over {len(results)} pairs: "
          f"mean Δ {np.nanmean(all_diff):+.4f}, "
          f"largest |Δ| {np.nanmax(np.abs(all_diff)):.4f}, "
          f"{int(all_sig.sum())} significant at p < {alpha}.")
