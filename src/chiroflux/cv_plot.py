"""
Figures for the tables cv-compare writes (see cv_compare).

All are read straight from a comparison folder, so they can be redrawn
anywhere the tables are:

  cv_overview.png          the effect map: the CVs that differ most (rows,
                           by ranking.csv) against the order parameter, colour
                           d = (A - B) / pooled SD, a dot where q < 0.05
  cv_overview_all.png      the same for every CV, grouped by family
  cv_ranking.png           the largest significant effect of each top CV,
                           and where along the path it is
  cv_shape.png             per CV and OP window: is the difference a shift
                           (mean) or a change of spread (SD)?
  cv_correlations.png      the CVs against each other: within-window
                           correlation of A, of B, and A - B, clustered
  cv_blocking.png          statistical inefficiency against block size
  profiles/cv_profiles_<family>[_<n>].png   every CV's mean along the OP,
                           A vs B with 95% bands, a family per page
  distributions/<cv>.png   per CV: the OP x value maps of A, B and A - B,
                           and the distribution in every OP window from the
                           entry stage's lambda_A to the escape stage's
                           lambda_B, one row per side of the midplane

A is blue and solid, B orange and dashed; filled markers are q < 0.05 (BH
within the table), hollow ones not significant or too few frames to test;
bands are 95% bootstrap intervals; dotted lines are stage boundaries.  For a
-mirror-null CV the profile panel also draws -B, and the distributions
compare A with B mirrored.
"""

from __future__ import annotations

import json
import os
from typing import Annotated

import matplotlib
import numpy as np
import pandas as pd
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from . import panels
from .chiral_plot import (AXIS, COL, DIV, INK, INK2, MUTED, NEUTRAL, SEQ, Z95, _markers,
                          _style, result_folders)
from .cv_compare import MIDPLANE, safe_name
from .cv_histograms import HIST_DPI
from .pathdata import _check_overwrite

SIDE = {"A": COL["L"], "B": COL["D"]}
LS = {"A": "-", "B": "--"}
TOP_OVERVIEW = 40
TOP_RANKING = 30
TOP_SHAPE = 30
PANELS_PER_PAGE = 20
#: Most window histograms in one row of a distributions figure; a side of the
#: midplane with more windows wraps onto a second row.
MAX_COLS = 6
PLOT_OUTPUTS = ("cv_overview.png", "cv_overview_all.png", "cv_ranking.png", "cv_shape.png",
                "cv_correlations.png", "cv_blocking.png")


def profile_pages(names):
    """{figure path: CV names on it}: one page per family, split into pages
    of PANELS_PER_PAGE."""
    from .cv_compare import family

    fams = {}
    for n in names:
        fams.setdefault(family(n), []).append(n)
    out = {}
    for fam, members in fams.items():
        chunks = [members[i:i + PANELS_PER_PAGE] for i in range(0, len(members), PANELS_PER_PAGE)]
        for k, chunk in enumerate(chunks):
            tag = f"_{k + 1}" if len(chunks) > 1 else ""
            out[os.path.join("profiles", f"cv_profiles_{safe_name(fam)}{tag}.png")] = chunk
    return out


def figure_names(names, top=0):
    """Every figure plot_all(top=top) can write for these CVs (in ranking
    order), relative to a folder."""
    dist = names[:top] if top > 0 else names
    return (list(PLOT_OUTPUTS) + list(profile_pages(names))
            + [os.path.join("distributions", f"{safe_name(n)}.png") for n in dist])


# ── Reading ──────────────────────────────────────────────────────────────────

def _meta(in_dir):
    path = os.path.join(in_dir, "meta.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _read(in_dir, name):
    path = os.path.join(in_dir, f"{name}.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    return None if df.empty else df


def _npz(in_dir, name):
    path = os.path.join(in_dir, f"{name}.npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _save(fig, in_dir, name, meta):
    """Save with the folder's title (which comparison, weighting, subset,
    stage) above everything else."""
    fig.text(0.01, 1.005, meta["title"], ha="left", va="bottom", fontsize=11, color=INK,
             fontweight="bold")
    path = os.path.join(in_dir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=HIST_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _labels(meta):
    return {"A": meta["labels"][0], "B": meta["labels"][1]}


def _stage_lines(ax, meta, axis="x", lo=-np.inf, hi=np.inf):
    for e in meta.get("stage_edges", []):
        if lo < e < hi:
            (ax.axvline if axis == "x" else ax.axhline)(e, color=MUTED, lw=0.8, ls=":", zorder=1)


def _key(lab, mirror=False):
    handles = [Line2D([], [], color=SIDE["A"], ls=LS["A"], marker="o", label=lab["A"]),
               Line2D([], [], color=SIDE["B"], ls=LS["B"], marker="o", label=lab["B"])]
    if mirror:
        handles.append(Line2D([], [], color=SIDE["B"], ls=":", label=f"−{lab['B']} (mirror null)"))
    handles += [Line2D([], [], ls="", marker="o", mfc=INK2, mec="white", label="q < 0.05"),
                Line2D([], [], ls="", marker="o", mfc="white", mec=INK2,
                       label="n.s. / too few frames")]
    return handles


def _d_norm(values, floor=1.0):
    """A symmetric diverging scale for d: at least +-floor, else the 98th
    percentile of |d|, so one extreme cell does not wash out the rest."""
    v = np.abs(np.asarray(values, float))
    v = v[np.isfinite(v)]
    lim = max(floor, float(np.percentile(v, 98))) if v.size else floor
    return TwoSlopeNorm(0.0, -lim, lim)


def _finite_bins(prof):
    """The finite OP bins of profiles.csv, in order: (lo, hi) edges."""
    b = prof[["op_lo", "op_hi"]].drop_duplicates()
    b = b[np.isfinite(b["op_lo"]) & np.isfinite(b["op_hi"])].sort_values("op_lo")
    return np.r_[b["op_lo"].to_numpy(), b["op_hi"].to_numpy()[-1:]] if len(b) else np.array([])


# ── Figures ──────────────────────────────────────────────────────────────────

def _effect_map(ax, prof, names, edges, norm):
    """Rows = names, columns = finite OP bins; colour d, dot where q < 0.05."""
    sub = prof[prof["cv"].isin(names) & np.isfinite(prof["op_lo"]) & np.isfinite(prof["op_hi"])]
    piv_d = sub.pivot_table(index="cv", columns="op_lo", values="d", dropna=False)
    piv_q = sub.pivot_table(index="cv", columns="op_lo", values="q_bh", dropna=False)
    tested = sub.assign(t=sub["p_boot"].notna()).pivot_table(index="cv", columns="op_lo",
                                                             values="t", aggfunc="max")
    cols = edges[:-1]
    D = piv_d.reindex(index=names, columns=cols).to_numpy(float)
    Q = piv_q.reindex(index=names, columns=cols).to_numpy(float)
    T = tested.reindex(index=names, columns=cols).fillna(False).to_numpy(bool)
    D = np.where(T, D, np.nan)                       # untested bins stay neutral
    ax.set_facecolor(NEUTRAL)
    y = np.arange(len(names) + 1)
    im = ax.pcolormesh(edges, y, D, cmap=DIV, norm=norm)
    yy, xx = np.nonzero(np.nan_to_num(Q, nan=1.0) < 0.05)
    mids = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(mids[xx], yy + 0.5, "o", ms=2.6, color=INK, mec="white", mew=0.4)
    ax.set_yticks(y[:-1] + 0.5)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_ylim(len(names), 0)
    ax.grid(False)
    ax.set_xlabel("order parameter")
    return im


def plot_overview(in_dir, meta, prof, rank):
    lab = _labels(meta)
    edges = _finite_bins(prof)
    if edges.size < 2:
        return
    norm = _d_norm(prof.loc[prof["p_boot"].notna(), "d"])
    cbar = (f"d = ({lab['A']} − {lab['B']}) / pooled SD\n"
            f"blue: {lab['A']} higher, orange: {lab['B']} higher")
    note = ("grey: not tested (too few frames); dot: q < 0.05; dotted: stage boundaries;"
            " A + B for mirror-null CVs")
    top = rank["cv"].head(TOP_OVERVIEW).tolist()
    fig, ax = plt.subplots(figsize=(10.5, 1.6 + 0.2 * len(top)))
    im = _effect_map(ax, prof, top, edges, norm)
    _stage_lines(ax, meta, "x", edges[0], edges[-1])
    fig.colorbar(im, ax=ax, label=cbar, pad=0.015, fraction=0.035)
    ax.set_title(f"The {len(top)} CVs that differ most (by ranking.csv), along the path\n"
                 + note, loc="left", fontsize=9, color=INK2)
    _save(fig, in_dir, "cv_overview.png", meta)

    order = rank.sort_values(["family", "rank"])
    names = order["cv"].tolist()
    fig, ax = plt.subplots(figsize=(10.5, 1.6 + 0.16 * len(names)))
    im = _effect_map(ax, prof, names, edges, norm)
    ax.tick_params(axis="y", labelsize=5.5)
    fam = order["family"].to_numpy()
    for i in np.flatnonzero(fam[1:] != fam[:-1]) + 1:
        ax.axhline(i, color=INK, lw=0.8)
    _stage_lines(ax, meta, "x", edges[0], edges[-1])
    fig.colorbar(im, ax=ax, label=cbar, pad=0.015, fraction=0.035)
    ax.set_title("Every CV along the path, grouped by family (lines)\n" + note,
                 loc="left", fontsize=9, color=INK2)
    _save(fig, in_dir, "cv_overview_all.png", meta)


def plot_ranking(in_dir, meta, rank):
    lab = _labels(meta)
    top = rank.head(TOP_RANKING).iloc[::-1]
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 1.2 + 0.26 * len(top)))
    y = np.arange(len(top))
    d = top["d_at_max"].to_numpy(float)
    sig = top["significant_at_max"].to_numpy(bool)
    colors = np.where(d >= 0, SIDE["A"], SIDE["B"])
    for yi, di, c in zip(y, d, colors):
        if np.isfinite(di):
            ax.plot([0, di], [yi, yi], color=c, lw=2, alpha=0.35, solid_capstyle="round")
    for yi, di, c, s in zip(y, d, colors, sig):
        if np.isfinite(di):
            _markers(ax, [di], [yi], [s], c, size=7)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(top["cv"], fontsize=8)
    lim = np.nanmax(np.abs(d)) if np.isfinite(d).any() else 1.0
    ax.set_xlim(-1.1 * lim, 1.1 * lim)
    # Where, in a column of its own right of the plot, clear of every marker.
    for yi, r in zip(y, top.itertuples()):
        if np.isfinite(r.d_at_max):
            ax.text(1.02, yi, f"{_win_label(r.op_lo_at_max, r.op_hi_at_max)}, "
                    f"{r.bins_significant} bins q < 0.05", transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=7, color=MUTED)
    ax.set_xlabel(f"largest d = ({lab['A']} − {lab['B']}) / pooled SD "
                  f"(right: {lab['A']} higher)")
    ax.set_title("CVs ranked by their largest significant effect (filled: q < 0.05;"
                 " hollow: no significant bin, largest |d| shown)", loc="left", fontsize=9,
                 color=INK2)
    ax.grid(axis="y", visible=False)
    _save(fig, in_dir, "cv_ranking.png", meta)


def plot_profiles(in_dir, meta, prof):
    lab = _labels(meta)
    for name, members in profile_pages(list(dict.fromkeys(prof["cv"]))).items():
        ncol = 4
        nrow = int(np.ceil(len(members) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow), squeeze=False)
        any_mirror = False
        for ax, cv in zip(axes.ravel(), members):
            df = prof[prof["cv"] == cv]
            x = df["op_mid"].to_numpy(float)
            sig = (df["q_bh"] < 0.05).to_numpy()
            for side in ("A", "B"):
                y = df[side].to_numpy(float)
                se = df[f"{side}_se"].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                ax.fill_between(x[ok], (y - Z95 * se)[ok], (y + Z95 * se)[ok],
                                color=SIDE[side], alpha=0.15, lw=0)
                ax.plot(x[ok], y[ok], color=SIDE[side], ls=LS[side], lw=1.4, zorder=3)
                _markers(ax, x[ok], y[ok], sig[ok], SIDE[side], size=3.5)
            if (df["tested"] == "A+B").any():
                any_mirror = True
                y = -df["B"].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                ax.plot(x[ok], y[ok], color=SIDE["B"], ls=":", lw=1.1)
            ok = df["p_boot"].notna()
            dmax = df.loc[ok, "d"].abs().max() if ok.any() else np.nan
            ax.set_title(f"{cv}" + (f"   max |d| {dmax:.2f}" if np.isfinite(dmax) else ""),
                         loc="left", fontsize=8.5)
            _stage_lines(ax, meta, "x", np.nanmin(x) if np.isfinite(x).any() else -np.inf,
                         np.nanmax(x) if np.isfinite(x).any() else np.inf)
        for ax in axes.ravel()[len(members):]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("order parameter")
        fig.legend(handles=_key(lab, any_mirror), loc="upper right", ncol=6,
                   bbox_to_anchor=(1.0, 1.0))
        fig.suptitle("Mean along the path (bands: 95% CI)", x=0.01, ha="left", color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save(fig, in_dir, name, meta)


def _win_label(lo, hi):
    if not np.isfinite(lo):
        return f"OP < {hi:g}"
    if not np.isfinite(hi):
        return f"OP > {lo:g}"
    return f"OP {lo:g} to {hi:g}"


def window_rows(win_edges, stage_edges=(), split=MIDPLANE, max_cols=MAX_COLS):
    """
    The OP windows a distributions figure draws as histograms, as rows of
    (side label, window indices); the indices are into the per-window arrays,
    whose entry 0 is the open window below the first edge.  Every finite
    window from the first stage boundary (the entry stage's lambda_A) to the
    last (the escape stage's lambda_B) is drawn, the [0-] windows below it
    and the open ends are not.  The windows below `split` (the midplane) make
    one row and those above it the next, each in OP order; a side with more
    than max_cols windows wraps.  Without stage edges every finite window is
    drawn.
    """
    e = np.asarray(win_edges, dtype=float)
    if not e.size:
        return []
    lo, hi = np.r_[-np.inf, e], np.r_[e, np.inf]
    first, last = ((min(stage_edges), max(stage_edges)) if len(stage_edges)
                   else (e[0], e[-1]))
    tol = 1e-6
    inside = [w for w in range(lo.size) if np.isfinite(lo[w]) and np.isfinite(hi[w])
              and lo[w] >= first - tol and hi[w] <= last + tol]
    mid = 0.5 * (lo + hi)
    rows = []
    for label, ws in ((f"OP < {split:g}", [w for w in inside if mid[w] < split]),
                      (f"OP > {split:g}", [w for w in inside if mid[w] >= split])):
        rows += [(label, ws[k:k + max_cols]) for k in range(0, len(ws), max_cols)]
    return rows


def _count(n):
    """A frame count short enough for a panel title."""
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e4:
        return f"{n / 1e3:.0f}k"
    return f"{n:,.0f}"


def _window_panel(ax, m, w, sl, x, cv, lab, mirror, n_a, n_b, lo, hi, fmin, hmin):
    """One window's histograms of one CV.  A side is drawn with at least
    fmin frames; the window is tested (q markers) with hmin on both sides;
    a window without frames keeps its place, empty."""
    title = f"{_win_label(lo, hi)}\n{lab['A']} {_count(n_a)}, {lab['B']} {_count(n_b)} frames"
    ax.set_title(title, loc="left", fontsize=8)
    ax.set_xlabel(cv)
    if n_a <= 0 and n_b <= 0:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, "no frames", transform=ax.transAxes, ha="center", va="center",
                color=MUTED, fontsize=8)
        return
    for side, flip, n in (("A", False, n_a), ("B", mirror, n_b)):
        if n < fmin:
            continue
        y = m[f"{side}_1d"][w, sl]
        se = m[f"{side}_se_1d"][w, sl]
        ok = np.isfinite(y)
        ax.fill_between(x[ok], (y - Z95 * se)[ok], (y + Z95 * se)[ok],
                        color=SIDE[side], alpha=0.15, lw=0)
        ax.plot(x[ok], y[ok], color=SIDE[side], ls=LS[side], lw=1.4, zorder=3)
        if flip:
            ym = y[::-1]
            okm = np.isfinite(ym)
            ax.plot(x[okm], ym[okm], color=SIDE["B"], ls=":", lw=1.1)
    if n_a >= hmin and n_b >= hmin:
        q = np.nan_to_num(m["q_1d"][w, sl], nan=1.0) < 0.05
        ya = m["A_1d"][w, sl]
        ax.plot(x[q], ya[q], "o", ms=4, color=SIDE["A"], mec="white", mew=0.6, zorder=4)
    else:
        ax.text(0.98, 0.96, f"not tested\n(< {hmin:,} frames)", transform=ax.transAxes,
                ha="right", va="top", color=MUTED, fontsize=7)
    ax.set_ylim(bottom=0)


def plot_distributions(in_dir, meta, dist, names):
    """One figure per CV: OP x value maps (A, B, A - B; each OP row
    normalised, grey: too few frames) and the distribution in every OP
    window of the stages, one row per side of the midplane (window_rows).
    A row whose windows have no frames of this CV is left out (a stage
    folder has none on the far side of its stage)."""
    lab = _labels(meta)
    m = dist
    all_names = [str(n) for n in m["names"]]
    off, eoff = m["bin_offsets"], m["edge_offsets"]
    op_edges, win_edges = m["op_edges"], m["win_edges"]
    wlo, whi = np.r_[-np.inf, win_edges], np.r_[win_edges, np.inf]
    fmin = meta.get("min_frames", 50)
    hmin = meta.get("hist_min_frames", 200)
    all_rows = window_rows(win_edges, meta.get("stage_edges", ()))

    def per_cv(key, j):
        """Frames with a value of CV j (older tables: one count for all)."""
        a = m[key]
        return a[:, j] if a.ndim == 2 else a

    for cv in names:
        if cv not in all_names:
            continue
        j = all_names.index(cv)
        fA_op, fB_op = per_cv("frames_A_op", j), per_cv("frames_B_op", j)
        fA_w, fB_w = per_cv("frames_A_win", j), per_cv("frames_B_win", j)
        thin = (fA_op < fmin) | (fB_op < fmin)
        has = (fA_w + fB_w) > 0
        rows = [(side, ws) for side, ws in all_rows if has[ws].any()]
        sl = slice(off[j], off[j + 1])
        e = m["value_edges"][eoff[j]:eoff[j + 1]]
        x = 0.5 * (e[:-1] + e[1:])
        mirror = bool(m["mirror"][j])
        A2 = m["A_2d"][1:-1, sl].copy()             # finite OP bins only
        B2 = m["B_2d"][1:-1, sl].copy()
        A2[thin[1:-1]], B2[thin[1:-1]] = np.nan, np.nan
        ncol = max((len(ws) for _, ws in rows), default=0)
        nrow_1d = len(rows)
        fig = plt.figure(figsize=(3.2 * max(ncol, 4), 3.8 + 2.9 * nrow_1d))
        outer = fig.add_gridspec(1 + nrow_1d, 1, height_ratios=[1.45] + [1] * nrow_1d,
                                 hspace=0.55)
        maps = [fig.add_subplot(c) for c in outer[0].subgridspec(1, 3, wspace=0.45)]
        gs = (outer[1:].subgridspec(nrow_1d, ncol, hspace=0.75, wspace=0.3)
              if nrow_1d else None)
        if np.isfinite(A2).any() or np.isfinite(B2).any():
            vmax = np.nanmax([np.nanmax(A2) if np.isfinite(A2).any() else 0,
                              np.nanmax(B2) if np.isfinite(B2).any() else 0]) or 1.0
            for ax, side, h in zip(maps[:2], ("A", "B"), (A2, B2)):
                ax.set_facecolor(NEUTRAL)
                im = ax.pcolormesh(e, op_edges, h, cmap=SEQ, vmin=0, vmax=vmax)
                ax.set_title(lab[side], loc="left")
            fig.colorbar(im, ax=maps[:2], label="fraction of the OP bin's frames", pad=0.02)
            diff = A2 - (B2[:, ::-1] if mirror else B2)
            lim = np.nanmax(np.abs(diff)) if np.isfinite(diff).any() else 1.0
            maps[2].set_facecolor(NEUTRAL)
            im = maps[2].pcolormesh(e, op_edges, diff, cmap=DIV,
                                    norm=TwoSlopeNorm(0, -(lim or 1.0), lim or 1.0))
            maps[2].set_title(f"{lab['A']} − {lab['B']}" + (" mirrored" if mirror else ""),
                              loc="left")
            fig.colorbar(im, ax=maps[2], label=f"blue: {lab['A']} more, orange: {lab['B']} more",
                         pad=0.02)
        for ax in maps:
            ax.grid(False)
            ax.set_xlabel(cv)
            _stage_lines(ax, meta, "y", op_edges[0], op_edges[-1])
        maps[0].set_ylabel("order parameter")
        for r, (side_label, ws) in enumerate(rows):
            for k, w in enumerate(ws):
                ax = fig.add_subplot(gs[r, k])
                _window_panel(ax, m, w, sl, x, cv, lab, mirror, fA_w[w], fB_w[w], wlo[w],
                              whi[w], fmin, hmin)
                if k == 0:
                    ax.set_ylabel(f"{side_label}\nfraction of frames")
        handles = [Line2D([], [], color=SIDE["A"], ls=LS["A"], label=lab["A"]),
                   Line2D([], [], color=SIDE["B"], ls=LS["B"], label=lab["B"])]
        if mirror:
            handles.append(Line2D([], [], color=SIDE["B"], ls=":",
                                  label=f"{lab['B']} mirrored (value → −value)"))
        handles.append(Line2D([], [], ls="", marker="o", mfc=SIDE["A"], mec="white",
                              label="bin differs, q < 0.05"))
        fig.legend(handles=handles, loc="upper center", ncol=len(handles),
                   bbox_to_anchor=(0.5, 0.03))
        fig.suptitle(f"{cv}: distribution along the path (rows normalised; grey: too few "
                     f"frames) and per OP window (bands: 95% CI)", x=0.01, y=1.0, ha="left",
                     color=INK)
        _save(fig, in_dir, os.path.join("distributions", f"{safe_name(cv)}.png"), meta)


def plot_shape(in_dir, meta, shift, rank):
    """Is the difference a shift or a change of spread?  Per top CV and
    window: (mean_A - mean_B) / pooled SD and (SD_A - SD_B) / pooled SD."""
    lab = _labels(meta)
    top = rank["cv"].head(TOP_SHAPE).tolist()
    s = shift[shift["cv"].isin(top)].copy()
    s = s[np.isfinite(s["win_lo"]) & np.isfinite(s["win_hi"])]
    if s.empty:
        return
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.sqrt(0.5 * (s["sd_A"] ** 2 + s["sd_B"] ** 2))
        s["mean_rel"] = np.where(pooled > 0, s["mean_diff"] / pooled, np.nan)
        s["sd_rel"] = np.where(pooled > 0, s["sd_diff"] / pooled, np.nan)
    tested = s["mean_p"].notna()
    s.loc[~tested, ["mean_rel", "sd_rel"]] = np.nan
    edges = np.unique(np.r_[s["win_lo"], s["win_hi"]])
    fig, axes = plt.subplots(1, 2, figsize=(13, 1.8 + 0.22 * len(top)), sharey=True)
    norm = _d_norm(np.r_[s["mean_rel"], s["sd_rel"]], floor=0.5)
    for ax, val, q, title in ((axes[0], "mean_rel", "mean_q", "shift: Δmean / pooled SD"),
                              (axes[1], "sd_rel", "sd_q", "spread: ΔSD / pooled SD")):
        V = s.pivot_table(index="cv", columns="win_lo", values=val, dropna=False).reindex(
            index=top, columns=edges[:-1]).to_numpy(float)
        Q = s.pivot_table(index="cv", columns="win_lo", values=q, dropna=False).reindex(
            index=top, columns=edges[:-1]).to_numpy(float)
        ax.set_facecolor(NEUTRAL)
        im = ax.pcolormesh(edges, np.arange(len(top) + 1), V, cmap=DIV, norm=norm)
        yy, xx = np.nonzero(np.nan_to_num(Q, nan=1.0) < 0.05)
        mids = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(mids[xx], yy + 0.5, "o", ms=3, color=INK, mec="white", mew=0.4)
        ax.set_ylim(len(top), 0)
        ax.set_yticks(np.arange(len(top)) + 0.5)
        ax.set_yticklabels(top, fontsize=7)
        ax.grid(False)
        ax.set_title(title, loc="left")
        ax.set_xlabel("order parameter (window)")
        _stage_lines(ax, meta, "x", edges[0], edges[-1])
    fig.colorbar(im, ax=axes, pad=0.015, fraction=0.03,
                 label=f"blue: {lab['A']} higher / wider, orange: {lab['B']}")
    fig.text(0.01, 0.0, "How the distributions differ, per OP window (dot: q < 0.05; grey: "
             "not tested). Shift alone: the CV moves; spread alone: it fluctuates more or less.",
             ha="left", va="top", color=INK2, fontsize=9, transform=fig.transFigure)
    _save(fig, in_dir, "cv_shape.png", meta)


def _cluster_order(r):
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    if len(r) < 3:
        return np.arange(len(r))
    d = 1.0 - np.abs(np.nan_to_num(r, nan=0.0))
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return leaves_list(linkage(squareform(np.clip(d, 0, None), checks=False), "average"))


def plot_correlations(in_dir, meta, corr):
    lab = _labels(meta)
    names = np.array([str(n) for n in corr["names"]])
    A, B = corr["A"], corr["B"]
    order = _cluster_order(A)
    A, B, names = A[np.ix_(order, order)], B[np.ix_(order, order)], names[order]
    n = len(names)
    size = max(5.0, 0.11 * n + 2.0)
    fig, axes = plt.subplots(1, 3, figsize=(3 * size + 1.5, size + 0.6), layout="constrained")
    dlim = (max(0.2, float(np.nanpercentile(np.abs(A - B), 99)))
            if np.isfinite(A - B).any() else 1.0)
    for ax, M, title, cmap, lim, unit in (
            (axes[0], A, lab["A"], "RdBu_r", 1.0, "r"),
            (axes[1], B, lab["B"], "RdBu_r", 1.0, "r"),
            (axes[2], A - B, f"{lab['A']} − {lab['B']}", "PuOr_r", dlim, "Δr")):
        ax.set_facecolor(NEUTRAL)
        im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_title(title, loc="left")
        fs = max(3.0, min(7.0, 600 / max(n, 1) / 4))
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=90, fontsize=fs)
        ax.set_yticklabels(names, fontsize=fs)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=unit)
    fig.suptitle("CV x CV correlation within OP windows (each window's own means removed, so "
                 "a common trend along the path is not counted); ordered by clustering of "
                 f"{lab['A']}", x=0.01, ha="left", color=INK)
    _save(fig, in_dir, "cv_correlations.png", meta)


def plot_blocking(in_dir, meta, blocking):
    if blocking is None or blocking.empty:
        return
    sides = list(dict.fromkeys(blocking["side"]))
    fig, axes = plt.subplots(1, len(sides), figsize=(5.2 * len(sides), 3.8), squeeze=False,
                             sharey=True)
    # Stages told apart by marker, not shade: every line here is black.
    marks = ("o", "s", "^", "D", "v")
    for ax, side in zip(axes[0], sides):
        g = blocking[(blocking["side"] == side) & blocking["reliable"]]
        for k, (stage, h) in enumerate(g.groupby("stage")):
            agg = h.groupby("block")["inefficiency"]
            med, top = agg.median(), agg.quantile(0.9)
            mk = marks[k % len(marks)]
            ax.plot(med.index, med.values, color=INK, marker=mk, ms=4,
                    label=f"stage {stage}, median")
            ax.plot(top.index, top.values, color=INK, ls="--", lw=1.1, marker=mk, ms=3,
                    mfc="white", label=f"stage {stage}, 90th pct")
        ax.axvline(meta.get("block", 1), color=INK, lw=1.2, ls="-.",
                   label=f"block in use ({meta.get('block')})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("block size (paths)")
        ax.set_title(side, loc="left")
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("statistical inefficiency over the CVs")
    fig.suptitle("Blocking analysis: where the curves level off, the block holds the "
                 "correlation between consecutive paths", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, in_dir, "cv_blocking.png", meta)


def plot_all(in_dir, top=0):
    """Draw every figure the tables in `in_dir` allow.  `top` > 0 limits the
    per-CV distribution figures to the top CVs of ranking.csv."""
    meta = _meta(in_dir)
    prof, rank = _read(in_dir, "profiles"), _read(in_dir, "ranking")
    if meta is None or prof is None or rank is None:
        print(f"  [!] {in_dir}: no cv-compare tables; nothing drawn")
        return
    _style()
    plot_overview(in_dir, meta, prof, rank)
    plot_ranking(in_dir, meta, rank)
    plot_profiles(in_dir, meta, prof)
    shift = _read(in_dir, "shifts")
    if shift is not None:
        plot_shape(in_dir, meta, shift, rank)
    corr = _npz(in_dir, "correlations")
    if corr is not None:
        plot_correlations(in_dir, meta, corr)
    plot_blocking(in_dir, meta, _read(in_dir, "blocking"))
    dist = _npz(in_dir, "distributions")
    if dist is not None:
        names = rank["cv"].tolist()
        plot_distributions(in_dir, meta, dist, names[:top] if top > 0 else names)


def cv_plot(
    in_dir: Annotated[str, typer.Option("-in-dir", help="Folder cv-compare wrote (its -out-dir); the figures are written next to the tables", rich_help_panel=panels.INPUT)] = "cv_compare",
    top: Annotated[int, typer.Option("-top", help="Draw the per-CV distribution figures only for the top N CVs of ranking.csv; 0 draws every CV", rich_help_panel=panels.OUTPUT)] = 0,
    overw: Annotated[bool, typer.Option("-O", help="Force overwriting of existing figures", rich_help_panel=panels.OUTPUT)] = False,
):
    """Draw the figures of a cv-compare result: the effect map of every CV
    along the path, the ranking, every CV's profile, the per-CV distributions,
    shift vs spread, and the CV x CV correlations.

    Needs only the tables, so it runs wherever the comparison folder is.
    cv-compare already draws them; this redraws them, e.g. after copying the
    folder off the server.  Subfolders (stages, reactive/, unreactive/,
    unweighted/) are drawn too.
    """
    if not os.path.isdir(in_dir):
        raise typer.BadParameter(f"-in-dir: {in_dir} is not a folder")
    folders = result_folders(in_dir)
    for folder in folders:
        rank = _read(folder, "ranking")
        names = rank["cv"].tolist() if rank is not None else []
        for name in figure_names(names, top):
            _check_overwrite(os.path.join(folder, name), overw)
    for folder in folders:
        plot_all(folder, top)
