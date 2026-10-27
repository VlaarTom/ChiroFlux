"""
Figures for the tables chiral-compare writes (see chiral_interface_analysis).

One PNG per question, all read straight from the comparison folder, so they
can be redrawn anywhere the CSVs are, without the cache or the runs:

  chiral_free_energy.png   G(depth) of L and D, ddG with its CI, and each
                           depth's share of the permeation resistance
  chiral_profiles.png      every depth profile, L vs D
  chiral_lifetimes.png     mean contact / H-bond lifetimes per partner class
  chiral_survival.png      survival S(tau) of the same runs, log-log
  chiral_contacts.png      H-bond partner fractions per donor, sn-2 share,
                           and the contacted lipid's glycerol response
  chiral_pose.png          the pose maps around the lipid stereocentre
  chiral_pose_depth.png    the same at matched contact depth, per depth, and
                           with the lipid C2's substituents marked
  chiral_split_half.png    with -split-half: do the L - D pose maps from the
                           first and second half of the runs agree?
  chiral_split_half_profiles.png   the same question for every depth profile
  chiral_blocking.png      statistical inefficiency against block size: how
                           many paths a bootstrap block (-block) needs
  chiral_drift.png         the same quantities per quarter of the run

chiral-compare on staged runs also writes reactive/ and unreactive/
subfolders (plus paths that cross their stage, and plus paths that return),
pooled over the stages and with one stage<k>_<run> folder per stage inside,
and all of it once more without the WHAM reweighting under unweighted/.
Their figures are the same minus the free-energy one, and are labelled.

Throughout, L is blue and D orange, a filled marker means q_bh < 0.05 (BH
within its own table), a hollow one means not significant or too few frames
to test, and bands are 95% bootstrap intervals (+-1.96 SE for L and D).

cos_chi_n is a pseudoscalar measured against the membrane normal, which is
not chiral.  A chiral permeant with any preferred orientation therefore gives
D = -L even if the lipids' handedness played no role; chi_lz inherits the
same through the tilt of the lipid's e_z along the normal (D ~ -L).  So
chiral-compare tests L + D for both (the `tested` column of
depth_profiles.csv), and their panels also draw -D: the deviation of L from
-D is what the lipids' handedness adds, and its markers are filled where that
deviation is significant.
"""

from __future__ import annotations

import os
from typing import Annotated

import matplotlib
import numpy as np
import pandas as pd
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

from . import panels
from .cv_histograms import HIST_DPI
from .pathdata import _check_overwrite

#: The observables chiral-compare histograms (chiral_interface_analysis.HIST_OBS).
HIST_NAMES = ("d_c2", "cos_axis_n", "cos_chi_n", "n_hb_lipid", "n_lipids_bound",
              "n_wat_polar", "pose_z", "chi_lz", "axis_lz")
HIST_DIR = "histograms"
PLOT_OUTPUTS = (("chiral_free_energy.png", "chiral_profiles.png",
                 "chiral_lifetimes.png", "chiral_survival.png",
                 "chiral_contacts.png", "chiral_pose.png", "chiral_pose_depth.png",
                 "chiral_split_half.png", "chiral_split_half_profiles.png",
                 "chiral_blocking.png", "chiral_drift.png")
                + ("chiral_reorientation.png", "chiral_barrier.png", "chiral_glycerol_dihedrals.png",
                   "chiral_orient_fe_cos_axis_n.png", "chiral_orient_fe_cos_chi_n.png")
                + ("chiral_orient_axis.png", "chiral_orient_chirality.png",
                   "chiral_split_half_orient_axis.png", "chiral_split_half_orient_chirality.png")
                + tuple(os.path.join(HIST_DIR, f"chiral_hist{d}_{n}.png")
                        for n in HIST_NAMES for d in ("1d", "2d")))

#: Subfolders chiral-compare writes for path subsets, and how figures name them.
SUBSETS = {"reactive": "reactive paths only", "unreactive": "unreactive paths only"}

COL = {"L": "#2a78d6", "D": "#eb6834"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
SEQ = LinearSegmentedColormap.from_list(
    "chiral_seq", ["#fcfcfb", "#e1e0d9", "#c3c2b7", "#898781", "#52514e", "#0b0b0b"])
DIV = LinearSegmentedColormap.from_list(          # D more <- 0 -> L more
    "chiral_div", ["#a3411b", COL["D"], NEUTRAL, COL["L"], "#184f95"])
Z95 = 1.96

OBS_LABELS = {
    "p_contact": "fraction of frames at a\nlipid stereocentre",
    "d_c2": "distance to nearest\nlipid C2 (nm)",
    "cos_axis_n": "cos(molecular axis,\noutward normal)",
    "cos_chi_n": "cos(chirality pseudovector,\noutward normal)",
    "n_hb_lipid": "H-bonds to lipid",
    "n_lipids_bound": "lipids H-bonded at once",
    "n_wat_polar": "waters at the polar groups",
    "pose_z": "pose_z: face of the lipid\nstereocentre (contact)",
    "chi_lz": "pseudovector . lipid e_z\n(contact)",
    "axis_lz": "axis . lipid e_z\n(contact)",
}
# Tables written before depth_profiles.csv had a `tested` column.
MIRROR_ODD_VS_NORMAL = ("cos_chi_n", "chi_lz")


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.labelsize": 9, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
        "lines.linewidth": 1.6, "font.size": 9,
    })


def _depth_mid(labels):
    """Bin centres from depth labels 'a-b'; the open bulk bin '>x' is NaN."""
    out = []
    for lab in labels:
        a, sep, b = str(lab).partition("-")
        out.append((float(a) + float(b)) / 2 if sep and a else np.nan)
    return np.array(out)


def _sig(df):
    return (df["q_bh"] < 0.05).to_numpy() if "q_bh" in df else np.zeros(len(df), bool)


def _markers(ax, x, y, sig, color, size=5.5):
    """Filled where significant, hollow (white face) elsewhere."""
    x, y, sig = np.asarray(x), np.asarray(y), np.asarray(sig, bool)
    ax.plot(x[sig], y[sig], "o", ms=size, color=color, mec="white", mew=0.8, zorder=4)
    ax.plot(x[~sig], y[~sig], "o", ms=size, mfc="white", mec=color, mew=1.2, zorder=4)


def _series(ax, x, y, se, sig, color, label):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, se, sig = x[ok], y[ok], se[ok], sig[ok]
    ax.fill_between(x, y - Z95 * se, y + Z95 * se, color=color, alpha=0.15, lw=0)
    ax.plot(x, y, color=color, label=label, zorder=3)
    _markers(ax, x, y, sig, color)


def _stage_edges(in_dir):
    """|lambda_A| / |lambda_B| of the stages, where the histogram is stitched."""
    edges = set()
    for side in ("L", "D"):
        path = os.path.join(in_dir, f"stages_{side}.csv")
        if os.path.exists(path):
            st = pd.read_csv(path)
            edges |= set(np.round(np.abs(st[["lam_a", "lam_b"]].to_numpy().ravel()), 3))
    return sorted(edges)


def _mark_stages(ax, edges, label=False):
    for e in edges:
        ax.axvline(e, color=MUTED, lw=0.8, ls=":", zorder=1)
    if label and edges:
        ax.text(edges[0], 1.0, " stage boundary", transform=ax.get_xaxis_transform(),
                color=MUTED, fontsize=7, va="top")


def _sig_handles():
    return [Line2D([], [], color=COL["L"], marker="o", label="L"),
            Line2D([], [], color=COL["D"], marker="o", label="D"),
            Line2D([], [], ls="", marker="o", mfc=INK2, mec="white", label="q < 0.05"),
            Line2D([], [], ls="", marker="o", mfc="white", mec=INK2,
                   label="n.s. / too few frames")]


def _sig_legend(ax, loc="best"):
    ax.legend(handles=_sig_handles(), loc=loc, ncol=2)


def _fig_legend(fig):
    """The L/D/significance key in one row at the top right, clear of the data."""
    fig.legend(handles=_sig_handles(), loc="upper right", ncol=4,
               bbox_to_anchor=(1.0, 1.0))


UNWEIGHTED = "unweighted"


def _subset(in_dir):
    """The figure tag of a result subfolder, e.g. 'reactive paths only',
    'reactive paths only, stage1_infinit_internal', 'unweighted' or
    'unweighted, reactive paths only, stage1_infinit_internal'; None for
    the top folder (WHAM-weighted, all paths)."""
    # Read from the end, so folder names further up never count.
    parts = os.path.normpath(os.path.abspath(in_dir)).split(os.sep)
    i, stage, subset = len(parts) - 1, None, None
    if i > 0 and parts[i].startswith("stage"):
        stage, i = parts[i], i - 1
    if parts[i] in SUBSETS:
        subset, i = parts[i], i - 1
    tag = ["unweighted (no WHAM reweighting)"] if parts[i] == UNWEIGHTED else []
    if subset:
        tag.append(SUBSETS[subset])
    elif stage:
        tag.append("all paths")
    if stage:
        tag.append(stage)
    return ", ".join(tag) or None


def result_folders(in_dir):
    """`in_dir` and every result folder chiral-compare wrote inside it: the
    pooled reactive/ and unreactive/ folders with their per-stage folders,
    and the same again under unweighted/."""
    out = []

    def stages(folder):
        return sorted(os.path.join(folder, d) for d in os.listdir(folder)
                      if d.startswith("stage") and os.path.isdir(os.path.join(folder, d)))

    for top in (in_dir, os.path.join(in_dir, UNWEIGHTED)):
        if not os.path.isdir(top):
            continue
        out.append(top)
        out += stages(top)                       # all paths, per stage
        for s in SUBSETS:
            sub = os.path.join(top, s)
            if os.path.isdir(sub):
                out.append(sub)
                out += sorted(os.path.join(sub, d) for d in os.listdir(sub)
                              if d.startswith("stage") and os.path.isdir(os.path.join(sub, d)))
    return out


def _save(fig, in_dir, name):
    tag = _subset(in_dir)
    if tag:
        # Above everything else, so it clears every title and legend; the
        # tight bounding box grows to include it.
        fig.text(0.01, 1.005, tag, ha="left", va="bottom", fontsize=11, color=INK,
                 fontweight="bold")
    path = os.path.join(in_dir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=HIST_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _read(in_dir, name):
    path = os.path.join(in_dir, f"{name}.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    return None if df.empty else df


# ── Figures ──────────────────────────────────────────────────────────────────

def plot_free_energy(in_dir):
    dd = _read(in_dir, "ddG_depth")
    if dd is None:
        if not _subset(in_dir):       # subsets have no free energies by design
            print("  [!] no ddG_depth.csv; free-energy figure skipped")
        return
    pr, sh = _read(in_dir, "permeability_ratio"), _read(in_dir, "resistance_shares")
    edges = _stage_edges(in_dir)
    x = _depth_mid(dd["depth"])
    fig, axes = plt.subplots(3 if sh is not None else 2, 1, sharex=True,
                             figsize=(6.4, 8.2 if sh is not None else 6.0),
                             gridspec_kw={"height_ratios": [3, 2, 2][:3 if sh is not None else 2]})
    ax = axes[0]
    for side in ("L", "D"):
        _series(ax, x, dd[f"G_{side}"].to_numpy(), dd[f"{side}_se"].to_numpy(),
                np.zeros(len(dd), bool), COL[side], side)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel("G(depth) relative to bulk (kcal/mol)")
    ax.legend(loc="upper right")
    title = "Free energy across the membrane"
    if pr is not None:
        r = pr.iloc[0]
        title += (f"\nln(P_L/P_D) = {r.ln_PL_over_PD:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]"
                  f"  ->  P_L/P_D = {r.PL_over_PD:.2g}  (equal D(z) assumed)")
    ax.set_title(title, loc="left")
    _mark_stages(ax, edges, label=True)

    ax = axes[1]
    d = dd["ddG_kcal"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(d)
    ax.fill_between(x[ok], dd["ci_lo"][ok], dd["ci_hi"][ok], color=INK2, alpha=0.15, lw=0)
    ax.plot(x[ok], d[ok], color=INK2)
    _markers(ax, x[ok], d[ok], _sig(dd)[ok], INK2)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel("ddG = G_L - G_D (kcal/mol)\n< 0: L more favourable")
    _mark_stages(ax, edges)

    if sh is not None:
        ax = axes[2]
        xs = _depth_mid(sh["depth"])
        sig = _sig(sh)
        for side in ("L", "D"):
            y = sh[f"share_{side}"].to_numpy()
            m = np.isfinite(xs) & np.isfinite(y) & (y > 0)
            ax.plot(xs[m], y[m], color=COL[side], label=side)
            _markers(ax, xs[m], y[m], sig[m], COL[side])
        ax.set_yscale("log")
        ax.set_ylabel("share of resistance\nint exp(G/kT) dz")
        _mark_stages(ax, edges)
    axes[-1].set_xlabel("depth |OP| (A): 0 = membrane centre, right = bulk water")
    _save(fig, in_dir, "chiral_free_energy.png")


def plot_profiles(in_dir):
    dp = _read(in_dir, "depth_profiles")
    if dp is None:
        return
    edges = _stage_edges(in_dir)
    names = list(dict.fromkeys(dp["observable"]))
    ncol = 5
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.7 * nrow), sharex=True)
    for ax, name in zip(axes.ravel(), names):
        df = dp[dp["observable"] == name]
        x = _depth_mid(df["depth"])
        sig = _sig(df)
        for side in ("L", "D"):
            _series(ax, x, df[side].to_numpy(), df[f"{side}_se"].to_numpy(), sig,
                    COL[side], side)
        mirror = ((df["tested"] == "L+D").any() if "tested" in df
                  else name in MIRROR_ODD_VS_NORMAL)
        if mirror:
            y = -df["D"].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            ax.plot(x[ok], y[ok], color=COL["D"], ls="--", lw=1.2, label="-D (mirror null)")
            ax.axhline(0, color=AXIS, lw=0.8)
            ax.legend(loc="lower right", fontsize=7)
        ax.set_title(OBS_LABELS.get(name, name), fontsize=8.5)
        _mark_stages(ax, edges)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("depth |OP| (A)")
    fig.suptitle("Depth profiles, L vs D (bands: 95% CI)", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _fig_legend(fig)
    _save(fig, in_dir, "chiral_profiles.png")


def _dumbbell(ax, labels, vL, vD, eL=None, eD=None, sig=None):
    """One row per label: L and D as dots joined by a hairline."""
    y = np.arange(len(labels))[::-1]
    sig = np.zeros(len(labels), bool) if sig is None else np.asarray(sig, bool)
    for yi, a, b in zip(y, vL, vD):
        ax.plot([a, b], [yi, yi], color=AXIS, lw=1.0, zorder=2)
    for side, v, e, dy in (("L", vL, eL, 0.12), ("D", vD, eD, -0.12)):
        v = np.asarray(v, float)
        if e is not None:
            ax.errorbar(v, y + dy, xerr=Z95 * np.asarray(e, float), fmt="none",
                        ecolor=COL[side], elinewidth=1.0, capsize=0, zorder=3)
        _markers(ax, v, y + dy, sig, COL[side], size=6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="y", visible=False)


def plot_lifetimes(in_dir):
    lt = _read(in_dir, "lifetimes")
    if lt is None:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    _dumbbell(ax, lt["kind"].tolist(), lt["tau_L_ps"], lt["tau_D_ps"],
              lt["L_se"], lt["D_se"], _sig(lt))
    ax.set_xlabel("mean continuous lifetime (ps)")
    ax.set_title("H-bond and stereocentre-contact lifetimes per partner", loc="left")
    _sig_legend(ax, loc="lower right")
    _save(fig, in_dir, "chiral_lifetimes.png")


def plot_survival(in_dir):
    sv = _read(in_dir, "lifetime_survival")
    if sv is None:
        return
    kinds = list(dict.fromkeys(sv["kind"]))
    ncol = 4
    nrow = int(np.ceil(len(kinds) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.6 * nrow),
                             sharex=True, sharey=True)
    for ax, kind in zip(axes.ravel(), kinds):
        df = sv[sv["kind"] == kind]
        x = df["tau_ps"].to_numpy()
        sig = _sig(df)
        for side in ("L", "D"):
            y = df[f"S_{side}"].to_numpy()
            m = np.isfinite(y) & (y > 0) & (x > 0)
            ax.plot(x[m], y[m], color=COL[side], label=side)
            _markers(ax, x[m], y[m], sig[m], COL[side], size=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(kind, fontsize=9)
    for ax in axes.ravel()[len(kinds):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("tau (ps)")
    for ax in axes[:, 0]:
        ax.set_ylabel("S(tau): runs lasting >= tau")
    fig.suptitle("Lifetime survival, L vs D", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _fig_legend(fig)
    _save(fig, in_dir, "chiral_survival.png")


def plot_contacts(in_dir):
    hp = _read(in_dir, "hbond_partners")
    sn = _read(in_dir, "sn2_preference")
    gc = _read(in_dir, "glycerol_conformers")
    if hp is None and sn is None and gc is None:
        return
    donors = sorted(hp.loc[hp["depth"] == "all", "donor"].unique()) if hp is not None else []
    n = len(donors) + (sn is not None) + (gc is not None)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.2), squeeze=False)
    axes = list(axes.ravel())
    for k in donors:
        ax = axes.pop(0)
        df = hp[(hp["depth"] == "all") & (hp["donor"] == k)]
        _dumbbell(ax, df["partner"].tolist(), df["L"], df["D"], df["L_se"], df["D_se"],
                  _sig(df))
        ax.set_xlabel("fraction of frames")
        ax.set_title(f"donor {k}: H-bond partner, all depths", loc="left", fontsize=9)
    if sn is not None:
        ax = axes.pop(0)
        _dumbbell(ax, [f"donor {d}" for d in sn["donor"]], sn["L"], sn["D"],
                  sn["L_se"], sn["D_se"], _sig(sn))
        ax.axvline(0.5, color=AXIS, lw=0.8, ls="--")
        ax.set_xlabel("sn-2 / (sn-2 + sn-1) of glycerol-ester H-bonds")
        ax.set_title("sn-2 preference", loc="left", fontsize=9)
    if gc is not None:
        ax = axes.pop(0)
        df = gc[gc["lipid"] == "contact_minus_bulk"]
        labels = [f"{d.split('_')[0]} {s}" for d, s in zip(df["dihedral"], df["state"])]
        _dumbbell(ax, labels, df["L"], df["D"], df["L_se"], df["D_se"], _sig(df))
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_xlabel("population: contacted lipid - bulk")
        ax.set_title("glycerol dihedrals (a: O11-C1-C2-O21,\nb: O21-C2-C3-O31)",
                     loc="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _fig_legend(fig)
    _save(fig, in_dir, "chiral_contacts.png")


def plot_pose(in_dir):
    path = os.path.join(in_dir, "pose_maps.npz")
    if not os.path.exists(path):
        return
    with np.load(path) as z:
        maps = {k: z[k] for k in ("L", "D", "L_minus_D")}
        phi, cos = z["phi_edges"], z["cos_edges"]
    test = _read(in_dir, "pose_jsd")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), sharey=True)
    vmax = max(maps["L"].max(), maps["D"].max())
    for ax, side in zip(axes[:2], ("L", "D")):
        im = ax.pcolormesh(phi, cos, maps[side].T, cmap=SEQ, vmin=0, vmax=vmax)
        ax.set_title(f"{side}: where the permeant sits", loc="left")
    fig.colorbar(im, ax=axes[:2], label="fraction of contact frames", shrink=0.9, pad=0.02)
    lim = np.abs(maps["L_minus_D"]).max() or 1.0
    im = axes[2].pcolormesh(phi, cos, maps["L_minus_D"].T, cmap=DIV,
                            norm=TwoSlopeNorm(0, -lim, lim))
    title = "L - D"
    if test is not None:
        t = test.iloc[0]
        title += f"   JSD = {t.jsd:.4f} (null 95%: {t.null_95:.4f}, p = {t.p_perm:.3g})"
    axes[2].set_title(title, loc="left")
    fig.colorbar(im, ax=axes[2], label="blue: L more, orange: D more", pad=0.02)
    for ax in axes:
        ax.set_xlabel("phi in the glycerol e_x-e_y plane (deg)")
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.grid(False)
    axes[0].set_ylabel("cos(theta) = pose_z\n(+: e_z face of the lipid C2)")
    _save(fig, in_dir, "chiral_pose.png")


#: Where the lipid C2's own substituents point in the pose-map frame (phi in
#: degrees, cos(theta)), for an R lipid.  Mean over the 132 R lipids of a
#: DOPC/POPC conf.gro; C1, C3, O21 and HS are fixed by the tetrahedral
#: bonding (SD of cos <= 0.03), the sn-2 carbonyl O22 follows the ester's
#: rotation (SD 0.11), so it is drawn as approximate.  An S lipid has the
#: same phi and the opposite cos.
SUBSTITUENTS = (
    ("O21", 90.0, 0.00, True),
    ("C1 (P side)", -22.0, 0.45, True),
    ("C3 (sn-1 side)", -158.0, 0.45, True),
    ("H", -91.0, -0.92, True),
    ("O22 (approx.)", 87.0, -0.70, False),
)


def _lipid_sign(in_dir):
    """+1 for R lipids (the default), -1 if stereochemistry_L.csv says S.
    A subset folder reads it from the comparison folder above it."""
    folder = os.path.normpath(in_dir)
    for _ in range(4):                    # up to unweighted/<subset>/<stage>
        path = os.path.join(folder, "stereochemistry_L.csv")
        if os.path.exists(path):
            break
        folder = os.path.dirname(folder)
    try:
        st = pd.read_csv(path, usecols=["lipids_R", "lipids_S"], nrows=1)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return 1.0
    return -1.0 if st.empty is False and st.iloc[0, 1] > st.iloc[0, 0] else 1.0


def _substituents(ax, sign, labels=True):
    from matplotlib import patheffects

    halo = [patheffects.withStroke(linewidth=2.2, foreground="white")]
    for name, phi, cos, fixed in SUBSTITUENTS:
        c = sign * cos
        ax.plot(phi, c, "o" if fixed else "D", ms=5 if fixed else 4.5,
                mfc="white" if fixed else "none", mec=INK, mew=1.1, zorder=5)
        if labels:
            ax.annotate(name, (phi, c), xytext=(5, 4), textcoords="offset points",
                        fontsize=7, color=INK, path_effects=halo, zorder=6)


def _pose_axes(ax, ylabel=True):
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlim(-180, 180)
    ax.set_ylim(-1, 1)
    ax.grid(False)
    if ylabel:
        ax.set_ylabel("cos(theta) = pose_z")


def plot_pose_depth(in_dir):
    """
    The pose maps at matched contact depth (pose_maps_depth.npz), next to the
    pooled ones of chiral_pose.png: top row L, D and L - D with both on the
    same depth weights; bottom row L - D within each depth bin, and how L's
    and D's contacts are spread over depth.  The lipid C2's substituents are
    marked, so the faces can be read off: pose_z < 0 is the H face, > 0 the
    face the C1/C3 backbone leans to (for R lipids).
    """
    path = os.path.join(in_dir, "pose_maps_depth.npz")
    if not os.path.exists(path):
        print("  [!] no pose_maps_depth.npz (tables from an older chiral-compare); "
              "depth-matched pose figure skipped")
        return
    with np.load(path) as z:
        m = {k: z[k] for k in z.files}
    test = _read(in_dir, "pose_jsd_depth")
    sign = _lipid_sign(in_dir)
    phi, cos = m["phi_edges"], m["cos_edges"]
    edges = m["depth_edges"]
    labels = [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])] + [f">{edges[-1]:g}"]
    # the table's own labels (Grid.pose_depth_labels), to look rows up by
    keys = [f"{a:.2f}-{b:.2f}" for a, b in zip(edges[:-1], edges[1:])] + [f">{edges[-1]:.2f}"]
    tested = np.flatnonzero(m["tested"])
    if not tested.size:
        print("  [!] no depth bin has enough contact frames in both L and D; "
              "depth-matched pose figure skipped")
        return

    def row(name, depth=None):
        if test is None:
            return None
        sel = test["map"] == name
        if depth is not None:
            sel &= test["depth"].astype(str) == depth
        return test[sel].iloc[0] if sel.any() else None

    fig = plt.figure(figsize=(14.5, 9.4))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.15, 1.0], hspace=0.04)

    axes = top.subplots(1, 3, sharey=True)
    vmax = np.nanmax([m["L_matched"].max(), m["D_matched"].max()])
    for ax, side in zip(axes[:2], ("L", "D")):
        im = ax.pcolormesh(phi, cos, m[f"{side}_matched"].T, cmap=SEQ, vmin=0, vmax=vmax)
        ax.set_title(f"{side}, depth-matched", loc="left")
        _pose_axes(ax, ylabel=side == "L")
        _substituents(ax, sign)
    top.colorbar(im, ax=axes[:2], label="fraction of contact frames", shrink=0.9, pad=0.02)
    diff = m["L_minus_D_matched"]
    lim = np.nanmax(np.abs(diff)) or 1.0
    im = axes[2].pcolormesh(phi, cos, diff.T, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim))
    title = "L - D, depth-matched"
    r, p0 = row("depth_matched"), row("pooled")
    if r is not None:
        title += f"\nJSD {r.jsd:.4f} (null 95% {r.null_95:.4f}, p = {r.p_perm:.3g})"
    if p0 is not None:
        title += f";  pooled: JSD {p0.jsd:.4f}, p = {p0.p_perm:.3g}"
    axes[2].set_title(title, loc="left", fontsize=9)
    _pose_axes(axes[2], ylabel=False)
    _substituents(axes[2], sign)
    top.colorbar(im, ax=axes[2], label="blue: L more, orange: D more", pad=0.02)
    for ax in axes:
        ax.set_xlabel("phi in the glycerol e_x-e_y plane (deg)")
    face = "H face" if sign > 0 else "backbone face"
    other = "backbone face" if sign > 0 else "H face"
    top.suptitle(f"Pose around the nearest lipid C2 at matched contact depth   "
                 f"(cos < 0: {face}, cos > 0: {other})",
                 x=0.01, ha="left", color=INK, fontsize=10.5)

    n = len(tested)
    axes = bottom.subplots(1, n + 1, gridspec_kw={"width_ratios": [1] * n + [1.25]})
    axes = np.atleast_1d(axes)
    per = [m["L_depth"][d] - m["D_depth"][d] for d in tested]
    # Scale on the depths that carry the comparison: a bin with a sliver of
    # the contacts is noisy and would wash out the rest (it saturates).
    weight = m["match_weight"][tested]
    main = [x for x, wt in zip(per, weight) if wt >= 0.01] or per
    lim = max([np.nanmax(np.abs(x)) for x in main] + [1e-12])
    for k, (ax, d, x) in enumerate(zip(axes, tested, per)):
        im = ax.pcolormesh(phi, cos, x.T, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim))
        _pose_axes(ax, ylabel=k == 0)
        _substituents(ax, sign, labels=False)
        r = row("depth", keys[d])
        t = f"{labels[d]} A ({m['match_weight'][d]:.0%})"
        if r is not None:
            t += f"\nJSD {r.jsd:.3f}, q = {r.q_bh:.2g}"
        ax.set_title(t, fontsize=8.5, loc="left")
        ax.set_xticks([-180, 0, 180])
        ax.set_xlabel("phi (deg)")
        if k:
            ax.set_yticklabels([])
    if n:
        bottom.colorbar(im, ax=axes[:n], label="L - D within the depth bin",
                        pad=0.01, shrink=0.85)

    ax = axes[-1]
    x = np.arange(len(labels))
    for side, dx in (("L", -0.2), ("D", 0.2)):
        ax.bar(x + dx, m[f"contact_share_{side}"], width=0.38, color=COL[side],
               label=f"{side} contacts")
    ax.plot(x[tested], m["match_weight"][tested], "o", ms=6, mfc="white", mec=INK,
            mew=1.2, label="matching weight", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("depth |OP| (A)")
    ax.set_ylabel("share of contact frames")
    ax.set_title("where the contacts happen\n(% in the panel titles: matching weight)",
                 fontsize=8.5, loc="left")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(axis="x", visible=False)
    _save(fig, in_dir, "chiral_pose_depth.png")


def plot_split_half(in_dir):
    """
    The split-half replication (split_half.csv, split_half_maps.npz): the
    L - D pose map from the first half of every stage above the one from
    the second half, for the depth-matched map and every pose depth bin, with
    the correlation between the two and its block-relabelling null.
    """
    path = os.path.join(in_dir, "split_half_maps.npz")
    test = _read(in_dir, "split_half")
    if not os.path.exists(path) or test is None:
        return
    with np.load(path, allow_pickle=False) as z:
        h1, h2, labels = z["half1"], z["half2"], [str(x) for x in z["labels"]]
        phi, cos = z["phi_edges"], z["cos_edges"]
    _split_half_figure(in_dir, h1, h2, labels, phi, cos, test, "cos(theta)",
                       "Does the L - D pose difference replicate?  L - D maps from the first "
                       "and second half of every stage (blue: L more, orange: D more)",
                       "chiral_split_half.png")


def _split_half_figure(in_dir, h1, h2, labels, phi, cos, test, ylabel, title, out):
    sign = _lipid_sign(in_dir)
    n = len(labels)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n + 1.2, 6.2), sharex=True, sharey=True,
                             squeeze=False)
    for k in range(n):
        lim = np.nanmax(np.abs(np.r_[h1[k].ravel(), h2[k].ravel()])) or 1.0
        for row, h in enumerate((h1, h2)):
            ax = axes[row, k]
            im = ax.pcolormesh(phi, cos, h[k].T, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim))
            _pose_axes(ax, ylabel=False)
            _substituents(ax, sign, labels=False)
            ax.set_xticks([-180, 0, 180])
        r = test.iloc[k]
        name = "depth-matched" if labels[k] == "matched" else \
            labels[k].replace(".00", "").replace(".50", ".5") + " A"
        axes[0, k].set_title(f"{name}\nr = {r.corr_halves:+.2f} (null 95% {r.null_95:+.2f}, "
                             f"p = {r.p_perm:.2g})", fontsize=8.5, loc="left")
        axes[1, k].set_xlabel("phi (deg)")
    axes[0, 0].set_ylabel(f"first half\n{ylabel}")
    axes[1, 0].set_ylabel(f"second half\n{ylabel}")
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, in_dir, out)


#: Stage colours for the blocking figures: categorical slots that are not the
#: L blue / D orange, which here only mark the line style.
STAGE_COL = ("#1baf7a", "#eda100", "#4a3aa7", "#e87ba4", "#008300")


def _stage_legend(fig, stages):
    handles = [Line2D([], [], color=STAGE_COL[k % len(STAGE_COL)], label=f"stage {k}")
               for k in stages]
    handles += [Line2D([], [], color=INK2, ls="-", label="L"),
                Line2D([], [], color=INK2, ls="--", label="D")]
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), bbox_to_anchor=(1.0, 1.0))


def plot_blocking(in_dir):
    """
    blocking.csv: statistical inefficiency against block size, per quantity,
    stage and enantiomer.  A curve that levels off says how many paths a
    block needs; one still rising at the right edge means the correlation
    is longer than any block tried.  Hollow markers: fewer than 10 blocks.
    """
    bl = _read(in_dir, "blocking")
    if bl is None:
        return
    bl = bl[np.isfinite(bl["inefficiency"])]
    if bl.empty or bl["block"].max() < 10:
        print("  [!] too few paths for a blocking analysis; blocking figure skipped")
        return
    names = list(dict.fromkeys(bl["observable"]))
    stages = sorted(bl["stage"].unique())
    choice = _read(in_dir, "block_choice")
    chosen = int(choice["chosen_block"].iloc[0]) if choice is not None else None
    mode = str(choice["mode"].iloc[0]) if choice is not None else ""
    # The bulk-lipid quantities can have a block of their own (block_choice.csv
    # rows used_for glycerol_conformers); each panel shows the one it is used with.
    per_obs = (choice.groupby("observable")["chosen_block"].first().astype(int).to_dict()
               if choice is not None and "observable" in choice else {})
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.9 * nrow), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        here = per_obs.get(name, chosen)
        if here:
            ax.axvline(here, color=INK2, lw=0.9, ls=":", zorder=1)
            if here != chosen:
                ax.text(here, 0.97, f" block {here}", transform=ax.get_xaxis_transform(),
                        fontsize=7, color=INK2, va="top")
        for (side, stage), g in bl[bl["observable"] == name].groupby(["side", "stage"]):
            g = g.sort_values("block")
            c = STAGE_COL[int(stage) % len(STAGE_COL)]
            ls = "-" if side == "L" else "--"
            ax.plot(g["block"], g["inefficiency"], color=c, ls=ls, lw=1.4)
            ok = g["reliable"].astype(bool).to_numpy()
            x, y = g["block"].to_numpy(), g["inefficiency"].to_numpy()
            ax.plot(x[ok], y[ok], "o", ms=4, color=c, mec="white", mew=0.6)
            ax.plot(x[~ok], y[~ok], "o", ms=4, mfc="white", mec=c, mew=1.0)
        ax.axhline(1, color=AXIS, lw=0.8)
        ax.set_xscale("log")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("block size (paths)")
    for ax in axes[:, 0]:
        ax.set_ylabel("statistical inefficiency")
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    title = "How many consecutive paths count as one independent path"
    if chosen:
        how = "chosen automatically" if mode == "auto" else "fixed with -block"
        title += (f"   (dotted: block {chosen}, {how})" if len(set(per_obs.values()) | {chosen}) == 1
                  else f"   (dotted: block in use, {how})")
    fig.suptitle(title, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _stage_legend(fig, stages)
    _save(fig, in_dir, "chiral_blocking.png")


def plot_drift(in_dir):
    """blocking_quarters.csv: every quantity in each quarter of the run (MC
    order), against its value over the whole stage (dotted).  A trend over
    the quarters is a drift the bootstrap cannot see."""
    q = _read(in_dir, "blocking_quarters")
    if q is None:
        return
    names = list(dict.fromkeys(q["observable"]))
    stages = sorted(q["stage"].unique())
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.9 * nrow), squeeze=False)
    xq = np.arange(1, 5)
    for ax, name in zip(axes.ravel(), names):
        for r in q[q["observable"] == name].itertuples():
            c = STAGE_COL[int(r.stage) % len(STAGE_COL)]
            ls = "-" if r.side == "L" else "--"
            y = [r.quarter1, r.quarter2, r.quarter3, r.quarter4]
            ax.plot(xq, y, color=c, ls=ls, marker="o", ms=4, lw=1.4)
            ax.axhline(r.whole_stage, color=c, ls=":", lw=0.9)
        ax.set_xticks(xq)
        ax.set_xticklabels(["1st", "2nd", "3rd", "4th"])
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("quarter of the run")
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    fig.suptitle("Drift over the run: each quantity per quarter (dotted: whole stage)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _stage_legend(fig, stages)
    _save(fig, in_dir, "chiral_drift.png")


def plot_split_half_profiles(in_dir):
    """
    split_half_profiles.csv: each depth profile's L - D (L + D for the
    mirror-null observables) from the first half of every stage (solid) and
    the second half (dashed), with 95% bands.  A difference that is real
    shows up in both halves at the same depths; filled markers where both
    halves are clear of zero with the same sign.
    """
    rows = _read(in_dir, "split_half_profiles")
    summ = _read(in_dir, "split_half_profiles_summary")
    if rows is None:
        return
    names = list(dict.fromkeys(rows["observable"]))
    ncol = 5
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.8 * nrow), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        g = rows[rows["observable"] == name]
        x = _depth_mid(g["depth"])
        both = g["both_clear_of_zero"].astype(bool).to_numpy()
        for col, ls, c in (("half1", "-", INK), ("half2", "--", MUTED)):
            y, e = g[col].to_numpy(), g[f"se_{col}"].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            ax.fill_between(x[ok], (y - Z95 * e)[ok], (y + Z95 * e)[ok], color=c, alpha=0.10, lw=0)
            ax.plot(x[ok], y[ok], ls=ls, color=c, lw=1.4)
            _markers(ax, x[ok], y[ok], both[ok], c, size=4.5)
        ax.axhline(0, color=AXIS, lw=0.8)
        title = OBS_LABELS.get(name, name)
        if summ is not None:
            r = summ[summ["observable"] == name]
            if len(r) and np.isfinite(r["corr_halves"].iloc[0]):
                r = r.iloc[0]
                title += f"\nr = {r.corr_halves:+.2f} (chance {r.null_95:+.2f}), p = {r.p_perm:.2g}"
        ax.set_title(title, fontsize=8)
        tested = g["tested"].iloc[0] if len(g) else "L-D"
        ax.set_ylabel(tested.replace("-", " \u2212 ").replace("+", " + "), fontsize=8)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("depth |OP| (A)")
    handles = [Line2D([], [], color=INK, ls="-", label="first half"),
               Line2D([], [], color=MUTED, ls="--", label="second half"),
               Line2D([], [], ls="", marker="o", mfc=INK2, mec="white",
                      label="both halves clear of 0, same sign")]
    fig.suptitle("Do the L - D depth profiles replicate between the halves of the run?",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.legend(handles=handles, loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.0))
    _save(fig, in_dir, "chiral_split_half_profiles.png")


#: The orientation maps chiral-compare writes (chiral_interface_analysis.
#: ORIENT_MAPS): name, what the vector is, whether D is shown inverted.
ORIENT = (("axis", "molecular axis (N \u2192 C)", False),
          ("chirality", "chirality vector n", True))


def plot_orient(in_dir):
    """
    orient_maps_depth.npz / orient_jsd_depth.csv: where the molecular axis
    and the chirality vector point in the contacted lipid's glycerol frame
    (phi in its e_x-e_y plane, cos(theta) along e_z; the markers show the
    directions of the C2 substituents).  Top: L, D and L - D at matched
    contact depth; bottom: L - D in each depth bin.  The chirality vector's
    D is shown inverted (-n: phi + 180, cos -> -cos), the map D would match
    if the lipid's handedness played no role; it is what L is tested against.
    """
    path = os.path.join(in_dir, "orient_maps_depth.npz")
    if not os.path.exists(path):
        print("  [!] no orient_maps_depth.npz (tables from an older chiral-compare); "
              "orientation figures skipped")
        return
    with np.load(path) as z:
        allm = {k: z[k] for k in z.files}
    test = _read(in_dir, "orient_jsd_depth")
    sign = _lipid_sign(in_dir)
    for name, what, inverted in ORIENT:
        m = {k[len(name) + 1:]: v for k, v in allm.items() if k.startswith(name + "_")}
        if not m:
            continue
        tested = np.flatnonzero(m["tested"])
        if not tested.size:
            print(f"  [!] {name}: no depth bin has enough contact frames in both L and D; "
                  "orientation figure skipped")
            continue
        t = test[test["vector"] == name] if test is not None and "vector" in test else None
        phi, cos, edges = m["phi_edges"], m["cos_edges"], m["depth_edges"]
        labels = [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])] + [f">{edges[-1]:g}"]
        keys = [f"{a:.2f}-{b:.2f}" for a, b in zip(edges[:-1], edges[1:])] + [f">{edges[-1]:.2f}"]
        dname = "D inverted (\u2212n)" if inverted else "D"

        def row(kind, depth=None):
            if t is None:
                return None
            sel = t["map"] == kind
            if depth is not None:
                sel &= t["depth"].astype(str) == depth
            return t[sel].iloc[0] if sel.any() else None

        fig = plt.figure(figsize=(14.5, 9.0))
        top, bottom = fig.subfigures(2, 1, height_ratios=[1.15, 1.0], hspace=0.04)
        axes = top.subplots(1, 3, sharey=True)
        vmax = np.nanmax([m["L_matched"].max(), m["D_matched"].max()])
        for ax, side, lab in zip(axes[:2], ("L", "D"), ("L", dname)):
            im = ax.pcolormesh(phi, cos, m[f"{side}_matched"].T, cmap=SEQ, vmin=0, vmax=vmax)
            ax.set_title(f"{lab}, depth-matched", loc="left")
            _pose_axes(ax, ylabel=False)
            _substituents(ax, sign)
        axes[0].set_ylabel("cos(theta) along lipid e_z")
        top.colorbar(im, ax=axes[:2], label="fraction of contact frames", shrink=0.9, pad=0.02)
        diff = m["L_minus_D_matched"]
        lim = np.nanmax(np.abs(diff)) or 1.0
        im = axes[2].pcolormesh(phi, cos, diff.T, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim))
        title = f"L \u2212 {dname}, depth-matched"
        r, p0 = row("depth_matched"), row("pooled")
        if r is not None:
            title += f"\nJSD {r.jsd:.4f} (null 95% {r.null_95:.4f}, p = {r.p_perm:.3g})"
        if p0 is not None:
            title += f";  pooled: p = {p0.p_perm:.3g}"
        axes[2].set_title(title, loc="left", fontsize=9)
        _pose_axes(axes[2], ylabel=False)
        _substituents(axes[2], sign)
        top.colorbar(im, ax=axes[2], label="blue: L more, orange: D more", pad=0.02)
        for ax in axes:
            ax.set_xlabel("phi in the lipid's e_x-e_y plane (deg)")
        top.suptitle(f"Where the {what} points in the contacted lipid's glycerol frame"
                     + ("   (D inverted: the map D would have without chiral recognition)"
                        if inverted else ""),
                     x=0.01, ha="left", color=INK, fontsize=10.5)

        n = len(tested)
        axes = np.atleast_1d(bottom.subplots(1, n))
        per = [m["L_depth"][d] - m["D_depth"][d] for d in tested]
        weight = m["match_weight"][tested]
        main = [x for x, wt in zip(per, weight) if wt >= 0.01] or per
        lim = max([np.nanmax(np.abs(x)) for x in main] + [1e-12])
        for k, (ax, d, x) in enumerate(zip(axes, tested, per)):
            im = ax.pcolormesh(phi, cos, x.T, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim))
            _pose_axes(ax, ylabel=False)
            _substituents(ax, sign, labels=False)
            r = row("depth", keys[d])
            ttl = f"{labels[d]} A ({m['match_weight'][d]:.0%})"
            if r is not None:
                ttl += f"\nJSD {r.jsd:.3f}, q = {r.q_bh:.2g}"
            ax.set_title(ttl, fontsize=8.5, loc="left")
            ax.set_xticks([-180, 0, 180])
            ax.set_xlabel("phi (deg)")
            if k:
                ax.set_yticklabels([])
        axes[0].set_ylabel("cos(theta)")
        bottom.colorbar(im, ax=list(axes), label=f"L \u2212 {dname} within the depth bin",
                        pad=0.01, shrink=0.85)
        _save(fig, in_dir, f"chiral_orient_{name}.png")


def plot_split_half_orient(in_dir):
    """split_half_orient.csv / split_half_orient_maps.npz: the split-half
    replication of the orientation maps, one figure per vector."""
    path = os.path.join(in_dir, "split_half_orient_maps.npz")
    test = _read(in_dir, "split_half_orient")
    if not os.path.exists(path) or test is None:
        return
    with np.load(path, allow_pickle=False) as z:
        allm = {k: z[k] for k in z.files}
    for name, what, inverted in ORIENT:
        if f"{name}_half1" not in allm:
            continue
        t = test[test["vector"] == name].reset_index(drop=True)
        dname = "D inverted" if inverted else "D"
        _split_half_figure(
            in_dir, allm[f"{name}_half1"], allm[f"{name}_half2"],
            [str(x) for x in allm[f"{name}_labels"]], allm[f"{name}_phi_edges"],
            allm[f"{name}_cos_edges"], t, "cos(theta) along e_z",
            f"Does the L \u2212 {dname} map of the {what} replicate?  First and second half "
            "of every stage (blue: L more, orange: D more)",
            f"chiral_split_half_orient_{name}.png")


def plot_reorientation(in_dir):
    """
    reorientation.npz / .csv: how fast the molecular axis, the chirality
    vector and the position around the lipid C2 turn relative to the
    contacted lipid.  Top: C(t) = <u(0) . u(t)> over all contact depths, L
    and D with 95% bands.  Bottom: the correlation time (integral of C to the
    longest lag) per contact depth, filled where L and D differ (q < 0.05).
    """
    path = os.path.join(in_dir, "reorientation.npz")
    tab = _read(in_dir, "reorientation")
    if not os.path.exists(path) or tab is None:
        return
    with np.load(path) as z:
        m = {k: z[k] for k in z.files}
    names = [str(v) for v in m["vectors"]]
    depths = [str(d) for d in m["depths"]]
    lag = m["lag_ps"]
    if not np.isfinite(m["C_L"][:, -1]).any():
        print("  [!] no contact pairs for the reorientation figure; skipped")
        return
    fig, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 7.0), squeeze=False)
    titles = {"axis": "molecular axis", "chirality": "chirality vector",
              "position": "position around C2"}
    for k, name in enumerate(names):
        ax = axes[0, k]
        for side in ("L", "D"):
            c, se = m[f"C_{side}"][k, -1], m[f"C_{side}se"][k, -1]
            ok = np.isfinite(c)
            ax.fill_between(lag[ok], (c - Z95 * se)[ok], (c + Z95 * se)[ok], color=COL[side],
                            alpha=0.15, lw=0)
            ax.plot(lag[ok], c[ok], "o-", ms=3.5, color=COL[side], label=side)
        ax.set_xscale("log")
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title(f"{titles.get(name, name)}, all contact depths", loc="left")
        ax.set_xlabel("lag (ps)")
        ax.legend(loc="upper right")
        if k == 0:
            ax.set_ylabel("C(t) = <u(0) . u(t)>")
        ax = axes[1, k]
        tq = tab[tab["quantity"] == "tau_ps"] if "quantity" in tab else tab
        t = tq[tq["vector"] == name].set_index("depth").reindex(depths)
        x = np.arange(len(depths))
        sig = (t["q_bh"] < 0.05).to_numpy()
        for side, dx in (("L", -0.12), ("D", 0.12)):
            y = t[side].to_numpy()
            se = t[f"{side}_se"].to_numpy()
            ok = np.isfinite(y) & np.r_[m["dense"]]
            ax.errorbar(x[ok] + dx, y[ok], yerr=Z95 * se[ok], fmt="none", ecolor=COL[side], lw=1)
            _markers(ax, x[ok] + dx, y[ok], sig[ok], COL[side])
        ax.set_xticks(x)
        ax.set_xticklabels([d.replace(".00", "").replace(".50", ".5") for d in depths],
                           rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("contact depth |OP| (A)")
        ax.grid(axis="x", visible=False)
        if k == 0:
            ax.set_ylabel(f"correlation time (ps, to {lag[-1]:g} ps)")
    fig.suptitle("How fast the proline turns relative to the lipid it touches (frame pairs in "
                 "contact with the same lipid; 95% CI; filled: L vs D q < 0.05)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, in_dir, "chiral_reorientation.png")


def plot_barrier(in_dir):
    """
    barrier.csv: every depth profile pooled over the depth bins that carry
    the resistance to crossing, as the L - D difference (L + D for the
    pseudovector quantities) in units of its own bootstrap standard error,
    with the 95% interval (+-1.96), for all paths and the crossing paths.
    """
    t = _read(in_dir, "barrier")
    if t is None:
        return
    sets = list(dict.fromkeys(t["paths"]))
    names = list(dict.fromkeys(t["observable"]))
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(names) + 1.8))
    y = np.arange(len(names))[::-1]
    for k, (label, col, dy) in enumerate(zip(sets, (INK, COL["L"]), (0.15, -0.15))):
        g = t[t["paths"] == label].set_index("observable").reindex(names)
        se = np.sqrt(g["L_se"] ** 2 + g["D_se"] ** 2)
        zval = (g["tested_value"] / se).to_numpy()
        sig = (g["q_bh"] < 0.05).to_numpy()
        ax.errorbar(zval, y + dy, xerr=Z95, fmt="none", ecolor=col, lw=1, alpha=0.6)
        ax.plot(zval[sig], (y + dy)[sig], "o", color=col, ms=6, label=f"{label}, q < 0.05")
        ax.plot(zval[~sig], (y + dy)[~sig], "o", mfc="white", mec=col, ms=6, label=label)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([("L + D: " if t[t["observable"] == n]["tested"].iloc[0] == "L+D" else "")
                        + OBS_LABELS.get(n, n).replace("\n", " ") for n in names], fontsize=8)
    ax.set_xlabel("difference / its standard error (L > D to the right)")
    ax.legend(loc="lower right", fontsize=7)
    r = t.iloc[0]
    ax.set_title(f"At the barrier: depth bins {r.barrier_depths} A, "
                 f"{r.resistance_covered:.0%} of the resistance", loc="left", fontsize=9)
    fig.tight_layout()
    _save(fig, in_dir, "chiral_barrier.png")


def plot_orientation_free_energy(in_dir):
    """
    orientation_free_energy.npz: G(z, cos theta) for L and D (an orientation
    with no preference sits at G(z)), and ddG = G_L - G_D (D mirrored for the
    chirality vector), dots where q < 0.05.  Relative to bulk water; rows
    with too few frames are blank.
    """
    path = os.path.join(in_dir, "orientation_free_energy.npz")
    if not os.path.exists(path):
        return
    with np.load(path) as z:
        m = {k: z[k] for k in z.files}
    if "depth_edges" not in m:            # no bulk-water anchor: nothing was computed
        return
    depth = m["depth_edges"]
    nrow = len(depth) - 1
    if "rows_used" in m:
        nrow = min(nrow, int(np.asarray(m["rows_used"], bool)[:nrow].sum()))
    depth = depth[:nrow + 1]
    stage = [e for e in _stage_edges(in_dir) if e < depth[-1]]
    for name in ("cos_axis_n", "cos_chi_n"):
        if f"G_L_{name}" not in m:
            continue
        edges, mirror = m[f"{name}_edges"], bool(m[f"{name}_mirror"])
        gL, gD = m[f"G_L_{name}"][:nrow], m[f"G_D_{name}"][:nrow]
        dd, qq = m[f"ddG_{name}"][:nrow], m[f"q_{name}"][:nrow]
        if not np.isfinite(gL).any():
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
        lo = np.nanmin([np.nanmin(gL), np.nanmin(gD)])
        hi = np.nanpercentile(np.r_[gL[np.isfinite(gL)], gD[np.isfinite(gD)]], 98)
        for ax, h, lab in zip(axes[:2], (gL, gD), ("L", "D")):
            ax.set_facecolor(NEUTRAL)
            im = ax.pcolormesh(edges, depth, h, cmap=SEQ.reversed(), vmin=lo, vmax=hi)
            ax.set_title(f"G, {lab}", loc="left")
        fig.colorbar(im, ax=axes[:2], label="kcal/mol relative to bulk water", pad=0.02)
        lim = (np.nanmax(np.abs(dd)) if np.isfinite(dd).any() else 0.0) or 1.0
        axes[2].set_facecolor(NEUTRAL)
        im = axes[2].pcolormesh(edges, depth, dd, cmap=DIV.reversed(), norm=TwoSlopeNorm(0, -lim, lim))
        rows, cols = np.nonzero(np.nan_to_num(qq, nan=1.0) < 0.05)
        cx = 0.5 * (edges[cols] + edges[cols + 1])
        cy = 0.5 * (depth[rows] + depth[rows + 1])
        axes[2].plot(cx, cy, "o", ms=3, mfc=INK, mec="white", mew=0.5)
        axes[2].set_title("\u0394\u0394G = G_L \u2212 G_D" + (" mirrored" if mirror else "")
                          + "  (dots: q < 0.05)", loc="left")
        fig.colorbar(im, ax=axes[2], label="blue: cheaper for L, orange: cheaper for D", pad=0.02)
        for ax in axes:
            ax.set_xlabel(_hist_label(name))
            ax.grid(False)
            for e in stage:
                ax.axhline(e, color=MUTED, lw=0.8, ls=":")
        axes[0].set_ylabel("depth |OP| (A)")
        fig.suptitle(f"Free energy of each depth and orientation: {_hist_label(name)}   "
                     "(an orientation with no preference sits at G(depth))",
                     x=0.01, ha="left", color=INK)
        _save(fig, in_dir, f"chiral_orient_fe_{name}.png")


def plot_glycerol_dihedrals(in_dir):
    """
    glycerol_dihedrals.npz: the distribution of each glycerol dihedral (rows)
    for the lipid the permeant touches and for all lipids (columns), L and D
    with 95% bands, dots where L and D differ (q < 0.05).  The shading marks
    the gauche-, trans and gauche+ ranges glycerol_conformers sums over.
    """
    path = os.path.join(in_dir, "glycerol_dihedrals.npz")
    if not os.path.exists(path):
        return
    with np.load(path) as z:
        m = {k: z[k] for k in z.files}
    edges = m["edges"]
    x = 0.5 * (edges[1:] + edges[:-1])
    lipids = [str(v) for v in m["lipids"]]
    names = [str(v) for v in m["dihedrals"]]
    titles = {"contact": "lipid the proline touches", "bulk": "all lipids"}
    fig, axes = plt.subplots(len(names), len(lipids), figsize=(11.5, 3.3 * len(names)),
                             sharex=True, squeeze=False)
    for r, name in enumerate(names):
        for c, lip in enumerate(lipids):
            ax = axes[r, c]
            for lo, hi, lab in ((-180, -120, "t"), (-120, 0, "g\u2212"), (0, 120, "g+"),
                                (120, 180, "t")):
                ax.axvspan(lo, hi, color=NEUTRAL if lab == "t" else "white", zorder=0)
                ax.text(0.5 * (lo + hi), 1.0, lab, transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=8, color=MUTED)
            sig = np.nan_to_num(m["q"][c, r], nan=1.0) < 0.05
            for side in ("L", "D"):
                y, se = 100 * m[side][c, r], 100 * m[f"{side}se"][c, r]
                if np.isfinite(y).any():
                    _hist_curve(ax, x, y, se, sig, COL[side], side)
            ax.set_xlim(-180, 180)
            ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
            ax.set_ylim(bottom=0)
            ax.set_title(f"{name}, {titles.get(lip, lip)}", loc="left", fontsize=9, pad=14)
            if c == 0:
                ax.set_ylabel("% of frames per 10\u00b0")
            if r == len(names) - 1:
                ax.set_xlabel("dihedral angle (\u00b0)")
            if r == 0 and c == len(lipids) - 1:
                ax.legend(loc="upper right")
    nL, nD = m["contact_frames"]
    fig.suptitle(f"Glycerol dihedrals of the lipids (bands: 95% CI; dots: L vs D q < 0.05; "
                 f"touched lipid from {int(nL):,} L and {int(nD):,} D contact frames)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, in_dir, "chiral_glycerol_dihedrals.png")


#: Histogram windows or depth rows with fewer frames than this (L or D) are
#: left blank, as chiral-compare leaves them untested.
HIST_MIN_FRAMES = 200


def _hist_npz(in_dir):
    path = os.path.join(in_dir, "histograms.npz")
    if not os.path.exists(path):
        print("  [!] no histograms.npz (tables from an older chiral-compare); "
              "histogram figures skipped")
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _hist_label(name):
    return OBS_LABELS.get(name, name).replace("\n", " ")


def _window_title(lab):
    a, _, b = str(lab).partition("-")
    return f"{float(a):g}\u2013{float(b):g} \u00c5"


def _hist_curve(ax, x, y, se, sig, color, label, ls="-"):
    """A histogram as a line with its 95% band; filled dots on significant bins."""
    ok = np.isfinite(y)
    x, y, sig = x[ok], y[ok], np.asarray(sig, bool)[ok]
    if se is not None:
        se = np.asarray(se)[ok]
        ax.fill_between(x, y - Z95 * se, y + Z95 * se, color=color, alpha=0.15, lw=0)
    ax.plot(x, y, color=color, ls=ls, label=label, zorder=3, lw=1.4)
    ax.plot(x[sig], y[sig], "o", ms=4.5, color=color, mec="white", mew=0.6, zorder=4)


def plot_hist1d(in_dir):
    """
    histograms.npz: per observable, the distribution of its per-frame values
    in each depth window, L and D with 95% bands (bootstrap, as the profiles),
    as the fraction of the window's frames in each value bin.  Dots: bins
    where L and D differ with q < 0.05.  For cos_chi_n and chi_lz, where D
    would be the mirror image of L without chiral recognition, the dashed
    line is D mirrored (value -> -value) and the dots test L against it.
    """
    m = _hist_npz(in_dir)
    if m is None:
        return
    wins = [str(w) for w in m["windows"]]
    for name in (str(n) for n in m["observables"]):
        edges = m[f"{name}_edges"]
        x = 0.5 * (edges[1:] + edges[:-1])
        mirror = bool(m[f"{name}_mirror"])
        ncol = 4
        nrow = int(np.ceil(len(wins) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.8 * nrow),
                                 sharex=True, squeeze=False)
        for w, ax in enumerate(axes.ravel()[:len(wins)]):
            nL, nD = m[f"nL_{name}_1d"][w], m[f"nD_{name}_1d"][w]
            title = _window_title(wins[w])
            if min(nL, nD) < HIST_MIN_FRAMES:
                ax.set_title(f"{title}: too few frames", loc="left", color=MUTED)
                continue
            sig = np.nan_to_num(m[f"q_{name}_1d"][w], nan=1.0) < 0.05
            L, D = m[f"L_{name}_1d"][w], m[f"D_{name}_1d"][w]
            Lse, Dse = m[f"Lse_{name}_1d"][w], m[f"Dse_{name}_1d"][w]
            _hist_curve(ax, x, L, Lse, sig, COL["L"], "L")
            if mirror:
                _hist_curve(ax, x, D, Dse, np.zeros_like(sig), COL["D"], "D")
                _hist_curve(ax, x, D[::-1], None, sig, COL["D"], "D mirrored", ls="--")
            else:
                _hist_curve(ax, x, D, Dse, sig, COL["D"], "D")
            ax.set_ylim(bottom=0)
            ax.set_title(f"{title}   frames L {int(nL):,}, D {int(nD):,}", loc="left",
                         fontsize=8.5)
        for ax in axes.ravel()[len(wins):]:
            ax.set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("fraction of frames")
        for ax in axes.ravel()[max(0, len(wins) - ncol):len(wins)]:
            ax.set_xlabel(_hist_label(name))
            ax.tick_params(labelbottom=True)
        handles = [Line2D([], [], color=COL["L"], label="L"),
                   Line2D([], [], color=COL["D"], label="D")]
        if mirror:
            handles.append(Line2D([], [], color=COL["D"], ls="--",
                                  label="D mirrored (value \u2192 \u2212value)"))
        handles.append(Line2D([], [], ls="", marker="o", mfc=INK2, mec="white",
                              label="q < 0.05 (L vs " + ("mirrored D)" if mirror else "D)")))
        fig.legend(handles=handles, loc="upper right", ncol=len(handles),
                   bbox_to_anchor=(1.0, 1.0))
        fig.suptitle(f"Distribution of {_hist_label(name)} by depth (bands: 95% CI)",
                     x=0.01, ha="left", color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        _save(fig, in_dir, os.path.join(HIST_DIR, f"chiral_hist1d_{name}.png"))


def plot_hist2d(in_dir):
    """
    histograms.npz: per observable, depth x value maps for L and D, each
    depth row normalised to its own frames (so every depth is visible,
    however often it was sampled), and L - D; for cos_chi_n and chi_lz,
    L - D mirrored.  Grey rows: too few frames.  The cells are not tested;
    the 1D figures carry the tests.
    """
    m = _hist_npz(in_dir)
    if m is None:
        return
    # The rows chiral-compare covered (-hist-max-depth); the open bulk-water
    # row has no upper edge and is never drawn.
    nrow = len(m["depth_edges"]) - 1
    if "rows_used" in m:
        nrow = min(nrow, int(np.asarray(m["rows_used"], bool)[:nrow].sum()))
    if nrow < 1:
        print("  [!] no depth bin below the histogram depth limit; 2D histograms skipped")
        return
    depth = m["depth_edges"][:nrow + 1]
    stage = [e for e in _stage_edges(in_dir) if e < depth[-1]]
    for name in (str(n) for n in m["observables"]):
        edges = m[f"{name}_edges"]
        mirror = bool(m[f"{name}_mirror"])
        L2, D2 = m[f"L_{name}_2d"][:nrow].copy(), m[f"D_{name}_2d"][:nrow].copy()
        thin = (m[f"nL_{name}_2d"][:nrow] < HIST_MIN_FRAMES) | (m[f"nD_{name}_2d"][:nrow] < HIST_MIN_FRAMES)
        L2[thin], D2[thin] = np.nan, np.nan
        if not np.isfinite(L2).any():
            print(f"  [!] {name}: no depth with enough frames; 2D histogram skipped")
            continue
        diff = L2 - (D2[:, ::-1] if mirror else D2)
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
        vmax = np.nanmax([np.nanmax(L2), np.nanmax(D2)])
        for ax, side, h in zip(axes[:2], ("L", "D"), (L2, D2)):
            ax.set_facecolor(NEUTRAL)
            im = ax.pcolormesh(edges, depth, h, cmap=SEQ, vmin=0, vmax=vmax)
            ax.set_title(side, loc="left")
        fig.colorbar(im, ax=axes[:2], label="fraction of the depth's frames", pad=0.02)
        lim = np.nanmax(np.abs(diff)) or 1.0
        axes[2].set_facecolor(NEUTRAL)
        im = axes[2].pcolormesh(edges, depth, diff, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim))
        axes[2].set_title("L \u2212 D mirrored" if mirror else "L \u2212 D", loc="left")
        fig.colorbar(im, ax=axes[2], label="blue: L more, orange: D more", pad=0.02)
        for ax in axes:
            ax.set_xlabel(_hist_label(name))
            ax.grid(False)
            for e in stage:
                ax.axhline(e, color=MUTED, lw=0.8, ls=":")
        axes[0].set_ylabel("depth |OP| (A)")
        fig.suptitle(f"Distribution of {_hist_label(name)} at each depth   "
                     "(rows normalised; grey: too few frames; dotted: stage boundaries)",
                     x=0.01, ha="left", color=INK)
        _save(fig, in_dir, os.path.join(HIST_DIR, f"chiral_hist2d_{name}.png"))


def plot_all(in_dir):
    """Draw every figure the tables in `in_dir` allow; missing tables are skipped."""
    _style()
    for fn in (plot_free_energy, plot_profiles, plot_lifetimes, plot_survival,
               plot_contacts, plot_pose, plot_pose_depth, plot_split_half,
               plot_split_half_profiles, plot_blocking, plot_drift,
               plot_orient, plot_split_half_orient, plot_reorientation, plot_barrier,
               plot_glycerol_dihedrals,
               plot_orientation_free_energy, plot_hist1d, plot_hist2d):
        fn(in_dir)


def chiral_plot(
    in_dir: Annotated[str, typer.Option("-in-dir", help="Folder chiral-compare wrote (its -out-dir); the figures are written next to the tables", rich_help_panel=panels.INPUT)] = "chiral_interface",
    overw: Annotated[bool, typer.Option("-O", help="Force overwriting of existing figures", rich_help_panel=panels.OUTPUT)] = False,
):
    """Draw the figures of a chiral-compare result: free energy, depth profiles,
    lifetimes, H-bond partners, pose maps and the 1D/2D histograms of every
    observable (in a histograms/ subfolder), L vs D.

    Needs only the tables, so it runs wherever the comparison folder is.
    chiral-compare already draws them; this redraws them, e.g. after copying
    the folder off the server.  The reactive/ and unreactive/ subfolders, and
    their per-stage folders, are drawn too when they are there.
    """
    if not os.path.isdir(in_dir):
        raise typer.BadParameter(f"-in-dir: {in_dir} is not a folder")
    folders = result_folders(in_dir)
    for folder in folders:
        for name in PLOT_OUTPUTS:
            _check_overwrite(os.path.join(folder, name), overw)
    for folder in folders:
        plot_all(folder)
