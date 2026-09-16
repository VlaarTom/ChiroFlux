"""Permeability from the inhomogeneous solubility-diffusion model.

Marrink & Berendsen (J. Phys. Chem. 100 (1996) 16729) write the permeability of
a bilayer as one integral of a local resistance across the membrane normal,

    1/P = int R(z)  dz = int exp(beta dG(z)) / D(z)  dz

with dG(z) the free energy of the permeant relative to bulk water and D(z) its
local diffusion coefficient. The integrand is a *resistance density*: the whole
profile matters, but because dG enters exponentially the answer is dominated by
the few angstroms around the free-energy maximum, and that is also where D is
worst determined. The resistance profile is therefore the more informative
output of the two, and it is written and plotted alongside the scalar.

Inputs are the files the rest of the package already writes:

* a run directory per stage, each holding the standard ``wham/`` folder:
  ``histo_probability.txt`` (bin centre, weighted count, with lA/lB in its
  header), ``Pcross.txt`` (total crossing probability in its last row) and
  ``runav_rate.txt`` (the path counter in its last row, column 0). Everything
  needed to reference the stages to each other is in there, so nothing has to
  be typed in by hand.
* ``diffusion_aggregate.csv`` from ``membrane_spatial`` - ``slab_center``
  (already the order-parameter axis) with ``D_z`` in A^2/ps, plus its ``_err``
  and ``_neff`` jackknife columns. ``D_z_lag2`` is the same estimate at twice
  the lag: if it disagrees with ``D_z``, the motion is not yet diffusive at the
  frame spacing.

A single staged run covers only part of the membrane, so the three are combined.
They cannot simply be spliced: each stage's WHAM histogram is *conditional* on
having started at that stage's own state A, so putting stage k on the scale of
stage 1 costs the probability of ever getting there. That is a multiplicative
factor on the **histogram**, which is where it is applied here - the stages are
rescaled, cut at their own lA and summed, and the free energy is taken as
``-ln`` of the sum at the end. Doing it on the histogram rather than on each
stage's free energy is what lets the counts simply add. The factor is the one
`cv_histograms` uses for its escape correction, chained over the junctions; see
`stage_factors`.

Conditioning on the start also biases each stage on its own: a forward
histogram climbs to ``-ln(P_tot)`` by its far interface, so the stages do not
join up until every region also has its backward histogram. In a symmetric
bilayer those are the mirror images of the same runs, so the combined
histogram is added to its own reflection about z = 0 - see
`combine_stage_histograms`. That is the default, and it is what reproduces the
free-energy profile these simulations are usually quoted with.

A partial profile yields a partial resistance, which *underestimates* 1/P and
so overestimates P - the report says so rather than quietly returning a number.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import matplotlib
import numpy as np
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import panels
from .pathdata import _check_overwrite

#: 1 A/ps in cm/s. P comes out in A/ps because z is in A and D in A^2/ps.
ANGSTROM_PER_PS_TO_CM_PER_S = 1.0e4

#: Interfaces are written to a couple of decimals, so exact equality is too
#: strict when looking a mirrored stage up by its lA/lB.
INTERFACE_TOL = 1.0e-4


def _split_list(value):
    """Split a comma- or space-separated option into tokens."""
    if not value:
        return []
    return [tok for tok in value.replace(",", " ").split() if tok]


def load_free_energy(path):
    """(z, beta_dG) from a wham histo_free_energy.txt, unsampled bins dropped.

    The file marks unsampled bins with ``inf``; they are not zero-probability
    regions of a profile, they are places the simulation never went, so they
    are removed rather than carried through the exponential.
    """
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path}: expected two columns (position, free energy).")
    finite = np.isfinite(data[:, 1])
    if finite.sum() < 2:
        raise ValueError(f"{path}: fewer than two sampled bins.")
    z, g = data[finite, 0], data[finite, 1]
    order = np.argsort(z)
    return z[order], g[order]


@dataclass(frozen=True)
class Stage:
    """One staged run: its weighted OP histogram and what references it.

    `counts` is the run's own weighted histogram, conditional on having
    started in its state A at `lam_a`; `pcross` is the total crossing
    probability of reaching `lam_b` from there, and `n_paths` the total number
    of accepted paths the histogram was accumulated over.
    """

    name: str
    centers: np.ndarray
    counts: np.ndarray
    lam_a: float
    lam_b: float
    pcross: float
    n_paths: float


def read_wham_header(path):
    """The ``key=value`` pairs wham writes on the first line, as floats.

    ``# lm1=-35, lA=-25.0, lB=-13.5, first_bin_after_lA_index=60, ...``
    """
    with open(path) as handle:
        first = handle.readline()
    if not first.lstrip().startswith("#"):
        return {}
    pairs = re.findall(
        r"([A-Za-z_]\w*)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", first
    )
    return {key: float(value) for key, value in pairs}


def _last_row(path):
    """The last data row of a whitespace table, as a 1-D array."""
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        return data
    if data.size == 0:
        raise ValueError(f"{path}: no data rows.")
    return data[-1]


def load_stage(run_dir, wham_subdir="wham", pcross_col=3):
    """Read one stage's histogram and the numbers that reference it.

    Everything comes from the run's own ``wham/`` folder, which is assumed to
    have the standard layout:

    * ``histo_probability.txt`` - bin centre and weighted count, with the
      stage's own ``lA``/``lB`` on the header line.
    * ``Pcross.txt`` - the total crossing probability is the last row;
      `pcross_col` picks the column (3 = P-wham2, the one the free-energy
      notebooks use; 1 = P-wham).
    * ``runav_rate.txt`` - the path counter is column 0 of the last row. The
      histogram is a sum over those paths, so the count is what puts two runs
      of different length on the same footing.
    """
    wham = Path(run_dir) / wham_subdir
    histo = wham / "histo_probability.txt"
    if not histo.is_file():
        raise ValueError(
            f"{histo} does not exist. -runs takes the run directories "
            f"themselves; each must contain a '{wham_subdir}' folder."
        )

    header = read_wham_header(histo)
    missing = [key for key in ("lA", "lB") if key not in header]
    if missing:
        raise ValueError(
            f"{histo}: no {', '.join(missing)} on the header line; the stage "
            "cannot be placed without its own interfaces."
        )

    data = np.loadtxt(histo, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{histo}: expected two columns (bin centre, count).")

    row = _last_row(wham / "Pcross.txt")
    if row.size <= pcross_col:
        raise ValueError(
            f"{wham / 'Pcross.txt'}: no column {pcross_col} "
            f"(the file has {row.size})."
        )
    pcross = float(row[pcross_col])

    rate = wham / "runav_rate.txt"
    n_paths = float(_last_row(rate)[0]) if rate.is_file() else np.nan

    return Stage(
        name=os.path.basename(os.path.normpath(str(run_dir))) or str(run_dir),
        centers=data[:, 0],
        counts=data[:, 1],
        lam_a=header["lA"],
        lam_b=header["lB"],
        pcross=pcross,
        n_paths=n_paths,
    )


def mirror_back_pcross(stages):
    """The backward escape probability at each junction, by mirror symmetry.

    Stage k+1's histogram is conditional on starting at its own state A, and
    referencing it needs the probability of falling back *out* of that state
    the way the permeant came in - across the interval stage k covers, but in
    reverse. That run does not exist, but in a symmetric bilayer its mirror
    image does: reversing [lA_k, lB_k] is the same as traversing
    [-lB_k, -lA_k] forwards, so the answer is the total crossing probability
    of whichever stage spans that. With entry/internal/escape this recovers
    exactly the substitutions made by hand in `cv_histograms` - the backward
    escape out of the entry run's state B is the escape run, and out of the
    internal run's state B is the internal run itself.

    Raises if the mirrored stage is not among `stages`; pass the values with
    ``-pcross-back`` when the staging is not symmetric.
    """
    backs = []
    for index, stage in enumerate(stages[:-1]):
        want_a, want_b = -stage.lam_b, -stage.lam_a
        match = [
            other for other in stages
            if abs(other.lam_a - want_a) < INTERFACE_TOL
            and abs(other.lam_b - want_b) < INTERFACE_TOL
        ]
        if not match:
            raise ValueError(
                f"junction {index}: the backward escape out of "
                f"'{stages[index + 1].name}' spans [{stage.lam_a}, "
                f"{stage.lam_b}] in reverse, whose mirror image "
                f"[{want_a}, {want_b}] is not among the stages given. Pass the "
                "crossing probabilities with -pcross-back."
            )
        backs.append(match[0].pcross)
    return backs


def stage_factors(stages, pcross_back=None, path_norm=False):
    """The factor each stage's histogram is multiplied by, relative to stage 1.

    A staged run's WHAM histogram is a *conditional* density: stage k is
    referenced to its own state A, so it says where the permeant is **given
    that it started there**. Putting it on stage 1's scale costs the
    probability of ever getting there, which for one junction is

        factor_{k+1} / factor_k = (N_k / N_{k+1}) * (P_k^+ / P_{k+1}^-)

    with P_k^+ the total crossing probability of stage k (its own
    ``Pcross.txt``), P_{k+1}^- the probability of escaping backward out of
    stage k+1's state A (see `mirror_back_pcross`), and N the total number of
    paths each histogram was accumulated over. Calling the three stages A
    (entry), M (internal) and B (escape) - each letter naming a run, so P_A is
    the entry run's own crossing probability - the chain is

        factor_M = N_A * (P_A / P_M_min) / N_M
        factor_B = N_A * (P_A / P_M_min) * (P_M / P_B_min) / N_B

    which is the escape-correction factor `cv_histograms` applies, written
    there with C and D for M and B.

    It cannot be recovered by matching the stages where they meet. An overlap
    fit measures how two conditional densities happen to line up locally, not
    how improbable it was to reach the second stage at all, and the difference
    between the two is the whole barrier the later stage sits behind.

    `path_norm` includes the N ratio, for histograms not normalised with the
    WHAM weights.
    """
    if not stages:
        raise ValueError("No stages given.")
    backs = (list(pcross_back) if pcross_back is not None
             else mirror_back_pcross(stages))
    if len(backs) != len(stages) - 1:
        raise ValueError(
            f"{len(backs)} backward crossing probabilities for "
            f"{len(stages)} stages; there must be one per junction."
        )

    factors = [1.0]
    for index, (prev, nxt) in enumerate(zip(stages, stages[1:])):
        forward, back = float(prev.pcross), float(backs[index])
        for value, what in ((forward, f"'{prev.name}' forward"),
                            (back, f"junction {index} backward")):
            if not 0.0 < value <= 1.0:
                raise ValueError(
                    f"{what} crossing probability is {value}; it must be in "
                    "(0, 1]."
                )
        step = forward / back
        if path_norm:
            if not (np.isfinite(prev.n_paths) and np.isfinite(nxt.n_paths)
                    and prev.n_paths > 0 and nxt.n_paths > 0):
                raise ValueError(
                    f"junction {index}: no usable path count for "
                    f"'{prev.name}'/'{nxt.name}' (runav_rate.txt). Pass "
                    "-no-path-norm if the histograms are already per path."
                )
            step *= prev.n_paths / nxt.n_paths
        factors.append(factors[-1] * step)
    return factors


def combine_stage_histograms(stages, factors, symmetrize=False):
    """Rescale, cut and add the staged histograms into one.

    Every stage is multiplied by its factor and then cut below its own lA:
    past the first stage, the region before a run's state A is the previous
    stage's to describe, and it is only in this run's histogram because paths
    dip back over the interface so 0^- ensembles are cut off. The first stage
    keeps everything, since below its lA is the bulk the whole profile is referenced to.

    `symmetrize` then adds the whole thing to its own mirror image about
    z = 0. Each stage's histogram is conditional on paths that *started* at
    its state A, which biases it towards paths in one direction. On its own it
    climbs to -ln(P_tot) by its state, so the forward stages alone do not join up.
    In a symmetric bilayer the mirror of a stage is the reverse-direction ensemble
    of the region opposite it, so mirroring the sum hands every region its backward
    histogram: with the stages labelled A (entry), M (internal) and B (escape),
    A mirrored is the B region run backwards, M mirrored is M itself reversed, and
    B mirrored is the A region run backwards. Forward plus backward is the unbiased
    density.

    The result is a single histogram on the shared bin grid, so the free
    energy taken from it afterwards has no jumps at the junctions.
    """
    stitched = stitch_stage_histograms(stages, factors, symmetrize)
    return stitched.centers, stitched.total


@dataclass
class Stitching:
    """Every intermediate of the stitched histogram, kept for inspection.

    `parts[k]` is stage k after its factor and its cut below lA - exactly what
    it contributes. `forward` is their sum and `mirror` that sum reflected
    about z = 0 (None when the bin grid is not symmetric); `total` is what the
    free energy is taken from.
    """

    stages: list
    factors: list
    centers: np.ndarray
    parts: list
    forward: np.ndarray
    mirror: Optional[np.ndarray]
    symmetrize: bool

    @property
    def total(self):
        return self.forward + self.mirror if self.symmetrize else self.forward


def stitch_stage_histograms(stages, factors, symmetrize=False):
    """The stitched histogram and all its parts; see `combine_stage_histograms`."""
    if not stages:
        raise ValueError("No stages given.")
    centers = np.asarray(stages[0].centers, dtype=float)

    parts = []
    for index, (stage, factor) in enumerate(zip(stages, factors)):
        if stage.centers.shape != centers.shape or not np.allclose(
            stage.centers, centers
        ):
            raise ValueError(
                f"'{stage.name}' is on a different bin grid from "
                f"'{stages[0].name}'. The stages must share one histogram grid "
                "to be added."
            )
        counts = np.asarray(stage.counts, dtype=float) * factor
        if index > 0:
            counts = np.where(centers > stage.lam_a, counts, 0.0)
        parts.append(counts)
    forward = np.sum(parts, axis=0)

    grid_is_symmetric = np.allclose(centers, -centers[::-1])
    if symmetrize and not grid_is_symmetric:
        raise ValueError(
            "the bin grid is not symmetric about 0, so it cannot be "
            "mirrored; drop -symmetrize."
        )
    mirror = forward[::-1].copy() if grid_is_symmetric else None

    return Stitching(stages=list(stages), factors=list(factors),
                     centers=centers, parts=parts, forward=forward,
                     mirror=mirror, symmetrize=symmetrize)


def junction_steps(stitched):
    """The free-energy step across each stage's lA, forward-only and total.

    Taken between the nearest sampled bins either side of the interface, so a
    well-stitched profile shows only its local slope times a bin width there,
    while a junction that does not close shows up as a jump. NaN where either
    side has no sampled bin adjacent to the interface.
    """
    centers = stitched.centers
    width = float(np.median(np.diff(centers))) if centers.size > 1 else np.nan

    def step(counts, x):
        with np.errstate(divide="ignore", invalid="ignore"):
            free = np.where(counts > 0, -np.log(counts), np.nan)
        left = np.flatnonzero((centers <= x) & np.isfinite(free))
        right = np.flatnonzero((centers > x) & np.isfinite(free))
        if not left.size or not right.size:
            return np.nan
        i, j = left[-1], right[0]
        if centers[j] - centers[i] > 3.0 * width:
            return np.nan
        return float(free[j] - free[i])

    return [
        {"stage": stage.name, "lam_a": stage.lam_a,
         "forward": step(stitched.forward, stage.lam_a),
         "total": step(stitched.total, stage.lam_a)}
        for stage in stitched.stages[1:]
    ]


def free_energy_from_counts(centers, counts):
    """beta*dG = -ln(counts), with the empty bins dropped.

    A zero bin is a place nothing was sampled, not a place of zero
    probability, so it is removed rather than carried into the exponential as
    an infinite barrier.
    """
    counts = np.asarray(counts, dtype=float)
    keep = np.isfinite(counts) & (counts > 0)
    if keep.sum() < 2:
        raise ValueError("Fewer than two sampled bins in the combined histogram.")
    z = np.asarray(centers, dtype=float)[keep]
    with np.errstate(divide="ignore"):
        return z, -np.log(counts[keep])


def stitch_runs(run_dirs, wham_subdir="wham", pcross_col=3,
                pcross_back=None, path_norm=False, symmetrize=False):
    """Read, reference and stitch staged run directories; see `Stitching`."""
    pcrosses = []
    stages = [load_stage(d, wham_subdir, pcross_col) for d in run_dirs]
    for stage in stages:
        pcrosses.append(stage.pcross)
        print(f"  {stage.name:<20} lA={stage.lam_a:>7.2f} lB={stage.lam_b:>7.2f} "
              f"P_tot={stage.pcross:.6g} paths={stage.n_paths:.0f}")

    factors = ([1.0] if len(stages) == 1
               else stage_factors(stages, pcross_back, path_norm))
    for stage, factor in zip(stages[1:], factors[1:]):
        print(f"  '{stage.name}' scaled by {factor:.6g} "
              f"({np.log(factor):+.4f} kT) and cut below its lA = {stage.lam_a}")

    stitched = stitch_stage_histograms(stages, factors, symmetrize)
    if symmetrize:
        print("  mirrored about z = 0 and added to itself, which supplies each "
              "region's backward histogram")
    elif len(stages) > 1:
        print("  [warn] -no-symmetrize: the forward histograms are conditional "
              "on starting at their own state A, so the profile will step at "
              "the junctions by roughly ln(P_tot) of each stage.")
    return stitched, pcrosses


def free_energy_from_runs(run_dirs, wham_subdir="wham", pcross_col=3,
                          pcross_back=None, path_norm=False, symmetrize=False):
    """(z, beta*dG) from a list of staged run directories.

    The whole chain: read each stage, reference them to one another on the
    histogram, add them, and take the log once at the end.
    """
    stitched, _ = stitch_runs(run_dirs, wham_subdir, pcross_col, pcross_back,
                           path_norm, symmetrize)
    return free_energy_from_counts(stitched.centers, stitched.total)


def stage_shifts_from_pcross(pcross):
    """Free-energy offset of each stage from the crossing probabilities.

    A staged run's WHAM histogram is a *conditional* density: stage k is
    referenced to its own state A, so it says where the permeant is **given
    that it started at the beginning of stage k**. Putting stage k on the scale
    of stage 1 costs the probability of ever getting there, which is the
    product of the total crossing probabilities of the stages in between:

        P_ref(z) = [prod over j<k of P_tot,j] * P_k(z)

    and because the profile is F = -ln(P), that product becomes an additive
    offset in free energy,

        F_ref(z) = F_k(z) - sum over j<k of ln(P_tot,j)

    which is positive, since every P_tot < 1. This is the same physical
    referencing `cv_histograms.escape_correction_factor` applies to its merged
    OP histogram, written additively because these profiles are already logs.

    It cannot be recovered by matching the stages on their overlap. The overlap
    fit measures how the two conditional densities happen to line up there, not
    how improbable it was to reach the second stage at all, and the difference
    between the two is the whole barrier the later stage sits behind.

    `pcross` holds one total crossing probability per junction - the last row
    of each earlier stage's ``wham/Pcross.txt`` - so N stages need N-1 of them.
    """
    shifts = [0.0]
    total = 0.0
    for index, p in enumerate(pcross):
        p = float(p)
        if not 0.0 < p <= 1.0:
            raise ValueError(
                f"crossing probability {index} is {p}; it must be in (0, 1]."
            )
        total -= np.log(p)
        shifts.append(total)
    return shifts


def join_profiles(profiles, shifts=None, overlap=None):
    """Splice staged free-energy profiles into one continuous beta*dG(z).

    `shifts` is the offset applied to each stage, normally from
    `stage_shifts_from_pcross`. The offset the overlap implies is reported
    alongside it as a cross-check: the two answer different questions, so they
    are not expected to agree, and the gap between them is the free energy the
    later stage is conditioned on having already climbed.

    With `shifts` left None the overlap fit is used instead. That is **not**
    physically referenced - it treats the conditional densities as though they
    shared a normalisation - so it warns, and the permeability that follows is
    not comparable with the RETIS one.
    """
    if not profiles:
        raise ValueError("No free-energy profiles given.")
    if shifts is not None and len(shifts) != len(profiles):
        raise ValueError(
            f"{len(shifts)} shift(s) for {len(profiles)} profile(s); there must "
            "be one per stage (the first is normally 0)."
        )

    z_all = list(profiles[0][0])
    g_all = list(profiles[0][1] + (shifts[0] if shifts is not None else 0.0))

    for index, (z_next, g_next) in enumerate(profiles[1:], start=1):
        lo = max(min(z_all), z_next.min())
        hi = min(max(z_all), z_next.max())
        overlapping = hi > lo

        fitted = np.nan
        if overlapping:
            flo, fhi = (max(lo, hi - overlap), hi) if overlap else (lo, hi)
            z_prev = np.asarray(z_all)
            g_prev = np.asarray(g_all)
            grid = z_prev[(z_prev >= flo) & (z_prev <= fhi)]
            if grid.size == 0:
                grid = np.linspace(flo, fhi, 16)
            fitted = float(np.mean(np.interp(grid, z_prev, g_prev)
                                   - np.interp(grid, z_next, g_next)))

        if shifts is not None:
            shift = shifts[index]
            note = (f"fitted {fitted:+.4f} kT, gap {shift - fitted:+.4f}"
                    if np.isfinite(fitted) else "no overlap to cross-check")
            print(f"  stage {index}: shifted {shift:+.4f} kT from the crossing "
                  f"probabilities ({note})")
        else:
            if not overlapping:
                raise ValueError(
                    f"stage {index} spans [{z_next.min():.3f}, "
                    f"{z_next.max():.3f}], which does not overlap the stages "
                    f"before it ([{min(z_all):.3f}, {max(z_all):.3f}]). Without "
                    "-pcross there is nothing to reference it by."
                )
            shift = fitted
            print(f"  [warn] stage {index}: shifted {shift:+.4f} kT by matching "
                  "the overlap. These are conditional histograms, so this is "
                  "NOT physically referenced - pass -pcross.")

        keep = z_next > max(z_all)
        z_all.extend(z_next[keep])
        g_all.extend(g_next[keep] + shift)

    z = np.asarray(z_all)
    g = np.asarray(g_all)
    order = np.argsort(z)
    return z[order], g[order]


def load_diffusion(path, column="D_z", min_neff=0.0,
                   midplane=0.0, flip=False):
    """(z, D, D_err) from a membrane_spatial diffusion aggregate, on the OP axis.

    `membrane_spatial` bins the permeant on ``membrane_z - permeant_z``, the
    same membrane-relative, sign-flipped coordinate the order parameter uses,
    so its ``slab_center`` needs no transformation at all: the defaults
    ``midplane=0, flip=False`` leave it alone. The mapping
    ``op = sign * (slab_center - midplane)`` is there for a diffusion profile
    that really is on lab-frame z, from somewhere else. Getting it wrong shows
    up immediately as a resistance profile whose peak sits away from the
    free-energy barrier, which is worth a look at the plot before trusting any
    number.

    Bins below `min_neff` effective paths are dropped rather than kept at
    whatever value one or two sojourns produced.
    """
    import pandas as pd

    frame = pd.read_csv(path)
    if "slab_center" not in frame.columns:
        raise ValueError(f"{path}: no 'slab_center' column.")
    if column not in frame.columns:
        raise ValueError(
            f"{path}: no '{column}' column. Available: "
            + ", ".join(c for c in frame.columns if c.startswith("D_"))
        )

    z = frame["slab_center"].to_numpy(dtype=float)
    d = frame[column].to_numpy(dtype=float)
    err_col = f"{column}_err"
    err = (frame[err_col].to_numpy(dtype=float) if err_col in frame.columns
           else np.full_like(d, np.nan))

    keep = np.isfinite(d) & (d > 0)
    neff_col = f"{column}_neff"
    if min_neff > 0 and neff_col in frame.columns:
        keep &= frame[neff_col].to_numpy(dtype=float) >= min_neff

    z, d, err = z[keep], d[keep], err[keep]
    op = (-1.0 if flip else 1.0) * (z - midplane)
    order = np.argsort(op)
    return op[order], d[order], err[order]


def _read_op_and_z(path, op_col, z_col):
    """(order parameter, lab-frame permeant z) from one per-path CV file."""
    with open(path) as handle:
        names = [handle.readline() for _ in range(4)][3].split()
    for col in (op_col, z_col):
        if col not in names:
            raise ValueError(
                f"{path}: no '{col}' column. Available z columns: "
                + ", ".join(n for n in names if n.lower().startswith("z"))
            )
    data = np.loadtxt(path, comments="#", skiprows=4,
                      usecols=(names.index(op_col), names.index(z_col)))
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data[:, 0], data[:, 1]


def midplane_from_cv_files(paths, op_col="OP_Lamb", z_col="z_PRO"):
    """Locate the bilayer midplane on the lab z axis from the CV trajectories.

    The free energy lives on the order parameter and the diffusion slabs on
    lab-frame z, so the two have to be tied together. Rather than assume which
    atoms define the midplane, this measures the relation the simulation
    actually used: every frame reports both the order parameter and the
    permeant's own z, and the order parameter is that z measured from the
    membrane centre, so

        op = +/- (z_lab - midplane)   =>   midplane = z_lab -/+ op

    averaged over frames. The sign comes from the regression slope rather than
    being assumed, so `-flip` is measured here instead of guessed, and the
    slope's magnitude says whether the CV files write z in nm (|slope| ~ 0.1)
    or in Angstrom (~1) - the diffusion slabs are in Angstrom either way.

    Returns (midplane in Angstrom, flip, info).
    """
    ops, zs = [], []
    for path in paths:
        op_values, z_values = _read_op_and_z(path, op_col, z_col)
        ops.append(op_values)
        zs.append(z_values)
    if not ops:
        raise ValueError("No CV trajectory files to take the midplane from.")

    op = np.concatenate(ops)
    z = np.concatenate(zs)
    good = np.isfinite(op) & np.isfinite(z)
    op, z = op[good], z[good]
    if op.size < 2 or np.ptp(op) <= 0:
        raise ValueError(
            "The order parameter does not vary over the frames read; the "
            "midplane cannot be fitted. Pass -midplane as a number."
        )

    slope = float(np.polyfit(op, z, 1)[0])
    magnitude = abs(slope)
    if 0.04 <= magnitude <= 0.4:
        scale, unit = 10.0, "nm"
    elif 0.4 < magnitude <= 4.0:
        scale, unit = 1.0, "A"
    else:
        raise ValueError(
            f"'{z_col}' changes by {slope:.4g} per unit of '{op_col}', which is "
            "neither the ~1 of a lab z in Angstrom nor the ~0.1 of one in nm. "
            "Check the columns, or pass -midplane as a number."
        )

    flip = slope < 0
    per_frame = scale * z + op if flip else scale * z - op
    midplane = float(np.mean(per_frame))
    info = {
        "n_frames": int(op.size),
        "n_files": len(paths),
        "slope": slope,
        "unit": unit,
        "flip": flip,
        "spread": float(np.std(per_frame)),
    }
    return midplane, flip, info


def find_cv_files(cv_dir, max_files):
    """Up to `max_files` per-path CV files, spread evenly over the directory.

    An evenly spaced sample rather than the first N: the membrane drifts over
    a run, and the first few paths written are not a fair sample of where its
    centre sat.
    """
    files = sorted(Path(cv_dir).glob("*.txt"))
    if not files:
        raise ValueError(f"No per-path .txt files in {cv_dir}.")
    if max_files <= 0 or len(files) <= max_files:
        return files
    index = np.linspace(0, len(files) - 1, max_files).round().astype(int)
    return [files[i] for i in sorted(set(index.tolist()))]


def resistance_profile(z, beta_g, d_z):
    """exp(beta dG) / D at each z - the integrand, in ps/A^2."""
    with np.errstate(over="ignore"):
        return np.exp(beta_g) / d_z


def permeability_from_resistance(z, resistance):
    """(P in cm/s, 1/P in ps/A) from a resistance profile by the trapezoid rule."""
    inv_p = float(np.trapezoid(resistance, z)) if hasattr(np, "trapezoid") \
        else float(np.trapz(resistance, z))
    inv_p = abs(inv_p)
    if inv_p <= 0:
        return np.inf, 0.0
    return ANGSTROM_PER_PS_TO_CM_PER_S / inv_p, inv_p


def _bootstrap_permeability(z, beta_g, d_z, d_err, n_bootstrap, rng):
    """Spread in P from the uncertainty on D alone.

    D is resampled per bin from a lognormal with the reported standard error,
    which keeps every draw positive - a normal draw would occasionally produce
    a negative or near-zero D in the barrier region and send P to absurd
    values. The free energy is held fixed, so this is a lower bound on the
    total uncertainty, not a full error budget.
    """
    usable = np.isfinite(d_err) & (d_err > 0)
    if not usable.any() or n_bootstrap <= 0:
        return None

    rel = np.where(usable, d_err / d_z, 0.0)
    sigma = np.sqrt(np.log1p(rel ** 2))
    draws = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        noise = rng.normal(size=d_z.size) * sigma
        d_draw = d_z * np.exp(noise - 0.5 * sigma ** 2)
        draws[i], _ = permeability_from_resistance(
            z, resistance_profile(z, beta_g, d_draw)
        )
    return draws


#: Categorical slots in fixed order: one hue per stage, never cycled.
STAGE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948")
_INK = "#0b0b0b"          # primary text / the result line
_INK_SECONDARY = "#52514e"
_RULE = "#c9c8c2"         # recessive interface markers
_SURFACE = "#fcfcfb"


def _csv_names(stages):
    """Unique, CSV-safe column names for the stage parts."""
    names, seen = [], {}
    for stage in stages:
        base = "stage_" + re.sub(r"[^0-9A-Za-z]+", "_", stage.name).strip("_")
        seen[base] = seen.get(base, 0) + 1
        names.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return names


def write_stitching(stitched, out_dir, label, overw):
    """stitching.png and stitching.csv: how the stages were put together.

    The top panel is each stage exactly as it enters the sum - after its
    factor and its cut - with the forward sum, its mirror image and the total
    drawn over them; the bottom panel the free energy of the forward sum alone
    beside the total's, both shifted by the same constant, over the full range
    before any cut to where D exists. A junction that closes shows the total
    running straight through the interface; one that does not shows a step,
    and the step is annotated in kT. Returns `junction_steps`.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    png = os.path.join(out_dir, "stitching.png")
    table = os.path.join(out_dir, "stitching.csv")
    _check_overwrite(png, overw)
    _check_overwrite(table, overw)

    z = stitched.centers
    total = stitched.total
    with np.errstate(divide="ignore", invalid="ignore"):
        free_forward = np.where(stitched.forward > 0, -np.log(stitched.forward), np.nan)
        free_total = np.where(total > 0, -np.log(total), np.nan)
    shift = np.nanmin(free_total) if np.isfinite(free_total).any() else 0.0
    free_forward = free_forward - shift
    free_total = free_total - shift
    steps = junction_steps(stitched)

    # ---- table ------------------------------------------------------------
    columns = [z] + list(stitched.parts) + [stitched.forward]
    header = ["z"] + _csv_names(stitched.stages) + ["forward"]
    if stitched.mirror is not None:
        columns.append(stitched.mirror)
        header.append("mirror")
    columns += [total, free_forward, free_total]
    header += ["total", "beta_dG_forward", "beta_dG_total"]
    np.savetxt(table, np.column_stack(columns), delimiter=",", fmt="%.6e",
               header=",".join(header), comments="")

    # ---- figure -----------------------------------------------------------
    def masked(values):
        return np.where(values > 0, values, np.nan)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1]}, facecolor=_SURFACE,
    )
    for ax in (top, bottom):
        ax.set_facecolor(_SURFACE)
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(_INK_SECONDARY)
        ax.tick_params(colors=_INK_SECONDARY, labelsize=10)
        for stage in stitched.stages[1:]:
            ax.axvline(stage.lam_a, color=_RULE, lw=1, ls=":", zorder=0)

    # The total goes underneath and the stages on top: where a stage alone
    # makes up the total its colour shows, and the black line only appears
    # where the mirror term lifts the total above it. The forward sum is not
    # drawn here - the cut stages do not overlap, so it is just their union.
    if stitched.mirror is not None and stitched.symmetrize:
        top.plot(z, masked(total), color=_INK, lw=2, label="total", zorder=2)
        top.plot(z, masked(stitched.mirror), color=_INK_SECONDARY, lw=1.5,
                 ls="--", label="mirror (backward)", zorder=4)
    for index, (stage, part) in enumerate(zip(stitched.stages, stitched.parts)):
        color = STAGE_COLORS[index % len(STAGE_COLORS)]
        top.plot(z, masked(part), color=color, lw=2, label=stage.name, zorder=3)
        inside = np.flatnonzero(part > 0)
        if inside.size:
            # Along the bottom edge, centred on the stage's own range: that
            # strip is empty, where any spot on the curves is crossed by the
            # total or the mirror.
            centre = 0.5 * (z[inside[0]] + z[inside[-1]])
            top.text(centre, 0.03, stage.name, transform=top.get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=9, color=_INK_SECONDARY)
    top.set_yscale("log")
    top.set_ylabel("weighted count (scaled)", fontsize=11, color=_INK)
    handles, labels = top.get_legend_handles_labels()
    top.legend(handles, labels, frameon=False, fontsize=9,
               labelcolor=_INK_SECONDARY, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=min(len(labels), 5))

    if stitched.symmetrize:
        bottom.plot(z, free_forward, color=_INK_SECONDARY, lw=1.5,
                    label="forward sum only")
        bottom.plot(z, free_total, color=_INK, lw=2, label="total (symmetrised)")
        bottom.legend(frameon=False, fontsize=9, labelcolor=_INK_SECONDARY,
                      loc="upper left")
    else:
        bottom.plot(z, free_total, color=_INK, lw=2)
    finite = free_total[np.isfinite(free_total)]
    top_of_axis = float(finite.max()) if finite.size else 1.0
    for item in steps:
        if np.isfinite(item["total"]):
            bottom.annotate(f"step {item['total']:+.2f} kT",
                            (item["lam_a"], top_of_axis),
                            textcoords="offset points", xytext=(4, -2),
                            fontsize=9, color=_INK_SECONDARY, va="top")
    bottom.set_ylabel(r"$\beta\,\Delta G$  (kT, shared offset)", fontsize=11,
                      color=_INK)
    bottom.set_xlabel("position along the order parameter (Å)", fontsize=11,
                      color=_INK)

    fig.suptitle(f"Stitched histogram — {label}", fontsize=12, color=_INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(png, dpi=200, facecolor=_SURFACE)
    plt.close(fig)

    print(f"  Stitching plot -> {png}")
    print(f"  Stitching table -> {table}")
    for item in steps:
        print(f"    step across lA = {item['lam_a']:>6.2f} ('{item['stage']}'): "
              f"forward {item['forward']:+.3f} kT, total {item['total']:+.3f} kT")
    return steps


def read_perm_data(file_path):
    if os.path.isfile(file_path):
        comments = []
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    comments.append(line.strip())
        #... = {DeltaZ} [angstrom] ... obtains value between "=" and "["
        delta_z = float(comments[1].split("=")[1].split("[")[0].strip())

        perm_data = np.loadtxt(file_path, comments="#", dtype=float)
        return delta_z, perm_data[-1,4], perm_data[-1,1] #delta_z, tau_ref, xi
    else:
        raise FileNotFoundError("{file_path} not found")


def calculate_p_transit(P_A, P_M, P_B):
    """
    Calculates the total membrane crossing probability accoring to a Markov state model
    with the formula:
    P_transit =  (P_{AC} * P_{CD} * P_{DB}) / (1 - P_{DC} *P_{CD})

    "Wouter Vervust, Daniel T. Zhang, Titus S. van Erp, An Ghysels,
    Path sampling with memory reduction and replica exchange to reach long permeation timescales,
    Biophysical Journal, volume 122, Issue 14, 2023"
    """
    return (P_A * P_M * P_B) / (1 - (P_M * P_B))



def permeability_from_retis(delta_z, xi, tau_ref, P_transit):
    """
    Calculates the permeability from RETIS data with the formula:
    P_permeability = xi * Delta z / tau_ref [o^{-'}] * P_A (lambda_B|lambda_A)

    An Ghysels, Sander Roet, Smaneh Davoudi and Titus S. van Erp,
    Exact non-Markovian permeability from rare event simulations
    Phys. Rev. Research, 3, 2021, 033068
    """
    Angstrom_to_cm = 1e-8
    prefactor = ((xi * delta_z) / (tau_ref)) * Angstrom_to_cm #  runav_tau_ref is in the _units file already in seconds
    return prefactor * P_transit # in cm/s


def _plot(z, beta_g, d_z, resistance, out_path, label, overw):
    _check_overwrite(out_path, overw)
    cumulative = np.concatenate(([0.0], np.cumsum(
        0.5 * (resistance[1:] + resistance[:-1]) * np.diff(z))))

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1.3, 1]})

    axes[0].plot(z, beta_g, color="#2a7fbf", lw=2)
    axes[0].set_ylabel(r"$\beta\,\Delta G$", fontsize=12)
    axes[0].axhline(0.0, color="grey", lw=0.8, ls=":")

    axes[1].plot(z, d_z, color="#d1701f", lw=2)
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$D$  ($\mathrm{\AA^2\,ps^{-1}}$)", fontsize=12)

    axes[2].plot(z, resistance, color="#7b3294", lw=2)
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"$e^{\beta\Delta G}/D$  (ps $\mathrm{\AA^{-2}}$)", fontsize=12)
    peak = z[int(np.argmax(resistance))]
    axes[2].axvline(peak, color="grey", lw=0.8, ls="--")
    axes[2].annotate(f"peak at {peak:.2f}", (peak, np.nanmax(resistance)),
                     textcoords="offset points", xytext=(6, -12), fontsize=9,
                     color="#444444")

    axes[3].plot(z, 100.0 * cumulative / cumulative[-1], color="#444444", lw=2)
    axes[3].set_ylabel("cumulative\nresistance (%)", fontsize=11)
    axes[3].set_xlabel("position along the order parameter (Å)", fontsize=12)
    axes[3].set_ylim(0, 100)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Resistance to permeation — {label}", fontsize=12,
                 color="#444444")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Profile plot -> {out_path}")
    return cumulative


def permeation_process(
    runs: Annotated[Optional[str], typer.Option("-runs", help="Run directories of the staged simulations, comma-separated, in order along the permeation coordinate (e.g. infinit_entry,infinit_internal,infinit_escape). Each must hold the standard wham/ folder; the histograms, crossing probabilities and path counts are read from it", rich_help_panel=panels.INPUT)] = None,
    free_energy: Annotated[str, typer.Option("-fe", help="Ready-made free-energy profile(s) instead of -runs, comma- or space-separated, in order along the permeation coordinate. Needs -pcross to reference the stages", rich_help_panel=panels.INPUT)] = "wham/histo_free_energy.txt",
    diffusion: Annotated[str, typer.Option("-diffusion", help="diffusion_aggregate.csv from membrane-spatial", rich_help_panel=panels.INPUT)] = "membrane_plots/diffusion_aggregate.csv",
    cv_dir: Annotated[Optional[str], typer.Option("-cv-dir", help="Folder of per-path CV .txt files used to locate the midplane with -midplane auto; defaults to ML/ inside the first -runs directory", rich_help_panel=panels.INPUT)] = None,
    label: Annotated[str, typer.Option("-label", help="Name for this run in the plot title and the summary", rich_help_panel=panels.INPUT)] = "permeant",
    d_column: Annotated[str, typer.Option("-d-column", help="Which diffusion column to use: D_z is the local short-time estimate, D_z_lag2 the same at twice the lag (a diffusive-regime check, not a rival estimate)", rich_help_panel=panels.DATASET)] = "D_z",
    midplane: Annotated[str, typer.Option("-midplane", help="Offset in Angstrom subtracted from the diffusion slab centres. membrane-spatial already bins on membrane_z - permeant_z, the order-parameter coordinate itself, so the default 0 is right for its diffusion_aggregate.csv. Use a number, or 'auto' to measure it from the CV trajectories, only for a profile that is genuinely on lab-frame z", rich_help_panel=panels.DATASET)] = "0",
    flip: Annotated[bool, typer.Option("-flip/-no-flip", help="Negate the diffusion axis, op = -(slab_center - midplane). Off by default because membrane-spatial's axis already runs the same way as the order parameter. Ignored when -midplane auto measures the sign", rich_help_panel=panels.DATASET)] = False,
    op_col: Annotated[str, typer.Option("-op-col", help="Order-parameter column in the CV files, for -midplane auto", rich_help_panel=panels.DATASET)] = "OP_Lamb",
    z_col: Annotated[str, typer.Option("-z-col", help="Lab-frame permeant z column in the CV files, for -midplane auto", rich_help_panel=panels.DATASET)] = "z_PRO",
    max_cv_files: Annotated[int, typer.Option("-max-cv-files", help="How many per-path CV files -midplane auto reads, spread evenly over the folder; 0 reads all", rich_help_panel=panels.DATASET)] = 60,
    wham_subdir: Annotated[str, typer.Option("-wham-subdir", help="Name of the wham folder inside each -runs directory", rich_help_panel=panels.DATASET)] = "wham",
    pcross_col: Annotated[int, typer.Option("-pcross-col", help="Column of wham/Pcross.txt holding the total crossing probability: 3 is P-wham2, 1 is P-wham", rich_help_panel=panels.DATASET)] = 3,
    pcross_back: Annotated[Optional[str], typer.Option("-pcross-back", help="Backward escape probability out of each later stage's state A, comma-separated, one per junction. Only needed when the staging is not mirror-symmetric, so the mirrored run is not among -runs", rich_help_panel=panels.DATASET)] = None,
    path_norm: Annotated[bool, typer.Option("-path-norm/-no-path-norm", help="Divide each stage's histogram by its number of paths (runav_rate.txt) when referencing the stages. Turn on for histograms not yet normalised per path", rich_help_panel=panels.DATASET)] = False,
    symmetrize: Annotated[bool, typer.Option("-symmetrize/-no-symmetrize", help="Add the combined histogram to its own mirror image about z = 0. In a symmetric bilayer this supplies each region's reverse-direction histogram, which is what removes the conditioning bias of the forward ones - leave it on unless the membrane is genuinely asymmetric", rich_help_panel=panels.DATASET)] = True,
    pcross: Annotated[Optional[str], typer.Option("-pcross", help="-fe route only: total crossing probability of each junction, comma-separated, one per stage boundary (N stages need N-1). Last row of each earlier stage's wham/Pcross.txt. Required to reference conditional staged profiles to each other", rich_help_panel=panels.DATASET)] = None,
    overlap: Annotated[Optional[float], typer.Option("-overlap", help="-fe route only: Angstroms of each junction used to match staged profiles; unset uses the whole shared range", rich_help_panel=panels.DATASET)] = None,
    min_neff: Annotated[float, typer.Option("-min-neff", help="Drop diffusion bins with fewer effective paths than this", rich_help_panel=panels.DATASET)] = 0.0,
    bulk: Annotated[Optional[str], typer.Option("-bulk", help="'lo,hi' window treated as bulk water, where beta*dG is set to zero. Unset uses the profile's own minimum", rich_help_panel=panels.DATASET)] = None,
    n_bootstrap: Annotated[int, typer.Option("-n-bootstrap", help="Resamples of D within its reported error, for a confidence interval on P; 0 disables", rich_help_panel=panels.MODEL)] = 2000,
    seed: Annotated[int, typer.Option("-seed", help="Random seed for the bootstrap", rich_help_panel=panels.MODEL)] = 42,
    out_dir: Annotated[str, typer.Option("-out-dir", help="Directory for the profile table and plot", rich_help_panel=panels.OUTPUT)] = "permeation",
    overw: Annotated[bool, typer.Option("-O", help="Force overwriting of existing files", rich_help_panel=panels.OUTPUT)] = False,
):
    """Permeability from the inhomogeneous solubility-diffusion model.

    Combines the WHAM free energy with the local diffusion coefficient from
    `membrane-spatial` into the Marrink-Berendsen resistance integral,

        1/P = integral of exp(beta dG(z)) / D(z) dz

    and reports P in cm/s. This is an independent estimate of what RETIS gets
    from reactive paths, resting on different assumptions (overdamped Markovian
    motion along z), so the two agreeing is meaningful and the two disagreeing
    is a result in itself.

    Point -runs at the staged run directories and everything else is read from
    their wham/ folders. The stages are conditional histograms, so each is
    rescaled by the crossing probabilities and path counts of the junctions
    before it, cut at its own lA, added, and the sum added to its own mirror
    image (which supplies each region's backward histogram and is what makes
    the junctions close). The free energy is `-ln` of that, already beta*dG in
    kT, so no temperature is needed.

    D is read in A^2/ps. `membrane-spatial` already bins it on
    `membrane_z - permeant_z`, which is the order-parameter coordinate, so no
    axis conversion is needed; `-midplane`/`-flip` are there for a diffusion
    profile that is on actual box z-values, and `-midplane auto` measures
    that offset from the CV trajectories.

    Writes to -out-dir:
      permeation_profile.csv   z, beta_dG, D, resistance, cumulative fraction
      permeation_profile.png   the four panels of the resistance decomposition
      stitching.png / .csv     (-runs only) each stage as it enters the stitched
                               histogram, the forward sum, its mirror and the
                               total, with the free-energy step at every junction

    The resistance profile is the part worth reading. Because dG enters
    exponentially, the integral is dominated by a few angstroms near the
    barrier, and the cumulative panel says exactly how few - if 90% of the
    resistance comes from three bins, P is a statement about those three bins
    and about D there, which is where D is least well determined.
    """
    if not os.path.isfile(diffusion):
        raise ValueError(f"{diffusion} is not a file.")

    run_dirs = _split_list(runs)
    n_stages = len(run_dirs)
    if run_dirs:
        for directory in run_dirs:
            if not os.path.isdir(directory):
                raise ValueError(f"{directory} is not a directory.")
        print(f"{n_stages} staged run(s); D from {d_column}")
        stitched, pcrosses = stitch_runs(
            run_dirs, wham_subdir=wham_subdir, pcross_col=pcross_col,
            pcross_back=([float(v) for v in _split_list(pcross_back)]
                         if pcross_back else None),
            path_norm=path_norm, symmetrize=symmetrize,
        )
        z, beta_g = free_energy_from_counts(stitched.centers, stitched.total)
        print(f"  combined profile [{z.min():.3f}, {z.max():.3f}]  "
              f"{z.size} sampled bins")
        write_stitching(stitched, out_dir, label, overw)
    else:
        profile_paths = _split_list(free_energy)
        n_stages = len(profile_paths)
        for path in profile_paths:
            if not os.path.isfile(path):
                raise ValueError(f"{path} is not a file.")

        print(f"{n_stages} free-energy profile(s); D from {d_column}")
        profiles = [load_free_energy(p) for p in profile_paths]
        for path, (z_p, _) in zip(profile_paths, profiles):
            print(f"  {os.path.basename(path):<28} "
                  f"[{z_p.min():8.3f}, {z_p.max():8.3f}]  {z_p.size} bins")

        shifts = None
        if pcross:
            pcrosses = [float(v) for v in _split_list(pcross)]
            if len(pcrosses) != len(profiles) - 1:
                raise ValueError(
                    f"-pcross has {len(pcrosses)} value(s) but {len(profiles)} "
                    f"stage(s) need {len(profiles) - 1} (one per junction)."
                )
            shifts = stage_shifts_from_pcross(pcrosses)
        elif len(profiles) > 1:
            print("  [warn] no -pcross given for a staged profile; falling back "
                  "to matching the overlaps, which is not physically "
                  "referenced. Prefer -runs, which reads the crossing "
                  "probabilities itself.")

        z, beta_g = join_profiles(profiles, shifts=shifts, overlap=overlap)

    if bulk:
        lo, hi = (float(v) for v in _split_list(bulk))
        window = (z >= min(lo, hi)) & (z <= max(lo, hi))
        if not window.any():
            raise ValueError(f"-bulk {bulk} selects no bin of the joined profile.")
        reference = float(np.mean(beta_g[window]))
        print(f"  beta*dG referenced to the mean over [{lo}, {hi}] "
              f"({int(window.sum())} bins)")
    else:
        reference = float(np.min(beta_g))
        print(f"  beta*dG referenced to its own minimum at z = "
              f"{z[int(np.argmin(beta_g))]:.3f}")
    beta_g = beta_g - reference

    if str(midplane).strip().lower() in ("auto", ""):
        folder = cv_dir or (os.path.join(run_dirs[0], "ML") if run_dirs else None)
        if not folder or not os.path.isdir(folder):
            raise ValueError(
                f"-midplane auto needs the CV trajectories: {folder or '-cv-dir'} "
                "is not a directory. Pass -cv-dir, or give -midplane as a number."
            )
        midplane_value, detected_flip, info = midplane_from_cv_files(
            find_cv_files(folder, max_cv_files), op_col=op_col, z_col=z_col,
        )
        print(f"  midplane measured from {info['n_files']} CV file(s), "
              f"{info['n_frames']} frames: {midplane_value:.3f} A "
              f"(+/- {info['spread']:.3f} between frames; '{z_col}' read as "
              f"{info['unit']})")
        if detected_flip != flip:
            print(f"  [warn] the fit says the order parameter runs "
                  f"{'opposite to' if detected_flip else 'with'} lab z "
                  f"(d{z_col}/d{op_col} = {info['slope']:+.4g}), which "
                  f"contradicts -{'' if flip else 'no-'}flip. Using the "
                  "measured sign.")
        flip = detected_flip
    else:
        try:
            midplane_value = float(midplane)
        except ValueError:
            raise ValueError(
                f"-midplane {midplane!r} is neither a number nor 'auto'."
            ) from None

    z_d, d_vals, d_err = load_diffusion(
        diffusion, column=d_column, min_neff=min_neff,
        midplane=midplane_value, flip=flip,
    )
    if z_d.size < 2:
        raise ValueError("Fewer than two usable diffusion bins.")
    print(f"  D bins on the order-parameter axis: "
          f"[{z_d.min():.3f}, {z_d.max():.3f}]  {z_d.size} slabs")

    inside = (z >= z_d.min()) & (z <= z_d.max())
    if not inside.any():
        raise ValueError(
            f"The free-energy profile ([{z.min():.3f}, {z.max():.3f}]) and the "
            f"diffusion profile ([{z_d.min():.3f}, {z_d.max():.3f}]) do not "
            "overlap. Check -midplane and -flip: membrane-spatial's own "
            "diffusion_aggregate.csv is already on the order-parameter axis "
            "and wants the defaults (-midplane 0 -no-flip)."
        )
    if not inside.all():
        print(f"  [warn] {int((~inside).sum())} free-energy bins fall outside "
              "the diffusion profile and are dropped; the integral is "
              "truncated there.")
    z, beta_g = z[inside], beta_g[inside]

    d_z = np.interp(z, z_d, d_vals)
    e_z = np.interp(z, z_d, d_err)

    resistance = resistance_profile(z, beta_g, d_z)
    permeability_resistance, inv_p = permeability_from_resistance(z, resistance)

    if run_dirs:
        entry_dir =  run_dirs[0]
        runav_perm_file_path = os.path.join(entry_dir, wham_subdir, "runav_permeability_units.txt")
        if not os.path.isfile(runav_perm_file_path):
            raise FileNotFoundError(f"Permeability data file not found.\n{runav_perm_file_path}")
        delta_z, tau_ref, xi = read_perm_data(runav_perm_file_path)
        P_transit = calculate_p_transit(pcrosses[0], pcrosses[1], pcrosses[2])
        permeability_retis = permeability_from_retis(delta_z, xi, tau_ref, P_transit)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cumulative = _plot(z, beta_g, d_z, resistance,
                       os.path.join(out_dir, "permeation_profile.png"),
                       label, overw)

    table = os.path.join(out_dir, "permeation_profile.csv")
    _check_overwrite(table, overw)
    fraction = 100.0 * cumulative / cumulative[-1] if cumulative[-1] > 0 else cumulative
    np.savetxt(
        table, np.column_stack([z, beta_g, d_z, resistance, fraction]),
        delimiter=",", fmt="%.6e",
        header="z,beta_dG,D_A2_per_ps,resistance_ps_per_A2,cumulative_percent",
        comments="",
    )
    print(f"  Table -> {table}")

    print(f"\n1/P = {inv_p:.6g} ps/A")
    print(f"P_resistance   = {permeability_resistance:.6g} cm/s   ({label})")
    print(f"P_retis        = {permeability_retis:.6g} cm/s")

    draws = _bootstrap_permeability(z, beta_g, d_z, e_z, n_bootstrap,
                                    np.random.default_rng(seed))
    if draws is not None:
        lo_p, hi_p = np.percentile(draws, [2.5, 97.5])
        print(f"      95% CI from the uncertainty on D alone: "
              f"[{lo_p:.6g}, {hi_p:.6g}] cm/s")
    else:
        print("      no usable D uncertainties; no interval reported")

    # Where the answer actually comes from.
    peak = int(np.argmax(resistance))
    half = int(np.searchsorted(fraction, 50.0))
    span = z[min(half + 1, z.size - 1)] - z[max(half - 1, 0)]
    print(f"\nResistance peaks at z = {z[peak]:.3f} "
          f"(beta*dG = {beta_g[peak]:.3f}, D = {d_z[peak]:.4g} A^2/ps)")
    print(f"Half the total resistance accumulates within ~{abs(span):.2f} A "
          f"of z = {z[half]:.3f}")
    if n_stages == 1:
        print("\n[warn] one stage only, so this covers part of the membrane. A "
              "partial integral underestimates 1/P and therefore OVERESTIMATES "
              "P - pass every stage to -runs for a complete number.")
