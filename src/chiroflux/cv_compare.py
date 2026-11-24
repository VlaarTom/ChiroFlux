"""
Every per-path CV (the generate-cvs files) compared between two sets of
paths along the order parameter, built the way chiral-compare is.

What it answers
---------------
`histograms` bins each CV of one simulation into its own figures, one per
column and path group, and leaves the comparing to the reader.  This command
asks the comparative question for all CVs at once: at the same position
along the permeation path, does a CV take different values in set A than in
set B, by how much compared with its spread, and is that more than the
sampling noise?  The two sets are either

  * two simulations, -runs-a and -runs-b (e.g. the L and D enantiomers),
    each given as its stages in chain order (entry, internal, escape), or
  * the reactive and the unreactive paths of one simulation (-runs-a only).

The results are collected in a few tables, which `chiroflux cv-plot` draws:

  profiles.csv        mean of every CV per order-parameter bin for A and B,
                      the difference with its 95% CI, p and BH q, and the
                      standardised effect d = difference / pooled SD
  distributions.csv   the distribution of every CV per OP window, compared
                      bin by bin; distributions.npz holds the same for the
                      figures, plus the OP x value maps of every CV
  shifts.csv          per CV and window, HOW the distributions differ: a
                      shift (mean, median), a change of spread (SD) or of a
                      tail (10th / 90th percentile), each with CI and q, and
                      the Wasserstein distance between them
  ranking.csv         one row per CV: its largest significant effect, where
                      along the path, and how many bins differ.  The figures
                      order the CVs by it
  correlations.npz    the CVs against each other: the within-window
                      correlation matrix of A and of B, so redundant CVs and
                      changes in how they couple show up (WHAM-weighted
                      folders only)
  blocking.csv, block_choice.csv   how many consecutive paths a bootstrap
                      block needs (chiral-compare's blocking analysis)

Two simulations also get the reactive and the unreactive paths alone, in
reactive/ and unreactive/; every stage gets its own folder; and the whole
analysis is repeated without the WHAM reweighting in unweighted/.  The
layout is chiral-compare's.

The order-parameter axis
------------------------
Bins are on the SIGNED order parameter (-op-col), not on |OP| as in
chiral-compare.  Folding onto |OP| puts frames from both sides of the
bilayer into one bin, and most CVs here name a leaflet (_u_/_l_) or point
along +z, so they would first need a per-column transformation.  With entry
(lambda -25 -> -13.5), internal (-13.5 -> 13.5) and escape (13.5 -> 25)
stages the signed axis covers one complete crossing.  Every set of bin edges
contains the stage boundaries, so no bin mixes two stages: within a bin only
the stage's own WHAM weights matter, and the stage factors (wham/Pcross.txt)
only set how the occupancy is shared between stages.  OP 0, the midplane, is
an edge too, so no bin or window takes frames from both halves of the
bilayer.

A second simulation that entered from the other face needs NO change to its
axis.  The OP column is each run's own infretis order parameter (column 1 of
order.txt), so both runs' entry stages already sit at lambda -25 -> -13.5,
whichever face they came in from.  What differs is the lab frame: B's entry
leaflet is A's exit leaflet, and anything signed along lab z changes sign.
Hence -swap-leaflets-b, plus the per-column symmetry corrections the other
two-simulation commands use: -mirror-b for CVs signed along z (Mem_*_def,
*_defDeep, *_curvH, the raw cosines Mem_*_tilt and PRO_mode_*), -flip-b for
angles in degrees against +z, and, for the other enantiomer, -mirror-b on
the chirality-odd pseudoscalars and -phase-shift-b on PRO_CP_phi2.  The raw
lab-z columns (z_*) cannot be corrected by a sign and are best excluded.
A signed azimuth about lab +z (PRO_azim_*) is odd under both the face flip
and the enantiomer, so between opposite-face enantiomers it needs neither.

-flip-op-b is something else: it bins B on -OP, so B's entry stage lands
where A's escape is and the two are compared at the same lab z rather than
at the same point of their paths.  Using it for opposite-face runs compares
A going up with B coming down, and the stage folders then share no frames.

Direction of motion
-------------------
-direction inbound keeps only the frames at which the order parameter is
increasing along the path, -direction outbound those at which it is
decreasing; the default, all, keeps every frame.  The direction is the sign
of the path's own OP gradient over its phase points in path order (central
differences, one-sided at the two ends), taken before the frames are split
at lambda_A or binned.  OP_Lamb increases from the entry side to the exit
side in every run, so inbound is the way a crossing goes and outbound the way
back; with -flip-op-b, B's direction is still that of its own OP.  A path
with fewer than two phase points has no direction and adds nothing.

What to expect: an unreactive path starts and ends at lambda_A, so its time
reverse is in the same ensemble with the same weight, and reversing a path
swaps its inbound and outbound frames (exactly, with central differences).
For the unreactive paths, inbound and outbound frames therefore have the same
distribution of every configurational CV, and a difference between an
inbound and an outbound run of them is sampling noise: a convergence check.
Reactive paths have no reversed partner; their inbound frames are the
crossing itself and their outbound frames its recrossings.

Weights and errors
------------------
Path weights, the [0-] frames below lambda_A of the entry stage (scaled by
1/xi), the stage factors and their errors are chiral-compare's, through its
setup_stages; see "Path-ensemble reweighting" there.  load/<path>/order.txt
is read for xi, as there; the trajectories are not needed.  Error bars come
from a block bootstrap over paths in MC order within each stage, with the
block chosen per folder by the blocking analysis (-block auto).  In the
reactive vs unreactive mode both classes come from one run, so they share
every bootstrap replicate.

Values outside a CV's value range (from -ranges, else from the 0.1-99.9%
quantiles of a sample of the paths, padded) are counted in its outermost
bins; profiles and the means and SDs in shifts.csv are exact either way.

Populations: -indicators
------------------------
The mean of an angle with two states says little about how the frames split
between them, and it is no mean at all on a periodic axis (anti at +179 and
-179 averages to syn).  -indicators adds 0/1 columns, one per state, e.g.

    -indicators 'PRO_anti=abs(PRO_dih_OH)>90,PRO_endo=PRO_dih_chi1>0'

whose mean per OP bin is the population of the state, so profiles.csv gives
it along the path with CI and q like any CV.  Each test reads its CV in the
file's own units (never the cos of -angle-cols) and takes that CV's B
corrections first, so one test holds for both sides: with -mirror-b PRO_dih_,
D's chi1 is negated first and chi1 > 0 is Cg-endo in both (see generate-cvs,
PRO_dih_chi1).

Tests joined by & must all hold: a joint state, e.g. Cg-endo with the N-H on
the carboxyl side, 'endo_cis=PRO_dih_chi1>0&abs(PRO_dih_NH)<90'.  Two
enantiomers can agree on every state alone and still differ in how often
they combine.  flip(...) turns a state into the frames at which it changes
before the next frame: its mean is the switching rate per frame step (divide
by the frame spacing for a rate in time), resolved by where along the path
it happens.  Switches faster than the frame spacing are missed.

The blocking analysis covers indicators too; a slow state (one the paths
inherit rather than sample, such as the anti COOH) then rightly sets a large
block.

Usage
-----
    chiroflux cv-compare \\
        -runs-a L/infinit_entry,L/infinit_internal,L/infinit_escape \\
        -runs-b D/infinit_entry,D/infinit_internal,D/infinit_escape \\
        -label-a L -label-b D -data-a post/infretis_data.txt \\
        -data-b post/infretis_data.txt -out-dir comparison/cv_compare

    chiroflux cv-compare -runs-a L/infinit_entry,L/infinit_internal,L/infinit_escape \\
        -out-dir L/cv_compare_classes          (reactive vs unreactive)
"""

from __future__ import annotations

import json
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import typer

from . import chiral_interface_analysis as cia
from . import panels
from .cv_histograms import SPHERICAL_ANGLE_COLS
from .pathdata import _check_overwrite, _matches_any, _matches_exclude

DEFAULT_CV_DIR = "analysis/ML_mda"
OP_COL = "OP_Lamb"
CLASSES = ("reactive", "unreactive", "minus")
PATH_SUBSETS = ("reactive", "unreactive")
UNWEIGHTED_DIR = "unweighted"
QUANTILES = (("p10", 0.10), ("median", 0.50), ("p90", 0.90))
SHIFT_MEASURES = ("mean", "sd") + tuple(n for n, _ in QUANTILES)
PARTS = ("above", "below")
DIRECTIONS = ("all", "inbound", "outbound")
DIRECTION_TITLE = {"all": "", "inbound": ", inbound frames (OP increasing)",
                   "outbound": ", outbound frames (OP decreasing)"}
#: The order parameter of the bilayer midplane: an edge of every grid, and
#: where the distribution figures split their windows into two rows.
MIDPLANE = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Columns: which CVs, under which name, transformed how
# ══════════════════════════════════════════════════════════════════════════════

def swap_leaflets(name):
    """_u_ <-> _l_: every leaflet-marked generate-cvs column spells the
    leaflet as a medial field (CA_C2_u_DOPC, Mem_l_def, PRO_hCN_u_P)."""
    return name.replace("_u_", "\0").replace("_l_", "_u_").replace("\0", "_l_")


def family(name):
    """The prefix a CV is grouped under in the figures: PRO, Mem, CA, z, ...;
    single-word columns (Water, DOPC, BoxZ) are 'other'."""
    for pre in ("cos2(", "cos("):
        if name.startswith(pre):
            name = name[len(pre):-1]
    head, sep, _ = name.partition("_")
    return head if sep else "other"


_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_TERM = r"(?:abs\(\s*(?P<abs>[^\s()<>=&]+)\s*\)|(?P<col>[^\s()<>=&]+))"
_ONE_SIDED = re.compile(rf"^\s*{_TERM}\s*(?P<op>[<>]=?)\s*(?P<v>{_NUM})\s*$")
_TWO_SIDED = re.compile(rf"^\s*(?P<lo>{_NUM})\s*(?P<op1><=?)\s*{_TERM}\s*(?P<op2><=?)\s*"
                        rf"(?P<hi>{_NUM})\s*$")
_FLIP = re.compile(r"^\s*flip\((?P<inner>.*)\)\s*$")
_INDICATOR_NAME = re.compile(r"^[A-Za-z_][\w.\-]*$")
INDICATOR_FORMS = ("NAME=TEST, TEST being COL>v, COL<=v (etc.) or lo<COL<hi, abs(COL) for "
                   "|COL|; several TESTs joined by & must all hold; NAME=flip(...) counts "
                   "the frames at which that state changes")


@dataclass(frozen=True)
class Clause:
    """One test of an indicator: the value of `column` (|value| if
    `absolute`) inside the bounds.  A bound of None is open-ended; *_incl
    makes it inclusive."""
    column: str
    absolute: bool = False
    lo: float | None = None
    lo_incl: bool = False
    hi: float | None = None
    hi_incl: bool = False

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        v = np.abs(x) if self.absolute else x
        inside = np.ones(v.shape, bool)
        with np.errstate(invalid="ignore"):
            if self.lo is not None:
                inside &= (v >= self.lo) if self.lo_incl else (v > self.lo)
            if self.hi is not None:
                inside &= (v <= self.hi) if self.hi_incl else (v < self.hi)
        return np.where(np.isfinite(x), inside.astype(float), np.nan)


@dataclass(frozen=True)
class Indicator:
    """
    A 0/1 column made from one or more CVs: 1 in the frames where every
    clause holds (a joint state when there are several), 0 where one does
    not, NaN where a clause's CV has no value.  Its weighted mean in an OP
    bin is the population of that state there, so the profiles give a
    population along the path with its CI.

    With `flip`, the column is instead 1 at a frame where the state differs
    from the next frame's and 0 where it does not (NaN at the last frame of
    the path): its mean is the number of changes per frame step, i.e. the
    rate of entering or leaving the state, by where along the path it
    happens.  Frames must be in path order, as generate-cvs writes them.
    """
    name: str
    clauses: tuple
    flip: bool = False
    text: str = ""

    @property
    def columns(self):
        return tuple(c.column for c in self.clauses)

    def __call__(self, values):
        """`values`: one array per clause, in clause order."""
        out = None
        for clause, x in zip(self.clauses, values):
            v = clause(x)
            out = v if out is None else out * v
        if self.flip:
            change = np.abs(np.diff(out))
            out = np.r_[change, np.nan] if out.size else out
        return out


def _parse_clause(text, spec):
    m = _TWO_SIDED.match(text)
    if m:
        lo, hi = float(m["lo"]), float(m["hi"])
        if lo >= hi:
            raise ValueError(f"{spec!r}: the range {lo:g} to {hi:g} is empty")
        bounds = dict(lo=lo, lo_incl=m["op1"] == "<=", hi=hi, hi_incl=m["op2"] == "<=")
    else:
        m = _ONE_SIDED.match(text)
        if not m:
            raise ValueError(f"{spec!r}: {text.strip()!r} is not a test; {INDICATOR_FORMS}")
        v, op = float(m["v"]), m["op"]
        bounds = (dict(lo=v, lo_incl=op == ">=") if op.startswith(">")
                  else dict(hi=v, hi_incl=op == "<="))
    column = m["abs"] or m["col"]
    if re.fullmatch(_NUM, column):
        raise ValueError(f"{spec!r}: {column!r} is a number, not a CV")
    return Clause(column=column, absolute=bool(m["abs"]), **bounds)


def parse_indicator(spec):
    """One -indicators entry (see INDICATOR_FORMS) as an Indicator;
    ValueError if it is not one of those forms."""
    name, sep, test = spec.partition("=")
    name = name.strip()
    if not sep or not _INDICATOR_NAME.match(name) or not test.strip():
        raise ValueError(f"{spec!r}: write it as {INDICATOR_FORMS}")
    m = _FLIP.match(test)
    flip = m is not None
    body = m["inner"] if flip else test
    parts = body.split("&")
    if not all(p.strip() for p in parts):
        raise ValueError(f"{spec!r}: an empty test; {INDICATOR_FORMS}")
    clauses = tuple(_parse_clause(p, spec) for p in parts)
    return Indicator(name=name, clauses=clauses, flip=flip, text=spec.strip())


@dataclass
class ColumnPlan:
    """The compared CVs.  `names` are as reported (A's naming, cos(...) for
    converted angles); `src_a`/`src_b` the file columns they come from;
    `angle` '', 'cos' or 'cos2'; `sym_b` the corrections applied to B only;
    `mirror_null` the CVs tested as A + B (see -mirror-null).

    A -indicators column is computed, not read: its `indicator` entry is
    (Indicator, positions of its clauses' CVs in the aux columns), None for
    a CV as it is.  `aux_a`/`aux_b` are the file columns the clauses read,
    each once, and `aux_sym` B's corrections of each (those of that CV)."""
    names: list
    src_a: list
    src_b: list
    angle: list
    sym_b: list
    mirror_null: np.ndarray
    indicator: list = None
    aux_a: list = field(default_factory=list)
    aux_b: list = field(default_factory=list)
    aux_sym: list = field(default_factory=list)

    def __post_init__(self):
        if self.indicator is None:
            self.indicator = [None] * len(self.names)

    @property
    def n(self):
        return len(self.names)

    def families(self):
        return [family(n) for n in self.names]


def read_header(path):
    """The column names on line 4 of a generate-cvs file."""
    with open(path, encoding="utf-8") as fh:
        lines = [fh.readline() for _ in range(4)]
    return lines[3].strip().lstrip("#").split()


def _ranges(nums, limit=8):
    """'1000-1499, 2003, ...': sorted path numbers as runs of consecutive ones."""
    nums = sorted(nums)
    runs, start = [], None
    for a, b in zip(nums, nums[1:] + [None]):
        start = a if start is None else start
        if b != a + 1:
            runs.append(f"{start}-{a}" if a != start else f"{a}")
            start = None
    return ", ".join(runs[:limit]) + (f", ... ({len(runs) - limit} more runs)"
                                      if len(runs) > limit else "")


def _headers(files, workers=16):
    """{path: column names} of every file that exists; the header is four
    lines, so reading every one of them is quick even for many paths."""
    from concurrent.futures import ThreadPoolExecutor

    def one(path):
        try:
            return path, read_header(path)
        except OSError:
            return path, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return {p: h for p, h in ex.map(one, files) if h is not None}


def side_header(stages, op_col, flag):
    """
    The columns of one side, from the header of EVERY CV file it will use.
    `stages` is [(stage name, {path number: file})], the weighted paths of
    each stage.  A column is

      kept      if every stage has it in at least one of its files.  Files
                that lack it (written before generate-cvs gained it) just add
                no values to it; they are reported per stage, with the path
                numbers to rerun generate-cvs -O on;
      left out  if some stage has it in none of its files, since that stage
                could not be compared on it at all; that stage is named.

    Returns (columns in the order the files list them, table of every path
    that lacks a column other paths of its stage have).
    """
    per_stage, union = [], {}
    for name, files in stages:
        heads = _headers(list(files.values()))
        if not heads:
            where = os.path.dirname(next(iter(files.values()), "")) or "its CV folder"
            raise typer.BadParameter(f"{flag}: none of the weighted paths of {name} has a "
                                     f"CV file in {where}")
        no_op = [p for p, h in heads.items() if op_col not in h]
        if len(no_op) == len(heads):
            raise typer.BadParameter(f"-op-col {op_col!r} is not a column of the CV files "
                                     f"of {name}, e.g. {no_op[0]}")
        for h in heads.values():
            union.update(dict.fromkeys(h))
        per_stage.append((name, files, heads))
    cols = [c for c in union if c != op_col]
    present = [{c for h in heads.values() for c in h} for _, _, heads in per_stage]
    kept = [c for c in cols if all(c in p for p in present)]
    dropped = [c for c in cols if c not in kept]
    if dropped:
        print(f"  [!] {flag}: {len(dropped)} column(s) left out, because no weighted path "
              f"of these stages has them:")
        for (name, _, _), p in zip(per_stage, present):
            lacks = [c for c in dropped if c not in p]
            if lacks:
                print(f"        {name}: {', '.join(lacks)}")
    rows = []
    for (name, files, heads), here in zip(per_stage, present):
        number = {f: pn for pn, f in files.items()}
        lacking = {}
        # Only columns the stage has in some files: one it has nowhere is
        # reported for the stage above, not path by path.
        for f, h in heads.items():
            miss = tuple(c for c in cols if c in here and c not in h)
            if miss:
                lacking.setdefault(miss, []).append(number[f])
                rows += [{"side": flag, "stage": name, "path": number[f],
                          "missing": ",".join(miss)}]
        # The same files usually lack the same columns: one line per set.
        for miss, pns in lacking.items():
            still = [c for c in miss if c in kept]
            if still:
                print(f"  [!] {flag} {name}: {len(pns)} of {len(heads)} weighted paths have no "
                      f"{', '.join(still)}, so they add no values to "
                      f"{'it' if len(still) == 1 else 'them'}: paths {_ranges(pns)}.  "
                      f"Rerun generate-cvs -O on them to include them.")
    return kept, pd.DataFrame(rows, columns=["side", "stage", "path", "missing"])


def plan_columns(cols_a, cols_b=None, cv_cols=None, exclude=None, angle_cols=(),
                 sym_angle_cols=(), swap_b=False, mirror_b=(), flip_b=(),
                 phase_b=(), mirror_null=(), indicators=()):
    """
    The CVs both sides have (B's names swapped first with swap_b), filtered by
    -cv-cols or -exclude.  Angle conversions match exact names, as in the
    other commands; the symmetry corrections and -mirror-null match
    substrings of the name (a trailing '$' anchors one to the end).  B's corrections are applied in the order flip,
    mirror, phase shift; a flip on a cos-converted column negates it
    (-cos t = cos(180 - t)).

    `indicators` (Indicator) are appended after the CVs, whatever -cv-cols
    and -exclude leave.  Their clauses read their CVs as aux columns, in the
    file's own units (no cos conversion), each with that CV's corrections in
    B before the test, so one test written in A's convention holds for both
    sides.
    """
    if cols_b is None:
        base, src_b = list(cols_a), list(cols_a)
    else:
        named_b = {swap_leaflets(c) if swap_b else c: c for c in cols_b}
        base = [c for c in cols_a if c in named_b]
        only_a = [c for c in cols_a if c not in named_b]
        only_b = [named_b[c] for c in named_b if c not in cols_a]
        # cols_a / cols_b are already what every stage of a side has, so a
        # column dropped above for one stage also shows up here.
        for side, other, cols in (("A", "B", only_a), ("B", "A", only_b)):
            if cols:
                print(f"  [!] {len(cols)} column(s) only in {side} are left out, as not "
                      f"every stage of {other} has them: " + ", ".join(cols))
        src_b = [named_b[c] for c in base]
    indicators = list(indicators or ())
    own = {k.name for k in indicators}
    if cv_cols:
        listed = [c for c in cv_cols if c not in own]   # indicators come anyway
        missing = [c for c in listed if c not in base]
        if missing:
            raise typer.BadParameter(f"-cv-cols: not a column of both sides: {missing}")
        keep = [base.index(c) for c in listed]
    else:
        keep = [i for i, c in enumerate(base) if not _matches_exclude(c, exclude)]
    if not keep and not indicators:
        raise typer.BadParameter("no CV column left to compare")

    def corrections(c):
        return tuple(op for op, subs in (("flip", flip_b), ("mirror", mirror_b),
                                         ("phase", phase_b))
                     if _matches_any(c, subs))

    names, angle, sym = [], [], []
    for i in keep:
        c = base[i]
        kind = "cos" if c in angle_cols else "cos2" if c in sym_angle_cols else ""
        names.append(f"{kind}({c})" if kind else c)
        angle.append(kind)
        sym.append(corrections(c))
    src_a_out, src_b_out = [base[i] for i in keep], [src_b[i] for i in keep]
    null = [_matches_any(base[i], mirror_null) for i in keep]
    ind = [None] * len(keep)
    aux_a, aux_b, aux_sym = [], [], []
    for k in indicators:
        if k.name in names or k.name in base:
            raise typer.BadParameter(f"-indicators: {k.name!r} is already the name of a CV "
                                     f"or of another indicator")
        idx = []
        for c in k.columns:
            if c not in base:
                raise typer.BadParameter(f"-indicators {k.text!r}: {c!r} is not a column "
                                         f"of both sides")
            if c not in aux_a:
                aux_a.append(c)
                aux_b.append(src_b[base.index(c)])
                aux_sym.append(corrections(c))
            idx.append(aux_a.index(c))
        names.append(k.name)
        angle.append("")
        sym.append(())
        # Nothing is read into the indicator's own column; it is computed.
        src_a_out.append(None)
        src_b_out.append(None)
        null.append(False)
        ind.append((k, tuple(idx)))
    # -angle-cols defaults to histograms' angle list, not all of which every
    # run has; an explicit -sym-angle-cols that misses is a typo.
    unknown = [c for c in sym_angle_cols if c not in base]
    if unknown:
        print(f"  [!] -sym-angle-cols: no such column: {', '.join(unknown)}")
    return ColumnPlan(names=names, src_a=src_a_out, src_b=src_b_out, angle=angle, sym_b=sym,
                      mirror_null=np.array(null, bool), indicator=ind,
                      aux_a=aux_a, aux_b=aux_b, aux_sym=aux_sym)


def transform(X, angle, sym=None, indicator=None, aux=None, aux_sym=None):
    """Angle conversions, then (for B) the symmetry corrections, in place.
    Then each -indicators column is computed from `aux` (frames, aux
    columns, in path order), after B's corrections `aux_sym` of those."""
    for j, kind in enumerate(angle):
        if kind == "cos":
            X[:, j] = np.cos(np.deg2rad(X[:, j]))
        elif kind == "cos2":
            X[:, j] = np.cos(np.deg2rad(X[:, j])) ** 2
    for j, ops in enumerate(sym or ()):
        for op in ops:
            if op == "flip":
                X[:, j] = -X[:, j] if angle[j] else 180.0 - X[:, j]
            elif op == "mirror":
                X[:, j] = -X[:, j]
            elif op == "phase":
                s = (X[:, j] + 180.0) % 360.0
                X[:, j] = np.where(s > 180.0, s - 360.0, s)
    if aux is not None and aux.shape[1]:
        aux = transform(np.array(aux, dtype=float), [""] * aux.shape[1], aux_sym)
    for j, spec in enumerate(indicator or ()):
        if spec is not None:
            test, idx = spec
            X[:, j] = test([aux[:, i] for i in idx])
    return X


def read_cv_file(path, op_col, src):
    """(path class, op, X) of one generate-cvs file: X holds the `src`
    columns (NaN for any the file lacks), one row per kept phase point."""
    with open(path, encoding="utf-8") as fh:
        label = fh.readline().strip().lstrip("#").strip().lower()
        ensemble = fh.readline().strip().lstrip("#").strip().lower()
        fh.readline()
        names = fh.readline().strip().lstrip("#").split()
    if label not in ("reactive", "non-reactive") or not ensemble.endswith("ensemble"):
        raise ValueError(f"not a generate-cvs file (line 1 {label!r}, line 2 {ensemble!r})")
    cls = ("minus" if ensemble.startswith("minus")
           else "reactive" if label == "reactive" else "unreactive")
    df = pd.read_csv(path, sep=r"\s+", skiprows=4, header=None, names=names)
    if op_col not in df:
        raise ValueError(f"no {op_col} column")
    n = len(df)
    X = np.empty((n, len(src)))
    for j, c in enumerate(src):
        X[:, j] = df[c].to_numpy(float) if c is not None and c in df else np.nan
    return cls, df[op_col].to_numpy(float), X


# ══════════════════════════════════════════════════════════════════════════════
# Grids: order-parameter bins and windows, value bins per CV
# ══════════════════════════════════════════════════════════════════════════════

def make_edges(lo, hi, step, extra=()):
    """Edges every `step` from lo to hi, plus every value of `extra` (the
    stage boundaries) inside the range, so no bin straddles one."""
    e = np.arange(lo, hi + 1e-9, step)
    inside = [x for x in extra if lo < x < hi]
    return np.unique(np.round(np.r_[e, hi, inside], 6))


@dataclass
class Grid:
    """Bins on the signed OP.  Below the first and above the last edge is one
    open bin each.  `op_edges` are the profile bins; `win_edges` the coarser
    windows the distributions are compared in."""
    op_edges: np.ndarray
    win_edges: np.ndarray

    @property
    def n_op(self):
        return len(self.op_edges) + 1

    @property
    def n_win(self):
        return len(self.win_edges) + 1

    def op_bin(self, op):
        return np.digitize(op, self.op_edges)

    def win_bin(self, op):
        return np.digitize(op, self.win_edges)

    @staticmethod
    def bounds(edges):
        return np.r_[-np.inf, edges], np.r_[edges, np.inf]


@dataclass
class ValueBins:
    """Value bins of every CV, laid end to end: CV j owns the bins
    offsets[j]:offsets[j+1] of a flat histogram."""
    edges: list
    discrete: np.ndarray

    @property
    def offsets(self):
        return np.r_[0, np.cumsum([len(e) - 1 for e in self.edges])].astype(int)

    @property
    def total(self):
        return int(self.offsets[-1])

    def index(self, X):
        """Flat bin of every value of X (frames, cvs); -1 for NaN.  Values
        outside a CV's range go to its outermost bin."""
        off = self.offsets
        out = np.full(X.shape, -1, np.int64)
        for j, e in enumerate(self.edges):
            x = X[:, j]
            ok = np.isfinite(x)
            k = np.clip(np.searchsorted(e, x[ok], side="right") - 1, 0, len(e) - 2)
            out[ok, j] = k + off[j]
        return out


def value_bins(samples, plan, n_bins, overrides=None, max_discrete=40):
    """
    Value edges per CV from a sample of frames (list of arrays (frames,
    cvs)): integer-valued CVs with at most `max_discrete` values get one bin
    per integer, with room above the largest sampled value; the others
    n_bins bins over the 0.1-99.9% quantiles, padded 5% on either side.  A
    -mirror-null CV gets a range symmetric about 0, so B can be mirrored bin
    for bin.  `overrides` {name: (min, max, n_bins)} (the -ranges file)
    replace the automatic range of a CV, by reported or file column name.
    """
    overrides = overrides or {}
    X = np.concatenate(samples) if samples else np.empty((0, plan.n))
    edges, discrete = [], []
    for j, name in enumerate(plan.names):
        if plan.indicator[j] is not None:
            # 0 and 1, whatever the sample held: a rare state may be absent
            # from it, and the CV's own -ranges entry is in other units.
            edges.append(np.array([-0.5, 0.5, 1.5]))
            discrete.append(True)
            continue
        o = overrides.get(name, overrides.get(plan.src_a[j]))
        x = X[:, j][np.isfinite(X[:, j])]
        if o is not None:
            lo, hi, nb = float(o[0]), float(o[1]), int(o[2])
            e, disc = np.linspace(lo, hi, nb + 1), False
        elif x.size == 0:
            e, disc = np.linspace(0.0, 1.0, n_bins + 1), False
        elif np.all(x == np.round(x)) and np.unique(x).size <= max_discrete:
            lo, hi = float(x.min()), float(x.max())
            room = max(1.0, np.ceil(0.25 * (hi - lo)))
            if plan.mirror_null[j]:
                m = max(abs(lo), abs(hi)) + room
                lo, hi = -m, m
            else:
                hi += room
                lo = lo - room if lo < 0 else max(0.0, lo - room) if lo > 0 else lo
            e, disc = np.arange(lo - 0.5, hi + 0.5 + 1e-9, 1.0), True
        else:
            lo, hi = np.quantile(x, [0.001, 0.999])
            if hi <= lo:
                lo, hi = lo - 0.5, hi + 0.5
            pad = 0.05 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            if plan.mirror_null[j]:
                m = max(abs(lo), abs(hi))
                lo, hi = -m, m
            e, disc = np.linspace(lo, hi, n_bins + 1), False
        edges.append(np.asarray(e, float))
        discrete.append(disc)
    return ValueBins(edges, np.array(discrete, bool))


def load_ranges(path):
    """{column: (min, max, n_bins)} from a cv_histograms-style -ranges TOML
    ([ranges] name = [min, max, n_bins])."""
    import tomli

    with open(path, "rb") as fh:
        cfg = tomli.load(fh)
    table = cfg.get("ranges", cfg)
    return {k: tuple(v) for k, v in table.items() if isinstance(v, (list, tuple)) and len(v) == 3}


# ══════════════════════════════════════════════════════════════════════════════
# Reduction: every path to binned sums
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SideConfig:
    """What a worker needs to reduce one side's files.  `direction` is one
    of DIRECTIONS: which frames of every path are kept (direction_mask)."""
    op_col: str
    src: list
    angle: list
    sym: list
    op_sign: float
    grid: Grid
    vbins: ValueBins
    correlations: bool
    direction: str = "all"
    indicator: list = None
    aux: list = field(default_factory=list)
    aux_sym: list = None


def _groups(keys):
    """(key, row indices) of every distinct key."""
    if keys.size == 0:
        return []
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    starts = np.r_[0, np.flatnonzero(np.diff(k)) + 1]
    ends = np.r_[starts[1:], k.size]
    return [(int(k[a]), order[a:b]) for a, b in zip(starts, ends)]


def direction_mask(op, direction):
    """
    The frames of one path moving `direction` along it: 'inbound' where the
    order parameter increases, 'outbound' where it decreases, by the sign of
    its gradient over the phase points in path order.  np.gradient takes
    central differences inside and one-sided ones at the ends, so reversing
    a path swaps the two sets exactly.  'all' keeps every frame; a path of
    fewer than two frames has no direction and keeps none.
    """
    op = np.asarray(op, dtype=float)
    if direction == "all":
        return np.ones(op.size, bool)
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {', '.join(DIRECTIONS)}, not {direction!r}")
    if op.size < 2:
        return np.zeros(op.size, bool)
    g = np.gradient(op)
    return g > 0 if direction == "inbound" else g < 0


def reduce_path(X, op_raw, unit, cfg: SideConfig):
    """
    One path's binned sums, unweighted (the path weight is applied later),
    in its two parts: frames above lambda_A and the [0-] frames below it
    that stage 0 keeps (cia.frame_parts).  With cfg.direction other than
    'all', only the frames moving that way count, decided on the whole path
    (op_raw, in path order) before it is split.  Returns
      prof  [(part, bins, N, C, S, S2)]  per OP bin: frames, finite values,
            sums of x and x^2 per CV
      hist  [(part, wins, N, H, C, S, S2)] the same per window, with H the
            flat value histogram of every CV
      maps  {part: (n_op, n_vbins)} the OP x value counts
      corr  {part: (n_win, 4, cv, cv)}  X'X, X'M, (X^2)'M, M'M per window,
            M the finite mask (pairwise-complete moments)
    """
    above, below = cia.frame_parts(op_raw, unit)
    if cfg.direction != "all":
        moving = direction_mask(op_raw, cfg.direction)
        above, below = above * moving, below * moving
    op = cfg.op_sign * op_raw
    grid, vb = cfg.grid, cfg.vbins
    fb, wb = grid.op_bin(op), grid.win_bin(op)
    fin = np.isfinite(X)
    X0 = np.where(fin, X, 0.0)
    VB = vb.index(X)
    TB = vb.total
    out = {"prof": [], "hist": [], "maps": {}, "corr": {}}
    for part, mult in zip(PARTS, (above, below)):
        sel = np.flatnonzero(mult > 0)
        if not sel.size:
            continue
        x0, f, v = X0[sel], fin[sel], VB[sel]
        for key, kind in ((fb[sel], "prof"), (wb[sel], "hist")):
            groups = _groups(key)
            ids = np.array([g for g, _ in groups])
            N = np.array([rows.size for _, rows in groups], float)
            C = np.stack([f[rows].sum(0) for _, rows in groups]).astype(np.float32)
            S = np.stack([x0[rows].sum(0) for _, rows in groups])
            S2 = np.stack([(x0[rows] ** 2).sum(0) for _, rows in groups])
            if kind == "prof":
                out["prof"].append((part, ids, N, C, S, S2))
                continue
            H = np.zeros((len(groups), TB), np.float32)
            for i, (_, rows) in enumerate(groups):
                flat = v[rows].ravel()
                H[i] = np.bincount(flat[flat >= 0], minlength=TB)
            out["hist"].append((part, ids, N, H, C, S, S2))
            if cfg.correlations:
                acc = np.zeros((grid.n_win, 4, X.shape[1], X.shape[1]))
                for g, rows in groups:
                    xr, mr = x0[rows], f[rows].astype(float)
                    acc[g, 0] = xr.T @ xr
                    acc[g, 1] = xr.T @ mr
                    acc[g, 2] = (xr ** 2).T @ mr
                    acc[g, 3] = mr.T @ mr
                out["corr"][part] = acc
        flat = (fb[sel][:, None] * TB + v)[v >= 0]
        out["maps"][part] = np.bincount(flat, minlength=grid.n_op * TB).reshape(grid.n_op, TB)
    return out


#: Held by each worker so its thread limit stays in force (see below).
_BLAS_LIMIT = None


def _one_blas_thread():
    """Worker initializer: one BLAS/OpenMP thread per worker process.

    The workers are forked from a parent whose numpy has already started its
    BLAS pool, by default one thread per core.  Every worker would then run
    that many threads for the small X'X products of the correlations, and
    -workers N would put N x cores threads on the machine, fighting over the
    cores and spin-waiting.  Parallelism here is over paths, so each worker
    gets one thread and -workers can go up to the core count.  Setting
    OMP_NUM_THREADS in the worker does not do it (the pool exists already);
    threadpoolctl changes the live pools.
    """
    global _BLAS_LIMIT
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:          # scikit-learn brings it; without it, as before
        return
    _BLAS_LIMIT = threadpool_limits(limits=1)


def _reduce_job(item):
    """Reduce one chunk of one stage's paths.  The OP x value maps and the
    correlation moments are summed here over the chunk, weighted (they have
    no error bars, so no per-path copy is kept); the binned sums are
    returned per path for the bootstrap."""
    cfg, members = item
    prof, hist, meta = [], [], []
    maps, corr = {}, {}
    for i, unit, path in members:
        try:
            cls, op, X = read_cv_file(path, cfg.op_col, list(cfg.src) + list(cfg.aux))
            n = len(cfg.src)
            X, aux = X[:, :n], X[:, n:]
            transform(X, cfg.angle, cfg.sym, cfg.indicator, aux, cfg.aux_sym)
            red = reduce_path(X, op, unit, cfg)
        except Exception as exc:
            meta.append({"index": i, "uid": unit.uid, "stage": unit.stage,
                         "error": f"{type(exc).__name__}: {exc}"})
            continue
        meta.append({"index": i, "uid": unit.uid, "stage": unit.stage, "path_class": cls,
                     "frames": len(op), "error": None})
        prof += [(i, *p) for p in red["prof"]]
        hist += [(i, *h) for h in red["hist"]]
        for part, m in red["maps"].items():
            for wkey, w in (("w", unit.weight), ("u", unit.weight_uniform)):
                key = (cls, part, wkey)
                maps[key] = maps.get(key, 0.0) + w * m
        for part, c in red["corr"].items():
            corr[(cls, part)] = corr.get((cls, part), 0.0) + unit.weight * c
    return prof, hist, maps, corr, meta


def _stack(entries, n_cv, TB=None):
    """Per-path entries -> flat arrays, one row per (path, part, bin)."""
    if not entries:
        out = {"unit": np.zeros(0, int), "bin": np.zeros(0, int), "below": np.zeros(0, bool),
               "N": np.zeros(0), "C": np.zeros((0, n_cv), np.float32),
               "S": np.zeros((0, n_cv)), "S2": np.zeros((0, n_cv))}
        if TB is not None:
            out["H"] = np.zeros((0, TB), np.float32)
        return out
    rows = lambda e: len(e[2])
    out = {"unit": np.concatenate([np.full(rows(e), e[0]) for e in entries]),
           "below": np.concatenate([np.full(rows(e), e[1] == "below") for e in entries]),
           "bin": np.concatenate([e[2] for e in entries]),
           "N": np.concatenate([e[3] for e in entries])}
    k = 4
    if TB is not None:
        out["H"] = np.concatenate([e[4] for e in entries])
        k = 5
    for name, off in (("C", 0), ("S", 1), ("S2", 2)):
        out[name] = np.concatenate([e[k + off] for e in entries])
    return out


class SideData:
    """
    One side (simulation) reduced: its units in MC order within each stage,
    their weights and classes, the per-path binned sums (`prof` per OP bin,
    `hist` per window), and the summed OP x value maps and correlation
    moments, keyed by (stage, class, part[, weighting]).
    """

    def __init__(self, label, units, factor_model, prof, hist, maps, corr, n_cv):
        self.label = label
        self.units = units
        self.n = len(units)
        self.w = np.array([u.weight for u in units], float)
        self.wu = np.array([u.weight_uniform for u in units], float)
        self.stage = np.array([u.stage for u in units], int)
        self.cls = np.array([u.path_class or "" for u in units])
        end = np.array([u.xi_end or "" for u in units])
        self.xi_R, self.xi_L = end == "R", end == "L"
        self.factor_model = factor_model
        self.prof, self.hist, self.maps, self.corr = prof, hist, maps, corr
        self.prof_groups = _groups(prof["bin"])
        self.hist_groups = _groups(hist["bin"])
        # Per-path sums over every bin, for the blocking analysis.
        self.unit_S = {p: np.zeros((self.n, n_cv)) for p in PARTS}
        self.unit_C = {p: np.zeros((self.n, n_cv)) for p in PARTS}
        for p, below in zip(PARTS, (False, True)):
            m = prof["below"] == below
            np.add.at(self.unit_S[p], prof["unit"][m], prof["S"][m])
            np.add.at(self.unit_C[p], prof["unit"][m], prof["C"][m].astype(float))

    def base(self, weighting):
        return self.w if weighting == "w" else self.wu

    def xi(self, W):
        """xi per row of W (replicate weights over all units): the weighted
        share of the entry stage's [0-] paths that end at lambda_A; 1 when
        there are none (single stage without [0-] paths, uniform weights)."""
        W = np.atleast_2d(W)
        r, l = W[:, self.xi_R].sum(1), W[:, self.xi_L].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(r + l > 0, np.where(r > 0, r / (r + l), np.nan), 1.0)

    def replicates(self, weighting, n_boot, block, rng):
        """(n_boot + 1, n) unit weights: row 0 the point estimate, then one
        block-bootstrap replicate per row, drawn within each stage
        (cia.block_bootstrap_indices), times the drawn stage factors."""
        base = self.base(weighting)
        W = np.empty((n_boot + 1, self.n))
        W[0] = base
        members = [np.flatnonzero(self.stage == s) for s in np.unique(self.stage)]
        for b in range(1, n_boot + 1):
            idx = np.concatenate([m[cia.block_bootstrap_indices(len(m), block, rng)]
                                  for m in members])
            W[b] = np.bincount(idx, minlength=self.n) * base
            if self.factor_model is not None:
                W[b] *= self.factor_model.draw(rng)[self.stage]
        return W


def reduce_side(label, units, factor_model, cfg, cv_file, workers, per_job):
    """Reduce every unit of one side in parallel; units whose file is missing
    or unreadable are reported and dropped, and their stage's other paths
    take over their weight (cia.rescale_missing).  Returns (SideData,
    per-path table)."""
    jobs, missing = [], []
    for s in sorted({u.stage for u in units}):
        members = []
        for i, u in enumerate(units):
            if u.stage != s:
                continue
            path = cv_file(u)
            if os.path.isfile(path):
                members.append((i, u, path))
            else:
                missing.append({"index": i, "uid": u.uid, "stage": s,
                                "error": f"no CV file {path}"})
        jobs += [(cfg, members[k:k + per_job]) for k in range(0, len(members), per_job)]
    prof, hist, meta = [], [], list(missing)
    maps, corr = {}, {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_one_blas_thread) as ex:
        for (_, members), (p, h, m, c, mt) in zip(jobs, ex.map(_reduce_job, jobs)):
            stage = members[0][1].stage
            prof += p
            hist += h
            meta += mt
            for key, arr in m.items():
                maps[(stage, *key)] = maps.get((stage, *key), 0.0) + arr
            for key, arr in c.items():
                corr[(stage, *key)] = corr.get((stage, *key), 0.0) + arr
    table = pd.DataFrame(meta).sort_values("index")
    ok = table[table["error"].isna()]
    for r in table[table["error"].notna()].head(10).itertuples():
        print(f"    [!] {label} stage {r.stage} path {r.uid}: {r.error}")
    n_bad = int(table["error"].notna().sum())
    if n_bad:
        print(f"    [!] {label}: {n_bad} path(s) left out")
    if ok.empty:
        raise RuntimeError(f"no path of {label} could be read; see the [!] messages above")
    keep = ok["index"].to_numpy(int)
    kept = [replace(units[i], path_class=c) for i, c in zip(keep, ok["path_class"])]
    new_w, miss = cia.rescale_missing(units, kept)
    new_u, _ = cia.rescale_missing(units, kept, attr="weight_uniform")
    final = [replace(a, weight_uniform=b.weight_uniform) for a, b in zip(new_w, new_u)]
    for r in miss[miss["paths_missing"] > 0].itertuples():
        print(f"  [!] {label} stage {r.stage}: {r.paths_missing} path(s) missing, "
              f"{r.weight_missing_share:.2%} of the stage's weight; the others are "
              f"scaled by {r.weight_rescale:.4f} to stand in for them")
    # The maps and moments were summed with the unscaled weights.
    scale_w = {u.stage: a.weight / u.weight for u, a in zip(kept, final) if u.weight > 0}
    scale_u = {u.stage: a.weight_uniform / u.weight_uniform for u, a in zip(kept, final)
               if u.weight_uniform > 0}
    maps = {k: v * (scale_w if k[3] == "w" else scale_u).get(k[0], 1.0) for k, v in maps.items()}
    corr = {k: v * scale_w.get(k[0], 1.0) for k, v in corr.items()}
    position = {int(i): n for n, i in enumerate(keep)}
    n_cv, TB = len(cfg.src), cfg.vbins.total
    P, H = _stack(prof, n_cv), _stack(hist, n_cv, TB)
    for arr in (P, H):
        arr["unit"] = np.array([position[int(i)] for i in arr["unit"]], int)
    side = SideData(label, final, factor_model, P, H, maps, corr, n_cv)
    mb = sum(a.nbytes for a in (*P.values(), *H.values())) / 1e6
    print(f"  {label}: {side.n} paths reduced, {len(P['unit'])} OP-bin and "
          f"{len(H['unit'])} window entries ({mb:.0f} MB)")
    return side, table.drop(columns="index")


def sample_frames(units, cv_file, cfg_angle, cfg_sym, src, op_col, n_paths,
                  frames_per_path=200, workers=8, indicator=None, aux=(), aux_sym=None):
    """Frames of up to n_paths evenly spaced paths, for the value ranges."""
    pick = [u for u in units if os.path.isfile(cv_file(u))]
    if len(pick) > n_paths:
        pick = [pick[i] for i in np.linspace(0, len(pick) - 1, n_paths).round().astype(int)]
    items = [(cv_file(u), op_col, src, cfg_angle, cfg_sym, frames_per_path, indicator,
              list(aux), aux_sym) for u in pick]
    with ProcessPoolExecutor(max_workers=workers, initializer=_one_blas_thread) as ex:
        return [x for x in ex.map(_sample_job, items, chunksize=8) if x is not None]


def _sample_job(item):
    path, op_col, src, angle, sym, n, indicator, aux, aux_sym = item
    try:
        _, _, X = read_cv_file(path, op_col, list(src) + aux)
    except Exception:
        return None
    k = len(src)
    X, A = X[:, :k], X[:, k:]
    transform(X, angle, sym, indicator, A, aux_sym)   # whole path: flips need it
    step = max(1, len(X) // n)
    return X[::step]


# ══════════════════════════════════════════════════════════════════════════════
# Views and weighted sums
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class View:
    """The paths of one side that one comparison counts: these stages and
    path classes, under one weighting ('w' WHAM, 'u' uniform)."""
    side: SideData
    label: str
    stages: tuple
    classes: tuple
    weighting: str

    @property
    def mask(self):
        return np.isin(self.side.stage, self.stages) & np.isin(self.side.cls, self.classes)

    @property
    def n_counted(self):
        return int((self.mask & (self.side.base(self.weighting) > 0)).sum())

    @property
    def n_frames(self):
        """Frames the counted paths bring; 0 when -direction left none."""
        counted = self.mask & (self.side.base(self.weighting) > 0)
        prof = self.side.prof
        return float(prof["N"][counted[prof["unit"]]].sum()) if prof["unit"].size else 0.0


def binned_sums(view, W, xi, table, groups, keys, n_bins):
    """
    Weighted sums per bin of every key of `keys` ({name: dtype}), for every
    row of W (point + replicates): the counted units' entries, the [0-] part
    below lambda_A divided by that row's xi.  One matrix product per bin and
    key.  The histograms are summed in float32 (they are the bulk of the
    data); the moments in float64, since an SD is a small difference of two
    large ones.  Also returns, per bin and CV, the raw frames with a value
    and the paths they come from: a CV that some files lack (an older
    generate-cvs) has fewer than the bin.
    """
    unit, below = table["unit"], table["below"]
    mask = view.mask
    out = {k: np.zeros((W.shape[0], n_bins) + table[k].shape[1:], dt) for k, dt in keys.items()}
    n_cv = table["C"].shape[1]
    frames = np.zeros((n_bins, n_cv))
    paths = np.zeros((n_bins, n_cv), int)
    for b, rows in groups:
        rows = rows[mask[unit[rows]]]
        if not rows.size:
            continue
        Wr = W[:, unit[rows]]
        bl = below[rows]
        if bl.any():
            Wr = Wr.copy()
            Wr[:, bl] /= xi[:, None]
        cast = {}
        for k, dt in keys.items():
            if dt not in cast:
                cast[dt] = Wr.astype(dt, copy=False)
            out[k][:, b] = cast[dt] @ table[k][rows].astype(dt, copy=False)
        C = table["C"][rows]
        frames[b] = C.sum(0)
        uniq, inv = np.unique(unit[rows], return_inverse=True)
        has = np.zeros((uniq.size, n_cv), bool)
        np.logical_or.at(has, inv, C > 0)
        paths[b] = has.sum(0)
    return out, frames, paths


def _ratio(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / b, np.nan)


def _bh(p):
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = cia.benjamini_hochberg(p[ok])
    return q


def paired_stats(pA, pB, rA, rB, ci=0.95):
    """
    A - B for flat arrays of point estimates (pA, pB) and their replicates
    (rA, rB: replicate x element): percentile CI of the replicate
    differences, a two-sided bootstrap p, and the replicate SDs.  As in
    chiral-compare, a p with no replicate on the other side of zero is taken
    from the normal tail with the bootstrap's own SD rather than floored at
    1/n_boot.
    """
    from scipy.special import ndtr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        diff = rA - rB
        lo, hi = np.nanpercentile(diff, [50 * (1 - ci), 50 * (1 + ci)], axis=0)
        n = np.sum(np.isfinite(diff), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            k = np.minimum(np.sum(diff <= 0, axis=0), np.sum(diff >= 0, axis=0))
            p = 2.0 * k / n
            z = np.abs(pA - pB) / np.nanstd(diff, axis=0)
            tail = (n > 0) & (k == 0) & np.isfinite(z)
            p = np.where(tail, 2.0 * ndtr(-z), np.clip(p, 1.0 / max(len(diff), 1), 1.0))
        p = np.where((n > 0) & np.isfinite(pA - pB), p, np.nan)
        return {"diff": pA - pB, "ci_lo": lo, "ci_hi": hi, "p_boot": p,
                "p_normal_tail": tail, "A_se": np.nanstd(rA, axis=0),
                "B_se": np.nanstd(rB, axis=0)}


# ══════════════════════════════════════════════════════════════════════════════
# The comparisons
# ══════════════════════════════════════════════════════════════════════════════

def _sign(plan):
    """+1, or -1 for the -mirror-null CVs, whose B is compared negated."""
    return np.where(plan.mirror_null, -1.0, 1.0)


def profile_table(sA, sB, fA, fB, nA, nB, plan, grid, min_frames):
    """
    profiles.csv: per CV and OP bin, the weighted mean of A and of B, A - B
    (A + B for the -mirror-null CVs, `tested`) with CI, p and BH q, the
    weighted SDs, d = tested difference / pooled SD, the frames and paths
    with a value of the CV on each side (fA/nA: bin x CV; fewer than the
    bin's for a CV that some files lack), and the occupancy (weighted share
    of the side's frames in the bin).  A CV is not tested in a bin where
    either side has fewer than `min_frames` such frames.
    """
    mA, mB = _ratio(sA["S"], sA["C"]), _ratio(sB["S"], sB["C"])     # (R, op, cv)
    sign = _sign(plan)
    T = lambda x: np.moveaxis(x, -1, -2)                            # -> (cv, op)
    R = mA.shape[0]
    st = paired_stats(T(mA[0]).ravel(), T(sign * mB[0]).ravel(),
                      T(mA[1:]).reshape(R - 1, -1), T(sign * mB[1:]).reshape(R - 1, -1))
    lo, hi = grid.bounds(grid.op_edges)
    n_cv, n_op = plan.n, grid.n_op

    def sd(s, m):
        return np.sqrt(np.clip(_ratio(s["S2"][0], s["C"][0]) - m[0] ** 2, 0, None))

    sdA, sdB = sd(sA, mA), sd(sB, mB)
    occ = lambda s: s["N"][0] / s["N"][0].sum() if s["N"][0].sum() > 0 else s["N"][0] * np.nan
    df = pd.DataFrame({
        "cv": np.repeat(plan.names, n_op),
        "family": np.repeat(plan.families(), n_op),
        "op_lo": np.tile(lo, n_cv), "op_hi": np.tile(hi, n_cv),
        "A": T(mA[0]).ravel(), "B": T(mB[0]).ravel(),
        "tested": np.repeat(np.where(plan.mirror_null, "A+B", "A-B"), n_op),
    })
    df["op_mid"] = np.where(np.isfinite(df["op_lo"] + df["op_hi"]),
                            0.5 * (df["op_lo"] + df["op_hi"]), np.nan)
    for k in ("diff", "ci_lo", "ci_hi", "p_boot", "p_normal_tail", "A_se", "B_se"):
        df[k] = st[k]
    df["sd_A"], df["sd_B"] = T(sdA).ravel(), T(sdB).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.sqrt(0.5 * (df["sd_A"] ** 2 + df["sd_B"] ** 2))
        df["d"] = np.where(pooled > 0, df["diff"] / pooled, np.nan)
    df["frames_A"], df["frames_B"] = fA.T.ravel(), fB.T.ravel()          # per CV
    df["paths_A"], df["paths_B"] = nA.T.ravel(), nB.T.ravel()
    df["occ_A"], df["occ_B"] = np.tile(occ(sA), n_cv), np.tile(occ(sB), n_cv)
    sparse = (df["frames_A"] < min_frames) | (df["frames_B"] < min_frames)
    df.loc[sparse, "p_boot"] = np.nan
    df["q_bh"] = _bh(df["p_boot"])
    return df[["cv", "family", "op_lo", "op_hi", "op_mid", "A", "B", "tested", "diff",
               "ci_lo", "ci_hi", "p_boot", "q_bh", "p_normal_tail", "A_se", "B_se",
               "sd_A", "sd_B", "d", "frames_A", "frames_B", "paths_A", "paths_B",
               "occ_A", "occ_B"]]


def _norm(h):
    with np.errstate(invalid="ignore", divide="ignore"):
        return h / h.sum(axis=-1, keepdims=True)


def quantiles(frac, edges, ps):
    """Quantiles of normalised histograms frac (..., bins) on `edges`, the
    density taken as uniform within a bin.  NaN for an empty histogram."""
    cdf = np.cumsum(frac, axis=-1)
    prev = np.concatenate([np.zeros(cdf.shape[:-1] + (1,)), cdf[..., :-1]], axis=-1)
    nb = frac.shape[-1]
    out = []
    for p in ps:
        k = np.clip(np.sum(cdf < p - 1e-12, axis=-1), 0, nb - 1)
        c0 = np.take_along_axis(prev, k[..., None], -1)[..., 0]
        c1 = np.take_along_axis(cdf, k[..., None], -1)[..., 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(c1 > c0, (p - c0) / (c1 - c0), 0.5)
        v = edges[k] + np.clip(t, 0, 1) * (edges[k + 1] - edges[k])
        out.append(np.where(np.isfinite(cdf[..., -1]), v, np.nan))
    return out


def wasserstein(fa, fb, edges):
    """W1 between two normalised histograms on the same edges, in the CV's
    units: the area between their cumulative distributions."""
    return np.sum(np.abs(np.cumsum(fa, -1) - np.cumsum(fb, -1)) * np.diff(edges), axis=-1)


def distribution_tables(hA, hB, wA, wB, plan, vbins, grid, min_frames):
    """
    The value distributions per CV and window (rows of each side's window
    histogram normalised to 1), compared two ways:

      distributions.csv  bin by bin, A - B with CI, p and q (BH over the
                         table), as chiral-compare's histograms.  The bins of
                         one histogram share their frames, so a run of
                         significant bins is one effect, not several.
      shifts.csv         per CV and window: mean, SD, 10th/50th/90th
                         percentile of A and B and their differences, each
                         measure with its own BH q over the table, and the
                         Wasserstein distance W1 (with its bootstrap CI, which
                         is biased upward like any distance; W1_rel = W1 /
                         pooled SD).  Mean and SD are exact (from the sums),
                         the percentiles from the histogram.

    For the -mirror-null CVs B is mirrored (value -> -value), on value bins
    symmetric about 0.  A CV is not tested in a window where either side has
    fewer than `min_frames` frames with a value of it (wA, wB: window x CV).
    Returns (distributions, shifts, arrays for distributions.npz).
    """
    off = vbins.offsets
    R = hA["H"].shape[0]
    wlo, whi = grid.bounds(grid.win_edges)
    tested_win = (wA >= min_frames) & (wB >= min_frames)
    rows, shifts = [], []
    arrays = {"A_1d": np.full((grid.n_win, vbins.total), np.nan),
              "B_1d": np.full((grid.n_win, vbins.total), np.nan),
              "A_se_1d": np.full((grid.n_win, vbins.total), np.nan),
              "B_se_1d": np.full((grid.n_win, vbins.total), np.nan)}
    sign = _sign(plan)
    for j, name in enumerate(plan.names):
        sl = slice(off[j], off[j + 1])
        e = vbins.edges[j]
        fa, fb = _norm(hA["H"][..., sl].astype(float)), _norm(hB["H"][..., sl].astype(float))
        fb_t = fb[..., ::-1] if plan.mirror_null[j] else fb
        arrays["A_1d"][:, sl], arrays["B_1d"][:, sl] = fa[0], fb[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            arrays["A_se_1d"][:, sl] = np.nanstd(fa[1:], axis=0)
            arrays["B_se_1d"][:, sl] = np.nanstd(fb[1:], axis=0)
        nb = e.size - 1
        st = paired_stats(fa[0].ravel(), fb_t[0].ravel(),
                          fa[1:].reshape(R - 1, -1), fb_t[1:].reshape(R - 1, -1))
        p = np.where(np.repeat(tested_win[:, j], nb), st["p_boot"], np.nan)
        rows.append(pd.DataFrame({
            "cv": name, "win_lo": np.repeat(wlo, nb), "win_hi": np.repeat(whi, nb),
            "value_lo": np.tile(e[:-1], grid.n_win), "value_hi": np.tile(e[1:], grid.n_win),
            "tested": "A vs mirrored B" if plan.mirror_null[j] else "A-B",
            "A": fa[0].ravel(), "B": fb[0].ravel(), "diff": st["diff"],
            "ci_lo": st["ci_lo"], "ci_hi": st["ci_hi"], "p_boot": p,
            "A_se": st["A_se"], "B_se": st["B_se"],
            "frames_A": np.repeat(wA[:, j], nb), "frames_B": np.repeat(wB[:, j], nb)}))

        mean = lambda h, k: _ratio(h["S"][..., k], h["C"][..., k])
        sd = lambda h, k, m: np.sqrt(np.clip(_ratio(h["S2"][..., k], h["C"][..., k]) - m ** 2,
                                             0, None))
        meA, meB = mean(hA, j), sign[j] * mean(hB, j)                  # (R, win)
        meas = {"mean": (meA, meB), "sd": (sd(hA, j, meA), sd(hB, j, meB))}
        qa, qb = quantiles(fa, e, [q for _, q in QUANTILES]), quantiles(fb_t, e, [q for _, q in QUANTILES])
        for (qn, _), a, b in zip(QUANTILES, qa, qb):
            meas[qn] = (a, b)
        row = {"cv": name, "win_lo": wlo, "win_hi": whi,
               "tested": "A vs mirrored B" if plan.mirror_null[j] else "A-B",
               "frames_A": wA[:, j], "frames_B": wB[:, j]}
        for mname, (a, b) in meas.items():
            s = paired_stats(a[0], b[0], a[1:], b[1:])
            row[f"{mname}_A"], row[f"{mname}_B"] = a[0], b[0]
            row[f"{mname}_diff"] = s["diff"]
            row[f"{mname}_lo"], row[f"{mname}_hi"] = s["ci_lo"], s["ci_hi"]
            row[f"{mname}_p"] = np.where(tested_win[:, j], s["p_boot"], np.nan)
        w1 = wasserstein(fa, fb_t, e)                                  # (R, win)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            row["w1"] = w1[0]
            row["w1_lo"], row["w1_hi"] = np.nanpercentile(w1[1:], [2.5, 97.5], axis=0)
            pooled = np.sqrt(0.5 * (meas["sd"][0][0] ** 2 + meas["sd"][1][0] ** 2))
            row["w1_rel"] = np.where(pooled > 0, w1[0] / pooled, np.nan)
        shifts.append(pd.DataFrame(row))
    dist = pd.concat(rows, ignore_index=True)
    dist["q_bh"] = _bh(dist["p_boot"])
    # q back onto (window, flat value bin): the rows run cv, window, bin.
    q = np.full((grid.n_win, vbins.total), np.nan)
    pos = 0
    for j in range(plan.n):
        nb = off[j + 1] - off[j]
        q[:, off[j]:off[j + 1]] = dist["q_bh"].to_numpy()[pos:pos + grid.n_win * nb].reshape(
            grid.n_win, nb)
        pos += grid.n_win * nb
    arrays["q_1d"] = q
    shift = pd.concat(shifts, ignore_index=True)
    for mname in SHIFT_MEASURES:
        shift[f"{mname}_q"] = _bh(shift[f"{mname}_p"])
    order = ["cv", "win_lo", "win_hi", "tested", "frames_A", "frames_B"]
    for mname in SHIFT_MEASURES:
        order += [f"{mname}_{k}" for k in ("A", "B", "diff", "lo", "hi", "p", "q")]
    order += ["w1", "w1_lo", "w1_hi", "w1_rel"]
    return dist, shift[order], arrays


def ranking_table(prof, shift):
    """
    ranking.csv: one row per CV, ordered by its largest |d| among the
    significant OP bins (q < 0.05; CVs without one follow, by their largest
    |d| in any tested bin).  Also where that bin is, the difference there,
    how many bins were tested and significant, the occupancy-weighted mean
    |d|, and from shifts.csv the windows where the mean, SD or median
    differ significantly and the largest W1_rel.
    """
    rows = []
    for name, g in prof.groupby("cv", sort=False):
        t = g[g["p_boot"].notna() & g["d"].notna()]
        sig = t[t["q_bh"] < 0.05]
        best = sig.loc[sig["d"].abs().idxmax()] if len(sig) else (
            t.loc[t["d"].abs().idxmax()] if len(t) else None)
        occ = 0.5 * (t["occ_A"].fillna(0) + t["occ_B"].fillna(0))
        rows.append({
            "cv": name, "family": g["family"].iloc[0],
            "score": float(abs(best["d"])) if len(sig) else 0.0,
            "max_abs_d": float(t["d"].abs().max()) if len(t) else np.nan,
            "d_at_max": float(best["d"]) if best is not None else np.nan,
            "diff_at_max": float(best["diff"]) if best is not None else np.nan,
            "op_lo_at_max": float(best["op_lo"]) if best is not None else np.nan,
            "op_hi_at_max": float(best["op_hi"]) if best is not None else np.nan,
            "significant_at_max": bool(len(sig)),
            "bins_tested": len(t), "bins_significant": len(sig),
            "mean_abs_d": (float((occ * t["d"].abs()).sum() / occ.sum())
                           if len(t) and occ.sum() > 0 else np.nan),
        })
    rank = pd.DataFrame(rows)
    if shift is not None and len(shift):
        s = shift.copy()
        sig = (s[["mean_q", "sd_q", "median_q"]] < 0.05)
        s["any"] = sig.any(axis=1)
        agg = s.groupby("cv", sort=False).agg(
            windows_shifted=("mean_q", lambda q: int((q < 0.05).sum())),
            windows_spread=("sd_q", lambda q: int((q < 0.05).sum())),
            windows_any=("any", "sum"),
            max_w1_rel=("w1_rel", "max"))
        rank = rank.merge(agg, left_on="cv", right_index=True, how="left")
    rank = rank.sort_values(["score", "bins_significant", "max_abs_d"],
                            ascending=False, na_position="last").reset_index(drop=True)
    rank.insert(0, "rank", np.arange(1, len(rank) + 1))
    return rank


def correlation_matrices(acc):
    """Pairwise-complete correlations from summed moments (win, 4, cv, cv):
    per window, and pooled within windows (each window's own means
    removed, so a common trend along the OP does not count as coupling)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        XtX, XtM, X2tM, n = acc[:, 0], acc[:, 1], acc[:, 2], acc[:, 3]
        mi = XtM / n
        mj = np.swapaxes(mi, -1, -2)
        cov = XtX / n - mi * mj
        vi = X2tM / n - mi ** 2
        vj = np.swapaxes(vi, -1, -2)
        r_win = cov / np.sqrt(vi * vj)
        ok = n > 0
        cp = np.where(ok, n * cov, 0).sum(0)
        Vi = np.where(ok, n * vi, 0).sum(0)
        pooled = cp / np.sqrt(Vi * np.swapaxes(Vi, -1, -2))
    return np.clip(r_win, -1, 1), np.clip(pooled, -1, 1), np.diagonal(n, axis1=1, axis2=2)


def _view_acc(view, xi_point, store, keys_extra=()):
    """Sum of a side's chunk-summed arrays (maps or corr) over the view's
    stages and classes, the [0-] part divided by the point xi."""
    total = None
    for key, arr in store.items():
        stage, cls, part = key[:3]
        if key[3:] != tuple(keys_extra) or stage not in view.stages or cls not in view.classes:
            continue
        a = arr / xi_point if part == "below" else arr
        total = a if total is None else total + a
    return total


def blocking_table(view):
    """
    chiral-compare's blocking analysis on every CV: per stage, the
    statistical inefficiency s(B) of the CV's mean over the whole path (all
    bins pooled) for blocks of B consecutive paths.  Same columns as there,
    so cia.choose_block picks the block.
    """
    side = view.side
    base = side.base(view.weighting)
    xi = side.xi(base)[0]
    S = side.unit_S["above"] + side.unit_S["below"] / xi
    C = side.unit_C["above"] + side.unit_C["below"] / xi
    mask = view.mask
    names = np.arange(S.shape[1])
    rows = []
    for s in view.stages:
        members = np.flatnonzero(side.stage == s)
        w = base[members] * mask[members]
        if not np.any(w > 0):
            continue
        Sm, Cm = S[members], C[members]
        tot = (w[:, None] * Cm).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            R_ = (w[:, None] * Sm).sum(0) / tot
        z = w[:, None] * (Sm - R_ * Cm)
        for B in cia.BLOCK_SIZES:
            m = len(z) // B
            if m < 2:
                break
            zz = z[: m * B]
            denom = (zz ** 2).sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                ineff = (zz.reshape(m, B, -1).sum(1) ** 2).sum(0) / denom
            good = (tot > 0) & (denom > 0)
            for k in names[good]:
                rows.append((view.label, int(s), int(k), B, m, float(ineff[k]),
                             m >= cia.MIN_BLOCKS))
    # Typed even when empty (a side whose CVs have no frames in these stages),
    # so concatenating it with the other side keeps the numeric columns numeric.
    return pd.DataFrame(rows, columns=["side", "stage", "observable", "block",
                                       "n_blocks", "inefficiency", "reliable"]).astype(
        {"stage": int, "observable": int, "block": int, "n_blocks": int,
         "inefficiency": float, "reliable": bool})


# ══════════════════════════════════════════════════════════════════════════════
# Folders
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Folder:
    sub: str
    title: str
    stages: tuple
    classes_a: tuple
    classes_b: tuple


def folder_plan(stages, stage_names, two_sides):
    """The comparisons written, each to its folder: chiral-compare's layout.
    Two simulations: all paths, each stage, and the reactive and the
    unreactive paths pooled and per stage.  One simulation: reactive vs
    unreactive, pooled and per stage.  With one stage the per-stage folders
    would repeat the pooled ones and are left out."""
    allst = tuple(stages)
    if len(allst) < 2:
        stages = ()
    if not two_sides:
        out = [Folder("", "reactive vs unreactive paths", allst, ("reactive",), ("unreactive",))]
        out += [Folder(stage_names[k], f"reactive vs unreactive paths, {stage_names[k]}",
                       (k,), ("reactive",), ("unreactive",)) for k in stages]
        return out
    out = [Folder("", "all paths", allst, CLASSES, CLASSES)]
    out += [Folder(stage_names[k], f"all paths, {stage_names[k]}", (k,), CLASSES, CLASSES)
            for k in stages]
    for cls in PATH_SUBSETS:
        out.append(Folder(cls, f"{cls} paths only", allst, (cls,), (cls,)))
        out += [Folder(os.path.join(cls, stage_names[k]), f"{cls} paths only, {stage_names[k]}",
                       (k,), (cls,), (cls,)) for k in stages]
    return out


#: Every table a folder gets, checked up front so -O is needed to replace them.
FOLDER_OUTPUTS = ("profiles.csv", "distributions.csv", "distributions.npz", "shifts.csv",
                  "ranking.csv", "correlations.npz", "blocking.csv", "block_choice.csv",
                  "meta.json")
TOP_OUTPUTS = ("columns.csv", "paths_A.csv", "paths_B.csv", "stages_A.csv", "stages_B.csv",
               "path_classes.csv", "paths_missing_columns.csv")


def analyse_folder(sideA, sideB, folder, weighting, labels, out_dir, plan, vbins, grid,
                   n_boot, block, seed, min_frames, hist_min_frames, correlations,
                   stage_edges, direction="all"):
    """Every table for one folder (see the module docstring), written to
    out_dir.  `block` None: chosen by the folder's own blocking analysis.
    `direction` only names the frames the sides were reduced with."""
    vA = View(sideA, labels[0], folder.stages, folder.classes_a, weighting)
    vB = View(sideB, labels[1], folder.stages, folder.classes_b, weighting)
    for v, label in ((vA, labels[0]), (vB, labels[1])):
        if not v.n_counted or not v.n_frames:
            what = "paths" if not v.n_counted else f"{direction} frames"
            print(f"\n  [!] {folder.title}{DIRECTION_TITLE[direction]}: no {what} in "
                  f"{label}; skipped")
            return None
    os.makedirs(out_dir, exist_ok=True)
    tag = (("unweighted (no WHAM reweighting), " if weighting == "u" else "") + folder.title
           + DIRECTION_TITLE[direction])
    print(f"\n  {tag}: {labels[0]} {vA.n_counted}, {labels[1]} {vB.n_counted} paths "
          f"(written to {out_dir})")
    blocking = pd.concat([blocking_table(vA), blocking_table(vB)], ignore_index=True)
    named = blocking.assign(observable=[plan.names[k] for k in blocking["observable"]])
    named.to_csv(os.path.join(out_dir, "blocking.csv"), index=False)
    chosen, _ = cia._choose_and_report(named, block, out_dir, tag)

    shared = sideB is sideA
    rngA = np.random.default_rng([seed, 0])
    WA = sideA.replicates(weighting, n_boot, chosen, rngA)
    WB = WA if shared else sideB.replicates(weighting, n_boot,
                                           chosen, np.random.default_rng([seed, 1]))
    xiA, xiB = sideA.xi(WA), sideB.xi(WB)
    keys = dict.fromkeys(("N", "C", "S", "S2"), np.float64)
    sA, fA, nA = binned_sums(vA, WA, xiA, sideA.prof, sideA.prof_groups, keys, grid.n_op)
    sB, fB, nB = binned_sums(vB, WB, xiB, sideB.prof, sideB.prof_groups, keys, grid.n_op)
    prof = profile_table(sA, sB, fA, fB, nA, nB, plan, grid, min_frames)
    hkeys = {"H": np.float32, "C": np.float64, "S": np.float64, "S2": np.float64}
    hA, wA, _ = binned_sums(vA, WA, xiA, sideA.hist, sideA.hist_groups, hkeys, grid.n_win)
    hB, wB, _ = binned_sums(vB, WB, xiB, sideB.hist, sideB.hist_groups, hkeys, grid.n_win)
    dist, shift, arrays = distribution_tables(hA, hB, wA, wB, plan, vbins, grid,
                                              hist_min_frames)
    rank = ranking_table(prof, shift)

    mA = _view_acc(vA, xiA[0], sideA.maps, (weighting,))
    mB = _view_acc(vB, xiB[0], sideB.maps, (weighting,))
    off = vbins.offsets
    for side, m in (("A", mA), ("B", mB)):
        m = np.zeros((grid.n_op, vbins.total)) if m is None else m
        arrays[f"{side}_2d"] = np.concatenate(
            [_norm(m[:, off[j]:off[j + 1]]) for j in range(plan.n)], axis=1)
    arrays.update({
        "names": np.array(plan.names, str), "families": np.array(plan.families(), str),
        "mirror": plan.mirror_null, "discrete": vbins.discrete,
        "bin_offsets": off,
        "edge_offsets": np.r_[0, np.cumsum([len(e) for e in vbins.edges])].astype(int),
        "value_edges": np.concatenate(vbins.edges),
        "op_edges": grid.op_edges, "win_edges": grid.win_edges,
        "frames_A_op": fA, "frames_B_op": fB, "frames_A_win": wA, "frames_B_win": wB,
    })
    np.savez_compressed(os.path.join(out_dir, "distributions.npz"), **arrays)

    if correlations and weighting == "w":
        cA = _view_acc(vA, xiA[0], sideA.corr)
        cB = _view_acc(vB, xiB[0], sideB.corr)
        if cA is not None and cB is not None:
            rwA, rpA, nwA = correlation_matrices(cA)
            rwB, rpB, nwB = correlation_matrices(cB)
            np.savez_compressed(
                os.path.join(out_dir, "correlations.npz"),
                names=np.array(plan.names, str), families=np.array(plan.families(), str),
                A=rpA, B=rpB, A_win=rwA.astype(np.float32), B_win=rwB.astype(np.float32),
                frames_A_win=nwA, frames_B_win=nwB, win_edges=grid.win_edges)

    for name, df in (("profiles", prof), ("distributions", dist), ("shifts", shift),
                     ("ranking", rank)):
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    meta = {"labels": list(labels), "title": tag, "weighting": weighting,
            "direction": direction, "n_boot": n_boot, "block": int(chosen), "min_frames": min_frames,
            "hist_min_frames": hist_min_frames, "stage_edges": [float(e) for e in stage_edges],
            "paths": [vA.n_counted, vB.n_counted]}
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    _report(rank, prof, shift, tag, labels)
    return rank


def _report(rank, prof, shift, tag, labels):
    tested = prof["q_bh"].notna().sum()
    hits = int((prof["q_bh"] < 0.05).sum())
    n_cv = int((rank["bins_significant"] > 0).sum())
    print(f"  {tag}: {hits} of {tested} tested CV x OP bins differ (q < 0.05), "
          f"in {n_cv} of {len(rank)} CVs")
    top = rank[rank["score"] > 0].head(10)
    if len(top):
        print(f"  largest significant effects (d = ({labels[0]} - {labels[1]}) / pooled SD):")
        for r in top.itertuples():
            print(f"    {r.cv:28s} d = {r.d_at_max:+6.2f} at OP {r.op_lo_at_max:g} "
                  f"to {r.op_hi_at_max:g}  ({r.bins_significant} bins)")


def path_class_table(sides, labels):
    rows = []
    for side, label in zip(sides, labels):
        for s in np.unique(side.stage):
            m = side.stage == s
            total = side.w[m].sum()
            for cls in CLASSES:
                c = m & (side.cls == cls)
                rows.append({"side": label, "stage": int(s), "path_class": cls,
                             "paths": int(c.sum()),
                             "weight_share": side.w[c].sum() / total if total > 0 else np.nan})
    return pd.DataFrame(rows)


def compare(sideA, sideB, labels, out_dir, plan, vbins, grid, stage_names, n_boot=500,
            block=None, seed=0, min_frames=50, hist_min_frames=200, correlations=True,
            unweighted=True, stage_edges=(), direction="all"):
    """Every folder of folder_plan, WHAM-weighted and (unweighted) without
    the reweighting under out_dir/unweighted.  sideB None: reactive vs
    unreactive within sideA.  `direction`: which frames the sides were
    reduced with, for the titles and meta.json."""
    two = sideB is not None
    sideB = sideB if two else sideA
    stages = sorted(set(sideA.stage) & set(sideB.stage))
    folders = folder_plan(stages, stage_names, two)
    weightings = ("w", "u") if unweighted else ("w",)
    results = {}
    for weighting in weightings:
        top = out_dir if weighting == "w" else os.path.join(out_dir, UNWEIGHTED_DIR)
        print(f"\n  ══ {'WHAM-weighted' if weighting == 'w' else 'unweighted'} ══")
        for f in folders:
            results[(weighting, f.sub)] = analyse_folder(
                sideA, sideB, f, weighting, labels, os.path.join(top, f.sub), plan, vbins,
                grid, n_boot, block, seed, min_frames, hist_min_frames, correlations,
                stage_edges, direction)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _split(value):
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


def _data_paths(data_file):
    """Every path number in an infretis_data.txt (first column)."""
    out = []
    with open(data_file) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            head = line.split(None, 1)[0]
            if head.isdigit():
                out.append(int(head))
    return out


def _cv_units(spec, path_numbers):
    """
    setup_stages' units for one stage: every path in its infretis_data.txt,
    each pointing at its order.txt (read for xi) and with `cache_dir` the CV
    folder its file is in.  A weighted path without a CV file is reported and
    its weight passed on (reduce_side).
    """
    pns = _data_paths(spec.data_file)
    if path_numbers is not None:
        keep = set(path_numbers)
        pns = [p for p in pns if p in keep]
    load = os.path.join(spec.run_dir, "load")
    return [cia.Unit(pn, [], order_file=os.path.join(load, str(pn), "order.txt"),
                     cache_dir=spec.cache_dir) for pn in dict.fromkeys(pns)]


def _cv_file(unit):
    return os.path.join(unit.cache_dir, f"{unit.uid}.txt")


def _side_specs(runs, data, toml, cv_dir, side):
    run_dirs = _split(runs)
    data_files = cia._per_run(data, run_dirs, f"-data-{side}")
    tomls = (cia._per_run(toml, run_dirs, f"-toml-{side}") if toml
             else [cia._find_toml(r) for r in run_dirs])
    cv_dirs = cia._per_run(cv_dir, run_dirs, f"-cv-dir-{side}", exists=os.path.isdir)
    return [cia.StageSpec(r, d, t, c) for r, d, t, c in zip(run_dirs, data_files, tomls, cv_dirs)]


def _parse_range(value):
    if str(value).strip().lower() == "auto":
        return None
    try:
        lo, hi = (float(v) for v in value.split(","))
    except ValueError:
        raise typer.BadParameter(f"-op-range: 'auto' or 'min,max', not {value!r}")
    if hi <= lo:
        raise typer.BadParameter("-op-range: max must exceed min")
    return lo, hi


def _stage_lambdas(specs, sign):
    """Signed (lambda_-1 of the entry stage, every lambda_A and lambda_B)."""
    lams, lm1 = [], None
    for k, spec in enumerate(specs):
        lam_a, lam_b, lam_m1 = _read_toml(spec.toml_file)
        lams += [sign * lam_a, sign * lam_b]
        if k == 0 and lam_m1 is not None:
            lm1 = sign * lam_m1
    return lams, lm1


def _read_toml(path):
    from .cv_generation import read_toml

    return read_toml(path)


def safe_name(name):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def cv_compare(
    # ── Input data ────────────────────────────────────────────────────────
    runs_a: Annotated[Optional[str], typer.Option("-runs-a", help="Run directories of simulation A, comma-separated, in chain order (entry, internal, escape). Each holds the CV folder (-cv-dir-a), load/<path>/order.txt (for xi), the wham/ folder inftools wrote, the toml and infretis_data.txt. Required", rich_help_panel=panels.INPUT)] = None,
    runs_b: Annotated[Optional[str], typer.Option("-runs-b", help="Run directories of simulation B, same rules. Leave out to compare the reactive with the unreactive paths of A", rich_help_panel=panels.INPUT)] = None,
    label_a: Annotated[str, typer.Option("-label-a", help="Name of A in tables and figures, e.g. L", rich_help_panel=panels.INPUT)] = "A",
    label_b: Annotated[str, typer.Option("-label-b", help="Name of B, e.g. D", rich_help_panel=panels.INPUT)] = "B",
    cv_dir_a: Annotated[str, typer.Option("-cv-dir-a", help="CV folder of each A stage: one relative path looked for inside every run directory, or one per run, comma-separated", rich_help_panel=panels.INPUT)] = DEFAULT_CV_DIR,
    cv_dir_b: Annotated[str, typer.Option("-cv-dir-b", help="CV folder of each B stage; see -cv-dir-a", rich_help_panel=panels.INPUT)] = DEFAULT_CV_DIR,
    data_a: Annotated[str, typer.Option("-data-a", help="infretis_data.txt of each A stage, same rules as -cv-dir-a", rich_help_panel=panels.INPUT)] = "infretis_data.txt",
    data_b: Annotated[str, typer.Option("-data-b", help="infretis_data.txt of each B stage", rich_help_panel=panels.INPUT)] = "infretis_data.txt",
    toml_a: Annotated[Optional[str], typer.Option("-toml-a", help="infretis.toml of each A stage, same rules; unset uses RUN_DIR/infretis.toml, else RUN_DIR/post/infretis.toml", rich_help_panel=panels.INPUT)] = None,
    toml_b: Annotated[Optional[str], typer.Option("-toml-b", help="infretis.toml of each B stage; see -toml-a", rich_help_panel=panels.INPUT)] = None,
    op_col: Annotated[str, typer.Option("-op-col", help="Order-parameter column of the CV files; it is the axis, on the same scale as the toml interfaces", rich_help_panel=panels.INPUT)] = OP_COL,

    # ── Dataset construction ──────────────────────────────────────────────
    path_numbers: Annotated[Optional[str], typer.Option("-path-numbers", help="Only these path numbers of every stage, e.g. '1-5000,6001'; default all", rich_help_panel=panels.DATASET)] = None,
    nskip: Annotated[int, typer.Option("-nskip", help="Burn-in rows of each infretis_data.txt skipped for the WHAM weights", rich_help_panel=panels.DATASET)] = 1000,
    wham_subdir: Annotated[str, typer.Option("-wham-subdir", help="Name of the wham folder inside each run directory", rich_help_panel=panels.DATASET)] = "wham",
    pcross_col: Annotated[int, typer.Option("-pcross-col", help="Column of wham/Pcross.txt holding the total crossing probability, as in permeation: 3 is P-wham2, 1 is P-wham", rich_help_panel=panels.DATASET)] = 3,
    op_range: Annotated[str, typer.Option("-op-range", help="'min,max' of the binned OP axis, or 'auto': from the entry stage's lambda_-1 to the last stage's lambda_B. Beyond it is one open bin on either side", rich_help_panel=panels.DATASET)] = "auto",
    op_step: Annotated[float, typer.Option("-op-step", help="Width of the OP bins of the profiles and the OP x value maps; every stage boundary is an extra edge", rich_help_panel=panels.DATASET)] = 2.5,
    window_step: Annotated[float, typer.Option("-window-step", help="Width of the OP windows the distributions are compared in (coarser than -op-step: a distribution needs more frames than a mean); stage boundaries are edges too", rich_help_panel=panels.DATASET)] = 5.0,
    direction: Annotated[str, typer.Option("-direction", help="Which frames of every path count: 'all'; 'inbound', where the OP increases along the path (the way a crossing goes); or 'outbound', where it decreases (back toward the entry side). From the sign of each path's own OP gradient. For unreactive paths the two have the same distributions (time reversal), so their difference is a convergence check; see the module docs", rich_help_panel=panels.DATASET)] = "all",

    # ── CV selection ──────────────────────────────────────────────────────
    cv_cols: Annotated[Optional[str], typer.Option("-cv-cols", help="Comma-separated CVs to compare (A's names); default all both sides have, except -op-col", rich_help_panel=panels.SELECT)] = None,
    exclude: Annotated[Optional[str], typer.Option("-exclude", help="Comma-separated substrings; CVs whose name contains one are left out (only without -cv-cols), e.g. z_,Box. A trailing '$' anchors one to the end of the name",rich_help_panel=panels.SELECT)] = None,

    # ── Representation ────────────────────────────────────────────────────
    angle_cols: Annotated[str, typer.Option("-angle-cols", help="Exact names of angle CVs (degrees) compared as cos(theta), on which an isotropic orientation is flat; '' for none", rich_help_panel=panels.REPR)] = ",".join(sorted(SPHERICAL_ANGLE_COLS)),
    sym_angle_cols: Annotated[Optional[str], typer.Option("-sym-angle-cols", help="Exact names of head-tail symmetric angle CVs compared as cos^2(theta)", rich_help_panel=panels.REPR)] = None,
    indicators: Annotated[Optional[str], typer.Option("-indicators", help="Extra 0/1 columns, comma-separated NAME=TEST, whose mean per OP bin is the population of a state there, with CI and q like any CV. TEST is COL>v, COL>=v, COL<v, COL<=v or lo<COL<hi (<= allowed), on the CV in its file units, abs(COL) for |COL|; join TESTs with & for a joint state; NAME=flip(TEST) is 1 at each frame where that state changes before the next frame (a rate per frame step). B's corrections of each COL (-mirror-b, -flip-b, -phase-shift-b) are applied first, so write it in A's convention. Added whatever -cv-cols/-exclude leave. E.g. 'PRO_anti=abs(PRO_dih_OH)>90,PRO_endo_bound=PRO_dih_chi1>0&PRO_CO_hb>=1,PRO_pucker_flips=flip(PRO_dih_chi1>0)'", rich_help_panel=panels.REPR)] = None,
    hist_bins: Annotated[int, typer.Option("-hist-bins", help="Value bins per continuous CV; integer-valued CVs get one bin per value", rich_help_panel=panels.REPR)] = 24,
    ranges: Annotated[Optional[str], typer.Option("-ranges", help="Optional TOML with fixed value ranges, [ranges] name = [min, max, n_bins] (the histograms format); other CVs get theirs from the data", rich_help_panel=panels.REPR)] = None,
    range_sample: Annotated[int, typer.Option("-range-sample", help="Paths per side read first to set the automatic value ranges", rich_help_panel=panels.REPR)] = 400,

    # ── Symmetry corrections (B only) ─────────────────────────────────────
    flip_op_b: Annotated[bool, typer.Option("-flip-op-b", help="Bin B on -OP, its lambdas mirrored too, so B's entry stage lands where A's escape is: B is then compared with A at the same lab z, not at the same point of its path. Not for runs that merely entered from opposite faces: OP_Lamb is each run's own infretis order parameter, which already lines their stages up; correct the lab-frame CVs with -swap-leaflets-b and -mirror-b/-flip-b instead",rich_help_panel=panels.SYMMETRY)] = False,
    swap_leaflets_b: Annotated[bool, typer.Option("-swap-leaflets-b", help="Rename B's _u_ columns _l_ and back, so the leaflets line up with A's: for a B that entered from the other face, whose entry leaflet is A's exit leaflet. No -flip-op-b with it: the OP is already aligned",rich_help_panel=panels.SYMMETRY)] = False,
    mirror_b: Annotated[Optional[str], typer.Option("-mirror-b", help="Substrings of CVs negated in B: theta -> -theta. For chirality-odd pseudoscalars (PRO_sign_vol, PRO_dih_, PRO_hCN_, PRO_nCos_) and, for a B that entered from the other face, CVs signed along lab z, raw cosines against +z included (_def, curvH, '_tilt$', PRO_mode_). A trailing '$' anchors a pattern to the end of the name",rich_help_panel=panels.SYMMETRY)] = None,
    flip_b: Annotated[Optional[str], typer.Option("-flip-b", help="Substrings of angle CVs (degrees, against +z) mapped to 180 - theta in B, negated if -angle-cols converted them to cos. A column that already holds a cosine (Mem_u_tilt, PRO_mode_*) needs -mirror-b instead: this would give 180 - cos",rich_help_panel=panels.SYMMETRY)] = None,
    phase_shift_b: Annotated[Optional[str], typer.Option("-phase-shift-b", help="Substrings of periodic phases advanced by 180 degrees in B (PRO_CP_phi2 between enantiomers)",rich_help_panel=panels.SYMMETRY)] = None,
    mirror_null: Annotated[Optional[str], typer.Option("-mirror-null", help="Substrings of CVs whose null hypothesis is B = -A (a pseudoscalar against the achiral normal, as chiral-compare's cos_chi_n): A + B is tested instead of A - B, and distributions against B mirrored", rich_help_panel=panels.SYMMETRY)] = None,

    # ── Model ─────────────────────────────────────────────────────────────
    n_boot: Annotated[int, typer.Option("-n-boot", help="Block-bootstrap replicates for every CI and p-value", rich_help_panel=panels.MODEL)] = 500,
    block: Annotated[str, typer.Option("-block", help="Paths per contiguous MC block in the bootstrap. 'auto': every folder runs the blocking analysis on every CV and takes chiral-compare's rule (the slowest CV decides, at least 15 blocks per stage); a number fixes it", rich_help_panel=panels.MODEL)] = "auto",
    seed: Annotated[int, typer.Option("-seed", help="Random seed of the bootstrap", rich_help_panel=panels.MODEL)] = 0,
    min_frames: Annotated[int, typer.Option("-min-frames", help="An OP bin is tested only with at least this many frames on both sides", rich_help_panel=panels.MODEL)] = 50,
    hist_min_frames: Annotated[int, typer.Option("-hist-min-frames", help="A window's distributions are tested only with at least this many frames on both sides", rich_help_panel=panels.MODEL)] = 200,
    correlations: Annotated[bool, typer.Option("-correlations/-no-correlations", help="Also the CV x CV correlation matrices (about cv^2 x windows numbers per stage and class)", rich_help_panel=panels.MODEL)] = True,
    unweighted: Annotated[bool, typer.Option("-unweighted/-no-unweighted", help="Repeat everything with every path equal within its set, in unweighted/", rich_help_panel=panels.MODEL)] = True,
    workers: Annotated[int, typer.Option("-workers", help="Worker processes reading the CV files, each with one BLAS thread, so up to the number of cores", rich_help_panel=panels.MODEL)] = 8,
    paths_per_job: Annotated[int, typer.Option("-paths-per-job", help="Paths a worker reduces per job", rich_help_panel=panels.MODEL)] = 200,

    # ── Output ────────────────────────────────────────────────────────────
    out_dir: Annotated[str, typer.Option("-out-dir", help="Directory for the tables and figures", rich_help_panel=panels.OUTPUT)] = "cv_compare",
    plot: Annotated[bool, typer.Option("-plot/-no-plot", help="Draw the figures at the end (cv-plot can draw them later from the tables)", rich_help_panel=panels.OUTPUT)] = True,
    overw: Annotated[bool, typer.Option("-O", help="Force overwriting of existing files", rich_help_panel=panels.OUTPUT)] = False,
):
    """Every per-path CV compared between two simulations (or reactive vs
    unreactive paths) along the order parameter, with block-bootstrap errors.

    Per CV and OP bin: mean, SD and the standardised difference d, with CI
    and BH-corrected q; per OP window: the distributions bin by bin and
    whether they differ by a shift, a spread or a tail; a ranking of the CVs
    by their largest effect; and the CV x CV correlations of each side.
    Stages are combined with the WHAM and stage weights of chiral-compare.
    Two simulations also get the reactive and the unreactive paths alone,
    every stage its own folder, and everything once more unweighted.  The
    figures are drawn at the end; `chiroflux cv-plot` redraws them from the
    tables.  See the module docstring for details.
    """
    if runs_a is None:
        raise typer.BadParameter("-runs-a is required.")
    if n_boot < 2:
        raise typer.BadParameter("-n-boot must be at least 2")
    if op_step <= 0 or window_step <= 0:
        raise typer.BadParameter("-op-step and -window-step must be positive")
    direction = direction.strip().lower()
    if direction not in DIRECTIONS:
        raise typer.BadParameter(f"-direction: one of {', '.join(DIRECTIONS)}, not {direction!r}")
    if str(block).strip().lower() == "auto":
        block_n = None
    else:
        try:
            block_n = int(block)
        except ValueError:
            raise typer.BadParameter(f"-block: 'auto' or a number of paths, not {block!r}")
        if block_n < 1:
            raise typer.BadParameter("-block must be at least 1")
    try:
        tests = [parse_indicator(s) for s in _split(indicators)]
    except ValueError as exc:
        raise typer.BadParameter(f"-indicators: {exc}")
    two = runs_b is not None
    labels = (label_a, label_b) if two else ("reactive", "unreactive")
    specs = {"a": _side_specs(runs_a, data_a, toml_a, cv_dir_a, "a")}
    if two:
        specs["b"] = _side_specs(runs_b, data_b, toml_b, cv_dir_b, "b")

    # The tables first; the figure names depend on the columns, which are
    # known only once the weighted paths are (below).
    stage_names = cia.stage_folder_names(pd.DataFrame(
        {"stage": range(len(specs["a"])), "run": [s.run_dir for s in specs["a"]]}))
    folders = folder_plan(range(len(specs["a"])), stage_names, two)
    tops = [out_dir] + ([os.path.join(out_dir, UNWEIGHTED_DIR)] if unweighted else [])
    for name in TOP_OUTPUTS:
        _check_overwrite(os.path.join(out_dir, name), overw)
    for top in tops:
        for f in folders:
            for name in FOLDER_OUTPUTS:
                _check_overwrite(os.path.join(top, f.sub, name), overw)

    sides, summaries = {}, {}
    pn = cia._parse_paths(path_numbers)
    lams, lm1s = [], []
    for key in specs:
        units, summary, model = cia.setup_stages(specs[key], nskip, wham_subdir, pcross_col,
                                                 trim=False, path_numbers=pn,
                                                 make_units=_cv_units)
        label = labels[0] if key == "a" else labels[1]
        print(f"  {label if two else label_a}:\n" + summary.to_string(index=False))
        if len(specs[key]) == 1:
            print("  [!] a single stage: its weighted ensemble is conditional on starting "
                  "in that run's state A")
        sign = -1.0 if key == "b" and flip_op_b else 1.0
        l, m = _stage_lambdas(specs[key], sign)
        lams += l
        lm1s += [m] if m is not None else []
        sides[key] = (units, model, sign)
        summaries[key] = summary

    # Which columns: from the header of every file a weighted path will be
    # read from, so neither one old file nor the unweighted burn-in paths
    # decide it.
    cols, missing = {}, []
    for key, (units, _, _) in sides.items():
        flag = f"-cv-dir-{key}"
        per_stage = [(os.path.basename(os.path.normpath(spec.run_dir)),
                      {u.uid: _cv_file(u) for u in units if u.stage == k})
                     for k, spec in enumerate(specs[key])]
        cols[key], table = side_header(per_stage, op_col, flag)
        missing.append(table)
    plan = plan_columns(cols["a"], cols.get("b"), _split(cv_cols), _split(exclude),
                        _split(angle_cols), _split(sym_angle_cols), swap_leaflets_b,
                        _split(mirror_b), _split(flip_b), _split(phase_shift_b),
                        _split(mirror_null), tests)
    print(f"  {plan.n} CVs compared" + (f"; tested as A + B: "
          + ", ".join(np.array(plan.names)[plan.mirror_null]) if plan.mirror_null.any() else ""))
    for spec in plan.indicator:
        if spec is not None:
            k, idx = spec
            fixes = [f"{plan.aux_a[i]}: {', '.join(plan.aux_sym[i])}" for i in idx
                     if plan.aux_sym[i]]
            print(f"  indicator {k.text}" + (f"  (B corrected first: {'; '.join(fixes)})"
                                            if two and fixes else ""))
    if plot:
        from .cv_plot import figure_names

        for top in tops:
            for f in folders:
                for name in figure_names(plan.names):
                    _check_overwrite(os.path.join(top, f.sub, name), overw)
    pd.concat(missing, ignore_index=True).to_csv(
        os.path.join(_mk(out_dir), "paths_missing_columns.csv"), index=False)

    rng_ = _parse_range(op_range)
    if rng_ is None:
        lo = min(lm1s + lams)
        hi = max(lams)
    else:
        lo, hi = rng_
    stage_edges = sorted(set(np.round(lams, 6)))
    cuts = [*stage_edges, MIDPLANE]
    grid = Grid(make_edges(lo, hi, op_step, cuts), make_edges(lo, hi, window_step, cuts))
    print(f"  OP axis {lo:g} to {hi:g}: {grid.n_op - 2} bins of the profiles, "
          f"{grid.n_win - 2} windows of the distributions (+ one open bin either side)")
    if direction != "all":
        print(f"  only {direction} frames: the OP "
              f"{'increasing' if direction == 'inbound' else 'decreasing'} along each path")

    overrides = load_ranges(ranges) if ranges else None
    samples = []
    for key, (units, _, _) in sides.items():
        src = plan.src_a if key == "a" else plan.src_b
        sym = None if key == "a" else plan.sym_b
        samples += sample_frames([u for u in units if u.weight > 0], _cv_file, plan.angle,
                                 sym, src, op_col, range_sample, workers=workers,
                                 indicator=plan.indicator,
                                 aux=plan.aux_a if key == "a" else plan.aux_b,
                                 aux_sym=None if key == "a" else plan.aux_sym)
    vbins = value_bins(samples, plan, hist_bins, overrides)
    del samples
    def _cols(src, aux):
        return [s if spec is None else "&".join(aux[i] for i in spec[1])
                for s, spec in zip(src, plan.indicator)]

    def _fixes(spec, ops):
        if spec is None:
            return ",".join(ops)
        return ";".join(f"{plan.aux_a[i]}:{','.join(plan.aux_sym[i])}" for i in spec[1]
                        if plan.aux_sym[i])

    pd.DataFrame({
        "cv": plan.names, "family": plan.families(),
        "column_A": _cols(plan.src_a, plan.aux_a),
        "column_B": _cols(plan.src_b, plan.aux_b) if two else _cols(plan.src_a, plan.aux_a),
        "angle": plan.angle,
        "corrections_B": ([_fixes(spec, ops) for spec, ops in zip(plan.indicator, plan.sym_b)]
                          if two else ""),
        "tested": np.where(plan.mirror_null, "A+B", "A-B"), "discrete": vbins.discrete,
        "value_min": [e[0] for e in vbins.edges], "value_max": [e[-1] for e in vbins.edges],
        "value_bins": [len(e) - 1 for e in vbins.edges],
        "indicator": [spec[0].text if spec is not None else "" for spec in plan.indicator],
    }).to_csv(os.path.join(_mk(out_dir), "columns.csv"), index=False)

    data = {}
    for key, (units, model, sign) in sides.items():
        cfg = SideConfig(op_col=op_col, src=plan.src_a if key == "a" else plan.src_b,
                         angle=plan.angle, sym=None if key == "a" else plan.sym_b,
                         op_sign=sign, grid=grid, vbins=vbins, correlations=correlations,
                         direction=direction, indicator=plan.indicator,
                         aux=plan.aux_a if key == "a" else plan.aux_b,
                         aux_sym=None if key == "a" else plan.aux_sym)
        label = labels[0] if (key == "a" and two) else labels[1] if key == "b" else label_a
        print(f"\n  Reducing {label} ...")
        side, table = reduce_side(label, [u for u in units if u.weight > 0 or u.weight_uniform > 0],
                                  model, cfg, _cv_file, workers, paths_per_job)
        tag = "A" if key == "a" else "B"
        table.to_csv(os.path.join(out_dir, f"paths_{tag}.csv"), index=False)
        summaries[key].to_csv(os.path.join(out_dir, f"stages_{tag}.csv"), index=False)
        data[key] = side
    path_class_table([data[k] for k in data], [labels[0] if two else label_a]
                     + ([labels[1]] if two else [])).to_csv(
        os.path.join(out_dir, "path_classes.csv"), index=False)

    compare(data["a"], data.get("b"), labels, out_dir, plan, vbins, grid, stage_names,
            n_boot=n_boot, block=block_n, seed=seed, min_frames=min_frames,
            hist_min_frames=hist_min_frames, correlations=correlations,
            unweighted=unweighted, stage_edges=stage_edges, direction=direction)
    if not plot:
        return

    # The tables are the result; a figure that fails to draw must not lose
    # them.  `chiroflux cv-plot` redraws from the tables alone.
    from .cv_plot import plot_all, result_folders
    print("\n  Figures:")
    for folder in result_folders(out_dir):
        try:
            plot_all(folder)
        except Exception as exc:
            print(f"  [!] plotting failed ({type(exc).__name__}: {exc}); the tables "
                  f"are written, rerun with `chiroflux cv-plot -in-dir {folder}`")


def _mk(path):
    os.makedirs(path, exist_ok=True)
    return path
