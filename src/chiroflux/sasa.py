#!/usr/bin/env python
"""
Solvent-accessible surface area (SASA) of the permeant along the membrane normal.

What this computes
------------------
For every phase point of every path in the (in)RETIS ensembles, the SASA of the
permeant is evaluated with the lipids as the OCCLUDING environment and water
left out.  Water is the solvent, so removing it is what makes the number mean
"how much of the molecule can still be touched by water".  A molecule sitting in
bulk water has its full free-molecule SASA; a molecule buried between the acyl
chains has almost none.  The change between those two limits, resolved along the
membrane normal, is the profile this script produces.

Three curves are accumulated per z bin, all weighted by the WHAM path weights:

    SASA_total     total SASA of the permeant                         [A^2]
    SASA_polar     contribution of N, O and the H bonded to them      [A^2]
    SASA_apolar    contribution of C and the H bonded to them         [A^2]

plus SASA_free, the SASA of the same conformer with the membrane deleted, which
turns the profile into a burial fraction  1 - SASA_total / SASA_free  that is
free of any conformational-size drift along the path.

Algorithm
---------
Shrake-Rupley: each atom is covered with N_SPHERE_POINTS quasi-uniform points on
a sphere of radius r_vdw + r_probe; a point counts as accessible when no other
atom's expanded sphere contains it.  Radii are the Bondi set that GROMACS itself
ships in vdwradii.dat, and the probe is 0.14 nm, so the numbers are directly
comparable to `gmx sasa`.  Verified against GROMACS 2024.4 on a test path:
49.3 / 42.3 / 45.5 A^2 here vs 49.1 / 42.3 / 44.7 A^2 from `gmx sasa` at
-ndots 4800, i.e. agreement to within the dot-discretisation noise.

Only occluders within (r_max_solute + r_max_occluder) of a solute atom enter the
calculation, found with a PBC-aware neighbour search, so the cost is set by the
size of the permeant and not by the size of the box (~10 ms per frame at
N_SPHERE_POINTS = 960).

The reaction coordinate
-----------------------
z is the signed distance of the permeant's centre of mass from the membrane
centre of mass, in ANGSTROM, which is exactly the infRETIS order parameter
OP_Lamb used by analysis_parallel.py.  It is recomputed here from coordinates
rather than read from ML/*.txt, because the ML files have had seam frames and
trailing rows removed and therefore do not align row-by-row with the raw
trajectory.  order.txt is read anyway and the two are cross-checked; the mean
absolute deviation is reported per group as a sanity check.

Both COMs are taken with a PBC-aware circular mean over z, and the difference is
wrapped into [-Lz/2, Lz/2), so paths that cross the periodic boundary are
handled correctly.

Binning and statistics
----------------------
- z is binned in 1 A bins (Z_RANGE / Z_BIN_WIDTH), the resolution asked for.
- Every frame of a path carries that path's weight, exactly as in
  analysis_parallel.py (factor = weight, applied to all rows of the path).
- The profile in a bin is the weighted mean  sum_p w_p S_p / sum_p w_p.
- Uncertainty comes from a bootstrap over PATHS -- the independent sampling unit
  -- not over frames, which are massively correlated within a path.  Unlike the
  chunk-level bootstrap in analysis_parallel.py this one is exact, because the
  per-path binned sums are small enough to keep on disk.
- Bins holding less than MIN_WEIGHT_FRACTION of the peak weight are masked.

Reused from analysis_parallel.py
--------------------------------
load_path_weights, build_bin_centers, update_histogram, column_stats_from_bins,
write_stats_csv, make_histogram_from_bins, make_2d_histogram,
make_2d_histogram_free_energy, resolve_data_path, HIST_DPI, HIST_COLOR.
update_histogram adds a constant factor per row, so the weighted SUM of a
per-frame quantity needs the sibling accumulate_weighted_sum() below, which uses
the same bin-index convention.

Input layout (from system_file_gen.py)
--------------------------------------
    <run>/load/<path_num>/traj.txt      ordered segments + direction per segment
    <run>/load/<path_num>/order.txt     order parameter per phase point
    <run>/load/<path_num>/accepted/*.xtc
    <run>/post/ML/<path_num>.txt        first two lines give reactivity/ensemble
    <run>/wham/path_weights.txt         WHAM path weights

Several runs (entry / escape / internal) can be listed in RUNS so that the
profile covers the whole membrane.  Each run carries its own `scale`: the path
weights of two runs are NOT on a common scale, and the factor that puts them
there is the crossing-probability correction documented in analysis_parallel.py
(section "Escape-run correction factor").  Leave it at 1.0 for a single run.

Output layout
-------------
    <SASA_OUTPUT_DIR>/
      intermediates/                    per-chunk .npz (per-path binned sums)
      sasa_profile_<tag>.png            SASA(z), total / polar / apolar + CI
      sasa_burial_<tag>.png             burial fraction and relative exposure
      sasa_hist_<tag>.png               1-D weighted SASA distribution
      sasa_vs_z_hist2d_<tag>.png        joint density
      sasa_vs_z_fes_<tag>.png           -ln p of the same
      sasa_profile_comparison.png       all groups on one axis
      sasa_profile_<tag>.csv            the numbers behind the profile plot
      stats_sasa_<tag>.csv              summary stats of the SASA distribution
      path_report.txt                   per-path diagnostics
      debug.log                         DEBUG: everything printed, plus errors
      debug/worker_<pid>.log            DEBUG: per-worker path-by-path trace

Debugging a silent death (DEBUG = True)
---------------------------------------
A batch job that stops without a traceback was almost certainly killed from
outside -- the scheduler taking the node back for exceeding memory or walltime
-- or crashed inside a C extension.  Neither produces Python output, and what
had been printed is lost anyway because stdout to a file is block-buffered.
DEBUG addresses all three: every print and every traceback is mirrored into
debug.log and flushed immediately, faulthandler dumps a stack on SIGSEGV and on
the SIGTERM/SIGUSR1 a scheduler sends before SIGKILL, and each worker keeps its
own log so the last "START path N" line names the path that killed it.  The
per-path RSS in those logs tells a leak (steady climb) from one pathological
path (one huge jump), and MAX_TASKS_PER_CHILD caps the former.
"""

import atexit
import csv
import faulthandler
import glob
import os
import platform
import signal
import sys
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import tomli
import typer

from . import panels

# (the original inserted this file's directory on sys.path so it could import
#  analysis_parallel from the same folder; as a package module it imports the
#  same helpers from chiroflux.cv_histograms instead)
from .cv_histograms import (
    HIST_COLOR,
    HIST_DPI,
    build_bin_centers,
    column_stats_from_bins,
    load_path_weights,
    make_2d_histogram,
    make_2d_histogram_free_energy,
    make_histogram_from_bins,
    resolve_data_path,
    update_histogram,
    write_stats_csv,
)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ── 1. CONFIGURATION ─────────────────────────────────────────────────────────

# Each run contributes its own paths.  `scale` multiplies every path weight of
# that run; it is the crossing-probability correction that puts two runs on a
# common scale (see analysis_parallel.py).  1.0 for a single run.
# `mirror_z` folds the run in with z -> -z, for a run that samples the opposite
# branch of a symmetric membrane.

def _parse_range(text, flag, n):
    """Parse 'a,b' or 'a,b,nbins' from a CLI option."""
    parts = text.split(",")
    if len(parts) != n:
        raise typer.BadParameter(
            f"{flag} needs {n} comma-separated values, got {text!r}"
        )
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise typer.BadParameter(f"{flag}: non-numeric value in {text!r}") from None
    if vals[1] <= vals[0]:
        raise typer.BadParameter(f"{flag}: max must exceed min, got {text!r}")
    if n == 3:
        if int(vals[2]) < 1:
            raise typer.BadParameter(f"{flag}: needs at least 1 bin, got {text!r}")
        return (vals[0], vals[1], int(vals[2]))
    return (vals[0], vals[1])


# ── Runs, loaded from the -runs file ─────────────────────────────────────────
# This was a literal list of dicts here, so comparing a different pair of
# simulations meant editing the module. It is now read from a TOML file given
# with -runs (see examples/sasa_runs.toml) and populated by _load_runs before
# the analysis starts; everything below still reads RUNS as a global.
RUNS = []

_RUN_REQUIRED = ("name", "load_dir", "weights", "ml_dir", "tpr")
_RUN_OPTIONAL = {"scale": 1.0, "mirror_z": False}


def _load_runs(path):
    """Read the [[run]] tables from a TOML file into the RUNS format.

    Raises rather than defaulting: which simulations are combined, and with
    what relative scaling, defines the whole result.
    """
    fpath = Path(path)
    if not fpath.is_file():
        raise typer.BadParameter(
            f"-runs: no such file {path!r}. This file lists the simulations to "
            "combine and their relative scaling; see examples/sasa_runs.toml."
        )
    with open(fpath, "rb") as fh:
        cfg = tomli.load(fh)

    runs = cfg.get("run", [])
    if not runs:
        raise typer.BadParameter(f"-runs: {path!r} defines no [[run]] tables.")

    out = []
    seen = set()
    for i, raw in enumerate(runs, 1):
        missing = [k for k in _RUN_REQUIRED if k not in raw]
        if missing:
            raise typer.BadParameter(
                f"-runs: [[run]] #{i} is missing {', '.join(missing)}."
            )
        unknown = set(raw) - set(_RUN_REQUIRED) - set(_RUN_OPTIONAL)
        if unknown:
            raise typer.BadParameter(
                f"-runs: [[run]] {raw['name']!r} has unknown key(s) "
                f"{', '.join(sorted(unknown))}."
            )
        if raw["name"] in seen:
            raise typer.BadParameter(
                f"-runs: duplicate run name {raw['name']!r}; names label the "
                "output files, so they must be distinct."
            )
        seen.add(raw["name"])

        run = {k: raw[k] for k in _RUN_REQUIRED}
        run["scale"] = float(raw.get("scale", _RUN_OPTIONAL["scale"]))
        run["mirror_z"] = bool(raw.get("mirror_z", _RUN_OPTIONAL["mirror_z"]))
        if run["scale"] <= 0:
            raise typer.BadParameter(
                f"-runs: {run['name']!r} has scale={run['scale']}; it multiplies "
                "a weighted histogram, so it must be positive."
            )
        for key in ("load_dir", "ml_dir", "tpr", "weights"):
            if not Path(run[key]).exists():
                raise typer.BadParameter(
                    f"-runs: {run['name']!r} {key}={run[key]!r} does not exist."
                )
        out.append(run)
    return out

OUTPUT_DIR       = "sasa_analysis_output"
INTERMEDIATE_DIR = os.path.join(OUTPUT_DIR, "intermediates")

N_WORKERS   = 32
FLUSH_EVERY = 500          # paths per intermediate chunk

SKIP_PARSING = False       # True: reuse existing intermediates, do not touch xtc

TESTING    = False
test_value = 20            # paths per run when TESTING

# Recycle a worker process after this many paths (None = never).  A worker that
# leaks -- the MDAnalysis reader holds on to per-trajectory state -- grows until
# the scheduler kills it, which on an HPC node looks like a silent stall.  Set
# to e.g. 50 if the RSS logged by DEBUG climbs steadily.
MAX_TASKS_PER_CHILD = None

# ── Debugging ────────────────────────────────────────────────────────────────
# A batch job that dies without a traceback is almost always killed from
# outside (OOM / walltime) or crashed inside a C extension, and in both cases
# whatever Python had buffered on stdout is lost.  DEBUG mirrors every print
# and every uncaught error into a file, flushed line by line, and installs
# faulthandler so a segfault or a scheduler signal still leaves a stack trace.
DEBUG            = True
DEBUG_LOG        = "debug.log"      # inside OUTPUT_DIR
DEBUG_WORKER_DIR = "debug"          # inside OUTPUT_DIR, one log per worker PID
DEBUG_TRUNCATE   = True             # False: append across runs

# Paths between plain-text progress lines.  These replace the tqdm bar whenever
# the output is not a terminal: a redraw-in-place bar written into a redirected
# batch log is useless at best, and tqdm's own rate-adaptive redraw logic makes
# it look frozen once the paths slow down, which is not a failure but reads like
# one.  A dated line every PROGRESS_EVERY paths says the same thing and survives
# being written to a file.
PROGRESS_EVERY = 25

# Seconds without a single completed path before the run is declared stalled.
# multiprocessing.Pool does NOT notice a worker that dies abruptly -- the OOM
# killer taking one out leaves imap_unordered waiting for a result that can
# never arrive, and the job then sits there until the scheduler's walltime kills
# it, which is indistinguishable from a silent crash.  This turns that hang into
# an error with a report.  Generous: a slow path is minutes, not hours.
STALL_TIMEOUT = 900.0

# ── Selections ───────────────────────────────────────────────────────────────
SOLUTE_SEL    = "resname ORP"
MEMBRANE_SEL  = "resname DOPC POPC"
WATER_SEL     = "resname TIP3 SOD CLA"
PHOSPHATE_SEL = "resname DOPC POPC and name P"

# Water is the solvent; it must NOT occlude, or the SASA stops meaning
# "reachable by water".  Set True only to get a packing-density-like measure.
OCCLUDE_WITH_WATER = False

# ── SASA parameters ──────────────────────────────────────────────────────────
PROBE_RADIUS     = 1.4      # A, GROMACS default 0.14 nm
N_SPHERE_POINTS  = 960      # dots per atom; 960 -> ~1 % noise on a small solute
DEFAULT_RADIUS   = 1.5      # A, for elements missing from ELEMENT_RADII

# Bondi radii, identical to the values GROMACS ships in vdwradii.dat.
ELEMENT_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52,
    "F": 1.47, "P": 1.80, "S": 1.80, "CL": 1.75,
    "NA": 2.27, "K": 2.75, "MG": 1.73, "CA": 2.31,
}

# Residue-name overrides for atoms whose force-field type does not start with
# their element symbol (ions, virtual sites).
RESNAME_ELEMENT = {
    "SOD": "NA", "CLA": "CL", "POT": "K", "MG": "MG", "CAL": "CA",
}
ZERO_RADIUS_NAMES = {"MW", "LP", "M"}

# ── Binning ──────────────────────────────────────────────────────────────────
# Z_RANGE only has to CONTAIN the sampled window; anything outside it is dropped
# and the plots crop themselves to what was actually sampled.  A single infRETIS
# run covers one stretch of the normal (the entry run runs from about -25 to
# -13 A), so the full membrane needs several runs listed in RUNS.
Z_RANGE      = (-40.0, 40.0)     # A, relative to the membrane centre
Z_BIN_WIDTH  = 1.0               # A, the requested resolution
SASA_RANGE   = (0.0, 320.0, 80)  # (min, max, n_bins) for the SASA axis

Z_LABEL    = "z from membrane centre [$\\mathrm{\\AA}$]"
SASA_LABEL = "SASA [$\\mathrm{\\AA}^2$]"

# Fold every frame in a second time at -z.  Legitimate for a symmetric bilayer
# and doubles the statistics, but it hides any genuine leaflet asymmetry.
FOLD_SYMMETRIC = False

# ── Statistics ───────────────────────────────────────────────────────────────
N_BOOTSTRAP         = 1000
ALPHA               = 0.05
MIN_WEIGHT_FRACTION = 0.005      # mask bins below this fraction of the peak

# ── Groups (from the ML/*.txt header, same convention as analysis_parallel) ──
ALL_GROUPS = [
    ("non-reactive", "minus"),
    ("non-reactive", "plus"),
    ("reactive",     "plus"),
]
COMBINED_PLUS = [("non-reactive", "plus"), ("reactive", "plus")]

PROFILE_COLORS = {
    "total":  "#1a5c8a",
    "polar":  "#2a7fbf",
    "apolar": "#e05c2a",
    "free":   "#777777",
}


# ── 2. DEBUG LOGGING ─────────────────────────────────────────────────────────

_LOG_FH      = None      # parent: debug.log; worker: its own debug/worker_<pid>.log
_ORIG_STDERR = sys.stderr
_ORIG_STDOUT = sys.stdout


class _Tee:
    """
    Write to the console and to the log file at once, flushing both every time.

    The flush is the whole point: a job killed by the scheduler loses anything
    still sitting in a block-buffered pipe, which is exactly why the crash looks
    silent when the output is redirected to a file by the batch system.
    """

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh     = fh

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        try:
            self._fh.write(data)
            self._fh.flush()
        except (ValueError, OSError):    # log closed during interpreter shutdown
            pass
        return len(data)

    def flush(self):
        self._stream.flush()
        try:
            self._fh.flush()
        except (ValueError, OSError):
            pass

    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()


def dbg(msg):
    """
    Timestamped line into the log only -- not onto the console.

    Used for the high-frequency bookkeeping (per-path timings, memory) that is
    worth having in the file but would drown the terminal.  A no-op when DEBUG
    is off, so calls can be left in the hot path.
    """
    if _LOG_FH is None:
        return
    try:
        _LOG_FH.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        _LOG_FH.flush()
    except (ValueError, OSError):
        pass


def rss_mb():
    """Current resident set size in MB, or nan where /proc is not available."""
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def _install_faulthandler(fh):
    """
    Dump a stack trace for the failures that never reach `except`.

    enable()  -> SIGSEGV / SIGBUS / SIGFPE, i.e. a crash inside numpy or the
                 compiled xtc reader.
    register() -> the signals a scheduler uses to take a job down.  SLURM sends
                 SIGTERM (and SIGUSR1 with --signal) before SIGKILL, and SIGKILL
                 itself cannot be caught, so this window is the only chance to
                 record where the process was.  chain=True keeps the default
                 behaviour afterwards.
    """
    faulthandler.enable(file=fh, all_threads=True)
    for signame in ("SIGTERM", "SIGUSR1", "SIGUSR2", "SIGXCPU"):
        sig = getattr(signal, signame, None)
        if sig is None:
            continue
        try:
            faulthandler.register(sig, file=fh, all_threads=True, chain=True)
        except (OSError, RuntimeError, ValueError):
            pass


def _log_environment():
    """Banner: everything needed to tell two runs apart after the fact."""
    dbg("=" * 78)
    dbg(f"SASA_analysis start  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    dbg(f"host        : {platform.node()}")
    dbg(f"python      : {sys.version.split()[0]}  ({sys.executable})")
    dbg(f"numpy       : {np.__version__}")
    try:
        import MDAnalysis
        dbg(f"MDAnalysis  : {MDAnalysis.__version__}")
    except Exception as e:
        dbg(f"MDAnalysis  : IMPORT FAILED: {e}")
    dbg(f"cwd         : {os.getcwd()}")
    dbg(f"pid         : {os.getpid()}")
    for var in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_NNODES",
                "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "PBS_JOBID"):
        if var in os.environ:
            dbg(f"{var:12s}: {os.environ[var]}")
    dbg(f"workers     : {N_WORKERS}  (max tasks/child = {MAX_TASKS_PER_CHILD})")
    dbg(f"dots/atom   : {N_SPHERE_POINTS}   probe {PROBE_RADIUS} A")
    dbg(f"occluders   : {'membrane + water' if OCCLUDE_WITH_WATER else 'membrane'}")
    dbg(f"flush every : {FLUSH_EVERY} paths")
    dbg(f"start RSS   : {rss_mb():.0f} MB")
    dbg("=" * 78)


def setup_debug_log():
    """
    Point stdout, stderr, uncaught exceptions and fatal signals at the log file.

    Called once from __main__ before anything else runs, so an import-time or
    configuration failure is captured too.
    """
    global _LOG_FH

    if not DEBUG:
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, DEBUG_LOG)
    _LOG_FH  = open(log_path, "w" if DEBUG_TRUNCATE else "a", buffering=1)

    sys.stdout = _Tee(_ORIG_STDOUT, _LOG_FH)
    sys.stderr = _Tee(_ORIG_STDERR, _LOG_FH)

    _install_faulthandler(_LOG_FH)
    _log_environment()

    def _excepthook(exc_type, exc_value, exc_tb):
        dbg("UNCAUGHT EXCEPTION in the parent process:")
        try:
            traceback.print_exception(exc_type, exc_value, exc_tb, file=_LOG_FH)
            _LOG_FH.flush()
        except (ValueError, OSError):
            pass
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    # A log that ends without this line means the process was killed from
    # outside rather than finishing or raising.
    atexit.register(lambda: dbg(f"interpreter exit, peak-ish RSS {rss_mb():.0f} MB"))

    print(f"DEBUG=True - full output and errors -> {os.path.relpath(log_path)}")
    return log_path


def _worker_init():
    """
    Pool initializer: give each worker its own log and its own faulthandler.

    Separate files rather than the shared one because the interesting case is a
    worker that dies mid-path -- its last line names the path that killed it,
    which interleaved output from 20 processes would not.
    """
    global _LOG_FH

    if not DEBUG:
        return

    worker_dir = os.path.join(OUTPUT_DIR, DEBUG_WORKER_DIR)
    os.makedirs(worker_dir, exist_ok=True)
    _LOG_FH = open(os.path.join(worker_dir, f"worker_{os.getpid()}.log"),
                   "a", buffering=1)

    # sys.stdout/stderr are the parent's Tee, inherited across the fork; leave
    # them alone (worker prints are rare) and only redirect the crash handlers.
    _install_faulthandler(_LOG_FH)
    dbg(f"worker up, pid {os.getpid()}, RSS {rss_mb():.0f} MB")


# ── 3. BIN GRIDS ─────────────────────────────────────────────────────────────

def build_sasa_bin_info():
    """
    Bin grids for both axes, built with analysis_parallel.build_bin_centers so
    the (centers, Minx, dx) convention is identical to the rest of the analysis.
    """
    n_z = int(round((Z_RANGE[1] - Z_RANGE[0]) / Z_BIN_WIDTH))
    ranges = {
        "z":    (Z_RANGE[0], Z_RANGE[1], n_z),
        "SASA": SASA_RANGE,
    }
    return build_bin_centers(ranges)


def accumulate_weighted_sum(values, x, weights, histogram, dx, Minx):
    """
    Sibling of analysis_parallel.update_histogram for a PER-ROW weight.

    update_histogram() adds one constant `factor` to the bin of every row, which
    is what a plain weighted histogram needs.  Here each row carries its own
    quantity (the frame's SASA), so the accumulated value is
    sum_frames  weight * value  per bin.  The bin-index convention -- floor
    division by dx from the lower edge Minx, rows outside the grid dropped --
    is exactly the one used there.
    """
    ix   = ((x - Minx) / dx).astype(int)
    n_x  = histogram.shape[0]
    mask = (ix >= 0) & (ix < n_x) & np.isfinite(values) & np.isfinite(x)
    histogram += np.bincount(ix[mask], weights=weights[mask] * values[mask],
                             minlength=n_x)
    return histogram


# ── 4. GEOMETRY / SASA ───────────────────────────────────────────────────────

def sphere_points(n_points):
    """
    Quasi-uniform points on the unit sphere (Fibonacci / golden-spiral lattice).
    Cheaper and more even than the recursive icosahedron GROMACS uses, and the
    two agree to within their own discretisation noise (see module docstring).
    """
    i     = np.arange(n_points) + 0.5
    phi   = np.arccos(1.0 - 2.0 * i / n_points)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack((np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(phi)))


def guess_elements(atomgroup):
    """
    Element symbol per atom, from the force-field type, then the atom name, with
    residue-name overrides for ions.  CHARMM types (CG2O2, OG311, HGA2, ...) all
    start with their element, which is what makes this reliable here.
    """
    names    = atomgroup.names
    types    = getattr(atomgroup, "types", names)
    resnames = atomgroup.resnames

    out = []
    for nm, ty, rn in zip(names, types, resnames):
        rn = rn.strip().upper()
        if rn in RESNAME_ELEMENT:
            out.append(RESNAME_ELEMENT[rn])
            continue
        if nm.strip().upper() in ZERO_RADIUS_NAMES:
            out.append("VS")
            continue
        elem = None
        for cand in (ty, nm):
            c = str(cand).strip().upper()
            if c and c[0].isalpha():
                elem = c[0]
                break
        out.append(elem or "C")
    return np.array(out, dtype=object)


def radii_for(atomgroup, probe=PROBE_RADIUS):
    """vdW radius + probe per atom; virtual sites get zero so they never occlude."""
    elements = guess_elements(atomgroup)
    radii    = np.empty(len(elements), dtype=np.float64)
    unknown  = set()
    for i, e in enumerate(elements):
        if e == "VS":
            radii[i] = 0.0
            continue
        if e in ELEMENT_RADII:
            radii[i] = ELEMENT_RADII[e] + probe
        else:
            unknown.add(e)
            radii[i] = DEFAULT_RADIUS + probe
    if unknown:
        print(f"  NOTE: no radius for element(s) {sorted(unknown)}, "
              f"using {DEFAULT_RADIUS} A")
    return radii, elements


def shrake_rupley(solute_pos, solute_r, occ_pos, occ_r, points):
    """
    Per-atom SASA of `solute_pos` in the field of solute + occluder spheres.

    All coordinates must already be in one continuous local frame (no PBC
    jumps); make_local_frame() below is what puts them there.

    Returns an array of per-atom areas in A^2.
    """
    n_pts = len(points)
    n_sol = len(solute_pos)
    areas = np.zeros(n_sol, dtype=np.float64)

    if len(occ_pos):
        all_pos = np.vstack((solute_pos, occ_pos))
        all_r   = np.concatenate((solute_r, occ_r))
    else:
        all_pos = solute_pos
        all_r   = solute_r

    for i in range(n_sol):
        R_i = solute_r[i]
        if R_i <= 0.0:
            continue

        # Only spheres that can reach the dot shell of atom i matter.
        d      = np.linalg.norm(all_pos - solute_pos[i], axis=1)
        occl   = (d < R_i + all_r) & (d > 1e-6)
        area_i = 4.0 * np.pi * R_i ** 2

        if not occl.any():
            areas[i] = area_i
            continue

        dots  = solute_pos[i] + points * R_i
        near  = all_pos[occl]
        nearr = all_r[occl]
        # (n_pts, n_near) distances; buried dots fall inside any neighbour.
        d2    = np.sum((dots[:, None, :] - near[None, :, :]) ** 2, axis=2)
        free  = ~np.any(d2 < nearr[None, :] ** 2, axis=1)
        areas[i] = area_i * free.sum() / n_pts

    return areas


def circular_com_z(z, masses, box_z):
    """
    Mass-weighted centre of mass along z, computed on the circle so that a group
    straddling the periodic boundary still gives the right answer (Bai & Breen).
    """
    theta = 2.0 * np.pi * z / box_z
    xi    = np.average(np.cos(theta), weights=masses)
    zeta  = np.average(np.sin(theta), weights=masses)
    return box_z * (np.arctan2(-zeta, -xi) + np.pi) / (2.0 * np.pi)


def wrap_delta(dz, box_z):
    """Signed separation wrapped into [-Lz/2, Lz/2)."""
    return dz - box_z * np.round(dz / box_z)


# ── 5. PATH / TRAJECTORY BOOKKEEPING ─────────────────────────────────────────

def read_traj_txt(traj_txt):
    """
    Ordered unique segment file names and their time direction.

    Mirrors system_file_gen.extract_sorted_traj_names(): column 2 is the file,
    column 4 the direction, .trr entries refer to a .xtc of the same stem, and a
    .g96 entry is a single phase point stored outside the trajectory files.
    That system_file_gen module cannot be imported (it parses argv and runs the
    whole analysis at import time), so the logic is repeated here.

    Returns (filenames, directions, g96_index).
    """
    filenames, directions, seen = [], [], set()
    g96_index = None

    with open(traj_txt, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(("#", "@")):
                continue
            cols = line.split()
            if len(cols) < 4:
                continue
            step, filename, direction = cols[0], cols[1], cols[3]
            if filename in seen:
                continue
            seen.add(filename)
            if filename.endswith(".trr"):
                filenames.append(filename[:-4] + ".xtc")
                directions.append(direction)
            elif filename.endswith(".g96"):
                g96_index = step
            else:
                filenames.append(filename)
                directions.append(direction)

    return filenames, directions, g96_index


def classify_path(ml_dir, path_num):
    """
    (reactivity, ensemble) for a path, from the first two lines of its
    ML/<path_num>.txt -- the same header convention analysis_parallel.classify_file
    parses.  Only the header is read, so this costs nothing.  Returns None when
    the file is missing or unlabelled.
    """
    fp = os.path.join(ml_dir, f"{path_num}.txt")
    try:
        with open(fp, "r") as fh:
            line1 = fh.readline().strip().lstrip("#").strip()
            line2 = fh.readline().strip().lstrip("#").strip()
    except OSError:
        return None

    if "non-reactive" in line1:
        reactivity = "non-reactive"
    elif "reactive" in line1:
        reactivity = "reactive"
    else:
        return None

    if "plus" in line2:
        ensemble = "plus"
    elif "minus" in line2:
        ensemble = "minus"
    else:
        return None

    return (reactivity, ensemble)


# ── 6. PER-PATH WORKER ───────────────────────────────────────────────────────

_CACHE = {}     # per-process, keyed by tpr path


def _get_universe(tpr):
    """
    One Universe (and one set of selections, radii and dot lattice) per worker
    process.  Parsing the tpr takes ~0.4 s, so it must not happen per path;
    swapping in a new trajectory with load_new() costs a few ms.
    """
    import MDAnalysis as mda

    if tpr in _CACHE:
        return _CACHE[tpr]

    # mda.Universe, not TPRParser: the parser class only produces a Topology
    # object, which has no selections, no masses and nothing to load frames
    # into.  The Universe wraps it and is what load_new() attaches the xtc to.
    #
    # The tpr is wanted for the topology alone -- names, types, masses, bonds --
    # since every frame arrives later through load_new().  Older MDAnalysis
    # builds have no TPR *coordinate* reader and warn "No coordinate reader
    # found ... Skipping this file"; that is the expected outcome here, not a
    # problem, so the warning is silenced rather than left to alarm the log.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No coordinate reader found")
        u = mda.Universe(tpr)

    solute = u.select_atoms(SOLUTE_SEL)
    memb   = u.select_atoms(MEMBRANE_SEL)
    water  = u.select_atoms(WATER_SEL)
    phos   = u.select_atoms(PHOSPHATE_SEL)

    if solute.n_atoms == 0:
        raise ValueError(f"SOLUTE_SEL {SOLUTE_SEL!r} selected no atoms in {tpr}")
    if memb.n_atoms == 0:
        raise ValueError(f"MEMBRANE_SEL {MEMBRANE_SEL!r} selected no atoms in {tpr}")

    occluders = memb + water if OCCLUDE_WITH_WATER else memb

    sol_r, sol_elements = radii_for(solute)
    occ_r, _            = radii_for(occluders)

    polar_mask = _polar_mask(solute, sol_elements)

    cache = {
        "u":          u,
        "solute":     solute,
        "memb":       memb,
        "phos":       phos,
        "occluders":  occluders,
        "sol_r":      sol_r,
        "occ_r":      occ_r,
        "polar_mask": polar_mask,
        "points":     sphere_points(N_SPHERE_POINTS),
        "cutoff":     float(sol_r.max() + occ_r.max()),
    }
    _CACHE[tpr] = cache
    return cache


def _polar_mask(solute, elements):
    """
    Polar atoms of the permeant: N and O, plus every H bonded to one of them.
    Bonds come from the tpr; if they are absent the H are assigned to their
    nearest heavy atom in the topology's reference coordinates.
    """
    polar = np.array([e in ("N", "O") for e in elements], dtype=bool)

    heavy_of_h = {}
    route = "bonds"
    try:
        for bond in solute.bonds:
            a, b = bond.atoms
            if a in solute and b in solute:
                ia = list(solute.indices).index(a.index)
                ib = list(solute.indices).index(b.index)
                if elements[ia] == "H" and elements[ib] != "H":
                    heavy_of_h[ia] = ib
                elif elements[ib] == "H" and elements[ia] != "H":
                    heavy_of_h[ib] = ia
    except Exception:
        heavy_of_h = {}

    if not heavy_of_h:
        route = "distance"
        # Only reachable when the tpr carried no bonds.  The reference
        # coordinates it needs are absent whenever MDAnalysis could not read
        # coordinates from the tpr either, so say so plainly instead of letting
        # a NoDataError surface once per path as an opaque "topology:" failure.
        try:
            pos = solute.positions
        except Exception as e:
            raise ValueError(
                "cannot work out which H are polar: the tpr provided no bonds "
                "and no coordinates to fall back on.  Give _get_universe a "
                "structure file with coordinates (gro/pdb) alongside the tpr, "
                f"or add bonds to the topology.  ({type(e).__name__}: {e})"
            ) from e

        for i, e in enumerate(elements):
            if e != "H":
                continue
            d = np.linalg.norm(pos - pos[i], axis=1)
            d[i] = np.inf
            d[[j for j, ee in enumerate(elements) if ee == "H"]] = np.inf
            heavy_of_h[i] = int(np.argmin(d))

    for ih, ihe in heavy_of_h.items():
        polar[ih] = polar[ihe]

    dbg(f"polar mask via {route}: {int(polar.sum())}/{len(polar)} atoms polar")
    return polar


def process_path(job):
    """
    DEBUG wrapper around _process_path_impl().

    Two things it buys.  Anything the impl does not catch -- a MemoryError, a
    numpy error deep in shrake_rupley, a malformed frame -- would otherwise
    propagate out of the worker and be re-raised by imap_unordered in the
    parent, killing the whole run and losing every path already done; here it
    comes back as a normal error record instead.  And the START/DONE pair means
    a worker that is killed outright still leaves its last START in the log,
    naming the path it died on.
    """
    path_num = job[0]
    group    = job[6]

    t0 = time.time()
    dbg(f"START path {path_num}  RSS {rss_mb():.0f} MB")
    try:
        res = _process_path_impl(job)
    except Exception as e:
        tb = traceback.format_exc()
        dbg(f"CRASH path {path_num}\n{tb}")
        return {"path_num": path_num, "group": group,
                "error": f"{type(e).__name__}: {e}", "traceback": tb}

    dt = time.time() - t0
    if res.get("error"):
        dbg(f"FAIL  path {path_num}  {res['error']}  ({dt:.1f} s)")
    else:
        dbg(f"DONE  path {path_num}  {res['n_frames']} frames  "
            f"{dt:.1f} s  RSS {rss_mb():.0f} MB")
    return res


def _process_path_impl(job):
    """
    One path: walk its segments in order, compute z and the three SASA numbers
    per frame, and return the per-path binned sums.

    Everything the parent needs is returned pre-binned, so the memory a path
    occupies is n_z floats and not n_frames.

    Returns a dict, or None when the path cannot be used.
    """
    (path_num, load_dir, ml_dir, tpr, weight, mirror_z, group,
     z_centers, z_min, z_dz, s_centers, s_min, s_ds) = job

    try:
        cache = _get_universe(tpr)
    except Exception as e:
        return {"path_num": path_num, "group": group, "error": f"topology: {e}"}

    u          = cache["u"]
    solute     = cache["solute"]
    memb       = cache["memb"]
    phos       = cache["phos"]
    occluders  = cache["occluders"]
    sol_r      = cache["sol_r"]
    occ_r_all  = cache["occ_r"]
    polar_mask = cache["polar_mask"]
    points     = cache["points"]
    cutoff     = cache["cutoff"]

    path_dir = os.path.join(load_dir, str(path_num))
    traj_txt = os.path.join(path_dir, "traj.txt")
    acc_dir  = os.path.join(path_dir, "accepted")
    if not os.path.isfile(traj_txt) or not os.path.isdir(acc_dir):
        return {"path_num": path_num, "group": group, "error": "no traj.txt / accepted/"}

    try:
        seg_names, directions, g96_index = read_traj_txt(traj_txt)
    except OSError as e:
        return {"path_num": path_num, "group": group, "error": f"traj.txt: {e}"}
    if not seg_names:
        return {"path_num": path_num, "group": group, "error": "no segments listed"}

    from MDAnalysis.lib.distances import capped_distance, minimize_vectors

    z_vals, s_tot, s_pol, s_apo, s_free = [], [], [], [], []
    zp_up, zp_lo = [], []
    prev_sig = None
    n_dupes  = 0

    for seg, direction in zip(seg_names, directions):
        seg_path = os.path.join(acc_dir, seg)
        if not os.path.isfile(seg_path):
            return {"path_num": path_num, "group": group, "error": f"missing segment {seg}"}
        # Logged before the read: a segfault in the compiled xtc reader leaves
        # this line as the last trace of which file was open.
        dbg(f"  path {path_num}: segment {seg} ({direction})")
        try:
            u.load_new(seg_path, refresh_offsets=True)
        except Exception as e:
            return {"path_num": path_num, "group": group, "error": f"{seg}: {e}"}

        frames = range(len(u.trajectory))
        if str(direction).strip() == "-1":
            frames = reversed(frames)

        for fi in frames:
            ts    = u.trajectory[fi]
            box   = ts.dimensions
            box_z = float(box[2])

            # Permeant made whole, then the membrane brought into its image.
            ref     = solute.positions[0]
            sol_pos = ref + minimize_vectors(solute.positions - ref, box)
            com     = np.average(sol_pos, axis=0, weights=solute.masses)

            # Seam frames are shared by consecutive segments; drop the repeat.
            sig = (round(float(com[0]), 4), round(float(com[1]), 4),
                   round(float(com[2]), 4), round(box_z, 4))
            if sig == prev_sig:
                n_dupes += 1
                continue
            prev_sig = sig

            z_memb = circular_com_z(memb.positions[:, 2], memb.masses, box_z)
            z_rel  = wrap_delta(float(com[2]) - z_memb, box_z)

            # Occluders that can possibly shadow a dot of the permeant.
            pairs = capped_distance(sol_pos, occluders.positions, cutoff,
                                    box=box, return_distances=False)
            if len(pairs):
                idx     = np.unique(pairs[:, 1])
                occ_pos = com + minimize_vectors(occluders.positions[idx] - com,
                                                 box)
                occ_r   = occ_r_all[idx]
            else:
                occ_pos = np.empty((0, 3))
                occ_r   = np.empty(0)

            per_atom = shrake_rupley(sol_pos, sol_r, occ_pos, occ_r, points)
            free_at  = shrake_rupley(sol_pos, sol_r,
                                     np.empty((0, 3)), np.empty(0), points)

            z_vals.append(z_rel)
            s_tot.append(per_atom.sum())
            s_pol.append(per_atom[polar_mask].sum())
            s_apo.append(per_atom[~polar_mask].sum())
            s_free.append(free_at.sum())

            # Leaflet phosphate planes, for the landmarks on the profile plot.
            if phos.n_atoms:
                dzp = wrap_delta(phos.positions[:, 2] - z_memb, box_z)
                if np.any(dzp > 0):
                    zp_up.append(float(dzp[dzp > 0].mean()))
                if np.any(dzp < 0):
                    zp_lo.append(float(dzp[dzp < 0].mean()))

    if not z_vals:
        return {"path_num": path_num, "group": group, "error": "no usable frames"}

    z_vals = np.asarray(z_vals)
    s_tot  = np.asarray(s_tot)
    s_pol  = np.asarray(s_pol)
    s_apo  = np.asarray(s_apo)
    s_free = np.asarray(s_free)

    # Cross-check against the order parameter infRETIS itself recorded.
    op_dev = np.nan
    order_txt = os.path.join(path_dir, "order.txt")
    if os.path.isfile(order_txt):
        try:
            order = np.loadtxt(order_txt, comments=("#", "@"))
            op    = order[:, 1] if order.ndim == 2 else order
            if g96_index is not None and 0 <= int(g96_index) < len(op):
                op = np.delete(op, int(g96_index))
            if len(op) == len(z_vals):
                op_dev = float(np.mean(np.abs(op - z_vals)))
        except (OSError, ValueError, IndexError):
            pass

    n_z = len(z_centers)
    n_s = len(s_centers)

    z_all = [z_vals]
    if mirror_z:
        z_all = [-z_vals]
    if FOLD_SYMMETRIC:
        z_all = z_all + [-z_all[0]]

    w_bin    = np.zeros(n_z)
    tot_bin  = np.zeros(n_z)
    pol_bin  = np.zeros(n_z)
    apo_bin  = np.zeros(n_z)
    fre_bin  = np.zeros(n_z)
    tot2_bin = np.zeros(n_z)
    hist2d   = np.zeros((n_s, n_z))

    for zz in z_all:
        wrow = np.full(zz.size, weight)
        update_histogram(zz, weight, w_bin, z_dz, z_min)
        accumulate_weighted_sum(s_tot,       zz, wrow, tot_bin,  z_dz, z_min)
        accumulate_weighted_sum(s_pol,       zz, wrow, pol_bin,  z_dz, z_min)
        accumulate_weighted_sum(s_apo,       zz, wrow, apo_bin,  z_dz, z_min)
        accumulate_weighted_sum(s_free,      zz, wrow, fre_bin,  z_dz, z_min)
        accumulate_weighted_sum(s_tot ** 2,  zz, wrow, tot2_bin, z_dz, z_min)
        update_histogram(np.column_stack((s_tot, zz)), weight, hist2d,
                         s_ds, s_min, Miny=z_min, dy=z_dz)

    return {
        "path_num": path_num,
        "group":    group,
        "error":    None,
        "n_frames": int(z_vals.size),
        "n_dupes":  int(n_dupes),
        "op_dev":   op_dev,
        "z_min":    float(z_vals.min()),
        "z_max":    float(z_vals.max()),
        "w":        w_bin,
        "tot":      tot_bin,
        "pol":      pol_bin,
        "apo":      apo_bin,
        "free":     fre_bin,
        "tot2":     tot2_bin,
        "hist2d":   hist2d,
        "zp_up":    float(np.mean(zp_up)) if zp_up else np.nan,
        "zp_lo":    float(np.mean(zp_lo)) if zp_lo else np.nan,
        "weight":   weight,
    }


# ── 7. INTERMEDIATES ─────────────────────────────────────────────────────────

def group_tag(group_key):
    react, ens = group_key
    return f"{react.replace('-', '')}_{ens}"


def intermediate_path(group_key, chunk_idx):
    return os.path.join(INTERMEDIATE_DIR,
                        f"{chunk_idx:04d}__{group_tag(group_key)}__sasa.npz")


def flush_chunk(buffer, chunk_idx):
    """
    Write one chunk per group: the per-path binned sums stacked row-wise (kept
    per path so the bootstrap can resample paths exactly) and the chunk-summed
    2-D histogram (too big to keep per path).
    """
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    for group_key, records in buffer.items():
        if not records:
            continue
        np.savez_compressed(
            intermediate_path(group_key, chunk_idx),
            w        = np.vstack([r["w"]    for r in records]),
            tot      = np.vstack([r["tot"]  for r in records]),
            pol      = np.vstack([r["pol"]  for r in records]),
            apo      = np.vstack([r["apo"]  for r in records]),
            free     = np.vstack([r["free"] for r in records]),
            tot2     = np.vstack([r["tot2"] for r in records]),
            hist2d   = np.sum([r["hist2d"] for r in records], axis=0),
            path_num = np.array([r["path_num"] for r in records]),
            zp_up    = np.array([r["zp_up"]  for r in records]),
            zp_lo    = np.array([r["zp_lo"]  for r in records]),
            weight   = np.array([r["weight"] for r in records]),
        )
    for group_key in buffer:
        buffer[group_key] = []


def load_group(group_key):
    """Concatenate every chunk of one group.  Returns None when it has none."""
    pattern = os.path.join(
        INTERMEDIATE_DIR, f"[0-9][0-9][0-9][0-9]__{group_tag(group_key)}__sasa.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        return None

    acc = defaultdict(list)
    hist2d = None
    for fp in files:
        d = np.load(fp)
        for key in ("w", "tot", "pol", "apo", "free", "tot2",
                    "path_num", "zp_up", "zp_lo", "weight"):
            acc[key].append(d[key])
        hist2d = d["hist2d"].copy() if hist2d is None else hist2d + d["hist2d"]

    out = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    out["hist2d"] = hist2d
    return out


def merge_groups(groups):
    """Pool several groups into one dataset (used for the combined plus ensemble)."""
    loaded = [g for g in (load_group(k) for k in groups) if g is not None]
    if not loaded:
        return None
    out = {}
    for key in ("w", "tot", "pol", "apo", "free", "tot2",
                "path_num", "zp_up", "zp_lo", "weight"):
        out[key] = np.concatenate([g[key] for g in loaded], axis=0)
    out["hist2d"] = np.sum([g["hist2d"] for g in loaded], axis=0)
    return out


# ── 8. PROFILE + BOOTSTRAP ───────────────────────────────────────────────────

def weighted_profile(data, n_bootstrap=N_BOOTSTRAP, alpha=ALPHA, rng=None):
    """
    Weighted mean SASA per z bin plus a path-level bootstrap confidence band.

    The sampling unit is the PATH.  Frames within a path are correlated by
    construction (they are consecutive points of one trajectory), so resampling
    frames would understate the uncertainty by orders of magnitude; resampling
    paths does not.  Each resample draws n_paths paths with replacement, which
    is done here as a multinomial over path indices so the whole bootstrap is
    two matrix products instead of a Python loop.

    Bins holding less than MIN_WEIGHT_FRACTION of the peak weight are masked.
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    W = data["w"]                       # (n_paths, n_z)
    n_paths = W.shape[0]

    w_tot = W.sum(axis=0)
    peak  = w_tot.max() if w_tot.size else 0.0
    mask  = (peak <= 0) | (w_tot < MIN_WEIGHT_FRACTION * peak)

    out = {"weight": w_tot, "mask": mask, "n_paths": n_paths}

    with np.errstate(invalid="ignore", divide="ignore"):
        for key in ("tot", "pol", "apo", "free"):
            mean = np.where(~mask, data[key].sum(axis=0) / w_tot, np.nan)
            out[f"mean_{key}"] = mean

        # Frame-to-frame spread within a bin (not an error bar on the mean).
        var = (data["tot2"].sum(axis=0) / w_tot) - out["mean_tot"] ** 2
        out["sd_tot"] = np.where(~mask, np.sqrt(np.clip(var, 0.0, None)), np.nan)

        exposure = np.where(~mask & (out["mean_free"] > 0),
                            out["mean_tot"] / out["mean_free"], np.nan)
        out["exposure"] = exposure
        out["burial"]   = 1.0 - exposure

    if n_paths < 2 or n_bootstrap < 1:
        for key in ("tot", "pol", "apo", "exposure"):
            out[f"ci_lo_{key}"] = np.full(W.shape[1], np.nan)
            out[f"ci_hi_{key}"] = np.full(W.shape[1], np.nan)
        return out

    lo_q, hi_q = alpha / 2 * 100, (1.0 - alpha / 2) * 100
    block = 200
    boots = {k: [] for k in ("tot", "pol", "apo", "exposure")}

    for start in range(0, n_bootstrap, block):
        n_b   = min(block, n_bootstrap - start)
        # Multinomial counts == drawing n_paths paths with replacement.
        counts = rng.multinomial(n_paths, np.full(n_paths, 1.0 / n_paths),
                                 size=n_b).astype(np.float64)
        w_b = counts @ W
        with np.errstate(invalid="ignore", divide="ignore"):
            m = {k: np.where(w_b > 0, (counts @ data[k]) / w_b, np.nan)
                 for k in ("tot", "pol", "apo", "free")}
            boots["tot"].append(m["tot"])
            boots["pol"].append(m["pol"])
            boots["apo"].append(m["apo"])
            boots["exposure"].append(np.where(m["free"] > 0,
                                              m["tot"] / m["free"], np.nan))

    # Unsampled bins are all-NaN columns by construction; that is what `mask`
    # records, so the percentile warning about them carries no information.
    with np.errstate(invalid="ignore"), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for key, chunks in boots.items():
            arr = np.vstack(chunks)
            lo  = np.nanpercentile(arr, lo_q, axis=0)
            hi  = np.nanpercentile(arr, hi_q, axis=0)
            out[f"ci_lo_{key}"] = np.where(~mask, lo, np.nan)
            out[f"ci_hi_{key}"] = np.where(~mask, hi, np.nan)

    return out


def phosphate_planes(data):
    """Weight-averaged z of the upper and lower leaflet phosphates."""
    w  = data["weight"]
    up = data["zp_up"]
    lo = data["zp_lo"]
    def _avg(v):
        ok = np.isfinite(v) & np.isfinite(w)
        return float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan
    return _avg(up), _avg(lo)


# ── 9. PLOTTING ──────────────────────────────────────────────────────────────

def _membrane_landmarks(ax, planes):
    """Dashed lines at the membrane centre and the two phosphate planes."""
    z_up, z_lo = planes
    ax.axvline(0.0, color="#999999", lw=1.0, ls=":", zorder=0)
    for z in (z_up, z_lo):
        if np.isfinite(z):
            ax.axvline(z, color="#b48a3a", lw=1.0, ls="--", zorder=0)
    if np.isfinite(z_up) and np.isfinite(z_lo):
        ax.axvspan(z_lo, z_up, color="#f2e6c9", alpha=0.30, zorder=-1,
                   label="acyl region")


def _set_z_limits(ax, z_centers, masks, planes, pad=2.0):
    """
    Frame the sampled z window together with the bilayer, so a run that only
    covers part of the membrane is still shown in context instead of being
    squeezed into a corner of the full Z_RANGE.
    """
    sampled = np.zeros(len(z_centers), dtype=bool)
    for m in masks:
        sampled |= ~m
    if not sampled.any():
        return

    lo = float(z_centers[sampled].min())
    hi = float(z_centers[sampled].max())
    for z in planes:
        if np.isfinite(z):
            lo, hi = min(lo, z), max(hi, z)
    ax.set_xlim(lo - pad, hi + pad)


def make_profile_plot(z_centers, prof, planes, group_label, out_path,
                      alpha=ALPHA):
    """SASA(z): total with its bootstrap band, plus the polar/apolar split."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if np.all(prof["mask"]):
        print(f"    SKIPPED (no sampled bins): {os.path.basename(out_path)}")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    _membrane_landmarks(ax, planes)

    ax.plot(z_centers, prof["mean_free"], color=PROFILE_COLORS["free"],
            lw=1.2, ls="--", label="free molecule (no membrane)")

    ax.fill_between(z_centers, prof["ci_lo_tot"], prof["ci_hi_tot"],
                    color=PROFILE_COLORS["total"], alpha=0.25,
                    label=f"total, {int((1 - alpha) * 100)} % CI")
    ax.plot(z_centers, prof["mean_tot"], color=PROFILE_COLORS["total"],
            lw=2.2, label="total SASA")
    ax.plot(z_centers, prof["mean_pol"], color=PROFILE_COLORS["polar"],
            lw=1.6, label="polar (N, O, their H)")
    ax.plot(z_centers, prof["mean_apo"], color=PROFILE_COLORS["apolar"],
            lw=1.6, label="apolar (C, their H)")

    ax.set_xlabel(Z_LABEL, fontsize=15)
    ax.set_ylabel(SASA_LABEL, fontsize=15)
    ax.set_ylim(bottom=0.0)
    _set_z_limits(ax, z_centers, [prof["mask"]], planes)
    ax.tick_params(labelsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.85, edgecolor="#cccccc")

    fig.suptitle(f"{group_label}   ({prof['n_paths']} paths, "
                 f"{Z_BIN_WIDTH:g} $\\mathrm{{\\AA}}$ bins)",
                 fontsize=11, color="#444444")
    fig.tight_layout()
    fig.savefig(out_path, dpi=HIST_DPI)
    plt.close(fig)


def make_burial_plot(z_centers, prof, planes, group_label, out_path,
                     alpha=ALPHA):
    """Fraction of the free-molecule surface still reachable by water."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if np.all(prof["mask"]):
        print(f"    SKIPPED (no sampled bins): {os.path.basename(out_path)}")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    _membrane_landmarks(ax, planes)

    ax.fill_between(z_centers, prof["ci_lo_exposure"] * 100.0,
                    prof["ci_hi_exposure"] * 100.0,
                    color=PROFILE_COLORS["total"], alpha=0.25,
                    label=f"{int((1 - alpha) * 100)} % CI")
    ax.plot(z_centers, prof["exposure"] * 100.0,
            color=PROFILE_COLORS["total"], lw=2.2, label="relative exposure")
    ax.axhline(100.0, color="grey", lw=1.0, ls="--",
               label="fully solvent exposed")

    ax.set_xlabel(Z_LABEL, fontsize=15)
    ax.set_ylabel("SASA / SASA$_{\\mathrm{free}}$ [%]", fontsize=15)
    ax.set_ylim(0.0, 110.0)
    _set_z_limits(ax, z_centers, [prof["mask"]], planes)
    ax.tick_params(labelsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.85, edgecolor="#cccccc")

    fig.suptitle(f"{group_label}   (burial of the permeant)",
                 fontsize=11, color="#444444")
    fig.tight_layout()
    fig.savefig(out_path, dpi=HIST_DPI)
    plt.close(fig)


def make_comparison_plot(z_centers, profiles, planes, out_path):
    """Every group's total-SASA profile on one axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not profiles:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    _membrane_landmarks(ax, planes)

    for (label, prof, color) in profiles:
        ax.fill_between(z_centers, prof["ci_lo_tot"], prof["ci_hi_tot"],
                        color=color, alpha=0.18)
        ax.plot(z_centers, prof["mean_tot"], color=color, lw=2.0,
                label=f"{label} ({prof['n_paths']} paths)")

    ax.set_xlabel(Z_LABEL, fontsize=15)
    ax.set_ylabel(SASA_LABEL, fontsize=15)
    ax.set_ylim(bottom=0.0)
    _set_z_limits(ax, z_centers, [p["mask"] for _, p, _ in profiles], planes)
    ax.tick_params(labelsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.85, edgecolor="#cccccc")

    fig.suptitle("SASA of the permeant through the membrane",
                 fontsize=11, color="#444444")
    fig.tight_layout()
    fig.savefig(out_path, dpi=HIST_DPI)
    plt.close(fig)


def write_profile_csv(z_centers, prof, out_path):
    fields = ["z", "weight", "mean_sasa", "ci_lo", "ci_hi", "sd_frames",
              "mean_polar", "mean_apolar", "mean_free", "exposure", "burial"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for i, z in enumerate(z_centers):
            if prof["mask"][i]:
                continue
            writer.writerow([
                f"{z:.4g}",
                f"{prof['weight'][i]:.6g}",
                f"{prof['mean_tot'][i]:.6g}",
                f"{prof['ci_lo_tot'][i]:.6g}",
                f"{prof['ci_hi_tot'][i]:.6g}",
                f"{prof['sd_tot'][i]:.6g}",
                f"{prof['mean_pol'][i]:.6g}",
                f"{prof['mean_apo'][i]:.6g}",
                f"{prof['mean_free'][i]:.6g}",
                f"{prof['exposure'][i]:.6g}",
                f"{prof['burial'][i]:.6g}",
            ])
    print(f"  Profile -> {os.path.relpath(out_path)}")


# ── 10. MAIN ──────────────────────────────────────────────────────────────────

def build_job_list(bin_info):
    """
    (path_num, weight, group) for every weighted path of every run, together
    with everything a worker needs to run standalone.
    """
    z_centers, z_min, z_dz = bin_info["z"]
    s_centers, s_min, s_ds = bin_info["SASA"]

    jobs, report = [], []

    for run in RUNS:
        name     = run["name"]
        load_dir = resolve_data_path(run["load_dir"], must_be_dir=True)
        weights  = resolve_data_path(run["weights"],  must_be_dir=False)
        ml_dir   = resolve_data_path(run["ml_dir"],   must_be_dir=True)
        tpr      = resolve_data_path(run["tpr"],      must_be_dir=False)

        print(f"\n  Run {name!r}")
        for label, cfg, got in (("load_dir", run["load_dir"], load_dir),
                                ("weights",  run["weights"],  weights),
                                ("ml_dir",   run["ml_dir"],   ml_dir),
                                ("tpr",      run["tpr"],      tpr)):
            print(f"    {label:9s}: {cfg!r} -> {got or 'NOT FOUND'}")

        if not all((load_dir, weights, ml_dir, tpr)):
            print(f"    SKIPPED: could not locate every input for run {name!r}")
            continue

        ordered = load_path_weights(weights)
        if TESTING:
            ordered = ordered[:test_value]
            print(f"    TESTING: limited to the first {len(ordered)} paths")

        n_unclassified = n_missing = 0
        for path_num, weight in ordered:
            group = classify_path(ml_dir, path_num)
            if group is None or group not in ALL_GROUPS:
                n_unclassified += 1
                continue
            if not os.path.isdir(os.path.join(load_dir, str(path_num))):
                n_missing += 1
                continue
            jobs.append((path_num, load_dir, ml_dir, tpr,
                         weight * float(run.get("scale", 1.0)),
                         bool(run.get("mirror_z", False)), group,
                         z_centers, z_min, z_dz, s_centers, s_min, s_ds))

        report.append((name, len(ordered), n_unclassified, n_missing))
        print(f"    {len(ordered)} weighted paths, {n_unclassified} without an "
              f"ML label, {n_missing} without a load/ folder")

    return jobs, report


def _unfinished_paths():
    """
    Paths whose worker logged START but never DONE or FAIL -- i.e. still running,
    or gone down with the process that was running them.
    """
    out = []
    for fp in sorted(glob.glob(os.path.join(OUTPUT_DIR, DEBUG_WORKER_DIR,
                                            "worker_*.log"))):
        try:
            with open(fp, "r", errors="replace") as fh:
                started, finished = None, set()
                for line in fh:
                    if "START path " in line:
                        started = line.split("START path ")[1].split()[0]
                    for marker in ("DONE  path ", "FAIL  path ", "CRASH path "):
                        if marker in line:
                            finished.add(line.split(marker)[1].split()[0])
            if started is not None and started not in finished:
                out.append((os.path.basename(fp), started))
        except OSError:
            continue
    return out


def _stall_report(pool, n_seen, n_jobs):
    """
    Why nothing has come back for STALL_TIMEOUT seconds.

    Pool DOES respawn a child that dies -- what it never does is re-queue the
    task that child was holding, so the census usually shows every worker alive
    and healthy while the run waits forever for a result that no longer has a
    process behind it.  The worker logs are the reliable signal: the path a log
    STARTed and never finished is the one that went down.
    """
    lines = [f"no path completed in {STALL_TIMEOUT:.0f} s "
             f"({n_seen}/{n_jobs} done)."]

    stuck = _unfinished_paths()
    if stuck:
        lines.append("  Paths started but never finished:")
        for logname, path_num in stuck:
            lines.append(f"    path {path_num}  ({logname})")
    else:
        lines.append("  No worker log shows an unfinished path "
                     "(DEBUG off, or the stall is elsewhere).")

    lines.append("  Worker census:")
    try:
        for p in pool._pool:               # private, diagnostics only
            alive = p.is_alive()
            lines.append(f"    pid {p.pid}: {'alive' if alive else 'DEAD'}"
                         f"{'' if alive else f' (exitcode {p.exitcode})'}")
    except Exception as e:                 # never let the report itself fail
        lines.append(f"    (could not inspect the pool: {e})")

    lines.append("  A worker killed mid-path takes that path with it; Pool "
                 "respawns the process but never re-queues the work, so the "
                 "run would otherwise hang here until walltime.  The usual "
                 "cause is the OOM killer -- check the RSS trend in the worker "
                 "logs and lower N_WORKERS or set MAX_TASKS_PER_CHILD.")

    report = "\n".join(lines)
    dbg("STALLED\n" + report)
    return report


def _fmt_hms(seconds):
    """h:mm:ss, or --:--:-- when the estimate is not meaningful yet."""
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    seconds = int(seconds)
    return f"{seconds // 3600}:{(seconds // 60) % 60:02d}:{seconds % 60:02d}"


def _progress_out(line):
    """
    Emit a progress line to stderr, flushed.

    stderr on purpose, and not print()'s stdout: it is where the tqdm bar used
    to go, so a batch script that captures the two streams separately keeps
    finding progress in the same file; and stdout redirected to a file is
    block-buffered, which would hold these lines back for minutes -- the very
    problem the debug log exists to avoid.  With DEBUG on, sys.stderr is the tee
    and the line reaches debug.log as well.
    """
    print(line, file=sys.stderr, flush=True)


def _progress_line(n_seen, n_jobs, t_start, n_err):
    """One self-contained progress line for a batch log."""
    elapsed = time.time() - t_start
    rate    = n_seen / elapsed if elapsed > 0 else float("nan")
    eta     = (n_jobs - n_seen) / rate if rate > 0 else float("nan")
    return (f"  {n_seen}/{n_jobs} paths ({100.0 * n_seen / n_jobs:.1f} %)  "
            f"{rate:.2f} path/s  elapsed {_fmt_hms(elapsed)}  "
            f"ETA {_fmt_hms(eta)}  RSS {rss_mb():.0f} MB  {n_err} failed")


def parse_all(bin_info):
    """Run every path through the pool, flushing per-path sums every FLUSH_EVERY."""
    from multiprocessing import Pool
    from multiprocessing import TimeoutError as mp_TimeoutError

    jobs, _ = build_job_list(bin_info)
    if not jobs:
        print("\nERROR: no paths to process.  Check RUNS at the top of the file.")
        raise SystemExit(1)

    print(f"\n{len(jobs)} paths to process with {N_WORKERS} workers "
          f"({N_SPHERE_POINTS} dots/atom, probe {PROBE_RADIUS} A, "
          f"occluders = {'membrane + water' if OCCLUDE_WITH_WATER else 'membrane'})")

    buffer   = {k: [] for k in ALL_GROUPS}
    n_done   = 0
    chunk    = 0
    errors   = []
    op_devs  = []
    n_frames = 0
    n_dupes  = 0

    dbg(f"{len(jobs)} jobs queued")
    t_start   = time.time()
    n_seen    = 0
    crashed   = None

    try:
        with Pool(processes=N_WORKERS, initializer=_worker_init,
                  maxtasksperchild=MAX_TASKS_PER_CHILD) as pool:
            it = pool.imap_unordered(process_path, jobs, chunksize=1)

            # A bar only where a bar makes sense: an interactive terminal.  It
            # writes to the real stderr rather than through the tee, so the log
            # does not fill up with carriage returns.  miniters=1 forces a
            # redraw on every path -- tqdm's default adapts the redraw interval
            # to the observed rate, and once that rate collapses the bar can sit
            # unchanged for a very long time and look like it has died.
            # Redirected output gets _progress_line() instead.
            interactive = HAS_TQDM and getattr(_ORIG_STDERR, "isatty",
                                               lambda: False)()
            bar = tqdm(total=len(jobs), desc="SASA", unit="path",
                       file=_ORIG_STDERR, miniters=1,
                       mininterval=0.5) if interactive else None
            if bar is None:
                _progress_out(f"  (no terminal: progress every "
                              f"{PROGRESS_EVERY} paths)")

            # Driven by hand rather than `for res in it` so each result can be
            # waited for with a timeout; see STALL_TIMEOUT.
            while True:
                try:
                    res = it.next(timeout=STALL_TIMEOUT)
                except StopIteration:
                    break
                except mp_TimeoutError:
                    raise RuntimeError(_stall_report(pool, n_seen, len(jobs)))

                n_seen += 1
                if bar is not None:
                    try:
                        bar.update(1)
                    except Exception as e:
                        # Reporting must never be able to end the run.
                        dbg(f"progress bar disabled after {type(e).__name__}: {e}")
                        bar = None
                if res is None:
                    continue
                if res.get("error"):
                    errors.append((res["path_num"], res["error"]))
                    if res.get("traceback"):
                        dbg(f"traceback for path {res['path_num']}:\n"
                            f"{res['traceback']}")
                    continue

                group = res["group"]
                if group not in buffer:
                    errors.append((res["path_num"], f"unknown group {group!r}"))
                    continue

                buffer[group].append(res)
                n_done   += 1
                n_frames += res["n_frames"]
                n_dupes  += res["n_dupes"]
                if np.isfinite(res["op_dev"]):
                    op_devs.append(res["op_dev"])

                if n_done % FLUSH_EVERY == 0:
                    flush_chunk(buffer, chunk)
                    dbg(f"flushed chunk {chunk} after {n_done} paths")
                    chunk += 1

                # Printed rather than dbg()'d when there is no bar, so a batch
                # job shows progress in its own output and not only in the log.
                if n_seen % PROGRESS_EVERY == 0 or n_seen == len(jobs):
                    line = _progress_line(n_seen, len(jobs), t_start, len(errors))
                    if bar is None:
                        _progress_out(line)
                    else:
                        dbg(line.strip())

            if bar is not None:
                bar.close()
    except BaseException as e:
        # Includes a worker dying abruptly and Ctrl-C.  Whatever is buffered is
        # written out before re-raising, so a restart with SKIP_PARSING=True can
        # still use everything that had been computed.
        crashed = e
        dbg(f"POOL ABORTED after {n_seen}/{len(jobs)} results: "
            f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        print(f"\n  Pool aborted after {n_seen}/{len(jobs)} paths: "
              f"{type(e).__name__}: {e}")

    if any(buffer.values()):
        flush_chunk(buffer, chunk)
        dbg(f"flushed final chunk {chunk}")

    print(f"\n  {n_done} paths, {n_frames:,} phase points "
          f"({n_dupes:,} seam duplicates dropped)")
    if op_devs:
        print(f"  Mean |z_computed - order.txt| = {np.mean(op_devs):.4f} A "
              f"over {len(op_devs)} paths "
              f"(large values mean z and OP_Lamb are not the same coordinate)")
    if errors:
        print(f"  {len(errors)} paths failed:")
        for pn, err in errors[:10]:
            print(f"    {pn}: {err}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")

    # Results that never came back at all: a worker was killed between taking a
    # job and returning it, which is what a silent stall looks like from here.
    n_lost = len(jobs) - n_seen
    if n_lost:
        print(f"  WARNING: {n_lost} of {len(jobs)} paths produced no result "
              f"(worker killed?).  See {os.path.join(OUTPUT_DIR, DEBUG_WORKER_DIR)}"
              f"/ for the last path each worker started.")
        dbg(f"{n_lost} jobs never returned a result")

    with open(os.path.join(OUTPUT_DIR, "path_report.txt"), "w") as fh:
        fh.write(f"paths processed : {n_done}\n")
        fh.write(f"phase points    : {n_frames}\n")
        fh.write(f"seam duplicates : {n_dupes}\n")
        if op_devs:
            fh.write(f"mean |z - OP|   : {np.mean(op_devs):.6f} A\n")
        fh.write(f"failed paths    : {len(errors)}\n")
        fh.write(f"lost paths      : {n_lost}\n")
        for pn, err in errors:
            fh.write(f"  {pn}: {err}\n")

    dbg(f"parse_all finished: {n_done} ok, {len(errors)} failed, {n_lost} lost, "
        f"{time.time() - t_start:.0f} s")

    if crashed is not None:
        raise crashed


def analyse(bin_info):
    """Turn the intermediates into profiles, histograms, CSVs and plots."""
    z_centers, z_min, z_dz = bin_info["z"]
    s_centers, s_min, s_ds = bin_info["SASA"]

    datasets = [(f"{r} | {e} ensemble", group_tag((r, e)), load_group((r, e)),
                 HIST_COLOR[(r, e)])
                for (r, e) in ALL_GROUPS]

    combined = merge_groups(COMBINED_PLUS)
    if combined is not None:
        datasets.append(("non-reactive + reactive | plus ensemble",
                         "combined_plus", combined, "#4a1a6b"))

    comparison = []
    planes_all = (np.nan, np.nan)
    rng = np.random.default_rng(seed=42)

    for label, tag, data, color in datasets:
        if data is None:
            print(f"  [{label}] no intermediates, skipping.")
            continue

        print(f"\n  [{label}] {data['w'].shape[0]} paths, "
              f"bootstrap n={N_BOOTSTRAP}")
        prof   = weighted_profile(data, rng=rng)
        planes = phosphate_planes(data)
        if tag == "combined_plus" or not np.isfinite(planes_all[0]):
            planes_all = planes

        make_profile_plot(z_centers, prof, planes, label,
                          os.path.join(OUTPUT_DIR, f"sasa_profile_{tag}.png"))
        make_burial_plot(z_centers, prof, planes, label,
                         os.path.join(OUTPUT_DIR, f"sasa_burial_{tag}.png"))
        write_profile_csv(z_centers, prof,
                          os.path.join(OUTPUT_DIR, f"sasa_profile_{tag}.csv"))

        # 1-D SASA distribution and the joint (z, SASA) density, drawn with the
        # same helpers the rest of the analysis uses.
        sasa_1d = np.nansum(data["hist2d"], axis=1)
        make_histogram_from_bins(
            s_centers, sasa_1d, SASA_LABEL, label,
            os.path.join(OUTPUT_DIR, f"sasa_hist_{tag}.png"), color)
        make_2d_histogram(
            z_centers, s_centers, data["hist2d"],
            Z_LABEL, SASA_LABEL, label,
            os.path.join(OUTPUT_DIR, f"sasa_vs_z_hist2d_{tag}.png"))
        make_2d_histogram_free_energy(
            z_centers, s_centers, data["hist2d"],
            Z_LABEL, SASA_LABEL, label,
            os.path.join(OUTPUT_DIR, f"sasa_vs_z_fes_{tag}.png"))

        write_stats_csv(
            {"SASA":       column_stats_from_bins(s_centers, sasa_1d),
             "z_sampling": column_stats_from_bins(z_centers, prof["weight"])},
            os.path.join(OUTPUT_DIR, f"stats_sasa_{tag}.csv"))

        comparison.append((label, prof, color))
        print(f"  [{label}] phosphate planes at z = "
              f"{planes[0]:.2f} / {planes[1]:.2f} A")

    make_comparison_plot(z_centers, comparison, planes_all,
                         os.path.join(OUTPUT_DIR, "sasa_profile_comparison.png"))
    print(f"\n  Plots -> {os.path.relpath(OUTPUT_DIR)}/")


def sasa(
    # ── Input data ────────────────────────────────────────────────────────
    runs: Annotated[str, typer.Option("-runs", help="REQUIRED. TOML file listing the simulations to combine as [[run]] tables (load_dir, weights, ml_dir, tpr, scale, mirror_z). See examples/sasa_runs.toml.", rich_help_panel=panels.INPUT)] = ...,

    # ── Dataset construction ──────────────────────────────────────────────
    z_range: Annotated[Optional[str], typer.Option("-z-range", help="z axis relative to the membrane centre as 'min,max' in Angstrom.", rich_help_panel=panels.DATASET)] = None,
    z_bin_width: Annotated[float, typer.Option("-z-bin-width", help="z bin width in Angstrom.", rich_help_panel=panels.DATASET)] = Z_BIN_WIDTH,
    sasa_range: Annotated[Optional[str], typer.Option("-sasa-range", help="SASA axis as 'min,max,nbins' in Angstrom^2.", rich_help_panel=panels.DATASET)] = None,
    fold_symmetric: Annotated[bool, typer.Option("-fold-symmetric/-no-fold-symmetric", help="Fold the profile about the membrane centre.", rich_help_panel=panels.DATASET)] = FOLD_SYMMETRIC,
    occlude_with_water: Annotated[bool, typer.Option("-occlude-with-water/-no-occlude-with-water", help="Count water and ions as occluding the solute surface.", rich_help_panel=panels.DATASET)] = OCCLUDE_WITH_WATER,

    # ── CV corrections (representation) ───────────────────────────────────
    probe_radius: Annotated[float, typer.Option("-probe-radius", help="Shrake-Rupley probe radius in Angstrom (GROMACS default 1.4).", rich_help_panel=panels.REPR)] = PROBE_RADIUS,
    n_sphere_points: Annotated[int, typer.Option("-n-sphere-points", help="Surface dots per atom; more is less noisy and slower.", rich_help_panel=panels.REPR)] = N_SPHERE_POINTS,
    solute_sel: Annotated[str, typer.Option("-solute-sel", help="MDAnalysis selection for the solute.", rich_help_panel=panels.REPR)] = SOLUTE_SEL,
    membrane_sel: Annotated[str, typer.Option("-membrane-sel", help="MDAnalysis selection for the membrane.", rich_help_panel=panels.REPR)] = MEMBRANE_SEL,
    water_sel: Annotated[str, typer.Option("-water-sel", help="MDAnalysis selection for water and ions.", rich_help_panel=panels.REPR)] = WATER_SEL,
    phosphate_sel: Annotated[str, typer.Option("-phosphate-sel", help="MDAnalysis selection for the phosphate landmark atoms.", rich_help_panel=panels.REPR)] = PHOSPHATE_SEL,

    # ── Model and training ────────────────────────────────────────────────
    workers: Annotated[int, typer.Option("-workers", help="Worker processes; each handles one path at a time.", rich_help_panel=panels.MODEL)] = N_WORKERS,
    n_bootstrap: Annotated[int, typer.Option("-n-bootstrap", help="Path-level bootstrap resamples for the profile confidence band.", rich_help_panel=panels.MODEL)] = N_BOOTSTRAP,
    skip_parsing: Annotated[bool, typer.Option("-skip-parsing", help="Reuse existing intermediates instead of re-reading the trajectories.", rich_help_panel=panels.MODEL)] = SKIP_PARSING,

    # ── Output ────────────────────────────────────────────────────────────
    out_dir: Annotated[str, typer.Option("-out-dir", help="Directory for profiles, plots and intermediates.", rich_help_panel=panels.OUTPUT)] = OUTPUT_DIR,
    debug: Annotated[bool, typer.Option("-debug/-no-debug", help="Write a debug log plus one log per worker.", rich_help_panel=panels.OUTPUT)] = DEBUG,
):
    """Weighted solvent-accessible surface area profile across the membrane.

    Computes the solute's SASA per frame with a Shrake-Rupley construction,
    occluded by the membrane (and optionally water), and bins it against the
    solute's z position relative to the membrane centre, using the same WHAM
    path weights as the other analyses. Several runs can be combined on one z
    axis via the -runs file, each with its own scale factor and z convention.

    Reads trajectories, so it needs the load/ directories and topologies - not
    just the CV .txt files.
    """
    global RUNS, OUTPUT_DIR, INTERMEDIATE_DIR, N_WORKERS, N_BOOTSTRAP
    global SKIP_PARSING, DEBUG, PROBE_RADIUS, N_SPHERE_POINTS
    global SOLUTE_SEL, MEMBRANE_SEL, WATER_SEL, PHOSPHATE_SEL
    global OCCLUDE_WITH_WATER, FOLD_SYMMETRIC, Z_RANGE, Z_BIN_WIDTH, SASA_RANGE

    RUNS = _load_runs(runs)
    print(f"Loaded {len(RUNS)} run(s) from {runs!r}: "
          f"{', '.join(r['name'] for r in RUNS)}")

    OUTPUT_DIR = out_dir
    INTERMEDIATE_DIR = os.path.join(OUTPUT_DIR, "intermediates")
    N_WORKERS = workers
    N_BOOTSTRAP = n_bootstrap
    SKIP_PARSING = skip_parsing
    DEBUG = debug
    PROBE_RADIUS = probe_radius
    N_SPHERE_POINTS = n_sphere_points
    SOLUTE_SEL, MEMBRANE_SEL = solute_sel, membrane_sel
    WATER_SEL, PHOSPHATE_SEL = water_sel, phosphate_sel
    OCCLUDE_WITH_WATER = occlude_with_water
    FOLD_SYMMETRIC = fold_symmetric
    Z_BIN_WIDTH = z_bin_width

    if z_range is not None:
        Z_RANGE = _parse_range(z_range, "-z-range", 2)
    if sasa_range is not None:
        SASA_RANGE = _parse_range(sasa_range, "-sasa-range", 3)

    os.makedirs(OUTPUT_DIR,       exist_ok=True)
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

    setup_debug_log()

    bin_info = build_sasa_bin_info()
    z_centers, z_min, z_dz = bin_info["z"]
    s_centers, s_min, s_ds = bin_info["SASA"]
    print(f"\nz    : {len(z_centers)} bins of {z_dz:g} A over "
          f"[{Z_RANGE[0]:g}, {Z_RANGE[1]:g}]")
    print(f"SASA : {len(s_centers)} bins of {s_ds:g} A^2 over "
          f"[{SASA_RANGE[0]:g}, {SASA_RANGE[1]:g}]")

    if SKIP_PARSING:
        existing = glob.glob(os.path.join(INTERMEDIATE_DIR, "*.npz"))
        if not existing:
            print("ERROR: SKIP_PARSING=True but no intermediates were found.")
            raise SystemExit(1)
        print(f"\nSKIP_PARSING=True - reusing {len(existing)} intermediate files.")
    else:
        for stale in glob.glob(os.path.join(INTERMEDIATE_DIR, "*__sasa.npz")):
            os.remove(stale)
        parse_all(bin_info)

    print("\nBuilding profiles from intermediates...")
    analyse(bin_info)
    print("\nDone!")
    dbg("clean exit")
