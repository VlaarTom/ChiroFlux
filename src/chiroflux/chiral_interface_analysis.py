"""
Trajectory-level L/D comparison built around the mechanism proposed in
Hu, Cochrane, Jones, Blackmond & Paegel, "Chiral lipid bilayers are
enantioselectively permeable", Nat. Chem. 2021 (doi:10.1038/s41557-021-00708-z).

What the experiments pin down
-----------------------------
* R-DOPC/R-POPC bilayers pass L-amino-acid propargylamides faster than D
  (Pro-yne 2.1 vs 0.36, Phe-yne 2 vs 0.6, Val-yne 1.7 vs 0.89 x1e-4 cm/s).
* Inverting ONLY the lipid glycerol stereocentre (S-DOPC/S-POPC/CH) inverts
  the preference (Pro-yne 0.39 L vs 0.7 D) -> the discriminating element is the
  sn-2 glycerol carbon, i.e. the interfacial region, not the achiral tails.
* Cholesterol shrinks the selectivity; an achiral oleic-acid-rich membrane
  removes it -> packing around the stereocentre modulates the effect.
* Dipeptides respond to both of their stereocentres (pV >> Pv, PV >> pv) ->
  the permeant's backbone (amine/amide) makes the stereo-sensitive contacts.

Working hypothesis tested here: while crossing the headgroup/glycerol/carbonyl
region the permeant forms diastereomeric contacts with the lipid stereocentre.
L and D differ in which lipid atoms they bind, in what geometry, and for how
long, and that produces a different interfacial free-energy barrier or trap.

Each testable consequence gets its own output:

  WHERE     ddG(depth) and every observable resolved by depth  -> ddG_depth.csv,
            depth_profiles.csv (mean, SD)
            ln(P_L/P_D) implied by the profiles, with each depth's share of
            the resistance       -> permeability_ratio.csv, resistance_shares.csv
  WHAT      H-bond partner classes per permeant donor, especially sn-2
            (on the stereocentre's own branch) vs sn-1 ester/carbonyl O
                                          -> hbond_partners.csv, sn2_preference.csv
            which H-bond state carries ddG     -> ddG_by_hbond_state.csv
  HOW       permeant pose in the body frame of the nearest lipid stereocentre
            (diastereomeric contact geometry) -> pose_maps.npz, pose_jsd.csv
  HOW LONG  continuous H-bond / contact lifetimes (interfacial trapping)
                                          -> lifetimes.csv, lifetime_survival.csv
  RESPONSE  glycerol backbone conformation of the contacted lipid vs bulk,
            and the contact-induced shift     -> glycerol_conformers.csv,
            and each dihedral's full distribution -> glycerol_dihedrals.csv, .npz
  WHICH WAY where the molecular axis and the chirality vector point in the
            contacted lipid's frame (phi, cos)  -> orient_jsd_depth.csv,
                                                   orient_maps_depth.npz
  HOW FAST  reorientation of the axis, chirality vector and position
            relative to the contacted lipid       -> reorientation.csv, .npz
  WHERE x HOW  free energy over depth x orientation
                                  -> orientation_free_energy.csv, .npz
  BARRIER   every profile pooled over the depths that carry the
            resistance, all and crossing paths    -> barrier.csv
  SHAPE     the full distribution of every per-frame observable, per depth
            (2D) and per depth window (1D, tested bin by bin)
                                          -> histograms.csv, histograms.npz

Symmetry: why every observable is fair game
-------------------------------------------
L-permeant + R-lipids and D-permeant + R-lipids are DIASTEREOMERIC systems.
The mirror image of D+R is L+S, not L+R, so no symmetry relates the two runs.
Mirror-invariant quantities (depth, distances, H-bond counts, the PMF itself)
can and must differ; that is what a permeability difference is. Error bars
here therefore come only from sampling: a block bootstrap over paths in MC
order, and a block permutation test for whole distributions. (An exact
negative control needs no new simulation: reflect the D+R trajectory
x -> -x and you have L+S.)

The exceptions are the permeant's pseudovector measured against the membrane
normal, which is not chiral (cos_chi_n), and against the contacted lipid's
e_z (chi_lz).  Without any chiral recognition cos_chi_n would be exactly
D = -L, nonzero whenever the permeant has a preferred orientation; e_z leans
along the normal, so that orientation passes into chi_lz with opposite signs
too (D ~ -L).  For both L + D is tested instead of L - D (MIRROR_NULL_OBS).

Pipeline
--------
  extract  Per path (in parallel), read its xtc files once with MDAnalysis's
           low-level reader, which writes no offset files next to them, and
           compute one row of features for every phase point traj.txt lists,
           in path order.  Paths are stored -paths-per-file to an .npz in
           RUN_DIR/chiral_cache, each tagged with the SHA-1 of its traj.txt.
  compare  Take every path's frames from the cache (no trajectory is read),
           check them against order.txt and trim them.  Weight
           each frame so that the entry/internal/escape stages combine into
           ONE equilibrium ensemble; a single TIS run only gives a density
           conditional on its own start.  Reduce to |OP|-resolved
           accumulators, then bootstrap L vs D.  The reweighting follows
           inftools Wham_Pcross and chiroflux permeation_process; see the
           "Path-ensemble reweighting" section.

Path assembly follows chiroflux's cv_generation exactly. traj.txt names one
(xtc, frame) per phase point, via chiroflux.pathdata.read_traj_plan. The xtc
files also carry frames outside the path, so whole segments are never
concatenated. Next, the order parameter from order.txt is checked row by row
against the plan, after the g96 row is dropped. Finally the first and/or last
phase point is deleted by the plus/minus ensemble criteria of write_ML_txt.
The path is then one continuous, time-ordered sequence, so lifetimes run
through segment boundaries.

Usage
-----
    for stage in infinit_entry infinit_internal infinit_escape; do
      chiroflux chiral-extract \\
          -tpr L_PRO_yne_neutral/$stage/gromacs_input/topol.tpr \\
          -permeant pro_yne -load-dir L_PRO_yne_neutral/$stage/load -workers 32
    done                                              (same for D)

    chiroflux chiral-compare \\
        -runs-l L_PRO_yne_neutral/infinit_entry,L_PRO_yne_neutral/infinit_internal,L_PRO_yne_neutral/infinit_escape \\
        -runs-d D_PRO_yne_neutral/infinit_entry,D_PRO_yne_neutral/infinit_internal,D_PRO_yne_neutral/infinit_escape \\
        -data-l post/infretis_data.txt -data-d post/infretis_data.txt \\
        -out-dir comparison/chiral_interface

Each stage's cache lands in RUN_DIR/chiral_cache, where compare looks for it,
so the stages can be extracted separately, from anywhere.  An interrupted or
repeated extract only does the paths not cached yet (or whose traj.txt
changed).  compare needs the cache and load/*/traj.txt, load/*/order.txt, but
not the trajectories, so it can run on a copy without them.

Each run directory must hold load/ and the wham/ folder that inftools wrote
(Pcross.txt, histo_probability.txt), the same folder permeation_process reads.
Stages go in chain order.  The escape stage must be the mirror image of the
entry stage, because the backward crossing probabilities are taken from it.
-data-l/-data-d name each stage's infretis_data.txt: one relative path looked
for inside every run directory, or one file per run, comma-separated.  The
toml of each stage (-toml-l/-toml-d, same rules; default RUN_DIR/infretis.toml,
else RUN_DIR/post/infretis.toml) gives the interfaces for the WHAM weights and
the lambdas for trimming.  Its lambda_A is checked against the wham/ header.
-no-trim keeps every phase point.

Plain MD / umbrella windows: pass -xtc (chiral-extract) and -xtc-l/-xtc-d
(chiral-compare) instead.  Each file is then its own unweighted, untrimmed
unit, and depth is the stereocentre's z.

Caveats
-------
* Assumes an orthorhombic box; that is checked on every frame.
* Lifetimes are cut only at the ends of a path, after trimming. Runs still in
  progress at a path end are counted as they are, so absolute lifetimes are
  lower bounds. Both enantiomers are biased the same way.
* Path weights are WFtot, as in inftools' wham/path_weights.txt: a plus
  path carries its plus-ensemble WHAM weight (pathdata), a [0-] path its
  [0-] weight, each set normalised to sum to 1, with the first -nskip rows
  of infretis_data.txt skipped as burn-in.  ddG(depth) and the depth profiles
  are equilibrium quantities only with all stages and these weights.  With a
  single stage they are conditional on that run's start.  The unweighted/
  folder repeats the analysis with the WHAM reweighting left out (every
  path equal within its set); it describes where the sampled paths spent
  their time, and so has no free energies.
* The bootstrap resamples paths within each stage, recomputes xi from each
  resample of the entry stage, and draws every stage's Pcross from a
  log-normal with the relative error inftools' block analysis wrote to
  wham/errPtot.txt, chaining the stage factors anew.  The WHAM path weights
  themselves are held fixed.
* Frames at OP <= lambda_A are scaled or cut frame by frame.  inftools and
  permeation_process do it bin by bin (on the bin centre), so the one bin
  that straddles lambda_A can differ slightly.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import typer

from . import panels
from .pathdata import (
    _check_overwrite,
    _compute_path_weights,
    _load_interfaces_from_toml,
    _load_path_table,
    read_traj_plan,
)

KB_KCAL = 0.0019872041  # kcal / (mol K)


# ══════════════════════════════════════════════════════════════════════════════
# Molecule definitions
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PermeantSpec:
    resname: str
    stereo: str        # the stereocentre
    cip: tuple         # its four substituents in CIP priority order, lowest last
    l_config: str      # CIP descriptor of the L enantiomer
    axis: tuple        # (tail, head) of the molecular axis, a true vector
    donors: tuple      # ((heavy, H), ...)
    acceptors: tuple   # heavy-atom acceptors

    @property
    def polar(self):
        out = []
        for a in [d for d, _ in self.donors] + list(self.acceptors):
            if a not in out:
                out.append(a)
        return tuple(out)


PERMEANTS = {
    # Neutral proline: ring N-H and COOH.
    "pro": PermeantSpec("ORP", "CA", ("N", "C", "CB", "HA"), "S", ("N", "C"),
                        (("N", "H01"), ("O01", "H02")), ("O", "O01", "N")),
    # Proline N-propargylamide (the species measured in the paper): ring N-H and
    # amide N-H; C4 is the alpha carbon.
    "pro_yne": PermeantSpec("ORP", "C4", ("N9", "C2", "C6", "H5"), "S", ("N9", "C2"),
                            (("N9", "H04"), ("N01", "H03")), ("O3", "N9")),
}

LIPID_RESNAMES = ("DOPC", "POPC")
# The sn-2 glycerol carbon C2 carries O21 > C1 (-O-P) > C3 (-O-C=O) > HS.
# Natural sn-glycero-3-phosphocholine is R.
LIPID_FRAME_ATOMS = ("C2", "C1", "C3", "O21", "O11", "O31")

ACCEPTOR_CLASSES = (
    ("PO4",       ("O13", "O14")),   # non-bridging phosphate O
    ("PO4_ester", ("O11", "O12")),   # bridging phosphate O
    ("CO_sn2",    ("O22",)),         # sn-2 carbonyl: the stereocentre's own ester
    ("CO_sn1",    ("O32",)),
    ("OE_sn2",    ("O21",)),         # sn-2 ester O, bonded to the stereocentre
    ("OE_sn1",    ("O31",)),
)
CLASS_NAMES = [c for c, _ in ACCEPTOR_CLASSES] + ["water"]
WATER_CLASS = len(ACCEPTOR_CLASSES)
WATER_SEL = "resname TIP3 and name OH2"

HB_CUTOFF = 3.5      # Angstrom, donor-acceptor (gmx hbond default)
HB_ANGLE = 30.0      # degrees, H-D-A (gmx hbond default)

FLOAT_COLS = ("z", "cos_axis_n", "cos_chi_n", "d_c2",
              "pose_x", "pose_y", "pose_z",
              "axis_lx", "axis_ly", "axis_lz",
              "chi_lx", "chi_ly", "chi_lz",
              "glyc_a", "glyc_b",
              "n_hb_lipid", "n_lipids_bound", "n_wat_polar")
N_GLYC_BINS = 36


# ══════════════════════════════════════════════════════════════════════════════
# Geometry (Angstrom internally, like every MDAnalysis trajectory reader)
# ══════════════════════════════════════════════════════════════════════════════

def _mic(v, L):
    """Orthorhombic minimum image; L is (3,), broadcast over leading axes."""
    return v - L * np.round(v / L)


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    out = np.zeros_like(v)
    np.divide(v, n, out=out, where=n > 1e-12)
    return out


def _dihedral(p0, p1, p2, p3, L):
    """IUPAC dihedral p0-p1-p2-p3 in degrees, vectorised over leading axes."""
    b0 = _mic(p0 - p1, L)
    b1 = _unit(_mic(p2 - p1, L))
    b2 = _mic(p3 - p2, L)
    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)
    return np.degrees(np.arctan2(y, x))


def signed_volume(center, subs, L):
    """
    a.(b x c) of the bond vectors to the three highest-priority substituents.
    With the lowest-priority group pointing away from the viewer, a->b->c
    clockwise (R) gives a NEGATIVE volume, counter-clockwise (S) a positive one.
    center (..., 3), subs (..., 3, 3) in CIP order.
    """
    b = _mic(subs - center[..., None, :], L)
    return np.einsum("...i,...i->...", b[..., 0, :],
                     np.cross(b[..., 1, :], b[..., 2, :]))


def cip_label(volume):
    return np.where(volume < 0, "R", "S")


def _midplane(z_slab, z_p, Lz):
    """
    Bilayer midplane z, correct wherever the bilayer sits relative to the box
    edge (trajectories are not made whole or centred).

    A circular mean of the phosphorus z alone does NOT work here.  The two P
    layers are ~Lz/2 apart in these boxes (3.7 nm in Lz 7-8 nm), so their unit
    vectors nearly cancel.  Once the P-P distance exceeds Lz/2 the mean points
    at the middle of the WATER slab instead.  The whole lipid slab is one
    connected band thinner than Lz, so its circular mean always points at the
    bilayer.  That gives a coarse centre; the P atoms, minimum-imaged about it,
    then give the precise one.
    """
    ang = 2.0 * np.pi * z_slab / Lz
    s, c = np.sin(ang).mean(), np.cos(ang).mean()
    if np.hypot(s, c) < 0.05:
        raise ValueError("lipid slab has no well-defined centre in z")
    coarse = Lz / (2.0 * np.pi) * np.arctan2(s, c)
    d = z_p - coarse
    d -= Lz * np.round(d / Lz)
    return (coarse + d.mean()) % Lz


# ══════════════════════════════════════════════════════════════════════════════
# Extraction: one path -> one row of features per phase point
# ══════════════════════════════════════════════════════════════════════════════

class _Topology:
    """Atom indices resolved once per worker from the tpr."""

    def __init__(self, tpr, spec: PermeantSpec):
        import MDAnalysis as mda
        warnings.filterwarnings("ignore", module="MDAnalysis")
        self.u = mda.Universe(tpr)
        self.n_atoms = self.u.atoms.n_atoms
        self.spec = spec
        perm = self.u.select_atoms(f"resname {spec.resname}")
        if perm.n_residues != 1:
            raise ValueError(f"expected one {spec.resname} residue, found {perm.n_residues}")

        def one(name):
            g = perm.select_atoms(f"name {name}")
            if g.n_atoms != 1:
                raise ValueError(f"{spec.resname} atom {name!r}: {g.n_atoms} matches")
            return g.indices[0]

        self.stereo = one(spec.stereo)
        self.cip = np.array([one(n) for n in spec.cip])
        self.axis = (one(spec.axis[0]), one(spec.axis[1]))
        polar = spec.polar
        self.polar = np.array([one(n) for n in polar])
        self.donor_rows = [polar.index(d) for d, _ in spec.donors]
        self.donor_h = np.array([one(h) for _, h in spec.donors])

        lip_sel = "resname " + " ".join(LIPID_RESNAMES)
        lip = self.u.select_atoms(lip_sel)
        self.lipid_resix = lip.residues.resindices
        per = {}
        for name in LIPID_FRAME_ATOMS:
            g = lip.select_atoms(f"name {name}")
            if not np.array_equal(g.resindices, self.lipid_resix):
                raise ValueError(f"lipid atom {name} is not exactly one per lipid")
            per[name] = g.indices
        self.C2, self.C1, self.C3 = per["C2"], per["C1"], per["C3"]
        self.O21, self.O11, self.O31 = per["O21"], per["O11"], per["O31"]
        self.P = lip.select_atoms("name P").indices
        self.lipid_heavy = lip.select_atoms("not name H*").indices

        tgt, cls = [], []
        for c, (_, names) in enumerate(ACCEPTOR_CLASSES):
            g = lip.select_atoms("name " + " ".join(names))
            tgt.append(g.indices)
            cls.append(np.full(g.n_atoms, c))
        wat = self.u.select_atoms(WATER_SEL)
        tgt.append(wat.indices)
        cls.append(np.full(wat.n_atoms, WATER_CLASS))
        self.targets = np.concatenate(tgt)
        self.target_class = np.concatenate(cls).astype(np.int32)
        self.target_resix = self.u.atoms[self.targets].resindices.astype(np.int32)


def _frame_features(T: _Topology, pos, box):
    """Features of one frame.  Returns (float row, int row, glyc dihedrals of
    all lipids (2, n_lipid))."""
    from MDAnalysis.lib.distances import capped_distance

    if not np.allclose(box[3:], 90.0):
        raise ValueError(f"non-orthorhombic box {box}")
    L = box[:3].astype(float)
    pos = pos.astype(float)
    f = dict.fromkeys(FLOAT_COLS, np.nan)

    # Depth, leaflet-folded with the local outward normal.  Folding by |z| is a
    # C2 rotation about an in-plane axis: proper, so it preserves handedness.
    mid = _midplane(pos[T.lipid_heavy, 2], pos[T.P, 2], L[2])
    X = pos[T.stereo]
    z = X[2] - mid
    z -= L[2] * np.round(z / L[2])
    s = 1.0 if z >= 0 else -1.0
    f["z"] = z / 10.0

    sub = _mic(pos[T.cip] - X, L)
    n_perm = _unit(np.cross(sub[0], sub[1]))      # pseudovector, as compute_body_frame
    ax = _unit(_mic(pos[T.axis[1]] - pos[T.axis[0]], L))
    f["cos_axis_n"] = s * ax[2]
    f["cos_chi_n"] = s * n_perm[2]

    # Nearest lipid stereocentre and its right-handed glycerol body frame:
    # ex along C3->C1, ey toward the sn-2 ester O21, ez = ex x ey.  Under a
    # reflection pose_x/pose_y survive and pose_z flips, so pose_z says which
    # face of the stereocentre the permeant sits on.
    r = _mic(X - pos[T.C2], L)
    d = np.linalg.norm(r, axis=1)
    j = int(np.argmin(d))
    f["d_c2"] = d[j] / 10.0
    ex = _unit(_mic(pos[T.C1[j]] - pos[T.C3[j]], L))
    t = _mic(pos[T.O21[j]] - pos[T.C2[j]], L)
    ey = _unit(t - np.dot(t, ex) * ex)
    R = np.stack([ex, ey, np.cross(ex, ey)])
    f["pose_x"], f["pose_y"], f["pose_z"] = R @ _unit(r[j])
    f["axis_lx"], f["axis_ly"], f["axis_lz"] = R @ ax
    f["chi_lx"], f["chi_ly"], f["chi_lz"] = R @ n_perm

    # Glycerol backbone of every lipid; the contacted one is picked out below.
    glyc = np.stack([
        _dihedral(pos[T.O11], pos[T.C1], pos[T.C2], pos[T.O21], L),
        _dihedral(pos[T.O21], pos[T.C2], pos[T.C3], pos[T.O31], L),
    ])
    f["glyc_a"], f["glyc_b"] = glyc[:, j]

    # Hydrogen bonds: one neighbour search for all permeant polar atoms.
    pairs, dist = capped_distance(pos[T.polar], pos[T.targets], max_cutoff=HB_CUTOFF,
                                  box=box, return_distances=True)
    ints = [T.lipid_resix[j]]
    bound = set()
    n_hb_lipid = 0
    for k, row in enumerate(T.donor_rows):
        m = pairs[:, 0] == row
        cls, res = -1, -1
        if m.any():
            cand = pairs[m, 1]
            D = pos[T.polar[row]]
            v1 = _unit(_mic(pos[T.donor_h[k]] - D, L))
            v2 = _unit(_mic(pos[T.targets[cand]] - D, L))
            ang = np.degrees(np.arccos(np.clip(v2 @ v1, -1.0, 1.0)))
            ok = ang <= HB_ANGLE
            if ok.any():
                best = cand[ok][np.argmin(dist[m][ok])]
                cls = int(T.target_class[best])
                res = int(T.target_resix[best])
                if cls != WATER_CLASS:
                    n_hb_lipid += 1
                    bound.add(res)
        ints += [cls, res]
    f["n_hb_lipid"] = n_hb_lipid
    f["n_lipids_bound"] = len(bound)
    wat = pairs[T.target_class[pairs[:, 1]] == WATER_CLASS, 1]
    f["n_wat_polar"] = np.unique(wat).size

    return np.array([f[c] for c in FLOAT_COLS]), np.array(ints), glyc


def check_stereochemistry(T: _Topology, pos, box):
    """CIP descriptors of the permeant and of every lipid stereocentre."""
    L = box[:3].astype(float)
    pos = pos.astype(float)
    perm = str(cip_label(signed_volume(pos[T.stereo], pos[T.cip[:3]], L)))
    subs = np.stack([pos[T.O21], pos[T.C1], pos[T.C3]], axis=1)
    lip = cip_label(signed_volume(pos[T.C2], subs, L))
    return {"permeant": perm,
            "enantiomer": "L" if perm == T.spec.l_config else "D",
            "lipids_R": int(np.sum(lip == "R")),
            "lipids_S": int(np.sum(lip == "S"))}


#: Bump when the cached content changes; old chunks are then ignored and rebuilt.
CACHE_VERSION = 4
#: Every cache chunk ends in this; anything else in a cache folder is ignored.
CHUNK_SUFFIX = ".chiral.npz"
#: Cache folder inside each run directory, unless one is given explicitly.
DEFAULT_CACHE = "chiral_cache"


def _read_xtc(xtc, n_atoms):
    """
    Yield (frame index, positions [A], box dimensions [A, deg], time [ps]) for
    every frame of `xtc`, in file order.

    Uses MDAnalysis's low-level XTC reader instead of a Universe trajectory.
    The Universe reader builds a frame-offset index on opening and saves it,
    with a lock file, as two hidden files next to every xtc it reads
    (.<name>.xtc_offsets.npz/.lock) - two inodes per trajectory, inside the
    run's own load/ folder.  Reading every frame in order needs no offsets,
    so nothing is written here.
    """
    from MDAnalysis.lib.formats.libmdaxdr import XTCFile
    from MDAnalysis.lib.mdamath import triclinic_box

    with XTCFile(xtc) as fh:
        if fh.n_atoms != n_atoms:
            raise ValueError(f"{xtc} has {fh.n_atoms} atoms, the topology {n_atoms}")
        for i, frame in enumerate(fh):
            box = np.asarray(triclinic_box(*frame.box), dtype=np.float32)
            box[:3] *= 10.0                                   # nm -> A
            yield i, frame.x * 10.0, box, float(frame.time)


def extract_unit(T: _Topology, unit):
    """
    Features of exactly the phase points `unit.plan` lists, in path order.

    Each xtc is read once, front to back, and features are computed only for
    the frames the plan uses: the xtc files also carry frames outside the
    path, which no analysis here looks at.  A frame index of None (plain MD)
    takes every frame of the file.

    Returns {F, I, G, dt, stereo}: F/I/G one row per phase point; dt the frame
    spacing in ps (the first positive spacing found); stereo the CIP check of
    the first frame (see check_stereochemistry).
    """
    need = {}                                     # file -> frame indices, None = all
    for fname, idx in unit.plan:
        if idx is None:
            need[fname] = None
        elif need.get(fname, ()) is not None:
            need.setdefault(fname, set()).add(idx)
    edges = np.linspace(-180.0, 180.0, N_GLYC_BINS + 1)
    feats, n_frames = {}, {}
    dt, stereo = None, None
    for fname, idx in need.items():
        prev_t, n = None, 0
        for i, pos, box, t in _read_xtc(fname, T.n_atoms):
            n = i + 1
            if dt is None and prev_t is not None and t > prev_t:
                dt = t - prev_t
            prev_t = t
            if idx is not None and i not in idx:
                continue
            if stereo is None:
                stereo = check_stereochemistry(T, pos, box)
            f_row, i_row, glyc = _frame_features(T, pos, box)
            # Bulk glycerol dihedral histogram per frame, so trimmed frames
            # drop out of the bulk reference too.
            g_row = np.stack([np.histogram(glyc[a], bins=edges)[0] for a in range(2)])
            feats[(fname, i)] = (f_row, i_row, g_row)
        n_frames[fname] = n
        if idx and max(idx) >= n:
            raise ValueError(f"{fname}: traj.txt asks for frame {max(idx)}, "
                             f"but the file has {n} frames")
    rows = []
    for fname, idx in unit.plan:
        if idx is None:
            rows += [feats[(fname, k)] for k in range(n_frames[fname])]
        else:
            rows.append(feats[(fname, idx)])
    n_int = 1 + 2 * len(T.spec.donors)
    return {
        "F": np.array([r[0] for r in rows], dtype=np.float32).reshape(-1, len(FLOAT_COLS)),
        "I": np.array([r[1] for r in rows], dtype=np.int32).reshape(-1, n_int),
        "G": np.array([r[2] for r in rows], dtype=np.uint16).reshape(-1, 2, N_GLYC_BINS),
        "dt": 0.0 if dt is None else dt,
        "stereo": stereo or {},
    }


# ── The cache: chunks of paths ────────────────────────────────────────────────
#
# One .npz holds up to `per_file` units (paths), with each unit's phase points
# concatenated and an index of where each starts.  Compared with one file per
# xtc this needs a few dozen files per stage instead of three inodes per
# trajectory, and the frames of one path are stored exactly once, in path
# order.  Each unit's entry carries its signature (see Unit), which is what
# says whether the entry is still valid: compare needs only the cache and the
# small text files, never the trajectories.
#
# A chunk is written to a temporary name and renamed into place once all its
# units are done, so a run killed half-way loses only the unfinished chunks,
# and rerunning extract resumes with them.

def _natural(key):
    return (0, int(key), "") if key.isdigit() else (1, 0, key)


def _chunk_name(tag, unit, pos, per_file):
    """Infretis paths are grouped by path number, so a chunk's membership does
    not shift when paths are added; plain-MD files by position."""
    number = unit.uid if isinstance(unit.uid, (int, np.integer)) else pos
    lo = (int(number) // per_file) * per_file
    return f"{tag}_{lo:07d}-{lo + per_file - 1:07d}{CHUNK_SUFFIX}"


def _chunk_index(path):
    """(version, permeant, {key: signature}) of a chunk without decompressing
    its frames, or None if the file cannot be read (e.g. truncated)."""
    try:
        with np.load(path) as d:
            return (int(d["version"]), str(d["permeant"]),
                    dict(zip(d["keys"].tolist(), d["signatures"].tolist())))
    except Exception:
        return None


def load_chunk(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def chunk_entry(chunk, pos):
    """(arrays {F, I, G}, dt, stereo) of the unit at position `pos` of a chunk."""
    a, b = int(chunk["starts"][pos]), int(chunk["starts"][pos + 1])
    stereo = {k: chunk[f"stereo_{k}"][pos].item()
              for k in ("permeant", "enantiomer", "lipids_R", "lipids_S")}
    return ({k: chunk[k][a:b] for k in ("F", "I", "G")},
            float(chunk["dt"][pos]), stereo)


def _chunk_entries(chunk):
    """{key: (signature, result)} of every unit in a loaded chunk."""
    out = {}
    for pos, (key, sig) in enumerate(zip(chunk["keys"].tolist(),
                                         chunk["signatures"].tolist())):
        arr, dt, stereo = chunk_entry(chunk, pos)
        out[key] = (sig, {**arr, "dt": dt, "stereo": stereo})
    return out


def _write_chunk(path, entries, permeant):
    """entries: {key: (signature, result)} -> one compressed .npz, atomically."""
    keys = sorted(entries, key=_natural)
    res = [entries[k][1] for k in keys]
    lengths = [len(r["F"]) for r in res]
    stereo = {k: [r["stereo"].get(k, "" if k in ("permeant", "enantiomer") else -1)
                  for r in res]
              for k in ("permeant", "enantiomer", "lipids_R", "lipids_S")}
    tmp = path[:-len(".npz")] + ".tmp.npz"
    np.savez_compressed(
        tmp, version=CACHE_VERSION, permeant=permeant, columns=np.array(FLOAT_COLS),
        keys=np.array(keys), signatures=np.array([entries[k][0] for k in keys]),
        starts=np.r_[0, np.cumsum(lengths)].astype(np.int64),
        dt=np.array([r["dt"] for r in res], dtype=float),
        F=np.concatenate([r["F"] for r in res]),
        I=np.concatenate([r["I"] for r in res]),
        G=np.concatenate([r["G"] for r in res]),
        **{f"stereo_{k}": np.array(v) for k, v in stereo.items()})
    os.replace(tmp, path)


def _update_chunk(path, new, permeant, overwrite):
    """Write `new` {key: (signature, result)} into the chunk at `path`, keeping
    the old chunk's still-valid entries for every other unit."""
    entries = {}
    if not overwrite and os.path.exists(path):
        info = _chunk_index(path)
        if info is not None and info[0] == CACHE_VERSION and info[1] == permeant:
            entries = _chunk_entries(load_chunk(path))
    entries.update(new)
    if entries:
        _write_chunk(path, entries, permeant)


_TOPO: _Topology | None = None


def _init_extract(tpr, permeant):
    global _TOPO
    _TOPO = _Topology(tpr, PERMEANTS[permeant])


def _extract_job(job):
    name, unit = job
    try:
        return name, unit.key, unit.signature, extract_unit(_TOPO, unit), None
    except Exception as exc:
        return name, unit.key, unit.signature, None, f"{type(exc).__name__}: {exc}"


def extract_many(units, tpr, permeant, cache_dir, tag, workers=8, overwrite=False,
                 per_file=500):
    """
    Extract every unit not yet cached (or cached from a different traj.txt,
    permeant or cache version) into chunks of `per_file` units in `cache_dir`.
    `tag` names the chunks, so several stages can share one folder.  Returns
    [(key, error)] of the units that failed; they are simply absent from the
    cache and are retried by the next run.
    """
    os.makedirs(cache_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(cache_dir, f"*{CHUNK_SUFFIX[:-4]}.tmp.npz")):
        os.remove(stale)                      # left behind by a killed run
    groups = {}
    for pos, u in enumerate(units):
        groups.setdefault(_chunk_name(tag, u, pos, per_file), []).append(u)
    todo = {}
    for name, members in groups.items():
        info = None if overwrite else _chunk_index(os.path.join(cache_dir, name))
        valid = info is not None and info[0] == CACHE_VERSION and info[1] == permeant
        have = info[2] if valid else {}
        missing = [u for u in members if have.get(u.key) != u.signature]
        if missing:
            todo[name] = missing
    n_todo = sum(len(m) for m in todo.values())
    print(f"  {len(units) - n_todo} units cached already, {n_todo} to extract "
          f"into {len(todo)} of {len(groups)} chunk file(s) in {cache_dir}")
    errors = []
    if not todo:
        return errors
    # Resolve the topology once here: a wrong -tpr/-permeant then fails with
    # its own message, instead of killing every worker's initializer and
    # surfacing as an opaque BrokenProcessPool.
    _Topology(tpr, PERMEANTS[permeant])
    remaining = {name: len(m) for name, m in todo.items()}
    new = {name: {} for name in todo}
    jobs = [(name, u) for name in sorted(todo) for u in todo[name]]
    # map() hands results back in submission order, which is chunk order, so
    # at most about one chunk's results are held here at a time.
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_extract,
                             initargs=(tpr, permeant)) as ex:
        for k, (name, key, sig, res, err) in enumerate(ex.map(_extract_job, jobs), 1):
            if err:
                errors.append((key, err))
            else:
                new[name][key] = (sig, res)
            remaining[name] -= 1
            if remaining[name] == 0:
                _update_chunk(os.path.join(cache_dir, name), new.pop(name),
                              permeant, overwrite)
            if k % 200 == 0:
                print(f"    ... {k}/{n_todo} units")
    print(f"  extracted {n_todo - len(errors)}, failed {len(errors)}")
    for key, err in errors[:10]:
        print(f"    [!] {key}: {err}")
    return errors


# ══════════════════════════════════════════════════════════════════════════════
# Units (paths) and per-unit reduction
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Unit:
    """
    One statistical unit.  `plan` lists (xtc path, frame index) per phase
    point in path order.  A frame index of None means every frame of that
    file.  It is empty on units built for compare, which reads the cache
    instead.  Units with an `order_file` are infretis paths.

    `signature` identifies the unit's content for the cache: the SHA-1 of
    traj.txt for a path (it names every file and frame), size and mtime of the
    xtc for a plain-MD file.  `cache_dir` is where its extracted features live.

    Stage context, set by `setup_stages` (see "Path-ensemble reweighting"):
      weight      WFtot of the path times its stage factor
      weight_uniform  the same without the WHAM reweighting: 1/N_plus for a
                  plus path, 1/N_minus for a [0-] path, times the stage factor
      trim        (lambda_A, lambda_minus_one) for end-point trimming, or None
      stage       index of the stage in the chain (0 = entry)
      lam_a       the stage's lambda_A on the order-parameter axis
      keep_below  frames with OP <= lam_a: kept on stage 0, where the
                  Ensemble scales them by 1/xi; cut on every later stage
      xi_end      'R' or 'L' for a [0-] path of stage 0 that ends at lambda_A
                  or at lambda_-1 (what xi counts), else None
      lam_b, lam_m1  the stage's lambda_B and lambda_-1, to classify the path
      path_class  'reactive' (a plus path that ends at lambda_B),
                  'unreactive' (a plus path back at lambda_A) or 'minus' (a
                  [0-] path); set when the path is reduced (assemble_path)
    """
    uid: int | str
    plan: list
    weight: float = 1.0
    order_file: str | None = None
    g96_index: str | None = None
    trim: tuple | None = None
    stage: int = 0
    lam_a: float | None = None
    keep_below: bool = True
    xi_end: str | None = None
    weight_uniform: float = 1.0
    lam_b: float | None = None
    lam_m1: float | None = None
    path_class: str | None = None
    signature: str = ""
    cache_dir: str | None = None

    @property
    def files(self):
        return list(dict.fromkeys(f for f, _ in self.plan))

    @property
    def key(self):
        return str(self.uid)


def infretis_path_unit(path_dir, uid, cache_dir=None, keep_plan=True):
    """
    A path as cv_generation reads it: the traj.txt plan, plus order.txt.

    Only extract follows the plan.  compare needs just the signature and
    g96_index, so it passes keep_plan=False and the plan is left empty: kept,
    it costs ~180 bytes per phase point, which over every path of every
    stage runs to tens of GB.
    """
    traj = os.path.join(path_dir, "traj.txt")
    plan, g96_index = read_traj_plan(traj)
    with open(traj, "rb") as fh:
        signature = hashlib.sha1(fh.read()).hexdigest()
    acc = os.path.join(path_dir, "accepted")
    return Unit(uid, [(os.path.join(acc, f), i) for f, i in plan] if keep_plan else [],
                order_file=os.path.join(path_dir, "order.txt"), g96_index=g96_index,
                signature=signature, cache_dir=cache_dir)


def units_from_load_dir(load_dir, path_numbers=None, cache_dir=None, keep_plan=True):
    if path_numbers is None:
        path_numbers = sorted(int(p.name) for p in Path(load_dir).iterdir()
                              if p.name.isdigit() and (p / "traj.txt").exists())
    return [infretis_path_unit(os.path.join(load_dir, str(pn)), pn, cache_dir, keep_plan)
            for pn in path_numbers]


def units_from_files(xtc_files, cache_dir=None):
    units = []
    for x in sorted(xtc_files):
        st = os.stat(x)
        units.append(Unit(os.path.abspath(x), [(x, None)],
                          signature=f"{st.st_size}:{int(st.st_mtime)}",
                          cache_dir=cache_dir))
    return units


# ── Trimming, as chiroflux.cv_generation does it ─────────────────────────────

def read_lambdas(toml_file):
    """(lambda_A, lambda_minus_one) from an infretis.toml, via cv_generation's
    own read_toml.  Imported lazily: cv_generation pulls in typer and sets
    OMP_NUM_THREADS on import."""
    from .cv_generation import read_toml
    if not os.path.isfile(toml_file):
        raise FileNotFoundError(f"{toml_file} not found; pass -toml-l/-toml-d "
                                f"or -no-trim")
    lambda_A, _, lambda_minus_one = read_toml(toml_file)
    return lambda_A, lambda_minus_one


def wham_weights(data_file, toml_file, nskip=1000):
    """
    {path number: WHAM weight}, computed exactly as chiroflux does
    (pathdata._load_path_table + _compute_path_weights).  The first `nskip`
    rows of infretis_data.txt are burn-in; paths that contribute to no plus
    ensemble are absent, and so get no weight.
    """
    interfaces = _load_interfaces_from_toml(toml_file)
    with warnings.catch_warnings():
        # numpy's loadtxt note about the blank header line, and 0/0 in rows
        # that np.where then discards: both harmless.
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        pnr, maxop, path_f, path_w = _load_path_table(data_file, nskip, len(interfaces))
        w = _compute_path_weights(maxop, path_f, path_w, interfaces)
    return dict(zip(pnr.tolist(), w.tolist()))


# ── Path-ensemble reweighting ─────────────────────────────────────────────────
#
# A TIS run samples paths that START in its own state A.  Its WHAM-weighted
# histogram is therefore a CONDITIONAL density, and -ln of it is a conditional
# free energy, not the free energy of the permeant.  It rises towards the
# stage's far interface by -ln P_cross.  Two corrections turn it into the
# equilibrium density.  They are the same ones inftools' Wham_Pcross and
# chiroflux's permeation_process apply to the order-parameter histogram, and
# here they are applied frame by frame to every accumulator:
#
#   1. Within a stage (inftools calculate_free_energy): every frame carries its
#      path's WFtot: its plus-ensemble WHAM weight if it is a plus path, its
#      [0-] weight if it is a [0-] path (no path is both).  Each of the two
#      sets is normalised to sum to 1 over its paths.  Frames at OP <= lambda_A come from the [0-]
#      ensemble, cut off at lambda_A, and are divided by xi, the weighted
#      fraction of [0-] paths that end at lambda_A rather than lambda_-1.
#   2. Across stages (permeation_process.stage_factors): stage k is multiplied
#      by the probability of reaching it, chained over the junctions from each
#      run's wham/Pcross.txt.  Every stage after the first is cut below its own
#      lambda_A.  The forward histograms are then added to their mirror image
#      about OP = 0, which supplies each region's backward histogram.  Binning
#      on |OP| is exactly that sum.
#
# Every observable here is invariant under the C2 rotation that maps one
# leaflet onto the other, so the mirror needs no value transformation.

def minus_weights(data_file, toml_file, nskip=1000):
    """{path: [0-] weight}: f/w of the [0-] ensemble column, normalised to 1,
    as Wham_Pcross computes WHAMfactorsMIN."""
    n_intf = len(_load_interfaces_from_toml(toml_file))
    rows = []
    with open(data_file) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line.split())
    out = {}
    for cols in rows[nskip:]:
        f, w = cols[3], cols[3 + n_intf]
        if f.startswith("--") or w.startswith("--") or float(w) <= 0:
            continue
        if float(f) > 0:
            out[int(cols[0])] = float(f) / float(w)
    total = sum(out.values())
    return {p: v / total for p, v in out.items()} if total > 0 else {}


def combine_weights(plus, minus):
    """WFtot from the plus-ensemble WHAM weights and the [0-] weights (two
    disjoint sets of paths, each summing to 1)."""
    out = dict(plus)
    for p, v in minus.items():
        out[p] = out.get(p, 0.0) + v
    return out


def uniform_weights(plus, minus):
    """
    The unweighted counterpart of WFtot: the same paths (after the burn-in,
    contributing to some ensemble), but every plus path weighs 1/N_plus and
    every [0-] path 1/N_minus.  Each set still sums to 1, as in WFtot, so the
    stages are stitched on the same scale and only the WHAM reweighting
    within each set is left out.
    """
    P = [p for p, v in plus.items() if v > 0]
    M = [p for p, v in minus.items() if v > 0]
    out = {p: 1.0 / len(P) for p in P}
    for p in M:
        out[p] = out.get(p, 0.0) + 1.0 / len(M)
    return out


def stage_path_weights(data_file, toml_file, nskip=1000):
    """{path: WFtot}, the per-path weight inftools writes to
    wham/path_weights.txt and histograms with: the plus-ensemble WHAM weight
    for a plus path, the [0-] weight for a [0-] path.  The two sets are
    disjoint and each sums to 1, so WFtot sums to 2 over the stage; the
    addition below never actually adds two weights for one path."""
    return combine_weights(wham_weights(data_file, toml_file, nskip),
                           minus_weights(data_file, toml_file, nskip))


def path_type(op, lambda_minus_one, lambda_A):
    """Wham_Pcross's extract_pathtype_and_tau_ref classification, from the raw
    order.txt order parameter."""
    first, second, last = op[0], op[1], op[-1]
    if first >= lambda_A and last >= lambda_A and second < first:
        return "RMR"
    if first >= lambda_A and last <= lambda_minus_one:
        return "RML"
    if first <= lambda_minus_one and last <= lambda_minus_one:
        return "LML"
    if first <= lambda_minus_one and last >= lambda_A:
        return "LMR"
    return "0+"


def wham_xi(units, weights, lambda_A, lambda_minus_one):
    """
    xi = R_END / (R_END + L_END), the WFtot-weighted fraction of [0-] paths
    ending at lambda_A (types LMR, RMR) rather than at lambda_-1 (LML, RML).
    This is the final value of Wham_Pcross's running average.  Returns
    (xi, number of weighted paths whose order.txt was missing), and sets
    each counted unit's `xi_end`, so the bootstrap can recompute xi from
    every resample (Ensemble.xi).
    """
    r_end = l_end = 0.0
    missing = 0
    for u in units:
        w = weights.get(u.uid, 0.0)
        if w <= 0:
            continue
        if not u.order_file or not os.path.exists(u.order_file):
            missing += 1
            continue
        op = np.loadtxt(u.order_file, comments=("#", "@"), ndmin=2)[:, 1]
        kind = path_type(op, lambda_minus_one, lambda_A)
        if kind in ("LMR", "RMR"):
            r_end += w
            u.xi_end = "R"
        elif kind in ("LML", "RML"):
            l_end += w
            u.xi_end = "L"
    xi = r_end / (r_end + l_end) if r_end + l_end > 0 else np.nan
    return xi, missing


@dataclass
class StageSpec:
    run_dir: str
    data_file: str
    toml_file: str
    cache_dir: str | None = None      # default: RUN_DIR/chiral_cache


def _find_toml(run_dir):
    for cand in ("infretis.toml", os.path.join("post", "infretis.toml")):
        path = os.path.join(run_dir, cand)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"no infretis.toml in {run_dir} or {run_dir}/post; "
                            f"give it with -toml-l/-toml-d")


def read_pcross_rel_err(run_dir, wham_subdir="wham", pcross_col=3):
    """
    Relative standard error of a stage's total crossing probability, as
    inftools' block analysis writes it on the first line of
    wham/errPtot.txt ("#averaged rel-error PM, WHAM: [pm, wham]").  Columns
    1 and 3 of Pcross.txt (P-wham, P-wham2) take the WHAM error, column 2
    (P-point-matching) the point-matching one.  NaN if the file is missing
    or unreadable.
    """
    path = os.path.join(run_dir, wham_subdir, "errPtot.txt")
    try:
        with open(path) as fh:
            head = fh.readline()
    except OSError:
        return np.nan
    if "rel-error" not in head or ":" not in head:
        return np.nan
    text = re.sub(r"np\.float\d*\(", "", head.split(":", 1)[1])
    vals = [float(v) for v in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text)]
    if len(vals) < 2:
        return np.nan
    return vals[0] if pcross_col == 2 else vals[1]


@dataclass
class StageFactorModel:
    """
    The stage factors of one enantiomer and how uncertain they are.  Each
    bootstrap replicate draws every stage's Pcross from a log-normal around
    its value, with the log-SD that its relative error implies, and chains
    the factors again exactly as setup did (permeation_process.stage_factors,
    so the mirror substitutions keep their correlations).  A stage without an
    error estimate keeps its Pcross fixed.
    """
    stages: list
    rel_err: np.ndarray
    factors: np.ndarray

    def draw(self, rng):
        """Drawn factor / point factor, per stage."""
        from dataclasses import replace

        from .permeation_process import stage_factors

        if len(self.stages) < 2:
            return np.ones(len(self.stages))
        err = np.nan_to_num(np.asarray(self.rel_err, float), nan=0.0)
        sig = np.sqrt(np.log1p(err ** 2))
        drawn = [replace(s, pcross=float(min(s.pcross * np.exp(g * rng.standard_normal()), 1.0)))
                 for s, g in zip(self.stages, sig)]
        return np.asarray(stage_factors(drawn), float) / self.factors


def setup_stages(specs, nskip=1000, wham_subdir="wham", pcross_col=3,
                 trim=True, path_numbers=None):
    """
    Units of every stage of one enantiomer, weighted onto one equilibrium
    scale (see the section comment).  `specs` are in chain order, entry first.
    Each unit carries both weights: `weight` (WFtot, the WHAM reweighting)
    and `weight_uniform` (the same paths, equal within plus and within [0-]).
    Returns (units, per-stage summary table, StageFactorModel).
    """
    from .permeation_process import load_stage, stage_factors

    stages = [load_stage(s.run_dir, wham_subdir, pcross_col) for s in specs]
    factors = stage_factors(stages) if len(stages) > 1 else [1.0]
    rel_err = np.array([read_pcross_rel_err(s.run_dir, wham_subdir, pcross_col)
                        for s in specs])
    model = StageFactorModel(stages, rel_err, np.asarray(factors, float))
    all_units, rows = [], []
    for k, (spec, stage, factor) in enumerate(zip(specs, stages, factors)):
        lam_a, lam_m1 = read_lambdas(spec.toml_file)
        if abs(lam_a - stage.lam_a) > 1e-6:
            raise ValueError(f"{spec.toml_file} has lambda_A = {lam_a} but "
                             f"{spec.run_dir}/{wham_subdir} was made with "
                             f"lA = {stage.lam_a}; wrong toml for this run?")
        cache_dir = spec.cache_dir or os.path.join(spec.run_dir, DEFAULT_CACHE)
        units = units_from_load_dir(os.path.join(spec.run_dir, "load"), path_numbers,
                                    cache_dir, keep_plan=False)
        plus = wham_weights(spec.data_file, spec.toml_file, nskip)
        minus = minus_weights(spec.data_file, spec.toml_file, nskip)
        wtot = combine_weights(plus, minus)
        wuni = uniform_weights(plus, minus)
        xi, missing = (wham_xi(units, wtot, lam_a, lam_m1) if k == 0
                       else (np.nan, 0))
        # The same ratio with the uniform weights; xi_end is set by wham_xi.
        r = sum(wuni.get(u.uid, 0.0) for u in units if u.xi_end == "R")
        l = sum(wuni.get(u.uid, 0.0) for u in units if u.xi_end == "L")
        xi_uni = r / (r + l) if k == 0 and r + l > 0 else np.nan
        if k == 0 and not np.isfinite(xi):
            raise ValueError(f"xi undefined for {spec.run_dir}: no weighted [0-] "
                             f"path ends at lambda_A or lambda_-1")
        for u in units:
            u.weight = wtot.get(u.uid, 0.0) * factor
            u.weight_uniform = wuni.get(u.uid, 0.0) * factor
            u.trim = (lam_a, lam_m1) if trim else None
            u.stage, u.lam_a, u.lam_b, u.lam_m1 = k, lam_a, stage.lam_b, lam_m1
            u.keep_below = k == 0
        kept = [u for u in units if u.weight > 0]
        all_units += kept
        rows.append({"stage": k, "run": spec.run_dir, "lam_a": lam_a,
                     "lam_b": stage.lam_b, "pcross": stage.pcross,
                     "pcross_rel_err": rel_err[k],
                     "factor": factor, "xi": xi, "xi_unweighted": xi_uni,
                     "paths_in_load": len(units),
                     "paths_weighted": len(kept),
                     "weighted_without_order_txt": missing})
    return all_units, pd.DataFrame(rows), model


def path_class(op_raw, lambda_A, lambda_B, lambda_minus_one):
    """
    'minus' for a [0-] path; for a plus path 'reactive' if it ends at or
    beyond the stage's lambda_B (it made it across the stage), else
    'unreactive' (it went back to lambda_A).  From the raw order.txt OP.
    """
    if path_ensemble(op_raw, lambda_A, lambda_minus_one) == "minus":
        return "minus"
    return "reactive" if op_raw[-1] >= lambda_B else "unreactive"


def path_ensemble(op_raw, lambda_A, lambda_minus_one):
    """'plus' or 'minus' from the RAW order.txt order parameter (before the g96
    row is dropped), the rule in cv_generation.process_single_path.  A missing
    lambda_minus_one is taken as -inf; cv_generation would raise there."""
    lm1 = -np.inf if lambda_minus_one is None else lambda_minus_one
    if lm1 < op_raw[0] < lambda_A or lm1 < op_raw[-1] < lambda_A:
        return "plus"
    return "minus"


def endpoint_keep_mask(op, ensemble, lambda_A, lambda_minus_one):
    """Keep-mask implementing write_ML_txt's first/last phase-point deletion."""
    lm1 = -np.inf if lambda_minus_one is None else lambda_minus_one

    def drop(x):
        return ((ensemble == "plus" and x < lambda_A) or
                (ensemble == "minus" and (x < lm1 or x > lambda_A)))

    keep = np.ones(len(op), dtype=bool)
    keep[0] = not drop(op[0])
    keep[-1] = keep[-1] and not drop(op[-1])
    return keep


@dataclass
class Grid:
    # |OP| in the order parameter's own unit (Angstrom here); plain-MD units
    # use the stereocentre depth z converted to Angstrom instead.
    depth_edges: np.ndarray = field(default_factory=lambda: np.arange(0.0, 35.01, 2.5))
    contact_cut: float = 0.7          # nm, permeant stereocentre to lipid C2
    n_phi: int = 24
    n_cos: int = 12
    life_edges: np.ndarray = field(default_factory=lambda: np.unique(
        np.round(np.logspace(0, 4, 25)).astype(int)))
    # Coarser depth bins for the depth-resolved pose maps: a (phi, cos) map
    # per bin is 288 numbers per path, and contacts end around 30 A.
    pose_depth_edges: np.ndarray = field(default_factory=lambda: np.arange(0.0, 30.01, 5.0))
    # The histograms cover the depth bins whose centres lie below this |OP|;
    # None: compare sets |lambda_A| of the entry stage, so the [0-] paths,
    # which only reach beyond it, stay out (no stages: every finite bin).
    hist_max_depth: Optional[float] = None

    @property
    def n_depth(self):
        return len(self.depth_edges)   # last bin is open-ended bulk water

    @property
    def n_pose_depth(self):
        return len(self.pose_depth_edges)

    @staticmethod
    def _bin(z, edges):
        return np.clip(np.digitize(np.abs(z), edges) - 1, 0, len(edges) - 1)

    @staticmethod
    def _labels(e):
        return [f"{a:.2f}-{b:.2f}" for a, b in zip(e[:-1], e[1:])] + [f">{e[-1]:.2f}"]

    def depth_bin(self, z):
        return self._bin(z, self.depth_edges)

    def depth_labels(self):
        return self._labels(self.depth_edges)

    def pose_depth_bin(self, z):
        return self._bin(z, self.pose_depth_edges)

    def pose_depth_labels(self):
        return self._labels(self.pose_depth_edges)


#: (name, column or derived key, only in frames in contact with a stereocentre)
PROFILE_OBS = (
    ("p_contact",      "contact",        False),  # fraction of frames at a stereocentre
    ("d_c2",           "d_c2",           False),  # nm to nearest lipid stereocentre
    ("cos_axis_n",     "cos_axis_n",     False),  # molecular axis . outward normal
    ("cos_chi_n",      "cos_chi_n",      False),  # permeant pseudovector . outward normal
    ("n_hb_lipid",     "n_hb_lipid",     False),
    ("n_lipids_bound", "n_lipids_bound", False),  # >1: bridging two lipids
    ("n_wat_polar",    "n_wat_polar",    False),  # hydration of the polar groups
    ("pose_z",         "pose_z",         True),   # face of the lipid stereocentre
    ("chi_lz",         "chi_lz",         True),   # relative handedness n_perm . n_lipid
    ("axis_lz",        "axis_lz",        True),
)
#: Observables that are D = -L when the lipids' handedness plays no role in
#: where and how the permeant sits, and L + D is tested for them instead of
#: L - D.  The mirror image of L+R is D+S, so with no chiral recognition D+R
#: would behave as D+S.
#:   cos_chi_n  the permeant's pseudovector against the membrane normal,
#:              which is not chiral: D = -L exactly.
#:   chi_lz     the same pseudovector against the contacted lipid's e_z.
#:              It is a true scalar under a mirror of the whole system, but
#:              e_z is built from the chiral lipid and leans along the
#:              membrane normal (by ~0.18 in contact), so any preferred
#:              orientation of the permeant (cos_chi_n) passes into it with
#:              opposite signs for L and D: D ~ -L without any recognition.
#:              Only the departure from that, L + D, needs the lipid's
#:              handedness to act on the permeant.
#: (pose_z and axis_lz pair e_z with true vectors (the direction to the
#: permeant, its N->C axis), which are alike for L and D without
#: recognition, so for them L - D stays the test.)
MIRROR_NULL_OBS = ("cos_chi_n", "chi_lz")

#: The per-frame distributions compare histograms, per depth bin: (observable
#: = FLOAT_COLS column, value-bin edges, only in contact frames as in
#: PROFILE_OBS).  Values outside the edges go to the end bins.  The
#: MIRROR_NULL_OBS need edges symmetric about 0, so a histogram can be
#: mirrored by reversing it.
_COS_EDGES = np.linspace(-1.0, 1.0, 31)
HIST_OBS = (
    ("d_c2",           np.linspace(0.2, 2.0, 37), False),   # nm
    ("cos_axis_n",     _COS_EDGES,                False),
    ("cos_chi_n",      _COS_EDGES,                False),
    ("n_hb_lipid",     np.arange(-0.5, 3.0),      False),   # 0, 1, 2
    ("n_lipids_bound", np.arange(-0.5, 3.0),      False),
    ("n_wat_polar",    np.arange(-0.5, 16.0),     False),   # 0 ... 15 (and more)
    ("pose_z",         _COS_EDGES,                True),
    ("chi_lz",         _COS_EDGES,                True),
    ("axis_lz",        _COS_EDGES,                True),
)
#: Where each observable's value bins start in the per-path "hist" array,
#: which is (depth bin, all value bins side by side), float32: about 12 kB a
#: path, against ~30 kB for everything else reduce_path keeps.
HIST_OFFSETS = np.r_[0, np.cumsum([len(e) - 1 for _, e, _ in HIST_OBS])].astype(int)
HIST_NBINS = int(HIST_OFFSETS[-1])
#: Depth windows (|OP|, OP units) of the 1D histograms: the depth bins whose
#: centres fall inside a window are pooled.  12.5-15 is the molecular-axis
#: result on its own.
HIST_WINDOWS = ((0.0, 5.0), (5.0, 10.0), (10.0, 12.5), (12.5, 15.0), (15.0, 17.5),
                (17.5, 20.0), (20.0, 25.0)) #,(25.0, 30.0))  # [0-] paths are currently not in included
# The -hist-max-depth sets the inclusion of certain frames for the 2D histograms.


#: Reorientation relative to the contacted lipid: the vectors followed (name,
#: their lipid-frame columns) and the lags (frames) of the time correlation
#: C(t) = <u(0) . u(t)>, taken over frame pairs in contact with the SAME lipid
#: at both times.  Log-spaced, 1 to 2000 frames (0.2 to 400 ps at 0.2 ps):
#: relative to the touched lipid the permeant turns slowly (C of the axis is
#: still ~0.7 at 40 ps), and paths last ~1 ns, so the long lags are sampled.
ROT_VECTORS = (("axis", ("axis_lx", "axis_ly", "axis_lz")),
               ("chirality", ("chi_lx", "chi_ly", "chi_lz")),
               ("position", ("pose_x", "pose_y", "pose_z")))
ROT_LAGS = np.array([1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 500, 800, 1200, 2000])
#: Lags (frames) at which reorientation reports C itself, next to tau: values
#: that compare directly even where C has not decayed by the longest lag.
ROT_REPORT_LAGS = (20, 200, 2000)


def _hist_layout():
    """(observable, edges, only_contact, slice into the value axis) per HIST_OBS."""
    return [(name, edges, only, slice(int(a), int(b)))
            for (name, edges, only), a, b in zip(HIST_OBS, HIST_OFFSETS[:-1], HIST_OFFSETS[1:])]
LIFE_KINDS = CLASS_NAMES + ["lipid_contact"]


def cache_index(cache_dir):
    """{key: [(chunk path, position, signature)]} over every current-version
    chunk in `cache_dir`.  Reads only the chunks' small index arrays."""
    index = {}
    for path in sorted(glob.glob(os.path.join(cache_dir, f"*{CHUNK_SUFFIX}"))):
        info = _chunk_index(path)
        if info is None or info[0] != CACHE_VERSION:
            continue
        for pos, (key, sig) in enumerate(info[2].items()):
            index.setdefault(key, []).append((path, pos, sig))
    return index


def _locate(unit, index):
    """(chunk path, position) of `unit` in its cache index; raises if it is
    missing or was extracted from a different traj.txt/xtc."""
    hits = index.get(unit.key, [])
    for path, pos, sig in hits:
        if sig == unit.signature:
            return path, pos
    why = ("its traj.txt (or xtc) changed since it was extracted" if hits
           else f"no v{CACHE_VERSION} entry in {unit.cache_dir}")
    raise LookupError(f"not cached: {why}; run `chiroflux chiral-extract` first")


def load_unit(unit):
    """(arrays {F, I, G}, dt, stereo) of one unit, from its cache."""
    path, pos = _locate(unit, cache_index(unit.cache_dir))
    return chunk_entry(load_chunk(path), pos)


def _runs(keys):
    """(value, length) of every maximal run of equal consecutive entries."""
    if keys.size == 0:
        return keys, keys
    b = np.flatnonzero(np.diff(keys)) + 1
    starts = np.r_[0, b]
    return keys[starts], np.diff(np.r_[starts, keys.size])


def assemble_path(unit: Unit, arr, dt):
    """
    The cached frames of one unit, checked and trimmed.

    `arr` holds one row per phase point exactly as `unit.plan` names them, in
    path order (reversed segments list descending indices, so they need no
    special handling); extract_unit built it that way.  For infretis paths
    the order parameter is then checked row by row against the plan, and, if
    `unit.trim` = (lambda_A, lambda_minus_one), the end points are trimmed,
    both as in cv_generation.

    Returns (arrays {F, I, G}, op, dt, meta).  `op` is the order parameter
    per kept frame; plain-MD units have none and get the stereocentre depth z
    in Angstrom instead.
    """
    meta = {"stage": unit.stage, "uid": unit.uid, "phase_points": len(arr["F"]),
            "ensemble": None, "path_class": None, "deleted_first": False,
            "deleted_last": False}
    if unit.order_file is None:
        return arr, arr["F"][:, FLOAT_COLS.index("z")].astype(float) * 10.0, dt, meta

    # The row-by-row check runs even without trimming: it is what guarantees
    # that frame i is phase point i.
    order = np.loadtxt(unit.order_file, comments=("#", "@"), ndmin=2)
    op_raw = order[:, 1]
    op = op_raw if unit.g96_index is None else np.delete(op_raw, int(unit.g96_index))
    if unit.lam_a is not None and unit.lam_b is not None:
        meta["path_class"] = path_class(op_raw, unit.lam_a, unit.lam_b, unit.lam_m1)
    if len(op) != len(arr["F"]):
        raise ValueError(f"traj.txt maps {len(arr['F'])} phase points but "
                         f"order.txt has {len(op)} rows after the g96 "
                         f"deletion; refusing to guess the correspondence")
    if unit.trim is not None:
        ensemble = path_ensemble(op_raw, *unit.trim)
        keep = endpoint_keep_mask(op, ensemble, *unit.trim)
        arr = {k: v[keep] for k, v in arr.items()}
        op = op[keep]
        meta.update(ensemble=ensemble, deleted_first=not keep[0],
                    deleted_last=not keep[-1])
    return arr, op, dt, meta


def frame_parts(op, unit: Unit):
    """
    (above, below): per-frame weights on top of the unit weight, split at
    lambda_A.  `above` is 1 for OP > lambda_A.  `below` is 1 for the [0-]
    frames at OP <= lambda_A that stage 0 keeps; the Ensemble scales that
    part by 1/xi, so xi can be resampled.  Later stages cut those frames
    (both parts 0).  Units without stages have everything in `above`.
    """
    if unit.lam_a is None:
        return np.ones(len(op)), np.zeros(len(op))
    low = op <= unit.lam_a
    return (~low).astype(float), (low & unit.keep_below).astype(float)


#: Frame counts, used only to decide which bins have enough data.  They are
#: kept with the `above` part only, so they are not counted twice.
RAW_KEYS = ("N_raw", "C_raw", "hb_raw", "pose_n_raw", "rot_n_raw")


def reduce_path(arr, op, dt, grid: Grid, mult=None, live=None):
    """
    Depth-resolved sums for one assembled path, binned on |OP| and weighted
    per frame by `mult`.  Every entry is additive over units, so any weighted
    average is a ratio of weighted sums of these.  Frames that are not
    `live` (default: mult > 0) are left out, and runs for the lifetimes
    break at them.  Given `live`, every weighted entry is linear in `mult`,
    so a path can be reduced in parts that add up to the whole.
    """
    nz, nobs = grid.n_depth, len(PROFILE_OBS)
    col = {c: i for i, c in enumerate(FLOAT_COLS)}
    mult = np.ones(len(op)) if mult is None else np.asarray(mult, dtype=float)
    F, I = arr["F"].astype(float), arr["I"]
    n_don = (I.shape[1] - 1) // 2
    # N/C/S/S2 carry the frame weights; N_raw/C_raw count frames, and are only
    # used to decide which bins have enough data to report.
    out = {"N": np.zeros(nz), "C": np.zeros((nobs, nz)),
           "N_raw": np.zeros(nz), "C_raw": np.zeros((nobs, nz)),
           "S": np.zeros((nobs, nz)), "S2": np.zeros((nobs, nz)),
           "hb": np.zeros((n_don, len(CLASS_NAMES) + 1, nz)),
           "hb_raw": np.zeros((n_don, len(CLASS_NAMES) + 1, nz)),
           "pose": np.zeros((grid.n_phi, grid.n_cos)),
           "pose_depth": np.zeros((grid.n_pose_depth, grid.n_phi, grid.n_cos)),
           "pose_n_raw": np.zeros(grid.n_pose_depth),
           "glyc_partner": np.zeros((2, N_GLYC_BINS)),
           "glyc_bulk": np.tensordot(mult, arr["G"].astype(float), axes=(0, 0)),
           "life_n": np.zeros(len(LIFE_KINDS)),
           "life_sum": np.zeros(len(LIFE_KINDS)),
           "life_hist": np.zeros((len(LIFE_KINDS), len(grid.life_edges) - 1)),
           "hist": np.zeros((nz, HIST_NBINS), dtype=np.float32),
           "rot_sum": np.zeros((len(ROT_VECTORS), grid.n_pose_depth, len(ROT_LAGS))),
           "rot_n": np.zeros((grid.n_pose_depth, len(ROT_LAGS))),
           "rot_n_raw": np.zeros((grid.n_pose_depth, len(ROT_LAGS))),
           "axis_depth": np.zeros((grid.n_pose_depth, grid.n_phi, grid.n_cos), np.float32),
           "chi_depth": np.zeros((grid.n_pose_depth, grid.n_phi, grid.n_cos), np.float32),
           "dt": dt}
    if len(F) == 0:
        return out

    zb = grid.depth_bin(op)
    live = mult > 0 if live is None else np.asarray(live, bool)
    contact = (F[:, col["d_c2"]] < grid.contact_cut) & live
    out["N"] += np.bincount(zb, weights=mult, minlength=nz)
    out["N_raw"] += np.bincount(zb[live], minlength=nz)

    for o, (_, key, only_contact) in enumerate(PROFILE_OBS):
        v = contact.astype(float) if key == "contact" else F[:, col[key]]
        ok = np.isfinite(v) & live & (contact if only_contact else True)
        m = mult[ok]
        out["C"][o] += np.bincount(zb[ok], weights=m, minlength=nz)
        out["C_raw"][o] += np.bincount(zb[ok], minlength=nz)
        out["S"][o] += np.bincount(zb[ok], weights=m * v[ok], minlength=nz)
        out["S2"][o] += np.bincount(zb[ok], weights=m * v[ok] ** 2, minlength=nz)

    # Distributions: one weighted histogram per observable and depth bin.
    flat = np.zeros(nz * HIST_NBINS)
    for name, edges, only_contact, sl in _hist_layout():
        v = F[:, col[name]]
        ok = np.isfinite(v) & live & (contact if only_contact else True)
        vb = np.clip(np.digitize(v[ok], edges) - 1, 0, len(edges) - 2)
        flat += np.bincount(zb[ok] * HIST_NBINS + sl.start + vb, weights=mult[ok],
                            minlength=nz * HIST_NBINS)
    out["hist"] += flat.reshape(nz, HIST_NBINS).astype(np.float32)

    for k in range(n_don):
        cls = I[:, 1 + 2 * k].copy()
        cls[cls < 0] = len(CLASS_NAMES)          # last slot: no H-bond
        np.add.at(out["hb"][k], (cls[live], zb[live]), mult[live])
        np.add.at(out["hb_raw"][k], (cls[live], zb[live]), 1.0)

    px, py, pz = (F[contact, col[c]] for c in ("pose_x", "pose_y", "pose_z"))
    phi = np.degrees(np.arctan2(py, px))
    out["pose"] += np.histogram2d(phi, pz, bins=(grid.n_phi, grid.n_cos),
                                  range=((-180, 180), (-1, 1)),
                                  weights=mult[contact])[0]
    pdb = grid.pose_depth_bin(op[contact])
    out["pose_depth"] += np.histogramdd(
        (pdb, phi, pz), bins=(grid.n_pose_depth, grid.n_phi, grid.n_cos),
        range=((-0.5, grid.n_pose_depth - 0.5), (-180, 180), (-1, 1)),
        weights=mult[contact])[0]
    out["pose_n_raw"] += np.bincount(pdb, minlength=grid.n_pose_depth)
    # Reorientation relative to the contacted lipid: u(t0) . u(t0 + lag) for
    # pairs of frames in contact with the same lipid at both times (contact
    # may break in between), weighted by the frame weight at t0 and binned
    # by its contact depth.
    near = I[:, 0]
    pdb_all = grid.pose_depth_bin(op)
    npd = grid.n_pose_depth
    vecs = [F[:, [col[c] for c in cols]] for _, cols in ROT_VECTORS]
    for k, lag in enumerate(ROT_LAGS):
        n0 = len(F) - lag
        if n0 <= 0:
            break
        ok = contact[:n0] & contact[lag:] & (near[:n0] == near[lag:]) & (near[:n0] >= 0)
        if not ok.any():
            continue
        w, d = mult[:n0][ok], pdb_all[:n0][ok]
        out["rot_n"][:, k] += np.bincount(d, weights=w, minlength=npd)
        out["rot_n_raw"][:, k] += np.bincount(d, minlength=npd)
        for v, u in enumerate(vecs):
            dot = np.einsum("ij,ij->i", u[:n0][ok], u[lag:][ok])
            out["rot_sum"][v, :, k] += np.bincount(d, weights=w * dot, minlength=npd)

    # Where the molecular axis and the chirality vector point in the same
    # lipid frame: (phi in e_x-e_y, cos(theta) = component along e_z).
    for key, c in (("axis_depth", "axis_l"), ("chi_depth", "chi_l")):
        vx, vy, vz = (F[contact, col[c + k]] for k in "xyz")
        out[key] += np.histogramdd(
            (pdb, np.degrees(np.arctan2(vy, vx)), vz),
            bins=(grid.n_pose_depth, grid.n_phi, grid.n_cos),
            range=((-0.5, grid.n_pose_depth - 0.5), (-180, 180), (-1, 1)),
            weights=mult[contact])[0].astype(np.float32)
    ge = np.linspace(-180.0, 180.0, N_GLYC_BINS + 1)
    for a, c in enumerate(("glyc_a", "glyc_b")):
        out["glyc_partner"][a] += np.histogram(F[contact, col[c]], bins=ge,
                                               weights=mult[contact])[0]

    # Lifetimes: continuous runs of the same (acceptor class, residue) per
    # donor, and of contact with the same lipid stereocentre.  The path is one
    # time-ordered sequence, so runs carry straight across segment boundaries;
    # they break only at the path ends and at cut frames.  Each run carries
    # the frame weight of its first frame.
    seqs = []
    for k in range(n_don):
        cls, res = I[:, 1 + 2 * k], I[:, 2 + 2 * k]
        seqs.append((np.where((cls >= 0) & live, cls * 10_000_000 + res, -1), cls))
    near = np.where(contact, I[:, 0], -1)
    seqs.append((near, np.where(contact, len(CLASS_NAMES), -1)))
    for key, kind in seqs:
        vals, lens = _runs(key)
        starts = np.r_[0, np.cumsum(lens)[:-1]]
        kinds, w = kind[starts], mult[starts]
        keep = vals >= 0
        np.add.at(out["life_n"], kinds[keep], w[keep])
        np.add.at(out["life_sum"], kinds[keep], w[keep] * lens[keep])
        h_idx = np.clip(np.digitize(lens[keep], grid.life_edges) - 1,
                        0, len(grid.life_edges) - 2)
        np.add.at(out["life_hist"], (kinds[keep], h_idx), w[keep])
    return out


STEREO_COLS = ("permeant", "enantiomer", "lipids_R", "lipids_S")


def _reduce_entry(unit: Unit, arr, dt, stereo, grid: Grid):
    """((above, below or None), meta): the path reduced in the two parts of
    `frame_parts`, with the same live frames, so runs cross lambda_A."""
    arr, op, dt, meta = assemble_path(unit, arr, dt)
    meta.update(stereo)
    above, below = frame_parts(op, unit)
    live = (above + below) > 0
    red = reduce_path(arr, op, dt, grid, above, live)
    red_below = None
    if below.any():
        red_below = reduce_path(arr, op, dt, grid, below, live)
        for k in RAW_KEYS + ("dt",):
            del red_below[k]
    return (red, red_below), meta


def reduce_unit(unit: Unit, grid: Grid):
    """Reduce one unit straight from its cache (see reduce_units for many)."""
    return _reduce_entry(unit, *load_unit(unit), grid)


_RED: dict = {}


def _init_reduce(grid):
    _RED.update(grid=grid)


def _reduce_chunk_job(item):
    """Reduce every requested unit of one chunk; the chunk is loaded once."""
    path, members = item
    try:
        chunk = load_chunk(path)
    except Exception as exc:
        return [(i, None, None, f"{path}: {type(exc).__name__}: {exc}")
                for i, _, _ in members]
    out = []
    for i, unit, pos in members:
        try:
            red, meta = _reduce_entry(unit, *chunk_entry(chunk, pos), _RED["grid"])
            out.append((i, red, meta, None))
        except Exception as exc:
            out.append((i, None, None, f"{type(exc).__name__}: {exc}"))
    return out


def reduce_units(units, grid, workers=8):
    """
    Reduce all units in parallel, one chunk per job.  Returns (kept units in
    input order, their reductions, a per-unit table of trimming and the CIP
    check).  Results are keyed by position: path numbers repeat across stages.
    Input order is MC order within each stage, which the block bootstrap needs.
    """
    indexes = {d: cache_index(d) for d in {u.cache_dir for u in units}}
    jobs, metas, res = {}, [], {}

    def fail(i, err):
        print(f"    [!] stage {units[i].stage} path {units[i].uid}: {err}")
        metas.append({"stage": units[i].stage, "uid": units[i].uid, "error": err})

    for i, u in enumerate(units):
        try:
            path, pos = _locate(u, indexes[u.cache_dir])
        except LookupError as exc:
            fail(i, str(exc))
            continue
        jobs.setdefault(path, []).append((i, u, pos))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_reduce,
                             initargs=(grid,)) as ex:
        for out in ex.map(_reduce_chunk_job, list(jobs.items())):
            for i, red, meta, err in out:
                if err:
                    fail(i, err)
                else:
                    res[i] = red
                    metas.append(meta)
                    units[i].path_class = meta.get("path_class")
    order = sorted(res)
    return [units[i] for i in order], [res[i] for i in order], pd.DataFrame(metas)


def pose_depth_edges(edges, summaries):
    """
    The pose-map depth edges with every stage's |lambda_A| and |lambda_B|
    inside their range added, so no pose depth bin straddles a stage
    boundary.  Inside one bin, the balance between two stages would then
    hang on the (uncertain) stage factors, and pose changes with depth.
    `summaries` are setup_stages' tables (None for plain-MD sides).
    """
    edges = np.asarray(edges, float)
    lams = [abs(v) for st in summaries if st is not None
            for v in np.r_[st["lam_a"].to_numpy(), st["lam_b"].to_numpy()]]
    inside = [v for v in lams if edges[0] < v < edges[-1]]
    return np.unique(np.round(np.r_[edges, inside], 6))


def rescale_missing(units, kept, attr="weight"):
    """
    Give every stage's reduced units the stage's full weight again.  Units
    that could not be reduced (not cached, trajectories lost) would otherwise
    shrink their stage's histogram against the other stages, which the
    stage factors assume to be complete, and so shift the stitched profile.
    Missing paths are assumed to be lost for reasons unrelated to their
    content, so the remaining paths of the stage stand in for them.
    `attr` names the weight to rescale (weight, or weight_uniform).
    Returns (rescaled copies of `kept`, per-stage table).
    """
    from dataclasses import replace

    rows, scale = [], {}
    for s in sorted({u.stage for u in units}):
        w_all = sum(getattr(u, attr) for u in units if u.stage == s)
        mine = [u for u in kept if u.stage == s]
        w_kept = sum(getattr(u, attr) for u in mine)
        scale[s] = w_all / w_kept if w_kept > 0 else np.nan
        rows.append({"stage": s,
                     "paths_missing": sum(u.stage == s for u in units) - len(mine),
                     "weight_missing_share": 1.0 - w_kept / w_all if w_all > 0 else np.nan,
                     "weight_rescale": scale[s]})
    return ([replace(u, **{attr: getattr(u, attr) * scale[u.stage]}) for u in kept],
            pd.DataFrame(rows))


def _report_stereo(label, meta, want):
    """CIP check per path, so a mislabelled run or inverted lipids is caught
    before any comparison is trusted."""
    if meta.empty or "enantiomer" not in meta:
        return pd.DataFrame()
    st = meta.loc[meta["enantiomer"].isin(["L", "D"]), ["stage", "uid", *STEREO_COLS]]
    if st.empty:
        return st
    bad = st[st["enantiomer"] != want]
    mixed = st[(st["lipids_R"] > 0) & (st["lipids_S"] > 0)]
    print(f"  {label}: {len(st)} paths, permeant {st['enantiomer'].value_counts().to_dict()}, "
          f"lipids R/S = {st['lipids_R'].iloc[0]}/{st['lipids_S'].iloc[0]}")
    if len(bad):
        print(f"  [!] {len(bad)} {label} path(s) contain the OTHER enantiomer")
    if len(mixed):
        print(f"  [!] {len(mixed)} {label} path(s) have lipids of both handedness")
    return st


# ══════════════════════════════════════════════════════════════════════════════
# Weighted bootstrap over units
# ══════════════════════════════════════════════════════════════════════════════

def jsd(p, q):
    """Jensen-Shannon divergence in nats, in [0, ln 2], between two
    histograms of the same shape (normalised here).  Dimensionless and
    bounded, and unlike KS it works for multi-dimensional maps."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    ps, qs = p.sum(), q.sum()
    if ps <= 0 or qs <= 0:
        return np.nan
    p = p / ps
    q = q / qs
    m = 0.5 * (p + q)
    out = 0.0
    for a in (p, q):
        mask = (a > 0) & (m > 0)
        out += 0.5 * np.sum(a[mask] * np.log(a[mask] / m[mask]))
    return float(out)


def block_bootstrap_indices(n, block, rng):
    """
    One circular block-bootstrap replicate over `n` units in MC order.

    Blocks of `block` consecutive units start anywhere in [0, n) and wrap
    around, so every unit is equally likely to be drawn.  (The version this
    replaces drew starts from [0, n - block) and so never drew the last unit;
    when n <= block it returned the original order unchanged, i.e. zero error
    bars.)  The block is capped at n // 2 so a short stage still resamples.
    """
    if n <= 1:
        return np.zeros(n, dtype=int)
    block = max(1, min(block, n // 2))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return idx.ravel()[:n]


def benjamini_hochberg(pvals):
    """BH step-up FDR correction.  Returns q-values in the original order."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def _wsum(w, arr):
    """sum_i w_i arr_i.  A float32 array (hist) is summed in float32: mixing
    in float64 weights would copy the whole array to float64 every call."""
    if arr.dtype == np.float32:
        return np.tensordot(np.asarray(w, np.float32), arr, axes=(0, 0)).astype(float)
    return np.tensordot(w, arr, axes=(0, 0))


class Ensemble:
    """
    Stacked per-unit reductions with weights, grouped by stage and in MC
    order within each stage.

    `A` holds every unit's frames above lambda_A, `B` the [0-] frames below
    it of the stage-0 units that have any (rows `b_rows`); the total is
    A + B / xi.  A replicate is a vector of effective unit weights (see
    `draw`), from which xi is recomputed, so the bootstrap carries the
    uncertainty of xi and, through `factor_model`, of the stage factors.

    `subset(mask)` gives a view restricted to some units (e.g. only the
    reactive paths) that shares these arrays: the other units keep their
    place in the MC chain, so blocks and resampling are those of the whole
    run, but contribute nothing to the sums.  xi is always taken from the
    whole run, since a subset of plus paths has no [0-] paths of its own.
    """

    def __init__(self, units, reds, factor_model=None):
        above = [r[0] for r in reds]
        self.w = np.array([u.weight for u in units], dtype=float)
        self.stage = np.array([u.stage for u in units])
        self.keys = [k for k in above[0] if k != "dt"]
        self.A = {k: np.stack([r[k] for r in above]) for k in self.keys}
        self.b_rows = np.array([i for i, r in enumerate(reds) if r[1] is not None], dtype=int)
        self.b_keys = [k for k in self.keys if k not in RAW_KEYS]
        self.B = ({k: np.stack([reds[i][1][k] for i in self.b_rows]) for k in self.b_keys}
                  if self.b_rows.size else {})
        end = np.array([u.xi_end or "" for u in units])
        self.xi_R, self.xi_L = end == "R", end == "L"
        self.dt = above[0]["dt"]
        self.n = len(units)
        self.factor_model = factor_model
        self.stages = np.unique(self.stage)
        self.mask = None                      # None: every unit counts

    def subset(self, mask):
        """A view on the units where `mask` is True (see the class doc)."""
        import copy

        view = copy.copy(self)
        view.mask = np.asarray(mask, bool) & (True if self.mask is None else self.mask)
        return view

    def restrict(self, a, b):
        """
        A view on units a..b-1 alone (one stage, in practice), sharing the
        arrays.  Unlike `subset`, the other units are gone altogether, so
        resampling, blocks and sums run over this stretch only, which is
        both the right bootstrap for one stage and faster.  xi comes from
        the [0-] paths inside the stretch: for stage 0 those are all of them.
        """
        import copy

        v = copy.copy(self)
        v.w, v.stage = self.w[a:b], self.stage[a:b]
        v.A = {k: arr[a:b] for k, arr in self.A.items()}
        i0, i1 = np.searchsorted(self.b_rows, [a, b])      # b_rows is sorted
        v.b_rows = self.b_rows[i0:i1] - a
        v.B = {k: arr[i0:i1] for k, arr in self.B.items()}
        v.xi_R, v.xi_L = self.xi_R[a:b], self.xi_L[a:b]
        v.n, v.stages = b - a, np.unique(self.stage[a:b])
        v.mask = None if self.mask is None else self.mask[a:b]
        return v

    def with_weights(self, w):
        """The same paths, sharing every array, with other weights (e.g. the
        unweighted ones).  xi follows the new weights."""
        import copy

        v = copy.copy(self)
        v.w = np.asarray(w, dtype=float)
        if v.w.shape != self.w.shape:
            raise ValueError("one weight per unit is needed")
        return v

    def unmasked(self):
        """The same view with every unit counted again (its subset mask
        dropped): per-unit values at their place in the chain."""
        import copy

        v = copy.copy(self)
        v.mask = None
        return v

    def stage_range(self, s):
        """(a, b): the units of stage s are a..b-1 (stages are contiguous)."""
        members = np.flatnonzero(self.stage == s)
        return (int(members[0]), int(members[-1]) + 1) if members.size else (0, 0)

    @property
    def sel(self):
        """Index of the counted units (a slice when all count)."""
        return slice(None) if self.mask is None else self.mask

    @property
    def n_counted(self):
        return self.n if self.mask is None else int(self.mask.sum())

    def raw(self, key):
        """Unweighted sum of a frame count (RAW_KEYS) over the counted units."""
        return self.A[key][self.sel].sum(axis=0)

    def xi(self, we=None):
        """R_END / (R_END + L_END) over the [0-] paths, weighted by `we`
        (default the point weights); 1 when no path carries an end type
        (uniform weights, or no stages)."""
        we = self.w if we is None else we
        r, l = we[self.xi_R].sum(), we[self.xi_L].sum()
        if r + l <= 0:
            return 1.0
        return r / (r + l) if r > 0 else np.nan

    #: Large per-unit arrays most statistics do not need; agg sums them only
    #: when asked for by name (pose_depth is read through _blocks, hist by
    #: histograms' own bootstrap).
    LAZY_KEYS = ("pose_depth", "hist", "axis_depth", "chi_depth")

    def agg(self, we=None, keys=None):
        """Weighted sums over units; `we` from `draw`, default the point
        estimate.  `keys` default: all but LAZY_KEYS."""
        we = self.w if we is None else we
        keys = [k for k in self.keys if k not in self.LAZY_KEYS] if keys is None else keys
        inv_xi = 1.0 / self.xi(we) if self.b_rows.size else 1.0   # whole run
        if self.mask is not None:
            we = we * self.mask
        out = {k: _wsum(we, self.A[k]) for k in keys}
        if self.b_rows.size:
            wb = we[self.b_rows]
            for k in keys:
                if k in self.B:
                    out[k] = out[k] + inv_xi * _wsum(wb, self.B[k])
        return out

    def unit_values(self, key):
        """Per-unit `key` at the point xi, A + B/xi (unweighted), of the
        counted units."""
        a = self.A[key].astype(float, copy=True)
        if self.b_rows.size and key in self.B:
            a[self.b_rows] += self.B[key] / self.xi()
        return a[self.sel]

    def resample(self, rng, block):
        """One block-bootstrap replicate, drawn within each stage separately:
        the stages are independent simulations, and an MC block must not
        straddle two of them.  Each stage keeps its own path count."""
        out = []
        for s in self.stages:
            members = np.flatnonzero(self.stage == s)
            out.append(members[block_bootstrap_indices(len(members), block, rng)])
        return np.concatenate(out)

    def draw(self, rng, block):
        """Effective unit weights of one replicate: how often each unit was
        drawn, times its weight, times its stage's drawn factor ratio."""
        we = np.bincount(self.resample(rng, block), minlength=self.n) * self.w
        if self.factor_model is not None:
            ratio = self.factor_model.draw(rng)
            we = we * ratio[self.stage]
        return we

    def block_starts(self, block):
        """Start index of every contiguous MC block, never crossing a stage."""
        starts = []
        for s in np.unique(self.stage):
            members = np.flatnonzero(self.stage == s)
            starts += list(members[::block])
        return np.array(sorted(starts))


def bootstrap(eL, eD, stat, n_boot=500, block=50, seed=0, keys=None):
    """
    stat(agg) -> 1-D array.  Returns point estimates and replicates for L and D.
    Units are resampled in contiguous MC blocks within each stage, never
    frames: frames within a path are almost perfectly correlated.  Each
    replicate also recomputes xi from its resampled stage-0 paths and draws
    the stage factors from the Pcross errors (Ensemble.draw).  `stat` may be
    one function, or a pair (for L, for D).  `keys`: what agg sums (default
    all but the LAZY_KEYS).
    """
    statL, statD = stat if isinstance(stat, tuple) else (stat, stat)
    rng = np.random.default_rng(seed)
    pL, pD = statL(eL.agg(keys=keys)), statD(eD.agg(keys=keys))
    rL = np.empty((n_boot, pL.size))
    rD = np.empty((n_boot, pD.size))
    for b in range(n_boot):
        rL[b] = statL(eL.agg(eL.draw(rng, block), keys=keys))
        rD[b] = statD(eD.agg(eD.draw(rng, block), keys=keys))
    return pL, pD, rL, rD


def _diff_table(pL, pD, rL, rD, index, ci=0.95):
    # Empty bins are NaN by design; numpy's all-NaN warnings are just noise here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _diff_table_impl(pL, pD, rL, rD, index, ci)


def _diff_table_impl(pL, pD, rL, rD, index, ci):
    from scipy.special import ndtr

    diff = rL - rD
    lo, hi = np.nanpercentile(diff, [50 * (1 - ci), 50 * (1 + ci)], axis=0)
    n = np.sum(np.isfinite(diff), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.minimum(np.sum(diff <= 0, axis=0), np.sum(diff >= 0, axis=0))
        p = 2.0 * k / n
        # No replicate on the other side of zero: the empirical p can only say
        # "below 1/n_boot", and a floor shared by every such result makes the
        # BH q of one strong result depend on how many others hit it too.
        # There the tail is taken as normal, with the bootstrap's own SD.
        z = np.abs(pL - pD) / np.nanstd(diff, axis=0)
        tail = (n > 0) & (k == 0) & np.isfinite(z)
        p = np.where(tail, 2.0 * ndtr(-z), np.clip(p, 1.0 / max(len(diff), 1), 1.0))
    p = np.where(n > 0, p, np.nan)
    df = index.copy()
    df["L"], df["D"], df["L_minus_D"] = pL, pD, pL - pD
    df["ci_lo"], df["ci_hi"], df["p_boot"] = lo, hi, p
    df["p_normal_tail"] = tail
    df["L_se"] = np.nanstd(rL, axis=0)
    df["D_se"] = np.nanstd(rD, axis=0)
    ok = np.isfinite(p)
    df["q_bh"] = np.nan
    df.loc[ok, "q_bh"] = benjamini_hochberg(p[ok])
    return df


def _ratio(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / b, np.nan)


# ══════════════════════════════════════════════════════════════════════════════
# The comparisons
# ══════════════════════════════════════════════════════════════════════════════

def depth_profiles(eL, eD, grid, min_frames=50, **boot):
    """
    WHERE + HOW: every PROFILE_OBS as a depth profile, L vs D.

    `tested` says what ci_lo/ci_hi/p_boot/q_bh are for: L - D, or for the
    MIRROR_NULL_OBS L + D, the departure from the mirror image (`L_minus_D`
    is always L - D).
    """
    stat = lambda a: _ratio(a["S"], a["C"]).ravel()
    pL, pD, rL, rD = bootstrap(eL, eD, stat, **boot)
    names = [o[0] for o in PROFILE_OBS]
    idx = pd.DataFrame([(n, lab) for n in names for lab in grid.depth_labels()],
                       columns=["observable", "depth"])
    # Flipping D's sign makes _diff_table's L - D the L + D of the mirror null.
    sign = np.where(idx["observable"].isin(MIRROR_NULL_OBS), -1.0, 1.0)
    df = _diff_table(pL, sign * pD, rL, sign * rD, idx)
    df.insert(2, "tested", np.where(sign < 0, "L+D", "L-D"))
    df.insert(3, "tested_value", df["L_minus_D"])
    df["D"], df["L_minus_D"] = pD, pL - pD
    # Spread of each observable within a bin (weighted SD), point estimates:
    # a shift in the mean with an unchanged SD reads differently from a change
    # in shape.
    for side, e in (("L", eL), ("D", eD)):
        a = e.agg()
        mean = _ratio(a["S"], a["C"])
        with np.errstate(invalid="ignore"):
            df[f"sd_{side}"] = np.sqrt(np.clip(_ratio(a["S2"], a["C"]) - mean ** 2,
                                               0, None)).ravel()
    df["frames_L"] = eL.raw("C_raw").ravel()
    df["frames_D"] = eD.raw("C_raw").ravel()
    sparse = (df["frames_L"] < min_frames) | (df["frames_D"] < min_frames)
    df.loc[sparse, ["p_boot", "q_bh"]] = np.nan
    return df


def with_hist_limit(grid, stages):
    """`grid` with hist_max_depth set, when unset, to |lambda_A| of the entry
    stage in `stages` (setup_stages' table): the [0-] paths only reach beyond
    it.  Without stages the grid is returned as it is (every finite bin)."""
    if grid.hist_max_depth is not None or stages is None or not len(stages):
        return grid
    from dataclasses import replace

    lam_a = abs(float(stages.sort_values("stage")["lam_a"].iloc[0]))
    print(f"  histograms cover |OP| < {lam_a:g} (|lambda_A| of the entry stage: the [0-] "
          "paths, beyond it, stay out; -hist-max-depth to change)")
    return replace(grid, hist_max_depth=lam_a)


def _hist_rows(grid):
    """Depth bins the histograms cover: finite ones whose centres lie below
    grid.hist_max_depth (the open bulk bin never)."""
    e = np.asarray(grid.depth_edges, float)
    mids = np.r_[0.5 * (e[:-1] + e[1:]), np.inf]
    limit = np.inf if grid.hist_max_depth is None else float(grid.hist_max_depth)
    return mids < limit, mids, limit


def _hist_windows(grid):
    """(labels, membership matrix window x depth bin) of the HIST_WINDOWS,
    cut at grid.hist_max_depth, that hold at least one covered depth bin.
    A window pools the depth bins whose centres fall inside it, and is
    labelled by THEIR edges: with 2.5 A bins a 10-13.5 window holds the
    10-12.5 bin and says so.  Windows that come out the same are kept once."""
    use, mids, limit = _hist_rows(grid)
    e = np.asarray(grid.depth_edges, float)
    labels, rows = [], []
    for a, b in HIST_WINDOWS:
        b = min(b, limit)
        m = use & (mids >= a) & (mids < b)
        idx = np.flatnonzero(m)
        if idx.size:
            lab = f"{e[idx[0]]:.2f}-{e[idx[-1] + 1]:.2f}"
            if lab not in labels:
                labels.append(lab)
                rows.append(m.astype(float))
    return labels, np.array(rows).reshape(len(rows), grid.n_depth)


def histograms(eL, eD, grid, min_frames=200, **boot):
    """
    DISTRIBUTIONS: every HIST_OBS observable as a frame-weighted histogram,
    per depth bin (2D, depth x value) and pooled over the HIST_WINDOWS (1D),
    each normalised to the fraction of that depth's (window's) frames in each
    value bin.  Weights, [0-] scaling, subsets and the block bootstrap are
    those of depth_profiles, so a histogram's mean is the profile's value.

    The 1D histograms are compared bin by bin.  For the MIRROR_NULL_OBS, L is
    compared with D mirrored (value -> -value), the histogram D would have
    if it were the mirror image of L; for the others with D as it is.  The
    bins of one histogram share their frames (they sum to 1), so neighbouring
    tests are not independent: read a run of significant bins as one shift.
    Windows with fewer than `min_frames` frames on either side are not
    tested.  Only depth bins below grid.hist_max_depth are covered (the 2D
    rows beyond it are NaN; windows are cut there).

    Returns (table of the 1D comparison, dict of arrays for histograms.npz).
    """
    wlab, M = _hist_windows(grid)
    nz, nw = grid.n_depth, len(wlab)
    layout = _hist_layout()

    def norm(h):
        with np.errstate(invalid="ignore", divide="ignore"):
            return h / h.sum(axis=-1, keepdims=True)

    def stat(a):
        H = np.asarray(a["hist"], float)
        return np.concatenate([np.r_[norm(H[:, sl]).ravel(), norm(M @ H[:, sl]).ravel()]
                               for _, _, _, sl in layout])

    pL, pD, rL, rD = bootstrap(eL, eD, stat, keys=("hist",), **boot)
    obs_index = {name: o for o, (name, _, _) in enumerate(PROFILE_OBS)}
    cL, cD = eL.raw("C_raw"), eD.raw("C_raw")
    use, _, limit = _hist_rows(grid)
    arrays = {"depth_edges": np.asarray(grid.depth_edges, float),
              "max_depth": np.array(limit), "rows_used": use,
              "windows": np.array(wlab, dtype=str),
              "observables": np.array([n for n, _, _ in HIST_OBS], dtype=str)}
    rows, t_pL, t_pD, t_rL, t_rD, frames = [], [], [], [], [], []
    start = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name, edges, _, sl in layout:
            nb = sl.stop - sl.start
            n2, n1 = nz * nb, nw * nb
            part = lambda x, a, n, shape: x[..., a:a + n].reshape(x.shape[:-1] + shape)
            L2, D2 = part(pL, start, n2, (nz, nb)).copy(), part(pD, start, n2, (nz, nb)).copy()
            L2[~use], D2[~use] = np.nan, np.nan
            L1, D1 = part(pL, start + n2, n1, (nw, nb)), part(pD, start + n2, n1, (nw, nb))
            rL1 = part(rL, start + n2, n1, (nw, nb))
            rD1 = part(rD, start + n2, n1, (nw, nb))
            mirror = name in MIRROR_NULL_OBS
            arrays.update({
                f"{name}_edges": np.asarray(edges, float),
                f"{name}_mirror": np.array(mirror),
                f"L_{name}_2d": L2, f"D_{name}_2d": D2,
                f"Lse_{name}_2d": np.where(use[:, None], np.nanstd(part(rL, start, n2, (nz, nb)), axis=0), np.nan),
                f"Dse_{name}_2d": np.where(use[:, None], np.nanstd(part(rD, start, n2, (nz, nb)), axis=0), np.nan),
                f"L_{name}_1d": L1, f"D_{name}_1d": D1,
                f"Lse_{name}_1d": np.nanstd(rL1, axis=0), f"Dse_{name}_1d": np.nanstd(rD1, axis=0),
                f"nL_{name}_2d": cL[obs_index[name]], f"nD_{name}_2d": cD[obs_index[name]],
                f"nL_{name}_1d": M @ cL[obs_index[name]], f"nD_{name}_1d": M @ cD[obs_index[name]],
            })
            start += n2 + n1
            flip = (lambda x: x[..., ::-1]) if mirror else (lambda x: x)
            t_pL.append(L1.ravel()), t_pD.append(flip(D1).ravel())
            t_rL.append(rL1.reshape(len(rL1), -1)), t_rD.append(flip(rD1).reshape(len(rD1), -1))
            fL, fD = M @ cL[obs_index[name]], M @ cD[obs_index[name]]
            for w, lab in enumerate(wlab):
                for k in range(nb):
                    rows.append((name, lab, edges[k], edges[k + 1],
                                 "L vs mirrored D" if mirror else "L-D"))
                    frames.append((fL[w], fD[w]))
    index = pd.DataFrame(rows, columns=["observable", "window", "value_lo", "value_hi", "tested"])
    df = _diff_table(np.concatenate(t_pL), np.concatenate(t_pD),
                     np.concatenate(t_rL, axis=1), np.concatenate(t_rD, axis=1), index)
    df["frames_L"], df["frames_D"] = np.array(frames).T
    sparse = (df["frames_L"] < min_frames) | (df["frames_D"] < min_frames)
    df.loc[sparse, "p_boot"] = np.nan
    ok = df["p_boot"].notna().to_numpy()
    df["q_bh"] = np.nan
    df.loc[ok, "q_bh"] = benjamini_hochberg(df.loc[ok, "p_boot"].to_numpy())
    for name, _, _, _ in layout:
        q = df.loc[df["observable"] == name, "q_bh"].to_numpy()
        arrays[f"q_{name}_1d"] = q.reshape(nw, -1)
    return df, arrays


def _reference_bin(eL, eD, min_frames):
    """Depth bins both ensembles sample with >= min_frames frames, and the
    outermost of them: the bulk-water anchor for every free energy here."""
    ok = (eL.raw("N_raw") >= min_frames) & (eD.raw("N_raw") >= min_frames)
    return ok, (np.flatnonzero(ok)[-1] if ok.any() else None)


def depth_ddg(eL, eD, grid, temperature=310.0, min_frames=50, **boot):
    """
    WHERE: ddG(depth) = G_L - G_D, each G = -kT ln N(|OP|) of the stitched,
    mirror-summed histogram, anchored at the outermost depth bin both
    ensembles sample.  With the full entry/internal/escape chain that bin is
    bulk water, and G is the free energy relative to bulk, the same profile
    permeation_process integrates.  Negative ddG = L relatively more stable.
    With a single stage G is only the conditional free energy of that run.
    """
    kt = KB_KCAL * temperature
    ok, ref = _reference_bin(eL, eD, min_frames)
    if ref is None:
        print("  [!] no depth bin is sampled by both ensembles; ddG skipped")
        return pd.DataFrame()

    def stat(a):
        with np.errstate(divide="ignore", invalid="ignore"):
            g = -kt * np.log(a["N"] / a["N"][ref])
        g[~ok | ~np.isfinite(g)] = np.nan
        return g

    pL, pD, rL, rD = bootstrap(eL, eD, stat, **boot)
    idx = pd.DataFrame({"depth": grid.depth_labels()})
    df = _diff_table(pL, pD, rL, rD, idx).rename(
        columns={"L": "G_L", "D": "G_D", "L_minus_D": "ddG_kcal"})
    df["reference_bin"] = grid.depth_labels()[ref]
    return df


def permeability_ratio(eL, eD, grid, min_frames=50, n_boot=500, block=50, seed=0):
    """
    WHERE, as one number comparable with the experiment: ln(P_L / P_D).

    Inhomogeneous solubility-diffusion (the model permeation_process
    integrates): 1/P = int exp(beta G(z)) / D(z) dz.  With D(z) taken EQUAL
    for the two enantiomers, D cancels from the ratio and

        ln(P_L / P_D) = ln R_D - ln R_L,    R = sum_bins width * exp(beta G)

    with G anchored at bulk water, as in depth_ddg.  This isolates the
    thermodynamic part of the selectivity.  A difference in D(z), e.g. slower
    diffusion of one enantiomer through the headgroups, is NOT in it; for
    that, run permeation_process on each enantiomer with its own
    diffusion_aggregate.csv.  The experimental value for Pro-yne in DOPC/POPC
    is ln(2.1/0.36) = 1.76 (Hu et al. 2021, SI Fig. 9).

    Also returns each depth bin's share of each enantiomer's resistance, and
    the difference, which says where the selectivity is decided.  If a
    membrane bin inside the reference bin is unsampled the integral misses
    part of the barrier; the table flags that as `complete = False`.
    """
    ok, ref = _reference_bin(eL, eD, min_frames)
    if ref is None:
        return pd.DataFrame(), pd.DataFrame()
    width = np.r_[np.diff(grid.depth_edges), np.nan]        # last bin is open
    use = ok & np.isfinite(width)
    complete = bool(np.all(ok[:ref]))

    def stat(a):
        with np.errstate(divide="ignore", invalid="ignore"):
            boltz = np.where(use, width * a["N"][ref] / a["N"], 0.0)  # width*exp(beta G)
        boltz[~np.isfinite(boltz)] = 0.0
        total = boltz.sum()
        share = np.where(use, boltz / total, np.nan) if total > 0 else np.full_like(boltz, np.nan)
        return np.r_[np.log(total) if total > 0 else np.nan, share]

    pL, pD, rL, rD = bootstrap(eL, eD, stat, n_boot=n_boot, block=block, seed=seed)
    d = -(rL[:, 0] - rD[:, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
    summary = pd.DataFrame([{
        "ln_PL_over_PD": -(pL[0] - pD[0]), "ci_lo": lo, "ci_hi": hi,
        "PL_over_PD": np.exp(-(pL[0] - pD[0])),
        "ln_resistance_L": pL[0], "ln_resistance_D": pD[0],
        "reference_bin": grid.depth_labels()[ref], "complete": complete,
        "assumes": "equal D(z) for L and D"}])
    shares = _diff_table(pL[1:], pD[1:], rL[:, 1:], rD[:, 1:],
                         pd.DataFrame({"depth": grid.depth_labels()}))
    shares = shares.rename(columns={"L": "share_L", "D": "share_D",
                                    "L_minus_D": "share_L_minus_D"})
    return summary, shares


def state_resolved_ddg(eL, eD, grid, temperature=310.0, min_frames=50, **boot):
    """
    WHAT carries the free-energy difference: ddG(depth, H-bond state).

    For donor k in H-bond state c (an acceptor class, water, or none),
    G_kc(z) = -kT ln N_kc(z) / N(ref), anchored at bulk like depth_ddg.  Summed
    over c, exp(-beta G_kc) gives exp(-beta G): the states partition the
    total.  So if L's advantage at the interface lives in, say, the sn-2
    carbonyl-bound state, ddG is negative there and ~0 for the other states.
    Donors are in PermeantSpec.donors order.
    """
    kt = KB_KCAL * temperature
    ok, ref = _reference_bin(eL, eD, min_frames)
    if ref is None:
        return pd.DataFrame()
    dense = ((eL.raw("hb_raw") >= min_frames)
             & (eD.raw("hb_raw") >= min_frames) & ok[None, None, :])

    def stat(a):
        with np.errstate(divide="ignore", invalid="ignore"):
            g = -kt * np.log(a["hb"] / a["N"][ref])
        g[~dense | ~np.isfinite(g)] = np.nan
        return g.ravel()

    pL, pD, rL, rD = bootstrap(eL, eD, stat, **boot)
    n_don = eL.A["hb"].shape[1]
    rows = [(k, c, z) for k in range(n_don) for c in CLASS_NAMES + ["none"]
            for z in grid.depth_labels()]
    df = _diff_table(pL, pD, rL, rD,
                     pd.DataFrame(rows, columns=["donor", "state", "depth"]))
    df = df.rename(columns={"L": "G_L", "D": "G_D", "L_minus_D": "ddG_kcal"})
    df["reference_bin"] = grid.depth_labels()[ref]
    return df.dropna(subset=["G_L", "G_D"], how="all")


def hbond_partners(eL, eD, grid, **boot):
    """WHAT: per donor, the fraction of frames H-bonded to each acceptor class,
    per depth bin and pooled over all depths."""
    def stat(a):
        hb = a["hb"]                                   # (don, cls+1, z)
        per_z = _ratio(hb, hb.sum(axis=1, keepdims=True))
        pooled = _ratio(hb.sum(axis=2), hb.sum(axis=(1, 2))[:, None])
        return np.concatenate([per_z.ravel(), pooled.ravel()])

    pL, pD, rL, rD = bootstrap(eL, eD, stat, **boot)
    n_don = eL.A["hb"].shape[1]
    cls = CLASS_NAMES + ["none"]
    rows = [(k, c, z) for k in range(n_don) for c in cls for z in grid.depth_labels()]
    rows += [(k, c, "all") for k in range(n_don) for c in cls]
    idx = pd.DataFrame(rows, columns=["donor", "partner", "depth"])
    return _diff_table(pL, pD, rL, rD, idx)


def sn2_preference(eL, eD, **boot):
    """
    WHAT, reduced to the single number the hypothesis is most directly about:
    of all H-bonds to glycerol-ester oxygens, the fraction on the sn-2 branch
    (O21/O22), which hangs off the stereocentre itself.  Reported per donor
    and for all donors together.
    """
    sn2 = [CLASS_NAMES.index(c) for c in ("CO_sn2", "OE_sn2")]
    sn1 = [CLASS_NAMES.index(c) for c in ("CO_sn1", "OE_sn1")]

    def stat(a):
        hb = a["hb"].sum(axis=2)                       # (don, cls+1)
        s2, s1 = hb[:, sn2].sum(axis=1), hb[:, sn1].sum(axis=1)
        return np.r_[_ratio(s2, s2 + s1), _ratio(s2.sum(), s2.sum() + s1.sum())]

    pL, pD, rL, rD = bootstrap(eL, eD, stat, **boot)
    n_don = eL.A["hb"].shape[1]
    idx = pd.DataFrame({"donor": [str(k) for k in range(n_don)] + ["all"]})
    return _diff_table(pL, pD, rL, rD, idx)


def lifetimes(eL, eD, **boot):
    """HOW LONG: mean continuous-run length (ps) per H-bond partner class and
    for stereocentre contact with the same lipid."""
    dtL, dtD = eL.dt, eD.dt
    if not np.isclose(dtL, dtD):
        print(f"  [!] frame spacing differs (L {dtL} ps, D {dtD} ps); lifetimes "
              f"are converted with each ensemble's own dt")
    stat_L = lambda a: _ratio(a["life_sum"], a["life_n"]) * dtL
    stat_D = lambda a: _ratio(a["life_sum"], a["life_n"]) * dtD
    pL, pD, rL, rD = bootstrap(eL, eD, (stat_L, stat_D), **boot)
    df = _diff_table(pL, pD, rL, rD, pd.DataFrame({"kind": LIFE_KINDS}))
    df = df.rename(columns={"L": "tau_L_ps", "D": "tau_D_ps", "L_minus_D": "dtau_ps"})
    df["runs_L"] = eL.unit_values("life_n").sum(axis=0)
    df["runs_D"] = eD.unit_values("life_n").sum(axis=0)
    return df


def lifetime_survival(eL, eD, grid, **boot):
    """
    HOW LONG, beyond the mean: survival S(tau) = fraction of runs lasting at
    least tau, per partner class and for lipid-stereocentre contact.  Two
    enantiomers with the same mean lifetime can still differ in the long-lived
    tail, and the tail (a subpopulation trapped at the interface) is what a
    trapping mechanism would change.  tau is in frames on the grid's
    logarithmic life_edges, converted with L's frame spacing.
    """
    def stat(a):
        h = a["life_hist"]                                  # (kind, len bin)
        tail = np.cumsum(h[:, ::-1], axis=1)[:, ::-1]       # runs with length >= edge
        return _ratio(tail, h.sum(axis=1, keepdims=True)).ravel()

    pL, pD, rL, rD = bootstrap(eL, eD, stat, **boot)
    edges = grid.life_edges[:-1]
    rows = [(k, int(t), t * eL.dt) for k in LIFE_KINDS for t in edges]
    df = _diff_table(pL, pD, rL, rD,
                     pd.DataFrame(rows, columns=["kind", "tau_frames", "tau_ps"]))
    return df.rename(columns={"L": "S_L", "D": "S_D", "L_minus_D": "S_L_minus_D"})


def _blocks(e, key, block, drop_empty=True):
    """Weighted `key` summed over contiguous MC blocks of units, with weights
    rescaled to mean 1 so two ensembles with differently normalised infretis
    weights pool on equal footing.  drop_empty=False keeps every block, so
    the blocks of two keys stay aligned."""
    starts = e.block_starts(block)
    ends = np.r_[starts[1:], e.n]
    w = e.w if e.mask is None else e.w * e.mask
    w = w / w[e.sel].mean()
    A = e.A[key]
    B = e.B.get(key)
    b_pos = np.full(e.n, -1)
    b_pos[e.b_rows] = np.arange(e.b_rows.size)
    inv_xi = 1.0 / e.xi()
    out = np.empty((len(starts),) + A.shape[1:])
    # One block at a time, so no weighted copy of the whole per-unit array.
    for i, (a, b) in enumerate(zip(starts, ends)):
        out[i] = np.tensordot(w[a:b], A[a:b], axes=(0, 0))
        if B is not None:
            rows = b_pos[a:b]
            m = rows >= 0
            if m.any():
                out[i] += inv_xi * np.tensordot(w[a:b][m], B[rows[m]], axes=(0, 0))
    # Blocks whose units are all outside a subset carry nothing: leave them
    # out, so they do not pad the permutation pool.
    return out[out.reshape(len(out), -1).any(axis=1)] if drop_empty else out


def permutation_jsd(eL, eD, key, block=50, n_perm=1000, seed=0):
    """
    JSD between the L and D distributions stored under `key`, with a p-value
    from a block permutation test: under the null L and D are exchangeable, so
    contiguous MC blocks of units are pooled and randomly relabelled.  Blocks,
    not single units, keep the path-to-path MC correlation inside the null.
    (A bootstrap CI on JSD is not used: resampling noise biases JSD upward.)
    """
    rng = np.random.default_rng(seed)
    bL, bD = _blocks(eL, key, block), _blocks(eD, key, block)
    obs = jsd(bL.sum(axis=0).ravel(), bD.sum(axis=0).ravel())
    pool = np.concatenate([bL, bD])
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(pool))
        null[i] = jsd(pool[perm[:len(bL)]].sum(axis=0).ravel(),
                      pool[perm[len(bL):]].sum(axis=0).ravel())
    return {"jsd": obs, "null_mean": null.mean(), "null_95": np.percentile(null, 95),
            "p_perm": (1 + np.sum(null >= obs)) / (1 + n_perm),
            "blocks_L": len(bL), "blocks_D": len(bD)}


def pose_maps(eL, eD, block=50, n_perm=1000, seed=0):
    """
    HOW: the permeant's position around the nearest lipid stereocentre, in the
    lipid's glycerol body frame (phi in the ex-ey plane, cos(theta) = pose_z),
    in frames where the two are in contact.  Returns normalised maps and a
    block-permutation test of JSD(L, D).
    """
    mL, mD = eL.agg()["pose"], eD.agg()["pose"]
    out = {"L": mL / mL.sum(), "D": mD / mD.sum()}
    out["L_minus_D"] = out["L"] - out["D"]
    test = permutation_jsd(eL, eD, "pose", block, n_perm, seed)
    return out, pd.DataFrame([{"map": "pose", **test}])


def _matched(L, D, use):
    """
    Depth-matched maps from per-depth weighted sums L, D (depth, phi, cos):
    each depth's map normalised on its own, then averaged with ONE set of
    depth weights w, the mean of L's and D's contact-depth distributions
    over the depths in `use`.  A difference between the two results can then
    not come from L and D contacting the lipids at different depths.
    """
    mL, mD = L.sum(axis=(1, 2)), D.sum(axis=(1, 2))
    ok = use & (mL > 0) & (mD > 0)
    w = np.zeros(len(mL))
    if not ok.any():
        return np.full(L.shape[1:], np.nan), np.full(L.shape[1:], np.nan), w
    w[ok] = 0.5 * (mL[ok] / mL[ok].sum() + mD[ok] / mD[ok].sum())
    pL = np.tensordot(w[ok], L[ok] / mL[ok, None, None], axes=(0, 0))
    pD = np.tensordot(w[ok], D[ok] / mD[ok, None, None], axes=(0, 0))
    return pL, pD, w


def pose_maps_depth(eL, eD, grid, block=50, n_perm=1000, seed=0, min_frames=1000,
                    key="pose_depth", flip_d=None):
    """
    HOW, at matched depth: the pose maps of `pose_maps`, but resolved into
    grid.pose_depth_edges bins of |OP|.  The pooled map mixes depths, and
    pose_z changes with depth, so L and D contacting the lipids at somewhat
    different depths would by itself make their pooled maps differ.  Here

      depth_matched  both maps are built with the same depth weights (see
                     `_matched`) over the depths both sample with
                     >= min_frames contact frames (a map has 288 bins, so
                     fewer than ~1000 frames is mostly noise)
      depth <bin>    L vs D within each such depth bin (BH q over these)
      pooled         all depths mixed as they come, as `pose_maps` does

    each with a block-permutation p-value; one set of permutations serves
    every row.  Returns (maps for pose_maps_depth.npz, table).

    `key` names any (depth, phi, cos) map of the contact frames (the
    orientation maps use it too); `flip_d` transforms D's maps first, so L is
    compared with, and permuted against, that image of D (see
    orientation_maps).
    """
    rng = np.random.default_rng(seed)
    bL, bD = _blocks(eL, key, block), _blocks(eD, key, block)
    if flip_d is not None:
        bD = flip_d(bD)
    nL = len(bL)
    fL = eL.raw("pose_n_raw")
    fD = eD.raw("pose_n_raw")
    use = (fL >= min_frames) & (fD >= min_frames)
    depths = np.flatnonzero(use)

    def stats(L, D):
        pL, pD, _ = _matched(L, D, use)
        return np.r_[jsd(L.sum(axis=0), D.sum(axis=0)), jsd(pL, pD),
                     [jsd(L[d], D[d]) for d in depths]]

    L, D = bL.sum(axis=0), bD.sum(axis=0)
    obs = stats(L, D)
    pool = np.concatenate([bL, bD])
    null = np.empty((n_perm, obs.size))
    for i in range(n_perm):
        perm = rng.permutation(len(pool))
        null[i] = stats(pool[perm[:nL]].sum(axis=0), pool[perm[nL:]].sum(axis=0))

    labels = grid.pose_depth_labels()
    pL, pD, w = _matched(L, D, use)
    mL, mD = L.sum(axis=(1, 2)), D.sum(axis=(1, 2))
    rows = [("pooled", "all"), ("depth_matched", "matched")] + \
           [("depth", labels[d]) for d in depths]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df = pd.DataFrame(rows, columns=["map", "depth"])
        df["jsd"] = obs
        df["null_mean"] = np.nanmean(null, axis=0)
        df["null_95"] = np.nanpercentile(null, 95, axis=0)
        df["p_perm"] = (1 + np.sum(null >= obs, axis=0)) / (1 + n_perm)
    per = df["map"] == "depth"
    df["q_bh"] = np.nan
    if per.any():
        df.loc[per, "q_bh"] = benjamini_hochberg(df.loc[per, "p_perm"].to_numpy())
    sel = {"pooled": slice(None), "depth_matched": use}
    df["frames_L"] = [fL[sel[m]].sum() if m in sel else fL[labels.index(z)]
                      for m, z in rows]
    df["frames_D"] = [fD[sel[m]].sum() if m in sel else fD[labels.index(z)]
                      for m, z in rows]
    df["contact_share_L"] = [np.nan if m in sel else mL[labels.index(z)] / mL.sum()
                             for m, z in rows]
    df["contact_share_D"] = [np.nan if m in sel else mD[labels.index(z)] / mD.sum()
                             for m, z in rows]
    df["match_weight"] = [np.nan if m in sel else w[labels.index(z)] for m, z in rows]
    df["blocks_L"], df["blocks_D"] = nL, len(bD)

    with np.errstate(invalid="ignore", divide="ignore"):
        maps = {"L_depth": L / mL[:, None, None], "D_depth": D / mD[:, None, None],
                "L_matched": pL, "D_matched": pD, "L_minus_D_matched": pL - pD,
                "match_weight": w, "contact_share_L": mL / mL.sum(),
                "contact_share_D": mD / mD.sum(), "frames_L": fL, "frames_D": fD,
                "tested": use, "depth_edges": grid.pose_depth_edges,
                "phi_edges": np.linspace(-180, 180, grid.n_phi + 1),
                "cos_edges": np.linspace(-1, 1, grid.n_cos + 1)}
    return maps, df


def _invert(maps):
    """(..., phi, cos) maps of a vector v as the maps of -v: phi + 180 deg,
    cos -> -cos.  Exact on the grid: phi spans -180..180 in an even number
    of bins, cos -1..1."""
    return np.roll(maps[..., ::-1], maps.shape[-2] // 2, axis=-2)


#: The orientation maps: (name, per-unit key, transform of D's maps).  The
#: molecular axis is an ordinary vector: without chiral recognition its
#: distribution in the (R) lipid's frame is the same for L and D.  The
#: chirality vector is a pseudovector: its component along the membrane
#: normal is opposite for L and D, so without recognition D's map is L's for
#: -n (phi + 180, cos -> -cos), the map analogue of testing L + D for chi_lz.
ORIENT_MAPS = (("axis", "axis_depth", None), ("chirality", "chi_depth", _invert))


def orientation_maps(eL, eD, grid, block=50, n_perm=1000, seed=0, min_frames=1000):
    """
    HOW, as directions: where the molecular axis and the chirality vector
    point in the glycerol frame of the contacted lipid (phi in its e_x-e_y
    plane, cos(theta) along e_z), contact frames only.  cos(theta) alone is
    axis_lz / chi_lz; phi is the side the vector leans to, which only a
    chiral frame can tell apart.  Each map is tested as the pose maps are
    (pose_maps_depth: depth-matched, per depth, pooled), the chirality map
    against D inverted (ORIENT_MAPS).
    Returns (table, maps for orient_maps_depth.npz, keys prefixed axis_ /
    chirality_).
    """
    tables, maps = [], {}
    for name, key, flip in ORIENT_MAPS:
        m, df = pose_maps_depth(eL, eD, grid, block=block, n_perm=n_perm, seed=seed,
                                min_frames=min_frames, key=key, flip_d=flip)
        df.insert(0, "vector", name)
        df.insert(1, "tested", "L vs D" if flip is None else "L vs D inverted (-n)")
        tables.append(df)
        maps.update({f"{name}_{k}": v for k, v in m.items()})
    return pd.concat(tables, ignore_index=True), maps


def reorientation(eL, eD, grid, min_pairs=500, **boot):
    """
    HOW FAST: how quickly the molecular axis, the chirality vector and the
    permeant's position (the direction from the lipid C2) turn relative to
    the lipid it touches.  C(t) = <u(0) . u(t)> over frame pairs in contact
    with the same lipid at both times (ROT_LAGS), per contact depth bin and
    over all of them.  Reported per vector and depth: the correlation time
    tau = integral of C from 0 to the longest lag (a lower bound when C has
    not decayed by then), and C at ROT_REPORT_LAGS.  A true scalar for every
    vector (a pseudovector dotted with itself), so L - D is the test, with
    one BH family over the whole table.  Depth bins with fewer than
    `min_pairs` pairs at 12 frames are left out.
    Returns (table, curves for reorientation.npz).
    """
    names = [n for n, _ in ROT_VECTORS]
    labels = grid.pose_depth_labels() + ["all contact"]
    t_ps = np.r_[0.0, ROT_LAGS * eL.dt]

    def curves(a):
        with np.errstate(invalid="ignore", divide="ignore"):
            per = a["rot_sum"] / a["rot_n"][None]                       # (v, depth, lag)
            pooled = a["rot_sum"].sum(1) / a["rot_n"].sum(0)[None]      # (v, lag)
        return np.concatenate([per, pooled[:, None, :]], axis=1)        # (v, depth+1, lag)

    def tau(c):
        full = np.concatenate([np.ones(c.shape[:-1] + (1,)), c], axis=-1)
        ok = np.isfinite(full).all(axis=-1)
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        out = trapezoid(np.nan_to_num(full), t_ps, axis=-1)
        return np.where(ok, out, np.nan)

    def stat(a):
        c = curves(a)
        return np.r_[tau(c).ravel(), c.ravel()]

    nv, nd, nl = len(names), len(labels), len(ROT_LAGS)
    pL, pD, rL, rD = bootstrap(eL, eD, stat, keys=("rot_sum", "rot_n"), **boot)
    # enough pairs: at the lag of 2.4 ps (12 frames) in the depth bin, and overall
    k12 = int(np.searchsorted(ROT_LAGS, 12))
    nL, nD = eL.raw("rot_n_raw"), eD.raw("rot_n_raw")
    pairs_L = np.r_[nL[:, k12], nL[:, k12].sum()]
    pairs_D = np.r_[nD[:, k12], nD[:, k12].sum()]
    dense = (pairs_L >= min_pairs) & (pairs_D >= min_pairs)

    n_tau = nv * nd
    # the reported quantities: tau, then C at each ROT_REPORT_LAGS lag
    ks = [int(np.searchsorted(ROT_LAGS, lag)) for lag in ROT_REPORT_LAGS]
    cols = [np.arange(n_tau)]
    for k in ks:
        cols.append(n_tau + (np.arange(nv)[:, None] * nd + np.arange(nd)[None, :]).ravel() * nl + k)
    sel = np.concatenate(cols)
    quantity = (["tau_ps"] * n_tau
                + [f"C({ROT_LAGS[k] * eL.dt:g} ps)" for k in ks for _ in range(n_tau)])
    idx = pd.DataFrame({"quantity": quantity,
                        "vector": [v for _ in range(1 + len(ks)) for v in names for _ in labels],
                        "depth": [d for _ in range(1 + len(ks)) for _ in names for d in labels]})
    df = _diff_table(pL[sel], pD[sel], rL[:, sel], rD[:, sel], idx)
    df["pairs_L"] = np.tile(pairs_L, nv * (1 + len(ks)))
    df["pairs_D"] = np.tile(pairs_D, nv * (1 + len(ks)))
    sparse = ~np.tile(dense, nv * (1 + len(ks)))
    df.loc[sparse, ["p_boot"]] = np.nan
    ok = df["p_boot"].notna().to_numpy()
    df["q_bh"] = np.nan
    df.loc[ok, "q_bh"] = benjamini_hochberg(df.loc[ok, "p_boot"].to_numpy())
    shape = (nv, nd, nl)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        maps = {"vectors": np.array(names, dtype=str), "depths": np.array(labels, dtype=str),
                "lag_ps": ROT_LAGS * eL.dt, "dense": dense,
                "C_L": pL[n_tau:].reshape(shape), "C_D": pD[n_tau:].reshape(shape),
                "C_Lse": np.nanstd(rL[:, n_tau:], axis=0).reshape(shape),
                "C_Dse": np.nanstd(rD[:, n_tau:], axis=0).reshape(shape),
                "tau_q": df["q_bh"].to_numpy()[:n_tau].reshape(nv, nd)}
    return df, maps


#: The orientation observables whose free-energy surface over depth x
#: cos(theta) orientation_free_energy reports (measured against the membrane
#: normal, every frame counts).
ORIENT_FE_OBS = ("cos_axis_n", "cos_chi_n")


def orientation_free_energy(eL, eD, grid, temperature=310.0, min_frames=1000, **boot):
    """
    WHERE x HOW: the free energy of each depth AND orientation,
        G(z, c) = -kT ln( n_c N(z, c) / N(ref) ),
    N(z, c) the weighted frame count in depth bin z and cos(theta) bin c
    (the HIST_OBS histogram), n_c the number of cos bins, N(ref) the bulk
    anchor of depth_ddg.  n_c makes an orientation with no preference come
    out at G(z) (cos(theta) bins are equal areas on the sphere), so
        G(z, c) = G(z) - kT ln( n_c p(c | z) ).
    ddG(z, c) = G_L - G_D says whether an orientation is cheaper for L or for
    D at that depth; for cos_chi_n, D is mirrored (c -> -c) first, as the
    histograms do.  Depth rows follow grid.hist_max_depth; rows with fewer
    than `min_frames` frames, and empty cells, are left out.
    Returns (table, arrays for orientation_free_energy.npz).
    """
    kt = KB_KCAL * temperature
    ok, ref = _reference_bin(eL, eD, min_frames=50)
    if ref is None:
        return pd.DataFrame(), {}
    use, _, limit = _hist_rows(grid)
    lay = {name: (edges, sl) for name, edges, _, sl in _hist_layout()}
    cL, cD = eL.raw("C_raw"), eD.raw("C_raw")
    names = [o[0] for o in PROFILE_OBS]

    def stat(a):
        H = np.asarray(a["hist"], float)
        out = []
        for name in ORIENT_FE_OBS:
            edges, sl = lay[name]
            with np.errstate(divide="ignore", invalid="ignore"):
                g = -kt * np.log((sl.stop - sl.start) * H[:, sl] / a["N"][ref])
            g[~np.isfinite(g)] = np.nan
            out.append(g.ravel())
        return np.concatenate(out)

    pL, pD, rL, rD = bootstrap(eL, eD, stat, keys=("hist", "N"), **boot)
    rows, tL, tD, trL, trD, keep = [], [], [], [], [], []
    arrays = {"depth_edges": np.asarray(grid.depth_edges, float), "rows_used": use,
              "max_depth": np.array(limit), "temperature": np.array(temperature)}
    start, nz, labels = 0, grid.n_depth, grid.depth_labels()
    for name in ORIENT_FE_OBS:
        edges, sl = lay[name]
        nb = sl.stop - sl.start
        mirror = name in MIRROR_NULL_OBS
        flip = (lambda x: x[..., ::-1]) if mirror else (lambda x: x)
        part = lambda x: x[..., start:start + nz * nb].reshape(x.shape[:-1] + (nz, nb))
        gL, gD = part(pL), part(pD)
        rows_ok = use & ok & (cL[names.index(name)] >= min_frames) & (cD[names.index(name)] >= min_frames)
        arrays.update({f"{name}_edges": np.asarray(edges, float), f"{name}_mirror": np.array(mirror),
                       f"G_L_{name}": np.where(rows_ok[:, None], gL, np.nan),
                       f"G_D_{name}": np.where(rows_ok[:, None], gD, np.nan)})
        for z in np.flatnonzero(rows_ok):
            for c in range(nb):
                rows.append((name, labels[z], edges[c], edges[c + 1],
                             "L vs mirrored D" if mirror else "L-D"))
            tL.append(gL[z]), tD.append(flip(gD[z]))
            trL.append(part(rL)[:, z]), trD.append(flip(part(rD)[:, z]))
        start += nz * nb
    if not rows:
        return pd.DataFrame(), arrays
    idx = pd.DataFrame(rows, columns=["observable", "depth", "value_lo", "value_hi", "tested"])
    df = _diff_table(np.concatenate(tL), np.concatenate(tD),
                     np.concatenate(trL, axis=1), np.concatenate(trD, axis=1), idx)
    df = df.rename(columns={"L": "G_L", "D": "G_D", "L_minus_D": "ddG_kcal"})
    df["reference_bin"] = labels[ref]
    for name in ORIENT_FE_OBS:
        sub = df[df["observable"] == name]
        nb = len(arrays[f"{name}_edges"]) - 1
        dd, qq = np.full((nz, nb), np.nan), np.full((nz, nb), np.nan)
        for z, g in sub.groupby("depth", sort=False):
            dd[labels.index(z)] = g["ddG_kcal"].to_numpy()
            qq[labels.index(z)] = g["q_bh"].to_numpy()
        arrays[f"ddG_{name}"], arrays[f"q_{name}"] = dd, qq
    return df, arrays


def barrier_depths(shares, cover=0.5):
    """The depth bins that carry the resistance to crossing: in order of
    their mean L/D share of 1/P (permeability_ratio's resistance_shares),
    the fewest bins that together hold at least `cover` of it."""
    if shares is None or shares.empty:
        return []
    mean = 0.5 * (shares["share_L"] + shares["share_D"])
    order = mean.sort_values(ascending=False).dropna()
    cum = order.cumsum()
    n = int(np.searchsorted(cum.to_numpy(), cover) + 1)
    return list(shares.loc[order.index[:n], "depth"])


def barrier_comparison(eL, eD, grid, shares, crossing=None, cover=0.5, **boot):
    """
    WHERE IT MATTERS: every PROFILE_OBS pooled over the depth bins that
    carry the resistance to crossing (barrier_depths), L vs D (L + D for the
    MIRROR_NULL_OBS), for all paths and, given `crossing` = (mask L, mask D)
    of the reactive units, for the crossing paths alone.  One small family of
    tests, so a difference at the barrier has more power than it has among
    the 150 cells of the depth profiles.
    """
    depths = barrier_depths(shares, cover)
    if not depths:
        return pd.DataFrame()
    labels = grid.depth_labels()
    sel = np.zeros(grid.n_depth, bool)
    sel[[labels.index(d) for d in depths]] = True
    covered = float(0.5 * (shares["share_L"] + shares["share_D"])[shares["depth"].isin(depths)].sum())
    names = [o[0] for o in PROFILE_OBS]
    sign = np.array([-1.0 if n in MIRROR_NULL_OBS else 1.0 for n in names])
    stat = lambda a: _ratio(a["S"][:, sel].sum(1), a["C"][:, sel].sum(1))
    sets = [("all paths", eL, eD)]
    if crossing is not None:
        sets.append(("crossing paths", eL.subset(crossing[0]), eD.subset(crossing[1])))
    tables = []
    for label, a, b in sets:
        if not a.n_counted or not b.n_counted:
            continue
        pL, pD, rL, rD = bootstrap(a, b, stat, **boot)
        idx = pd.DataFrame({"paths": label, "observable": names,
                            "tested": np.where(sign < 0, "L+D", "L-D")})
        t = _diff_table(pL, sign * pD, rL, sign * rD, idx)
        t.insert(3, "tested_value", t["L_minus_D"])
        t["D"], t["L_minus_D"] = pD, pL - pD
        t["frames_L"] = a.raw("C_raw")[:, sel].sum(1)
        t["frames_D"] = b.raw("C_raw")[:, sel].sum(1)
        tables.append(t)
    df = pd.concat(tables, ignore_index=True)
    ok = df["p_boot"].notna().to_numpy()
    df["q_bh"] = np.nan
    df.loc[ok, "q_bh"] = benjamini_hochberg(df.loc[ok, "p_boot"].to_numpy())
    df["barrier_depths"] = ", ".join(depths)
    df["resistance_covered"] = covered
    return df


def halves(e):
    """Masks of the first and second half of every stage, in MC order."""
    first = np.zeros(e.n, bool)
    for s in e.stages:
        members = np.flatnonzero(e.stage == s)
        first[members[: len(members) // 2]] = True
    return first, ~first


def split_half_profiles(eL, eD, grid, block=50, n_perm=200, n_boot=200, seed=0,
                        min_frames=50):
    """
    Do the L - D depth profiles replicate?  Every PROFILE_OBS profile is
    computed separately from the first and the second half of every stage
    (MC order), with a block-bootstrap standard error per half.  For the
    MIRROR_NULL_OBS the tested quantity is L + D, as in depth_profiles.

    Returns (rows, summary):
      rows     per observable and depth: the tested difference in each half,
               its SE, and whether both halves agree in sign
      summary  per observable: the correlation r of the two halves'
               difference profiles over the depth bins both sample, with a
               null from relabelling MC blocks L/D within each half
               (null_95, p_perm; for the MIRROR_NULL_OBS the D blocks are
               mirrored before relabelling, so the null is "D is the mirror
               image of L"), and how many bins agree in sign
    A single depth bin replicates when both halves give the same sign and
    neither interval covers zero; a whole profile when r beats its null.

    The null is conservative when a real difference exists: relabelled
    blocks still carry part of it, so shuffled halves can share its shape
    (with a random size and sign) and null_95 comes out high.  Its false
    alarm rate is right when L and D do not differ, but a replicating
    profile may still fail it; the per-bin both_clear_of_zero count is the
    more sensitive check.
    """
    rng = np.random.default_rng(seed)
    names = [o[0] for o in PROFILE_OBS]
    nz = grid.n_depth
    sign = np.repeat(np.array([-1.0 if n in MIRROR_NULL_OBS else 1.0 for n in names]), nz)
    labels = grid.depth_labels()
    hL, hD = halves(eL), halves(eD)
    halfL = [eL.subset(m) for m in hL]
    halfD = [eD.subset(m) for m in hD]
    ok = np.ones(len(names) * nz, bool)
    for e in halfL + halfD:
        ok &= e.raw("C_raw").ravel() >= min_frames

    stat = lambda a: _ratio(a["S"], a["C"]).ravel()
    diff, se = [], []
    for h in (0, 1):
        pL, pD, rL, rD = bootstrap(halfL[h], halfD[h], stat, n_boot=n_boot,
                                   block=block, seed=seed + h)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            diff.append(pL - sign * pD)
            se.append(np.nanstd(rL - sign * rD, axis=0))

    def corrs(d1, d2):
        out = []
        for o in range(len(names)):
            sl = slice(o * nz, (o + 1) * nz)
            m = ok[sl] & np.isfinite(d1[sl]) & np.isfinite(d2[sl])
            a, b = d1[sl][m], d2[sl][m]
            out.append(float(np.corrcoef(a, b)[0, 1])
                       if m.sum() >= 3 and a.std() > 0 and b.std() > 0 else np.nan)
        return np.array(out)

    obs = corrs(*diff)
    # Null: per half, MC blocks of S and C (kept aligned) relabelled L/D.
    # For the MIRROR_NULL_OBS the D blocks are mirrored first (S -> -S, C
    # unchanged): with no chiral recognition a mirrored D block is just
    # another L block, and ratio(L) - ratio(mirrored D) is L + D.  Pooling
    # the unmirrored blocks instead would mix opposite signs, and the
    # relabelled "L + D" would carry the orientation profile itself.
    mirror = sign.reshape(len(names), nz)
    blocks = []
    for h in (0, 1):
        pair = []
        for e, flip in ((halfL[h], 1.0), (halfD[h], mirror)):
            S = _blocks(e, "S", block, drop_empty=False) * flip
            C = _blocks(e, "C", block, drop_empty=False)
            keep = C.reshape(len(C), -1).any(axis=1)
            pair.append((S[keep], C[keep]))
        blocks.append(pair)
    null = np.empty((n_perm, len(names)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(n_perm):
            ds = []
            for (SL, CL), (SD, CD) in blocks:
                S, C = np.concatenate([SL, SD]), np.concatenate([CL, CD])
                perm = rng.permutation(len(S))
                a, b = perm[:len(SL)], perm[len(SL):]
                ds.append((_ratio(S[a].sum(0), C[a].sum(0))
                           - _ratio(S[b].sum(0), C[b].sum(0))).ravel())
            null[i] = corrs(*ds)

    rows = pd.DataFrame([(n, z) for n in names for z in labels], columns=["observable", "depth"])
    rows["tested"] = np.where(sign < 0, "L+D", "L-D")
    rows["half1"], rows["se_half1"] = diff[0], se[0]
    rows["half2"], rows["se_half2"] = diff[1], se[1]
    rows["same_sign"] = np.sign(diff[0]) == np.sign(diff[1])
    rows["both_clear_of_zero"] = rows["same_sign"] & \
        (np.abs(diff[0]) > Z95_SE * se[0]) & (np.abs(diff[1]) > Z95_SE * se[1])
    rows = rows[ok].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        summary = pd.DataFrame({
            "observable": names, "corr_halves": obs,
            "null_95": np.nanpercentile(null, 95, axis=0),
            "p_perm": (1 + np.sum(null >= obs, axis=0)) / (1 + n_perm)})
    counts = rows.groupby("observable").agg(bins=("depth", "size"),
                                            same_sign=("same_sign", "sum"),
                                            both_clear=("both_clear_of_zero", "sum"))
    summary = summary.merge(counts, left_on="observable", right_index=True, how="left")
    return rows, summary


#: Standard errors either side of zero for a half's difference to count as
#: clear of it (a 95% interval).
Z95_SE = 1.96


def split_half_pose(eL, eD, grid, block=50, n_perm=200, seed=0, min_frames=1000,
                    key="pose_depth", flip_d=None):
    """
    Does the L - D pose difference replicate?  The pose maps of
    `pose_maps_depth` are built separately from the first and the second
    half of every stage (MC order), and the two L - D difference maps are
    correlated (Pearson over the 288 bins), per pose depth bin and for the
    depth-matched map.  Two halves of one run are not fully independent,
    but blocks of `block` paths apart are close to it.

    Null: within each half, MC blocks are relabelled L/D at random; the
    correlation of the two relabelled difference maps gives `null_95` and
    `p_perm`.  Relabelled blocks still carry part of any real difference,
    so with one present the null comes out high: the test then has the
    right false-alarm rate but is conservative.  `key` and `flip_d` as in
    pose_maps_depth.
    Returns (table, maps for split_half_maps.npz).
    """
    flip = (lambda b: b) if flip_d is None else flip_d
    rng = np.random.default_rng(seed)
    fL = eL.raw("pose_n_raw")
    fD = eD.raw("pose_n_raw")
    use = (fL >= min_frames) & (fD >= min_frames)
    depths = np.flatnonzero(use)
    hL, hD = halves(eL), halves(eD)
    blocks = [(_blocks(eL.subset(hL[h]), key, block),
               flip(_blocks(eD.subset(hD[h]), key, block))) for h in (0, 1)]

    def diffs(L, D):
        """Per-depth and matched L - D maps of normalised maps."""
        with np.errstate(invalid="ignore", divide="ignore"):
            mL, mD = L.sum(axis=(1, 2)), D.sum(axis=(1, 2))
            per = L / mL[:, None, None] - D / mD[:, None, None]
        pL, pD, _ = _matched(L, D, use)
        return [pL - pD] + [per[d] for d in depths]

    def corr(a, b):
        a, b = np.ravel(a), np.ravel(b)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
            return np.nan
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    obs_maps = [diffs(bl.sum(axis=0), bd.sum(axis=0)) for bl, bd in blocks]
    obs = np.array([corr(a, b) for a, b in zip(*obs_maps)])
    null = np.empty((n_perm, obs.size))
    for i in range(n_perm):
        perm_maps = []
        for bl, bd in blocks:
            pool = np.concatenate([bl, bd])
            perm = rng.permutation(len(pool))
            perm_maps.append(diffs(pool[perm[:len(bl)]].sum(axis=0),
                                   pool[perm[len(bl):]].sum(axis=0)))
        null[i] = [corr(a, b) for a, b in zip(*perm_maps)]

    labels = grid.pose_depth_labels()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df = pd.DataFrame({"map": ["depth_matched"] + ["depth"] * len(depths),
                           "depth": ["matched"] + [labels[d] for d in depths],
                           "corr_halves": obs,
                           "null_mean": np.nanmean(null, axis=0),
                           "null_95": np.nanpercentile(null, 95, axis=0),
                           "p_perm": (1 + np.sum(null >= obs, axis=0)) / (1 + n_perm)})
    df["blocks_L_half1"], df["blocks_L_half2"] = len(blocks[0][0]), len(blocks[1][0])
    df["blocks_D_half1"], df["blocks_D_half2"] = len(blocks[0][1]), len(blocks[1][1])
    maps = {"half1": np.array(obs_maps[0]), "half2": np.array(obs_maps[1]),
            "labels": np.array(df["depth"], dtype=str), "phi_edges": np.linspace(-180, 180, grid.n_phi + 1),
            "cos_edges": np.linspace(-1, 1, grid.n_cos + 1)}
    return df, maps


# ── How long the paths stay correlated ───────────────────────────────────────

def _obs_index(name):
    return [o[0] for o in PROFILE_OBS].index(name)


def _glyc_trans():
    centers = np.linspace(-180, 180, N_GLYC_BINS + 1)[:-1] + 180.0 / N_GLYC_BINS
    return np.abs(centers) >= 120


#: Per-path (S, C) pairs whose weighted ratio sum(w S) / sum(w C) is one of the
#: quantities compare reports, pooled over depth: what the blocking analysis
#: follows.  Each takes a (stage) Ensemble view and returns per-unit arrays.
BLOCKING_OBS = {
    "contact fraction": lambda v: (v.unit_values("S")[:, _obs_index("p_contact")].sum(-1),
                                   v.unit_values("C")[:, _obs_index("p_contact")].sum(-1)),
    "pose_z (contact)": lambda v: (v.unit_values("S")[:, _obs_index("pose_z")].sum(-1),
                                   v.unit_values("C")[:, _obs_index("pose_z")].sum(-1)),
    "H-face share (contact)": lambda v: (
        v.unit_values("pose")[:, :, : v.unit_values("pose").shape[2] // 2].sum((1, 2)),
        v.unit_values("pose").sum((1, 2))),
    "cos_axis_n": lambda v: (v.unit_values("S")[:, _obs_index("cos_axis_n")].sum(-1),
                             v.unit_values("C")[:, _obs_index("cos_axis_n")].sum(-1)),
    "H-bonds to lipid": lambda v: (v.unit_values("S")[:, _obs_index("n_hb_lipid")].sum(-1),
                                   v.unit_values("C")[:, _obs_index("n_hb_lipid")].sum(-1)),
    "donor 0 without H-bond": lambda v: (v.unit_values("hb")[:, 0, -1, :].sum(-1),
                                         v.unit_values("hb")[:, 0].sum((-1, -2))),
    "stereocentre contact lifetime": lambda v: (
        v.unit_values("life_sum")[:, LIFE_KINDS.index("lipid_contact")],
        v.unit_values("life_n")[:, LIFE_KINDS.index("lipid_contact")]),
    "bulk glycerol b trans": lambda v: (v.unit_values("glyc_bulk")[:, 1, _glyc_trans()].sum(-1),
                                        v.unit_values("glyc_bulk")[:, 1].sum(-1)),
}
BLOCK_SIZES = (1, 10, 25, 50, 100, 250, 500, 1000, 2000)
MIN_BLOCKS = 10        # fewer blocks than this: the inefficiency itself is too noisy


def blocking_analysis(e, side, block_sizes=BLOCK_SIZES):
    """
    The statistical inefficiency s(B) of each BLOCKING_OBS quantity, per
    stage, for blocks of B consecutive paths (MC order): how many paths
    together carry as much information as one independent path.

    For a ratio R = sum(w S) / sum(w C), path i contributes
    z_i = w_i (S_i - R C_i) to its error, and
        s(B) = sum over blocks (sum of z in the block)^2 / sum_i z_i^2,
    1 for independent paths.  It is weight-aware, so a few heavy paths
    count as what they are.  Where s(B) levels off, B is longer than the
    correlation and that s is the true one; the bootstrap with -block B
    underestimates standard errors by about sqrt(s(true) / s(B)).

    On a subset view (e.g. the reactive paths) only the subset's paths
    carry weight, but every path keeps its place in the run: blocks are
    stretches of the run, as in the bootstrap, and the correlation measured
    is the subset's correlation along it.

    Returns (inefficiency table, table of R in each quarter of the run and
    in the whole stage: a drift over the run shows up there).
    """
    rows, quarters = [], []
    for s in e.stages:
        a, b = e.stage_range(s)
        v = e.restrict(a, b)
        w = v.w if v.mask is None else v.w * v.mask
        if not np.any(w > 0):
            continue
        whole = v.unmasked()
        for name, fn in BLOCKING_OBS.items():
            S, C = (np.asarray(x, float) for x in fn(whole))
            tot = (w * C).sum()
            if not tot > 0:
                continue
            R = (w * S).sum() / tot
            z = w * (S - R * C)
            for B in block_sizes:
                m = len(z) // B
                if m < 2:
                    break
                zz = z[: m * B]
                denom = (zz ** 2).sum()
                ineff = (zz.reshape(m, B).sum(1) ** 2).sum() / denom if denom > 0 else np.nan
                rows.append({"side": side, "stage": int(s), "observable": name, "block": B,
                             "n_blocks": m, "inefficiency": ineff,
                             "reliable": m >= MIN_BLOCKS})
            parts = np.array_split(np.arange(len(z)), 4)
            q = [((w[p] * S[p]).sum() / (w[p] * C[p]).sum()) if (w[p] * C[p]).sum() > 0
                 else np.nan for p in parts]
            quarters.append({"side": side, "stage": int(s), "observable": name,
                             "whole_stage": R, **{f"quarter{i + 1}": x for i, x in enumerate(q)}})
    return pd.DataFrame(rows), pd.DataFrame(quarters)


#: The automatic block choice: a quantity has levelled off at B when no
#: larger block raises its inefficiency by more than AUTO_TOL, and only block
#: sizes that leave AUTO_MIN_BLOCKS blocks in a stage are considered.  A
#: quantity whose inefficiency stays within AUTO_Z standard errors (one-sided
#: 95%) of 1 at every size counts as uncorrelated.
AUTO_TOL = 0.15
AUTO_MIN_BLOCKS = 15
AUTO_Z = 1.645

#: BLOCKING_OBS that follow the whole membrane rather than the permeant's
#: surroundings.  The lipids relax far more slowly than the permeant does
#: between paths, so these set only the block of the table that reports them
#: (glycerol_conformers), not the block of every permeant comparison.
LIPID_BLOCKING_OBS = ("bulk glycerol b trans",)


def choose_block(table, tol=AUTO_TOL, min_blocks=AUTO_MIN_BLOCKS, z=AUTO_Z):
    """
    The block size for one analysis folder, from its blocking table.

    Per side, stage and quantity, among the sizes that leave at least
    `min_blocks` blocks:
      - 1, if s(B) <= 1 + tol + z sqrt(2 / (m - 1)) at every size (m blocks):
        no evidence of correlation beyond the scatter of s itself, which
        from m blocks is about sqrt(2 / (m - 1)) relative.  Without this a
        single noisy s at 15-20 blocks (+-35%) can ask for the largest block
        when the paths are in fact independent;
      - else the smallest B after which s(B') <= (1 + tol) s(B) for every
        larger B'.
    The folder takes the largest of these (the slowest quantity decides).  A
    quantity that never levels off asks for the largest size allowed and is
    marked as not levelled.  If the stages allow different largest sizes,
    the choice is capped at the smallest of them, so every stage keeps
    `min_blocks` blocks.

    Returns (block, per-quantity table, summary dict).
    """
    recs = []
    ok = table[np.isfinite(table["inefficiency"]) & (table["n_blocks"] >= min_blocks)]
    for (side, stage, obs), h in ok.groupby(["side", "stage", "observable"]):
        h = h.sort_values("block")
        B, s = h["block"].to_numpy(), h["inefficiency"].to_numpy()
        m = h["n_blocks"].to_numpy()
        independent = bool(np.all(s <= 1 + tol + z * np.sqrt(2.0 / np.maximum(m - 1, 1))))
        need, levelled, at = int(B[-1]), False, s[-1]
        if independent:
            need, levelled, at = int(B[0]), True, s[0]
        else:
            for i in range(len(B) - 1):
                if np.all(s[i + 1:] <= (1 + tol) * s[i]):
                    need, levelled, at = int(B[i]), True, s[i]
                    break
        recs.append({"side": side, "stage": int(stage), "observable": obs,
                     "needed_block": need, "levelled": levelled, "independent": independent,
                     "inefficiency_there": at, "largest_allowed": int(B[-1])})
    rec = pd.DataFrame(recs)
    if rec.empty:
        return 1, rec, {"block": 1, "levelled": False, "capped": False,
                        "decided_by": "too few paths for any blocking"}
    cap = int(rec.groupby(["side", "stage"])["largest_allowed"].max().min())
    worst = rec.loc[rec["needed_block"].idxmax()]
    block = int(min(worst["needed_block"], cap))
    info = {"block": block, "levelled": bool(rec["levelled"].all()),
            "capped": bool(worst["needed_block"] > cap),
            "decided_by": f"{worst['observable']} ({worst['side']} stage {worst['stage']})",
            "not_levelled": "; ".join(f"{r.observable} ({r.side} stage {r.stage})"
                                     for r in rec[~rec["levelled"]].itertuples())}
    return block, rec, info


def _choose_and_report(blocking, fixed, out_dir, label):
    """
    The blocks for one folder, `fixed` if given, else from choose_block:
    one for the permeant comparisons, from every quantity but the
    LIPID_BLOCKING_OBS, and one for glycerol_conformers, which also reports
    the bulk lipids (at least the permeant block).  Writes block_choice.csv
    (per quantity, the block it is used for in `chosen_block`) and prints
    one line.  Returns (block, lipid_block).
    """
    lipid = blocking["observable"].isin(LIPID_BLOCKING_OBS) if not blocking.empty \
        else pd.Series(dtype=bool)
    block, rec, info = choose_block(blocking[~lipid] if lipid.any() else blocking)
    l_block, l_rec, l_info = (choose_block(blocking[lipid]) if lipid.any()
                              else (block, pd.DataFrame(), {}))
    l_block = max(block, l_block) if not l_rec.empty else block
    if fixed is not None:
        block = l_block = int(fixed)
        info = {**info, "block": block}
    mode = "fixed" if fixed is not None else "auto"
    rec = pd.concat([rec.assign(chosen_block=block, used_for="permeant comparisons"),
                     l_rec.assign(chosen_block=l_block, used_for="glycerol_conformers")]
                    if not l_rec.empty else
                    [rec.assign(chosen_block=block, used_for="permeant comparisons")],
                    ignore_index=True).assign(mode=mode)
    rec.to_csv(os.path.join(out_dir, "block_choice.csv"), index=False)
    if fixed is not None:
        print(f"  {label}: -block {block} (fixed)")
    else:
        msg = f"  {label}: block {block} chosen, set by {info['decided_by']}"
        if info["capped"]:
            msg += f"; capped so every stage keeps {AUTO_MIN_BLOCKS} blocks"
        if not l_rec.empty:
            msg += f"; block {l_block} for glycerol_conformers (bulk lipids)"
        n_bad = int((~rec["levelled"].astype(bool)).sum()) if "levelled" in rec else 0
        n_all = len(rec)
        if n_bad:
            msg += (f".  [!] {n_bad} of {n_all} quantity/stage pairs had not levelled "
                    f"off by the largest size allowed (see block_choice.csv), so "
                    f"error bars here may still be too small")
        print(msg)
    _report_blocking(blocking[~lipid] if lipid.any() else blocking, block)
    if lipid.any():
        _report_blocking(blocking[lipid], l_block)
    return block, l_block


def _report_blocking(table, block):
    """Per side and stage: the inefficiency at the -block in use against the
    largest block size that still has MIN_BLOCKS blocks."""
    print(f"\n  Blocking analysis (statistical inefficiency; -block {block} in use):")
    for (side, stage), g in table.groupby(["side", "stage"]):
        ok = g[g["reliable"]]
        if ok.empty:
            continue
        b_max = ok["block"].max()
        now = g[g["block"] == block].set_index("observable")["inefficiency"] \
            if (g["block"] == block).any() else None
        top = ok[ok["block"] == b_max].set_index("observable")["inefficiency"]
        worst = (np.sqrt(top / now).max() if now is not None else np.nan)
        rising = (ok.groupby("observable")["inefficiency"]
                  .apply(lambda x: x.iloc[-1] > 1.3 * x.iloc[-2] if len(x) > 1 else False))
        print(f"    {side} stage {stage}: s at B={b_max} ({int(ok['n_blocks'].min())} blocks) "
              f"{top.min():.1f}-{top.max():.1f}"
              + (f"; standard errors with -block {block} too small by up to "
                 f"x{worst:.1f}" if np.isfinite(worst) else "")
              + (f"; still rising for {int(rising.sum())} of {len(rising)} quantities"
                 if rising.any() else "; levelled off"))


def glycerol_conformers(eL, eD, **boot):
    """
    RESPONSE: g+/t/g- populations of the two glycerol dihedrals
    (a: O11-C1-C2-O21, b: O21-C2-C3-O31) for the lipid in contact with the
    permeant, next to the bulk populations of the same trajectories, and the
    contact-induced shift (contact - bulk).  A diastereomeric contact predicts
    that L and D induce DIFFERENT shifts: the L_minus_D of the
    "contact_minus_bulk" rows is that test, and it cancels any small
    difference in the bulk conformers between the two runs.
    """
    centers = np.linspace(-180, 180, N_GLYC_BINS + 1)[:-1] + 180.0 / N_GLYC_BINS
    states = {"g+": (centers > 0) & (centers < 120), "t": np.abs(centers) >= 120,
              "g-": (centers < 0) & (centers > -120)}

    def fractions(h):
        return np.array([_ratio(h[dih, m].sum(), h[dih].sum())
                         for dih in range(2) for m in states.values()], dtype=float)

    def stat(a):
        contact, bulk = fractions(a["glyc_partner"]), fractions(a["glyc_bulk"])
        return np.r_[contact, bulk, contact - bulk]

    pL, pD, rL, rD = bootstrap(eL, eD, stat, **boot)
    rows = [(src, dih, s) for src in ("contact", "bulk", "contact_minus_bulk")
            for dih in ("a_O11-C1-C2-O21", "b_O21-C2-C3-O31") for s in states]
    return _diff_table(pL, pD, rL, rD, pd.DataFrame(rows, columns=["lipid", "dihedral", "state"]))


#: The glycerol dihedrals extracted for every lipid: (key, atoms).
GLYC_DIHEDRALS = (("a", "O11-C1-C2-O21"), ("b", "O21-C2-C3-O31"))


def glycerol_dihedrals(eL, eD, min_frames=200, **boot):
    """
    RESPONSE, in full: the distribution of each glycerol dihedral in 10 deg
    bins, for the lipid the permeant touches (contact frames) and for all
    lipids (bulk), L and D with bootstrap SEs, and an L - D test per bin.
    glycerol_conformers reports the same distributions summed into g+/t/g-.
    Contact distributions with fewer than `min_frames` contact frames on
    either side are not tested.
    Returns (table, arrays for glycerol_dihedrals.npz).
    """
    def norm(h):
        with np.errstate(invalid="ignore", divide="ignore"):
            return h / h.sum(axis=-1, keepdims=True)

    def stat(a):
        return np.r_[norm(a["glyc_partner"]).ravel(), norm(a["glyc_bulk"]).ravel()]

    pL, pD, rL, rD = bootstrap(eL, eD, stat, **boot)
    edges = np.linspace(-180.0, 180.0, N_GLYC_BINS + 1)
    rows = [(lip, f"{k}_{atoms}", edges[i], edges[i + 1])
            for lip in ("contact", "bulk") for k, atoms in GLYC_DIHEDRALS
            for i in range(N_GLYC_BINS)]
    idx = pd.DataFrame(rows, columns=["lipid", "dihedral", "angle_lo", "angle_hi"])
    df = _diff_table(pL, pD, rL, rD, idx)
    contact = int(np.argmax([o[2] for o in PROFILE_OBS]))     # a contact-only observable
    fL, fD = eL.raw("C_raw")[contact].sum(), eD.raw("C_raw")[contact].sum()
    if min(fL, fD) < min_frames:
        df.loc[df["lipid"] == "contact", "p_boot"] = np.nan
    ok = df["p_boot"].notna().to_numpy()
    df["q_bh"] = np.nan
    df.loc[ok, "q_bh"] = benjamini_hochberg(df.loc[ok, "p_boot"].to_numpy())
    shape = (2, len(GLYC_DIHEDRALS), N_GLYC_BINS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        arrays = {"edges": edges, "lipids": np.array(["contact", "bulk"]),
                  "dihedrals": np.array([f"{k}: {atoms}" for k, atoms in GLYC_DIHEDRALS]),
                  "L": pL.reshape(shape), "D": pD.reshape(shape),
                  "Lse": np.nanstd(rL, axis=0).reshape(shape),
                  "Dse": np.nanstd(rD, axis=0).reshape(shape),
                  "q": df["q_bh"].to_numpy().reshape(shape),
                  "contact_frames": np.array([fL, fD])}
    return df, arrays


def _report_trimming(label, meta):
    if meta.empty or "ensemble" not in meta or meta["ensemble"].isna().all():
        print(f"  {label}: no trimming applied")
        return
    for stage, ok in meta[meta["ensemble"].notna()].groupby("stage"):
        ens = ok["ensemble"].value_counts().to_dict()
        print(f"  {label} stage {stage}: {len(ok)} paths trimmed ({ens}); first point "
              f"deleted in {int(ok['deleted_first'].sum())}, last in "
              f"{int(ok['deleted_last'].sum())}")


def compare(unitsL, unitsD, out_dir, grid=None, workers=8,
            temperature=310.0, n_boot=500, block=None, seed=0,
            stagesL=None, stagesD=None, factorsL=None, factorsD=None,
            by_class=False, split_half=False, unweighted=False):
    """
    Run every comparison and write the tables to out_dir.  The units carry
    their own weights, stage, trimming context and cache folder (see
    `setup_stages`); stagesL/stagesD are the per-stage summaries, written
    alongside, and factorsL/factorsD the StageFactorModels whose Pcross
    errors the bootstrap propagates (None: factors held fixed).

    by_class also writes the comparisons that are defined for a subset of
    paths for the reactive and the unreactive plus paths alone, each to its
    subfolder (PATH_SUBSETS), pooled over the stages and, in a further
    subfolder per stage (stage_folder_names), for each stage on its own;
    split_half adds the split-half replication of the pose maps
    (split_half_pose) to every set analysed.

    block: paths per bootstrap/permutation block, or None to let every
    folder choose its own from its blocking analysis (choose_block).

    unweighted repeats it all with every path equally weighted within its
    set (Unit.weight_uniform) into out_dir/unweighted, from the same
    reduced paths: only the sums are redone.  Free energies are left out
    there, since without the WHAM weights they are not equilibrium ones.
    """
    grid = with_hist_limit(grid or Grid(), stagesL)
    os.makedirs(out_dir, exist_ok=True)

    uL, rL, mL = reduce_units(unitsL, grid, workers)
    uD, rD, mD = reduce_units(unitsD, grid, workers)
    if not rL or not rD:
        raise RuntimeError("no path of " + ("L" if not rL else "D") + " could be "
                           "reduced; see the [!] messages above")
    uL, missL = rescale_missing(unitsL, uL)
    uD, missD = rescale_missing(unitsD, uD)
    uL, _ = rescale_missing(unitsL, uL, attr="weight_uniform")
    uD, _ = rescale_missing(unitsD, uD, attr="weight_uniform")
    for label, st, miss in (("L", stagesL, missL), ("D", stagesD, missD)):
        for r in miss[miss["paths_missing"] > 0].itertuples():
            print(f"  [!] {label} stage {r.stage}: {r.paths_missing} path(s) missing, "
                  f"{r.weight_missing_share:.2%} of the stage's weight; the others "
                  f"are scaled by {r.weight_rescale:.4f} to stand in for them")
        if st is not None:
            st.merge(miss, on="stage", how="left").to_csv(
                os.path.join(out_dir, f"stages_{label}.csv"), index=False)
    for label, meta in (("L", mL), ("D", mD)):
        _report_stereo(label, meta, label).to_csv(
            os.path.join(out_dir, f"stereochemistry_{label}.csv"), index=False)
        meta.drop(columns=[c for c in STEREO_COLS if c in meta]).to_csv(
            os.path.join(out_dir, f"path_trimming_{label}.csv"), index=False)
        _report_trimming(label, meta)
    eL, eD = Ensemble(uL, rL, factorsL), Ensemble(uD, rD, factorsD)
    print(f"  units: L {eL.n}, D {eD.n}")
    common = dict(grid=grid, boot=dict(n_boot=n_boot, block=block, seed=seed),
                  by_class=by_class, split_half=split_half,
                  names=stage_folder_names(stagesL if stagesL is not None else stagesD))

    print("\n  ══ WHAM-weighted ══")
    tables = _analyse(eL, eD, uL, uD, out_dir, temperature=temperature, **common)
    if unweighted:
        sub = os.path.join(out_dir, UNWEIGHTED_DIR)
        os.makedirs(sub, exist_ok=True)
        from dataclasses import replace

        print(f"\n  ══ unweighted (every path equal within its set; written to {sub}) ══")
        _analyse(eL.with_weights([u.weight_uniform for u in uL]),
                 eD.with_weights([u.weight_uniform for u in uD]),
                 [replace(u, weight=u.weight_uniform) for u in uL],
                 [replace(u, weight=u.weight_uniform) for u in uD],
                 sub, temperature=None, **common)
    return tables


def _blocking_for(eL, eD, out_dir, fixed, label):
    """Blocking analysis of the paths of one folder (writes blocking.csv and
    blocking_quarters.csv) and the blocks it is to use (block_choice.csv):
    (block, lipid_block), as _choose_and_report."""
    (bLt, bLq), (bDt, bDq) = blocking_analysis(eL, "L"), blocking_analysis(eD, "D")
    blocking = pd.concat([bLt, bDt], ignore_index=True)
    pd.concat([bLq, bDq], ignore_index=True).to_csv(
        os.path.join(out_dir, "blocking_quarters.csv"), index=False)
    blocking.to_csv(os.path.join(out_dir, "blocking.csv"), index=False)
    return _choose_and_report(blocking, fixed, out_dir, label)


#: The subfolder compare writes the unweighted analysis to.
UNWEIGHTED_DIR = "unweighted"


def _analyse(eL, eD, uL, uD, out_dir, grid, boot, by_class, split_half, names,
             temperature=None):
    """
    Everything compare reports, for one weighting of the paths, into
    out_dir: the blocking analysis, the comparisons over all paths (with
    the free energies when `temperature` is given) and, with by_class, the
    reactive and unreactive subsets, pooled and per stage.  `uL`/`uD` are
    the units behind eL/eD, carrying the same weights.
    """
    fixed, seed = boot["block"], boot["seed"]           # fixed: None = automatic
    # How long paths stay correlated decides the block, before anything
    # relies on it; every folder below does the same for its own paths.
    block, lipid_block = _blocking_for(eL, eD, out_dir, fixed, "all paths")
    boot = {**boot, "block": block}

    tables, perm_summary = {}, pd.DataFrame()
    if temperature is not None:
        perm_summary, perm_shares = permeability_ratio(eL, eD, grid, **boot)
        tables = {
            "ddG_depth": depth_ddg(eL, eD, grid, temperature, **boot),
            "permeability_ratio": perm_summary,
            "resistance_shares": perm_shares,
            "ddG_by_hbond_state": state_resolved_ddg(eL, eD, grid, temperature, **boot),
        }
        # Free energy over depth x orientation, and the L/D comparison
        # pooled over the depths that carry the resistance to crossing.
        tables["orientation_free_energy"], ofe = orientation_free_energy(
            eL, eD, grid, temperature, **boot)
        np.savez(os.path.join(out_dir, "orientation_free_energy.npz"), **ofe)
        crossing = (np.array([u.path_class == "reactive" for u in uL]),
                    np.array([u.path_class == "reactive" for u in uD]))
        tables["barrier"] = barrier_comparison(eL, eD, grid, perm_shares, crossing, **boot)
    tables.update(_conditional_tables(eL, eD, grid, out_dir, boot, lipid_block))
    if split_half:
        tables.update(_split_half_tables(eL, eD, grid, out_dir, block, seed))
    _write_tables(tables, out_dir)
    _report(tables, "all paths")
    if not perm_summary.empty:
        r = perm_summary.iloc[0]
        print(f"\n  ln(P_L/P_D) from the free-energy profiles alone (equal D): "
              f"{r.ln_PL_over_PD:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  "
              f"-> P_L/P_D = {r.PL_over_PD:.3g}"
              + ("" if r.complete else "   [!] profile incomplete: an unsampled "
                 "membrane bin is missing from the integral"))

    if by_class:
        # Every stage on its own, all paths: stage factors and the stitching
        # between stages play no part here.
        for k in sorted(set(eL.stages) & set(eD.stages)):
            (aL, bL), (aD, bD) = eL.stage_range(k), eD.stage_range(k)
            _folder(eL.restrict(aL, bL), eD.restrict(aD, bD),
                    os.path.join(out_dir, names.get(k, f"stage{k}")),
                    f"all paths, stage {k}", grid, boot, fixed, split_half, seed)
        classes = path_class_table(uL, uD)
        classes.to_csv(os.path.join(out_dir, "path_classes.csv"), index=False)
        print("\n  Path classes (share of each stage's weight):\n"
              + classes.pivot_table(index=["side", "stage"], columns="path_class",
                                    values="weight_share").round(3).to_string())
        for cls in PATH_SUBSETS:
            mL = np.array([u.path_class == cls for u in uL])
            mD = np.array([u.path_class == cls for u in uD])
            # Pooled over the stages, then each stage on its own.  Within one
            # stage the stage factor scales every path alike, so the stage
            # folders do not depend on Pcross at all.
            sets = [(os.path.join(out_dir, cls), f"{cls} paths, all stages",
                     eL.subset(mL), eD.subset(mD))]
            for k in sorted(set(eL.stages) & set(eD.stages)):
                (aL, bL), (aD, bD) = eL.stage_range(k), eD.stage_range(k)
                sets.append((os.path.join(out_dir, cls, names.get(k, f"stage{k}")),
                             f"{cls} paths, stage {k}",
                             eL.restrict(aL, bL).subset(mL[aL:bL]),
                             eD.restrict(aD, bD).subset(mD[aD:bD])))
            for sub, label, sL, sD in sets:
                _folder(sL, sD, sub, label, grid, boot, fixed, split_half, seed)
    return tables


def _folder(sL, sD, sub, label, grid, boot, fixed, split_half, seed):
    """One analysis folder below the top one (a stage, a path subset, or
    both): its own blocking analysis and block, then every comparison that
    is defined for it, written to `sub`."""
    if not sL.n_counted or not sD.n_counted:
        print(f"\n  [!] {label}: no paths in " + ("L" if not sL.n_counted else "D")
              + "; skipped")
        return
    os.makedirs(sub, exist_ok=True)
    print(f"\n  {label}: L {sL.n_counted}, D {sD.n_counted} paths (written to {sub})")
    b_sub, b_lipid = _blocking_for(sL, sD, sub, fixed, label)
    boot_sub = {**boot, "block": b_sub}
    t = _conditional_tables(sL, sD, grid, sub, boot_sub, b_lipid)
    if split_half:
        t.update(_split_half_tables(sL, sD, grid, sub, b_sub, seed))
    _write_tables(t, sub)
    _report(t, label)


def stage_folder_names(summary):
    """{stage index: 'stage<k>_<run folder name>'} for the per-stage subset
    folders, e.g. stage1_infinit_internal.  `summary` is setup_stages' table."""
    if summary is None:
        return {}
    return {int(k): f"stage{int(k)}_{os.path.basename(os.path.normpath(str(run)))}"
            for k, run in zip(summary["stage"], summary["run"])}


#: The path subsets compare writes for staged runs (weighted or not), each
#: to its own subfolder of the output folder, with one more subfolder per
#: stage inside it.  Only plus-ensemble paths fall in
#: them; the [0-] paths, and with them the bulk-water reference of every
#: free energy, belong to neither.
PATH_SUBSETS = ("reactive", "unreactive")


def path_class_table(uL, uD):
    """Paths and share of the stage's weight per path class, per side and
    stage."""
    rows = []
    for side, units in (("L", uL), ("D", uD)):
        for s in sorted({u.stage for u in units}):
            mine = [u for u in units if u.stage == s]
            total = sum(u.weight for u in mine)
            for cls in ("reactive", "unreactive", "minus"):
                members = [u for u in mine if u.path_class == cls]
                rows.append({"side": side, "stage": s, "path_class": cls,
                             "paths": len(members),
                             "weight_share": (sum(u.weight for u in members) / total
                                              if total > 0 else np.nan)})
    return pd.DataFrame(rows)


def _conditional_tables(eL, eD, grid, out_dir, boot, lipid_block=None):
    """
    Every comparison that is an average WITHIN a depth bin (or pooled over
    them), so it is defined for any subset of paths: profiles, H-bond
    partners, lifetimes, conformers and the pose maps.  The free energies
    are not: they need the [0-] paths for their bulk-water reference.
    Writes the pose map .npz files to out_dir.  glycerol_conformers, which
    reports the bulk lipids too, uses `lipid_block` (default: the block).
    """
    block, seed = boot["block"], boot["seed"]
    lipid_boot = {**boot, "block": max(block, lipid_block or block)}
    tables = {
        "depth_profiles": depth_profiles(eL, eD, grid, **boot),
        "hbond_partners": hbond_partners(eL, eD, grid, **boot),
        "sn2_preference": sn2_preference(eL, eD, **boot),
        "lifetimes": lifetimes(eL, eD, **boot),
        "lifetime_survival": lifetime_survival(eL, eD, grid, **boot),
        "glycerol_conformers": glycerol_conformers(eL, eD, **lipid_boot),
    }
    tables["glycerol_dihedrals"], glyc = glycerol_dihedrals(eL, eD, **lipid_boot)
    np.savez(os.path.join(out_dir, "glycerol_dihedrals.npz"), **glyc)
    maps, tables["pose_jsd"] = pose_maps(eL, eD, block=block, seed=seed)
    np.savez(os.path.join(out_dir, "pose_maps.npz"), **maps,
             phi_edges=np.linspace(-180, 180, grid.n_phi + 1),
             cos_edges=np.linspace(-1, 1, grid.n_cos + 1))
    maps_z, tables["pose_jsd_depth"] = pose_maps_depth(eL, eD, grid, block=block, seed=seed)
    np.savez(os.path.join(out_dir, "pose_maps_depth.npz"), **maps_z)
    tables["histograms"], hist = histograms(eL, eD, grid, **boot)
    np.savez(os.path.join(out_dir, "histograms.npz"), **hist)
    tables["orient_jsd_depth"], orient = orientation_maps(eL, eD, grid, block=block, seed=seed)
    np.savez(os.path.join(out_dir, "orient_maps_depth.npz"), **orient)
    tables["reorientation"], rot = reorientation(eL, eD, grid, **boot)
    np.savez(os.path.join(out_dir, "reorientation.npz"), **rot)
    return tables


def _split_half_tables(eL, eD, grid, out_dir, block, seed):
    """Both split-half tests: the pose maps (split_half.csv, .npz) and the
    depth profiles (split_half_profiles.csv, split_half_profiles_summary.csv)."""
    rows, summary = split_half_profiles(eL, eD, grid, block=block, seed=seed)
    tables, maps = [], {}
    for name, key, flip in ORIENT_MAPS:
        df, m = split_half_pose(eL, eD, grid, block=block, seed=seed, key=key, flip_d=flip)
        df.insert(0, "vector", name)
        tables.append(df)
        maps.update({f"{name}_{k}": v for k, v in m.items()})
    np.savez(os.path.join(out_dir, "split_half_orient_maps.npz"), **maps)
    return {"split_half": _split_half(eL, eD, grid, out_dir, block, seed),
            "split_half_profiles": rows, "split_half_profiles_summary": summary,
            "split_half_orient": pd.concat(tables, ignore_index=True)}


def _split_half(eL, eD, grid, out_dir, block, seed):
    table, maps = split_half_pose(eL, eD, grid, block=block, seed=seed)
    np.savez(os.path.join(out_dir, "split_half_maps.npz"), **maps)
    return table


def _write_tables(tables, out_dir):
    for name, df in tables.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)


def _report(tables, label):
    print(f"\n  {label}, rows with q_bh < 0.05:")
    for name, df in tables.items():
        if "q_bh" in df:
            hits = df[df["q_bh"] < 0.05]
            print(f"    {name:20s} {len(hits):4d} / {df['q_bh'].notna().sum()}")
    z = tables.get("pose_jsd_depth")
    if z is not None:
        print("  pose maps (depth_matched: L and D on the same contact-depth weights):\n"
              + z[["map", "depth", "jsd", "null_95", "p_perm", "q_bh", "frames_L",
                   "frames_D"]].to_string(index=False))
    sp = tables.get("split_half_profiles_summary")
    if sp is not None:
        print("  split-half replication of the L - D depth profiles:\n"
              + sp.round(3).to_string(index=False))
    sh = tables.get("split_half")
    if sh is not None:
        print("  split-half replication of the L - D pose maps:\n"
              + sh[["map", "depth", "corr_halves", "null_95", "p_perm"]]
              .round(4).to_string(index=False))


#: Every file `compare` writes, checked up front so -O is needed to replace them.
COMPARE_OUTPUTS = (
    [f"{stem}_{side}.csv" for stem in ("stages", "stereochemistry", "path_trimming")
     for side in ("L", "D")]
    + [f"{name}.csv" for name in (
        "depth_profiles", "ddG_depth", "permeability_ratio", "resistance_shares",
        "ddG_by_hbond_state", "hbond_partners", "sn2_preference", "lifetimes",
        "lifetime_survival", "glycerol_conformers", "pose_jsd", "pose_jsd_depth",
        "path_classes", "blocking", "blocking_quarters", "block_choice", "histograms",
        "orient_jsd_depth", "reorientation", "orientation_free_energy", "barrier",
        "glycerol_dihedrals")]
    + ["pose_maps.npz", "pose_maps_depth.npz", "histograms.npz", "orient_maps_depth.npz",
       "reorientation.npz", "orientation_free_energy.npz", "glycerol_dihedrals.npz"]
)
#: What compare writes into each PATH_SUBSETS subfolder.
SUBSET_OUTPUTS = (
    [f"{name}.csv" for name in (
        "depth_profiles", "hbond_partners", "sn2_preference", "lifetimes",
        "lifetime_survival", "glycerol_conformers", "pose_jsd", "pose_jsd_depth",
        "blocking", "blocking_quarters", "block_choice", "histograms", "orient_jsd_depth",
        "reorientation", "glycerol_dihedrals")]
    + ["pose_maps.npz", "pose_maps_depth.npz", "histograms.npz", "orient_maps_depth.npz",
       "reorientation.npz", "glycerol_dihedrals.npz"]
)
#: What compare writes into UNWEIGHTED_DIR, besides its subset folders.
UNWEIGHTED_OUTPUTS = SUBSET_OUTPUTS + ["path_classes.csv"]
#: Added to the output folder and every subset folder by -split-half.
SPLIT_HALF_OUTPUTS = ("split_half.csv", "split_half_maps.npz", "split_half_profiles.csv",
                      "split_half_profiles_summary.csv", "split_half_orient.csv",
                      "split_half_orient_maps.npz")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_paths(spec):
    if not spec:
        return None
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def _units(load_dir, xtc_glob, paths):
    if load_dir:
        return units_from_load_dir(load_dir, _parse_paths(paths))
    return units_from_files(glob.glob(xtc_glob))


def _split(value):
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


def _per_run(value, run_dirs, flag, exists=os.path.isfile):
    """
    One path per run directory from `value`: a single path looked for inside
    every run directory, or one path per run, comma-separated, in run order.
    Each path is tried inside its run directory first, then as given.
    `exists` is os.path.isfile or os.path.isdir.
    """
    items = _split(value)
    if len(items) == 1:
        items = items * len(run_dirs)
    elif len(items) != len(run_dirs):
        raise typer.BadParameter(
            f"{flag}: give one path for every run directory or {len(run_dirs)} "
            f"comma-separated paths, got {len(items)}")
    out = []
    for run_dir, item in zip(run_dirs, items):
        for cand in (os.path.join(run_dir, item), item):
            if exists(cand):
                out.append(cand)
                break
        else:
            raise typer.BadParameter(
                f"{flag}: neither {os.path.join(run_dir, item)} nor {item} exists")
    return out


def _stage_specs(runs, data, toml, cache, side):
    run_dirs = _split(runs)
    data_files = _per_run(data, run_dirs, f"-data-{side}")
    tomls = (_per_run(toml, run_dirs, f"-toml-{side}") if toml
             else [_find_toml(r) for r in run_dirs])
    caches = _per_run(cache, run_dirs, f"-cache-{side}", exists=os.path.isdir)
    return [StageSpec(r, d, t, c)
            for r, d, t, c in zip(run_dirs, data_files, tomls, caches)]


def _check_cache_dirs(dirs):
    """
    Fail at once if any cache folder holds no current-version chunk.  A wrong
    -cache-l/-cache-d (say the folder above the cache) would otherwise only
    show up as 'not cached' for every path, after the slow setup.  `dirs` is
    [(flag side, what it is for, folder)].  Reads only the chunks' indexes.
    """
    bad = [f"  -cache-{side} for {what}: {d}" for side, what, d in dirs
           if not cache_index(d)]
    if bad:
        raise typer.BadParameter(
            f"no v{CACHE_VERSION} cache file (*{CHUNK_SUFFIX}) in\n" + "\n".join(bad)
            + "\nPoint -cache-l/-cache-d at the folder that holds the chunk files "
            "chiral-extract wrote, or run chiral-extract (again, if they are "
            "from an older version).")


def check_cached(units, label):
    """
    Look every unit up in its cache before anything is reduced, and print one
    line per stage.  Raises if a stage has none of its units cached, which
    means a wrong cache folder; otherwise the missing units are reported here,
    once, and left out of the analysis by reduce_units.
    """
    indexes = {d: cache_index(d) for d in {u.cache_dir for u in units}}
    stages = {}
    for u in units:
        stages.setdefault(u.stage, []).append(u)
    empty = []
    for s, members in sorted(stages.items()):
        absent, stale = [], []
        for u in members:
            try:
                _locate(u, indexes[u.cache_dir])
            except LookupError:
                (stale if indexes[u.cache_dir].get(u.key) else absent).append(u.uid)
        n, folder = len(members), members[0].cache_dir
        print(f"  {label} stage {s}: {n - len(absent) - len(stale)}/{n} paths "
              f"cached in {folder}")
        for what, uids in (("not in the cache", absent),
                           ("changed since extraction (traj.txt differs)", stale)):
            if uids:
                print(f"    [!] {len(uids)} {what}: "
                      + ", ".join(map(str, uids[:10])) + (" ..." if len(uids) > 10 else ""))
        if len(absent) + len(stale) == n:
            empty.append(f"  {label} stage {s}: {folder}")
        elif absent or stale:
            print("    they are left out, and the stage's other paths take over their "
                  "weight; rerun chiral-extract on this stage if their "
                  "trajectories still exist")
    if empty:
        raise typer.BadParameter(
            "none of the paths of\n" + "\n".join(empty) + "\nis in its cache "
            "folder; is -cache-l/-cache-d the folder chiral-extract wrote for "
            "that stage?")


def chiral_extract(
    # ── Input data ────────────────────────────────────────────────────────
    tpr: Annotated[Optional[str], typer.Option("-tpr", help="Topology (.tpr) of the simulated system; atom names and residues are resolved from it once per worker. Required", rich_help_panel=panels.INPUT)] = None,
    permeant: Annotated[Optional[str], typer.Option("-permeant", help=f"Which permeant definition to use (stereocentre, CIP substituents, H-bond donors/acceptors); one of: {', '.join(sorted(PERMEANTS))}. Required", rich_help_panel=panels.INPUT)] = None,
    load_dir: Annotated[Optional[str], typer.Option("-load-dir", help="infretis load/ folder: every path in it is read through its traj.txt plan. Give this or -xtc", rich_help_panel=panels.INPUT)] = None,
    xtc: Annotated[Optional[str], typer.Option("-xtc", help="Plain MD / umbrella windows instead of -load-dir: a glob of .xtc files (quote it); each file is one unit", rich_help_panel=panels.INPUT)] = None,

    # ── Dataset construction ──────────────────────────────────────────────
    path_numbers: Annotated[Optional[str], typer.Option("-path-numbers", help="Only these paths of -load-dir, e.g. '1-5000,6001'; default all", rich_help_panel=panels.DATASET)] = None,

    # ── Model and training ────────────────────────────────────────────────
    workers: Annotated[int, typer.Option("-workers", help="Worker processes; each extracts one path at a time", rich_help_panel=panels.MODEL)] = 8,

    # ── Output ────────────────────────────────────────────────────────────
    cache: Annotated[Optional[str], typer.Option("-cache", help=f"Cache folder. Default with -load-dir: {DEFAULT_CACHE}/ in the run directory (the parent of load/), which is where chiral-compare looks. Required with -xtc", rich_help_panel=panels.OUTPUT)] = None,
    paths_per_file: Annotated[int, typer.Option("-paths-per-file", help="Paths stored together in one cache file. Fewer files (inodes) with larger values; a killed run loses at most the unfinished files, and extending the simulation rebuilds only the files whose paths changed", rich_help_panel=panels.OUTPUT)] = 500,
    overw: Annotated[bool, typer.Option("-O", help="Re-extract every path, even ones already cached", rich_help_panel=panels.OUTPUT)] = False,
):
    """Extract per-frame stereocentre-contact features of every path, for chiral-compare.

    Each phase point gets the permeant's depth, orientation, pose in the body
    frame of the nearest lipid stereocentre, H-bond partners per donor, and
    the glycerol dihedrals of the contacted lipid. Only the frames traj.txt
    lists are computed, and they are stored in path order, -paths-per-file
    paths per cache file. The trajectories are read with a reader that writes
    no offset files next to them.

    A path is re-extracted only if its traj.txt changed (or the permeant or
    cache version did), so rerunning after an interruption or after extending
    the simulation resumes where it stopped. Run it once per stage (entry,
    internal, escape) of each enantiomer.
    """
    if tpr is None or permeant is None:
        raise typer.BadParameter("-tpr and -permeant are required.")
    if permeant not in PERMEANTS:
        raise typer.BadParameter(
            f"-permeant {permeant!r}: choose from {sorted(PERMEANTS)}.")
    if (load_dir is None) == (xtc is None):
        raise typer.BadParameter("Give exactly one of -load-dir and -xtc.")
    if paths_per_file < 1:
        raise typer.BadParameter("-paths-per-file must be at least 1.")
    if load_dir is not None:
        run_dir = os.path.dirname(os.path.abspath(load_dir))
        cache = cache or os.path.join(run_dir, DEFAULT_CACHE)
        tag = os.path.basename(run_dir)
        units = units_from_load_dir(load_dir, _parse_paths(path_numbers))
    else:
        if cache is None:
            raise typer.BadParameter("-cache is required with -xtc.")
        tag = "md"
        units = units_from_files(glob.glob(xtc))
    files = {f for u in units for f in u.files}
    print(f"  {len(units)} units, {len(files)} unique xtc files")
    errors = extract_many(units, tpr, permeant, cache, tag, workers, overw, paths_per_file)
    if errors:
        raise typer.Exit(code=1)


def chiral_compare(
    # ── Input data ────────────────────────────────────────────────────────
    runs_l: Annotated[Optional[str], typer.Option("-runs-l", help="Run directories of the L simulation, comma-separated, in chain order (entry, internal, escape). Each holds load/ (traj.txt and order.txt of every path; the trajectories themselves are not needed), the wham/ folder inftools wrote, and the cache. Give this or -xtc-l", rich_help_panel=panels.INPUT)] = None,
    runs_d: Annotated[Optional[str], typer.Option("-runs-d", help="Run directories of the D simulation; see -runs-l", rich_help_panel=panels.INPUT)] = None,
    data_l: Annotated[str, typer.Option("-data-l", help="infretis_data.txt of each L stage: one relative path looked for inside every run directory, or one file per run, comma-separated", rich_help_panel=panels.INPUT)] = "infretis_data.txt",
    data_d: Annotated[str, typer.Option("-data-d", help="infretis_data.txt of each D stage; see -data-l", rich_help_panel=panels.INPUT)] = "infretis_data.txt",
    toml_l: Annotated[Optional[str], typer.Option("-toml-l", help="infretis.toml of each L stage, same rules as -data-l; unset uses RUN_DIR/infretis.toml, else RUN_DIR/post/infretis.toml", rich_help_panel=panels.INPUT)] = None,
    toml_d: Annotated[Optional[str], typer.Option("-toml-d", help="infretis.toml of each D stage; see -toml-l", rich_help_panel=panels.INPUT)] = None,
    cache_l: Annotated[str, typer.Option("-cache-l", help=f"Cache folder of each L stage, same rules as -data-l (default: {DEFAULT_CACHE}/ inside every run directory, where chiral-extract puts it). With -xtc-l: the one folder chiral-extract -cache wrote", rich_help_panel=panels.INPUT)] = DEFAULT_CACHE,
    cache_d: Annotated[str, typer.Option("-cache-d", help="Cache folder of each D stage; see -cache-l", rich_help_panel=panels.INPUT)] = DEFAULT_CACHE,
    xtc_l: Annotated[Optional[str], typer.Option("-xtc-l", help="Plain MD instead of -runs-l: a glob of .xtc files (quote it); each file is its own unweighted, untrimmed unit", rich_help_panel=panels.INPUT)] = None,
    xtc_d: Annotated[Optional[str], typer.Option("-xtc-d", help="Plain MD instead of -runs-d; see -xtc-l", rich_help_panel=panels.INPUT)] = None,

    # ── Dataset construction ──────────────────────────────────────────────
    path_numbers: Annotated[Optional[str], typer.Option("-path-numbers", help="Only these path numbers of every stage, e.g. '1-5000,6001'; default all", rich_help_panel=panels.DATASET)] = None,
    trim: Annotated[bool, typer.Option("-trim/-no-trim", help="Delete the first/last phase point by the plus/minus ensemble criteria, as generate-cvs does; disable to keep every phase point", rich_help_panel=panels.DATASET)] = True,
    nskip: Annotated[int, typer.Option("-nskip", help="Burn-in rows of each infretis_data.txt skipped for the WHAM weights", rich_help_panel=panels.DATASET)] = 1000,
    wham_subdir: Annotated[str, typer.Option("-wham-subdir", help="Name of the wham folder inside each run directory", rich_help_panel=panels.DATASET)] = "wham",
    pcross_col: Annotated[int, typer.Option("-pcross-col", help="Column of wham/Pcross.txt holding the total crossing probability, as in permeation: 3 is P-wham2, 1 is P-wham", rich_help_panel=panels.DATASET)] = 3,
    temperature: Annotated[Optional[float], typer.Option("-temperature", help="Temperature in K for the free energies; unset reads [engine] temperature from the runs' tomls", rich_help_panel=panels.DATASET)] = None,
    depth_max: Annotated[float, typer.Option("-depth-max", help="Last depth-bin edge on |OP|, in OP units; beyond it is one open bulk-water bin", rich_help_panel=panels.DATASET)] = 35.0,
    depth_step: Annotated[float, typer.Option("-depth-step", help="Depth-bin width on |OP|, in OP units", rich_help_panel=panels.DATASET)] = 2.5,
    contact_cut: Annotated[float, typer.Option("-contact-cut", help="Permeant stereocentre to lipid C2 distance (nm) that counts as contact with a lipid stereocentre", rich_help_panel=panels.DATASET)] = 0.7,
    hist_max_depth: Annotated[Optional[float], typer.Option("-hist-max-depth", help="Deepest |OP| (OP units) the 1D and 2D histograms cover: depth bins whose centres lie below it. Default: |lambda_A| of the entry stage, so the [0-] paths, which only reach beyond it, stay out of the histograms (without stages: up to -depth-max). The profiles and free energies always use every depth", rich_help_panel=panels.DATASET)] = None,

    # ── Model and training ────────────────────────────────────────────────
    n_boot: Annotated[int, typer.Option("-n-boot", help="Block-bootstrap replicates for every confidence interval and p-value (the pose-map permutation test always uses 1000 permutations)", rich_help_panel=panels.MODEL)] = 500,
    block: Annotated[str, typer.Option("-block", help="Paths per contiguous MC block in the bootstrap and permutation tests. 'auto' (default): every analysis folder (all paths, each path subset, each stage) runs its own blocking analysis and takes the smallest block after which every quantity's statistical inefficiency has levelled off (no larger block raises it by over 15%; a quantity whose inefficiency stays within the scatter of 1 counts as uncorrelated), keeping at least 15 blocks per stage. The bulk-lipid conformers get their own block, used only for glycerol_conformers; the choice and why are in block_choice.csv. A number fixes the block everywhere", rich_help_panel=panels.MODEL)] = "auto",
    seed: Annotated[int, typer.Option("-seed", help="Random seed for the bootstrap and permutation test", rich_help_panel=panels.MODEL)] = 0,
    split_half: Annotated[bool, typer.Option("-split-half", help="Also test whether the L - D pose maps replicate: build them from the first and the second half of every stage (MC order) and correlate the two, against a block-relabelling null; for all paths and each path subset", rich_help_panel=panels.MODEL)] = False,
    workers: Annotated[int, typer.Option("-workers", help="Worker processes for reducing the paths; each takes one cache file at a time", rich_help_panel=panels.MODEL)] = 8,

    # ── Output ────────────────────────────────────────────────────────────
    out_dir: Annotated[str, typer.Option("-out-dir", help="Directory for the comparison tables", rich_help_panel=panels.OUTPUT)] = "chiral_interface",
    overw: Annotated[bool, typer.Option("-O", help="Force overwriting of existing files", rich_help_panel=panels.OUTPUT)] = False,
):
    """Trajectory-level L vs D comparison at the lipid stereocentres.

    Tests the diastereomeric-contact mechanism of Hu et al., Nat. Chem. 2021,
    on the features cached by chiral-extract: WHERE the free energies differ
    (ddG and every observable by depth, and the ln(P_L/P_D) they imply), WHAT
    the permeant H-bonds to (sn-2 vs sn-1 glycerol oxygens), HOW it sits at
    the stereocentre (pose maps), HOW LONG contacts last, and how the
    contacted lipid's glycerol backbone RESPONDS. Error bars come from a
    block bootstrap over paths within each stage.

    With all stages (entry, internal, escape) and WHAM weights, the depth
    profiles are equilibrium quantities; with a single stage they are
    conditional on that run's start. Staged runs also get the reactive and
    unreactive paths alone (pooled and per stage), and everything once more
    without the WHAM reweighting in unweighted/, from the same reduced paths.
    See the module docstring for details.
    The trajectories are not read: only the cache and each path's traj.txt
    and order.txt.
    """
    for side, runs, xtc in (("l", runs_l, xtc_l), ("d", runs_d, xtc_d)):
        if (runs is None) == (xtc is None):
            raise typer.BadParameter(f"Give exactly one of -runs-{side} and -xtc-{side}.")
    if str(block).strip().lower() == "auto":
        block = None
    else:
        try:
            block = int(block)
        except ValueError:
            raise typer.BadParameter(f"-block: 'auto' or a number of paths, not {block!r}")
        if block < 1:
            raise typer.BadParameter("-block must be at least 1")
    from .chiral_plot import PLOT_OUTPUTS
    by_class = runs_l is not None and runs_d is not None
    extra = list(SPLIT_HALF_OUTPUTS) if split_half else []
    for name in COMPARE_OUTPUTS + list(PLOT_OUTPUTS) + extra:
        _check_overwrite(os.path.join(out_dir, name), overw)
    if by_class:
        stage_dirs = [""] + [f"stage{k}_{os.path.basename(os.path.normpath(r))}"
                             for k, r in enumerate(_split(runs_l))]
        for top in ("", UNWEIGHTED_DIR):
            if top:
                for name in UNWEIGHTED_OUTPUTS + list(PLOT_OUTPUTS) + extra:
                    _check_overwrite(os.path.join(out_dir, top, name), overw)
            for sd in stage_dirs[1:]:                  # all paths, per stage
                for name in SUBSET_OUTPUTS + list(PLOT_OUTPUTS) + extra:
                    _check_overwrite(os.path.join(out_dir, top, sd, name), overw)
            for cls in PATH_SUBSETS:
                for sd in stage_dirs:
                    for name in SUBSET_OUTPUTS + list(PLOT_OUTPUTS) + extra:
                        _check_overwrite(os.path.join(out_dir, top, cls, sd, name), overw)

    grid = Grid(depth_edges=np.arange(0.0, depth_max + 1e-9, depth_step),
                contact_cut=contact_cut, hist_max_depth=hist_max_depth)
    # Resolve and check every cache folder of both enantiomers first: setup
    # reads every traj.txt and order.txt, which takes minutes on a large run.
    plans, dirs = [], []
    for side, runs, data, toml, cache, xtc in (
            ("l", runs_l, data_l, toml_l, cache_l, xtc_l),
            ("d", runs_d, data_d, toml_d, cache_d, xtc_d)):
        if runs is None:
            if not os.path.isdir(cache):
                raise typer.BadParameter(f"-cache-{side}: {cache} is not a folder")
            plans.append((side, None, xtc, cache))
            dirs.append((side, xtc, cache))
        else:
            specs = _stage_specs(runs, data, toml, cache, side)
            plans.append((side, specs, xtc, cache))
            dirs += [(side, sp.run_dir, sp.cache_dir) for sp in specs]
    _check_cache_dirs(dirs)

    sides, summaries, models, run_dirs = [], [], [], []
    for side, specs, xtc, cache in plans:
        if specs is None:
            units = units_from_files(glob.glob(xtc), cache)
            check_cached(units, side.upper())
            sides.append(units)
            summaries.append(None)
            models.append(None)
            continue
        run_dirs += [sp.run_dir for sp in specs]
        units, summary, model = setup_stages(specs, nskip, wham_subdir, pcross_col,
                                             trim=trim,
                                             path_numbers=_parse_paths(path_numbers))
        print(f"  {side.upper()}:\n" + summary.to_string(index=False))
        if len(specs) == 1:
            print(f"  [!] {side.upper()} has a single stage.  Its histogram is conditional "
                  f"on starting in that run's state A, so ddG and the depth "
                  f"profiles are conditional too; pass entry, internal and escape.")
        no_err = summary.loc[summary["pcross_rel_err"].isna(), "run"].tolist()
        if len(specs) > 1 and no_err:
            print(f"  [!] {side.upper()}: no Pcross error (wham/errPtot.txt) for "
                  + ", ".join(no_err) + "; those stages' Pcross is held fixed in "
                  "the bootstrap, so the intervals are too narrow")
        check_cached(units, side.upper())
        sides.append(units)
        summaries.append(summary)
        models.append(model)
    if summaries[0] is not None and summaries[1] is not None \
            and len(summaries[0]) != len(summaries[1]):
        print("  [!] L and D have different numbers of stages")
    grid.pose_depth_edges = pose_depth_edges(grid.pose_depth_edges, summaries)
    print("  pose-map depth bins (A): "
          + ", ".join(f"{e:g}" for e in grid.pose_depth_edges) + ", open")

    if temperature is None and run_dirs:
        from .permeation_process import temperature_from_runs
        temperature = temperature_from_runs(run_dirs)
    if temperature is None:
        raise typer.BadParameter("No temperature found in the runs' tomls; pass -temperature.")
    print(f"  T = {temperature} K")
    compare(sides[0], sides[1], out_dir, grid, workers, temperature, n_boot, block, seed,
            stagesL=summaries[0], stagesD=summaries[1],
            factorsL=models[0], factorsD=models[1], by_class=by_class,
            split_half=split_half, unweighted=by_class)

    # The tables are the result; a figure that fails to draw must not lose
    # them.  `chiroflux chiral-plot` redraws from the tables alone.
    from .chiral_plot import plot_all, result_folders
    print("\n  Figures:")
    for folder in result_folders(out_dir):
        try:
            plot_all(folder)
        except Exception as exc:
            print(f"  [!] plotting failed ({type(exc).__name__}: {exc}); the tables "
                  f"are written, rerun with `chiroflux chiral-plot -in-dir {folder}`")
