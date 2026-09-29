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
            and the contact-induced shift     -> glycerol_conformers.csv

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

Pipeline
--------
  extract  Read each xtc exactly once (MDAnalysis, parallel, cached per file),
           one row of features per frame -> <cache>/<hash>.npz
  compare  Assemble every path from its cached frames and trim it.  Weight
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
          -permeant pro_yne -load-dir L_PRO_yne_neutral/$stage/load \\
          -cache L_PRO_yne_neutral/chiral_cache -workers 32
    done                                              (same for D)

    chiroflux chiral-compare \\
        -runs-l L_PRO_yne_neutral/infinit_entry,L_PRO_yne_neutral/infinit_internal,L_PRO_yne_neutral/infinit_escape \\
        -runs-d D_PRO_yne_neutral/infinit_entry,D_PRO_yne_neutral/infinit_internal,D_PRO_yne_neutral/infinit_escape \\
        -data-l post/infretis_data.txt -data-d post/infretis_data.txt \\
        -cache-l L_PRO_yne_neutral/chiral_cache -cache-d D_PRO_yne_neutral/chiral_cache \\
        -out-dir comparison/chiral_interface

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
* Path weights are WFtot, as in inftools: chiroflux's plus-ensemble WHAM
  weight (pathdata) plus the [0-] weight, with the first -nskip rows of
  infretis_data.txt skipped as burn-in.  ddG(depth) and the depth profiles
  are equilibrium quantities only with all stages and these weights.  With a
  single stage they are conditional on that run's start; with -no-reweight
  they only describe where the sampled paths spent their time.
* The bootstrap resamples paths within each stage.  The stage factors come
  from Pcross.txt and are held fixed, so their uncertainty is not in the
  intervals.
* Frames at OP <= lambda_A are scaled or cut frame by frame.  inftools and
  permeation_process do it bin by bin (on the bin centre), so the one bin
  that straddles lambda_A can differ slightly.
"""

from __future__ import annotations

import glob
import hashlib
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    traj_plan_segments,
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
# Extraction: one xtc -> one row of features per frame
# ══════════════════════════════════════════════════════════════════════════════

class _Topology:
    """Atom indices resolved once per worker from the tpr."""

    def __init__(self, tpr, spec: PermeantSpec):
        import MDAnalysis as mda
        warnings.filterwarnings("ignore", module="MDAnalysis")
        self.u = mda.Universe(tpr)
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


#: Bump when the cached content changes; old caches then miss and are rebuilt.
CACHE_VERSION = 3


def cache_name(xtc):
    st = os.stat(xtc)
    key = f"v{CACHE_VERSION}|{os.path.abspath(xtc)}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(key.encode()).hexdigest()[:20] + ".npz"


def extract_file(T: _Topology, xtc, out):
    """Features for every frame of one xtc -> npz at `out`.  Every frame is
    kept, including ones no path uses; paths select theirs at assembly."""
    T.u.load_new(xtc)
    n = T.u.trajectory.n_frames
    n_don = len(T.spec.donors)
    F = np.full((n, len(FLOAT_COLS)), np.nan, dtype=np.float32)
    I = np.full((n, 1 + 2 * n_don), -1, dtype=np.int32)
    # Bulk glycerol dihedral histogram per frame, so trimmed frames drop out
    # of the bulk reference too.
    G = np.zeros((n, 2, N_GLYC_BINS), dtype=np.uint16)
    edges = np.linspace(-180.0, 180.0, N_GLYC_BINS + 1)
    stereo = None
    for i, ts in enumerate(T.u.trajectory):
        pos = ts.positions
        if stereo is None:
            stereo = check_stereochemistry(T, pos, ts.dimensions)
        F[i], I[i], glyc = _frame_features(T, pos, ts.dimensions)
        for a in range(2):
            G[i, a] = np.histogram(glyc[a], bins=edges)[0]
    tmp = f"{out}.tmp.npz"
    np.savez_compressed(tmp, F=F, I=I, G=G, dt=T.u.trajectory.dt,
                        xtc=os.path.abspath(xtc), columns=np.array(FLOAT_COLS),
                        **{f"stereo_{k}": v for k, v in (stereo or {}).items()})
    os.replace(tmp, out)
    return n


_TOPO: _Topology | None = None


def _init_extract(tpr, permeant):
    global _TOPO
    _TOPO = _Topology(tpr, PERMEANTS[permeant])


def _extract_job(xtc, cache_dir, overwrite):
    out = os.path.join(cache_dir, cache_name(xtc))
    if os.path.exists(out) and not overwrite:
        return xtc, -1, None
    try:
        return xtc, extract_file(_TOPO, xtc, out), None
    except Exception as exc:
        return xtc, 0, f"{type(exc).__name__}: {exc}"


def extract_many(xtc_files, tpr, permeant, cache_dir, workers=8, overwrite=False):
    """Extract (and cache) every file; shared segments are only read once."""
    os.makedirs(cache_dir, exist_ok=True)
    files = sorted(set(xtc_files))
    done = skipped = 0
    errors = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_extract,
                             initargs=(tpr, permeant)) as ex:
        futs = [ex.submit(_extract_job, x, cache_dir, overwrite) for x in files]
        for k, fut in enumerate(as_completed(futs), 1):
            xtc, n, err = fut.result()
            if err:
                errors.append((xtc, err))
            elif n < 0:
                skipped += 1
            else:
                done += 1
            if k % 200 == 0:
                print(f"    ... {k}/{len(files)} files")
    print(f"  extracted {done}, cached already {skipped}, failed {len(errors)}")
    for xtc, err in errors[:10]:
        print(f"    [!] {xtc}: {err}")
    return errors


# ══════════════════════════════════════════════════════════════════════════════
# Units (paths) and per-unit reduction
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Unit:
    """
    One statistical unit.  `plan` lists (xtc path, frame index) per phase
    point in path order.  A frame index of None means every frame of that
    file.  Units with an `order_file` are infretis paths.

    Stage context, set by `setup_stages` (see "Path-ensemble reweighting"):
      weight    WFtot of the path times its stage factor
      trim      (lambda_A, lambda_minus_one) for end-point trimming, or None
      stage     index of the stage in the chain (0 = entry)
      lam_a     the stage's lambda_A on the order-parameter axis
      below_la  multiplier for frames with OP <= lam_a: 1/xi on stage 0, 0
                (the cut) on every later stage
    """
    uid: int | str
    plan: list
    weight: float = 1.0
    order_file: str | None = None
    g96_index: str | None = None
    trim: tuple | None = None
    stage: int = 0
    lam_a: float | None = None
    below_la: float = 1.0

    @property
    def files(self):
        return list(dict.fromkeys(f for f, _ in self.plan))


def infretis_path_unit(path_dir, uid):
    """A path as cv_generation reads it: the traj.txt plan, plus order.txt."""
    plan, g96_index = read_traj_plan(os.path.join(path_dir, "traj.txt"))
    acc = os.path.join(path_dir, "accepted")
    return Unit(uid, [(os.path.join(acc, f), i) for f, i in plan],
                order_file=os.path.join(path_dir, "order.txt"), g96_index=g96_index)


def units_from_load_dir(load_dir, path_numbers=None):
    if path_numbers is None:
        path_numbers = sorted(int(p.name) for p in Path(load_dir).iterdir()
                              if p.name.isdigit() and (p / "traj.txt").exists())
    return [infretis_path_unit(os.path.join(load_dir, str(pn)), pn)
            for pn in path_numbers]


def units_from_files(xtc_files):
    return [Unit(i, [(x, None)]) for i, x in enumerate(sorted(xtc_files))]


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
#      path's WFtot, the plus-ensemble WHAM weight plus the [0-] weight, each
#      part normalised to 1.  Frames at OP <= lambda_A come from the [0-]
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


def stage_path_weights(data_file, toml_file, nskip=1000):
    """{path: WFtot} = plus WHAM weight + [0-] weight, the per-path weight
    inftools writes to wham/path_weights.txt and histograms with."""
    out = dict(wham_weights(data_file, toml_file, nskip))
    for p, v in minus_weights(data_file, toml_file, nskip).items():
        out[p] = out.get(p, 0.0) + v
    return out


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
    (xi, number of weighted paths whose order.txt was missing).
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
        elif kind in ("LML", "RML"):
            l_end += w
    xi = r_end / (r_end + l_end) if r_end + l_end > 0 else np.nan
    return xi, missing


@dataclass
class StageSpec:
    run_dir: str
    data_file: str
    toml_file: str


def _find_toml(run_dir):
    for cand in ("infretis.toml", os.path.join("post", "infretis.toml")):
        path = os.path.join(run_dir, cand)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"no infretis.toml in {run_dir} or {run_dir}/post; "
                            f"give it with -toml-l/-toml-d")


def setup_stages(specs, nskip=1000, wham_subdir="wham", pcross_col=3,
                 trim=True, weights="wham", path_numbers=None):
    """
    Units of every stage of one enantiomer, weighted onto one equilibrium
    scale (see the section comment).  `specs` are in chain order, entry first.
    Returns (units, per-stage summary table).
    """
    from .permeation_process import load_stage, stage_factors

    stages = [load_stage(s.run_dir, wham_subdir, pcross_col) for s in specs]
    factors = stage_factors(stages) if len(stages) > 1 else [1.0]
    all_units, rows = [], []
    for k, (spec, stage, factor) in enumerate(zip(specs, stages, factors)):
        lam_a, lam_m1 = read_lambdas(spec.toml_file)
        if abs(lam_a - stage.lam_a) > 1e-6:
            raise ValueError(f"{spec.toml_file} has lambda_A = {lam_a} but "
                             f"{spec.run_dir}/{wham_subdir} was made with "
                             f"lA = {stage.lam_a}; wrong toml for this run?")
        units = units_from_load_dir(os.path.join(spec.run_dir, "load"), path_numbers)
        if weights == "wham":
            wtot = stage_path_weights(spec.data_file, spec.toml_file, nskip)
            xi, missing = (wham_xi(units, wtot, lam_a, lam_m1) if k == 0
                           else (np.nan, 0))
        else:
            wtot = {u.uid: 1.0 for u in units}
            xi, missing = 1.0, 0
        if k == 0 and not np.isfinite(xi):
            raise ValueError(f"xi undefined for {spec.run_dir}: no weighted [0-] "
                             f"path ends at lambda_A or lambda_-1")
        for u in units:
            u.weight = wtot.get(u.uid, 0.0) * factor
            u.trim = (lam_a, lam_m1) if trim else None
            u.stage, u.lam_a = k, lam_a
            u.below_la = 1.0 / xi if k == 0 else 0.0
        kept = [u for u in units if u.weight > 0]
        all_units += kept
        rows.append({"stage": k, "run": spec.run_dir, "lam_a": lam_a,
                     "lam_b": stage.lam_b, "pcross": stage.pcross,
                     "factor": factor, "xi": xi, "paths_in_load": len(units),
                     "paths_weighted": len(kept),
                     "weighted_without_order_txt": missing})
    return all_units, pd.DataFrame(rows)


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

    @property
    def n_depth(self):
        return len(self.depth_edges)   # last bin is open-ended bulk water

    def depth_bin(self, z):
        return np.clip(np.digitize(np.abs(z), self.depth_edges) - 1, 0, self.n_depth - 1)

    def depth_labels(self):
        e = self.depth_edges
        return [f"{a:.2f}-{b:.2f}" for a, b in zip(e[:-1], e[1:])] + [f">{e[-1]:.2f}"]


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
LIFE_KINDS = CLASS_NAMES + ["lipid_contact"]


def load_cached(xtc, cache_dir):
    path = os.path.join(cache_dir, cache_name(xtc))
    if not os.path.exists(path):
        raise FileNotFoundError(f"no v{CACHE_VERSION} cache for {xtc} in {cache_dir}; "
                                f"run `chiroflux chiral-extract` first (caches from older versions, "
                                f"or of a since-modified xtc, are not reused)")
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def _runs(keys):
    """(value, length) of every maximal run of equal consecutive entries."""
    if keys.size == 0:
        return keys, keys
    b = np.flatnonzero(np.diff(keys)) + 1
    starts = np.r_[0, b]
    return keys[starts], np.diff(np.r_[starts, keys.size])


def assemble_path(unit: Unit, cache_dir):
    """
    The frames of one unit as a single time-ordered sequence.

    Frames are taken exactly as `unit.plan` names them (reversed segments list
    descending indices, so they need no special handling).  For infretis
    paths the order parameter is then checked row by row against the plan,
    and, if `unit.trim` = (lambda_A, lambda_minus_one), the end points are
    trimmed, both as in cv_generation.

    Returns (arrays {F, I, G}, op, dt, meta).  `op` is the order parameter
    per kept frame; plain-MD units have none and get the stereocentre depth z
    in Angstrom instead.
    """
    parts = {"F": [], "I": [], "G": []}
    dt = None
    for fname, idx in traj_plan_segments(unit.plan):
        d = load_cached(fname, cache_dir)
        n = d["F"].shape[0]
        if idx[0] is None:
            sel = slice(None)
        else:
            sel = np.asarray(idx)
            if sel.min() < 0 or sel.max() >= n:
                raise ValueError(f"{fname}: traj.txt asks for frame {sel.max()}, "
                                 f"but the file has {n} frames")
        for k in parts:
            parts[k].append(d[k][sel])
        dt = float(d["dt"]) if dt is None else dt
    arr = {k: np.concatenate(v) for k, v in parts.items()}

    meta = {"stage": unit.stage, "uid": unit.uid, "phase_points": len(arr["F"]),
            "ensemble": None, "deleted_first": False, "deleted_last": False}
    if unit.order_file is None:
        return arr, arr["F"][:, FLOAT_COLS.index("z")].astype(float) * 10.0, dt, meta

    # The row-by-row check runs even without trimming: it is what guarantees
    # that frame i is phase point i.
    order = np.loadtxt(unit.order_file, comments=("#", "@"), ndmin=2)
    op_raw = order[:, 1]
    op = op_raw if unit.g96_index is None else np.delete(op_raw, int(unit.g96_index))
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


def frame_multiplier(op, unit: Unit):
    """Per-frame weight on top of the unit weight: `below_la` (1/xi on the
    first stage, 0 = cut on later ones) for frames at OP <= lambda_A."""
    if unit.lam_a is None:
        return np.ones(len(op))
    return np.where(op <= unit.lam_a, unit.below_la, 1.0)


def reduce_path(arr, op, dt, grid: Grid, mult=None):
    """
    Depth-resolved sums for one assembled path, binned on |OP| and weighted
    per frame by `mult`.  Every entry is additive over units, so any weighted
    average is a ratio of weighted sums of these.  Frames with zero weight
    are left out, and runs for the lifetimes break at them.
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
           "glyc_partner": np.zeros((2, N_GLYC_BINS)),
           "glyc_bulk": np.tensordot(mult, arr["G"].astype(float), axes=(0, 0)),
           "life_n": np.zeros(len(LIFE_KINDS)),
           "life_sum": np.zeros(len(LIFE_KINDS)),
           "life_hist": np.zeros((len(LIFE_KINDS), len(grid.life_edges) - 1)),
           "dt": dt}
    if len(F) == 0:
        return out

    zb = grid.depth_bin(op)
    live = mult > 0
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

    for k in range(n_don):
        cls = I[:, 1 + 2 * k].copy()
        cls[cls < 0] = len(CLASS_NAMES)          # last slot: no H-bond
        np.add.at(out["hb"][k], (cls[live], zb[live]), mult[live])
        np.add.at(out["hb_raw"][k], (cls[live], zb[live]), 1.0)

    px, py, pz = (F[contact, col[c]] for c in ("pose_x", "pose_y", "pose_z"))
    out["pose"] += np.histogram2d(np.degrees(np.arctan2(py, px)), pz,
                                  bins=(grid.n_phi, grid.n_cos),
                                  range=((-180, 180), (-1, 1)),
                                  weights=mult[contact])[0]
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


def reduce_unit(unit: Unit, cache_dir, grid: Grid):
    arr, op, dt, meta = assemble_path(unit, cache_dir)
    return reduce_path(arr, op, dt, grid, frame_multiplier(op, unit)), meta


_RED: dict = {}


def _init_reduce(cache_dir, grid):
    _RED.update(cache_dir=cache_dir, grid=grid)


def _reduce_job(item):
    i, unit = item
    try:
        red, meta = reduce_unit(unit, _RED["cache_dir"], _RED["grid"])
        return i, red, meta, None
    except Exception as exc:
        return i, None, None, f"{type(exc).__name__}: {exc}"


def reduce_units(units, cache_dir, grid, workers=8):
    """Reduce all units in parallel.  Returns (kept units in input order, their
    reductions, a per-unit trimming table).  Results are keyed by position:
    path numbers repeat across stages."""
    res, metas = {}, []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_reduce,
                             initargs=(cache_dir, grid)) as ex:
        for i, red, meta, err in ex.map(_reduce_job, enumerate(units), chunksize=16):
            if err:
                print(f"    [!] stage {units[i].stage} path {units[i].uid}: {err}")
                metas.append({"stage": units[i].stage, "uid": units[i].uid, "error": err})
            elif red is not None:
                res[i] = red
                metas.append(meta)
    order = sorted(res)
    return [units[i] for i in order], [res[i] for i in order], pd.DataFrame(metas)


def stereo_summary(units, cache_dir):
    """Per-file CIP check, so a mislabelled run or inverted lipids is caught
    before any comparison is trusted."""
    rows = []
    for f in sorted({f for u in units for f in u.files}):
        d = load_cached(f, cache_dir)
        rows.append({k[len("stereo_"):]: d[k].item() for k in d if k.startswith("stereo_")})
    return pd.DataFrame(rows)


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


class Ensemble:
    """Stacked per-unit reductions with weights, grouped by stage and in MC
    order within each stage."""

    def __init__(self, units, reds):
        self.w = np.array([u.weight for u in units], dtype=float)
        self.stage = np.array([u.stage for u in units])
        self.keys = [k for k in reds[0] if k != "dt"]
        self.A = {k: np.stack([r[k] for r in reds]) for k in self.keys}
        self.dt = reds[0]["dt"]
        self.n = len(units)

    def agg(self, idx=None):
        if idx is None:
            idx = np.arange(self.n)
        w = self.w[idx]
        return {k: np.tensordot(w, self.A[k][idx], axes=(0, 0)) for k in self.keys}

    def resample(self, rng, block):
        """One block-bootstrap replicate, drawn within each stage separately:
        the stages are independent simulations, and an MC block must not
        straddle two of them.  Each stage keeps its own path count."""
        out = []
        for s in np.unique(self.stage):
            members = np.flatnonzero(self.stage == s)
            out.append(members[block_bootstrap_indices(len(members), block, rng)])
        return np.concatenate(out)

    def block_starts(self, block):
        """Start index of every contiguous MC block, never crossing a stage."""
        starts = []
        for s in np.unique(self.stage):
            members = np.flatnonzero(self.stage == s)
            starts += list(members[::block])
        return np.array(sorted(starts))


def bootstrap(eL, eD, stat, n_boot=500, block=50, seed=0):
    """
    stat(agg) -> 1-D array.  Returns point estimates and replicates for L and D.
    Units are resampled in contiguous MC blocks within each stage, never
    frames: frames within a path are almost perfectly correlated.  The stage
    factors (from Pcross) are held fixed, so their own uncertainty is not in
    these intervals.
    """
    rng = np.random.default_rng(seed)
    pL, pD = stat(eL.agg()), stat(eD.agg())
    rL = np.empty((n_boot, pL.size))
    rD = np.empty((n_boot, pD.size))
    for b in range(n_boot):
        rL[b] = stat(eL.agg(eL.resample(rng, block)))
        rD[b] = stat(eD.agg(eD.resample(rng, block)))
    return pL, pD, rL, rD


def _diff_table(pL, pD, rL, rD, index, ci=0.95):
    # Empty bins are NaN by design; numpy's all-NaN warnings are just noise here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _diff_table_impl(pL, pD, rL, rD, index, ci)


def _diff_table_impl(pL, pD, rL, rD, index, ci):
    diff = rL - rD
    lo, hi = np.nanpercentile(diff, [50 * (1 - ci), 50 * (1 + ci)], axis=0)
    n = np.sum(np.isfinite(diff), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = 2.0 * np.minimum(np.sum(diff <= 0, axis=0), np.sum(diff >= 0, axis=0)) / n
    p = np.clip(np.where(n > 0, p, np.nan), 1.0 / max(len(diff), 1), 1.0)
    df = index.copy()
    df["L"], df["D"], df["L_minus_D"] = pL, pD, pL - pD
    df["ci_lo"], df["ci_hi"], df["p_boot"] = lo, hi, p
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
    """WHERE + HOW: every PROFILE_OBS as a depth profile, L vs D."""
    stat = lambda a: _ratio(a["S"], a["C"]).ravel()
    pL, pD, rL, rD = bootstrap(eL, eD, stat, **boot)
    names = [o[0] for o in PROFILE_OBS]
    idx = pd.DataFrame([(n, lab) for n in names for lab in grid.depth_labels()],
                       columns=["observable", "depth"])
    df = _diff_table(pL, pD, rL, rD, idx)
    # Spread of each observable within a bin (weighted SD), point estimates:
    # a shift in the mean with an unchanged SD reads differently from a change
    # in shape.
    for side, e in (("L", eL), ("D", eD)):
        a = e.agg()
        mean = _ratio(a["S"], a["C"])
        with np.errstate(invalid="ignore"):
            df[f"sd_{side}"] = np.sqrt(np.clip(_ratio(a["S2"], a["C"]) - mean ** 2,
                                               0, None)).ravel()
    df["frames_L"] = eL.A["C_raw"].sum(axis=0).ravel()
    df["frames_D"] = eD.A["C_raw"].sum(axis=0).ravel()
    sparse = (df["frames_L"] < min_frames) | (df["frames_D"] < min_frames)
    df.loc[sparse, ["p_boot", "q_bh"]] = np.nan
    return df


def _reference_bin(eL, eD, min_frames):
    """Depth bins both ensembles sample with >= min_frames frames, and the
    outermost of them: the bulk-water anchor for every free energy here."""
    ok = ((eL.A["N_raw"].sum(axis=0) >= min_frames)
          & (eD.A["N_raw"].sum(axis=0) >= min_frames))
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
    dense = ((eL.A["hb_raw"].sum(axis=0) >= min_frames)
             & (eD.A["hb_raw"].sum(axis=0) >= min_frames) & ok[None, None, :])

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
    rng = np.random.default_rng(boot.get("seed", 0))
    n_boot, block = boot.get("n_boot", 500), boot.get("block", 50)
    pL, pD = stat_L(eL.agg()), stat_D(eD.agg())
    rL = np.array([stat_L(eL.agg(eL.resample(rng, block))) for _ in range(n_boot)])
    rD = np.array([stat_D(eD.agg(eD.resample(rng, block))) for _ in range(n_boot)])
    df = _diff_table(pL, pD, rL, rD, pd.DataFrame({"kind": LIFE_KINDS}))
    df = df.rename(columns={"L": "tau_L_ps", "D": "tau_D_ps", "L_minus_D": "dtau_ps"})
    df["runs_L"] = eL.A["life_n"].sum(axis=0)
    df["runs_D"] = eD.A["life_n"].sum(axis=0)
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


def _blocks(e, key, block):
    """Weighted `key` summed over contiguous MC blocks of units, with weights
    rescaled to mean 1 so two ensembles with differently normalised infretis
    weights pool on equal footing."""
    a = e.A[key]
    wa = a * (e.w / e.w.mean()).reshape((-1,) + (1,) * (a.ndim - 1))
    starts = e.block_starts(block)
    return np.add.reduceat(wa, starts, axis=0)


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


def _report_trimming(label, meta):
    if meta.empty or "ensemble" not in meta or meta["ensemble"].isna().all():
        print(f"  {label}: no trimming applied")
        return
    for stage, ok in meta[meta["ensemble"].notna()].groupby("stage"):
        ens = ok["ensemble"].value_counts().to_dict()
        print(f"  {label} stage {stage}: {len(ok)} paths trimmed ({ens}); first point "
              f"deleted in {int(ok['deleted_first'].sum())}, last in "
              f"{int(ok['deleted_last'].sum())}")


def compare(unitsL, unitsD, cacheL, cacheD, out_dir, grid=None, workers=8,
            temperature=310.0, n_boot=500, block=50, seed=0,
            stagesL=None, stagesD=None):
    """
    Run every comparison and write the tables to out_dir.  The units carry
    their own weights, stage and trimming context (see `setup_stages`);
    stagesL/stagesD are the per-stage summaries, written alongside.
    """
    grid = grid or Grid()
    os.makedirs(out_dir, exist_ok=True)
    for label, st in (("L", stagesL), ("D", stagesD)):
        if st is not None:
            st.to_csv(os.path.join(out_dir, f"stages_{label}.csv"), index=False)

    for label, units, cache, want in (("L", unitsL, cacheL, "L"), ("D", unitsD, cacheD, "D")):
        st = stereo_summary(units, cache)
        st.to_csv(os.path.join(out_dir, f"stereochemistry_{label}.csv"), index=False)
        bad = st[st["enantiomer"] != want]
        mixed = st[(st["lipids_R"] > 0) & (st["lipids_S"] > 0)]
        print(f"  {label}: {len(st)} files, permeant {st['enantiomer'].value_counts().to_dict()}, "
              f"lipids R/S = {st['lipids_R'].iloc[0]}/{st['lipids_S'].iloc[0]}")
        if len(bad):
            print(f"  [!] {len(bad)} {label} file(s) contain the OTHER enantiomer")
        if len(mixed):
            print(f"  [!] {len(mixed)} {label} file(s) have lipids of both handedness")

    uL, rL, mL = reduce_units(unitsL, cacheL, grid, workers)
    uD, rD, mD = reduce_units(unitsD, cacheD, grid, workers)
    for label, meta in (("L", mL), ("D", mD)):
        meta.to_csv(os.path.join(out_dir, f"path_trimming_{label}.csv"), index=False)
        _report_trimming(label, meta)
    eL, eD = Ensemble(uL, rL), Ensemble(uD, rD)
    print(f"  units: L {eL.n}, D {eD.n}")
    boot = dict(n_boot=n_boot, block=block, seed=seed)

    perm_summary, perm_shares = permeability_ratio(eL, eD, grid, **boot)
    tables = {
        "depth_profiles": depth_profiles(eL, eD, grid, **boot),
        "ddG_depth": depth_ddg(eL, eD, grid, temperature, **boot),
        "permeability_ratio": perm_summary,
        "resistance_shares": perm_shares,
        "ddG_by_hbond_state": state_resolved_ddg(eL, eD, grid, temperature, **boot),
        "hbond_partners": hbond_partners(eL, eD, grid, **boot),
        "sn2_preference": sn2_preference(eL, eD, **boot),
        "lifetimes": lifetimes(eL, eD, **boot),
        "lifetime_survival": lifetime_survival(eL, eD, grid, **boot),
        "glycerol_conformers": glycerol_conformers(eL, eD, **boot),
    }
    maps, pose_jsd = pose_maps(eL, eD, block=block, seed=seed)
    tables["pose_jsd"] = pose_jsd
    np.savez(os.path.join(out_dir, "pose_maps.npz"), **maps,
             phi_edges=np.linspace(-180, 180, grid.n_phi + 1),
             cos_edges=np.linspace(-1, 1, grid.n_cos + 1))
    for name, df in tables.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)

    print("\n  Rows with q_bh < 0.05:")
    for name, df in tables.items():
        if "q_bh" in df:
            hits = df[df["q_bh"] < 0.05]
            print(f"    {name:20s} {len(hits):4d} / {df['q_bh'].notna().sum()}")
    if not perm_summary.empty:
        r = perm_summary.iloc[0]
        print(f"\n  ln(P_L/P_D) from the free-energy profiles alone (equal D): "
              f"{r.ln_PL_over_PD:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  "
              f"-> P_L/P_D = {r.PL_over_PD:.3g}"
              + ("" if r.complete else "   [!] profile incomplete: an unsampled "
                 "membrane bin is missing from the integral"))
    print("\n" + pose_jsd.to_string(index=False))
    print("\n" + tables["sn2_preference"].to_string(index=False))
    return tables


#: Every file `compare` writes, checked up front so -O is needed to replace them.
COMPARE_OUTPUTS = (
    [f"{stem}_{side}.csv" for stem in ("stages", "stereochemistry", "path_trimming")
     for side in ("L", "D")]
    + [f"{name}.csv" for name in (
        "depth_profiles", "ddG_depth", "permeability_ratio", "resistance_shares",
        "ddG_by_hbond_state", "hbond_partners", "sn2_preference", "lifetimes",
        "lifetime_survival", "glycerol_conformers", "pose_jsd")]
    + ["pose_maps.npz"]
)


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


def _per_run(value, run_dirs, flag):
    """
    One file per run directory from `value`: a single path looked for inside
    every run directory, or one path per run, comma-separated, in run order.
    Each path is tried inside its run directory first, then as given.
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
            if os.path.isfile(cand):
                out.append(cand)
                break
        else:
            raise typer.BadParameter(
                f"{flag}: neither {os.path.join(run_dir, item)} nor {item} exists")
    return out


def _stage_specs(runs, data, toml, side):
    run_dirs = _split(runs)
    data_files = _per_run(data, run_dirs, f"-data-{side}")
    tomls = (_per_run(toml, run_dirs, f"-toml-{side}") if toml
             else [_find_toml(r) for r in run_dirs])
    return [StageSpec(r, d, t) for r, d, t in zip(run_dirs, data_files, tomls)]


def chiral_extract(
    # ── Input data ────────────────────────────────────────────────────────
    tpr: Annotated[Optional[str], typer.Option("-tpr", help="Topology (.tpr) of the simulated system; atom names and residues are resolved from it once per worker. Required", rich_help_panel=panels.INPUT)] = None,
    permeant: Annotated[Optional[str], typer.Option("-permeant", help=f"Which permeant definition to use (stereocentre, CIP substituents, H-bond donors/acceptors); one of: {', '.join(sorted(PERMEANTS))}. Required", rich_help_panel=panels.INPUT)] = None,
    load_dir: Annotated[Optional[str], typer.Option("-load-dir", help="infretis load/ folder: every path in it is read through its traj.txt plan. Give this or -xtc", rich_help_panel=panels.INPUT)] = None,
    xtc: Annotated[Optional[str], typer.Option("-xtc", help="Plain MD / umbrella windows instead of -load-dir: a glob of .xtc files (quote it)", rich_help_panel=panels.INPUT)] = None,

    # ── Dataset construction ──────────────────────────────────────────────
    path_numbers: Annotated[Optional[str], typer.Option("-path-numbers", help="Only these paths of -load-dir, e.g. '1-5000,6001'; default all", rich_help_panel=panels.DATASET)] = None,

    # ── Model and training ────────────────────────────────────────────────
    workers: Annotated[int, typer.Option("-workers", help="Worker processes; each reads one xtc at a time", rich_help_panel=panels.MODEL)] = 8,

    # ── Output ────────────────────────────────────────────────────────────
    cache: Annotated[Optional[str], typer.Option("-cache", help="Cache folder for the per-xtc feature files. Share one per enantiomer across its stages: segments shared between paths and stages are then read only once. Required", rich_help_panel=panels.OUTPUT)] = None,
    overw: Annotated[bool, typer.Option("-O", help="Re-extract files that already have a cache entry", rich_help_panel=panels.OUTPUT)] = False,
):
    """Read every xtc of a run once and cache per-frame features for chiral-compare.

    Each frame gets the permeant's depth, orientation, pose in the body frame
    of the nearest lipid stereocentre, H-bond partners per donor, and the
    glycerol dihedrals of the contacted lipid. Caches are keyed on the xtc's
    path, size and mtime, so a modified xtc is re-extracted automatically.
    Run it once per stage (entry, internal, escape) of each enantiomer.
    """
    if tpr is None or permeant is None or cache is None:
        raise typer.BadParameter("-tpr, -permeant and -cache are required.")
    if permeant not in PERMEANTS:
        raise typer.BadParameter(
            f"-permeant {permeant!r}: choose from {sorted(PERMEANTS)}.")
    if (load_dir is None) == (xtc is None):
        raise typer.BadParameter("Give exactly one of -load-dir and -xtc.")
    units = _units(load_dir, xtc, path_numbers)
    files = [f for u in units for f in u.files]
    print(f"  {len(units)} units, {len(set(files))} unique xtc files")
    errors = extract_many(files, tpr, permeant, cache, workers, overw)
    if errors:
        raise typer.Exit(code=1)


def chiral_compare(
    # ── Input data ────────────────────────────────────────────────────────
    runs_l: Annotated[Optional[str], typer.Option("-runs-l", help="Run directories of the L simulation, comma-separated, in chain order (entry, internal, escape). Each holds load/ and the wham/ folder inftools wrote. Give this or -xtc-l", rich_help_panel=panels.INPUT)] = None,
    runs_d: Annotated[Optional[str], typer.Option("-runs-d", help="Run directories of the D simulation; see -runs-l", rich_help_panel=panels.INPUT)] = None,
    data_l: Annotated[str, typer.Option("-data-l", help="infretis_data.txt of each L stage: one relative path looked for inside every run directory, or one file per run, comma-separated", rich_help_panel=panels.INPUT)] = "infretis_data.txt",
    data_d: Annotated[str, typer.Option("-data-d", help="infretis_data.txt of each D stage; see -data-l", rich_help_panel=panels.INPUT)] = "infretis_data.txt",
    toml_l: Annotated[Optional[str], typer.Option("-toml-l", help="infretis.toml of each L stage, same rules as -data-l; unset uses RUN_DIR/infretis.toml, else RUN_DIR/post/infretis.toml", rich_help_panel=panels.INPUT)] = None,
    toml_d: Annotated[Optional[str], typer.Option("-toml-d", help="infretis.toml of each D stage; see -toml-l", rich_help_panel=panels.INPUT)] = None,
    xtc_l: Annotated[Optional[str], typer.Option("-xtc-l", help="Plain MD instead of -runs-l: a glob of .xtc files (quote it); each file is its own unweighted, untrimmed unit", rich_help_panel=panels.INPUT)] = None,
    xtc_d: Annotated[Optional[str], typer.Option("-xtc-d", help="Plain MD instead of -runs-d; see -xtc-l", rich_help_panel=panels.INPUT)] = None,
    cache_l: Annotated[Optional[str], typer.Option("-cache-l", help="Cache folder chiral-extract wrote for L. Required", rich_help_panel=panels.INPUT)] = None,
    cache_d: Annotated[Optional[str], typer.Option("-cache-d", help="Cache folder chiral-extract wrote for D. Required", rich_help_panel=panels.INPUT)] = None,

    # ── Dataset construction ──────────────────────────────────────────────
    path_numbers: Annotated[Optional[str], typer.Option("-path-numbers", help="Only these path numbers of every stage, e.g. '1-5000,6001'; default all", rich_help_panel=panels.DATASET)] = None,
    trim: Annotated[bool, typer.Option("-trim/-no-trim", help="Delete the first/last phase point by the plus/minus ensemble criteria, as generate-cvs does; disable to keep every phase point", rich_help_panel=panels.DATASET)] = True,
    reweight: Annotated[bool, typer.Option("-reweight/-no-reweight", help="Weight every path by WFtot (plus-ensemble WHAM weight + [0-] weight) times its stage factor, which combines the stages into one equilibrium ensemble; disable to give every path weight 1, in which case nothing is an equilibrium average", rich_help_panel=panels.DATASET)] = True,
    nskip: Annotated[int, typer.Option("-nskip", help="Burn-in rows of each infretis_data.txt skipped for the WHAM weights", rich_help_panel=panels.DATASET)] = 1000,
    wham_subdir: Annotated[str, typer.Option("-wham-subdir", help="Name of the wham folder inside each run directory", rich_help_panel=panels.DATASET)] = "wham",
    pcross_col: Annotated[int, typer.Option("-pcross-col", help="Column of wham/Pcross.txt holding the total crossing probability, as in permeation: 3 is P-wham2, 1 is P-wham", rich_help_panel=panels.DATASET)] = 3,
    temperature: Annotated[Optional[float], typer.Option("-temperature", help="Temperature in K for the free energies; unset reads [engine] temperature from the runs' tomls", rich_help_panel=panels.DATASET)] = None,
    depth_max: Annotated[float, typer.Option("-depth-max", help="Last depth-bin edge on |OP|, in OP units; beyond it is one open bulk-water bin", rich_help_panel=panels.DATASET)] = 35.0,
    depth_step: Annotated[float, typer.Option("-depth-step", help="Depth-bin width on |OP|, in OP units", rich_help_panel=panels.DATASET)] = 2.5,
    contact_cut: Annotated[float, typer.Option("-contact-cut", help="Permeant stereocentre to lipid C2 distance (nm) that counts as contact with a lipid stereocentre", rich_help_panel=panels.DATASET)] = 0.7,

    # ── Model and training ────────────────────────────────────────────────
    n_boot: Annotated[int, typer.Option("-n-boot", help="Block-bootstrap replicates for every confidence interval and p-value (the pose-map permutation test always uses 1000 permutations)", rich_help_panel=panels.MODEL)] = 500,
    block: Annotated[int, typer.Option("-block", help="Paths per contiguous MC block in the bootstrap and permutation test; capped at half a stage's path count", rich_help_panel=panels.MODEL)] = 50,
    seed: Annotated[int, typer.Option("-seed", help="Random seed for the bootstrap and permutation test", rich_help_panel=panels.MODEL)] = 0,
    workers: Annotated[int, typer.Option("-workers", help="Worker processes for assembling and reducing the paths", rich_help_panel=panels.MODEL)] = 8,

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
    conditional on that run's start. See the module docstring for details.
    """
    if cache_l is None or cache_d is None:
        raise typer.BadParameter("-cache-l and -cache-d are required.")
    for side, runs, xtc in (("l", runs_l, xtc_l), ("d", runs_d, xtc_d)):
        if (runs is None) == (xtc is None):
            raise typer.BadParameter(f"Give exactly one of -runs-{side} and -xtc-{side}.")
    for name in COMPARE_OUTPUTS:
        _check_overwrite(os.path.join(out_dir, name), overw)

    grid = Grid(depth_edges=np.arange(0.0, depth_max + 1e-9, depth_step),
                contact_cut=contact_cut)
    sides, summaries, run_dirs = [], [], []
    for side, runs, data, toml, xtc in (("l", runs_l, data_l, toml_l, xtc_l),
                                        ("d", runs_d, data_d, toml_d, xtc_d)):
        if runs is None:
            sides.append(units_from_files(glob.glob(xtc)))
            summaries.append(None)
            continue
        specs = _stage_specs(runs, data, toml, side)
        run_dirs += [sp.run_dir for sp in specs]
        units, summary = setup_stages(specs, nskip, wham_subdir, pcross_col,
                                      trim=trim,
                                      weights="wham" if reweight else "uniform",
                                      path_numbers=_parse_paths(path_numbers))
        print(f"  {side.upper()}:\n" + summary.to_string(index=False))
        if len(specs) == 1:
            print(f"  [!] {side.upper()} has a single stage.  Its histogram is conditional "
                  f"on starting in that run's state A, so ddG and the depth "
                  f"profiles are conditional too; pass entry, internal and escape.")
        if not reweight:
            print(f"  [!] {side.upper()}: uniform weights, so nothing here is an "
                  f"equilibrium average")
        sides.append(units)
        summaries.append(summary)
    if summaries[0] is not None and summaries[1] is not None \
            and len(summaries[0]) != len(summaries[1]):
        print("  [!] L and D have different numbers of stages")

    if temperature is None and run_dirs:
        from .permeation_process import temperature_from_runs
        temperature = temperature_from_runs(run_dirs)
    if temperature is None:
        raise typer.BadParameter("No temperature found in the runs' tomls; pass -temperature.")
    print(f"  T = {temperature} K")
    compare(sides[0], sides[1], cache_l, cache_d, out_dir, grid, workers,
            temperature, n_boot, block, seed,
            stagesL=summaries[0], stagesD=summaries[1])
