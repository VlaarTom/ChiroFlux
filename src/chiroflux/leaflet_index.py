"""Build the per-leaflet CN_*.ndx index files by lipid, not by atom height.

The index files these CVs read are normally made with a `gmx select` expression
of the form ``resname DOPC and name C210 and z > 4.06``, which classifies
*atoms* by their own z. That is exact for the headgroup and glycerol markers,
whose z distributions do not overlap between leaflets - C2, P, N, O22 and O32
all split 55/55 in DOPC and 11/11 in POPC. It is wrong for the deep chain
carbons: C210 splits 54/56 and C310 51/59, because C10 sits near the bilayer
midplane and chains from opposing leaflets interdigitate through the cutoff.
DOPC's oleoyl sn-3 chain (18:1, long and kinked) misassigns four carbons where
POPC's palmitoyl (16:0) misassigns none.

Nothing is flip-flopping - a translocated lipid would carry its whole headgroup
across and every marker would be off together. It is simply that "which side of
a plane is this atom on" is not the same question as "which leaflet does this
atom's lipid belong to".

This module asks the second question. Each *residue* is assigned to a leaflet
once, from the height of its own headgroup phosphorus, and then every marker
atom of that residue is written to that leaflet's group. Interdigitation stops
mattering, because a chain carbon never votes on its own leaflet. The counts
come out equal by construction, which is the check to run afterwards.

The group order written here matches the files the CV generator expects
positionally (``CN_DOPC_GROUPS[i]``), so the output is a drop-in replacement:
five single-atom permeant groups, then each lipid marker as an upper/lower
pair, in ``-lipid-atoms`` order.

Depends on MDAnalysis, imported lazily so ``chiroflux --help`` stays cheap.
"""

from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import typer

from . import panels
from .pathdata import _check_overwrite

_MDANALYSIS_HINT = (
    "chiroflux leaflet-index needs MDAnalysis, which is a required dependency "
    "of chiroflux - if it is missing, the install is incomplete. Reinstall "
    "with:\n\n    pip install -e .\n"
)

#: Marker atoms written as upper/lower pairs, in the order the CV generator
#: reads them positionally: groups 5/6, 7/8, 9/10, 11/12, 13/14, 15/16, 17/18.
DEFAULT_LIPID_ATOMS = "C2,P,N,C210,C310,O22,O32"

#: Single-atom permeant groups, written first as groups 0-4.
DEFAULT_PERMEANT_ATOMS = "CA,O01,N,HA,CD"


def _require_mdanalysis():
    """Import MDAnalysis on demand, with an actionable error if it is absent."""
    try:
        import MDAnalysis as mda
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(f"{exc}\n\n{_MDANALYSIS_HINT}") from exc
    return mda


def _read_g96_positions(path, n_atoms):
    """Positions from a GROMOS-96 .g96 POSITION block, converted to Angstrom.

    MDAnalysis has no .g96 reader, and these systems ship `conf.g96` rather
    than a .gro, so the alternative would be a `gmx editconf` round trip before
    every run. Only the coordinates are needed here - the topology supplies
    names and residues - so the POSITION block is enough.

    Both `POSITION` (with residue/atom labels) and `POSITIONRED` (bare
    coordinates) are accepted; in each the last three fields of a line are the
    x, y, z in nm, and the block closes with END.
    """
    positions = []
    in_block = False
    with open(path) as f:
        for line in f:
            token = line.strip()
            if not in_block:
                if token in ("POSITION", "POSITIONRED"):
                    in_block = True
                continue
            if token == "END":
                break
            if not token or token.startswith("#"):
                continue
            fields = token.split()
            if len(fields) < 3:
                continue
            positions.append([float(v) for v in fields[-3:]])

    if not positions:
        raise ValueError(f"No POSITION block found in {path}.")
    if len(positions) != n_atoms:
        raise ValueError(
            f"{path} holds {len(positions)} positions but the topology has "
            f"{n_atoms} atoms. They must describe the same system."
        )
    # .g96 is in nm; MDAnalysis works in Angstrom.
    return np.asarray(positions, dtype=np.float32) * 10.0


def _split_list(value):
    """Split a comma- or space-separated option into a list of tokens."""
    if not value:
        return []
    return [tok for tok in value.replace(",", " ").split() if tok]


def _shared_midplane(universe, resnames, leaflet_atom):
    """One dividing height for the whole membrane, from every species at once.

    Computing a midplane per species would give each its own boundary - 38.69 A
    from 110 DOPC against 39.16 A from 22 POPC on the reference system - so the
    same bilayer would be cut in two slightly different places, and the minor
    species' plane would be the noisier of the two. The leaflet boundary is a
    property of the membrane, not of a species.
    """
    sel = universe.select_atoms(
        f"resname {' '.join(resnames)} and name {leaflet_atom}"
    )
    if len(sel) == 0:
        raise ValueError(
            f"No atoms matched 'resname {' '.join(resnames)} and name "
            f"{leaflet_atom}'. Check -resname and -leaflet-atom."
        )
    return float(sel.positions[:, 2].mean()), len(sel)


def _assign_leaflets(universe, resname, leaflet_atom, midplane=None):
    """Map each lipid residue to a leaflet from its own headgroup height.

    Returns
    -------
    by_resindex : dict resindex -> True (upper) / False (lower)
    midplane    : the dividing height actually used, in the topology's units
                  (Angstrom for everything MDAnalysis reads)
    """
    sel = universe.select_atoms(f"resname {resname} and name {leaflet_atom}")
    if len(sel) == 0:
        raise ValueError(
            f"No atoms matched 'resname {resname} and name {leaflet_atom}'. "
            f"Check -resname and -leaflet-atom against the topology."
        )
    if len(sel) != len(sel.residues):
        raise ValueError(
            f"'name {leaflet_atom}' matched {len(sel)} atoms across "
            f"{len(sel.residues)} {resname} residues. The leaflet atom must be "
            "unique per residue, or the assignment is ambiguous."
        )

    z = sel.positions[:, 2].astype(float)
    if midplane is None:
        # A symmetric bilayer centred in the box: the mean of the two headgroup
        # planes. Reported below so a wrapped or asymmetric system is visible
        # as a lopsided split rather than passing silently.
        midplane = float(z.mean())

    return {int(a.resindex): bool(zi > midplane)
            for a, zi in zip(sel.atoms, z)}, midplane


def _marker_groups(universe, resname, atom_name, by_resindex):
    """(upper, lower) 1-based atom indices for one marker, split by residue."""
    sel = universe.select_atoms(f"resname {resname} and name {atom_name}")
    upper, lower, orphan = [], [], 0
    for atom in sel.atoms:
        leaflet = by_resindex.get(int(atom.resindex))
        if leaflet is None:
            orphan += 1
            continue
        (upper if leaflet else lower).append(int(atom.index) + 1)
    return sorted(upper), sorted(lower), orphan


def _write_ndx(path, groups):
    """Write [name] sections with 15 one-based indices per line, gmx style."""
    with open(path, "w") as f:
        for name, idx in groups:
            f.write(f"[ {name} ]\n")
            for start in range(0, len(idx), 15):
                f.write(" ".join(f"{i:5d}" for i in idx[start:start + 15]) + "\n")
            f.write("\n")


def leaflet_index(
    topology: Annotated[str, typer.Option("-topology", help="Topology with atom names: a .tpr, or a .gro/.pdb", rich_help_panel=panels.INPUT)] = "gromacs_input/topol.tpr",
    coords: Annotated[Optional[str], typer.Option("-coords", help="Coordinate frame to read positions from, when -topology carries none (e.g. conf.gro)", rich_help_panel=panels.INPUT)] = None,
    resname: Annotated[str, typer.Option("-resname", help="Lipid residue name(s) to index, comma- or space-separated: 'DOPC' or 'DOPC,POPC'. One file is written per species", rich_help_panel=panels.INPUT)] = "DOPC",
    permeant_resname: Annotated[str, typer.Option("-permeant-resname", help="Residue name of the permeant", rich_help_panel=panels.INPUT)] = "ORP",
    leaflet_atom: Annotated[str, typer.Option("-leaflet-atom", help="Atom whose height decides the whole residue's leaflet. Must be unique per residue and well separated between leaflets - the headgroup phosphorus", rich_help_panel=panels.DATASET)] = "P",
    midplane: Annotated[Optional[float], typer.Option("-midplane", help="Dividing height in Angstrom; default is the mean headgroup height. Set it explicitly if the membrane straddles the periodic boundary", rich_help_panel=panels.DATASET)] = None,
    lipid_atoms: Annotated[str, typer.Option("-lipid-atoms", help="Comma-separated marker atoms, written as upper/lower pairs in this order", rich_help_panel=panels.SELECT)] = DEFAULT_LIPID_ATOMS,
    permeant_atoms: Annotated[str, typer.Option("-permeant-atoms", help="Comma-separated permeant atoms, written first as one group each", rich_help_panel=panels.SELECT)] = DEFAULT_PERMEANT_ATOMS,
    out: Annotated[Optional[str], typer.Option("-out", help="Output .ndx path(s), one per -resname entry, comma- or space-separated. Omit to write CN_<RESNAME>.ndx into -out-dir", rich_help_panel=panels.OUTPUT)] = None,
    out_dir: Annotated[str, typer.Option("-out-dir", help="Directory for the default CN_<RESNAME>.ndx names, when -out is not given", rich_help_panel=panels.OUTPUT)] = ".",
    overw: Annotated[bool, typer.Option("-O", help="Force overwriting of existing files", rich_help_panel=panels.OUTPUT)] = False,
):
    """Write a CN_*.ndx whose leaflet groups are split per lipid, not per atom.

    A `gmx select` z cutoff classifies atoms, so deep chain carbons whose
    chains interdigitate through the midplane land in the wrong leaflet group
    (DOPC C210 54/56, C310 51/59, against 55/55 for every headgroup marker).
    This assigns each residue once from its own phosphorus and writes all of
    that residue's markers to the matching group, so every group comes out the
    same size — which is the check to run on the summary it prints.

    The group order matches what `generate-cvs` reads positionally, so the
    output drops straight into `gromacs_input/`:

      0-4    permeant, one group per -permeant-atoms entry
      5,6    first -lipid-atoms entry, upper then lower
      7,8    second entry, and so on

    Several species can be done at once, one file each, and the midplane is
    then computed once over all of them rather than per species:

      chiroflux leaflet-index -resname DOPC,POPC -out-dir gromacs_input
      chiroflux leaflet-index -resname DOPC,POPC -out CN_DOPC.ndx,CN_POPC.ndx

    With -out omitted each species is written to CN_<RESNAME>.ndx under
    -out-dir; given, it must list one path per species in the same order.

    Build both simulations' index files this way. Two runs whose index files
    were made independently by a z cutoff misassign *different* chain carbons,
    which puts a small systematic difference into the CC2/CC3 coordination
    numbers that has nothing to do with chirality.
    """
    mda = _require_mdanalysis()

    resnames = _split_list(resname)
    if not resnames:
        raise ValueError("-resname is empty.")
    out_paths = _split_list(out)
    if out_paths and len(out_paths) != len(resnames):
        raise ValueError(
            f"-out lists {len(out_paths)} path(s) for {len(resnames)} species "
            f"({', '.join(resnames)}). Give one per species, in the same "
            "order, or omit -out to use CN_<RESNAME>.ndx under -out-dir."
        )
    if not out_paths:
        out_paths = [str(Path(out_dir) / f"CN_{r}.ndx") for r in resnames]
    for path in out_paths:
        _check_overwrite(path, overw)

    if coords and Path(coords).suffix.lower() == ".g96":
        universe = mda.Universe(topology)
        universe.load_new(
            _read_g96_positions(coords, len(universe.atoms))[np.newaxis, :, :],
            format="memory",
        )
    else:
        universe = mda.Universe(topology, coords) if coords else mda.Universe(topology)

    try:
        universe.atoms.positions
    except Exception as exc:
        raise ValueError(
            f"{topology} carries no coordinates ({exc}). Pass a frame with "
            "-coords, e.g. -coords gromacs_input/conf.g96."
        ) from exc

    if midplane is None:
        used_midplane, n_head = _shared_midplane(universe, resnames, leaflet_atom)
        print(f"Midplane z = {used_midplane:.2f} A, from {n_head} "
              f"'{leaflet_atom}' atoms across {', '.join(resnames)}.")
    else:
        used_midplane = midplane
        print(f"Midplane z = {used_midplane:.2f} A (given).")

    permeant_names = [a.strip() for a in permeant_atoms.split(",") if a.strip()]
    marker_names = [a.strip() for a in lipid_atoms.split(",") if a.strip()]

    permeant_groups = []
    for name in permeant_names:
        sel = universe.select_atoms(f"resname {permeant_resname} and name {name}")
        if len(sel) == 0:
            raise ValueError(
                f"No atoms matched 'resname {permeant_resname} and name {name}'."
            )
        permeant_groups.append((f"resname_{permeant_resname}_and_name_{name}",
                                sorted(int(i) + 1 for i in sel.indices)))

    all_balanced = True
    for species, path in zip(resnames, out_paths):
        by_resindex, _ = _assign_leaflets(
            universe, species, leaflet_atom, used_midplane
        )
        n_upper = sum(by_resindex.values())
        n_lower = len(by_resindex) - n_upper

        print(f"\n{len(by_resindex)} {species} residues: "
              f"{n_upper} upper, {n_lower} lower.")
        if n_upper == 0 or n_lower == 0:
            raise ValueError(
                f"Every {species} residue landed in one leaflet. The membrane "
                "is probably wrapped across the periodic boundary - centre it "
                "first, or set -midplane explicitly."
            )
        if abs(n_upper - n_lower) > 0.1 * len(by_resindex):
            print(f"  [warn] leaflets differ by {abs(n_upper - n_lower)} residues "
                  f"(>10%). Check that the membrane is centred in the box.")

        groups = list(permeant_groups)
        print(f"  {'marker':<8} {'upper':>6} {'lower':>6}")
        for name in marker_names:
            upper, lower, orphan = _marker_groups(
                universe, species, name, by_resindex
            )
            if orphan:
                print(f"  [warn] {orphan} {species} {name} atoms belong to a "
                      f"residue with no '{leaflet_atom}' and were dropped.")
            if len(upper) != n_upper or len(lower) != n_lower:
                all_balanced = False
            print(f"  {name:<8} {len(upper):>6} {len(lower):>6}")
            groups.append((f"resname_{species}_and_name_{name}_upper", upper))
            groups.append((f"resname_{species}_and_name_{name}_lower", lower))

        _write_ndx(path, groups)
        print(f"  -> {len(groups)} groups written to {path}.")

    if all_balanced:
        print("\nEvery marker splits like its headgroup, as it should.")
    else:
        print("\n[warn] a marker did not match the headgroup split — some "
              "residues are missing that atom. Check -lipid-atoms against the "
              "topology.")
