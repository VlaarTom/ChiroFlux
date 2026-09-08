"""Build the permeant-side .ndx files, with the atom selections checked.

These are the index files that used to come from a hand-maintained
`make_ndx.py` copied into each simulation directory. Copies drift, and the
copies in this project had: the L run selected ``C, O, H02`` for
``PRO_C_O_H.ndx`` where the D run selected ``C, O01, H02``. In the CHARMM
topology ``O`` is the carbonyl oxygen (type OG2D1, bonded only to C) and
``O01`` is the hydroxyl (OG311, bonded to C and H02), so the L file measured
the angle at the *carbonyl* oxygen subtended by a hydrogen it is not bonded to.
``PRO_ang_OH`` therefore meant different things in the two runs, and any L/D
comparison would have read that as a difference between the enantiomers.

So this module does not just generate the groups, it checks them. Every group
declares how its atoms should be connected — a chain for angles and dihedrals,
a closed ring for the pyrrolidine, a central atom bonded to the rest for the
tetrahedral stereocentre — and the connectivity is verified against the
topology's own bond list. The L/D divergence above shows up immediately as
``O01-H02 expected bonded`` rather than as a mysterious CV.

Atom *order* inside each group is taken from the spec, never from the topology,
which matters because the two runs order their ORP atoms differently (L starts
CA, C, CB...; D starts N, CA, C...) while using identical names. Selecting by
name into an explicit order makes the output independent of that.

Depends on MDAnalysis, imported lazily, and on `leaflet_index` for the shared
.ndx writer.
"""

from pathlib import Path
from typing import Annotated

import typer

from . import panels
from .leaflet_index import _require_mdanalysis, _write_ndx
from .pathdata import _check_overwrite

#: (filename, group name, atom names in order, connectivity rule).
#:
#: The rules are what turn a silent selection mistake into an error:
#:   "chain" - consecutive atoms bonded, as an angle or dihedral requires
#:   "ring"  - consecutive bonded and the last bonded back to the first
#:   "star"  - the first atom bonded to every other, for a stereocentre
#:   None    - atoms that are deliberately not bonded to each other
#: A name may also be written ``H@X``, meaning "the one hydrogen bonded to the
#: atom called X". Hydrogen naming is the least stable thing in a topology -
#: which of H01/H02 carries the hydroxyl, whether the alpha hydrogen is HA or
#: HA1 - and getting it wrong is silent: an angle or a stereocentre frame built
#: on the wrong hydrogen is still a number. Resolving through the bond graph
#: cannot pick the wrong one, and fails loudly when the answer is ambiguous.
PERMEANT_INDEX_SPEC = [
    ("PRO_backbone.ndx", "PRO_backbone", ["N", "CA", "C"], "chain"),
    ("PRO_CA.ndx", "PRO_CA", ["CA"], None),
    ("PRO_dihedral.ndx", "dihedral", ["CB", "CA", "C", "O"], "chain"),
    ("PRO_dihedral_OH.ndx", "dihedral", ["O", "C", "O01", "H@O01"], "chain"),
    ("PRO_C_O_H.ndx", "ORP_COH", ["C", "O01", "H@O01"], "chain"),
    ("PRO_C_CG.ndx", "ORP_C_CG", ["C", "CG"], None),
    ("PRO_ring_plane.ndx", "ORP_ring", ["CA", "CB", "CG", "CD", "N"], "ring"),
    ("PRO_ring_plane_chiral.ndx", "ORP_ring",
     ["C", "CA", "CB", "CG", "CD", "N"], "chain"),
    ("PRO_tetrahedral.ndx", "ORP_tetra", ["CA", "N", "C", "CB", "H@CA"], "star"),
]


def _hydrogen_on(permeant, host_name):
    """The single hydrogen bonded to `host_name`, found through the bonds.

    Used wherever a group needs "the OH hydrogen" or "the hydrogen on the
    stereocentre" rather than a particular spelling of its name.
    """
    host = _named_atom(permeant, host_name)
    bonded = _bonded_names(host)
    if bonded is None:
        raise ValueError(
            f"Resolving 'H@{host_name}' needs the topology's bond list, and "
            "this one has none. Pass a .tpr, or name the hydrogen explicitly."
        )
    hydrogens = [b for b in host.bonded_atoms if b.name.upper().startswith("H")]
    if len(hydrogens) != 1:
        found = ", ".join(sorted(h.name for h in hydrogens)) or "none"
        raise ValueError(
            f"'H@{host_name}' expects exactly one hydrogen bonded to "
            f"{host_name}, found {len(hydrogens)} ({found}). Name the intended "
            "one explicitly in PERMEANT_INDEX_SPEC."
        )
    return hydrogens[0]


def _resolve_atom(permeant, token):
    """One spec entry: a literal atom name, or ``H@<host>``."""
    if token.startswith("H@"):
        return _hydrogen_on(permeant, token[2:])
    return _named_atom(permeant, token)


def _named_atom(permeant, name):
    """The single permeant atom called `name`, or a clear error."""
    sel = permeant.select_atoms(f"name {name}")
    if len(sel) == 0:
        raise ValueError(
            f"The permeant has no atom named '{name}'. The two simulations in "
            "this project use identical ORP atom names but different orders, "
            "so a missing name usually means the wrong topology, not a "
            "reordering. Available: " + ", ".join(sorted(a.name for a in permeant))
        )
    if len(sel) > 1:
        raise ValueError(
            f"'{name}' matches {len(sel)} permeant atoms; the groups here "
            "assume one atom per name."
        )
    return sel[0]


def _bonded_names(atom):
    """Names bonded to `atom`, or None when the topology carries no bonds."""
    try:
        return {b.name for b in atom.bonded_atoms}
    except Exception:  # pragma: no cover - depends on the topology format
        return None


def _connectivity_problems(atoms, rule):
    """Bond expectations of `rule` that `atoms` fails, as readable strings."""
    if rule is None or len(atoms) < 2:
        return []

    bonded = {a.name: _bonded_names(a) for a in atoms}
    if any(v is None for v in bonded.values()):
        return []  # no bond information; nothing to check against

    def _linked(a, b):
        return b.name in bonded[a.name]

    problems = []
    if rule in ("chain", "ring"):
        pairs = list(zip(atoms, atoms[1:]))
        if rule == "ring":
            pairs.append((atoms[-1], atoms[0]))
        problems = [f"{a.name}-{b.name}" for a, b in pairs if not _linked(a, b)]
    elif rule == "star":
        centre = atoms[0]
        problems = [f"{centre.name}-{b.name}" for b in atoms[1:]
                    if not _linked(centre, b)]
    return problems


def permeant_index(
    topology: Annotated[str, typer.Option("-topology", help="Topology with atom names and bonds; a .tpr, so the connectivity checks can run", rich_help_panel=panels.INPUT)] = "topol.tpr",
    permeant_resname: Annotated[str, typer.Option("-permeant-resname", help="Residue name of the permeant", rich_help_panel=panels.INPUT)] = "ORP",
    lipid_resnames: Annotated[str, typer.Option("-lipid-resnames", help="Space-separated membrane residue names for the hydrogen-bond groups", rich_help_panel=panels.INPUT)] = "DOPC POPC",
    phosphate_sel: Annotated[str, typer.Option("-phosphate-sel", help="Membrane atom-name pattern for the phosphate oxygens", rich_help_panel=panels.SELECT)] = "O1*",
    carbonyl_sel: Annotated[str, typer.Option("-carbonyl-sel", help="Membrane atom-name pattern for the carbonyl oxygens", rich_help_panel=panels.SELECT)] = "O2* O3*",
    nitrogen_sel: Annotated[str, typer.Option("-nitrogen-sel", help="Membrane atom-name pattern for the headgroup nitrogens", rich_help_panel=panels.SELECT)] = "N*",
    strict: Annotated[bool, typer.Option("-strict", help="Fail instead of warning when a group's atoms are not bonded as its rule expects", rich_help_panel=panels.MODEL)] = False,
    out_dir: Annotated[str, typer.Option("-out-dir", help="Directory to write the .ndx files into", rich_help_panel=panels.OUTPUT)] = "gromacs_input",
    overw: Annotated[bool, typer.Option("-O", help="Force overwriting of existing files", rich_help_panel=panels.OUTPUT)] = False,
):
    """Write the permeant-side .ndx files and verify their atom selections.

    Replaces the per-simulation `make_ndx.py` copies. Groups are built by
    selecting atoms by name into an explicit order, so the output does not
    depend on how the topology happens to order them — the L and D runs here
    use the same ORP names in different orders.

    Each group also declares how its atoms should be bonded, and that is
    checked against the topology. This is not decoration: the L and D copies of
    the old script disagreed about `PRO_C_O_H.ndx`, one selecting the hydroxyl
    oxygen O01 and the other the carbonyl O, which silently made `PRO_ang_OH`
    a different quantity in each run. That failure surfaces here as an
    unbonded pair. Use `-strict` in a pipeline to make it fatal.

    Writes PRO_backbone, PRO_CA, PRO_dihedral, PRO_dihedral_OH, PRO_C_O_H,
    PRO_C_CG, PRO_ring_plane, PRO_ring_plane_chiral, PRO_tetrahedral,
    PRO_Non_H and PRO_HB. Run `chiroflux leaflet-index` for the CN_*.ndx
    leaflet files; together they are every index file `generate-cvs` reads.

    Generate both simulations' files with this command, from each one's own
    topology, so a convention can no longer differ between them.
    """
    mda = _require_mdanalysis()
    universe = mda.Universe(topology)

    permeant = universe.select_atoms(f"resname {permeant_resname}")
    if len(permeant) == 0:
        raise ValueError(
            f"No atoms matched 'resname {permeant_resname}' in {topology}."
        )
    membrane = universe.select_atoms(f"resname {lipid_resnames}")
    if len(membrane) == 0:
        raise ValueError(
            f"No atoms matched 'resname {lipid_resnames}' in {topology}."
        )

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    has_bonds = _bonded_names(permeant[0]) is not None
    print(f"{len(permeant)} {permeant_resname} atoms, {len(membrane)} membrane atoms.")
    if not has_bonds:
        print("  [warn] the topology carries no bond list, so the connectivity "
              "checks are skipped. Pass a .tpr to enable them.")

    failures = []
    for filename, group, names, rule in PERMEANT_INDEX_SPEC:
        path = out_root / filename
        _check_overwrite(str(path), overw)

        atoms = [_resolve_atom(permeant, n) for n in names]
        problems = _connectivity_problems(atoms, rule)
        _write_ndx(str(path), [(group, [a.index + 1 for a in atoms])])

        # Print what the H@ tokens resolved to, not the tokens, so the actual
        # hydrogen used is visible rather than implied.
        resolved = "-".join(a.name for a in atoms)
        note = ""
        if problems:
            note = f"   <-- expected bonded: {', '.join(problems)}"
            failures.append((filename, problems))
        print(f"  {filename:<28} {resolved}{note}")

    # Selection-based groups: written in topology order, which is what their
    # consumers (COM, radius of gyration, hydrogen-bond counting) expect.
    heavy = permeant.select_atoms("not name H*")
    path = out_root / "PRO_Non_H.ndx"
    _check_overwrite(str(path), overw)
    _write_ndx(str(path), [("ORP_Heavy", sorted(int(i) + 1 for i in heavy.indices))])
    print(f"  {'PRO_Non_H.ndx':<28} {len(heavy)} heavy atoms")

    hb_groups = [
        ("ORP", permeant),
        ("MEMB", membrane),
        ("PRO_phosphate", membrane.select_atoms(f"name {phosphate_sel}")),
        ("PRO_carbonyl", membrane.select_atoms(f"name {carbonyl_sel}")),
        ("PRO_Nitrogen", membrane.select_atoms(f"name {nitrogen_sel}")),
    ]
    path = out_root / "PRO_HB.ndx"
    _check_overwrite(str(path), overw)
    _write_ndx(str(path), [(name, sorted(int(i) + 1 for i in sel.indices))
                           for name, sel in hb_groups])
    print(f"  {'PRO_HB.ndx':<28} " +
          ", ".join(f"{name} {len(sel)}" for name, sel in hb_groups))

    if failures:
        message = (
            f"{len(failures)} group(s) are not bonded as their rule expects: "
            + "; ".join(f"{f} ({', '.join(p)})" for f, p in failures)
            + ". This is what a wrong atom name looks like — check the spec "
            "against the topology before using these files."
        )
        if strict:
            raise ValueError(message)
        print(f"\n[warn] {message}")
    else:
        print(f"\n{len(PERMEANT_INDEX_SPEC) + 2} files written to {out_root}; "
              "every group is bonded as expected.")
