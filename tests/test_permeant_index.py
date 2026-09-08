"""Index groups must name the atoms their CV actually needs.

The failure this guards against really happened here. Two copies of the old
`make_ndx.py` disagreed about `PRO_C_O_H.ndx`: one selected `C, O01, H02`, the
other `C, O, H02`. In the CHARMM topology `O` is the carbonyl oxygen, bonded
only to C, while `O01` is the hydroxyl, bonded to C and H02 — so the second
spelling asks for the angle at an oxygen that is not attached to the hydrogen
closing it. Nothing crashes; `PRO_ang_OH` just quietly means something else,
and an L/D comparison reads that as chemistry.

A connectivity rule per group turns that into a message. These tests use small
stand-ins for MDAnalysis atoms, since only `.name`, `.index` and
`.bonded_atoms` are ever touched.
"""

import pytest

from chiroflux.permeant_index import (
    PERMEANT_INDEX_SPEC,
    _connectivity_problems,
    _named_atom,
    _resolve_atom,
)


class _Atom:
    def __init__(self, name, index):
        self.name = name
        self.index = index
        self._bonded = []

    @property
    def bonded_atoms(self):
        return list(self._bonded)


class _NoBondAtom(_Atom):
    @property
    def bonded_atoms(self):
        raise AttributeError("this topology has no bond information")


def _bond(a, b):
    a._bonded.append(b)
    b._bonded.append(a)


@pytest.fixture
def proline():
    """The ORP skeleton, bonded as the real topology has it.

    Both hydrogens that share the H0x naming are present: H01 on the ring
    nitrogen and H02 on the hydroxyl oxygen. That pairing is the whole reason
    the spec resolves hydrogens through bonds — the names are adjacent and
    carry no hint of which is which.
    """
    names = ["N", "CA", "C", "O", "O01", "H01", "H02", "CB", "CG", "CD", "HA"]
    at = {n: _Atom(n, i) for i, n in enumerate(names)}
    for a, b in [("N", "CA"), ("CA", "C"), ("C", "O"), ("C", "O01"),
                 ("O01", "H02"), ("N", "H01"), ("CA", "CB"), ("CB", "CG"),
                 ("CG", "CD"), ("CD", "N"), ("CA", "HA")]:
        _bond(at[a], at[b])
    return at


class TestConnectivityRules:
    def test_the_real_mismatch_is_caught(self, proline):
        """C-O-H02: O is the carbonyl oxygen and is not bonded to H02."""
        wrong = [proline[n] for n in ("C", "O", "H02")]
        assert _connectivity_problems(wrong, "chain") == ["O-H02"]

    def test_the_corrected_selection_passes(self, proline):
        right = [proline[n] for n in ("C", "O01", "H02")]
        assert _connectivity_problems(right, "chain") == []

    def test_ring_rule_requires_the_ring_to_close(self, proline):
        ring = [proline[n] for n in ("CA", "CB", "CG", "CD", "N")]
        assert _connectivity_problems(ring, "ring") == []
        # The same atoms minus the closing bond are a chain, not a ring.
        open_chain = [proline[n] for n in ("CB", "CG", "CD", "N", "C")]
        assert _connectivity_problems(open_chain, "ring") != []

    def test_star_rule_checks_every_substituent_of_the_centre(self, proline):
        tetra = [proline[n] for n in ("CA", "N", "C", "CB", "HA")]
        assert _connectivity_problems(tetra, "star") == []
        # O hangs off C, not off the stereocentre.
        assert _connectivity_problems(
            [proline[n] for n in ("CA", "N", "O")], "star"
        ) == ["CA-O"]

    def test_no_rule_means_no_complaint(self, proline):
        """C and CG are deliberately unbonded — an angle they do not span."""
        assert _connectivity_problems([proline["C"], proline["CG"]], None) == []

    def test_a_topology_without_bonds_is_skipped_not_failed(self):
        atoms = [_NoBondAtom("C", 0), _NoBondAtom("O", 1)]
        assert _connectivity_problems(atoms, "chain") == []


class _Sel:
    def __init__(self, atoms):
        self.atoms = atoms

    def __len__(self):
        return len(self.atoms)

    def __getitem__(self, i):
        return self.atoms[i]

    def __iter__(self):
        return iter(self.atoms)

    def select_atoms(self, selection):
        name = selection.split()[-1]
        return _Sel([a for a in self.atoms if a.name == name])


class TestNamedAtom:
    def test_finds_the_atom(self, proline):
        sel = _Sel(list(proline.values()))
        assert _named_atom(sel, "O01").name == "O01"

    def test_missing_name_lists_what_is_available(self, proline):
        sel = _Sel(list(proline.values()))
        with pytest.raises(ValueError, match="no atom named 'HB'"):
            _named_atom(sel, "HB")

    def test_missing_name_error_names_the_alternatives(self, proline):
        sel = _Sel(list(proline.values()))
        with pytest.raises(ValueError, match="O01"):
            _named_atom(sel, "HB")

    def test_duplicate_name_is_refused(self):
        sel = _Sel([_Atom("CA", 0), _Atom("CA", 1)])
        with pytest.raises(ValueError, match="matches 2"):
            _named_atom(sel, "CA")


class TestHydrogenResolution:
    """Hydrogen names are the least stable part of a topology, so the groups
    that need a particular hydrogen ask the bond graph instead of guessing."""

    def test_resolves_the_hydroxyl_hydrogen_by_its_bond(self, proline):
        sel = _Sel(list(proline.values()))
        assert _resolve_atom(sel, "H@O01").name == "H02"

    def test_resolves_the_hydrogen_on_the_stereocentre(self, proline):
        sel = _Sel(list(proline.values()))
        assert _resolve_atom(sel, "H@CA").name == "HA"

    def test_follows_a_renamed_hydrogen(self, proline):
        """Swap the two H0x names, as a differently built topology might.

        A literal "H02" would now silently pick the amine hydrogen and build
        the C-O-H angle on an atom three bonds away. Resolving by bond cannot.
        """
        proline["H02"].name, proline["H01"].name = "H01", "H02"
        sel = _Sel(list(proline.values()))
        assert _resolve_atom(sel, "H@O01").name == "H01"
        assert _resolve_atom(sel, "H@N").name == "H02"

    def test_refuses_an_ambiguous_host(self, proline):
        """CB carries two hydrogens, so 'the' hydrogen on it is meaningless."""
        for name in ("H2B", "H3B"):
            h = _Atom(name, 90)
            _bond(proline["CB"], h)
            proline[name] = h
        sel = _Sel(list(proline.values()))
        with pytest.raises(ValueError, match="exactly one hydrogen"):
            _resolve_atom(sel, "H@CB")

    def test_refuses_a_host_with_no_hydrogen(self, proline):
        sel = _Sel(list(proline.values()))
        with pytest.raises(ValueError, match="found 0"):
            _resolve_atom(sel, "H@O")

    def test_needs_a_topology_with_bonds(self):
        sel = _Sel([_NoBondAtom("O01", 0)])
        with pytest.raises(ValueError, match="bond list"):
            _resolve_atom(sel, "H@O01")

    def test_a_plain_name_still_works(self, proline):
        sel = _Sel(list(proline.values()))
        assert _resolve_atom(sel, "CA").name == "CA"


class TestSpec:
    def test_the_angle_group_uses_the_hydroxyl_oxygen(self):
        """Pin the corrected selection so a future edit cannot quietly undo it."""
        spec = {name: atoms for name, _, atoms, _ in PERMEANT_INDEX_SPEC}
        assert spec["PRO_C_O_H.ndx"] == ["C", "O01", "H@O01"]

    def test_every_group_declares_a_known_rule(self):
        for filename, _, _, rule in PERMEANT_INDEX_SPEC:
            assert rule in (None, "chain", "ring", "star"), filename

    def test_the_tetrahedral_group_starts_on_the_stereocentre(self):
        """compute_body_frame reads CA, N, C in that order off this group."""
        spec = {name: atoms for name, _, atoms, _ in PERMEANT_INDEX_SPEC}
        assert spec["PRO_tetrahedral.ndx"][:3] == ["CA", "N", "C"]

    def test_every_hydrogen_in_the_spec_is_resolved_not_spelled(self):
        """A literal hydrogen name is the failure mode this guards against."""
        for filename, _, names, _ in PERMEANT_INDEX_SPEC:
            for name in names:
                if name.upper().startswith("H") and not name.startswith("H@"):
                    raise AssertionError(
                        f"{filename} names the hydrogen '{name}' literally; "
                        "use H@<host> so a renamed hydrogen cannot slip past"
                    )

    def test_every_group_is_verified_against_a_real_proline(self, proline):
        """The shipped spec must itself satisfy the rules it declares."""
        sel = _Sel(list(proline.values()))
        for filename, _, names, rule in PERMEANT_INDEX_SPEC:
            atoms = [_resolve_atom(sel, n) for n in names]
            assert _connectivity_problems(atoms, rule) == [], filename
