"""Leaflet groups must be decided per lipid, not per atom.

The bug this replaces is subtle enough to be worth stating: a `gmx select`
expression like `name C210 and z > 4.06` classifies *atoms*, so a chain carbon
that has interdigitated past the midplane is filed under the opposite leaflet
even though its own lipid plainly is not. The headgroup markers never show it
(their z distributions do not overlap) which is why the files look fine at a
glance — only the deep chain carbons come out lopsided, 54/56 and 51/59 in the
real DOPC file against 55/55 for everything else.

These tests use small stand-ins for an MDAnalysis Universe: the module only
needs `select_atoms`, so building a real one would require a topology file
without testing anything more.
"""

import numpy as np
import pytest

from chiroflux.cv_generation import parse_ndx
from chiroflux.leaflet_index import (
    _assign_leaflets,
    _marker_groups,
    _read_g96_positions,
    _shared_midplane,
    _split_list,
    _write_ndx,
)


class _Atom:
    def __init__(self, index, resindex):
        self.index = index
        self.resindex = resindex


class _Sel:
    def __init__(self, atoms, positions=None):
        self.atoms = atoms
        self._positions = positions

    def __len__(self):
        return len(self.atoms)

    @property
    def positions(self):
        return np.asarray(self._positions, dtype=float)

    @property
    def residues(self):
        return sorted({a.resindex for a in self.atoms})

    @property
    def indices(self):
        return [a.index for a in self.atoms]


class _Universe:
    def __init__(self, table):
        self.table = table

    def select_atoms(self, selection):
        return self.table.get(selection, _Sel([]))


def _bilayer():
    """Four lipids: residues 0,1 upper (P at z=+10), 2,3 lower (P at z=-10).

    Their C210 carbons straddle the midplane the *wrong* way — the upper
    lipids' chains reach down to -1 and the lower lipids' up to +1 — which is
    exactly the configuration an atom-wise z cutoff gets backwards.
    """
    p_atoms = [_Atom(100 + i, i) for i in range(4)]
    p_z = [[0, 0, 10.0], [0, 0, 10.0], [0, 0, -10.0], [0, 0, -10.0]]
    c210 = [_Atom(200 + i, i) for i in range(4)]
    return _Universe({
        "resname DOPC and name P": _Sel(p_atoms, p_z),
        "resname DOPC and name C210": _Sel(c210),
    })


class TestAssignLeaflets:
    def test_splits_on_the_mean_headgroup_height(self):
        by_res, midplane = _assign_leaflets(_bilayer(), "DOPC", "P")
        assert midplane == pytest.approx(0.0)
        assert by_res == {0: True, 1: True, 2: False, 3: False}

    def test_honours_an_explicit_midplane(self):
        by_res, midplane = _assign_leaflets(_bilayer(), "DOPC", "P", midplane=20.0)
        assert midplane == 20.0
        assert set(by_res.values()) == {False}

    def test_rejects_a_leaflet_atom_that_is_not_unique_per_residue(self):
        """Two candidates in one residue makes the assignment ambiguous."""
        atoms = [_Atom(0, 0), _Atom(1, 0), _Atom(2, 1)]
        u = _Universe({"resname DOPC and name P": _Sel(atoms, [[0, 0, 1.0]] * 3)})
        with pytest.raises(ValueError, match="unique per residue"):
            _assign_leaflets(u, "DOPC", "P")

    def test_missing_selection_is_an_error_not_an_empty_file(self):
        with pytest.raises(ValueError, match="No atoms matched"):
            _assign_leaflets(_Universe({}), "DOPC", "P")


class TestMarkerGroups:
    def test_chain_carbons_follow_their_own_headgroup(self):
        """The whole point: a C210's own z must not decide its leaflet."""
        u = _bilayer()
        by_res, _ = _assign_leaflets(u, "DOPC", "P")
        upper, lower, orphan = _marker_groups(u, "DOPC", "C210", by_res)

        # residues 0,1 are upper, so their C210s are, whatever their own z
        assert upper == [201, 202]
        assert lower == [203, 204]
        assert orphan == 0

    def test_groups_come_out_the_same_size_as_the_headgroups(self):
        u = _bilayer()
        by_res, _ = _assign_leaflets(u, "DOPC", "P")
        upper, lower, _ = _marker_groups(u, "DOPC", "C210", by_res)
        assert len(upper) == len(lower) == 2

    def test_atoms_of_an_unassigned_residue_are_counted_not_silently_kept(self):
        u = _bilayer()
        by_res, _ = _assign_leaflets(u, "DOPC", "P")
        del by_res[3]
        upper, lower, orphan = _marker_groups(u, "DOPC", "C210", by_res)
        assert orphan == 1
        assert 204 not in upper and 204 not in lower


class TestSplitList:
    def test_accepts_commas_and_spaces_alike(self):
        """-resname DOPC,POPC and -resname "DOPC POPC" must both work."""
        assert _split_list("DOPC,POPC") == ["DOPC", "POPC"]
        assert _split_list("DOPC POPC") == ["DOPC", "POPC"]
        assert _split_list("DOPC, POPC") == ["DOPC", "POPC"]

    def test_single_value_and_empty(self):
        assert _split_list("DOPC") == ["DOPC"]
        assert _split_list("") == []
        assert _split_list(None) == []


class TestSharedMidplane:
    def test_pools_every_species(self):
        """One boundary for the membrane, not one per species.

        A minor species would otherwise get its own, noisier plane: on the
        reference system 110 DOPC give 38.69 A and 22 POPC give 39.16 A, so
        the same bilayer would be cut in two different places.
        """
        major = [_Atom(i, i) for i in range(10)]
        minor = [_Atom(100 + i, 100 + i) for i in range(2)]
        z_major = [[0, 0, 10.0]] * 5 + [[0, 0, -10.0]] * 5
        z_minor = [[0, 0, 14.0], [0, 0, -6.0]]  # mean +4, on its own
        u = _Universe({
            "resname DOPC and name P": _Sel(major, z_major),
            "resname POPC and name P": _Sel(minor, z_minor),
            "resname DOPC POPC and name P": _Sel(major + minor, z_major + z_minor),
        })

        pooled, n = _shared_midplane(u, ["DOPC", "POPC"], "P")
        assert n == 12
        assert pooled == pytest.approx(8.0 / 12.0)

        # POPC alone would have put the boundary at +4.0, far from the pooled one.
        alone, _ = _shared_midplane(u, ["POPC"], "P")
        assert alone == pytest.approx(4.0)

    def test_missing_species_is_an_error(self):
        with pytest.raises(ValueError, match="No atoms matched"):
            _shared_midplane(_Universe({}), ["DOPC"], "P")


class TestWriteNdx:
    def test_round_trips_through_the_projects_own_parser(self, tmp_path):
        """Written 1-based; parse_ndx hands back 0-based. Must survive that."""
        out = tmp_path / "test.ndx"
        groups = [("first", [1, 2, 3]), ("second", list(range(10, 50)))]
        _write_ndx(str(out), groups)

        parsed = parse_ndx(str(out))
        assert list(parsed) == ["first", "second"]
        assert parsed["first"].tolist() == [0, 1, 2]
        assert parsed["second"].tolist() == list(range(9, 49))

    def test_wraps_at_fifteen_indices_per_line(self, tmp_path):
        out = tmp_path / "wrap.ndx"
        _write_ndx(str(out), [("g", list(range(1, 34)))])
        body = [ln for ln in out.read_text().splitlines() if ln and "[" not in ln]
        assert [len(ln.split()) for ln in body] == [15, 15, 3]

    def test_an_empty_group_still_gets_its_header(self, tmp_path):
        """Group numbering is positional, so a dropped section shifts every
        later group and silently repoints the CVs."""
        out = tmp_path / "empty.ndx"
        _write_ndx(str(out), [("a", [1]), ("empty", []), ("c", [2])])
        assert list(parse_ndx(str(out))) == ["a", "empty", "c"]


G96 = """TITLE
made up
END
POSITION
    1 DOPC  C1         1    1.000000000    2.000000000    3.000000000
    1 DOPC  C2         2    1.500000000    2.500000000    3.500000000
END
BOX
   5.0 5.0 5.0
END
"""


class TestG96Positions:
    def test_reads_positions_and_converts_nm_to_angstrom(self, tmp_path):
        p = tmp_path / "conf.g96"
        p.write_text(G96)
        pos = _read_g96_positions(str(p), 2)
        assert pos.shape == (2, 3)
        assert pos[0].tolist() == pytest.approx([10.0, 20.0, 30.0])
        assert pos[1].tolist() == pytest.approx([15.0, 25.0, 35.0])

    def test_reads_the_reduced_block_too(self, tmp_path):
        p = tmp_path / "red.g96"
        p.write_text("POSITIONRED\n 1.0 2.0 3.0\n 4.0 5.0 6.0\nEND\n")
        pos = _read_g96_positions(str(p), 2)
        assert pos[1].tolist() == pytest.approx([40.0, 50.0, 60.0])

    def test_atom_count_mismatch_is_refused(self, tmp_path):
        """Silently misaligned coordinates would corrupt every leaflet call."""
        p = tmp_path / "conf.g96"
        p.write_text(G96)
        with pytest.raises(ValueError, match="holds 2 positions"):
            _read_g96_positions(str(p), 3)

    def test_a_file_with_no_position_block_is_refused(self, tmp_path):
        p = tmp_path / "bad.g96"
        p.write_text("TITLE\nnothing here\nEND\n")
        with pytest.raises(ValueError, match="No POSITION block"):
            _read_g96_positions(str(p), 2)
