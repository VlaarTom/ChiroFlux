"""MDAnalysis leaves a hidden frame-offset index next to every XTC/TRR it opens
(.<name>_offsets.npz, plus the .lock its file lock leaves behind). On an
inode-limited file system those two files per trajectory add up, so every
analysis removes them once it is done with a path's trajectories."""

import numpy as np
import pytest

from chiroflux.pathdata import remove_offset_files


def _hidden(folder):
    return sorted(p.name for p in folder.iterdir() if p.name.startswith("."))


def test_removes_both_files_for_every_trajectory(tmp_path):
    for name in ("a.xtc", "b.xtc"):
        (tmp_path / name).write_bytes(b"")
        for ending in ("npz", "lock"):
            (tmp_path / f".{name}_offsets.{ending}").write_bytes(b"")
    (tmp_path / ".unrelated").write_bytes(b"")
    remove_offset_files([tmp_path / "a.xtc", str(tmp_path / "b.xtc")])
    assert _hidden(tmp_path) == [".unrelated"]
    assert (tmp_path / "a.xtc").exists() and (tmp_path / "b.xtc").exists()


def test_accepts_a_single_path_and_missing_files(tmp_path):
    (tmp_path / ".c.xtc_offsets.npz").write_bytes(b"")
    remove_offset_files(str(tmp_path / "c.xtc"))        # no .lock: fine
    remove_offset_files(tmp_path / "never_opened.xtc")  # nothing at all: fine
    remove_offset_files([])
    assert _hidden(tmp_path) == []


def test_matches_what_mdanalysis_actually_writes(tmp_path):
    """Pins the naming: if MDAnalysis ever changes it, this fails instead of
    the files silently piling up again."""
    mda = pytest.importorskip("MDAnalysis")
    u = mda.Universe.empty(3, trajectory=True)
    xtc = tmp_path / "seg.xtc"
    with mda.Writer(str(xtc), 3) as w:
        for i in range(3):
            u.atoms.positions = np.full((3, 3), float(i))
            u.dimensions = [30.0, 30.0, 30.0, 90.0, 90.0, 90.0]
            w.write(u)
    for kwargs in ({}, {"refresh_offsets": True}):
        reader = mda.Universe(u._topology, str(xtc), **kwargs)
        assert _hidden(tmp_path), "MDAnalysis wrote no offset files"
        assert len(reader.trajectory) == 3
        remove_offset_files(xtc)
        assert _hidden(tmp_path) == []
