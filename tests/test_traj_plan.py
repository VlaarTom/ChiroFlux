"""traj.txt is the only authoritative frame-to-phase-point mapping.

Every trajectory reader here used to rebuild a path by reading whole .xtc
segments and then trying to spot the extra frames - by value equality in
cv_generation, by a rounded COM signature in sasa, by "drop the first frame of
interior segments" in neighbours and membrane_spatial. None of those can work,
because the surplus frames are not repeats of their neighbours: they are
distinct frames the .xtc files carry outside the path, and on the reference set
there are exactly 2*(n_segments-1)+1 of them, at the end of each segment and
the start of one.

traj.txt names the file and the frame index for every phase point, so following
it removes the guesswork. These tests pin the parsing and the grouping the
readers depend on.
"""

import pytest

from chiroflux.pathdata import read_traj_plan, traj_plan_segments

# A backward segment followed by a forward one, as infretis writes them: the
# backward half counts down, which is what makes a `direction` column redundant.
TRAJ_TXT = """# Cycle: 9, status: ACC
#     Step              Filename       index    vel
         0  000_1_trajB.trr          3     -1
         1  000_1_trajB.trr          2     -1
         2  000_1_trajB.trr          1     -1
         3  000_2_trajF.trr          1      1
         4  000_2_trajF.trr          2      1
"""


def _write(tmp_path, text):
    p = tmp_path / "traj.txt"
    p.write_text(text)
    return str(p)


class TestReadTrajPlan:
    def test_one_entry_per_phase_point_in_file_order(self, tmp_path):
        plan, g96 = read_traj_plan(_write(tmp_path, TRAJ_TXT))
        assert plan == [
            ("000_1_trajB.xtc", 3),
            ("000_1_trajB.xtc", 2),
            ("000_1_trajB.xtc", 1),
            ("000_2_trajF.xtc", 1),
            ("000_2_trajF.xtc", 2),
        ]
        assert g96 is None

    def test_a_backward_segment_keeps_its_descending_order(self, tmp_path):
        """No reversal step: the file already lists the path order."""
        plan, _ = read_traj_plan(_write(tmp_path, TRAJ_TXT))
        backward = [k for name, k in plan if "trajB" in name]
        assert backward == [3, 2, 1]

    def test_trr_names_resolve_to_the_xtc_on_disk(self, tmp_path):
        plan, _ = read_traj_plan(_write(tmp_path, TRAJ_TXT))
        assert all(name.endswith(".xtc") for name, _ in plan)

    def test_a_non_trr_name_is_left_alone(self, tmp_path):
        txt = "#h\n   0  seg_a.xtc   5   1\n"
        plan, _ = read_traj_plan(_write(tmp_path, txt))
        assert plan == [("seg_a.xtc", 5)]

    def test_a_g96_point_is_skipped_and_its_step_returned(self, tmp_path):
        """It has no .xtc frame; the caller drops the matching order.txt row."""
        txt = TRAJ_TXT + "         5  000_3_shoot.g96          0      1\n"
        plan, g96 = read_traj_plan(_write(tmp_path, txt))
        assert g96 == "5"
        assert all(not name.endswith(".g96") for name, _ in plan)
        assert len(plan) == 5

    def test_comment_and_blank_lines_are_ignored(self, tmp_path):
        txt = "# a\n@ b\n\n   0  s.trr   0   1\n\n"
        plan, _ = read_traj_plan(_write(tmp_path, txt))
        assert plan == [("s.xtc", 0)]

    def test_an_empty_file_gives_an_empty_plan(self, tmp_path):
        plan, g96 = read_traj_plan(_write(tmp_path, "# only a header\n"))
        assert plan == [] and g96 is None


class TestTrajPlanSegments:
    def test_groups_contiguous_runs_in_path_order(self, tmp_path):
        plan, _ = read_traj_plan(_write(tmp_path, TRAJ_TXT))
        assert traj_plan_segments(plan) == [
            ("000_1_trajB.xtc", [3, 2, 1]),
            ("000_2_trajF.xtc", [1, 2]),
        ]

    def test_a_file_revisited_later_becomes_two_segments(self, tmp_path):
        """Order matters more than uniqueness: a reader that collapsed these
        into one file entry would emit the frames in the wrong order."""
        txt = ("#h\n"
               "  0  a.trr  0  1\n"
               "  1  b.trr  7  1\n"
               "  2  a.trr  1  1\n")
        plan, _ = read_traj_plan(_write(tmp_path, txt))
        assert traj_plan_segments(plan) == [
            ("a.xtc", [0]), ("b.xtc", [7]), ("a.xtc", [1]),
        ]

    def test_segments_preserve_every_phase_point(self, tmp_path):
        plan, _ = read_traj_plan(_write(tmp_path, TRAJ_TXT))
        flat = [(name, k) for name, idxs in traj_plan_segments(plan) for k in idxs]
        assert flat == plan

    def test_an_empty_plan_gives_no_segments(self):
        assert traj_plan_segments([]) == []

    def test_segment_frame_lists_are_independent(self, tmp_path):
        """They are handed to callers that slice and iterate them."""
        plan, _ = read_traj_plan(_write(tmp_path, TRAJ_TXT))
        segs = traj_plan_segments(plan)
        segs[0][1].append(999)
        assert traj_plan_segments(plan)[0][1] == [3, 2, 1]


class TestFirstLastTrimmingComposes:
    """The readers trim the first/last phase point by slicing the plan, which
    only works because the plan is in path order and one row per phase point."""

    def test_dropping_the_first_phase_point(self, tmp_path):
        plan, _ = read_traj_plan(_write(tmp_path, TRAJ_TXT))
        segs = traj_plan_segments(plan[1:])
        assert segs[0] == ("000_1_trajB.xtc", [2, 1])

    def test_dropping_the_last_phase_point(self, tmp_path):
        plan, _ = read_traj_plan(_write(tmp_path, TRAJ_TXT))
        segs = traj_plan_segments(plan[:-1])
        assert segs[-1] == ("000_2_trajF.xtc", [1])

    def test_dropping_both_can_empty_a_segment_entirely(self, tmp_path):
        txt = "#h\n  0  a.trr  0  1\n  1  b.trr  4  1\n"
        plan, _ = read_traj_plan(_write(tmp_path, txt))
        assert traj_plan_segments(plan[1:-1]) == []


def test_pathdata_still_imports_nothing_internal():
    """pathdata is the bottom of the stack; the helper must not change that."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "chiroflux" / "pathdata.py"
    tree = ast.parse(src.read_text())
    relative = {n.module for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.level > 0}
    assert relative == set()


@pytest.mark.parametrize("module", ["cv_generation", "neighbours",
                                    "membrane_spatial", "sasa"])
def test_every_reader_uses_the_shared_helper(module):
    """Four copies of this parser drifted apart once already."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "chiroflux"
           / f"{module}.py").read_text()
    assert "read_traj_plan" in src, f"{module} does not use the shared plan"
    assert "def extract_sorted_traj_names" not in src, (
        f"{module} still carries its own traj.txt parser"
    )
