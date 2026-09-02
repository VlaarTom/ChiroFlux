"""The histogram binning table lives in a file, not in the code.

It is per-simulation - each run covers a different OP window - and keeping it
in the module is what produced four divergent copies of the original script.
"""

import inspect
from pathlib import Path

import pytest
import typer

from chiroflux.cv_histograms import _load_column_ranges, histograms

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "column_ranges.toml"


def test_the_example_file_ships_and_parses():
    """The command is unusable without one, so a working example must exist."""
    assert EXAMPLE.is_file(), "examples/column_ranges.toml is missing"
    ranges, nr_minus = _load_column_ranges(EXAMPLE)
    assert len(ranges) > 100
    assert "OP_Lamb" in ranges, "the reference column must be binned"
    assert "OP_Lamb" in nr_minus, "the non-reactive/minus override is the point"


def test_entries_become_min_max_nbins_tuples():
    ranges, _ = _load_column_ranges(EXAMPLE)
    lo, hi, nbins = ranges["OP_Lamb"]
    assert isinstance(nbins, int) and nbins > 0
    assert hi > lo


def test_the_ranges_file_is_required(monkeypatch):
    """No silent default: the binning determines every histogram produced, so
    a wrong-but-plausible fallback would be worse than refusing to run."""
    param = inspect.signature(histograms).parameters["ranges"]
    assert param.default is ..., "-ranges must be a required option"


@pytest.mark.parametrize(
    "body, expected",
    [
        ('[ranges]\n"A" = [1, 2]\n', "min, max, n_bins"),
        ('[ranges]\n"A" = [5, 1, 10]\n', "max <= min"),
        ('[ranges]\n"A" = [1, 5, 0]\n', "at least 1 bin"),
        ('[other]\nx = 1\n', "no [ranges] section"),
    ],
)
def test_malformed_tables_are_rejected(tmp_path, body, expected):
    p = tmp_path / "ranges.toml"
    p.write_text(body)
    with pytest.raises(typer.BadParameter, match=re_escape(expected)):
        _load_column_ranges(p)


def test_a_missing_file_points_at_the_example(tmp_path):
    with pytest.raises(typer.BadParameter, match="column_ranges.toml"):
        _load_column_ranges(tmp_path / "absent.toml")


def re_escape(s):
    import re

    return re.escape(s)
