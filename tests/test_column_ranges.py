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


# ── chiroflux sasa: the run table is an input file too ───────────────────────

SASA_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sasa_runs.toml"


def test_the_sasa_runs_example_ships():
    from chiroflux.sasa import sasa

    assert SASA_EXAMPLE.is_file(), "examples/sasa_runs.toml is missing"
    assert inspect.signature(sasa).parameters["runs"].default is ..., (
        "-runs must be required: which simulations are combined, and with what "
        "relative scaling, defines the whole result"
    )


def test_sasa_run_entries_are_validated(tmp_path):
    """A run table that names a path which does not exist, or scales a
    histogram by a non-positive number, must fail before any trajectory is
    read rather than producing a quietly wrong profile."""
    from chiroflux.sasa import _load_runs

    (tmp_path / "load").mkdir()
    (tmp_path / "ml").mkdir()
    (tmp_path / "topol.tpr").write_text("x")
    (tmp_path / "w.txt").write_text("x")
    base = (
        '[[run]]\nname="entry"\n'
        f'load_dir="{tmp_path}/load"\nweights="{tmp_path}/w.txt"\n'
        f'ml_dir="{tmp_path}/ml"\ntpr="{tmp_path}/topol.tpr"\n'
    )
    cfg = tmp_path / "runs.toml"

    cfg.write_text(base)
    run = _load_runs(cfg)[0]
    assert run["scale"] == 1.0 and run["mirror_z"] is False, "defaults applied"

    for body, expected in [
        (base + "scale=-1\n", "must be positive"),
        (base + base, "duplicate run name"),
        ('[[run]]\nname="x"\n', "is missing"),
        (base.replace(str(tmp_path) + "/load", "/nope"), "does not exist"),
        ("[other]\nx=1\n", "no [[run]] tables"),
    ]:
        cfg.write_text(body)
        with pytest.raises(typer.BadParameter, match=re_escape(expected)):
            _load_runs(cfg)
