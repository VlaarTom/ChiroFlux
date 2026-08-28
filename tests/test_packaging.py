"""Checks that the package is wired together correctly.

These do not touch simulation data - they cover the packaging itself: the
public API, the CLI's lazy command loading, and the optional DeepTDA extra.
"""

import subprocess
import sys

import pytest
from typer.testing import CliRunner

from chiroflux import cli

runner = CliRunner()


def test_version_is_exposed():
    import chiroflux

    assert chiroflux.__version__ != ""


def test_documented_entry_points_are_importable():
    """The import paths promised in the package docstring and README must work."""
    from importlib import import_module

    for module_name, function_name in [
        ("shap_analysis", "shap_ml"),
        ("shap_analysis_ld", "shap_enantiomer"),
        ("statistical_analysis", "statistics"),
        ("principal_component_analysis", "PCA"),
        ("prepare_deeptda_data", "prepare_deeptda_data"),
        ("prepare_deeptda_data", "prepare_deeptda_data_ld"),
        ("train_deeptda", "train_deeptda"),
    ]:
        module = import_module(f"chiroflux.{module_name}")
        assert callable(getattr(module, function_name)), function_name


def test_cli_registry_covers_every_entry_point():
    """Every analysis function must be reachable as a CLI subcommand."""
    registered = {(spec.module, spec.function) for spec in cli._COMMANDS.values()}
    assert registered == {
        ("shap_analysis", "shap_ml"),
        ("shap_analysis_ld", "shap_enantiomer"),
        ("statistical_analysis", "statistics"),
        ("principal_component_analysis", "PCA"),
        ("prepare_deeptda_data", "prepare_deeptda_data"),
        ("prepare_deeptda_data", "prepare_deeptda_data_ld"),
        ("train_deeptda", "train_deeptda"),
    }


def test_importing_the_package_stays_light():
    """`import chiroflux` must not drag in the heavy scientific stack.

    Runs in a subprocess so the check is not defeated by modules another
    test already imported into this interpreter.
    """
    code = (
        "import sys; import chiroflux; "
        "heavy = [m for m in ('shap', 'sklearn', 'lightgbm', 'torch') if m in sys.modules]; "
        "print(','.join(heavy))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", f"eagerly imported: {out.stdout.strip()}"


def test_every_registered_command_can_be_built():
    """Each entry in the CLI registry must point at a real, wrappable function."""
    for name in cli._COMMANDS:
        command = cli._build_command(name)
        assert command.name == name
        assert command.params, f"{name} exposes no options"


def test_help_listing_does_not_build_real_commands(monkeypatch):
    """The command listing must come from the static table, not from imports.

    This is what keeps `chiroflux --help` from importing shap, scikit-learn
    and lightgbm just to print seven one-line descriptions.
    """
    monkeypatch.setattr(
        cli,
        "_build_command",
        lambda name: pytest.fail(f"--help built the real command {name!r}"),
    )
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for name in cli._COMMANDS:
        assert name in result.output


@pytest.mark.parametrize("name", list(cli._COMMANDS))
def test_subcommand_help_runs(name):
    """Dispatch must swap the stub for the real command, options and all."""
    result = runner.invoke(cli.app, [name, "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "-toml" in result.output or "-npz" in result.output


def test_missing_deeptda_extra_gives_an_actionable_error(monkeypatch):
    """Without torch/lightning/mlcolvar the error must name the extra to install."""
    import builtins

    from chiroflux.train_deeptda import _import_deeptda_backend

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "lightning", "mlcolvar"}:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"chiroflux\[deeptda\]"):
        _import_deeptda_backend()


ML_STACK = ("shap", "sklearn", "lightgbm", "pandas", "torch")

# Modules that must stay clear of the machine-learning stack. shap_analysis and
# shap_analysis_ld are deliberately absent: SHAP is what they are for.
ML_FREE_MODULES = [
    "chiroflux.pathdata",
    "chiroflux.cvs",
    "chiroflux.plotting",
    "chiroflux.statistical_analysis",
    "chiroflux.principal_component_analysis",
    "chiroflux.prepare_deeptda_data",
    "chiroflux.train_deeptda",
]


@pytest.mark.parametrize("module", ML_FREE_MODULES)
def test_module_does_not_import_the_ml_stack(module):
    """These analyses fit no model, so importing one must not cost the caller
    shap + scikit-learn + lightgbm (~1.5s and several hundred MB of RAM).

    Run in a subprocess: an in-process check would pass simply because another
    test had already imported shap_analysis.
    """
    code = (
        "import sys, importlib; importlib.import_module(%r); "
        "print(','.join(m for m in %r if m in sys.modules))" % (module, ML_STACK)
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", f"{module} pulled in: {out.stdout.strip()}"


def test_support_module_layering_holds():
    """pathdata is the bottom of the stack and must import nothing internal;
    cvs and plotting sit directly on it and may import only pathdata.

    If a support module grows an import of an analysis module, the layering -
    and the ML-free guarantee above - is gone.
    """
    import ast
    from pathlib import Path

    allowed = {"pathdata": set(), "cvs": {"pathdata"}, "plotting": {"pathdata"}}
    src_dir = Path(__file__).resolve().parents[1] / "src" / "chiroflux"
    for name, permitted in allowed.items():
        tree = ast.parse((src_dir / f"{name}.py").read_text())
        relative = {
            n.module
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.level > 0
        }
        assert relative <= permitted, (
            f"{name}.py imports {sorted(relative - permitted)}; "
            f"only {sorted(permitted) or 'nothing'} is allowed"
        )
