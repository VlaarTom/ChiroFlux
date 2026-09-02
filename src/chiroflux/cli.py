"""The ``chiroflux`` command-line interface.

Subcommands are loaded lazily: the module behind a command is imported only
when that command is actually invoked. Importing the full analysis stack
(shap, scikit-learn, lightgbm, pandas, matplotlib) costs a couple of seconds,
which would otherwise be paid on every ``chiroflux --help`` and on commands
that do not need it. The short help strings below are therefore duplicated
here rather than read from the functions' docstrings.
"""

from dataclasses import dataclass
from importlib import import_module

import click
import typer
import typer.core

from . import __version__


@dataclass(frozen=True)
class _Command:
    """Where a subcommand lives, and what to print for it in ``--help``."""

    module: str
    function: str
    short_help: str


_COMMANDS = {
    "generate-cvs": _Command(
        "cv_generation",
        "generate_cvs",
        "Compute per-frame CVs from MD trajectories into per-path .txt files.",
    ),
    "histograms": _Command(
        "cv_histograms",
        "histograms",
        "Weighted CV histograms, statistics and 2D maps over a path ensemble.",
    ),
    "shap-ml": _Command(
        "shap_analysis",
        "shap_ml",
        "Per-interface SHAP feat. importance of CVs for reactivity (Single sim.).",
    ),
    "shap-enantiomer": _Command(
        "shap_analysis_ld",
        "shap_enantiomer",
        "Per-interface SHAP importance separating two simulations (e.g. L/D).",
    ),
    "statistics": _Command(
        "statistical_analysis",
        "statistics",
        "Model-free weighted effect sizes (Cohen's d, Spearman, KS) per interface.",
    ),
    "pca": _Command(
        "principal_component_analysis",
        "PCA",
        "Weighted PCA of the CV space, optionally on a joint two-simulation basis.",
    ),
    "prepare-deeptda-data": _Command(
        "prepare_deeptda_data",
        "prepare_deeptda_data",
        "Build a weighted frame-level DeepTDA dataset (reactive/non-reactive).",
    ),
    "prepare-deeptda-data-ld": _Command(
        "prepare_deeptda_data",
        "prepare_deeptda_data_ld",
        "Build a weighted frame-level DeepTDA dataset (two simulations).",
    ),
    "train-deeptda": _Command(
        "train_deeptda",
        "train_deeptda",
        "Train a 2-state DeepTDA CV on a prepare-deeptda-data dataset.",
    ),
}


def _build_command(name):
    """Import the module behind `name` and wrap its function as a click command."""
    spec = _COMMANDS[name]
    function = getattr(import_module(f".{spec.module}", __package__), spec.function)

    # A single-command Typer app collapses to a plain TyperCommand, which is
    # exactly the click object this group needs to hand back.
    sub = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})
    sub.command(name=name)(function)
    command = typer.main.get_command(sub)
    command.name = name
    command.short_help = spec.short_help
    return command


def _stub_command(name):
    """A parameter-less stand-in used only to render `name` in the help listing.

    Building the real command would import its module, and the help renderer
    asks for *every* command - so listing all seven would import the whole
    analysis stack just to print seven one-line descriptions.
    """
    spec = _COMMANDS[name]
    return click.Command(name=name, help=spec.short_help, short_help=spec.short_help)


class _LazyGroup(typer.core.TyperGroup):
    """A command group that imports a subcommand's module only on demand."""

    def list_commands(self, ctx):
        return list(_COMMANDS)

    def get_command(self, ctx, name):
        # Only ever reached for help/listing: dispatch goes via resolve_command.
        if name not in _COMMANDS:
            return None
        return _stub_command(name)

    def resolve_command(self, ctx, args):
        # This is the one place click looks a command up in order to *run* it,
        # so this is where the stub gets swapped for the real thing.
        name, command, rest = super().resolve_command(ctx, args)
        if command is not None and command.name in _COMMANDS:
            command = _build_command(command.name)
        return name, command, rest


def _version_callback(value):
    if value:
        typer.echo(f"chiroflux {__version__}")
        raise typer.Exit()


app = typer.Typer(
    cls=_LazyGroup,
    no_args_is_help=True,
    add_completion=False,
    help=(
        "ChiroFlux - collective-variable analysis for TIS/RETIS path sampling.\n\n"
        "Most commands read an infretis .toml config, an infretis_data.txt path "
        "table and a folder of per-path CV trajectories, then rank the CVs by how "
        "well they separate reactive from non-reactive paths (or one simulation "
        "from another). Run 'chiroflux COMMAND --help' for a command's options."
    ),
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback()
def _main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show the chiroflux version and exit.",
    ),
):
    pass


def main():
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
