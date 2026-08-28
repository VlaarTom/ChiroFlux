"""ChiroFlux - collective-variable analysis for TIS/RETIS path sampling.

The package bundles the analyses that take an infretis simulation (its
``.toml`` config, its ``infretis_data.txt`` path table and a folder of
per-path CV trajectories) and ask which collective variables actually
distinguish reactive from non-reactive paths - or one simulation from
another, e.g. two enantiomers.

Each analysis lives in its own module and is imported from there::

    from chiroflux.shap_analysis import shap_ml
    from chiroflux.shap_analysis_ld import shap_enantiomer
    from chiroflux.statistical_analysis import statistics
    from chiroflux.principal_component_analysis import PCA
    from chiroflux.prepare_deeptda_data import (
        prepare_deeptda_data,
        prepare_deeptda_data_ld,
    )
    from chiroflux.train_deeptda import train_deeptda

They are deliberately *not* re-exported here. Two of them share a name with
their own module (``prepare_deeptda_data``, ``train_deeptda``), so a
top-level alias would resolve to the function or the module depending on
what had been imported first. Importing from the module is unambiguous, and
it keeps ``import chiroflux`` from dragging in shap, scikit-learn, lightgbm
and torch.

What each one does:

``shap_ml``
    WHAM-weighted classifiers (random forest, logistic regression, gradient
    boosting, LightGBM, SVM) fitted per TIS interface and explained with SHAP.
``shap_enantiomer``
    The same, but classifying which of two simulations a path came from.
``statistics``
    Model-free effect sizes (weighted Cohen's d, Spearman, KS) per interface -
    a cheap sanity check on the SHAP rankings.
``PCA``
    Weighted principal-component analysis of the CV space, optionally on a
    joint basis fitted over two simulations.
``prepare_deeptda_data`` / ``prepare_deeptda_data_ld``
    Frame-level, WHAM-weighted training sets for DeepTDA.
``train_deeptda``
    Trains a DeepTDA collective variable on those datasets (needs the
    optional ``deeptda`` extra).

Every entry point is also a subcommand of the ``chiroflux`` CLI.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("chiroflux")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
