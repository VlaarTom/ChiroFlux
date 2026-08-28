"""Plots shared by more than one analysis module.

``_plot_importance_bar`` and ``_plot_interface_heatmap`` each used to exist as
two near-identical copies - one in ``shap_analysis`` (labelled for mean |SHAP|)
and one in ``statistical_analysis`` (labelled for |Cohen's d|) - differing only
in axis labels and titles. Those are parameters here.

The SHAP-specific plots (beeswarm, dependence, ROC, calibration) deliberately
stay in ``shap_analysis``: they need shap and scikit-learn, and pulling them in
here would cost every importer of this module the whole ML stack.

Depends only on numpy and matplotlib.
"""

import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .pathdata import _check_overwrite


def _plot_importance_bar(
    values,
    feature_names,
    out_path,
    overw=False,
    xlabel="Mean |SHAP value|",
    title="SHAP Feature Importance",
    message="SHAP bar plot saved to",
):
    """Horizontal bar chart of a per-feature importance score.

    `values` is plotted as given and sorted ascending, so the most important
    feature ends up at the top. Callers that want magnitudes (e.g. signed
    Cohen's d) pass ``np.abs(...)`` themselves - permutation importance can
    legitimately be negative, so this does not take the absolute value.
    """
    _check_overwrite(out_path, overw)
    order = np.argsort(values)  # ascending so most important is at top
    fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(feature_names))))
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, values[order], color="#1f77b4")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"{message} {out_path}")


def _plot_interface_heatmap(
    results,
    cv_names,
    out_path,
    overw=False,
    value_label="Mean |SHAP|",
    title="Mean |SHAP| across interfaces",
):
    """Heatmap of a per-CV score across all interfaces.

    Gives a single-figure overview of which CVs matter most and at which
    stage of the reaction they become important. Each entry of `results`
    needs a ``"lambda"`` and a ``"ranking"`` of ``(cv_name, value)`` pairs;
    entries whose ranking is None are skipped.
    """
    _check_overwrite(out_path, overw)
    valid = [r for r in results if r["ranking"] is not None]
    if not valid:
        return

    lambdas = [r["lambda"] for r in valid]
    cv_to_idx = {name: j for j, name in enumerate(cv_names)}
    mat = np.zeros((len(valid), len(cv_names)))
    unknown = set()
    for row_i, r in enumerate(valid):
        for cv_name, val in r["ranking"]:
            col = cv_to_idx.get(cv_name)
            if col is None:
                # A ranking naming a CV outside cv_names means the two were
                # built from different column sets; warn rather than either
                # crashing or silently dropping it.
                unknown.add(cv_name)
                continue
            mat[row_i, col] = val
    if unknown:
        warnings.warn(
            f"{len(unknown)} CV(s) in the ranking are absent from cv_names and "
            f"were left out of {out_path}: {', '.join(sorted(unknown))}",
            stacklevel=2,
        )

    fig, ax = plt.subplots(
        figsize=(max(6, 0.9 * len(cv_names)), max(4, 0.5 * len(valid)))
    )
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label=value_label)
    ax.set_xticks(np.arange(len(cv_names)))
    ax.set_xticklabels(cv_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(valid)))
    ax.set_yticklabels([f"λ={l:.4f}" for l in lambdas], fontsize=8)
    ax.set_xlabel("Collective Variable")
    ax.set_ylabel("Interface")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Interface heatmap saved to {out_path}")
