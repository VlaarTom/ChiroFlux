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

#: Hard ceiling on a figure dimension, in inches.
#:
#: These figures size themselves per item - 0.4 in per bar, 0.9 in per heatmap
#: column - which is fine for the ~40 CVs they were written for and fatal for
#: the ~240 features a `dynamics` run produces. matplotlib's Agg backend cannot
#: render a dimension past 2**16 pixels, which at 300 dpi is 218 inches, so an
#: uncapped figure raised at save time and left an unusable file behind after
#: the whole analysis had already run. 200 in stays under that; `top_n` below
#: is the real fix, and this is the backstop for whatever still slips past.
MAX_FIG_INCHES = 200.0


def _fig_inches(per_item, n_items, minimum):
    """Figure extent for `n_items`, floored at `minimum` and hard-capped."""
    return float(min(max(minimum, per_item * n_items), MAX_FIG_INCHES))


def _top_n_indices(scores, top_n):
    """Indices of the `top_n` largest scores, back in their original order.

    Original order rather than ranked order on purpose: it keeps a CV's
    features (``X_var``, ``X_tau``, ...) adjacent and keeps the axis
    comparable between two runs that rank things differently.
    """
    scores = np.nan_to_num(np.asarray(scores, dtype=float))
    if not top_n or top_n <= 0 or len(scores) <= top_n:
        return np.arange(len(scores))
    return np.sort(np.argsort(scores)[::-1][:top_n])


def _truncation_note(n_shown, n_total):
    return "" if n_shown >= n_total else f"  (top {n_shown} of {n_total})"


def _plot_importance_bar(
    values,
    feature_names,
    out_path,
    overw=False,
    xlabel="Mean |SHAP value|",
    title="SHAP Feature Importance",
    message="SHAP bar plot saved to",
    top_n=None,
):
    """Horizontal bar chart of a per-feature importance score.

    `values` is plotted as given and sorted ascending, so the most important
    feature ends up at the top. Callers that want magnitudes (e.g. signed
    Cohen's d) pass ``np.abs(...)`` themselves - permutation importance can
    legitimately be negative, so this does not take the absolute value.

    `top_n` keeps only that many highest-valued bars, which is what makes the
    figure legible once a caller has hundreds of features rather than dozens.
    The title then says so. Left as None, every feature is drawn.
    """
    _check_overwrite(out_path, overw)
    order = np.argsort(values)  # ascending so most important is at top
    if top_n and top_n > 0 and len(order) > top_n:
        order = order[-top_n:]  # the largest, still ascending

    n_shown = len(order)
    fig, ax = plt.subplots(figsize=(6, _fig_inches(0.4, n_shown, 3)))
    y_pos = np.arange(n_shown)
    ax.barh(y_pos, np.asarray(values)[order], color="#1f77b4")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title + _truncation_note(n_shown, len(feature_names)))
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
    top_n=None,
):
    """Heatmap of a per-CV score across all interfaces.

    Gives a single-figure overview of which CVs matter most and at which
    stage of the reaction they become important. Each entry of `results`
    needs a ``"lambda"`` and a ``"ranking"`` of ``(cv_name, value)`` pairs;
    entries whose ranking is None are skipped.

    `top_n` keeps the columns with the largest value at *any* interface, so a
    CV that only matters late is not dropped for being quiet early. They stay
    in their original order rather than being re-sorted by score.
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

    keep = _top_n_indices(np.nanmax(np.abs(mat), axis=0), top_n)
    shown_names = [cv_names[i] for i in keep]
    mat = mat[:, keep]

    fig, ax = plt.subplots(
        figsize=(_fig_inches(0.9, len(shown_names), 6),
                 _fig_inches(0.5, len(valid), 4))
    )
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label=value_label)
    ax.set_xticks(np.arange(len(shown_names)))
    ax.set_xticklabels(shown_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(valid)))
    ax.set_yticklabels([f"λ={l:.4f}" for l in lambdas], fontsize=8)
    ax.set_xlabel("Collective Variable")
    ax.set_ylabel("Interface")
    ax.set_title(title + _truncation_note(len(shown_names), len(cv_names)))
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Interface heatmap saved to {out_path}")
