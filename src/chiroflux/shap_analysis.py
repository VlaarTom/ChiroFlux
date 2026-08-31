"""SHAP feature-importance analysis of collective variables (CVs) for
TIS/RETIS path sampling.

For each TIS interface lambda_i (read from the infretis .toml file):
  1. Take every path's CV values at its first crossing of lambda_i.
  2. Label each path reactive / non-reactive (from its trajectory file header).
  3. For each model to compare (random forest / logistic regression /
     gradient boosting), fit a WHAM-weighted classifier CV -> label on all
     the paths that reach lambda_i, and explain it with SHAP on that same
     data.
  4. Rank CVs by mean |SHAP| per model, and plot a per-interface comparison
     of the top CVs across models.

Trajectory file layout (per path, e.g. ``ML/<path_nr>.txt``):
  line 1: "# reactive" / "# non-reactive"
  line 2: "# <ensemble info>"
  line 3: "# <duplicate/edit info>"
  line 4: column names (no leading "#")
  line 5+: data
"""

import datetime
import warnings
from pathlib import Path
from typing import Annotated, Optional
import os
os.environ["OMP_NUM_THREADS"] = "1"


import joblib
import matplotlib
import numpy as np
import shap
import tomli
import typer
from joblib import Parallel, delayed
from shap.utils._exceptions import ExplainerError

matplotlib.use("Agg")  # non-interactive backend; safe for CLI use
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .cvs import _apply_angle_transforms, _apply_z_corrections
from .pathdata import (
    _check_overwrite,
    _compute_path_weights,
    _extract_cv_crossings,
    _extract_path_metadata,
    _load_path_table,
)
from .plotting import _plot_importance_bar, _plot_interface_heatmap

_MODEL_LABELS = {
    "rf":     "RandomForest",
    "gbm":    "GradientBoosting",
    "lgbm":   "LightGBM",
    "logreg": "LogisticRegression",
    "svm":    "SupportVectorMachine"
}
MODEL_CHOICES = tuple(_MODEL_LABELS.keys())

# Per-model hyperparameter search spaces used by _optimize_hyperparams.
_PARAM_GRIDS = {
    "rf": {
        "n_estimators":     [100, 200, 300, 500],
        "max_depth":        [None, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5],
        "max_features":     ["sqrt", "log2"],
    },
    "gbm": {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [3, 5, 7],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "min_samples_leaf": [1, 5, 10],
    },
    "lgbm": {
        "n_estimators":       [100, 200, 300, 500],
        "num_leaves":         [31, 63, 127],
        "learning_rate":      [0.01, 0.05, 0.1],
        "min_child_samples":  [5, 10, 20],
    },
    "logreg": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "svm": {
        "C":     [0.1, 0.5, 1.0, 5.0, 10.0],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    },
}


# Maximum training samples for SVM per fold.  SVM scales O(n²)–O(n³), so
# large datasets prevent convergence and make permutation importance useless.
# Stratified subsampling keeps class ratios intact while capping the cost.
_MAX_SVM_TRAIN = 5000


def _chunk_shap_values(model, X_chunk):
    """SHAP values for one chunk of rows, normalised to the positive class.

    check_additivity=False: RandomForest is fit with WHAM sample_weight,
    whose dynamic range across paths can span many orders of magnitude;
    combined with the recursive Tree SHAP algorithm, this occasionally
    triggers float precision artefacts that fail shap's strict per-sample
    additivity check on an individual row even though the attributions are
    otherwise fine. shap's own warning recommends disabling the check
    rather than the much slower feature_perturbation='interventional' mode.
    """
    sv = shap.TreeExplainer(model).shap_values(X_chunk, check_additivity=False)
    # Normalise across shap versions: older shap returns a [class0, class1]
    # list, newer shap returns one (N, N_feat, 2) array.
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def _parallel_shap_values(model, X_test, n_jobs):
    """SHAP values for X_test, splitting rows across n_jobs worker processes.

    shap.TreeExplainer has no n_jobs/threading knob of its own (it calls
    straight into a single-threaded C extension for sklearn models), so for
    a non-trivial test set it is by far the slowest part of each fold - much
    slower than fitting the forest itself, which sklearn already
    parallelises via n_jobs. Splitting rows into independent chunks (SHAP
    values per row don't depend on other rows, given the fitted model) and
    explaining each chunk in its own process is the only way to use more
    than one core here.
    """
    n_jobs_eff = joblib.effective_n_jobs(n_jobs)
    if n_jobs_eff <= 1 or len(X_test) < 2 * n_jobs_eff:
        return _chunk_shap_values(model, X_test)

    chunks = [c for c in np.array_split(X_test, n_jobs_eff) if len(c)]
    results = Parallel(n_jobs=n_jobs_eff)(
        delayed(_chunk_shap_values)(model, c) for c in chunks
    )
    return np.concatenate(results, axis=0)


def _linear_shap_values(model, X_train, X_test):
    """SHAP values for a linear model via shap.LinearExplainer.

    Uses X_train as the background distribution (its column means become the
    SHAP reference point, which after StandardScaler are all ~0). Returns a
    (N_test, N_features) array for the positive class; fast and exact for
    any sklearn linear model (LogisticRegression, Ridge, etc.).

    max_samples is set explicitly to avoid shap's default subsampling to 100,
    which would trigger a noisy warning even though the effect on values is
    negligible after StandardScaler.
    """
    masker = shap.maskers.Independent(X_train, max_samples=len(X_train))
    explainer = shap.LinearExplainer(model, masker)
    sv = explainer.shap_values(X_test)
    # Older shap returns list [class0, class1]; newer returns single array.
    if isinstance(sv, list):
        sv = sv[1]
    return np.asarray(sv, dtype=float)


def _optimize_hyperparams(
    model_type, X_f, y_f, sw, groups_f,
    n_splits, n_jobs, random_state, n_iter,
):
    """Random search over _PARAM_GRIDS[model_type] using the same CV strategy
    as the main k-fold loop.  Returns the best parameter dict, which callers
    merge into the model constructor kwargs so the best params override the
    defaults for every fold.

    Scale-sensitive models (logreg, svm) receive a globally-standardised X for
    the search; tree models receive raw X (scaling has no effect on trees).
    This is acceptable because the search is only used to select
    hyperparameters — the reported AUC and SHAP values come from the main
    k-fold loop where scaling is done correctly per fold.
    """
    import math

    from sklearn.model_selection import RandomizedSearchCV

    param_grid = _PARAM_GRIDS[model_type]
    n_combos   = math.prod(len(v) for v in param_grid.values())
    n_iter     = min(n_iter, n_combos)

    if model_type in ("logreg", "svm"):
        X_search = StandardScaler().fit_transform(X_f)
    else:
        X_search = X_f

    if model_type == "rf":
        base = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
    elif model_type == "gbm":
        base = GradientBoostingClassifier(random_state=random_state)
    elif model_type == "lgbm":
        base = lgb.LGBMClassifier(random_state=random_state, n_jobs=n_jobs, verbose=-1)
    elif model_type == "logreg":
        base = LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=n_jobs)
    else:  # svm
        base = SVC(kernel="rbf", probability=True, random_state=random_state, max_iter=100000)

    cv = (
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if groups_f is not None
        else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    )
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring="roc_auc",
        n_jobs=n_jobs,
        random_state=random_state,
        refit=False,
        error_score=np.nan,
    )
    fit_kwargs = {"sample_weight": sw}
    if groups_f is not None:
        fit_kwargs["groups"] = groups_f
    search.fit(X_search, y_f, **fit_kwargs)
    print(
        f"  HPO ({n_iter} iterations): best CV AUC = {search.best_score_:.4f}"
        f"  params = {search.best_params_}"
    )
    return search.best_params_


def _model_shap_kfold(
    X,
    y,
    feature_names,
    model_type="rf",
    sample_weight=None,
    n_splits=5,
    n_estimators=300,
    n_jobs=-1,
    random_state=0,
    groups=None,
    optimize=False,
    n_search_iter=20,
):
    """Stratified k-fold CV with SHAP explanation, supporting three model types.

    For each fold: standardise (fit scaler on train only), fit the chosen
    classifier, compute out-of-fold SHAP values on the held-out test split.
    All folds are pooled back into a single array aligned with the input rows.

    model_type choices:
      "rf"     — RandomForestClassifier; shap.TreeExplainer, parallelised by
                  splitting test rows across processes (see _parallel_shap_values).
      "gbm"    — GradientBoostingClassifier; shap.TreeExplainer, same parallel
                  strategy.  Boosting often captures different non-linear patterns
                  than RF's bagging, so comparing the two rankings tests robustness.
      "lgbm"   — LightGBMClassifier; shap.TreeExplainer.  Histogram-based boosting
                  that is significantly faster and more memory-efficient than GBM,
                  especially for large datasets.  Supports n_jobs for parallel training.
      "logreg" — LogisticRegression; shap.LinearExplainer (exact, fast, no
                  parallelisation needed).  Provides a linear baseline: if its
                  AUC matches the tree models, the CV--label relationship is
                  roughly linear; a large gap reveals important non-linearity.

    Parameters
    ----------
    X             : (N_samples, N_features) CV values.
    y             : (N_samples,) binary labels (1 = reactive, 0 = non-reactive).
    feature_names : length N_features CV names.
    model_type    : one of MODEL_CHOICES ("rf", "gbm", "lgbm", "logreg").
    sample_weight : optional (N_samples,) WHAM path weights.
    n_splits      : number of stratified CV folds.
    n_estimators  : trees for RF / GBM / LGBM (ignored for logreg).
    n_jobs        : CPU cores for RF / LGBM / logreg solver; -1 = all cores.
                    sklearn GBM training is sequential (boosting) so n_jobs is ignored.
    random_state  : seed for the fold splitter and the model.
    optimize      : whether to run hyperparameter search before the main k-fold loop.
    n_search_iter : number of random hyperparameter configurations to evaluate when optimize is set.

    Returns
    -------
    shap_values   : (N_samples, N_features) OOF SHAP values (reactive class);
                    NaN for rows dropped as non-finite.
    mean_abs_shap : (N_features,) mean |SHAP| across samples.
    fold_auc      : (n_splits,) held-out ROC AUC per fold.
    fold_roc      : list of (fpr, tpr, auc) tuples for folds with both classes.
    oof_true      : (N_samples,) true binary labels, NaN for dropped rows.
    oof_proba     : (N_samples,) OOF predicted reactive probability, NaN for
                    dropped rows.
    """
    if model_type not in MODEL_CHOICES:
        raise ValueError(f"model_type {model_type!r} not in {MODEL_CHOICES}")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    finite = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        finite &= np.isfinite(sample_weight)
    n_dropped = len(y) - int(np.sum(finite))
    if n_dropped:
        warnings.warn(
            f"Dropping {n_dropped}/{len(y)} rows with NaN CV/label/weight.",
            stacklevel=2,
        )

    X_f, y_f = X[finite], y[finite]
    sw = sample_weight[finite] if sample_weight is not None else np.ones(len(y_f))
    groups_f = groups[finite] if groups is not None else None

    shap_values_f = np.full((len(y_f), X_f.shape[1]), np.nan)
    oof_proba_f = np.full(len(y_f), np.nan)
    fold_auc = np.full(n_splits, np.nan)
    fold_roc = []
    perm_importances = []  # SVM only: list of (n_features,) arrays, one per fold

    if groups_f is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_iter = splitter.split(X_f, y_f, groups_f)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_iter = splitter.split(X_f, y_f)

    best_params: dict = {}
    if optimize:
        print(f"  Optimizing hyperparameters ({n_search_iter} random iterations)…")
        best_params = _optimize_hyperparams(
            model_type, X_f, y_f, sw, groups_f,
            n_splits=n_splits, n_jobs=n_jobs,
            random_state=random_state, n_iter=n_search_iter,
        )

    for fold, (train_idx, test_idx) in enumerate(fold_iter):
        scaler = StandardScaler().fit(X_f[train_idx])
        X_train = scaler.transform(X_f[train_idx])
        X_test = scaler.transform(X_f[test_idx])
        if model_type == "lgbm":
            fn = list(feature_names)
            X_train = pd.DataFrame(X_train, columns=fn)
            X_test  = pd.DataFrame(X_test,  columns=fn)

        if model_type == "rf":
            kw = dict(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
            kw.update(best_params)
            model = RandomForestClassifier(**kw)
        elif model_type == "gbm":
            kw = dict(n_estimators=n_estimators, random_state=random_state)
            kw.update(best_params)
            model = GradientBoostingClassifier(**kw)
        elif model_type == "lgbm":
            kw = dict(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs, verbose=-1)
            kw.update(best_params)
            model = lgb.LGBMClassifier(**kw)
        elif model_type == "svm":
            kw = dict(kernel="rbf", max_iter=1000000, random_state=random_state,
                      C=0.5, gamma=0.5, probability=True, class_weight="balanced")
            kw.update(best_params)
            model = SVC(**kw)
        else:  # logreg
            kw = dict(max_iter=1000, random_state=random_state, n_jobs=n_jobs)
            kw.update(best_params)
            model = LogisticRegression(**kw)

        if model_type == "svm" and X_train.shape[0] > _MAX_SVM_TRAIN:
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=_MAX_SVM_TRAIN, random_state=random_state + fold
            )
            sub_idx, _ = next(sss.split(X_train, y_f[train_idx]))
            model.fit(X_train[sub_idx], y_f[train_idx][sub_idx],
                      sample_weight=sw[train_idx][sub_idx])
        else:
            model.fit(X_train, y_f[train_idx], sample_weight=sw[train_idx])

        proba = model.predict_proba(X_test)[:, 1]
        oof_proba_f[test_idx] = proba
        if len(np.unique(y_f[test_idx])) > 1:
            fold_auc[fold] = roc_auc_score(
                y_f[test_idx], proba, sample_weight=sw[test_idx]
            )
            fpr, tpr, _ = roc_curve(y_f[test_idx], proba)
            fold_roc.append((fpr, tpr, float(fold_auc[fold])))

        if model_type == "svm":
            try:
                pi = permutation_importance(
                    model, X_test, y_f[test_idx],
                    scoring="roc_auc",
                    n_repeats=10,
                    random_state=random_state,
                    sample_weight=sw[test_idx],
                    n_jobs=n_jobs,
                )
                perm_importances.append(pi.importances_mean)
            except Exception as exc:
                warnings.warn(
                    f"Fold {fold}: permutation importance failed: {exc}", stacklevel=2
                )
        else:
            try:
                if model_type == "logreg":
                    sv = _linear_shap_values(model, X_train, X_test)
                else:
                    sv = _parallel_shap_values(model, X_test, n_jobs)
            except ExplainerError as exc:
                warnings.warn(
                    f"Fold {fold}: SHAP explanation failed: {exc}", stacklevel=2
                )
                continue
            shap_values_f[test_idx] = sv

    if model_type == "svm":
        if perm_importances:
            mean_abs_shap = np.mean(perm_importances, axis=0)
        else:
            mean_abs_shap = np.zeros(X_f.shape[1])
        shap_values = None
        score_label = "mean permutation importance"
    else:
        mean_abs_shap = np.nanmean(np.abs(shap_values_f), axis=0)
        shap_values = np.full((len(finite), X.shape[1]), np.nan)
        shap_values[finite] = shap_values_f
        score_label = "mean |SHAP|"

    order = np.argsort(mean_abs_shap)[::-1]

    print(
        f"{n_splits}-fold CV ROC AUC: "
        f"{np.nanmean(fold_auc):.3f} +/- {np.nanstd(fold_auc):.3f}"
    )
    print(f"{'rank':>4}  {'CV':<25}  {score_label:>26}")
    for rank, idx in enumerate(order, 1):
        print(f"{rank:>4}  {feature_names[idx]:<25}  {mean_abs_shap[idx]:26.4f}")

    oof_proba = np.full(len(finite), np.nan)
    oof_proba[finite] = oof_proba_f

    oof_true = np.full(len(finite), np.nan)
    oof_true[finite] = y_f

    return shap_values, mean_abs_shap, fold_auc, fold_roc, oof_true, oof_proba


def _plot_shap_summary(shap_values, X, feature_names, out_path, overw=False):
    """Beeswarm summary plot of out-of-fold SHAP values from _rf_shap_kfold."""
    _check_overwrite(out_path, overw)

    finite = np.all(np.isfinite(shap_values), axis=1)
    shap.summary_plot(
        shap_values[finite],
        np.asarray(X, dtype=float)[finite],
        feature_names=feature_names,
        show=False,
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to {out_path}")


def _plot_shap_dependence(shap_values, X, feature_names, out_dir, prefix="", top_n=3, overw=False):
    """Scatter of CV value vs its SHAP value for the top_n most important CVs.

    Each dot is one path; colour encodes the CV value on a coolwarm scale so
    you can read off both direction and magnitude of the effect at a glance.
    """
    mean_abs = np.nanmean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1]

    finite = np.all(np.isfinite(shap_values), axis=1)
    sv = shap_values[finite]
    X_f = np.asarray(X, dtype=float)[finite]

    for rank, idx in enumerate(order[:top_n]):
        fname = feature_names[idx]
        safe_name = fname.replace("/", "_").replace(" ", "_")
        out_path = str(Path(out_dir) / f"{prefix}shap_dep_{safe_name}.png")
        _check_overwrite(out_path, overw)

        fig, ax = plt.subplots(figsize=(5, 4))
        sc = ax.scatter(
            X_f[:, idx], sv[:, idx],
            c=X_f[:, idx], cmap="coolwarm", alpha=0.5, s=8, rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label=fname)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(fname)
        ax.set_ylabel(f"SHAP value for {fname}")
        ax.set_title(f"SHAP Dependence: {fname} (rank {rank + 1})")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"SHAP dependence plot saved to {out_path}")


def _plot_roc_curves(fold_roc, out_path, overw=False):
    """Per-fold ROC curves; title shows mean ± std AUC across folds."""
    _check_overwrite(out_path, overw)
    if not fold_roc:
        warnings.warn("No fold ROC data available — skipping ROC plot.", stacklevel=2)
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    aucs = []
    for i, (fpr, tpr, auc) in enumerate(fold_roc):
        ax.plot(fpr, tpr, alpha=0.5, lw=1.2, label=f"Fold {i + 1} (AUC={auc:.3f})")
        aucs.append(auc)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    mean_auc = float(np.nanmean(aucs))
    std_auc = float(np.nanstd(aucs))
    ax.set_title(f"ROC curves — mean AUC = {mean_auc:.3f} ± {std_auc:.3f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved to {out_path}")


def _plot_calibration(oof_true, oof_proba, out_path, n_bins=10, overw=False):
    """Reliability diagram: does predicted P(reactive) match observed frequency?

    Top panel: calibration curve vs perfect diagonal.
    Bottom panel: histogram of predicted probabilities (shows whether the model
    is over-confident, under-confident, or well-spread).
    """
    _check_overwrite(out_path, overw)
    finite = np.isfinite(oof_true) & np.isfinite(oof_proba)
    if int(np.sum(finite)) < n_bins * 2:
        warnings.warn("Too few OOF samples for calibration plot — skipping.", stacklevel=2)
        return

    fraction_pos, mean_pred = calibration_curve(
        oof_true[finite].astype(int), oof_proba[finite],
        n_bins=n_bins, strategy="quantile",
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(5, 6), gridspec_kw={"height_ratios": [3, 1]}
    )
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    ax.plot(mean_pred, fraction_pos, "o-", color="#d62728", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration (reliability diagram)")
    ax.legend(fontsize=8)

    axes[1].hist(oof_proba[finite], bins=30, color="#1f77b4", alpha=0.7)
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Calibration plot saved to {out_path}")


def shap_ml(
    toml: Annotated[str, typer.Option("-toml", help="The .toml file")] = "infretis.toml",
    data: Annotated[str, typer.Option("-data", help="The infretis_data.txt file")] = "infretis_data.txt",
    cv_dir: Annotated[str, typer.Option("-cv-dir", help="Path data folder with CV values in .txt files")] = "ML",
    op_col: Annotated[str, typer.Option("-op-col", help="Order-parameter column name")] = "OP_Lamb",
    cv_cols: Annotated[Optional[str], typer.Option("-cv-cols", help="Comma-separated CV columns to use; default = all except -op-col")] = None,
    exclude: Annotated[Optional[str], typer.Option("-exclude", help="Comma-separated substrings; CVs whose name matches one are dropped (only applied when -cv-cols is unset)")] = None,
    angle_cols: Annotated[Optional[str], typer.Option("-angle-cols", help="Comma-separated CV columns in degrees to convert to cos(θ); use for asymmetric molecules where 0° and 180° are distinct orientations")] = None,
    sym_angle_cols: Annotated[Optional[str], typer.Option("-sym-angle-cols", help="Comma-separated CV columns in degrees to convert to cos²(θ); use for head-tail symmetric molecules where θ and 180°−θ are equivalent")] = None,
    nskip: Annotated[int, typer.Option("-nskip", help="Skip the first nskip rows of infretis_data.txt")] = 1000,
    models: Annotated[str, typer.Option("-models", help=f"Comma-separated models to run; choices: {', '.join(MODEL_CHOICES)}")] = "rf,gbm,lgbm,logreg,svm",
    n_splits: Annotated[int, typer.Option("-n-splits", help="Number of stratified CV folds")] = 5,
    n_estimators: Annotated[int, typer.Option("-n-estimators", help="Trees per RandomForest / GradientBoosting / LightGBM")] = 300,
    n_jobs: Annotated[int, typer.Option("-n-jobs", help="CPU cores for RF, LGBM, and LogReg; -1 = all (sklearn GBM is always single-threaded)")] = -1,
    seed: Annotated[int, typer.Option("-seed", help="Random seed for fold splits and models")] = 42,
    plot_dir: Annotated[str, typer.Option("-plot-dir", help="Root directory for plots; each model gets a subdirectory")] = "shap_ml_plots",
    top_n: Annotated[int, typer.Option("-top-n", help="Number of top CVs for SHAP dependence plots per interface")] = 3,
    out: Annotated[str, typer.Option("-out", help="Base name for ranking files; '_<model>.txt' is appended")] = "shap_ranking.txt",
    drop_z_ref: Annotated[bool, typer.Option("-drop-z-ref", help="Remove the z-reference column (z_Memb) from the feature set after z-corrections are applied; the column is still used as the reference during correction")] = False,
    optimize: Annotated[bool, typer.Option("-optimize", help="Run random hyperparameter search before the main k-fold loop; best params override -n-estimators and model defaults")] = False,
    n_search_iter: Annotated[int, typer.Option("-n-search-iter", help="Number of random hyperparameter configurations to evaluate when -optimize is set")] = 20,
    overw: Annotated[bool, typer.Option("-O", help="Force overwriting of existing files")] = False,
):
    """Per-interface SHAP feature-importance analysis across one or more classifiers.

    Runs the following models (controlled via -models, default = all four):

      rf     — RandomForestClassifier.  Bagging ensemble; captures non-linear
               and interaction effects.  SHAP via TreeExplainer (parallelised
               across CPU cores by splitting test rows).

      gbm    — GradientBoostingClassifier.  Boosting ensemble; residual-
               correction gives different inductive bias than RF.  Same
               TreeExplainer path.  If RF and GBM agree on CV rankings the
               result is robust; disagreement flags instability worth
               investigating.

      lgbm   — LightGBMClassifier.  Histogram-based gradient boosting; faster
               and more memory-efficient than GBM for large datasets.  Supports
               parallel training (n_jobs).  SHAP via TreeExplainer.

      logreg — LogisticRegression.  Linear baseline; SHAP via LinearExplainer
               (analytically exact, no parallelisation needed).  Comparing
               AUC with the tree models tells you whether the CV--label
               relationship at each interface is linear (LR ≈ trees) or
               non-linear (LR << trees).

    For each model, data is loaded once; results go to:
      <plot-dir>/<model>/interface_NNN_*.png  (beeswarm, bar, dependence, ROC,
                                              calibration, heatmap)
      shap_ranking_<model>.txt

    Interfaces with fewer than -n-splits paths in either class are skipped.
    """
    z_cols_list   = [
        "z_NTop", "z_NBot", "z_PTop", "z_PBot",
        "z_O2_T", "z_O2_B", "z_O3_T", "z_O3_B",
        "z_C2_T", "z_C2_B", "z_C3_T", "z_C3_B",
    ]
    model_list = [m.strip() for m in models.split(",")]
    unknown = [m for m in model_list if m not in MODEL_CHOICES]
    if unknown:
        raise typer.BadParameter(
            f"Unknown model(s): {unknown}. Available: {list(MODEL_CHOICES)}"
        )

    with open(toml, "rb") as f:
        cfg = tomli.load(f)
    interfaces = np.asarray(cfg["simulation"]["interfaces"], dtype=float)
    M = len(interfaces)

    pnr, maxop, path_f, path_w = _load_path_table(data, nskip, M)
    path_weights = _compute_path_weights(maxop, path_f, path_w, interfaces)

    # Split once: both the column discovery and the z-correction need the
    # substrings as a list, not as the raw comma-separated string.
    exclude_list = exclude.split(",") if exclude else None
    cv_array, cv_names, _ = _extract_cv_crossings(
        cv_dir=cv_dir,
        pnr_expected=pnr,
        subgrid=interfaces,
        op_col=op_col,
        cv_cols=cv_cols.split(",") if cv_cols else None,
        exclude=exclude_list,
    )
    cv_array, cv_names = _apply_angle_transforms(
        cv_array, cv_names,
        cos_cols=angle_cols.split(",") if angle_cols else None,
        cos2_cols=sym_angle_cols.split(",") if sym_angle_cols else None,
    )
    cv_array, cv_names = _apply_z_corrections(
        cv_array, cv_names,
        z_cols=z_cols_list,
        exclude_list=exclude_list,
        z_ref="z_Memb",
        drop_ref=drop_z_ref,
    )
    labels, is_plus = _extract_path_metadata(cv_dir, pnr)

    cv_array, labels, path_weights = (
        cv_array[is_plus], labels[is_plus], path_weights[is_plus]
    )
    N_cvs = len(cv_names)
    print(f"{int(np.sum(is_plus))} plus-ensemble paths  |  {N_cvs} CVs  |  {M} interfaces.")

    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    min_per_class = max(n_splits, 2)
    out_stem = Path(out).stem
    out_suffix = Path(out).suffix

    for model_type in model_list:
        label = _MODEL_LABELS[model_type]
        sep = "=" * 62
        print(f"\n{sep}\n  Model: {label}\n{sep}")

        model_plot_dir = Path(plot_dir) / model_type
        model_plot_dir.mkdir(parents=True, exist_ok=True)
        model_out = str(Path(out).parent / f"{out_stem}_{model_type}{out_suffix}")
        _check_overwrite(model_out, overw)

        results = []
        for i in range(M):
            start = datetime.datetime.now()
            X_i = cv_array[:, :, i]

            finite = np.all(np.isfinite(X_i), axis=1) & np.isfinite(labels)
            n_pos = int(np.sum(labels[finite] == 1))
            n_neg = int(np.sum(labels[finite] == 0))
            print(
                f"\n[interface {i + 1:3d}/{M}] lambda = {interfaces[i]:.4f}  "
                f"({n_pos} reactive, {n_neg} non-reactive)"
            )

            if min(n_pos, n_neg) < min_per_class:
                print(f"  SKIP: fewer than {min_per_class} paths in one class.")
                results.append({"interface": i, "lambda": interfaces[i], "ranking": None})
                continue

            shap_values, mean_abs_shap, fold_auc, fold_roc, oof_true, oof_proba = (
                _model_shap_kfold(
                    X_i, labels, cv_names,
                    model_type=model_type,
                    sample_weight=path_weights,
                    n_splits=n_splits,
                    n_estimators=n_estimators,
                    n_jobs=n_jobs,
                    random_state=seed,
                    optimize=optimize,
                    n_search_iter=n_search_iter,
                )
            )

            prefix = f"interface_{i:03d}_"

            if shap_values is not None:
                _plot_shap_summary(
                    shap_values, X_i, cv_names,
                    str(model_plot_dir / f"{prefix}shap_beeswarm.png"), overw=overw,
                )
            _plot_importance_bar(
                mean_abs_shap, cv_names,
                str(model_plot_dir / f"{prefix}shap_bar.png"), overw=overw,
                xlabel="Mean permutation importance" if model_type == "svm" else "Mean |SHAP value|",
                title="Feature Importance (permutation)" if model_type == "svm" else "SHAP Feature Importance",
            )
            if shap_values is not None:
                _plot_shap_dependence(
                    shap_values, X_i, cv_names, str(model_plot_dir),
                    prefix=prefix, top_n=top_n, overw=overw,
                )
            _plot_roc_curves(
                fold_roc, str(model_plot_dir / f"{prefix}roc.png"), overw=overw,
            )
            _plot_calibration(
                oof_true, oof_proba,
                str(model_plot_dir / f"{prefix}calibration.png"), overw=overw,
            )

            order = np.argsort(mean_abs_shap)[::-1]
            results.append({
                "interface": i,
                "lambda": interfaces[i],
                "ranking": [(cv_names[idx], float(mean_abs_shap[idx])) for idx in order],
                "mean_fold_auc": float(np.nanmean(fold_auc)),
            })
            end = datetime.datetime.now()
            print(f"Interface {i + 1}/{M} done in {end - start}.")

        with open(model_out, "w") as f:
            f.write("# model\tinterface\tlambda\trank\tCV\tmean_abs_shap\tmean_fold_AUC\n")
            for r in results:
                if r["ranking"] is None:
                    continue
                for rank, (cv_name, val) in enumerate(r["ranking"], 1):
                    f.write(
                        f"{model_type}\t{r['interface']}\t{r['lambda']:.6f}\t{rank}\t"
                        f"{cv_name}\t{val:.6f}\t{r['mean_fold_auc']:.4f}\n"
                    )
        print(f"\nRanking saved to {model_out}.")

        _plot_interface_heatmap(
            results, cv_names,
            str(model_plot_dir / "interface_heatmap.png"), overw=overw,
        )


