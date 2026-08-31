"""Guards against CPU over-subscription in the SHAP analyses.

The failure mode these pin down: passing the same -n-jobs to an outer
parallel loop and to the estimator inside it multiplies rather than adds, so
-n-jobs 28 asks the machine for ~784 workers. OMP_NUM_THREADS does not bound
it, because scikit-learn's n_jobs is joblib, not OpenMP.
"""

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "chiroflux"


def test_hpo_pins_the_inner_estimator_to_one_job(monkeypatch):
    """RandomizedSearchCV owns the parallelism; the estimator must not also
    claim n_jobs, or the two multiply."""
    import sklearn.model_selection

    from chiroflux import shap_analysis as sa

    captured = {}

    class FakeSearch:
        def __init__(self, estimator, **kw):
            captured["estimator"] = estimator
            captured["search_n_jobs"] = kw.get("n_jobs")

        def fit(self, *a, **k):
            self.best_score_, self.best_params_ = 0.5, {}

    # _optimize_hyperparams imports RandomizedSearchCV inside the function
    # body, so it has to be replaced at the source module.
    monkeypatch.setattr(
        sklearn.model_selection, "RandomizedSearchCV", FakeSearch
    )
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = (rng.random(40) > 0.5).astype(int)

    for model_type in ("rf", "lgbm", "logreg"):
        captured.clear()
        sa._optimize_hyperparams(
            model_type, X, y, np.ones(40), None,
            n_splits=2, n_jobs=28, random_state=0, n_iter=2,
        )
        assert captured["search_n_jobs"] == 28, model_type
        assert captured["estimator"].n_jobs == 1, (
            f"{model_type}: inner estimator kept n_jobs="
            f"{captured['estimator'].n_jobs}; 28 x 28 workers"
        )


def test_single_threaded_workers_is_a_usable_context_manager():
    from chiroflux.shap_analysis import _single_threaded_workers

    with _single_threaded_workers() as cfg:
        assert cfg is not None


@pytest.mark.parametrize("module", ["shap_analysis", "shap_analysis_ld"])
def test_import_does_not_mutate_thread_env_vars(module):
    """Setting OMP_NUM_THREADS at import time is a side effect on every
    importer, and silently does nothing if NumPy was already imported - so it
    must not come back. Thread limits belong at the call site."""
    code = (
        "import os; "
        "before = {k: os.environ.get(k) for k in "
        "('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}; "
        f"import chiroflux.{module}; "
        "after = {k: os.environ.get(k) for k in before}; "
        "print('CHANGED' if before != after else 'OK')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "OK", f"{module} mutated thread env vars at import"


def test_every_process_parallel_call_limits_inner_threads():
    """Each joblib fan-out point must sit inside _single_threaded_workers(),
    or its workers each open a full-width BLAS/OpenMP pool."""
    tree = ast.parse((SRC / "shap_analysis.py").read_text())

    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            names = {
                n.func.id
                for item in node.items
                for n in [item.context_expr]
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            if "_single_threaded_workers" in names:
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                        guarded.add(inner.func.id)
                    elif isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                        guarded.add(inner.func.attr)

    for fanout in ("Parallel", "permutation_importance", "fit"):
        assert fanout in guarded, f"{fanout}() is not inside _single_threaded_workers()"
