"""The SVM can be fitted by cuML on the GPU instead of scikit-learn.

The SVM path is dominated by permutation_importance, which is predict-bound,
so a GPU backend attacks the right cost. It is only usable if the resulting
importance rankings match the CPU ones.
"""

import numpy as np
import pytest

from chiroflux.shap_analysis import SVM_DEVICE_CHOICES, _as_svm_array, _make_svc


def _cuml_available():
    try:
        import cuml.svm  # noqa: F401

        return True
    except Exception:
        return False


class TestMakeSVC:
    def test_cpu_returns_sklearn(self):
        from sklearn.svm import SVC

        model, on_gpu = _make_svc("cpu", kernel="rbf", C=0.5, probability=False)
        assert isinstance(model, SVC)
        assert on_gpu is False

    def test_cpu_keeps_the_probability_argument(self):
        """sklearn accepts it; only the cuML branch has to drop it."""
        model, _ = _make_svc("cpu", kernel="rbf", probability=False)
        assert model.probability is False

    def test_auto_falls_back_when_cuml_is_missing(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def no_cuml(name, *a, **k):
            if name.startswith("cuml"):
                raise ImportError("No module named 'cuml'")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_cuml)
        with pytest.warns(UserWarning, match="cuML unavailable"):
            model, on_gpu = _make_svc("auto", kernel="rbf", probability=False)
        assert on_gpu is False

    def test_explicit_gpu_raises_when_cuml_is_missing(self, monkeypatch):
        """-svm-device gpu asserts something about the machine; silently using
        the CPU would hide a broken setup behind a long runtime."""
        import builtins

        real_import = builtins.__import__

        def no_cuml(name, *a, **k):
            if name.startswith("cuml"):
                raise ImportError("No module named 'cuml'")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_cuml)
        with pytest.raises(RuntimeError, match="svm-device gpu was requested"):
            _make_svc("gpu", kernel="rbf", probability=False)

    def test_device_choices_are_the_documented_three(self):
        assert SVM_DEVICE_CHOICES == ("auto", "gpu", "cpu")


class TestSvmArrayCast:
    def test_gpu_path_casts_to_float32(self):
        a = np.arange(6, dtype=np.float64).reshape(3, 2)
        out = _as_svm_array(a, on_gpu=True)
        assert out.dtype == np.float32 and out.flags["C_CONTIGUOUS"]

    def test_cpu_path_leaves_the_array_alone(self):
        a = np.arange(6, dtype=np.float64)
        assert _as_svm_array(a, on_gpu=False) is a


@pytest.mark.skipif(not _cuml_available(), reason="cuML not installed")
class TestCumlBackend:
    def test_gpu_and_cpu_svms_rank_features_the_same(self):
        """The whole point: same conclusions, far faster. cuML has no
        predict_proba, so both paths score through decision_function."""
        from scipy.stats import spearmanr
        from sklearn.inspection import permutation_importance

        rng = np.random.default_rng(0)
        X = rng.normal(size=(1200, 12))
        y = (2.0 * X[:, 0] + 1.2 * X[:, 1] + rng.normal(size=1200) > 0).astype(int)
        w = 10.0 ** rng.uniform(0, 2, 1200)
        w /= w.mean()

        importances = {}
        for device in ("cpu", "gpu"):
            model, on_gpu = _make_svc(
                device, kernel="rbf", C=0.5, gamma="scale",
                probability=False, class_weight="balanced",
            )
            Xa = _as_svm_array(X, on_gpu)
            model.fit(Xa, _as_svm_array(y, on_gpu),
                      sample_weight=_as_svm_array(w, on_gpu))
            pi = permutation_importance(
                model, Xa, _as_svm_array(y, on_gpu), scoring="roc_auc",
                n_repeats=3, n_jobs=1, random_state=0,
            )
            importances[device] = pi.importances_mean

        rho = spearmanr(importances["cpu"], importances["gpu"]).statistic
        assert rho > 0.9, f"GPU and CPU rankings disagree (Spearman {rho:.3f})"
        top_cpu = set(np.argsort(importances["cpu"])[-2:])
        top_gpu = set(np.argsort(importances["gpu"])[-2:])
        assert top_cpu == top_gpu == {0, 1}

    def test_cuml_estimator_exposes_decision_function(self):
        """decision_function is all the SVM path needs, and it is the only
        scoring method cuML offers consistently: 26.2 has predict_proba while
        26.8 dropped it along with the `probability` parameter. Probabilities
        come from _platt_calibrate_oof either way, so the path does not care
        which version is installed - but it does require decision_function."""
        model, on_gpu = _make_svc("gpu", kernel="rbf", probability=False)
        assert on_gpu is True
        assert hasattr(model, "decision_function")

    def test_probability_is_stripped_for_cuml(self):
        """cuML 26.8 rejects `probability` outright; 26.2 accepts it. _make_svc
        drops it for both, which is required for the former and harmless for
        the latter since the SVM path never calls predict_proba."""
        import inspect

        from cuml.svm import SVC as cuSVC

        model, on_gpu = _make_svc("gpu", kernel="rbf", C=0.5, probability=True)
        assert on_gpu is True
        if "probability" in inspect.signature(cuSVC.__init__).parameters:
            assert model.probability is False, "must not inherit probability=True"
