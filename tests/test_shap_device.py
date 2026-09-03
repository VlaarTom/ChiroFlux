"""GPU TreeSHAP must be a drop-in for the CPU explainer.

Explaining, not fitting, dominates a fold: a 300-tree forest measured ~1s to
fit and ~177s to explain. shap ships a CUDA implementation of the same Tree
SHAP algorithm, so the GPU path exists purely to attack that, and it is only
worth having if it returns the same numbers.
"""

import numpy as np
import pytest

from chiroflux.shap_analysis import (
    SHAP_DEVICE_CHOICES,
    _parallel_shap_values,
    _positive_class_shap,
)


@pytest.fixture(scope="module")
def forest():
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 8))
    y = (X[:, 0] + 0.5 * rng.normal(size=400) > 0).astype(int)
    model = RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=1).fit(X, y)
    return model, X[:40]


def _gpu_available(model, X):
    import shap

    try:
        shap.explainers.GPUTree(model, data=None).shap_values(
            X[:1], check_additivity=False
        )
        return True
    except Exception:
        return False


class TestPositiveClassShap:
    """shap's return shape varies by version; the analyses want one array."""

    def test_unpacks_the_two_class_list_form(self):
        a, b = np.zeros((5, 3)), np.ones((5, 3))
        assert _positive_class_shap([a, b]) is b

    def test_slices_the_three_dimensional_form(self):
        sv = np.arange(5 * 3 * 2).reshape(5, 3, 2)
        assert np.array_equal(_positive_class_shap(sv), sv[:, :, 1])

    def test_passes_a_plain_matrix_through(self):
        sv = np.arange(15).reshape(5, 3)
        assert np.array_equal(_positive_class_shap(sv), sv)


class TestShapDevice:
    def test_cpu_path_is_the_reference(self, forest):
        model, X = forest
        sv = _parallel_shap_values(model, X, n_jobs=1, device="cpu")
        assert sv.shape == X.shape

    def test_gpu_matches_cpu_when_a_gpu_is_present(self, forest):
        """The whole point: same attributions, faster. Skipped without CUDA."""
        model, X = forest
        if not _gpu_available(model, X):
            pytest.skip("no usable CUDA TreeSHAP on this machine")
        cpu = _parallel_shap_values(model, X, n_jobs=1, device="cpu")
        gpu = _parallel_shap_values(model, X, n_jobs=1, device="gpu")
        assert gpu.shape == cpu.shape
        # float32 on the GPU against float64 on the CPU
        assert np.allclose(cpu, gpu, atol=1e-4), np.abs(cpu - gpu).max()

    def test_auto_falls_back_when_the_gpu_path_raises(self, forest, monkeypatch):
        """A missing or broken CUDA build must degrade to CPU, not crash."""
        model, X = forest
        import chiroflux.shap_analysis as sa

        def boom(*a, **k):
            raise RuntimeError("no CUDA device")

        monkeypatch.setattr(sa, "_gpu_shap_values", boom)
        with pytest.warns(UserWarning, match="falling back to the CPU"):
            sv = _parallel_shap_values(model, X, n_jobs=1, device="auto")
        expected = _parallel_shap_values(model, X, n_jobs=1, device="cpu")
        assert np.allclose(sv, expected)

    def test_explicit_gpu_raises_instead_of_falling_back(self, forest, monkeypatch):
        """-shap-device gpu is a claim about the machine; silently using the
        CPU would hide a broken setup behind a long runtime."""
        model, X = forest
        import chiroflux.shap_analysis as sa

        def boom(*a, **k):
            raise RuntimeError("no CUDA device")

        monkeypatch.setattr(sa, "_gpu_shap_values", boom)
        with pytest.raises(RuntimeError, match="shap-device gpu was requested"):
            _parallel_shap_values(model, X, n_jobs=1, device="gpu")

    def test_dataframe_input_stays_on_the_cpu(self, forest):
        """Only LightGBM is handed a DataFrame, and shap's GPU TreeSHAP warns
        that categorical features are unsupported there."""
        import pandas as pd

        model, X = forest
        df = pd.DataFrame(X, columns=[f"cv{i}" for i in range(X.shape[1])])
        called = {"gpu": False}
        import chiroflux.shap_analysis as sa

        def spy(*a, **k):
            called["gpu"] = True
            raise AssertionError("GPU path must not run on a DataFrame")

        sa_gpu = sa._gpu_shap_values
        try:
            sa._gpu_shap_values = spy
            sv = _parallel_shap_values(model, df, n_jobs=1, device="auto")
        finally:
            sa._gpu_shap_values = sa_gpu
        assert not called["gpu"]
        assert sv.shape == X.shape

    def test_device_choices_are_the_documented_three(self):
        assert SHAP_DEVICE_CHOICES == ("auto", "gpu", "cpu")
