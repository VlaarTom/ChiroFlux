"""-reweight/-no-reweight must reach the model in every weighted analysis.

With -no-reweight every path counts once (weight 1) instead of by its WHAM
weight. These tests build a tiny synthetic simulation whose WHAM weights are
deliberately non-uniform, run each command up to the point where the
weights are handed to the model/statistics, and check what arrives there.
"""

import inspect

import numpy as np
import pytest

from chiroflux.pathdata import _compute_path_weights, _load_path_table

INTERFACES = [0.0, 1.0, 2.0]

# pnr, maxop, reactive, f (minus, ens1, ens2), w (minus, ens1, ens2).
# Unequal path_w / maxop make the WHAM weights differ between paths.
_PATHS = [
    (1, 1.5, False, ("----", 1.0, "----"), ("----", 1.0, "----")),
    (2, 2.5, True, ("----", 1.0, 1.0), ("----", 2.0, 1.0)),
    (3, 1.2, False, ("----", 1.0, "----"), ("----", 4.0, "----")),
    (4, 2.2, True, ("----", 1.0, 1.0), ("----", 1.0, 3.0)),
    (5, 1.8, False, ("----", 1.0, 1.0), ("----", 3.0, 2.0)),
    (6, 2.8, True, ("----", 1.0, 1.0), ("----", 1.5, 1.0)),
]


def _write_sim(root):
    """Write infretis.toml, infretis_data.txt and ML/<pnr>.txt under root."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "infretis.toml").write_text(
        f"[simulation]\ninterfaces = {INTERFACES}\n"
    )
    rows = []
    for pnr, maxop, _, f, w in _PATHS:
        rows.append(" ".join(str(v) for v in (pnr, 10, maxop, *f, *w)))
    (root / "infretis_data.txt").write_text("\n".join(rows) + "\n")

    ml = root / "ML"
    ml.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for pnr, maxop, reactive, _, _ in _PATHS:
        op = np.linspace(-0.2, maxop, 8)
        lines = [
            "# reactive" if reactive else "# non-reactive",
            "# plus ensemble",
            "# info",
            "OP_Lamb cvA cvB z_Memb",
        ]
        for v in op:
            lines.append(f"{v:.4f} {rng.normal():.4f} {rng.normal():.4f} 0.0")
        (ml / f"{pnr}.txt").write_text("\n".join(lines) + "\n")
    return root


def _wham_weights(root):
    pnr, maxop, path_f, path_w = _load_path_table(
        str(root / "infretis_data.txt"), 0, len(INTERFACES)
    )
    return pnr, _compute_path_weights(maxop, path_f, path_w, np.asarray(INTERFACES))


@pytest.fixture
def sim(tmp_path):
    root = _write_sim(tmp_path / "sim")
    pnr, w = _wham_weights(root)
    assert np.ptp(w) > 0, "fixture must have non-uniform WHAM weights"
    return root, pnr, w


class _Captured(Exception):
    """Raised by a stub to stop the command once the weights are captured."""


# ── Every weighted command exposes the flag, defaulting to weighted ────────

@pytest.mark.parametrize("module_name, function_name", [
    ("shap_analysis", "shap_ml"),
    ("shap_analysis_ld", "shap_enantiomer"),
    ("statistical_analysis", "statistics"),
    ("principal_component_analysis", "PCA"),
    ("prepare_deeptda_data", "prepare_deeptda_data"),
    ("prepare_deeptda_data", "prepare_deeptda_data_ld"),
    ("train_deeptda", "train_deeptda"),
])
def test_command_has_reweight_flag(module_name, function_name):
    from importlib import import_module

    fn = getattr(import_module(f"chiroflux.{module_name}"), function_name)
    param = inspect.signature(fn).parameters["reweight"]
    assert param.default is True
    option = param.annotation.__metadata__[0]
    # Annotated typer options keep their first declaration in `default`
    assert "-reweight/-no-reweight" in (option.default, *option.param_decls)


# ── shap-ml ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("reweight", [True, False])
def test_shap_ml_passes_the_requested_weights(sim, monkeypatch, tmp_path, reweight):
    from chiroflux import shap_analysis

    root, _, wham = sim
    seen = {}

    def fake_kfold(X, y, names, sample_weight=None, **kw):
        seen["w"] = np.asarray(sample_weight)
        raise _Captured

    monkeypatch.setattr(shap_analysis, "_model_shap_kfold", fake_kfold)
    with pytest.raises(_Captured):
        shap_analysis.shap_ml(
            toml=str(root / "infretis.toml"), data=str(root / "infretis_data.txt"),
            cv_dir=str(root / "ML"), nskip=0, n_splits=2, models="logreg",
            reweight=reweight, out=str(tmp_path / "r.txt"),
            plot_dir=str(tmp_path / "plots"),
        )
    expected = wham if reweight else np.ones_like(wham)
    np.testing.assert_allclose(seen["w"], expected)


@pytest.mark.parametrize("model_type", ["logreg", "lgbm"])
def test_fit_does_not_depend_on_the_overall_weight_scale(model_type):
    """WHAM weights sum to 1 (and to far less at late interfaces). Only the
    relative weights may matter: at ~1e-5 per path LightGBM used to fit a
    constant model and logreg was heavily over-regularised."""
    from chiroflux.shap_analysis import _model_shap_kfold

    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3))
    y = (X[:, 0] + 0.5 * rng.normal(size=600) > 0).astype(float)
    w = rng.lognormal(sigma=1.5, size=600)

    runs = [
        _model_shap_kfold(X, y, ["a", "b", "c"], model_type=model_type,
                          sample_weight=w * scale, n_splits=3, n_estimators=50,
                          n_jobs=1, shap_device="cpu")
        for scale in (1e-6 / w.sum(), 1.0 / w.mean())
    ]
    (_, imp_small, auc_small, *_), (_, imp_mean1, auc_mean1, *_) = runs
    np.testing.assert_allclose(auc_small, auc_mean1, rtol=1e-6)
    np.testing.assert_allclose(imp_small, imp_mean1, rtol=1e-5, atol=1e-8)
    assert np.nanmean(auc_small) > 0.8  # a real model, not a constant one


# ── shap-enantiomer ────────────────────────────────────────────────────────

@pytest.mark.parametrize("reweight", [True, False])
def test_shap_enantiomer_passes_the_requested_weights(tmp_path, monkeypatch, reweight):
    from chiroflux import shap_analysis_ld

    dir_l = _write_sim(tmp_path / "L")
    dir_d = _write_sim(tmp_path / "D")
    _, wham = _wham_weights(dir_l)
    seen = {}

    def fake_kfold(X, y, names, sample_weight=None, **kw):
        seen["w"] = np.asarray(sample_weight)
        raise _Captured

    monkeypatch.setattr(shap_analysis_ld, "_model_shap_kfold", fake_kfold)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(_Captured):
        shap_analysis_ld.shap_enantiomer(
            dir_l=str(dir_l), dir_d=str(dir_d), nskip=0, n_splits=2,
            models="logreg", reweight=reweight,
        )
    # one row per (path, interface), L then D
    per_path = wham if reweight else np.ones_like(wham)
    expected = np.tile(np.repeat(per_path, len(INTERFACES)), 2)
    np.testing.assert_allclose(seen["w"], expected)


# ── statistics ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("reweight", [True, False])
def test_statistics_passes_the_requested_weights(sim, monkeypatch, tmp_path, reweight):
    from chiroflux import statistical_analysis

    root, _, wham = sim
    seen = {}

    def fake_metrics(X, y, weights, names):
        seen["w"] = np.asarray(weights)
        raise _Captured

    monkeypatch.setattr(statistical_analysis, "_compute_importance_metrics", fake_metrics)
    with pytest.raises(_Captured):
        statistical_analysis.statistics(
            toml=str(root / "infretis.toml"), data=str(root / "infretis_data.txt"),
            cv_dir=str(root / "ML"), nskip=0, reweight=reweight,
            out=str(tmp_path / "r.txt"), plot_dir=str(tmp_path / "plots"),
        )
    expected = wham if reweight else np.ones_like(wham)
    np.testing.assert_allclose(seen["w"], expected)


# ── DeepTDA data prep ──────────────────────────────────────────────────────

def _load_frames(root, reweight):
    from chiroflux.prepare_deeptda_data import _load_sim_frames

    return _load_sim_frames(
        toml=str(root / "infretis.toml"), data=str(root / "infretis_data.txt"),
        cv_dir=str(root / "ML"), nskip=0, op_col="OP_Lamb", cv_cols=None,
        stride=1, max_frames_per_path=None, force_proximity=True,
        reweight=reweight,
    )


def test_deeptda_frames_are_path_weight_times_proximity(sim):
    root, pnr, wham = sim
    _, w, prox, pnr_frame, *_ = _load_frames(root, reweight=True)
    path_w = dict(zip(pnr, wham))
    expected = np.array([path_w[p] for p in pnr_frame]) * prox
    np.testing.assert_allclose(w, expected)


def test_deeptda_no_reweight_keeps_only_the_proximity_ramp(sim):
    root, *_ = sim
    _, w, prox, *_ = _load_frames(root, reweight=False)
    np.testing.assert_allclose(w, prox)
    assert np.ptp(prox) > 0  # the ramp itself survives


def test_deeptda_proximity_column_does_not_depend_on_reweight(sim):
    root, *_ = sim
    prox_on = _load_frames(root, reweight=True)[2]
    prox_off = _load_frames(root, reweight=False)[2]
    np.testing.assert_array_equal(prox_on, prox_off)


def test_deeptda_ld_saves_consistent_weight_columns(tmp_path):
    """Dropping path weights at train time (w_prox of a weighted dataset)
    must equal building the dataset unweighted in the first place."""
    from chiroflux.prepare_deeptda_data import prepare_deeptda_data_ld

    dir_l = _write_sim(tmp_path / "L")
    dir_d = _write_sim(tmp_path / "D")
    common = dict(dir_l=str(dir_l), dir_d=str(dir_d), nskip=0,
                  lambda_low=0.0, lambda_high=1.0)
    prepare_deeptda_data_ld(**common, reweight=True, out=str(tmp_path / "on.npz"))
    prepare_deeptda_data_ld(**common, reweight=False, out=str(tmp_path / "off.npz"))
    on, off = np.load(tmp_path / "on.npz"), np.load(tmp_path / "off.npz")

    assert bool(on["reweight"]) and not bool(off["reweight"])
    assert not np.allclose(on["w"], off["w"])
    np.testing.assert_allclose(on["w_prox"], off["w"])
    np.testing.assert_allclose(off["w_prox"], off["w"])
