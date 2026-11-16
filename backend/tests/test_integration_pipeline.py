"""Pruebas de integración: datos → optimizador → métricas → servicio."""
from __future__ import annotations

import numpy as np
import pytest

from app.evaluation.metrics import classification_report, confusion, roc_auc, roc_curve
from app.services.serialization import downsample_indices, to_jsonable

# ------------------------------------------------------------------- datos


def test_split_sizes_labels_and_stratification(ds):
    assert ds.X_train.shape == (455, 30) and ds.X_test.shape == (114, 30)
    y_all = np.concatenate([ds.y_train, ds.y_test])
    assert int(y_all.sum()) == 212  # y = 1 ⇔ maligno
    assert abs(ds.y_train.mean() - ds.y_test.mean()) < 0.01  # partición estratificada
    assert set(np.unique(y_all)) == {0.0, 1.0}
    assert len(set(ds.idx_train) & set(ds.idx_test)) == 0


def test_standardization_uses_train_statistics_only(ds):
    np.testing.assert_allclose(ds.X_train.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(ds.X_train.std(axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(ds.standardize(ds.X_test_raw), ds.X_test, rtol=1e-12)
    assert np.max(np.abs(ds.X_test.mean(axis=0))) > 1e-3  # test NO está centrado: no hay fuga


# ---------------------------------------------------------------- métricas


def test_confusion_and_report_on_handcrafted_case():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.4, 0.7, 0.2, 0.1, 0.3, 0.05])
    assert confusion(y, p >= 0.5) == {"tp": 2, "tn": 4, "fp": 1, "fn": 1}
    rep = classification_report(y, p)
    assert rep["accuracy"] == pytest.approx(6 / 8)
    assert rep["sensitivity"] == pytest.approx(2 / 3)
    assert rep["specificity"] == pytest.approx(4 / 5)
    assert rep["precision"] == pytest.approx(2 / 3)


def test_auc_limits():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert roc_auc(y, np.full(4, 0.5)) == 0.5
    assert np.isnan(roc_auc(np.ones(4), np.arange(4.0)))


def test_roc_curve_endpoints():
    y = np.array([0, 1, 0, 1, 1])
    r = roc_curve(y, np.array([0.2, 0.7, 0.4, 0.9, 0.1]))
    assert (r["fpr"][0], r["tpr"][0]) == (0.0, 0.0)  # umbral 1.0: ningún score < 1 se clasifica positivo
    assert (r["fpr"][-1], r["tpr"][-1]) == (1.0, 1.0)


# ----------------------------------------------------------- serialización


def test_to_jsonable_maps_non_finite_to_none():
    out = to_jsonable({"a": np.array([1.0, np.nan, np.inf, -np.inf]), "b": np.int64(3), "c": np.bool_(True)})
    assert out == {"a": [1.0, None, None, None], "b": 3, "c": True}


def test_downsample_keeps_endpoints():
    idx = downsample_indices(10_001, 100)
    assert idx[0] == 0 and idx[-1] == 10_000 and len(idx) <= 100
    assert np.array_equal(downsample_indices(5, 100), np.arange(5))


# ------------------------------------------------------- servicio completo


def test_reported_test_metrics_match_independent_recomputation(service, ds):
    info = service.model_info()
    theta = service.active.theta
    p = 1.0 / (1.0 + np.exp(-(ds.X_test @ theta[:-1] + theta[-1])))  # rango seguro: |z| < 700
    acc = np.mean((p >= 0.5) == ds.y_test)
    assert info["metrics"]["test"]["accuracy"] == pytest.approx(acc)


def test_predict_every_test_sample_is_consistent(service, ds):
    preds = [service.predict(sample_index=i) for i in range(ds.X_test.shape[0])]
    for i, pr in enumerate(preds):
        assert pr["true_label"] == int(ds.y_test[i])
        assert pr["prediction"] == int(pr["probability_malignant"] >= 0.5)


def test_invalid_requests_raise_value_error(service):
    from app.services.model_service import TrainRequestDTO

    with pytest.raises(ValueError):
        service.predict(sample_index=10_000)
    with pytest.raises(ValueError):
        service.train(TrainRequestDTO(mode="otro"))
    with pytest.raises(ValueError):
        service.step_sweep(exp_min=-2, exp_max=-5)
