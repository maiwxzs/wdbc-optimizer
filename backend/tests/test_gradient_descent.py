import numpy as np
import pytest

from app.math_engine import LogisticObjective, sigmoid
from app.optimization import GDConfig, StopReason, gradient_descent


def test_converges_by_grad_norm(wdbc_objective):
    res = gradient_descent(wdbc_objective, np.zeros(wdbc_objective.dim), GDConfig(alpha=5.0, max_iter=5000, tol_grad=1e-5))
    assert res.stop_reason == StopReason.GRAD_NORM and res.converged
    assert res.final_grad_norm < 1e-5
    assert res.history.loss[0] == pytest.approx(np.log(2))  # J(0) = ln 2


def test_loss_monotone_for_small_step(wdbc_objective):
    L = wdbc_objective.lipschitz_constant()
    res = gradient_descent(wdbc_objective, np.zeros(wdbc_objective.dim), GDConfig(alpha=1.0 / L, max_iter=300))
    assert np.all(np.diff(res.history.loss) <= 1e-15)  # descenso garantizado si α ≤ 1/L


def test_stops_by_loss_change(wdbc_objective):
    cfg = GDConfig(alpha=0.5, max_iter=50_000, tol_grad=0.0, tol_loss=1e-9)
    res = gradient_descent(wdbc_objective, np.zeros(wdbc_objective.dim), cfg)
    assert res.stop_reason == StopReason.LOSS_CHANGE
    assert res.history.loss_change[-1] < 1e-9


def test_stops_by_max_iter(wdbc_objective):
    res = gradient_descent(wdbc_objective, np.zeros(wdbc_objective.dim), GDConfig(alpha=0.01, max_iter=25))
    assert res.stop_reason == StopReason.MAX_ITER and res.iterations == 25
    assert len(res.history.iteration) == 26  # k = 0..25


def test_divergence_by_threshold_returns_state(wdbc_objective):
    res = gradient_descent(wdbc_objective, np.zeros(wdbc_objective.dim), GDConfig(alpha=1e5, max_iter=100))
    assert res.stop_reason == StopReason.DIVERGENCE
    assert res.history.loss[-1] > 1e8 and "umbral" in res.divergence_detail


def test_divergence_by_nan_in_naive_mode(ds):
    obj = LogisticObjective(ds.X_train, ds.y_train, l2=1e-3, mode="naive")
    res = gradient_descent(obj, np.zeros(obj.dim), GDConfig(alpha=1000.0, max_iter=100))
    assert res.stop_reason == StopReason.DIVERGENCE
    assert res.numeric_flags["has_nan"] or res.numeric_flags["has_inf"]
    assert res.numeric_flags["exp_overflow_events"] > 0


def test_stable_mode_survives_same_alpha_as_naive(ds):
    obj = LogisticObjective(ds.X_train, ds.y_train, l2=0.0, mode="logits")
    res = gradient_descent(obj, np.zeros(obj.dim), GDConfig(alpha=1000.0, max_iter=3000))
    assert res.stop_reason != StopReason.DIVERGENCE
    assert np.isfinite(res.final_loss)


def test_trained_model_quality(ds, wdbc_objective):
    res = gradient_descent(wdbc_objective, np.zeros(wdbc_objective.dim), GDConfig(alpha=1.0, max_iter=8000))
    p = sigmoid(ds.X_test @ res.w + res.b)
    acc = np.mean((p >= 0.5) == ds.y_test)
    assert acc > 0.95


def test_record_every_keeps_last(wdbc_objective):
    res = gradient_descent(wdbc_objective, np.zeros(wdbc_objective.dim), GDConfig(alpha=0.1, max_iter=101, record_every=10))
    assert res.history.iteration[-1] == 101 and res.history.iteration[:3] == [0, 10, 20]


def test_invalid_config():
    with pytest.raises(ValueError):
        GDConfig(alpha=-1).validate()
