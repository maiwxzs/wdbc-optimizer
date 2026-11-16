import numpy as np
import pytest

from app.math_engine import (
    EPS_MACHINE,
    LogisticObjective,
    central_difference_gradient,
    forward_difference_gradient,
    gradient_check,
    logistic_gradient,
    logistic_loss,
    pack,
    relative_error,
    step_size_sweep,
)


def test_packed_gradient_matches_separate_form(toy):
    X, y = toy
    w, b = np.linspace(-1, 1, X.shape[1]), 0.3
    gw, gb = logistic_gradient(X, y, w, b, l2=0.1)
    obj = LogisticObjective(X, y, l2=0.1)
    np.testing.assert_allclose(obj.grad(pack(w, b)), np.append(gw, gb), rtol=1e-14)
    assert obj.loss(pack(w, b)) == pytest.approx(logistic_loss(X, y, w, b, l2=0.1), rel=1e-14)


def test_gradient_zero_point_closed_form(toy):
    # En θ = 0: ŷ = 1/2  ⇒  ∇_w J = (1/m) Xᵀ(1/2 - y)
    X, y = toy
    obj = LogisticObjective(X, y)
    g = obj.grad(np.zeros(obj.dim))
    np.testing.assert_allclose(g[:-1], X.T @ (0.5 - y) / X.shape[0], rtol=1e-14)
    assert g[-1] == pytest.approx(np.mean(0.5 - y))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gradient_check_wdbc_random_points(wdbc_objective, seed):
    theta = np.random.default_rng(seed).normal(0, 0.5, wdbc_objective.dim)
    res = gradient_check(wdbc_objective.loss, wdbc_objective.grad, theta, h=1e-5)
    assert res["relative_error"] < 1e-7
    assert res["verdict"] == "excelente"


def test_gradient_check_detects_wrong_gradient(toy):
    X, y = toy
    obj = LogisticObjective(X, y)
    wrong = lambda t: 2.0 * obj.grad(t)  # noqa: E731
    res = gradient_check(obj.loss, wrong, np.full(obj.dim, 0.2))
    assert res["relative_error"] > 0.3 and res["verdict"] == "incorrecto"


def test_relative_error_properties():
    a = np.array([1.0, 2.0, 3.0])
    assert relative_error(a, a) == 0.0
    assert relative_error(np.zeros(3), np.zeros(3)) == 0.0  # sin división por cero
    assert relative_error(a, -a) == pytest.approx(1.0)


def test_finite_difference_orders_on_quadratic():
    f = lambda t: float(np.sum(t**3))  # noqa: E731   f' = 3t², f'' = 6t
    t = np.array([1.0, -2.0])
    exact = 3 * t**2
    # central: error exactamente h² (de t³), progresiva: ~3 t h
    np.testing.assert_allclose(central_difference_gradient(f, t, 1e-3), exact, atol=2e-6)
    assert np.max(np.abs(forward_difference_gradient(f, t, 1e-3) - exact)) > 1e-3


def test_step_sweep_has_v_shape(wdbc_objective):
    theta = np.random.default_rng(0).normal(0, 0.5, wdbc_objective.dim)
    hs = np.logspace(-14, -1, 27)
    res = step_size_sweep(wdbc_objective.loss, wdbc_objective.grad, theta, hs)
    c = res["central_error"]
    i = int(np.argmin(c))
    assert 0 < i < len(hs) - 1, "el mínimo debe ser interior (forma de V)"
    assert c[0] > 100 * c[i] and c[-1] > 100 * c[i]
    # h óptimo central del orden de ε^{1/3} ≈ 6e-6 (una década y media de margen)
    assert 1e-7 < res["central_best"]["h"] < 1e-4
    # la central alcanza mejor precisión que la progresiva
    assert res["central_best"]["error"] < res["forward_best"]["error"]
    assert res["forward_h_opt_theory"] == pytest.approx(np.sqrt(EPS_MACHINE))


def test_lipschitz_bounds_hessian(toy):
    X, y = toy
    obj = LogisticObjective(X, y)
    L = obj.lipschitz_constant()
    # Hessiano exacto en un punto: (1/m) X̃ᵀ diag(σ(1-σ)) X̃
    theta = np.full(obj.dim, 0.1)
    from app.math_engine import sigmoid

    s = sigmoid(obj.logits(theta))
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    H = Xa.T @ (Xa * (s * (1 - s))[:, None]) / X.shape[0]
    assert np.linalg.eigvalsh(H)[-1] <= L + 1e-12
