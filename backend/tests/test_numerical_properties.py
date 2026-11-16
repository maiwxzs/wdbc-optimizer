"""Validación matemática del motor numérico.

Estas pruebas no repiten la implementación: comprueban propiedades que la teoría
exige al objetivo J(θ) y al optimizador (derivadas, convexidad, lema de descenso,
cota de suavidad L, unicidad del minimizador) usando referencias independientes
(diferencias finitas y un método de Newton escrito aquí mismo).
"""
from __future__ import annotations

import numpy as np
import pytest

from app.math_engine import (
    LogisticObjective,
    central_difference_gradient,
    forward_difference_gradient,
    gradient_check,
    naive_sigmoid,
    relative_error,
    sigmoid,
)
from app.math_engine.losses import bce_per_sample_from_logits
from app.optimization import GDConfig, StopReason, gradient_descent

# ---------------------------------------------------------------- utilidades


def _hessian(obj: LogisticObjective, theta: np.ndarray) -> np.ndarray:
    """∇²J(θ) = (1/m) X̃ᵀ diag(p(1-p)) X̃ + λ·diag(1,…,1,0)  (el sesgo no se regulariza)."""
    p = sigmoid(obj.logits(theta))
    Xa = np.hstack([obj.X, np.ones((obj.m, 1))])
    H = Xa.T @ (Xa * (p * (1.0 - p))[:, None]) / obj.m
    reg = np.full(obj.dim, obj.l2)
    reg[-1] = 0.0
    return H + np.diag(reg)


def _newton(obj: LogisticObjective, iters: int = 50) -> np.ndarray:
    """Método de Newton puro como referencia independiente del minimizador θ*."""
    theta = np.zeros(obj.dim)
    for _ in range(iters):
        g = obj.grad(theta)
        if np.linalg.norm(g) < 1e-13:
            break
        theta = theta - np.linalg.solve(_hessian(obj, theta), g)
    return theta


# --------------------------------------------------------------- activación


def test_sigmoid_derivative_matches_closed_form():
    z = np.linspace(-20, 20, 81)
    numeric = np.array([(sigmoid(t + 1e-5) - sigmoid(t - 1e-5)) / 2e-5 for t in z])
    s = sigmoid(z)
    np.testing.assert_allclose(numeric, s * (1 - s), atol=1e-10)


def test_sigmoid_is_monotone_bounded_and_finite_on_extreme_grid():
    z = np.concatenate([[-1e300, -1e10], np.linspace(-800, 800, 4001), [1e10, 1e300]])
    s = sigmoid(z)
    assert np.all(np.isfinite(s))
    assert np.all((s >= 0.0) & (s <= 1.0))
    assert np.all(np.diff(s) >= 0.0)


def test_naive_sigmoid_saturates_to_exactly_one():
    # 1 + e^{-z} == 1 en float64 cuando e^{-z} < ε/2, es decir z > -ln(ε/2) ≈ 36.74
    z_sat = -np.log(np.finfo(np.float64).eps / 2)
    assert naive_sigmoid(z_sat + 0.5) == 1.0
    assert naive_sigmoid(z_sat - 2.0) < 1.0


# ------------------------------------------------------------------ pérdida


def test_per_sample_loss_matches_textbook_bce_in_safe_range():
    rng = np.random.default_rng(11)
    z = rng.uniform(-15, 15, 200)
    y = rng.integers(0, 2, 200).astype(float)
    p = sigmoid(z)
    textbook = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    # atol cubre la cancelación de la forma de libro en log(1 - p) cuando p ≈ 1:
    # la discrepancia (~1e-15 absoluta) es error de la referencia, no del motor.
    np.testing.assert_allclose(bce_per_sample_from_logits(y, z), textbook, rtol=1e-10, atol=1e-14)


def test_loss_is_non_negative_and_equals_ln2_at_origin(wdbc_objective):
    assert wdbc_objective.loss(np.zeros(wdbc_objective.dim)) == pytest.approx(np.log(2.0), rel=1e-15)
    rng = np.random.default_rng(0)
    for _ in range(10):
        assert wdbc_objective.loss(rng.normal(0, 3, wdbc_objective.dim)) >= 0.0


def test_objective_rejects_inconsistent_shapes():
    with pytest.raises(ValueError):
        LogisticObjective(np.zeros((5, 2)), np.zeros(4))


def test_shapes_of_logits_and_gradient(wdbc_objective):
    theta = np.zeros(wdbc_objective.dim)
    assert wdbc_objective.dim == 31
    assert wdbc_objective.logits(theta).shape == (wdbc_objective.m,)
    assert wdbc_objective.grad(theta).shape == (31,)


# ------------------------------------------------------- gradiente analítico


@pytest.mark.parametrize("l2", [0.0, 1e-3, 1.0])
def test_gradient_check_for_several_regularizations(wdbc_objective, l2):
    obj = LogisticObjective(wdbc_objective.X, wdbc_objective.y, l2=l2)
    theta = np.random.default_rng(5).normal(0, 0.5, obj.dim)
    assert gradient_check(obj.loss, obj.grad, theta, h=1e-5)["relative_error"] < 1e-7


def test_gradient_check_clipped_mode_in_non_saturated_region(wdbc_objective):
    obj = LogisticObjective(wdbc_objective.X, wdbc_objective.y, l2=1e-3, mode="clipped")
    theta = np.random.default_rng(1).normal(0, 0.1, obj.dim)
    assert gradient_check(obj.loss, obj.grad, theta)["verdict"] == "excelente"


def test_forward_difference_is_less_accurate_than_central(wdbc_objective):
    theta = np.random.default_rng(2).normal(0, 0.5, wdbc_objective.dim)
    g = wdbc_objective.grad(theta)
    err_fwd = relative_error(g, forward_difference_gradient(wdbc_objective.loss, theta, 1e-7))
    err_cen = relative_error(g, central_difference_gradient(wdbc_objective.loss, theta, 1e-5))
    assert err_fwd < 1e-5
    assert err_cen < err_fwd


# ------------------------------------------- convexidad y suavidad (teoría)


def test_hessian_is_psd_and_bounded_by_lipschitz_constant(wdbc_objective):
    L = wdbc_objective.lipschitz_constant()
    rng = np.random.default_rng(3)
    for _ in range(5):
        eig = np.linalg.eigvalsh(_hessian(wdbc_objective, rng.normal(0, 1, wdbc_objective.dim)))
        assert eig[0] >= -1e-12
        assert eig[-1] <= L + 1e-12


def test_convexity_along_random_segments(wdbc_objective):
    rng = np.random.default_rng(4)
    J = wdbc_objective.loss
    for _ in range(20):
        a, b = rng.normal(0, 2, (2, wdbc_objective.dim))
        t = rng.uniform()
        assert J(t * a + (1 - t) * b) <= t * J(a) + (1 - t) * J(b) + 1e-12


def test_descent_lemma_with_step_one_over_L(wdbc_objective):
    """J(θ - ∇J/L) ≤ J(θ) - ‖∇J‖²/(2L) para toda función L-suave."""
    L = wdbc_objective.lipschitz_constant()
    rng = np.random.default_rng(6)
    for _ in range(10):
        theta = rng.normal(0, 1, wdbc_objective.dim)
        g = wdbc_objective.grad(theta)
        lhs = wdbc_objective.loss(theta - g / L)
        rhs = wdbc_objective.loss(theta) - g @ g / (2 * L)
        assert lhs <= rhs + 1e-12


# ---------------------------------------------- optimizador vs referencia


def test_gradient_descent_reaches_newton_minimizer(wdbc_objective):
    theta_star = _newton(wdbc_objective)
    assert np.linalg.norm(wdbc_objective.grad(theta_star)) < 1e-10
    res = gradient_descent(wdbc_objective, np.zeros(31), GDConfig(alpha=1.0, max_iter=20000, tol_grad=1e-7, tol_loss=0.0))
    assert res.stop_reason == StopReason.GRAD_NORM
    gap = res.final_loss - wdbc_objective.loss(theta_star)
    # Para J μ-fuertemente convexa: J(θ) - J* ≤ ‖∇J‖² / (2μ), con μ ≥ λ_min(∇²J(θ*)) > 0
    mu = np.linalg.eigvalsh(_hessian(wdbc_objective, theta_star))[0]
    assert -1e-12 <= gap <= res.final_grad_norm**2 / (2 * mu) + 1e-12
    assert np.linalg.norm(res.theta - theta_star) <= res.final_grad_norm / mu * 1.05


def test_gradient_descent_is_bitwise_reproducible(wdbc_objective):
    cfg = GDConfig(alpha=1.0, max_iter=500)
    r1 = gradient_descent(wdbc_objective, np.zeros(31), cfg)
    r2 = gradient_descent(wdbc_objective, np.zeros(31), cfg)
    np.testing.assert_array_equal(r1.theta, r2.theta)
    assert r1.history.loss == r2.history.loss


def test_gradient_descent_does_not_mutate_initial_point(wdbc_objective):
    theta0 = np.full(31, 0.1)
    gradient_descent(wdbc_objective, theta0, GDConfig(alpha=0.5, max_iter=10))
    np.testing.assert_array_equal(theta0, np.full(31, 0.1))


def test_without_regularization_weights_keep_growing(wdbc_objective):
    """Hallazgo documentado: con λ = 0 los datos de entrenamiento son casi separables y ‖w‖ crece sin cota."""
    obj = LogisticObjective(wdbc_objective.X, wdbc_objective.y, l2=0.0)
    res = gradient_descent(obj, np.zeros(31), GDConfig(alpha=5.0, max_iter=4000, tol_grad=0.0, tol_loss=0.0))
    wn = np.asarray(res.history.weight_norm)
    assert wn[-1] > wn[1000] > wn[100]
    assert res.final_loss < 0.02
