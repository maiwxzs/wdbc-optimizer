import numpy as np
import pytest

from app.math_engine import (
    EPSILON,
    bce_from_logits,
    binary_cross_entropy,
    log_sigmoid,
    naive_binary_cross_entropy,
    naive_sigmoid,
    sigmoid,
)


def test_sigmoid_matches_naive_in_safe_range():
    z = np.linspace(-30, 30, 1001)
    np.testing.assert_allclose(sigmoid(z), naive_sigmoid(z), rtol=1e-14, atol=0)


@pytest.mark.parametrize("z", [-1e6, -1000.0, -745.0, -710.0, 0.0, 710.0, 1e6])
def test_sigmoid_never_overflows(z):
    # pytest.ini convierte cualquier RuntimeWarning (overflow) en error
    s = sigmoid(z)
    assert np.isfinite(s) and 0.0 <= s <= 1.0


def test_sigmoid_symmetry_and_scalar():
    z = np.array([-50.0, -3.0, 0.0, 3.0, 50.0])
    np.testing.assert_allclose(sigmoid(z) + sigmoid(-z), 1.0, atol=1e-15)
    assert isinstance(sigmoid(0.0), float) and sigmoid(0.0) == 0.5


def test_sigmoid_preserves_shape():
    z = np.arange(12.0).reshape(3, 4) - 6
    assert sigmoid(z).shape == (3, 4)


def test_log_sigmoid_exact_for_extreme_values():
    assert log_sigmoid(-1000.0) == pytest.approx(-1000.0)
    assert log_sigmoid(1000.0) == pytest.approx(0.0, abs=1e-300)


def test_bce_formulations_agree(toy):
    X, y = toy
    z = X @ np.ones(X.shape[1]) * 0.5
    assert bce_from_logits(y, z) == pytest.approx(binary_cross_entropy(y, sigmoid(z)), rel=1e-12)


def test_bce_handles_saturation():
    y = np.array([1.0, 0.0])
    z = np.array([-800.0, 800.0])  # predicciones totalmente erróneas
    exact = bce_from_logits(y, z)
    clipped = binary_cross_entropy(y, sigmoid(z))
    assert exact == pytest.approx(800.0)
    # Techo del clip ≈ -log ε ≈ 34.54. No es exacto para y=0 porque 1 - 1e-15 no es
    # representable en float64: se redondea a 1 - 9.992e-16 (error de representación).
    assert clipped == pytest.approx(-np.log(EPSILON), rel=1e-4)
    assert 1.0 - (1.0 - EPSILON) != EPSILON
    assert np.isinf(naive_binary_cross_entropy(y, naive_sigmoid(z)))


def test_naive_bce_produces_nan_from_zero_times_log_zero():
    # σ(40) redondea a 1.0 exactamente → (1-y)·log(1-1) = 0·(-inf) = NaN
    assert naive_sigmoid(40.0) == 1.0
    assert np.isnan(naive_binary_cross_entropy(np.array([1.0]), np.array([naive_sigmoid(40.0)])))
