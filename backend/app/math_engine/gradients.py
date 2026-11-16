"""Gradiente analítico vectorizado y el objetivo J(w, b) empaquetado.

Para ŷ = σ(Xw + b) y la BCE promedio:

    ∇_w J = (1/m) Xᵀ (ŷ - y) + λ w
    ∂J/∂b = (1/m) Σ (ŷ_i - y_i)

Derivación: ∂ℓ_i/∂z_i = σ(z_i) - y_i (la derivada de log(1+e^z) es σ(z)),
y por regla de la cadena ∂z_i/∂w = x_i, ∂z_i/∂b = 1.

``LogisticObjective`` trabaja con el vector empaquetado θ = [w; b] ∈ ℝ^{d+1},
lo que permite reutilizar el mismo objetivo en el optimizador y en la
auditoría por diferencias finitas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .activations import count_exp_overflows, naive_sigmoid, sigmoid
from .losses import (
    EPSILON,
    bce_from_logits,
    binary_cross_entropy,
    l2_penalty,
    naive_binary_cross_entropy,
)

NumericMode = Literal["logits", "clipped", "naive"]


def logistic_gradient(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, l2: float = 0.0
) -> tuple[np.ndarray, float]:
    """Gradiente analítico exacto (∇_w J, ∂J/∂b) — forma vectorizada."""
    m = X.shape[0]
    residual = sigmoid(X @ w + b) - y  # ŷ - y
    grad_w = (X.T @ residual) / m
    if l2 > 0:
        grad_w = grad_w + l2 * w
    grad_b = float(np.sum(residual) / m)
    return grad_w, grad_b


def logistic_loss(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, l2: float = 0.0
) -> float:
    return bce_from_logits(y, X @ w + b) + l2_penalty(w, l2)


def pack(w: np.ndarray, b: float) -> np.ndarray:
    return np.concatenate([np.asarray(w, dtype=np.float64), [float(b)]])


def unpack(theta: np.ndarray) -> tuple[np.ndarray, float]:
    return theta[:-1], float(theta[-1])


@dataclass
class LogisticObjective:
    """J(θ) para regresión logística, con selección de formulación numérica.

    mode:
        - "logits":  pérdida Log-Sum-Exp + sigmoide estable (recomendado)
        - "clipped": sigmoide estable + clip(ŷ, ε, 1-ε)
        - "naive":   sigmoide ingenua + log sin protección (inestable)
    """

    X: np.ndarray
    y: np.ndarray
    l2: float = 0.0
    mode: NumericMode = "logits"
    eps: float = EPSILON
    overflow_events: int = field(default=0, init=False)
    n_loss_evals: int = field(default=0, init=False)
    n_grad_evals: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.X = np.ascontiguousarray(self.X, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64).reshape(-1)
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X e y deben tener el mismo número de muestras")

    @property
    def m(self) -> int:
        return self.X.shape[0]

    @property
    def dim(self) -> int:
        return self.X.shape[1] + 1

    def logits(self, theta: np.ndarray) -> np.ndarray:
        w, b = unpack(theta)
        with np.errstate(over="ignore", invalid="ignore"):
            return self.X @ w + b

    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.mode == "naive":
            self.overflow_events += count_exp_overflows(z)
            return naive_sigmoid(z)
        return sigmoid(z)

    def loss(self, theta: np.ndarray) -> float:
        self.n_loss_evals += 1
        w, _ = unpack(theta)
        z = self.logits(theta)
        if self.mode == "logits":
            data_term = bce_from_logits(self.y, z)
        elif self.mode == "clipped":
            data_term = binary_cross_entropy(self.y, sigmoid(z), self.eps)
        else:
            data_term = naive_binary_cross_entropy(self.y, self._activate(z))
        return data_term + l2_penalty(w, self.l2)

    def grad(self, theta: np.ndarray) -> np.ndarray:
        self.n_grad_evals += 1
        w, _ = unpack(theta)
        z = self.logits(theta)
        with np.errstate(over="ignore", invalid="ignore"):
            residual = self._activate(z) - self.y
            g_w = (self.X.T @ residual) / self.m
            if self.l2 > 0:
                g_w = g_w + self.l2 * w
            g_b = np.sum(residual) / self.m
        return np.concatenate([g_w, [g_b]])

    def loss_and_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        return self.loss(theta), self.grad(theta)

    def lipschitz_constant(self) -> float:
        """Cota de suavidad L: ∇²J ⪯ (1/4m) X̃ᵀX̃ + λI, con X̃ = [X | 1].

        Gradiente descendente con paso fijo converge en funciones convexas
        L-suaves si 0 < α < 2/L.
        """
        Xa = np.hstack([self.X, np.ones((self.m, 1))])
        lam_max = float(np.linalg.eigvalsh(Xa.T @ Xa / self.m)[-1])
        return 0.25 * lam_max + self.l2
