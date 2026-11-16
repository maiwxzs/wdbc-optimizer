"""Entropía cruzada binaria (BCE) en tres formulaciones numéricas.

1. ``bce_from_logits`` (Log-Sum-Exp, por defecto en el motor):
       ℓ(z, y) = log(1 + e^{z}) - y·z = logaddexp(0, z) - y·z
   Es algebraicamente idéntica a la BCE y exacta en float64 para cualquier z.

2. ``binary_cross_entropy`` (acotación por margen de seguridad ε = 1e-15):
       J = -(1/m) Σ [ y log(ŷ) + (1-y) log(1-ŷ) ],  ŷ ← clip(ŷ, ε, 1-ε)
   Evita log(0) pero introduce saturación: la pérdida por muestra queda
   acotada en [~1.1e-15, -log ε ≈ 34.54].

3. ``naive_binary_cross_entropy``: sin protección. Puede devolver ±inf/NaN.
   Se expone únicamente para la auditoría de error flotante.
"""
from __future__ import annotations

import numpy as np

EPSILON: float = 1e-15


def binary_cross_entropy(y: np.ndarray, y_hat: np.ndarray, eps: float = EPSILON) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(y_hat, dtype=np.float64), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def bce_from_logits(y: np.ndarray, z: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def bce_per_sample_from_logits(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, np.asarray(z, dtype=np.float64)) - np.asarray(y, dtype=np.float64) * z


def naive_binary_cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(y_hat, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(val)


def l2_penalty(w: np.ndarray, lam: float) -> float:
    """Término de Tikhonov (λ/2)‖w‖². No penaliza el sesgo b."""
    return 0.5 * lam * float(np.dot(w, w)) if lam > 0 else 0.0
