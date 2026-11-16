"""Funciones de activación con tratamiento explícito del error de punto flotante.

La sigmoide ingenua ``1 / (1 + exp(-z))`` evalúa ``exp(-z)``, que desborda en
IEEE-754 doble precisión cuando ``-z > ln(DBL_MAX) ≈ 709.78``. La versión
estable evalúa por tramos según el signo de ``z`` de modo que el argumento de
``exp`` es siempre no positivo:

    σ(z) = 1 / (1 + e^{-z})      si z ≥ 0
    σ(z) = e^{z} / (1 + e^{z})   si z < 0

Implementación: con ``e = exp(-|z|) ∈ (0, 1]`` ambos tramos se escriben sin
riesgo de overflow, y solo se calcula ``exp`` una vez por elemento (a
diferencia de ``np.where``, que evaluaría ambas ramas y emitiría warnings).
"""
from __future__ import annotations

import numpy as np

# ln(DBL_MAX): umbral a partir del cual exp(x) = +inf en float64.
EXP_OVERFLOW_THRESHOLD: float = float(np.log(np.finfo(np.float64).max))


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Sigmoide numéricamente estable (por tramos según el signo de z)."""
    z_arr = np.asarray(z, dtype=np.float64)
    flat = z_arr.reshape(-1)
    out = np.empty_like(flat)

    e = np.exp(-np.abs(flat))  # siempre en (0, 1]: nunca desborda
    pos = flat >= 0
    out[pos] = 1.0 / (1.0 + e[pos])
    neg = ~pos
    out[neg] = e[neg] / (1.0 + e[neg])

    out = out.reshape(z_arr.shape)
    return float(out) if np.ndim(z) == 0 else out


def naive_sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Sigmoide ingenua. SOLO para auditoría y demostración de inestabilidad."""
    z_arr = np.asarray(z, dtype=np.float64)
    with np.errstate(over="ignore"):
        out = 1.0 / (1.0 + np.exp(-z_arr))
    return float(out) if np.ndim(z) == 0 else out


def count_exp_overflows(z: np.ndarray) -> int:
    """Número de elementos para los que la sigmoide ingenua desbordaría exp(-z)."""
    return int(np.count_nonzero(-np.asarray(z, dtype=np.float64) > EXP_OVERFLOW_THRESHOLD))


def log_sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """log σ(z) = -log(1 + e^{-z}) evaluado vía Log-Sum-Exp (exacto para todo z)."""
    return -np.logaddexp(0.0, -np.asarray(z, dtype=np.float64))
