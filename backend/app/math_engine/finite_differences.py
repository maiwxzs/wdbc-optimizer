"""Derivación numérica por diferencias finitas y análisis del error de paso h.

Diferencia central:   ∂J/∂θ_j ≈ [J(θ + h e_j) - J(θ - h e_j)] / (2h)
    error total ≈ (h²/6)|J'''| (truncamiento)  +  ε_mach |J| / h (redondeo)
    h* ≈ (3 ε_mach |J| / |J'''|)^{1/3}  ~ 6e-6

Diferencia progresiva: ∂J/∂θ_j ≈ [J(θ + h e_j) - J(θ)] / h
    error total ≈ (h/2)|J''| + 2 ε_mach |J| / h,   h* ~ √ε_mach ~ 1.5e-8

Representado en escala log-log, el error total forma una "V": la rama
izquierda (pendiente -1) la domina el redondeo, la derecha el truncamiento
(pendiente +2 para central, +1 para progresiva).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

EPS_MACHINE: float = float(np.finfo(np.float64).eps)
ArrayFn = Callable[[np.ndarray], float]


def central_difference_gradient(f: ArrayFn, theta: np.ndarray, h: float = 1e-5) -> np.ndarray:
    theta = np.array(theta, dtype=np.float64, copy=True)
    grad = np.empty_like(theta)
    for j in range(theta.size):
        original = theta[j]
        theta[j] = original + h
        f_plus = f(theta)
        theta[j] = original - h
        f_minus = f(theta)
        theta[j] = original
        grad[j] = (f_plus - f_minus) / (2.0 * h)
    return grad


def forward_difference_gradient(f: ArrayFn, theta: np.ndarray, h: float = 1e-8) -> np.ndarray:
    theta = np.array(theta, dtype=np.float64, copy=True)
    f0 = f(theta)
    grad = np.empty_like(theta)
    for j in range(theta.size):
        original = theta[j]
        theta[j] = original + h
        grad[j] = (f(theta) - f0) / h
        theta[j] = original
    return grad


def relative_error(g_analytic: np.ndarray, g_numeric: np.ndarray) -> float:
    """‖g_a - g_n‖₂ / (‖g_a‖₂ + ‖g_n‖₂ + 1e-15)."""
    num = np.linalg.norm(g_analytic - g_numeric)
    den = np.linalg.norm(g_analytic) + np.linalg.norm(g_numeric) + 1e-15
    return float(num / den)


def classify_relative_error(err: float) -> tuple[str, str]:
    """Veredicto cualitativo estándar para chequeo de gradientes."""
    if not np.isfinite(err):
        return "fallo", "Error no finito: la evaluación de J produjo NaN/Inf."
    if err < 1e-7:
        return "excelente", "Gradiente analítico correcto (error < 1e-7)."
    if err < 1e-5:
        return "aceptable", "Correcto salvo ruido de redondeo / no suavidad local."
    if err < 1e-3:
        return "sospechoso", "Posible error de implementación o h mal escogido."
    return "incorrecto", "El gradiente analítico NO coincide con el numérico."


def gradient_check(
    loss_fn: ArrayFn,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    h: float = 1e-5,
) -> dict:
    g_a = grad_fn(theta)
    g_n = central_difference_gradient(loss_fn, theta, h)
    abs_err = np.abs(g_a - g_n)
    comp_rel = abs_err / (np.abs(g_a) + np.abs(g_n) + 1e-15)
    err = relative_error(g_a, g_n)
    verdict, message = classify_relative_error(err)
    return {
        "h": h,
        "analytic": g_a,
        "numeric": g_n,
        "abs_error": abs_err,
        "component_rel_error": comp_rel,
        "relative_error": err,
        "max_abs_error": float(abs_err.max()),
        "analytic_norm": float(np.linalg.norm(g_a)),
        "numeric_norm": float(np.linalg.norm(g_n)),
        "verdict": verdict,
        "message": message,
    }


def _fit_v_model(h: np.ndarray, err: np.ndarray, trunc_order: int) -> tuple[float, float]:
    """Ajuste robusto de E(h) ≈ a·h^p + c/h, rama por rama.

    Rama izquierda (redondeo, h < h_min):     c = mediana(E·h)
    Rama derecha (truncamiento, h > h_min):   a = mediana(E / h^p)
    Solo se usa la zona asintótica (E < 1e-2): para h extremos el error
    relativo satura en ~1 (cota del denominador) y deja de seguir el modelo.
    La media geométrica (mediana en log) evita que una década domine.
    """
    valid = np.isfinite(err) & (err > 0)
    if valid.sum() < 3:
        return 0.0, 0.0
    i_min = int(np.argmin(np.where(valid, err, np.inf)))
    asym = valid & (err < 1e-2)
    left = asym & (np.arange(h.size) < i_min - 1)
    right = asym & (np.arange(h.size) > i_min + 1)
    c = float(np.exp(np.median(np.log(err[left] * h[left])))) if left.any() else 0.0
    a = float(np.exp(np.median(np.log(err[right] / h[right] ** trunc_order)))) if right.any() else 0.0
    return a, c


def step_size_sweep(
    loss_fn: ArrayFn,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    hs: np.ndarray,
) -> dict:
    """Error relativo vs h para diferencias central y progresiva (curva en V)."""
    hs = np.asarray(hs, dtype=np.float64)
    g_a = grad_fn(theta)
    central = np.array([relative_error(g_a, central_difference_gradient(loss_fn, theta, h)) for h in hs])
    forward = np.array([relative_error(g_a, forward_difference_gradient(loss_fn, theta, h)) for h in hs])

    a_c, c_c = _fit_v_model(hs, central, 2)
    a_f, c_f = _fit_v_model(hs, forward, 1)

    def h_opt(a: float, c: float, p: int) -> float | None:
        # d/dh (a h^p + c/h) = 0  ->  h* = (c / (p a))^{1/(p+1)}
        return float((c / (p * a)) ** (1.0 / (p + 1))) if a > 0 and c > 0 else None

    i_c, i_f = int(np.nanargmin(central)), int(np.nanargmin(forward))
    return {
        "h": hs,
        "central_error": central,
        "forward_error": forward,
        "central_model": np.minimum(a_c * hs**2 + c_c / hs, 1.0),
        "forward_model": np.minimum(a_f * hs + c_f / hs, 1.0),
        "central_best": {"h": float(hs[i_c]), "error": float(central[i_c])},
        "forward_best": {"h": float(hs[i_f]), "error": float(forward[i_f])},
        "central_h_opt_fit": h_opt(a_c, c_c, 2),
        "forward_h_opt_fit": h_opt(a_f, c_f, 1),
        "central_h_opt_theory": float(EPS_MACHINE ** (1.0 / 3.0)),
        "forward_h_opt_theory": float(np.sqrt(EPS_MACHINE)),
        "eps_machine": EPS_MACHINE,
    }
