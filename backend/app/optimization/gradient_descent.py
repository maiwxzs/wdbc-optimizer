"""Gradiente descendente de paso fijo con telemetría completa.

    θ^{(k+1)} = θ^{(k)} - α ∇J(θ^{(k)}),     θ = [w; b]

En cada iteración k se registra J^{(k)}, ‖∇J^{(k)}‖₂, |ΔJ|, ‖w‖₂ y se evalúan,
en este orden, las cuatro condiciones de parada:

    4. Fallo numérico / divergencia: NaN, ±Inf o J^{(k)} > umbral (1e8)
    1. Convergencia por norma:        ‖∇J‖₂ < tol_grad
    2. Convergencia por cambio:       |J^{(k)} - J^{(k-1)}| < tol_loss
    3. Límite de iteraciones:         k = max_iter

La divergencia se comprueba primero: un estado no finito no debe confundirse
con convergencia (p. ej. ΔJ = inf - inf = NaN no es < tol). El optimizador
nunca lanza excepciones por inestabilidad: devuelve el estado numérico.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from app.math_engine.gradients import LogisticObjective, unpack


class StopReason(str, Enum):
    GRAD_NORM = "grad_norm"
    LOSS_CHANGE = "loss_change"
    MAX_ITER = "max_iter"
    DIVERGENCE = "divergence"


STOP_MESSAGES = {
    StopReason.GRAD_NORM: "Convergencia: ‖∇J‖₂ < tol_grad (punto estacionario de primer orden).",
    StopReason.LOSS_CHANGE: "Convergencia: |ΔJ| < tol_loss (estancamiento de la función objetivo).",
    StopReason.MAX_ITER: "Límite de iteraciones alcanzado sin satisfacer las tolerancias.",
    StopReason.DIVERGENCE: "Fallo numérico / divergencia detectada. Ejecución detenida.",
}


@dataclass
class GDConfig:
    alpha: float = 0.5
    max_iter: int = 5000
    tol_grad: float = 1e-5
    tol_loss: float = 1e-12
    divergence_threshold: float = 1e8
    record_every: int = 1

    def validate(self) -> None:
        if not (self.alpha > 0 and np.isfinite(self.alpha)):
            raise ValueError("alpha debe ser positivo y finito")
        if self.max_iter < 1:
            raise ValueError("max_iter debe ser >= 1")
        if self.tol_grad < 0 or self.tol_loss < 0:
            raise ValueError("las tolerancias deben ser no negativas")
        if self.record_every < 1:
            raise ValueError("record_every debe ser >= 1")


@dataclass
class GDHistory:
    iteration: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    loss_change: list[float] = field(default_factory=list)
    weight_norm: list[float] = field(default_factory=list)

    def append(self, k: int, J: float, gn: float, dJ: float, wn: float) -> None:
        self.iteration.append(k)
        self.loss.append(J)
        self.grad_norm.append(gn)
        self.loss_change.append(dJ)
        self.weight_norm.append(wn)


@dataclass
class GDResult:
    theta: np.ndarray
    stop_reason: StopReason
    iterations: int
    history: GDHistory
    final_loss: float
    final_grad_norm: float
    elapsed_ms: float
    numeric_flags: dict
    divergence_detail: str | None = None

    @property
    def w(self) -> np.ndarray:
        return unpack(self.theta)[0]

    @property
    def b(self) -> float:
        return unpack(self.theta)[1]

    @property
    def converged(self) -> bool:
        return self.stop_reason in (StopReason.GRAD_NORM, StopReason.LOSS_CHANGE)

    @property
    def message(self) -> str:
        return STOP_MESSAGES[self.stop_reason]


def _diagnose(J: float, g: np.ndarray, theta: np.ndarray, threshold: float) -> str | None:
    if np.isnan(J):
        return "J = NaN (indeterminación, p. ej. 0·log 0 o inf - inf)."
    if np.isinf(J):
        return "J = ±Inf (log(0) tras saturación de la sigmoide o overflow en exp)."
    if J > threshold:
        return f"J = {J:.3e} supera el umbral de divergencia {threshold:.0e}."
    if not np.all(np.isfinite(g)):
        return "∇J contiene NaN/Inf."
    if not np.all(np.isfinite(theta)):
        return "Los parámetros θ contienen NaN/Inf."
    return None


def gradient_descent(
    objective: LogisticObjective,
    theta0: np.ndarray,
    config: GDConfig | None = None,
    callback: Callable[[int, float, float], None] | None = None,
) -> GDResult:
    cfg = config or GDConfig()
    cfg.validate()

    theta = np.array(theta0, dtype=np.float64, copy=True)
    history = GDHistory()
    J_prev = np.nan
    stop: StopReason = StopReason.MAX_ITER
    detail: str | None = None
    J, gn = np.nan, np.nan
    k = 0
    t0 = time.perf_counter()

    with np.errstate(all="ignore"):
        for k in range(cfg.max_iter + 1):
            J, g = objective.loss_and_grad(theta)
            gn = float(np.linalg.norm(g))
            dJ = abs(J - J_prev) if k > 0 else np.nan
            w_norm = float(np.linalg.norm(theta[:-1]))

            is_last_record = False
            detail = _diagnose(J, g, theta, cfg.divergence_threshold)
            if detail is not None:
                stop, is_last_record = StopReason.DIVERGENCE, True
            elif gn < cfg.tol_grad:
                stop, is_last_record = StopReason.GRAD_NORM, True
            elif k > 0 and dJ < cfg.tol_loss:
                stop, is_last_record = StopReason.LOSS_CHANGE, True
            elif k == cfg.max_iter:
                stop, is_last_record = StopReason.MAX_ITER, True

            if k % cfg.record_every == 0 or is_last_record:
                history.append(k, float(J), gn, float(dJ), w_norm)
            if callback is not None:
                callback(k, float(J), gn)
            if is_last_record:
                break

            theta -= cfg.alpha * g
            J_prev = J

    elapsed = (time.perf_counter() - t0) * 1000.0
    flags = {
        "has_nan": bool(np.isnan(J) or np.any(np.isnan(theta))),
        "has_inf": bool(np.isinf(J) or np.any(np.isinf(theta))),
        "exp_overflow_events": int(objective.overflow_events),
        "loss_evaluations": int(objective.n_loss_evals),
        "gradient_evaluations": int(objective.n_grad_evals),
    }
    return GDResult(
        theta=theta,
        stop_reason=stop,
        iterations=k,
        history=history,
        final_loss=float(J),
        final_grad_norm=float(gn),
        elapsed_ms=elapsed,
        numeric_flags=flags,
        divergence_detail=detail if stop == StopReason.DIVERGENCE else None,
    )
