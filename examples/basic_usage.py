"""Uso mínimo del motor numérico sin la API web.

Ejecutar desde la raíz del repositorio:

    python examples/basic_usage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np

from app.data.loader import load_wdbc
from app.evaluation.metrics import classification_report
from app.math_engine import LogisticObjective, gradient_check, sigmoid
from app.optimization import GDConfig, gradient_descent


def main() -> None:
    # 1. Datos: partición estratificada 80/20 (semilla 42), estandarización ajustada solo en train.
    ds = load_wdbc()

    # 2. Objetivo J(θ) = BCE(Log-Sum-Exp) + (λ/2)‖w‖², con θ = [w; b] ∈ ℝ³¹.
    obj = LogisticObjective(ds.X_train, ds.y_train, l2=1e-3)
    L = obj.lipschitz_constant()
    print(f"Constante de suavidad L = {L:.4f}  →  cota de paso 2/L = {2 / L:.4f}")

    # 3. Verificación del gradiente analítico por diferencias centrales en un θ aleatorio.
    theta_r = np.random.default_rng(0).normal(0.0, 0.5, obj.dim)
    chk = gradient_check(obj.loss, obj.grad, theta_r, h=1e-5)
    print(f"Chequeo de gradiente: error relativo = {chk['relative_error']:.2e} ({chk['verdict']})")

    # 4. Gradiente descendente de paso fijo.
    res = gradient_descent(obj, np.zeros(obj.dim), GDConfig(alpha=1.0, max_iter=8000, tol_grad=1e-5))
    print(f"Parada: {res.stop_reason.value} en k = {res.iterations}, J = {res.final_loss:.6f}, "
          f"‖∇J‖ = {res.final_grad_norm:.2e}")

    # 5. Evaluación en el conjunto de prueba (114 pacientes).
    p = sigmoid(ds.X_test @ res.w + res.b)
    rep = classification_report(ds.y_test, p)
    print(f"Test: exactitud {rep['accuracy']:.3f}, sensibilidad {rep['sensitivity']:.3f}, "
          f"especificidad {rep['specificity']:.3f}, AUC {rep['auc']:.3f}")


if __name__ == "__main__":
    main()
