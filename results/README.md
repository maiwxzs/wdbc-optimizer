# Resultados

Salidas **generadas** por `scripts/reproduce_experiments.py`. No las edite a mano: vuelva a ejecutar el script.

```bash
python scripts/reproduce_experiments.py          # regenera metrics/ y figures/
```

## `metrics/`

| Archivo | Experimento |
|---|---|
| `environment.json` | versiones de Python, NumPy y scikit-learn, plataforma, semilla y λ |
| `e1_default_model.json` | E1 · modelo por defecto (α = 1): parada, iteraciones y métricas de entrenamiento y prueba |
| `e2_scenarios.json` | E2 · seis escenarios de α y de formulación (historiales submuestreados) |
| `e3_learning_rate_sweep.json` | E3 · barrido de 29 valores de α en [1e-2, 1e5] |
| `e4_regularization.json` | E4 · λ = 0 frente a λ = 1e-3 |
| `e5_gradient_audit.json` | E5 · chequeo de gradiente en tres puntos y barrido de h |
| `e6_float_audit.json` | E6 · laboratorio IEEE-754 |

Son JSON estrictos: los valores no finitos (NaN/±Inf) aparecen como `null`. El campo `elapsed_ms_reference_only` depende de la máquina.

## `figures/`

| Figura | Contenido |
|---|---|
| `default_model_convergence.png` | J − J_final y ‖∇J‖ por iteración (E1) |
| `roc_test.png` | curva ROC en prueba (E1) |
| `scenarios_loss.png` | J por iteración en los seis escenarios (E2) |
| `learning_rate_sweep.png` | iteraciones y J final frente a α, con la cota 2/L (E3) |
| `regularization_weight_norm.png` | ‖w‖ por iteración con y sin λ (E4) |
| `finite_difference_v_curve.png` | error relativo frente a h, central y progresiva (E5) |
| `float_loss_formulations.png` | pérdida por muestra: Log-Sum-Exp, *clip* e ingenua (E6) |

Interpretación de todos los resultados: [`docs/experiments/results.md`](../docs/experiments/results.md).
