# Metodología experimental

## Objetivo de los experimentos

Los experimentos no buscan el mejor clasificador posible para WDBC. Buscan **verificar empíricamente** las afirmaciones teóricas del proyecto:

| Afirmación teórica | Experimento |
|---|---|
| El GD converge al minimizador y el modelo resultante es útil | E1 |
| El paso $\alpha$ determina el régimen: lento, rápido, oscilatorio o divergente | E2, E3 |
| $2/L$ es suficiente pero no necesario; el umbral real lo marca la curvatura local | E3 |
| Sin regularización, el minimizador no existe (datos cuasi separables) | E4 |
| El gradiente analítico es correcto; las diferencias finitas tienen un $h$ óptimo | E5 |
| La formulación Log-Sum-Exp es estable y las alternativas no | E2 (modo ingenuo), E6 |

## Configuración común

| Parámetro | Valor |
|---|---|
| Datos | WDBC, partición estratificada 80/20, semilla 42 ([dataset](dataset.md)) |
| Precisión | `float64` |
| Regularización | $\lambda=10^{-3}$ (salvo en E4) |
| Formulación | `logits` / Log-Sum-Exp (salvo en los escenarios ingenuos de E2 y en E6) |
| Inicialización | $\boldsymbol\theta^{(0)}=\mathbf 0$ (salvo en E5, que usa un $\boldsymbol\theta$ aleatorio con semilla 0) |
| Umbral de divergencia | $J>10^{8}$ |
| Umbral de clasificación | $0.5$ |

## Experimentos

| ID | Descripción | Variables | Salida |
|---|---|---|---|
| **E1** | Modelo por defecto | $\alpha=1$, $k_{\max}=8000$, $\text{tol\_grad}=10^{-5}$, $\text{tol\_loss}=10^{-12}$ | `e1_default_model.json`, `default_model_convergence.png`, `roc_test.png` |
| **E2** | Seis escenarios comparativos (los mismos que muestra la pestaña de Telemetría) | $\alpha\in\{0.05,1/L,5,100,10^5\}$ en modo `logits` y $\alpha=1$ en modo `naive`; $k_{\max}=3000$ | `e2_scenarios.json`, `scenarios_loss.png` |
| **E3** | Barrido de la tasa de aprendizaje | 29 valores de $\alpha$ en escala logarítmica en $[10^{-2},10^{5}]$; $k_{\max}=5000$ | `e3_learning_rate_sweep.json`, `learning_rate_sweep.png` |
| **E4** | Efecto de la regularización | $\lambda\in\{0,10^{-3}\}$, $\alpha=5$, 4000 iteraciones sin criterio de convergencia | `e4_regularization.json`, `regularization_weight_norm.png` |
| **E5** | Auditoría del gradiente | chequeo con $h=10^{-5}$ en tres puntos; barrido de $h$ con 49 valores en $[10^{-16},1]$ | `e5_gradient_audit.json`, `finite_difference_v_curve.png` |
| **E6** | Laboratorio IEEE-754 | rejilla de 372 valores de $z\in[-800,60]$ y 11 valores de prueba | `e6_float_audit.json`, `float_loss_formulations.png` |

## Cómo reproducir

```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt   # incluye matplotlib
cd ..
python scripts/reproduce_experiments.py                   # ≈ 10 s
```

- Las salidas se escriben en `results/metrics/` (JSON estricto: `NaN`/`Inf` → `null`) y en `results/figures/` (PNG).
- `--no-figures` omite matplotlib; `--output DIR` cambia el destino.
- `results/metrics/environment.json` registra las versiones de Python, NumPy, scikit-learn y la plataforma.
- La prueba `tests/test_experiments_script.py` ejecuta el script en un directorio temporal y comprueba que E1 reproduce exactamente 4183 iteraciones.

## Qué es determinista y qué no

| Resultado | ¿Reproducible? |
|---|---|
| Partición, $\boldsymbol\theta$ final, iteraciones, motivos de parada, métricas | sí, con las mismas versiones de bibliotecas |
| Últimos bits de $J$ y $\boldsymbol\theta$ entre máquinas o versiones de BLAS distintas | pueden diferir ligeramente |
| Tiempos de ejecución (`elapsed_ms_reference_only`) | **no**: dependen del hardware; solo se registran como referencia |

## Amenazas a la validez

- **Una sola partición.** Las métricas de prueba dependen de qué 114 casos cayeron en prueba. No se hizo validación cruzada ni se calcularon intervalos de confianza.
- **Hiperparámetros fijados a mano.** $\lambda=10^{-3}$ no se seleccionó con validación. El experimento E4 solo compara $0$ frente a $10^{-3}$.
- **Sin comparación externa.** No se contrasta con otra implementación de regresión logística. La validación de optimalidad se hace frente a un método de Newton propio (en las pruebas), que comparte la definición de $J$.
