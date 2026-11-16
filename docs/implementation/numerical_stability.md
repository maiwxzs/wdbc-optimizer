# Estabilidad numérica

Resumen operativo de **dónde** puede fallar la aritmética de punto flotante en este programa y **qué** hace el código en cada caso. La teoría está en [aritmética de punto flotante](../theory/floating_point.md).

## Matriz de riesgos

| # | Riesgo | Dónde aparecería | Mitigación implementada | Verificación |
|---|---|---|---|---|
| 1 | `exp` desborda ($z<-709.78$) | sigmoide | sigmoide por tramos con $e^{-\lvert z\rvert}$ | `test_sigmoid_never_overflows` + `RuntimeWarning` como error |
| 2 | $\sigma$ satura a $1.0$ ($z>36.74$) y $\log(1-p)=-\infty$ | pérdida | la pérdida se calcula desde $z$ (Log-Sum-Exp), nunca desde $p$ | `test_bce_handles_saturation` |
| 3 | $0\cdot\log0=\texttt{NaN}$ | BCE de libro | ídem; el modo `naive` lo reproduce a propósito | `test_naive_bce_produces_nan_from_zero_times_log_zero` |
| 4 | Pérdida acotada artificialmente | modo `clipped` | se documenta; no es el modo por defecto | ídem (#2) |
| 5 | Cancelación en $1-p$ | BCE de libro | Log-Sum-Exp usa `log1p` internamente | `test_per_sample_loss_matches_textbook_bce_in_safe_range` (la referencia de libro es la que pierde precisión) |
| 6 | Paso $\alpha$ excesivo → $J\to\infty$ | optimizador | umbral $J>10^8$, detección de `NaN`/`Inf` y parada con diagnóstico | `test_divergence_by_threshold_returns_state` |
| 7 | $\infty-\infty=\texttt{NaN}$ en $\Delta J$ | criterio de parada | la divergencia se evalúa antes que $\lvert\Delta J\rvert<\text{tol}$ | `test_divergence_by_nan_in_naive_mode` |
| 8 | Cancelación en diferencias finitas con $h$ pequeño | auditoría | $h=10^{-5}$ por defecto (cerca de $h^\star$ central); barrido que lo hace visible | `test_step_sweep_has_v_shape` |
| 9 | Error relativo con denominador $\approx0$ | `relative_error` | $+10^{-15}$ en el denominador | `test_relative_error_properties` |
| 10 | `NaN`/`Inf` en la respuesta HTTP | serialización | `to_jsonable`: no finito → `null` | pruebas con `allow_nan=False` |
| 11 | Mal condicionamiento por escalas dispares | datos | estandarización por columna | `test_standardization_uses_train_statistics_only` |
| 12 | Ausencia de minimizador (separación) | objetivo | $\lambda=10^{-3}>0$ por defecto | `test_without_regularization_weights_keep_growing` |
| 13 | Probabilidades `NaN` en métricas | evaluación | `nan_to_num` para la confusión; AUC = `NaN` si hay no finitos | `test_service_train_divergent_is_json_safe` |

## Casos problemáticos conocidos (no mitigados)

- **Modo `naive` con datos de este conjunto:** diverge (por `NaN`) en cuanto algún $\lvert z\rvert$ supera ~36.7, aunque los parámetros sean buenos. Es intencional: así se demuestra el problema.
- **Chequeo de gradiente en el óptimo:** el veredicto baja a "aceptable" por razones puramente numéricas ([explicación](../theory/numerical_differentiation.md#4-chequeo-de-gradiente)).
- **Pesos sin regularizar:** con $\lambda=0$ los pesos crecen sin cota. El optimizador no lo detecta como divergencia, porque $J$ sigue siendo finito y decreciente.
- **$\exp$ en la prueba de integración** `test_reported_test_metrics_match_independent_recomputation`: se usa la sigmoide ingenua como referencia independiente, válida solo porque en el modelo entrenado $\lvert z\rvert<700$. Si eso cambiara, el `RuntimeWarning` haría fallar la prueba en lugar de dar un resultado silenciosamente erróneo.
