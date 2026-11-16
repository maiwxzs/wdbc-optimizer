# Métricas de evaluación

El proyecto mide dos cosas distintas, y conviene no mezclarlas.

## 1. Métricas del optimizador (calidad numérica)

| Métrica | Definición | Dónde se registra |
|---|---|---|
| $J(\boldsymbol\theta^{(k)})$ | valor del objetivo en cada iteración | `GDHistory.loss` |
| $\lVert\nabla J(\boldsymbol\theta^{(k)})\rVert_2$ | distancia a la estacionariedad | `GDHistory.grad_norm` |
| $\lvert J^{(k)}-J^{(k-1)}\rvert$ | progreso por iteración | `GDHistory.loss_change` |
| $\lVert\boldsymbol w^{(k)}\rVert_2$ | crecimiento de los pesos (diagnóstico de separación) | `GDHistory.weight_norm` |
| Iteraciones y motivo de parada | ver [criterios de parada](../theory/optimization_algorithms.md#5-criterios-de-parada) | `GDResult` |
| Banderas numéricas | `has_nan`, `has_inf`, `exp_overflow_events`, número de evaluaciones de $J$ y $\nabla J$ | `GDResult.numeric_flags` |
| $\alpha/(2/L)$ | posición del paso respecto a la cota teórica | respuesta de `/api/train` |
| Error relativo del gradiente | ver [chequeo de gradiente](../theory/numerical_differentiation.md#4-chequeo-de-gradiente) | `gradient_check` |

**Nota sobre `exp_overflow_events`.** Solo se cuenta en modo `naive`, y se incrementa **cada vez** que se activa la sigmoide ingenua, es decir, tanto al evaluar $J$ como $\nabla J$. Por tanto, cuenta elementos-evaluación, no elementos únicos. En E2, el escenario ingenuo con $\alpha=1$ registra $0$ desbordamientos: su `NaN` lo causa la **saturación**, no el desbordamiento.

## 2. Métricas del clasificador (calidad predictiva)

Definiciones completas en [probabilidad y estadística](../theory/probability_and_statistics.md#5-métricas-de-evaluación). Todas están implementadas en NumPy en `app/evaluation/metrics.py`, sin `sklearn.metrics`:

| Función | Devuelve |
|---|---|
| `confusion(y_true, y_pred)` | `{"tp", "tn", "fp", "fn"}` |
| `classification_report(y_true, proba, threshold=0.5)` | matriz de confusión, exactitud, precisión, sensibilidad, especificidad, F1, AUC |
| `roc_auc(y_true, scores)` | AUC por Mann-Whitney con rangos promedio; `NaN` si hay una sola clase o puntuaciones no finitas |
| `roc_curve(y_true, scores, n_points=101)` | `fpr`, `tpr` y umbrales en una rejilla uniforme |

**Divisiones por cero.** `_safe_div` devuelve $0$ cuando el denominador es $0$ (por ejemplo, la precisión cuando no hay ninguna predicción positiva). Es una convención que conviene recordar al leer métricas de modelos degenerados.

**Probabilidades no finitas.** Si un entrenamiento divergente produce `NaN` en $p$, `classification_report` los trata como $0$ para la matriz de confusión e informa la AUC como `NaN` (`null` en JSON).

## 3. Tolerancias usadas en las pruebas

| Comprobación | Tolerancia | Justificación |
|---|---|---|
| Sigmoide estable frente a ingenua en $[-30,30]$ | `rtol = 1e-14` | ambas son exactas hasta unos pocos ulp en ese rango |
| $\sigma(z)+\sigma(-z)=1$ | `atol = 1e-15` | redondeo de una suma de dos números en $[0,1]$ |
| Derivada de $\sigma$ frente a diferencia central | `atol = 1e-10` | error teórico $\approx h^2\lvert\sigma'''\rvert/6+\varepsilon/h\approx2\times10^{-11}$ con $h=10^{-5}$ |
| LSE frente a BCE de libro | `rtol = 1e-10`, `atol = 1e-14` | la referencia pierde precisión en $\log(1-p)$ cuando $p\approx1$ (cancelación) |
| Chequeo de gradiente | `< 1e-7` | el error esperado con $h=10^{-5}$ es $\sim10^{-11}$; el margen cubre otras plataformas |
| Gradiente empaquetado frente a separado | `rtol = 1e-14` | mismas operaciones en distinto orden |
| Convexidad, lema de descenso, $\nabla^2J\preceq LI$ | `+1e-12` | holgura para el redondeo en valores $O(1)$ |
| GD frente a Newton | $0\le J_{\text{GD}}-J^\star\le\lVert\nabla J\rVert^2/(2\mu)$ | cota teórica de convexidad fuerte, no una tolerancia arbitraria |
| Exactitud en prueba | `> 0.95` | umbral de regresión: detecta degradaciones sin fijar un valor exacto |
