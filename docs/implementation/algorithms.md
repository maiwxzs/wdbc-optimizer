# Algoritmos

Ficha técnica de cada algoritmo implementado. Las derivaciones matemáticas están en `docs/theory/`. Aquí se documentan el contrato, el pseudocódigo y la complejidad. Notación: $m$ = muestras, $d$ = características, $n=d+1$ = parámetros.

> **Complejidad teórica frente a rendimiento medido.** Las cotas $O(\cdot)$ son teóricas. Los tiempos indicados en otros documentos son medidas puntuales en una máquina concreta y no forman parte del contrato.

---

## A1 · Sigmoide estable

| | |
|---|---|
| **Objetivo** | $\sigma(z)=1/(1+e^{-z})$ sin desbordamiento |
| **Fundamento** | [floating_point.md §3](../theory/floating_point.md#3-sigmoide-estable-por-tramos) |
| **Entrada / salida** | escalar o arreglo de cualquier forma → mismo tipo y forma (`float` si la entrada es escalar) |
| **Complejidad** | tiempo $O(N)$, memoria $O(N)$ ($N$ = número de elementos) |
| **Implementación** | `app/math_engine/activations.py::sigmoid` |

```text
e ← exp(−|z|)                        ▷ e ∈ (0, 1]
σ ← 1/(1+e)   donde z ≥ 0
σ ← e/(1+e)   donde z < 0
```

**Limitación.** Para $z>36.74$ devuelve $1.0$ exacto (el valor correctamente redondeado). No sirve para calcular $\log(1-\sigma)$: para eso está Log-Sum-Exp.

## A2 · Entropía cruzada desde logits (Log-Sum-Exp)

| | |
|---|---|
| **Objetivo** | $\frac1m\sum_i[\log(1+e^{z_i})-y_iz_i]$ exacta |
| **Fundamento** | [loss_functions.md §2](../theory/loss_functions.md#2-formulación-log-sum-exp-la-que-usa-el-motor) |
| **Entrada / salida** | `y`, `z` de longitud $m$ → `float` |
| **Complejidad** | $O(m)$ |
| **Implementación** | `losses.py::bce_from_logits`; por muestra: `bce_per_sample_from_logits` |

## A3 · Objetivo y gradiente analítico

| | |
|---|---|
| **Objetivo** | $J(\boldsymbol\theta)$ y $\nabla J(\boldsymbol\theta)$ |
| **Fundamento** | [calculus.md §3](../theory/calculus.md#3-el-gradiente-de-j-regla-de-la-cadena) |
| **Parámetros** | `l2` ($\lambda\ge0$), `mode` ∈ {`logits`, `clipped`, `naive`}, `eps` ($10^{-15}$) |
| **Entrada / salida** | `theta` ($n$) → `float` / arreglo ($n$) |
| **Complejidad** | $O(md)$ en tiempo, $O(m+n)$ de memoria adicional |
| **Implementación** | `gradients.py::LogisticObjective.loss`, `.grad`, `.loss_and_grad`; forma separada: `logistic_loss`, `logistic_gradient` |

```text
w, b ← θ[:-1], θ[-1]
z ← X w + b
r ← σ(z) − y
∇w ← Xᵀ r / m + λ w
∂b ← Σ r / m
devolver [∇w; ∂b]
```

**Limitación.** `loss_and_grad` calcula $\boldsymbol z$ dos veces (una en `loss` y otra en `grad`). Es un coste $O(md)$ extra por iteración que se aceptó por claridad.

## A4 · Constante de suavidad L

| | |
|---|---|
| **Objetivo** | cota $L$ con $\nabla^2J\preceq LI$ |
| **Fundamento** | [optimization_algorithms.md §3](../theory/optimization_algorithms.md#3-constante-de-suavidad-l) |
| **Salida** | `float` ($3.3455$ en WDBC con $\lambda=10^{-3}$) |
| **Complejidad** | $O(mn^2+n^3)$: matriz de Gram + `eigvalsh` |
| **Implementación** | `LogisticObjective.lipschitz_constant` |

## A5 · Gradiente descendente de paso fijo

| | |
|---|---|
| **Objetivo** | aproximar $\boldsymbol\theta^\star=\arg\min J$ |
| **Fundamento** | [optimization_algorithms.md](../theory/optimization_algorithms.md) |
| **Parámetros** (`GDConfig`) | `alpha` (0.5 por defecto en la clase; 1.0 en el servicio), `max_iter` (5000; 8000 en el servicio), `tol_grad` ($10^{-5}$), `tol_loss` ($10^{-12}$), `divergence_threshold` ($10^8$), `record_every` (1) |
| **Entrada** | objetivo con `loss_and_grad`, `theta0`, `config`, `callback(k, J, ‖∇J‖)` opcional |
| **Salida** | `GDResult`: `theta`, `stop_reason`, `iterations`, `history`, `final_loss`, `final_grad_norm`, `elapsed_ms`, `numeric_flags`, `divergence_detail` |
| **Complejidad** | tiempo $O(k\cdot md)$; memoria $O(md)$ para los datos + $O(k/\texttt{record\_every})$ para el historial |
| **Implementación** | `optimization/gradient_descent.py::gradient_descent` |

Pseudocódigo en [optimization_algorithms.md §2](../theory/optimization_algorithms.md#2-pseudocódigo-tal-como-está-implementado). Garantías: no modifica `theta0` (trabaja sobre una copia; ver `test_gradient_descent_does_not_mutate_initial_point`) y nunca lanza excepciones por valores no finitos. Sí lanza `ValueError` si la configuración no es válida.

## A6 · Diferencias finitas (central y progresiva)

| | |
|---|---|
| **Objetivo** | aproximar $\nabla f(\boldsymbol\theta)$ evaluando solo $f$ |
| **Fundamento** | [numerical_differentiation.md](../theory/numerical_differentiation.md) |
| **Parámetros** | `h` (central: $10^{-5}$; progresiva: $10^{-8}$) |
| **Complejidad** | central: $2n$ evaluaciones de $f$; progresiva: $n+1$. Con $f=J$, $O(n\cdot md)$ |
| **Implementación** | `central_difference_gradient`, `forward_difference_gradient` |

## A7 · Chequeo de gradiente

| | |
|---|---|
| **Objetivo** | validar $\nabla J$ analítico contra la diferencia central |
| **Salida** | gradientes, errores absolutos y relativos por componente, error relativo global, veredicto |
| **Complejidad** | $O(n\cdot md)$ |
| **Implementación** | `gradient_check`, `relative_error`, `classify_relative_error` |

## A8 · Barrido de h y ajuste del modelo de error

| | |
|---|---|
| **Objetivo** | curva $E(h)$ y paso óptimo empírico y teórico |
| **Entrada** | `hs` (vector de pasos; el servicio usa `logspace(-16, 0, 49)`) |
| **Salida** | errores central y progresivo, curvas del modelo, mejores $h$, $h^\star$ ajustado y teórico |
| **Complejidad** | $O(\lvert hs\rvert\cdot 3n\cdot md)$: unas $49\times93$ evaluaciones de $J$ |
| **Implementación** | `step_size_sweep`, `_fit_v_model` |

```text
para cada h en hs:
    E_c(h) ← err_rel(∇J, central(J, θ, h))
    E_p(h) ← err_rel(∇J, progresiva(J, θ, h))
para cada rama (p = 2 central, p = 1 progresiva):
    i* ← argmin E;  usar solo E < 1e-2
    c ← mediana_geom(E·h)     a la izquierda de i*−1
    a ← mediana_geom(E/h^p)   a la derecha de i*+1
    h*_ajuste ← (c / (p·a))^{1/(p+1)}
```

## A9 · AUC por Mann-Whitney

| | |
|---|---|
| **Objetivo** | $P(s^+>s^-)+\frac12P(s^+=s^-)$ |
| **Fundamento** | [probability_and_statistics.md §5.1](../theory/probability_and_statistics.md#51-auc-mediante-el-estadístico-u-de-mann-whitney) |
| **Complejidad** | $O(N\log N)$ por la ordenación; el barrido de empates es $O(N)$ |
| **Implementación** | `evaluation/metrics.py::roc_auc` |

## A10 · Submuestreo del historial

| | |
|---|---|
| **Objetivo** | enviar como máximo `max_points` puntos por curva al navegador |
| **Algoritmo** | índices `round(linspace(0, n−1, max_points))` sin duplicados; conserva el primero y el último |
| **Complejidad** | $O(\text{max\_points})$ |
| **Implementación** | `services/serialization.py::downsample_indices` |

El submuestreo es **uniforme por índice**, no LTTB. El README original menciona LTTB para las curvas largas del frontend, que lo aplica ECharts (`sampling: "lttb"`) en el lado del cliente.

## A11 · Método de Newton (solo en las pruebas)

| | |
|---|---|
| **Objetivo** | referencia independiente de $\boldsymbol\theta^\star$ |
| **Iteración** | $\boldsymbol\theta\leftarrow\boldsymbol\theta-[\nabla^2J]^{-1}\nabla J$ (con `np.linalg.solve`) |
| **Complejidad** | $O(mn^2+n^3)$ por iteración; converge cuadráticamente en pocas iteraciones |
| **Implementación** | `backend/tests/test_numerical_properties.py::_newton` (**no** forma parte del paquete) |
