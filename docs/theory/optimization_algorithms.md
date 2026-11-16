# Algoritmos de optimización

## 1. Gradiente descendente de paso fijo

**Problema.** Encontrar $\boldsymbol\theta^\star=\arg\min J(\boldsymbol\theta)$ cuando no hay solución cerrada, usando solo evaluaciones de $J$ y $\nabla J$.

**Definición formal.**

```math
\boldsymbol\theta^{(k+1)}=\boldsymbol\theta^{(k)}-\alpha\,\nabla J\big(\boldsymbol\theta^{(k)}\big),\qquad k=0,1,2,\dots
```

que, desglosada para los dos bloques de parámetros, da

```math
\boldsymbol w^{(k+1)}=\boldsymbol w^{(k)}-\alpha\Big[\tfrac1m X^\top(\boldsymbol p^{(k)}-\boldsymbol y)+\lambda\boldsymbol w^{(k)}\Big],\qquad
b^{(k+1)}=b^{(k)}-\alpha\,\tfrac1m\textstyle\sum_i\big(p^{(k)}_i-y_i\big).
```

**Variables.** $\alpha>0$: tasa de aprendizaje (paso), constante en todas las iteraciones; $\boldsymbol\theta^{(0)}$: punto inicial (`zeros`, `random` con $\mathcal N(0,0.5^2)$ y semilla, o `trained`).

**Intuición.** $-\nabla J$ es la dirección de **máximo descenso local**: entre todas las direcciones unitarias $\boldsymbol u$, la derivada direccional $\nabla J^\top\boldsymbol u$ es mínima para $\boldsymbol u=-\nabla J/\lVert\nabla J\rVert$ (desigualdad de Cauchy-Schwarz). El paso $\alpha$ decide cuánto se avanza en esa dirección.

**Por qué gradiente descendente y no Newton.** El método de Newton también sería viable aquí: la Hessiana es de solo $31\times31$, y las pruebas lo usan como referencia. Se eligió el gradiente descendente porque el objetivo del proyecto es **estudiar explícitamente** el efecto del paso, la convergencia, la divergencia y la estabilidad numérica.

## 2. Pseudocódigo (tal como está implementado)

```text
entrada: objetivo J, θ⁰, α, k_max, tol_grad, tol_loss, umbral_div = 1e8
θ ← copia(θ⁰);  J_prev ← NaN
para k = 0, 1, …, k_max:
    J, g ← J(θ), ∇J(θ)
    ‖g‖ ← norma2(g);  ΔJ ← |J − J_prev| (si k > 0)
    si J es NaN/±Inf, J > umbral_div, o g/θ no finitos:  parar → DIVERGENCE
    si ‖g‖ < tol_grad:                                    parar → GRAD_NORM
    si k > 0 y ΔJ < tol_loss:                             parar → LOSS_CHANGE
    si k = k_max:                                         parar → MAX_ITER
    registrar (k, J, ‖g‖, ΔJ, ‖w‖)
    θ ← θ − α g;  J_prev ← J
devolver θ, motivo de parada, historial, banderas numéricas
```

**Orden de los criterios.** La divergencia se comprueba **primero**. Si $J=+\infty$ en dos iteraciones seguidas, $\Delta J=\infty-\infty=\texttt{NaN}$, y la comparación `NaN < tol` es falsa. Pero si el orden fuera otro, un estado no finito podría llegar a evaluarse contra criterios de convergencia. El optimizador **nunca lanza excepciones** por inestabilidad: devuelve el estado y un diagnóstico textual (`divergence_detail`).

**Correspondencia con el código:** `app/optimization/gradient_descent.py::gradient_descent`, con `GDConfig` (parámetros), `GDHistory` (telemetría), `GDResult` (salida) y `StopReason` (motivo).

## 3. Constante de suavidad L

**Definición.** $\nabla J$ es $L$-Lipschitz si $\lVert\nabla J(\boldsymbol a)-\nabla J(\boldsymbol b)\rVert\le L\lVert\boldsymbol a-\boldsymbol b\rVert$. Para $J\in C^2$ equivale a $\nabla^2J\preceq LI$.

**Cálculo para la regresión logística.** De la [Hessiana](calculus.md#4-la-hessiana) y la cota $p(1-p)\le\frac14$, para todo $\boldsymbol v$ unitario:

```math
\boldsymbol v^\top\nabla^2J\,\boldsymbol v=\frac1m\sum_i p_i(1-p_i)(\tilde{\boldsymbol x}_i^\top\boldsymbol v)^2+\lambda\sum_{j\le d}v_j^2
\le\frac{1}{4m}\boldsymbol v^\top\tilde X^\top\tilde X\boldsymbol v+\lambda
\le\frac{1}{4m}\lambda_{\max}\big(\tilde X^\top\tilde X\big)+\lambda=:L .
```

**Valor medido:** $L=3.3455$, de modo que $2/L=0.5978$ y $1/L=0.2989$.

**Código.**

```python
def lipschitz_constant(self) -> float:
    Xa = np.hstack([self.X, np.ones((self.m, 1))])
    lam_max = float(np.linalg.eigvalsh(Xa.T @ Xa / self.m)[-1])
    return 0.25 * lam_max + self.l2
```

**Observación.** $L$ es una cota **global y pesimista**: supone que todos los pacientes están en la frontera ($p_i=\frac12$). Cerca del óptimo casi todos están bien clasificados, $p_i(1-p_i)\ll\frac14$, y la curvatura real es mucho menor: $\lambda_{\max}(\nabla^2J(\boldsymbol\theta^\star))=0.1132$, unas 30 veces menor que $L$.

## 4. Convergencia

### 4.1 Descenso garantizado si 0 < α < 2/L

Con el [lema de descenso](calculus.md#5-expansiones-de-taylor-usadas) y $\boldsymbol\delta=-\alpha\nabla J$:

```math
J\big(\boldsymbol\theta-\alpha\nabla J\big)\le J(\boldsymbol\theta)-\alpha\Big(1-\frac{\alpha L}{2}\Big)\lVert\nabla J(\boldsymbol\theta)\rVert^2 .
```

El factor $\alpha(1-\alpha L/2)$ es positivo si y solo si $0<\alpha<2/L$. En ese caso, $J$ decrece estrictamente en cada iteración mientras $\nabla J\ne\mathbf 0$. El descenso por iteración es máximo con $\alpha=1/L$, que da $J^{(k+1)}\le J^{(k)}-\frac{1}{2L}\lVert\nabla J^{(k)}\rVert^2$. Esta desigualdad se verifica en `test_descent_lemma_with_step_one_over_L` y su consecuencia, la monotonía de $J^{(k)}$, en `test_loss_monotone_for_small_step`.

### 4.2 Velocidad lineal con convexidad fuerte

Si además $\nabla^2J\succeq\mu I$ con $\mu>0$, entonces con $\alpha=1/L$:

```math
J(\boldsymbol\theta^{(k)})-J^\star\le\Big(1-\frac{\mu}{L}\Big)^{k}\big(J(\boldsymbol\theta^{(0)})-J^\star\big).
```

El número de condición $\kappa=L/\mu$ controla la velocidad. Aquí, con la cota global, $\mu/L\approx1.0005\times10^{-3}/3.3455\approx3\times10^{-4}$, lo que da un factor de $0.9997$ por iteración. Eso explica por qué $\alpha=1/L$ **no alcanza** la tolerancia en 3000–5000 iteraciones (experimentos E2 y E3).

### 4.3 La cota 2/L es suficiente, no necesaria: análisis local

Cerca del óptimo, $\nabla J(\boldsymbol\theta)\approx H^\star(\boldsymbol\theta-\boldsymbol\theta^\star)$ con $H^\star=\nabla^2J(\boldsymbol\theta^\star)$. El error $\boldsymbol e_k=\boldsymbol\theta^{(k)}-\boldsymbol\theta^\star$ sigue entonces la iteración lineal

```math
\boldsymbol e_{k+1}\approx(I-\alpha H^\star)\,\boldsymbol e_k ,
```

que converge si y solo si $\lvert1-\alpha\lambda_j\rvert<1$ para todos los autovalores $\lambda_j$ de $H^\star$, es decir, si $\alpha<2/\lambda_{\max}(H^\star)$.

**Comparación con lo medido** (E3, barrido de $\alpha$ con $\boldsymbol\theta^{(0)}=\mathbf 0$):

| Magnitud | Valor |
|---|---|
| $2/L$ (cota global suficiente) | $0.598$ |
| $2/\lambda_{\max}(H^\star)$ (umbral local predicho) | $17.67$ |
| mayor $\alpha$ del barrido que converge | $17.78$ (243 iteraciones) |
| menor $\alpha$ del barrido que ya no converge | $31.6$ |

La transición observada está donde predice el análisis local. Que $\alpha=17.78$, ligeramente por encima de $17.67$, todavía converja indica que la linealización es solo una aproximación: la Hessiana cambia a lo largo de la trayectoria. No se investigó la causa en detalle.

### 4.4 Regímenes observados

| Régimen | Rango de $\alpha$ (este conjunto) | Comportamiento |
|---|---|---|
| Lento pero monótono | $\alpha\le2/L\approx0.6$ | $J$ decrece siempre; no llega a $\text{tol}=10^{-5}$ en 5000 iteraciones |
| Convergencia rápida | $1\le\alpha\le17.8$ | para por $\lVert\nabla J\rVert<10^{-5}$ en 4183 a 243 iteraciones |
| Oscilatorio | $31.6\le\alpha\le1.8\times10^{3}$ | $J$ oscila sin converger (p. ej. $J\approx18.75$ con $\alpha=100$) |
| Divergente | $\alpha\ge3.2\times10^{3}$ | $J>10^{8}$ en 2 a 7 iteraciones |

## 5. Criterios de parada

| Motivo (`StopReason`) | Condición | Interpretación |
|---|---|---|
| `divergence` | $J$ `NaN`/$\pm\infty$, $J>10^8$, o $\nabla J$/$\boldsymbol\theta$ no finitos | fallo numérico o paso demasiado grande |
| `grad_norm` | $\lVert\nabla J\rVert_2<\text{tol\_grad}$ ($10^{-5}$) | punto estacionario aproximado y, por convexidad, mínimo global aproximado |
| `loss_change` | $\lvert J^{(k)}-J^{(k-1)}\rvert<\text{tol\_loss}$ ($10^{-12}$) | estancamiento (puede ser prematuro con $\alpha$ pequeño) |
| `max_iter` | $k=k_{\max}$ | presupuesto agotado sin cumplir tolerancias |

**Garantía del criterio de gradiente.** Para $J$ $\mu$-fuertemente convexa, $J(\boldsymbol\theta)-J^\star\le\frac{1}{2\mu}\lVert\nabla J(\boldsymbol\theta)\rVert^2$ y $\lVert\boldsymbol\theta-\boldsymbol\theta^\star\rVert\le\frac1\mu\lVert\nabla J\rVert$. `test_gradient_descent_reaches_newton_minimizer` comprueba ambas desigualdades frente al $\boldsymbol\theta^\star$ que obtiene un método de Newton independiente.

## 6. Complejidad

| Recurso | Por iteración | Total |
|---|---|---|
| Tiempo | $O(md)$: dos productos matriz-vector ($X\boldsymbol w$ y $X^\top\boldsymbol r$) | $O(k\,md)$ |
| Memoria de trabajo | $O(m+d)$ (logits, residuos, gradiente) | — |
| Memoria del historial | $O(1)$ por registro | $O(k/\texttt{record\_every})$ |

**Rendimiento medido** (referencial, depende de la máquina): el modelo por defecto (4183 iteraciones) tarda unos 0.26 s en el entorno donde se generaron los resultados.

## 7. Limitaciones del método

- **Sensibilidad al paso:** un $\alpha$ fijo no se adapta a la curvatura local. No hay búsqueda de línea (Armijo) ni pasos adaptativos (momento, Adam).
- **Convergencia lenta en problemas mal condicionados:** el factor $1-\mu/L$ puede estar muy cerca de 1.
- **Lote completo:** cada iteración usa las 455 muestras. Para conjuntos grandes convendría un método estocástico.
- El modo `naive` puede marcar como `divergence` una ejecución cuyos **parámetros son finitos y útiles**: el fallo está en la *evaluación* de $J$, no en la iteración (ver [resultados E2](../experiments/results.md#e2--escenarios-comparativos)).
