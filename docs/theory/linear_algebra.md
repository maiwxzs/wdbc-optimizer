# Álgebra lineal

Todo el motor numérico está **vectorizado**: no hay bucles de Python sobre las muestras. Este documento describe las operaciones matriciales que aparecen en el código, sus dimensiones y por qué son correctas.

## 1. Producto matriz-vector: el modelo lineal

**Problema.** Calcular los $m$ logits $z_i=\boldsymbol w^\top\boldsymbol x_i+b$ de una sola vez.

**Definición formal.**

```math
\boldsymbol z = X\boldsymbol w + b\,\mathbf 1_m \;=\; \tilde X\boldsymbol\theta,
\qquad X\in\mathbb R^{m\times d},\;\boldsymbol w\in\mathbb R^{d},\;\boldsymbol z\in\mathbb R^{m}.
```

**Variables.** $X$: matriz de diseño estandarizada; $\boldsymbol w$: pesos; $b$: sesgo que se suma a cada componente (*broadcasting* de NumPy); $\tilde X=[X\mid\mathbf 1]$ y $\boldsymbol\theta=[\boldsymbol w;b]$ dan la forma compacta equivalente.

**Intuición.** Cada logit es una suma ponderada de las 30 medidas del paciente. Un peso $w_j>0$ indica que valores altos de la característica $j$ empujan hacia "maligno".

**Ejemplo numérico.** Con $X=\begin{pmatrix}1&0\\-2&1\end{pmatrix}$, $\boldsymbol w=(0.5,-1)^\top$, $b=0.2$:

```math
\boldsymbol z=\begin{pmatrix}1\cdot0.5+0\cdot(-1)\\-2\cdot0.5+1\cdot(-1)\end{pmatrix}+0.2=\begin{pmatrix}0.7\\-1.8\end{pmatrix}.
```

**Código.**

```python
# app/math_engine/gradients.py — LogisticObjective.logits
w, b = unpack(theta)
return self.X @ w + b
```

**Coste.** $O(md)$ operaciones: $455\times30\approx1.4\times10^4$ multiplicaciones-suma por evaluación.

## 2. Producto con la transpuesta: el gradiente

El gradiente de la pérdida respecto a $\boldsymbol w$ es una combinación lineal de las filas de $X$ ponderada por los residuos $\boldsymbol r=\boldsymbol p-\boldsymbol y$:

```math
\frac1m X^\top\boldsymbol r=\frac1m\sum_{i=1}^{m} r_i\,\boldsymbol x_i\in\mathbb R^{d}.
```

**Dimensiones.** $X^\top\in\mathbb R^{d\times m}$ y $\boldsymbol r\in\mathbb R^{m}$ dan un vector de $\mathbb R^{d}$. Esa comprobación de dimensiones es la forma más rápida de detectar un error de transposición; las pruebas la verifican (`test_shapes_of_logits_and_gradient`).

**Intuición.** Cada paciente "vota" en la dirección de su vector de características con un peso igual a su error de predicción. Los pacientes bien clasificados ($r_i\approx0$) apenas influyen.

**Código.** `g_w = (self.X.T @ residual) / self.m`

## 3. Normas

| Norma | Definición | Uso en el proyecto |
|---|---|---|
| $\lVert\boldsymbol v\rVert_2$ | $\sqrt{\sum_j v_j^2}$ | criterio de parada $\lVert\nabla J\rVert_2<\text{tol}$; telemetría $\lVert\boldsymbol w\rVert_2$; error relativo del chequeo de gradiente |
| $\lVert\boldsymbol w\rVert_2^2$ | $\boldsymbol w^\top\boldsymbol w$ | término de regularización $\frac\lambda2\lVert\boldsymbol w\rVert^2$ (`np.dot(w, w)`) |

`np.linalg.norm` calcula la norma 2 con escalado interno, así que no desborda aunque las componentes sean grandes.

## 4. Matrices simétricas semidefinidas positivas y autovalores

**Definición.** $A\in\mathbb R^{n\times n}$ simétrica es *semidefinida positiva* ($A\succeq0$) si $\boldsymbol v^\top A\boldsymbol v\ge0$ para todo $\boldsymbol v$. Sus autovalores son reales y no negativos.

**Cociente de Rayleigh.** Para $A$ simétrica,

```math
\lambda_{\max}(A)=\max_{\boldsymbol v\ne\mathbf 0}\frac{\boldsymbol v^\top A\boldsymbol v}{\boldsymbol v^\top\boldsymbol v}.
```

Esta identidad es la que convierte la cota $p(1-p)\le\frac14$ en una cota sobre la Hessiana (ver [optimización](optimization_algorithms.md#3-constante-de-suavidad-l)).

**Matriz de Gram.** $\tilde X^\top\tilde X$ siempre es simétrica y semidefinida positiva, porque $\boldsymbol v^\top\tilde X^\top\tilde X\boldsymbol v=\lVert\tilde X\boldsymbol v\rVert^2\ge0$.

**Código.**

```python
# LogisticObjective.lipschitz_constant
Xa = np.hstack([self.X, np.ones((self.m, 1))])
lam_max = float(np.linalg.eigvalsh(Xa.T @ Xa / self.m)[-1])
```

`eigvalsh` aprovecha la simetría: es más estable y más rápido que `eig`, y devuelve los autovalores **ordenados de menor a mayor**, por eso se toma el último.

**Coste.** Formar $\tilde X^\top\tilde X$ cuesta $O(md^2)$ y la descomposición, $O(d^3)$. Con $d+1=31$ es despreciable y se hace una sola vez por entrenamiento.

## 5. Estandarización (transformación afín por columnas)

Antes de optimizar, cada característica se centra y escala con los estadísticos **del conjunto de entrenamiento**:

```math
X_{ij}=\frac{X^{\text{raw}}_{ij}-\mu_j}{s_j},\qquad
\mu_j=\frac1m\sum_i X^{\text{raw}}_{ij},\quad s_j=\sqrt{\frac1m\sum_i\big(X^{\text{raw}}_{ij}-\mu_j\big)^2}.
```

**Por qué importa numéricamente.** Las características originales tienen escalas muy distintas: el área media llega a miles y la dimensión fractal está en torno a $0.06$. Sin estandarizar, $\tilde X^\top\tilde X$ estaría muy mal condicionada, $L$ sería enorme y el paso seguro $2/L$ minúsculo. Tras estandarizar, todas las columnas tienen media 0 y varianza 1 en entrenamiento, y lo comprueba `test_standardization_uses_train_statistics_only`.

**Implementación.** Esta transformación la realiza `sklearn.preprocessing.StandardScaler` en `app/data/loader.py`, uno de los cuatro usos de scikit-learn que permite el proyecto (ver [decisiones de diseño](../architecture/design_decisions.md)). Usa la desviación típica poblacional ($1/m$).

## 6. Proyección PCA (solo visualización)

Para dibujar el mapa 2D del panel clínico se proyectan los pacientes sobre las dos direcciones de máxima varianza del conjunto de entrenamiento estandarizado (análisis de componentes principales). Explican el $44.6\,\%$ y el $18.5\,\%$ de la varianza. La proyección **no interviene** en el entrenamiento ni en la predicción; la calcula `sklearn.decomposition.PCA`.

## 7. Consideraciones numéricas

- Las matrices se convierten a `float64` contiguo (`np.ascontiguousarray`) para que los productos usen BLAS sin copias.
- No se invierte ninguna matriz en el código de producción. La única resolución de sistemas lineales (`np.linalg.solve`) está en el método de Newton que **las pruebas** usan como referencia independiente (`tests/test_numerical_properties.py`).
