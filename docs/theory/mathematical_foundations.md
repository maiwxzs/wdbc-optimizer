# Fundamentos matemáticos

Este documento fija la notación que usa toda la documentación y resume, en una sola página, el problema matemático que resuelve el código. Los demás documentos de `docs/theory/` desarrollan cada pieza con detalle.

## 1. Notación

| Símbolo | Significado | Dimensión / valor en el proyecto | En el código |
|---|---|---|---|
| $m$ | número de muestras de entrenamiento | $455$ | `obj.m`, `X.shape[0]` |
| $d$ | número de características | $30$ | `ds.n_features` |
| $X$ | matriz de características **estandarizadas** (una fila por paciente) | $\mathbb R^{m\times d}$ | `ds.X_train` |
| $\boldsymbol x_i$ | fila $i$ de $X$ | $\mathbb R^{d}$ | `X[i]` |
| $y_i$ | etiqueta: $1$ = maligno, $0$ = benigno | $\{0,1\}$ | `ds.y_train` |
| $\boldsymbol w$ | vector de pesos | $\mathbb R^{d}$ | `theta[:-1]` |
| $b$ | sesgo (intercepto) | $\mathbb R$ | `theta[-1]` |
| $\boldsymbol\theta$ | parámetros empaquetados $[\boldsymbol w; b]$ | $\mathbb R^{d+1}=\mathbb R^{31}$ | `pack(w, b)` |
| $\tilde X$ | matriz aumentada $[X \mid \mathbf 1]$ | $\mathbb R^{m\times(d+1)}$ | `Xa` en `lipschitz_constant` |
| $z_i$ | logit $\boldsymbol w^\top\boldsymbol x_i+b$ | $\mathbb R$ | `obj.logits(theta)` |
| $p_i$ | probabilidad estimada $\sigma(z_i)=\hat y_i$ | $(0,1)$ | `sigmoid(z)` |
| $\lambda$ | coeficiente de regularización L2 | $10^{-3}$ por defecto | `l2` |
| $\alpha$ | tasa de aprendizaje (paso fijo) | $1$ por defecto | `GDConfig.alpha` |
| $L$ | constante de suavidad (Lipschitz de $\nabla J$) | $3.3455$ | `obj.lipschitz_constant()` |
| $\varepsilon_{\text{mach}}$ | épsilon de máquina de `float64` | $2^{-52}\approx 2.22\times10^{-16}$ | `EPS_MACHINE` |
| $h$ | paso de diferencias finitas | $10^{-5}$ por defecto | `h` |

Los vectores se escriben en negrita y son columnas. $\lVert\cdot\rVert$ es la norma euclidiana.

## 2. El problema que resuelve el programa

El programa estima la probabilidad de que un tumor de mama sea maligno a partir de 30 medidas morfológicas de núcleos celulares (conjunto WDBC, ver [dataset](../experiments/dataset.md)). Lo hace con **regresión logística**, que aquí no se trata como una caja negra de *machine learning*: se plantea como un **problema de optimización convexa sin restricciones**

```math
\boldsymbol\theta^\star=\arg\min_{\boldsymbol\theta\in\mathbb R^{31}} J(\boldsymbol\theta),\qquad
J(\boldsymbol w,b)=\frac1m\sum_{i=1}^{m}\Big[\log\big(1+e^{z_i}\big)-y_i z_i\Big]+\frac\lambda2\lVert\boldsymbol w\rVert_2^2,
\qquad z_i=\boldsymbol w^\top\boldsymbol x_i+b ,
```

que se resuelve numéricamente con **gradiente descendente de paso fijo**:

```math
\boldsymbol\theta^{(k+1)}=\boldsymbol\theta^{(k)}-\alpha\,\nabla J\big(\boldsymbol\theta^{(k)}\big).
```

No existe una fórmula cerrada para $\boldsymbol\theta^\star$: la condición de optimalidad $\nabla J(\boldsymbol\theta)=\mathbf 0$ es un sistema de 31 ecuaciones **no lineales** (contiene $\sigma(z_i)$). Por eso se necesita un método iterativo.

## 3. Mapa de la teoría y su implementación

| Pieza matemática | Documento | Implementación |
|---|---|---|
| Matrices, normas, autovalores | [Álgebra lineal](linear_algebra.md) | operaciones vectorizadas de NumPy |
| Gradiente, regla de la cadena, Hessiana, Taylor | [Cálculo](calculus.md) | `gradients.py`, `finite_differences.py` |
| Modelo de Bernoulli, verosimilitud, AUC | [Probabilidad y estadística](probability_and_statistics.md) | `losses.py`, `metrics.py` |
| Entropía cruzada y sus tres formulaciones | [Funciones de pérdida](loss_functions.md) | `losses.py` |
| Gradiente descendente, convergencia, $L$ | [Algoritmos de optimización](optimization_algorithms.md) | `gradient_descent.py` |
| Diferencias finitas y curva en V | [Derivación numérica](numerical_differentiation.md) | `finite_differences.py` |
| IEEE-754, desbordamiento, saturación | [Aritmética de punto flotante](floating_point.md) | `activations.py`, `losses.py` |
| Proceso completo de entrenamiento | [Entrenamiento del modelo](model_training.md) | `model_service.py` |

## 4. Propiedades clave (demostradas en los documentos enlazados)

1. **$J$ es convexa.** Su Hessiana $\nabla^2J=\frac1m\tilde X^\top S\tilde X+\lambda D$, con $S=\operatorname{diag}(p_i(1-p_i))\succeq0$, es semidefinida positiva. Todo mínimo local es global.
2. **$J$ es $L$-suave.** Como $p(1-p)\le\frac14$, se tiene $\nabla^2J\preceq L I$ con $L=\frac{1}{4m}\lambda_{\max}(\tilde X^\top\tilde X)+\lambda$.
3. **Con $\lambda>0$ el minimizador es único en la práctica de este conjunto.** La parte de $\boldsymbol w$ es fuertemente convexa por el término $\lambda$; la dirección del sesgo lo es gracias a los datos. En el óptimo medido, $\lambda_{\min}(\nabla^2J(\boldsymbol\theta^\star))\approx1.0005\times10^{-3}>0$.
4. **Con paso $0<\alpha<2/L$ el descenso está garantizado** (condición suficiente, no necesaria). En este conjunto, $2/L\approx0.598$, pero el método converge hasta $\alpha\approx17.8$ porque la curvatura **local** cerca de $\boldsymbol\theta^\star$ es mucho menor que la cota global $L$ (ver [resultados](../experiments/results.md)).
5. **La forma Log-Sum-Exp de la pérdida es exacta en `float64` para todo $z$**, a diferencia de la forma de libro $-[y\log p+(1-y)\log(1-p)]$, que produce `NaN` en cuanto $\sigma(z)$ redondea a $1.0$ ($z\gtrsim36.7$).
