# Cálculo multivariable

Este documento deriva, paso a paso, todas las derivadas que el código implementa de forma analítica: la de la sigmoide, la de la pérdida por muestra, el gradiente de $J$ y su Hessiana. También presenta las expansiones de Taylor en las que se apoyan el [análisis de diferencias finitas](numerical_differentiation.md) y la [convergencia del gradiente descendente](optimization_algorithms.md).

## 1. La función sigmoide y su derivada

**Definición.**

```math
\sigma:\mathbb R\to(0,1),\qquad \sigma(z)=\frac{1}{1+e^{-z}}.
```

**Propiedades.**

| Propiedad | Expresión | Comprobación |
|---|---|---|
| Rango | $0<\sigma(z)<1$ (en aritmética exacta) | `test_sigmoid_is_monotone_bounded_and_finite_on_extreme_grid` |
| Simetría | $\sigma(-z)=1-\sigma(z)$ | `test_sigmoid_symmetry_and_scalar` |
| Valor central | $\sigma(0)=\tfrac12$ | ídem |
| Monotonía | estrictamente creciente | primera prueba citada |
| Inversa | $\sigma^{-1}(p)=\log\frac{p}{1-p}$ (*logit*) | — |

**Derivación de $\sigma'(z)$.** Por la regla del cociente:

```math
\sigma'(z)=\frac{d}{dz}\big(1+e^{-z}\big)^{-1}
=-\big(1+e^{-z}\big)^{-2}\cdot\big(-e^{-z}\big)
=\frac{e^{-z}}{(1+e^{-z})^2}
=\underbrace{\frac{1}{1+e^{-z}}}_{\sigma(z)}\cdot\underbrace{\frac{e^{-z}}{1+e^{-z}}}_{1-\sigma(z)} .
```

Por tanto

```math
\boxed{\sigma'(z)=\sigma(z)\,\big(1-\sigma(z)\big)}\qquad\Longrightarrow\qquad 0<\sigma'(z)\le\tfrac14,
```

con el máximo $\tfrac14$ en $z=0$, porque $p(1-p)$ es una parábola con vértice en $p=\tfrac12$. **Esta cota es la que da la constante $L$.**

**Ejemplo.** $\sigma(2)=1/(1+e^{-2})=1/1.13534=0.880797$ y $\sigma'(2)=0.880797\times0.119203=0.104994$. La prueba `test_sigmoid_derivative_matches_closed_form` contrasta esta fórmula con una diferencia central en 81 puntos de $[-20,20]$.

## 2. La pérdida por muestra y su derivada respecto al logit

La pérdida de la muestra $i$ (ver [funciones de pérdida](loss_functions.md)) es

```math
\ell(z,y)=\log\big(1+e^{z}\big)-y\,z .
```

**Derivada.** Se usa $\frac{d}{dz}\log(1+e^{z})=\frac{e^{z}}{1+e^{z}}=\sigma(z)$:

```math
\boxed{\frac{\partial\ell}{\partial z}=\sigma(z)-y=p-y}\qquad\text{(el "residuo")}.
```

**Segunda derivada.** $\dfrac{\partial^2\ell}{\partial z^2}=\sigma'(z)=p(1-p)\ge0$, luego $\ell$ es convexa en $z$.

**Interpretación.** El residuo $p-y$ vale casi $0$ si el modelo acierta con confianza, casi $+1$ si predice "maligno" para un benigno y casi $-1$ en el caso contrario. Su magnitud nunca supera 1, lo que hace al gradiente robusto frente a valores atípicos.

## 3. El gradiente de J (regla de la cadena)

$J(\boldsymbol w,b)=\frac1m\sum_i\ell(z_i,y_i)+\frac\lambda2\lVert\boldsymbol w\rVert^2$, con $z_i=\boldsymbol w^\top\boldsymbol x_i+b$.

**Paso 1: derivadas del logit.**

```math
\frac{\partial z_i}{\partial w_j}=x_{ij},\qquad \frac{\partial z_i}{\partial b}=1 .
```

**Paso 2: regla de la cadena para cada muestra.**

```math
\frac{\partial\ell_i}{\partial w_j}=\frac{\partial\ell_i}{\partial z_i}\frac{\partial z_i}{\partial w_j}=(p_i-y_i)\,x_{ij},
\qquad
\frac{\partial\ell_i}{\partial b}=p_i-y_i .
```

**Paso 3: promedio y regularización.** $\frac{\partial}{\partial\boldsymbol w}\frac\lambda2\boldsymbol w^\top\boldsymbol w=\lambda\boldsymbol w$, y el sesgo **no** se regulariza:

```math
\boxed{\nabla_{\boldsymbol w}J=\frac1m X^\top(\boldsymbol p-\boldsymbol y)+\lambda\boldsymbol w,\qquad
\frac{\partial J}{\partial b}=\frac1m\sum_{i=1}^{m}(p_i-y_i).}
```

En forma empaquetada, $\nabla_{\boldsymbol\theta}J=\frac1m\tilde X^\top(\boldsymbol p-\boldsymbol y)+\lambda D\boldsymbol\theta$, con $D=\operatorname{diag}(1,\dots,1,0)$.

**Ejemplo numérico ($m=2$, $d=1$).** $X=\begin{pmatrix}1\\-2\end{pmatrix}$, $\boldsymbol y=(1,0)^\top$, $w=0$, $b=0$, $\lambda=0$. Entonces $\boldsymbol z=\mathbf 0$, $\boldsymbol p=(\tfrac12,\tfrac12)$ y $\boldsymbol p-\boldsymbol y=(-\tfrac12,\tfrac12)$:

```math
\frac{\partial J}{\partial w}=\tfrac12\big(1\cdot(-\tfrac12)+(-2)\cdot\tfrac12\big)=-0.75,\qquad
\frac{\partial J}{\partial b}=\tfrac12\big(-\tfrac12+\tfrac12\big)=0 .
```

El gradiente negativo en $w$ indica que aumentar $w$ reduce la pérdida. Tiene sentido: la muestra positiva tiene $x=1>0$ y la negativa $x=-2<0$.

**Caso particular verificable.** En $\boldsymbol\theta=\mathbf 0$ todas las $p_i=\tfrac12$, así que $\nabla_{\boldsymbol w}J=\frac1m X^\top(\tfrac12\mathbf 1-\boldsymbol y)$ y $J(\mathbf 0)=\log 2$. Ambas igualdades se comprueban en `test_gradient_zero_point_closed_form` y `test_loss_is_non_negative_and_equals_ln2_at_origin`.

**Código.**

```python
# app/math_engine/gradients.py — LogisticObjective.grad (modo "logits")
residual = self._activate(z) - self.y          # p - y
g_w = (self.X.T @ residual) / self.m
if self.l2 > 0:
    g_w = g_w + self.l2 * w                     # + λ w
g_b = np.sum(residual) / self.m
return np.concatenate([g_w, [g_b]])
```

## 4. La Hessiana

Derivando otra vez el gradiente (con $\partial p_i/\partial\boldsymbol\theta=p_i(1-p_i)\tilde{\boldsymbol x}_i$):

```math
\nabla^2J(\boldsymbol\theta)=\frac1m\sum_{i=1}^m p_i(1-p_i)\,\tilde{\boldsymbol x}_i\tilde{\boldsymbol x}_i^\top+\lambda D
=\frac1m\tilde X^\top S\tilde X+\lambda D,\qquad S=\operatorname{diag}\big(p_i(1-p_i)\big).
```

- **Semidefinida positiva:** $\boldsymbol v^\top\nabla^2J\,\boldsymbol v=\frac1m\sum_i p_i(1-p_i)(\tilde{\boldsymbol x}_i^\top\boldsymbol v)^2+\lambda\sum_{j\le d}v_j^2\ge0$. Por tanto, **$J$ es convexa**.
- **Acotada:** como $p_i(1-p_i)\le\frac14$ y $D\preceq I$, se cumple $\nabla^2J\preceq\frac1{4m}\tilde X^\top\tilde X+\lambda I\preceq L\,I$, con $L=\frac{1}{4m}\lambda_{\max}(\tilde X^\top\tilde X)+\lambda$.

El código de producción **no** calcula la Hessiana: el gradiente descendente no la necesita. Sí la calculan las pruebas para verificar la teoría (`test_hessian_is_psd_and_bounded_by_lipschitz_constant`, `test_lipschitz_bounds_hessian`) y para construir el método de Newton de referencia.

## 5. Expansiones de Taylor usadas

Para $f$ suficientemente suave:

```math
f(x+h)=f(x)+hf'(x)+\frac{h^2}{2}f''(x)+\frac{h^3}{6}f'''(x)+O(h^4).
```

- **Diferencia progresiva.** De la expansión se despeja $\dfrac{f(x+h)-f(x)}{h}=f'(x)+\dfrac h2f''(\xi)$: el error de truncamiento es $O(h)$.
- **Diferencia central.** Restando las expansiones en $x+h$ y $x-h$, los términos pares se cancelan: $\dfrac{f(x+h)-f(x-h)}{2h}=f'(x)+\dfrac{h^2}{6}f'''(\xi)$, con error $O(h^2)$.
- **Lema de descenso** (versión multivariable con resto de Lagrange): si $\nabla^2J\preceq LI$,

```math
J(\boldsymbol\theta+\boldsymbol\delta)\le J(\boldsymbol\theta)+\nabla J(\boldsymbol\theta)^\top\boldsymbol\delta+\frac L2\lVert\boldsymbol\delta\rVert^2 .
```

  Con $\boldsymbol\delta=-\alpha\nabla J$ se obtiene la garantía de descenso del [gradiente descendente](optimization_algorithms.md#4-convergencia).

## 6. Convexidad: definiciones equivalentes

Para $J$ dos veces diferenciable, son equivalentes:

1. $J(t\boldsymbol a+(1-t)\boldsymbol b)\le tJ(\boldsymbol a)+(1-t)J(\boldsymbol b)$ para todo $t\in[0,1]$ (definición). Se comprueba en `test_convexity_along_random_segments`.
2. $J(\boldsymbol b)\ge J(\boldsymbol a)+\nabla J(\boldsymbol a)^\top(\boldsymbol b-\boldsymbol a)$ (el plano tangente es una cota inferior).
3. $\nabla^2J(\boldsymbol\theta)\succeq0$ en todo punto.

**Consecuencia práctica.** Todo punto estacionario ($\nabla J=\mathbf 0$) es un mínimo global. Por eso el criterio de parada $\lVert\nabla J\rVert<\text{tol}$ tiene sentido como certificado de optimalidad aproximada.
