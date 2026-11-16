# Probabilidad y estadística

## 1. Modelo probabilístico de la regresión logística

**Problema.** Modelar una variable binaria $Y\in\{0,1\}$ (maligno/benigno) en función de un vector de características $\boldsymbol x$.

**Definición formal.** La regresión logística supone que, condicionado a $\boldsymbol x$, $Y$ sigue una distribución de Bernoulli cuyo parámetro es la sigmoide de una función lineal [Cox, 1958]:

```math
Y\mid\boldsymbol x\sim\operatorname{Bernoulli}(p(\boldsymbol x)),\qquad p(\boldsymbol x)=P(Y=1\mid\boldsymbol x)=\sigma(\boldsymbol w^\top\boldsymbol x+b).
```

**Interpretación de los logits.** Despejando: $\log\dfrac{p}{1-p}=\boldsymbol w^\top\boldsymbol x+b$. El logit es el **logaritmo de las probabilidades a favor** (*log-odds*) de malignidad, y cada peso $w_j$ es el cambio en esos log-odds cuando la característica estandarizada $x_j$ aumenta una desviación típica, con las demás fijas.

**Convención de etiquetas del proyecto.** scikit-learn codifica `target = 0` como maligno. El cargador la invierte (`y = 1 - target`), de modo que **$y=1\Leftrightarrow$ maligno** y $\sigma(z)$ se lee directamente como $P(\text{maligno}\mid\boldsymbol x)$.

## 2. Máxima verosimilitud y entropía cruzada

**Verosimilitud** de $m$ observaciones independientes:

```math
\mathcal L(\boldsymbol\theta)=\prod_{i=1}^m p_i^{\,y_i}(1-p_i)^{1-y_i}.
```

**Log-verosimilitud negativa promediada:**

```math
-\frac1m\log\mathcal L(\boldsymbol\theta)=-\frac1m\sum_{i=1}^m\Big[y_i\log p_i+(1-y_i)\log(1-p_i)\Big]=\text{BCE}(\boldsymbol\theta).
```

Minimizar la **entropía cruzada binaria (BCE)** equivale, por tanto, a estimar $\boldsymbol\theta$ por máxima verosimilitud. Las tres formulaciones numéricas de la BCE están en [funciones de pérdida](loss_functions.md).

## 3. Regularización L2 como información a priori

Añadir $\frac\lambda2\lVert\boldsymbol w\rVert^2$ equivale a una estimación de **máximo a posteriori** con prior gaussiano sobre los pesos. Multiplicando $J$ por $m$:

```math
m\,J(\boldsymbol\theta)=-\log\mathcal L(\boldsymbol\theta)+\frac{m\lambda}{2}\lVert\boldsymbol w\rVert^2
=-\log\mathcal L(\boldsymbol\theta)-\log\mathcal N\!\Big(\boldsymbol w;\mathbf 0,\tfrac{1}{m\lambda}I\Big)+\text{cte}.
```

Con $m=455$ y $\lambda=10^{-3}$, la varianza a priori es $1/(m\lambda)\approx2.2$, es decir, una desviación típica de $\approx1.48$ por peso: una restricción suave.

## 4. Existencia del estimador: el problema de la separación

**Resultado clásico** [Albert y Anderson, 1984]: si las clases son **linealmente separables** en el conjunto de entrenamiento, el estimador de máxima verosimilitud **no existe**. Basta tomar un hiperplano separador $\boldsymbol w_s$ y escalarlo: $J(c\,\boldsymbol w_s)\to0$ cuando $c\to\infty$, de modo que el ínfimo $0$ no se alcanza en ningún $\boldsymbol\theta$ finito.

**Evidencia en este proyecto** (experimento E4, ver [resultados](../experiments/results.md#e4--efecto-de-la-regularización)): con $\lambda=0$ y $\alpha=5$, tras 4000 iteraciones,

- $\lVert\boldsymbol w\rVert_2$ sigue creciendo ($22.7$ y en aumento) y $J=0.0195$ sigue bajando;
- la exactitud en prueba es $94.7\,\%$, frente al $97.4\,\%$ con $\lambda=10^{-3}$, donde $\lVert\boldsymbol w\rVert$ se estabiliza en $4.93$.

El comportamiento es **consistente con una separación (cuasi)completa** del conjunto de entrenamiento. No se verificó la separabilidad exacta, que requeriría resolver un programa lineal. Con $\lambda>0$ el objetivo es coercitivo en $\boldsymbol w$ y el minimizador existe.

## 5. Métricas de evaluación

Sea $\hat y_i=\mathbb 1[p_i\ge\tau]$ con umbral $\tau=0.5$, y sean $TP, TN, FP, FN$ las entradas de la matriz de confusión (positivo = maligno).

| Métrica | Fórmula | Pregunta que responde |
|---|---|---|
| Exactitud | $\frac{TP+TN}{TP+TN+FP+FN}$ | ¿Qué fracción se clasifica bien? |
| Precisión | $\frac{TP}{TP+FP}$ | De los que el modelo llama malignos, ¿cuántos lo son? |
| Sensibilidad (*recall*) | $\frac{TP}{TP+FN}$ | De los malignos, ¿cuántos detecta? |
| Especificidad | $\frac{TN}{TN+FP}$ | De los benignos, ¿cuántos descarta correctamente? |
| F1 | $\frac{2\cdot\text{prec}\cdot\text{sens}}{\text{prec}+\text{sens}}$ | Media armónica de precisión y sensibilidad |

En diagnóstico oncológico un **falso negativo** (maligno clasificado como benigno) suele ser más grave que un falso positivo. Por eso la interfaz muestra sensibilidad y especificidad por separado y no solo la exactitud. El umbral $\tau$ está fijo en $0.5$; elegirlo según costes clínicos queda fuera del alcance (ver [limitaciones](../introduction/scope_and_limitations.md)).

### 5.1 AUC mediante el estadístico U de Mann-Whitney

**Definición.** El área bajo la curva ROC es la probabilidad de que un positivo elegido al azar reciba una puntuación mayor que un negativo elegido al azar [Hanley y McNeil, 1982]:

```math
\text{AUC}=P(s^+>s^-)+\tfrac12P(s^+=s^-)=\frac{U}{n_+n_-},
\qquad U=\sum_{i:\,y_i=1}\operatorname{rango}(s_i)-\frac{n_+(n_++1)}{2},
```

donde los rangos se asignan en orden creciente de puntuación y los **empates reciben el rango promedio** [Mann y Whitney, 1947].

**Ejemplo.** Con puntuaciones $(0.1,0.4,0.35,0.8)$ y etiquetas $(0,0,1,1)$, los rangos son $(1,3,2,4)$. Entonces $U=(2+4)-\frac{2\cdot3}{2}=3$ y $\text{AUC}=3/(2\cdot2)=0.75$: de los 4 pares (positivo, negativo), 3 están bien ordenados.

**Código.** `app/evaluation/metrics.py::roc_auc` ordena con `mergesort` (estable) y recorre los bloques de empates. Complejidad $O(n\log n)$. `test_auc_matches_bruteforce` lo contrasta con el cálculo por fuerza bruta sobre todos los pares, con empates incluidos.

### 5.2 Curva ROC

`roc_curve` evalúa la tasa de verdaderos positivos (sensibilidad) frente a la de falsos positivos ($1-$especificidad) en 101 umbrales equiespaciados de $1$ a $0$. Es una aproximación en rejilla, adecuada para dibujar. El **valor numérico** de la AUC no se obtiene integrando esta curva, sino con la fórmula exacta de Mann-Whitney.

## 6. Partición de los datos

- **Estratificada 80/20** con semilla 42 (`train_test_split(..., stratify=y)`): 455 casos de entrenamiento y 114 de prueba, con la misma proporción de malignos ($\approx37.3\,\%$) en ambos.
- **Sin fuga de información:** el escalador y la PCA se ajustan **solo** con entrenamiento. La media del conjunto de prueba estandarizado no es exactamente cero, y la prueba `test_standardization_uses_train_statistics_only` lo comprueba.
- **Limitación estadística:** con 114 casos de prueba, cada error mueve la exactitud $\approx0.9$ puntos porcentuales. No se calcularon intervalos de confianza ni validación cruzada (ver [trabajo futuro](../introduction/scope_and_limitations.md#trabajo-futuro)).

## Referencias

[Cox, 1958], [Albert y Anderson, 1984], [Hanley y McNeil, 1982], [Mann y Whitney, 1947]: ver [bibliografía](../references/bibliography.md).
