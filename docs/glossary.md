# Glosario

| Término | Definición | Ver |
|---|---|---|
| **α (tasa de aprendizaje)** | tamaño del paso del gradiente descendente | [optimización](theory/optimization_algorithms.md) |
| **AUC** | área bajo la curva ROC; probabilidad de que un positivo al azar puntúe más alto que un negativo al azar | [probabilidad](theory/probability_and_statistics.md#51-auc-mediante-el-estadístico-u-de-mann-whitney) |
| **BCE** | *binary cross-entropy*, entropía cruzada binaria; log-verosimilitud negativa media del modelo de Bernoulli | [pérdidas](theory/loss_functions.md) |
| **Cancelación catastrófica** | pérdida de cifras significativas al restar dos números casi iguales en punto flotante | [punto flotante](theory/floating_point.md) |
| **Chequeo de gradiente** | comparación del gradiente analítico con uno aproximado por diferencias finitas | [derivación numérica](theory/numerical_differentiation.md#4-chequeo-de-gradiente) |
| **Clip** | restringir un valor a un intervalo; aquí, $p\in[\varepsilon,1-\varepsilon]$ | [pérdidas](theory/loss_functions.md#3-formulación-acotada-clip) |
| **Constante de Lipschitz / suavidad ($L$)** | cota superior de la curvatura: $\nabla^2J\preceq LI$ | [optimización](theory/optimization_algorithms.md#3-constante-de-suavidad-l) |
| **Convexidad fuerte ($\mu$)** | $\nabla^2J\succeq\mu I$ con $\mu>0$; garantiza un minimizador único y convergencia lineal | ídem |
| **Curva en V** | error de las diferencias finitas frente a $h$ en escala log-log | [derivación numérica](theory/numerical_differentiation.md#3-error-de-redondeo-y-paso-óptimo-la-curva-en-v) |
| **Desbordamiento (*overflow*)** | resultado mayor que $\text{DBL\_MAX}$, que se convierte en $+\infty$ | [punto flotante](theory/floating_point.md) |
| **Diferencia central / progresiva** | aproximaciones de la derivada de orden $O(h^2)$ y $O(h)$ | [derivación numérica](theory/numerical_differentiation.md) |
| **Divergencia** | estado en que $J$ no es finito o supera $10^8$; motivo de parada `divergence` | [optimización](theory/optimization_algorithms.md#5-criterios-de-parada) |
| **$\varepsilon_{\text{mach}}$** | épsilon de máquina: $2^{-52}\approx2.22\times10^{-16}$ en `float64` | [punto flotante](theory/floating_point.md) |
| **Especificidad** | fracción de benignos clasificados correctamente: $TN/(TN+FP)$ | [probabilidad](theory/probability_and_statistics.md#5-métricas-de-evaluación) |
| **Estandarización** | transformar cada columna a media 0 y desviación 1 con estadísticos de entrenamiento | [álgebra lineal](theory/linear_algebra.md#5-estandarización-transformación-afín-por-columnas) |
| **Fuga de datos** | usar información del conjunto de prueba al ajustar el modelo o el preprocesamiento | [dataset](experiments/dataset.md) |
| **Gradiente descendente (GD)** | método iterativo $\boldsymbol\theta\leftarrow\boldsymbol\theta-\alpha\nabla J$ | [optimización](theory/optimization_algorithms.md) |
| **Hessiana** | matriz de segundas derivadas $\nabla^2J$ | [cálculo](theory/calculus.md#4-la-hessiana) |
| **IEEE-754** | estándar de aritmética de punto flotante | [punto flotante](theory/floating_point.md) |
| **λ (regularización L2)** | peso de la penalización $\frac\lambda2\lVert\boldsymbol w\rVert^2$ | [pérdidas](theory/loss_functions.md#5-término-de-regularización) |
| **Logit ($z$)** | $\boldsymbol w^\top\boldsymbol x+b$; logaritmo de las probabilidades a favor de malignidad | [probabilidad](theory/probability_and_statistics.md) |
| **Log-Sum-Exp** | evaluación estable de $\log(1+e^{z})$ mediante `np.logaddexp` | [pérdidas](theory/loss_functions.md#2-formulación-log-sum-exp-la-que-usa-el-motor) |
| **Modelo activo** | $\boldsymbol\theta$ que usa la API para predecir; se reemplaza con `set_active=true` | [flujo de datos](architecture/data_flow.md#4-estado) |
| **Número de condición ($\kappa$)** | $\lambda_{\max}/\lambda_{\min}$ de la Hessiana; controla la velocidad del GD | [resultados E3](experiments/results.md#e3--barrido-de-la-tasa-de-aprendizaje) |
| **PCA** | análisis de componentes principales; aquí solo se usa para visualizar en 2D | [álgebra lineal](theory/linear_algebra.md#6-proyección-pca-solo-visualización) |
| **Saturación** | $\sigma(z)$ redondea a $1.0$ exacto para $z>36.74$ | [punto flotante](theory/floating_point.md) |
| **Sensibilidad (*recall*)** | fracción de malignos detectados: $TP/(TP+FN)$ | [probabilidad](theory/probability_and_statistics.md#5-métricas-de-evaluación) |
| **Separación** | existencia de un hiperplano que separa las clases; impide que exista el estimador de máxima verosimilitud | [probabilidad](theory/probability_and_statistics.md#4-existencia-del-estimador-el-problema-de-la-separación) |
| **Sigmoide ($\sigma$)** | $1/(1+e^{-z})$; transforma el logit en probabilidad | [cálculo](theory/calculus.md#1-la-función-sigmoide-y-su-derivada) |
| **θ (theta)** | vector empaquetado de parámetros $[\boldsymbol w;b]\in\mathbb R^{31}$ | [fundamentos](theory/mathematical_foundations.md) |
| **WDBC** | *Wisconsin Diagnostic Breast Cancer*, el conjunto de datos | [dataset](experiments/dataset.md) |
