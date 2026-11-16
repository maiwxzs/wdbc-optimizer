# Planteamiento del problema

## Contexto

El diagnóstico citológico de masas mamarias a partir de aspirados con aguja fina puede apoyarse en medidas cuantitativas de los núcleos celulares (tamaño, forma, textura). El conjunto WDBC reúne 30 de esas medidas para 569 casos con diagnóstico confirmado ([dataset](../experiments/dataset.md)).

## Problema técnico

Dado un paciente con vector de características $\boldsymbol x\in\mathbb R^{30}$, estimar $P(\text{maligno}\mid\boldsymbol x)$ con un modelo **interpretable**, y obtener sus parámetros resolviendo numéricamente

```math
\min_{\boldsymbol w\in\mathbb R^{30},\,b\in\mathbb R}\;\frac1m\sum_{i=1}^m\Big[\log\big(1+e^{\boldsymbol w^\top\boldsymbol x_i+b}\big)-y_i(\boldsymbol w^\top\boldsymbol x_i+b)\Big]+\frac\lambda2\lVert\boldsymbol w\rVert^2 .
```

## Problema académico

En un curso de Métodos Numéricos, el interés no está solo en que el clasificador funcione, sino en **poder justificar cada número**:

1. **¿Por qué converge el método?** Para eso hay que demostrar la convexidad, calcular $L$ y relacionar $\alpha$ con la convergencia.
2. **¿Cómo sabemos que el gradiente está bien?** Comparándolo con diferencias finitas, cuya precisión a su vez depende de $h$ por el equilibrio entre truncamiento y redondeo.
3. **¿Qué pasa cuando la aritmética finita rompe las fórmulas?** Hay desbordamiento de `exp`, saturación de $\sigma$, $0\cdot\log0=\texttt{NaN}$ y cancelación catastrófica.
4. **¿Cómo se detecta y se comunica un fallo numérico** sin que el sistema se caiga?

Las bibliotecas de *machine learning* resuelven el problema técnico, pero ocultan el académico. Por eso el proyecto implementa el núcleo desde cero y expone su comportamiento en una interfaz de auditoría.

## Restricciones

- El núcleo numérico solo puede usar NumPy (lo verifica una prueba automática).
- Precisión `float64` en todo el cálculo.
- El sistema nunca debe fallar por inestabilidad numérica: los estados no finitos se detectan y se informan.
- Las respuestas de la API deben ser JSON estándar, sin `NaN` ni `Infinity`.
