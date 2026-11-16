# Aritmética de punto flotante (IEEE-754)

Toda la aritmética del proyecto es `float64` (doble precisión IEEE-754 [IEEE, 2019]). Este documento reúne las constantes que determinan dónde fallan las formulaciones ingenuas y por qué las estables no fallan. Referencia introductoria: [Goldberg, 1991]; análisis de errores: [Higham, 2002].

## 1. Constantes de `float64`

| Constante | Valor | Significado | En el código |
|---|---|---|---|
| $\varepsilon_{\text{mach}}=2^{-52}$ | $2.220446\times10^{-16}$ | distancia de 1 al siguiente número representable | `EPS_MACHINE` |
| $\text{DBL\_MAX}$ | $1.797693\times10^{308}$ | mayor número finito | `np.finfo(np.float64).max` |
| $\ln(\text{DBL\_MAX})$ | $709.7827$ | $e^{x}=+\infty$ para $x$ mayor que este valor | `EXP_OVERFLOW_THRESHOLD` |
| menor normal | $2.225\times10^{-308}$ | por debajo empiezan los subnormales | `float_tiny` |
| $\ln(\text{menor subnormal})$ | $-744.44$ | $e^{x}=0$ exacto para $x$ menor que este valor | `underflow_threshold` |
| $-\ln(\varepsilon_{\text{mach}}/2)$ | $36.7368$ | $1+e^{-z}=1$ exacto para $z$ mayor que este valor | `sigmoid_saturation_z` |

**Origen de $36.74$.** Con redondeo al más cercano, $1+\delta$ se redondea a $1$ si $\delta<\varepsilon_{\text{mach}}/2$ (en el empate exacto, a 1 por "redondeo a par"). Con $\delta=e^{-z}$, la condición es $z>-\ln(\varepsilon_{\text{mach}}/2)$. `test_naive_sigmoid_saturates_to_exactly_one` lo comprueba.

## 2. Tres fenómenos que rompen las fórmulas de libro

1. **Desbordamiento (*overflow*).** $1/(1+e^{-z})$ con $z<-709.78$ evalúa $e^{-z}=+\infty$. El resultado final es $0$, que es correcto en el límite, pero NumPy emite un `RuntimeWarning`, y $\log(0)=-\infty$ aparece en la pérdida.
2. **Saturación.** Para $z>36.74$, $\sigma_{\text{ingenua}}(z)=1.0$ **exacto**, sin ningún desbordamiento. Entonces $\log(1-p)=-\infty$, y si $y=1$, $(1-y)\log(1-p)=0\cdot(-\infty)=\texttt{NaN}$.
3. **Cancelación catastrófica.** Al restar números casi iguales se pierden cifras significativas. Afecta a $1-p$ cuando $p\approx1$ y a las diferencias finitas con $h$ pequeño (ver [derivación numérica](numerical_differentiation.md#3-error-de-redondeo-y-paso-óptimo-la-curva-en-v)).

## 3. Sigmoide estable por tramos

**Formulación.** Se elige, según el signo de $z$, la expresión cuyo argumento de `exp` es no positivo:

```math
\sigma(z)=\begin{cases}\dfrac{1}{1+e^{-z}}, & z\ge0,\\[2ex]\dfrac{e^{z}}{1+e^{z}}, & z<0.\end{cases}
```

Las dos ramas son algebraicamente idénticas: basta multiplicar numerador y denominador por $e^{z}$.

**Implementación con una sola exponencial.** Con $e=\exp(-\lvert z\rvert)\in(0,1]$, ambas ramas se escriben como $1/(1+e)$ y $e/(1+e)$:

```python
e = np.exp(-np.abs(flat))          # siempre en (0, 1]: nunca desborda
pos = flat >= 0
out[pos] = 1.0 / (1.0 + e[pos])
out[~pos] = e[~pos] / (1.0 + e[~pos])
```

No se usa `np.where(z >= 0, 1/(1+exp(-z)), exp(z)/(1+exp(z)))`, porque `np.where` **evalúa ambas ramas en todo el arreglo** antes de seleccionar, así que seguiría desbordando y emitiendo avisos.

**Verificación.** `pytest.ini` convierte todo `RuntimeWarning` en error. `test_sigmoid_never_overflows` evalúa $z\in\{-10^6,-1000,-745,-710,0,710,10^6\}$ sin un solo aviso.

**Límite inevitable.** Para $z>36.74$, también la sigmoide estable devuelve $1.0$ exacto, porque es el `float64` más cercano al valor real. No es un error de la implementación: por eso la **pérdida** no debe calcularse a partir de $p$, sino directamente del logit (Log-Sum-Exp).

## 4. Tabla de valores especiales (experimento E6)

Valores reales producidos por `ModelService.float_audit()` (pérdida para $y=1$: $-\log\sigma(z)$):

| $z$ | $\sigma$ ingenua | $\sigma$ estable | ¿desborda $e^{-z}$? | pérdida ingenua | pérdida *clip* | pérdida exacta (LSE) |
|---|---|---|---|---|---|---|
| $-1000$ | $0$ | $0$ | sí | $+\infty$ | $34.54$ | $1000$ |
| $-710$ | $0$ | $4.48\times10^{-309}$ | sí | $+\infty$ | $34.54$ | $710$ |
| $-40$ | $4.25\times10^{-18}$ | $4.25\times10^{-18}$ | no | $40$ | **$34.54$** | $40$ |
| $0$ | $0.5$ | $0.5$ | no | $0.6931$ | $0.6931$ | $0.6931$ |
| $20$ | $1-2.06\times10^{-9}$ | $1-2.06\times10^{-9}$ | no | $2.0612\times10^{-9}$ | $2.0612\times10^{-9}$ | $2.0612\times10^{-9}$ |
| $37.5$ | $1.0$ | $1.0$ | no | $0$ | $9.99\times10^{-16}$ | $5.18\times10^{-17}$ |
| $40$ | $1.0$ | $1.0$ | no | $0$ | $9.99\times10^{-16}$ | $4.25\times10^{-18}$ |

Para $y=0$ y $z\ge37.5$, la forma ingenua da $-\log(1-1)=+\infty$, mientras que la exacta da $z$.

**Observación fina** (en $z=20$, $y=0$): la forma ingenua da $19.99999997$, frente al valor exacto $20.0000000021$. Es la cancelación en $1-p$ con $p\approx1$: se pierden unas 8 cifras significativas **antes** de llegar a la saturación.

## 5. JSON y valores no finitos

El estándar JSON no admite `NaN` ni `Infinity`. `app/services/serialization.py::to_jsonable` convierte cualquier valor no finito a `null` en las series. La tabla IEEE de la auditoría usa además las cadenas `"NaN"`, `"+inf"` y `"-inf"` para mostrarlos de forma explícita. Las pruebas usan `json.dumps(..., allow_nan=False)` para asegurar que **ninguna** respuesta, ni siquiera la de un entrenamiento divergente, contiene literales no estándar.

## Referencias

[IEEE, 2019], [Goldberg, 1991], [Higham, 2002]: ver [bibliografía](../references/bibliography.md).
