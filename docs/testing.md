# Estrategia de pruebas y validación

## Ejecución

```bash
cd backend
python -m pytest                 # las 72 pruebas
python -m pytest -m numerical    # solo validación matemática
python -m pytest -m integration  # datos → servicio → API
python -m pytest -ra             # con resumen de omitidas/fallidas
ruff check . ../scripts ../examples
```

`pytest.ini` convierte cualquier `RuntimeWarning` en **error**. Un desbordamiento o una división por cero en la ruta numérica estable hace fallar la prueba aunque el resultado parezca correcto.

## Inventario

| Archivo | Pruebas | Origen | Marcadores | Qué valida |
|---|---|---|---|---|
| `test_activations_losses.py` | 14 | original | unit, numerical | sigmoide estable frente a ingenua, ausencia de desbordamiento, simetría, forma; las tres BCE; saturación y `NaN` |
| `test_gradients_fd.py` | 10 | original | unit, numerical | gradiente empaquetado frente al separado; forma cerrada en $\boldsymbol\theta=\mathbf 0$; chequeo de gradiente en WDBC; detección de un gradiente erróneo; órdenes de las diferencias finitas; curva en V; $L$ acota la Hessiana |
| `test_gradient_descent.py` | 10 | original | unit, numerical | los cuatro motivos de parada; monotonía con $\alpha=1/L$; exactitud > 95 %; modo ingenuo frente al estable; `record_every`; configuración inválida |
| `test_service_api.py` | 6 | original | integration | AUC frente a fuerza bruta; servicio JSON-safe; rutas de predicción; escenarios y auditorías; *endpoints* FastAPI con `TestClient` |
| `test_architecture.py` | 2 | original | architecture | el núcleo no importa bibliotecas de ML; scikit-learn solo en el cargador y solo con 4 utilidades |
| `test_numerical_properties.py` | 19 | **nuevo** | numerical | $\sigma'=\sigma(1-\sigma)$; monotonía y acotación en rejilla extrema; saturación exacta en $z\approx36.7$; LSE frente a BCE de libro; $J\ge0$, $J(\mathbf 0)=\ln2$; dimensiones; chequeo de gradiente con $\lambda\in\{0,10^{-3},1\}$ y en modo `clipped`; central frente a progresiva; Hessiana semidefinida positiva y $\preceq LI$; convexidad en segmentos; lema de descenso; **GD frente a Newton**; reproducibilidad bit a bit; no mutación de $\boldsymbol\theta^{(0)}$; crecimiento de $\lVert\boldsymbol w\rVert$ sin regularización |
| `test_integration_pipeline.py` | 10 | **nuevo** | integration | tamaños, etiquetas y estratificación; estandarización sin fuga; métricas en un caso hecho a mano; límites de la AUC; extremos de la ROC; serialización de no finitos; submuestreo; métricas recalculadas de forma independiente; predicción de los 114 casos; errores de validación |
| `test_experiments_script.py` | 1 | **nuevo** | integration | el script de reproducción genera 7 JSON estrictos y E1 da exactamente 4183 iteraciones |
| **Total** | **72** | 42 originales + 30 nuevas | | |

Recuento por marcador: `unit` 34, `numerical` 53, `integration` 17, `architecture` 2. Un módulo puede tener más de un marcador.

## Validación matemática: en qué se basa cada prueba

Las pruebas nuevas evitan comparar el código consigo mismo. Cada una contrasta con una **referencia independiente**:

| Propiedad | Referencia independiente |
|---|---|
| Gradiente analítico | diferencias centrales (el código del gradiente no interviene en la aproximación) |
| Derivada de la sigmoide | diferencia central sobre `sigmoid` |
| Pérdida Log-Sum-Exp | fórmula de libro en el rango seguro |
| Optimalidad del GD | método de Newton escrito en la propia prueba, con Hessiana explícita |
| Distancia al óptimo | cotas teóricas de convexidad fuerte: $J-J^\star\le\lVert\nabla J\rVert^2/(2\mu)$ y $\lVert\boldsymbol\theta-\boldsymbol\theta^\star\rVert\le\lVert\nabla J\rVert/\mu$ |
| Constante $L$ | autovalores de la Hessiana exacta en puntos aleatorios |
| AUC | conteo por fuerza bruta de todos los pares |
| Métricas del servicio | recálculo con NumPy directo a partir de $\boldsymbol\theta$ |

### Prueba de gradiente por diferencias finitas

```math
\frac{\partial J}{\partial\theta_j}\approx\frac{J(\boldsymbol\theta+h\boldsymbol e_j)-J(\boldsymbol\theta-h\boldsymbol e_j)}{2h},\qquad h=10^{-5}.
```

**Tolerancia: error relativo $<10^{-7}$.** El error teórico es $\approx h^2\lvert J'''\rvert/6+\varepsilon_{\text{mach}}\lvert J\rvert/h\approx10^{-11}$, y el medido es $\sim10^{-11}$. El margen de cuatro órdenes absorbe diferencias entre plataformas y BLAS, y aun así detecta cualquier error real: un factor 2 da un error relativo de $\approx0.33$ y un signo cambiado, de $\approx1$. El resto de tolerancias y su justificación están en [métricas de evaluación](experiments/evaluation_metrics.md#3-tolerancias-usadas-en-las-pruebas).

**Distinción importante:** esta prueba **valida** el gradiente. El algoritmo de entrenamiento usa siempre el gradiente analítico.

## Integración continua

`.github/workflows/ci.yml` se ejecuta en cada *push* a `main` y en cada *pull request*:

| Trabajo | Pasos |
|---|---|
| **backend** (Python 3.10, 3.11, 3.12 y 3.13) | instalar `requirements.txt` + `requirements-dev.txt` → `pip check` → `ruff check` → `pytest -ra` → `examples/basic_usage.py` |
| **frontend** (Node 20) | `npm ci` → `npm run typecheck` → `npm run build` |

**Estado de la verificación:** en el entorno de preparación (Linux, Python 3.13.18, Node 22) pasaron las 72 pruebas, Ruff, el *typecheck* y la compilación. La primera ejecución en GitHub Actions ([run 43977248421](https://github.com/maiwxzs/wdbc-optimizer/actions/runs/43977248421)) pasó los cinco trabajos: Python 3.10, 3.11, 3.12 y 3.13, y frontend.

## Lo que no se prueba

- Componentes de React y flujos de extremo a extremo en el navegador.
- Rendimiento y tiempos: los tiempos no se afirman en ninguna prueba.
- Concurrencia real con varios clientes simultáneos.
