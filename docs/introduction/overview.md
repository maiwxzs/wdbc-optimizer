# Visión general

**WDBC Optimizer** es una plataforma web *full-stack* académica que resuelve la regresión logística sobre el conjunto *Wisconsin Diagnostic Breast Cancer* como un **problema de optimización numérica**, implementado desde cero con NumPy. La acompaña una interfaz que permite **auditar** el optimizador: ver cómo converge o diverge, verificar el gradiente y observar los límites de la aritmética IEEE-754.

Proyecto del curso **Métodos Numéricos** de la Universidad de los Llanos.

## Componentes

| Componente | Tecnología | Función |
|---|---|---|
| Motor numérico | Python 3.10+, NumPy | sigmoide estable, pérdida Log-Sum-Exp, gradiente analítico, diferencias finitas, constante $L$ |
| Optimizador | NumPy | gradiente descendente de paso fijo con telemetría y cuatro criterios de parada |
| Evaluación | NumPy | matriz de confusión, sensibilidad, especificidad, F1, AUC (Mann-Whitney), ROC |
| Datos | scikit-learn (solo carga, partición, escalado y PCA) | WDBC, partición estratificada 80/20 |
| API REST | FastAPI, Pydantic v2, Uvicorn | 10 endpoints JSON; cálculo en *threadpool* |
| Interfaz | React 18, TypeScript, Vite, Tailwind, ECharts, Framer Motion, KaTeX | tres pestañas: Clínico, Telemetría, Auditoría |
| Experimentos | `scripts/reproduce_experiments.py`, matplotlib | métricas JSON y figuras reproducibles |
| Calidad | pytest (72 pruebas), Ruff, GitHub Actions | validación matemática e integración continua |

## Qué hace "desde cero" y qué no

**Implementado explícitamente en el código del proyecto:**

- la función sigmoide (estable e ingenua) y $\log\sigma$;
- la entropía cruzada en tres formulaciones y la penalización L2;
- el gradiente analítico vectorizado y la constante de suavidad $L$;
- el gradiente descendente con sus criterios de parada y diagnóstico de divergencia;
- las diferencias finitas central y progresiva, el chequeo de gradiente y el barrido de $h$ con ajuste del modelo de error;
- todas las métricas de clasificación, incluida la AUC.

**Delegado en bibliotecas** (no forma parte del aprendizaje del modelo):

- `scikit-learn`: carga del conjunto, partición estratificada, `StandardScaler` y PCA 2D para visualización. La prueba `tests/test_architecture.py` lo **hace cumplir**: falla si el núcleo numérico importa `sklearn`, `scipy`, `torch`, `tensorflow` o `jax`, o si se usa `sklearn` fuera del cargador o con otras funciones.
- NumPy: álgebra lineal básica (productos, normas, `eigvalsh`).

## Mapa de la documentación

- **Introducción:** [problema](problem_statement.md), [objetivos](objectives.md), [alcance y limitaciones](scope_and_limitations.md)
- **Teoría:** [fundamentos](../theory/mathematical_foundations.md) y sus documentos enlazados
- **Arquitectura:** [sistema](../architecture/system_architecture.md), [módulos](../architecture/module_design.md), [flujo de datos](../architecture/data_flow.md), [decisiones](../architecture/design_decisions.md)
- **Implementación:** [detalles](../implementation/implementation_details.md), [algoritmos](../implementation/algorithms.md), [estabilidad numérica](../implementation/numerical_stability.md), [referencia de código](../implementation/code_reference.md), [API REST](../implementation/api_reference.md), [frontend](../implementation/frontend.md)
- **Experimentos:** [dataset](../experiments/dataset.md), [metodología](../experiments/experimental_methodology.md), [métricas](../experiments/evaluation_metrics.md), [resultados](../experiments/results.md)
- **Pruebas:** [estrategia de pruebas](../testing.md)
