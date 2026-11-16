# Documentación de WDBC Optimizer

Regresión logística como problema de optimización convexa: gradiente descendente, aritmética IEEE-754 estable y verificación del gradiente por diferencias finitas, implementados en NumPy.

## Ruta de lectura recomendada

1. [Visión general](introduction/overview.md) → [planteamiento del problema](introduction/problem_statement.md)
2. [Fundamentos matemáticos](theory/mathematical_foundations.md) (notación y mapa)
3. [Funciones de pérdida](theory/loss_functions.md) → [cálculo](theory/calculus.md) → [optimización](theory/optimization_algorithms.md)
4. [Derivación numérica](theory/numerical_differentiation.md) y [punto flotante](theory/floating_point.md)
5. [Resultados](experiments/results.md)

## Índice completo

### Introducción
- [Visión general](introduction/overview.md)
- [Planteamiento del problema](introduction/problem_statement.md)
- [Objetivos](introduction/objectives.md)
- [Alcance y limitaciones](introduction/scope_and_limitations.md) (incluye el trabajo futuro)

### Teoría
- [Fundamentos matemáticos y notación](theory/mathematical_foundations.md)
- [Álgebra lineal](theory/linear_algebra.md)
- [Cálculo multivariable](theory/calculus.md)
- [Probabilidad y estadística](theory/probability_and_statistics.md)
- [Funciones de pérdida](theory/loss_functions.md)
- [Algoritmos de optimización](theory/optimization_algorithms.md)
- [Derivación numérica y verificación del gradiente](theory/numerical_differentiation.md)
- [Aritmética de punto flotante](theory/floating_point.md)
- [Entrenamiento del modelo](theory/model_training.md)

### Arquitectura
- [Arquitectura del sistema](architecture/system_architecture.md)
- [Diseño de módulos y diagrama UML](architecture/module_design.md)
- [Flujo de datos](architecture/data_flow.md)
- [Decisiones de diseño](architecture/design_decisions.md)

### Implementación
- [Detalles de implementación](implementation/implementation_details.md)
- [Algoritmos (fichas técnicas y complejidad)](implementation/algorithms.md)
- [Estabilidad numérica](implementation/numerical_stability.md)
- [Referencia de código](implementation/code_reference.md)
- [API REST](implementation/api_reference.md)
- [Frontend](implementation/frontend.md)

### Experimentos
- [Conjunto de datos](experiments/dataset.md)
- [Metodología experimental](experiments/experimental_methodology.md)
- [Métricas de evaluación y tolerancias](experiments/evaluation_metrics.md)
- [Resultados](experiments/results.md)

### Otros
- [Estrategia de pruebas](testing.md)
- [Diagramas (fuentes Mermaid)](diagrams/README.md)
- [Bibliografía](references/bibliography.md)
- [Glosario](glossary.md)
