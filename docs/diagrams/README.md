# Diagramas

Fuentes Mermaid (`.mmd`) de los diagramas técnicos. Todos reflejan el código real y se validaron con el analizador oficial de Mermaid 11. GitHub los muestra dibujados en los documentos donde están incrustados.

| Archivo | Tipo | Representa | Documento que lo explica |
|---|---|---|---|
| `system_architecture.mmd` | componentes | frontend, capas del backend, herramientas fuera de línea | [arquitectura del sistema](../architecture/system_architecture.md) |
| `execution_flow.mmd` | secuencia | arranque de la API y atención de `POST /api/train` | [flujo de datos](../architecture/data_flow.md) |
| `training_pipeline.mmd` | flujo | bucle del gradiente descendente y criterios de parada | [flujo de datos](../architecture/data_flow.md#2-proceso-de-entrenamiento) |
| `class_diagram.mmd` | UML de clases | `LogisticObjective`, `GDConfig`, `GDResult`, `ModelService`, … | [diseño de módulos](../architecture/module_design.md#diagrama-uml-de-clases) |
| `module_dependencies.mmd` | dependencias | *imports* entre módulos y bibliotecas externas | [diseño de módulos](../architecture/module_design.md#dependencias) |

Para exportarlos a SVG/PNG: `npx @mermaid-js/mermaid-cli -i system_architecture.mmd -o system_architecture.svg`.
