# Alcance y limitaciones

## Dentro del alcance

- Regresión logística binaria con regularización L2 sobre WDBC.
- Gradiente descendente de **paso fijo**, en lote completo.
- Tres formulaciones numéricas de la pérdida (estable, acotada e ingenua), para compararlas.
- Verificación del gradiente por diferencias finitas y estudio del paso $h$.
- API REST y una interfaz web de demostración y auditoría.

## Fuera del alcance (no implementado)

| Funcionalidad | Situación |
|---|---|
| Otros optimizadores (Newton, BFGS, momento, Adam, SGD) | no implementados. Newton solo existe dentro de una prueba, como referencia |
| Búsqueda de línea (Armijo/Wolfe) o paso adaptativo | no implementada |
| Selección de $\lambda$ o del umbral por validación cruzada | no implementada; $\lambda=10^{-3}$ y umbral $0.5$ fijos |
| Intervalos de confianza de las métricas | no calculados |
| Clasificación multiclase o modelos no lineales | fuera del alcance |
| Persistencia del modelo en disco | no existe: el modelo activo vive en memoria y se reentrena al arrancar la API |
| Autenticación, límites de uso o despliegue en producción | no implementados |
| Pruebas automatizadas del frontend | no existen; la CI solo comprueba tipos y la compilación |

## Limitaciones técnicas

- **Estado global en memoria.** `get_service()` mantiene un único `ModelService` por proceso. Promover un modelo con `set_active=true` afecta a todos los clientes de ese proceso, y con varios *workers* de Uvicorn cada uno tendría su propio modelo activo.
- **Coste de arranque.** Al iniciar, la API carga los datos, entrena el modelo por defecto (≈0.3 s) y precalcula en segundo plano escenarios, barridos y la auditoría flotante.
- **Caché acotada.** Los resultados deterministas se memorizan en una LRU de 64 entradas, cuya clave incluye la versión del modelo activo.
- **Contador de desbordamientos.** `exp_overflow_events` cuenta activaciones en $J$ y en $\nabla J$ por separado, así que un mismo elemento puede contarse dos veces por iteración.
- **Bandas de riesgo** (0.2 / 0.5 / 0.8): son una convención de presentación, no umbrales clínicos validados.

## Limitaciones de validez

- Una sola partición entrenamiento/prueba con 114 casos de prueba: cada error vale ≈0.9 puntos de exactitud.
- Los datos proceden de una sola institución y época (ver [dataset](../experiments/dataset.md)).
- **No es un dispositivo médico** ni sustituye el criterio clínico.

## Trabajo futuro

Propuestas, ninguna implementada:

1. Validación cruzada estratificada con intervalos de confianza para las métricas.
2. Selección de $\lambda$ y del umbral de decisión según el coste relativo de falsos negativos y falsos positivos.
3. Búsqueda de línea de Armijo y comparación con Newton, que ya existe en las pruebas, como optimizador de producción.
4. Paso adaptativo basado en una estimación local de la curvatura (E3 muestra que $2/L$ es unas 30 veces conservador).
5. Persistencia y versionado del modelo activo.
6. Pruebas automatizadas del frontend (Vitest/Playwright).
