# Estado del proyecto

Informe de la reorganización del repositorio para su publicación (16 de noviembre de 2026).

## 1. Componentes que existían originalmente

| Componente | Estado original |
|---|---|
| Motor numérico `backend/app/math_engine/` | completo y funcional: sigmoide estable e ingenua, tres BCE, gradiente analítico, `LogisticObjective`, $L$, diferencias finitas, chequeo de gradiente, barrido de $h$ |
| Optimizador `backend/app/optimization/` | gradiente descendente con telemetría y cuatro criterios de parada |
| Métricas `backend/app/evaluation/` | confusión, exactitud, precisión, sensibilidad, especificidad, F1, AUC (Mann-Whitney), ROC |
| Datos `backend/app/data/` | WDBC con scikit-learn (carga, partición, escalado, PCA) |
| Servicio y API | `ModelService` con caché y modelo activo; FastAPI con 10 *endpoints* |
| Pruebas | 42 en 5 archivos, todas pasando; `RuntimeWarning` como error |
| Frontend `frontend/` | React + TypeScript + Vite, tres pestañas; compila y pasa el *typecheck* |
| Documentación | `README.md` de una sola página (arquitectura, especificación numérica, resultados, API) |
| Otros | `.gitignore` mínimo; presentación PDF de la sustentación en la raíz |

**No existían:** licencia, CI, `pyproject.toml`, configuración de *linter*, documentación en `docs/`, diagramas, script de experimentos, ejemplos ni archivos de comunidad.

## 2. Reorganización y cambios en el código

- **Estructura conservada.** No se movió ningún archivo: `backend/` + `frontend/` se mantienen (justificación en [DD-11](docs/architecture/design_decisions.md#dd-11--se-conserva-la-estructura-backend--frontend-en-lugar-de-src-reorganización)). Las pruebas se clasificaron con marcadores en lugar de moverlas a subcarpetas ([DD-12](docs/architecture/design_decisions.md#dd-12--pruebas-en-una-sola-carpeta-clasificadas-con-marcadores-reorganización)).
- **Código de la aplicación:** solo correcciones de estilo detectadas por Ruff, sin efecto funcional (detalle en [implementation_details.md §6](docs/implementation/implementation_details.md#6-cambios-realizados-durante-la-reorganización)). Las cifras que documentaba el README original (4183 iteraciones, 97.4 %, AUC 0.993, $L=3.3455$, error del gradiente $1.2\times10^{-11}$…) se reprodujeron **idénticas** después de los cambios.
- **`backend/pytest.ini`:** se registraron los marcadores; se mantiene `error::RuntimeWarning`.
- **`backend/tests/conftest.py`:** asignación de marcadores por módulo y eliminación de `noqa` innecesarios.
- **`README.md`:** reescrito; conserva toda la información útil del original (puesta en marcha, modo presentación, hallazgos y equipo).
- **`.gitignore`:** ampliado.

## 3. Archivos creados

| Tipo | Archivos |
|---|---|
| Documentación | `docs/index.md`; `docs/introduction/` (4); `docs/theory/` (9); `docs/architecture/` (4); `docs/implementation/` (6); `docs/experiments/` (4); `docs/testing.md`, `docs/glossary.md`, `docs/references/bibliography.md` |
| Diagramas | `docs/diagrams/`: arquitectura, flujo de ejecución, proceso de entrenamiento, clases UML y dependencias (`.mmd`) + `README.md`. Todos validados con el analizador de Mermaid 11 |
| Pruebas | `test_numerical_properties.py` (19), `test_integration_pipeline.py` (10), `test_experiments_script.py` (1) |
| Experimentos | `scripts/reproduce_experiments.py`; `results/metrics/*.json` (7); `results/figures/*.png` (7); `results/README.md` |
| Ejemplos | `examples/basic_usage.py` |
| Configuración | `backend/pyproject.toml`, `backend/requirements-dev.txt`, `ruff.toml`, `.editorconfig`, `.gitattributes`, `.github/workflows/ci.yml` |
| Comunidad y legal | `LICENSE` (MIT), `CITATION.cff` (validado con `cffconvert`), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`, este archivo |

## 4. Verificaciones ejecutadas

Entorno: Linux x86-64, Python 3.13.18, NumPy 2.5.3, scikit-learn 1.9.1, Node 22.

| Verificación | Resultado |
|---|---|
| `pytest` (entorno de trabajo) | **72 pasaron**, 0 fallaron |
| `pytest` en un *virtualenv* limpio con `requirements.txt` + `requirements-dev.txt` (simulación del trabajo de CI) | **72 pasaron**; 1 aviso de deprecación externo (Starlette recomienda `httpx2` en `TestClient`) |
| `pip check` | sin conflictos |
| `ruff check . ../scripts ../examples` | sin errores |
| `python examples/basic_usage.py` | correcto |
| `python scripts/reproduce_experiments.py` | 7 JSON + 7 figuras en ≈9 s |
| `npm ci && npm run typecheck && npm run build` | correcto (aviso de Vite: el *chunk* de ECharts supera 500 kB) |
| Enlaces internos y anclas de los 39 archivos Markdown | 0 rotos (comprobación automática) |
| Sintaxis de los 8 diagramas Mermaid | válida |

**GitHub Actions:** la primera ejecución tras la publicación ([run 43977248421](https://github.com/maiwxzs/wdbc-optimizer/actions/runs/43977248421), *commit* `abb5965`) terminó con éxito en los cinco trabajos: backend con Python 3.10, 3.11, 3.12 y 3.13, y frontend.

**No verificado:** el renderizado de las fórmulas LaTeX en GitHub no se comprobó visualmente.

## 5. Problemas y observaciones encontrados en la auditoría

Ninguno afecta a la corrección de los resultados. Todos están documentados.

1. **Falso diagnóstico de divergencia en modo `naive`.** Con $\alpha=1$, el modo ingenuo se detiene en $k=44$ por $J=\texttt{NaN}$, aunque $\boldsymbol\theta$ es finito y clasifica el 98.2 % de la prueba. Es el comportamiento esperado (no se puede certificar nada con $J$ no finito), pero conviene saberlo al interpretar la pestaña de Telemetría. → [results.md E2](docs/experiments/results.md#e2--escenarios-comparativos)
2. **`exp_overflow_events` cuenta dos veces por iteración** (en `loss` y en `grad`). → [evaluation_metrics.md](docs/experiments/evaluation_metrics.md)
3. **`loss_and_grad` calcula los logits dos veces.** Es un coste extra de $O(md)$ por iteración. → [algorithms.md A3](docs/implementation/algorithms.md#a3--objetivo-y-gradiente-analítico)
4. **Parámetro sin uso:** `ModelService._audit_point(point, seed, l2)` recibe `l2` pero no lo usa. No se eliminó para no cambiar firmas.
5. **La cota $2/L$ es unas 30 veces conservadora.** El umbral real lo marca $2/\lambda_{\max}(\nabla^2J(\boldsymbol\theta^\star))=17.67$, lo que coincide con la transición observada. Es un hallazgo nuevo de E3, no un error.
6. **Aviso de deprecación de Starlette** sobre `httpx` en `TestClient` (dependencia externa; no afecta a las pruebas).

## 6. Publicación y pendientes del propietario

**Publicado en** https://github.com/maiwxzs/wdbc-optimizer. El titular del copyright y el mantenedor es **maiwxzs**. Los *commits* se firman con la misma identidad que el *commit* inicial del repositorio. La historia está separada en *commits* temáticos para que cada cambio sea trazable: importación del original → estilo → pruebas → experimentos → documentación → CI, licencia y comunidad.

Ya completado: año y titular en `LICENSE`, URL en `CITATION.cff`, `README.md` (con la insignia de CI) y `CONTRIBUTING.md`, y contacto vía GitHub en `SECURITY.md` y `CODE_OF_CONDUCT.md`.

| Pendiente (opcional) | Dónde |
|---|---|
| Activar *Private vulnerability reporting* (lo citan `SECURITY.md` y `CODE_OF_CONDUCT.md`) y Dependabot | *Settings → Code security* del repositorio |
| Decidir si se publica la presentación PDF: `.gitignore` excluye `*.pdf` y no se subió | subirla como *release* o eliminar la línea `*.pdf` |
| Añadir ORCID, correo o afiliación a los autores, si se desea | `CITATION.cff` |

## 7. Funcionalidades no implementadas

Optimizadores alternativos (Newton solo existe en las pruebas), búsqueda de línea, validación cruzada e intervalos de confianza, selección de $\lambda$ y del umbral, persistencia del modelo, autenticación de la API y pruebas automatizadas del frontend. Detalle en [alcance y limitaciones](docs/introduction/scope_and_limitations.md).
