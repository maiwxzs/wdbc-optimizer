# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/). El proyecto usa [versionado semántico](https://semver.org/lang/es/).

## [1.0.0] — 2026-11-16 · Primera publicación

Base: el proyecto original (motor numérico en NumPy, API FastAPI, interfaz React con las pestañas Clínico, Telemetría y Auditoría, y 42 pruebas), importado sin cambios en el primer *commit*. Sobre esa base:

### Añadido
- Documentación técnica y matemática completa en `docs/`: teoría, arquitectura, implementación, experimentos, glosario y bibliografía.
- Diagramas Mermaid de arquitectura, flujo de ejecución, entrenamiento, clases UML y dependencias (`docs/diagrams/`).
- 30 pruebas nuevas: `test_numerical_properties.py` (19, incluida la comparación con un método de Newton independiente), `test_integration_pipeline.py` (10) y `test_experiments_script.py` (1).
- Marcadores de pytest `unit`, `numerical`, `integration` y `architecture`.
- `scripts/reproduce_experiments.py`: seis experimentos reproducibles con métricas JSON y figuras en `results/`.
- `examples/basic_usage.py`.
- `backend/pyproject.toml` (metadatos y configuración de Ruff), `backend/requirements-dev.txt` y `ruff.toml` en la raíz.
- Integración continua con GitHub Actions (`.github/workflows/ci.yml`).
- `LICENSE` (MIT), `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `PROJECT_STATUS.md`, `.editorconfig` y `.gitattributes`.

### Cambiado
- README reescrito como punto de entrada del repositorio.
- `.gitignore` ampliado.
- Correcciones de estilo sin efecto funcional en el backend: orden de *imports*, `collections.abc.Callable`, `zip(strict=True)`, un nombre de variable ambiguo y `noqa` innecesarios (detalle en `docs/implementation/implementation_details.md`).
