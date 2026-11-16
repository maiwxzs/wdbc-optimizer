# Decisiones de diseño

Registro de decisiones (formato ADR abreviado). Las marcadas como **[original]** ya estaban en el código antes de esta reorganización; las marcadas como **[reorganización]** se tomaron al preparar el repositorio para publicarlo.

---

### DD-01 · Núcleo numérico solo con NumPy [original]

**Contexto.** El propósito académico es entender e implementar los mecanismos de entrenamiento.
**Decisión.** `math_engine`, `optimization` y `evaluation` solo importan NumPy y la biblioteca estándar.
**Verificación.** `tests/test_architecture.py::test_numeric_core_has_no_ml_libraries` analiza el AST de cada archivo y falla si encuentra `sklearn`, `scipy`, `torch`, `tensorflow` o `jax`.

### DD-02 · scikit-learn limitado a cuatro utilidades de datos [original]

**Decisión.** Solo `load_breast_cancer`, `train_test_split`, `StandardScaler` y `PCA`, y únicamente en `data/loader.py`.
**Motivo.** Ninguna de ellas entrena el clasificador ni calcula su gradiente; reimplementarlas no aporta al objetivo del curso.
**Verificación.** `test_sklearn_only_for_allowed_utilities`.
**Consecuencia.** scikit-learn es una dependencia de ejecución, y la copia de WDBC viene con su instalación.

### DD-03 · Pérdida Log-Sum-Exp como formulación por defecto [original]

**Alternativas.** BCE de libro sobre $p$ o BCE con *clip*.
**Decisión.** $\ell=\operatorname{logaddexp}(0,z)-yz$: exacta y sin `NaN`/`inf` para todo $z$ finito ([teoría](../theory/loss_functions.md)). Las otras dos se conservan **solo** para compararlas en la auditoría.

### DD-04 · Sigmoide por tramos con una sola exponencial [original]

**Decisión.** $e=\exp(-\lvert z\rvert)$ y selección por máscara, en lugar de `np.where`, que evalúa ambas ramas y emite avisos ([detalle](../theory/floating_point.md#3-sigmoide-estable-por-tramos)).

### DD-05 · `RuntimeWarning` como error en las pruebas [original]

**Decisión.** `filterwarnings = error::RuntimeWarning` en `pytest.ini`. Así, cualquier desbordamiento o división por cero que se cuele en la ruta estable hace fallar las pruebas. Las rutas ingenuas silencian sus avisos de forma explícita con `np.errstate`.

### DD-06 · Divergencia evaluada antes que convergencia; el optimizador nunca lanza excepciones [original]

**Motivo.** Un $J$ no finito no debe poder pasar por un criterio de convergencia, y la interfaz necesita mostrar **cómo** falló un entrenamiento, no un error HTTP 500.
**Consecuencia.** `GDResult.stop_reason = DIVERGENCE` incluye un `divergence_detail` legible.

### DD-07 · Parámetros empaquetados θ = [w; b] [original]

**Motivo.** Un único vector permite reutilizar el mismo objetivo en el optimizador, en las diferencias finitas y en Newton (pruebas), y calcular $L$ con la matriz aumentada.

### DD-08 · Sesgo sin regularizar [original]

**Motivo.** Penalizar $b$ sesgaría la tasa base de malignidad hacia $0.5$. Es la práctica estándar.

### DD-09 · JSON estricto: no finitos → `null` [original]

**Motivo.** `NaN` e `Infinity` no son JSON válido; muchos analizadores (incluido `JSON.parse`) los rechazan.
**Verificación.** Las pruebas usan `json.dumps(..., allow_nan=False)`.

### DD-10 · Cálculo en *threadpool* y caché versionada [original]

**Motivo.** Los *endpoints* `async` no deben bloquearse con cálculo CPU. Los resultados deterministas se memorizan, y la clave incluye la versión del modelo activo para invalidar lo que depende de $\boldsymbol\theta^\star$.

### DD-11 · Se conserva la estructura `backend/` + `frontend/` en lugar de `src/` [reorganización]

**Contexto.** La estructura de referencia proponía `src/`.
**Decisión.** Mantener `backend/app` y `frontend/src`.
**Motivo.** Es un proyecto *full-stack* con dos ecosistemas (pip y npm). Mover `app/` rompería el comando `uvicorn app.main:app`, el `sys.path` de `conftest.py`, el *proxy* de Vite y las instrucciones que ya usa el equipo, sin ningún beneficio funcional.

### DD-12 · Pruebas en una sola carpeta, clasificadas con marcadores [reorganización]

**Decisión.** No mover las pruebas existentes a `tests/unit`, `tests/integration` y `tests/numerical`. Cada módulo se clasifica con marcadores de pytest (`unit`, `numerical`, `integration`, `architecture`), que se asignan en `conftest.py`.
**Motivo.** Conserva intactos los archivos originales y su historial, y permite ejecutar categorías con `pytest -m numerical`.

### DD-13 · Sin carpeta `data/` ni `notebooks/` [reorganización]

**Motivo.** Los datos vienen con scikit-learn (no hay archivos que versionar), y el análisis reproducible está en `scripts/reproduce_experiments.py`. Un cuaderno duplicaría ese código. Una carpeta vacía no tendría ninguna finalidad real.

### DD-14 · Ruff con `E501` ignorado y símbolos Unicode permitidos [reorganización]

**Motivo.** El código original usa líneas de hasta 141 caracteres y notación matemática Unicode (σ, ∇, ‖·‖) en comentarios y cadenas a propósito. Reformatearlo todo generaría un cambio masivo sin valor funcional. El resto de reglas (`E`, `F`, `W`, `I`, `B`, `UP`, `NPY`, `RUF`) sí se aplican y pasan.

### DD-15 · Licencia MIT [reorganización]

Ver la justificación en el [README](../../README.md#licencia).
