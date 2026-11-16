# Detalles de implementación

Recorrido por el código siguiendo una ejecución completa. Cada fragmento está copiado del repositorio. Las rutas son relativas a `backend/`.

## 1. Datos — `app/data/loader.py`

```python
@lru_cache(maxsize=4)
def load_wdbc(test_size: float = 0.2, random_state: int = 42) -> WDBCDataset:
    bunch = load_breast_cancer()
    X = bunch.data.astype(np.float64)
    y = (1 - bunch.target).astype(np.float64)  # 1 = maligno
    ...
    scaler = StandardScaler().fit(X_tr)  # ajustado solo con train: sin fuga de datos
```

- `lru_cache` hace que los datos se carguen una sola vez por proceso y combinación de argumentos.
- `WDBCDataset` es una *dataclass* congelada: nadie puede reasignar sus campos por accidente.
- `translate_feature` traduce los 30 nombres al español para la interfaz (por ejemplo, `"mean radius"` → `"Radio (media)"`).

## 2. Objetivo — `app/math_engine/gradients.py`

`LogisticObjective` reúne los datos ($X$, $\boldsymbol y$), la regularización y la **formulación numérica**:

```python
def loss(self, theta):
    self.n_loss_evals += 1
    w, _ = unpack(theta)
    z = self.logits(theta)
    if self.mode == "logits":
        data_term = bce_from_logits(self.y, z)                         # estable
    elif self.mode == "clipped":
        data_term = binary_cross_entropy(self.y, sigmoid(z), self.eps) # acotada
    else:
        data_term = naive_binary_cross_entropy(self.y, self._activate(z))  # ingenua
    return data_term + l2_penalty(w, self.l2)
```

- `__post_init__` convierte $X$ a `float64` contiguo y $\boldsymbol y$ a vector plano, y valida que tengan el mismo número de filas.
- `_activate` usa la sigmoide ingenua solo en modo `naive` y cuenta los elementos que desbordarían.
- `grad` evalúa el residuo dentro de `np.errstate(over="ignore", invalid="ignore")`. En los modos estables no hay nada que silenciar; en el ingenuo, los avisos se convierten en datos (`has_nan`, `exp_overflow_events`) en lugar de ruido.

## 3. Optimizador — `app/optimization/gradient_descent.py`

Puntos de diseño relevantes:

```python
with np.errstate(all="ignore"):
    for k in range(cfg.max_iter + 1):         # k = 0 … max_iter (max_iter + 1 evaluaciones)
        J, g = objective.loss_and_grad(theta)
        ...
        detail = _diagnose(J, g, theta, cfg.divergence_threshold)
        if detail is not None:   stop = StopReason.DIVERGENCE
        elif gn < cfg.tol_grad:  stop = StopReason.GRAD_NORM
        elif k > 0 and dJ < cfg.tol_loss: stop = StopReason.LOSS_CHANGE
        elif k == cfg.max_iter:  stop = StopReason.MAX_ITER
        ...
        theta -= cfg.alpha * g                 # actualización in situ sobre la copia
```

- El bucle llega a `max_iter` **inclusive**: con `max_iter=25`, el historial tiene 26 registros ($k=0..25$) y se ejecutan 25 actualizaciones (`test_stops_by_max_iter`).
- `record_every` reduce el historial, pero el último registro siempre se guarda (`test_record_every_keeps_last`).
- `_diagnose` devuelve un mensaje en español que distingue `NaN` ($0\cdot\log0$, $\infty-\infty$), $\pm\infty$ ($\log0$ o desbordamiento), umbral superado y gradiente o parámetros no finitos.

## 4. Servicio — `app/services/model_service.py`

- **Modelo por defecto:** `DEFAULT_CONFIG = GDConfig(alpha=1.0, max_iter=8000, tol_grad=1e-5, tol_loss=1e-12)` y `DEFAULT_L2 = 1e-3`.
- **`train`:** si `set_active=False`, el resultado se cachea porque no tiene efectos secundarios. Si `set_active=True` y no diverge, reemplaza el modelo activo bajo un candado e incrementa `_active_version`, lo que invalida la caché dependiente.
- **`compare`:** ejecuta seis escenarios predefinidos (o los recibidos) desde $\boldsymbol\theta=\mathbf 0$ e informa de $\alpha/(2/L)$ para cada uno.
- **`predict`:** acepta `sample_index` (índice en el conjunto de prueba) **o** 30 valores crudos. Devuelve $p$, $z$, banda de riesgo, proyección PCA y las contribuciones $w_j\tilde x_j$ ordenadas por magnitud.
- **`gradient_check` / `step_sweep`:** el punto de auditoría es `zeros`, `random` (semilla) o `trained`.
- **`float_audit`:** genera la rejilla de $z$, las curvas de las tres formulaciones, la tabla IEEE y las constantes de `float64`.

## 5. API — `app/main.py`

```python
async def _call(fn, *args, **kwargs):
    try:
        return await run_in_threadpool(fn, *args, **kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
```

Todos los errores de validación del dominio (`ValueError`) se convierten en HTTP 422 con un mensaje legible. Los errores de esquema los produce Pydantic con el mismo código. El catálogo completo de rutas está en [api_reference.md](api_reference.md).

## 6. Cambios realizados durante la reorganización

Para preparar la publicación solo se hicieron cambios **sin efecto funcional** en el código de la aplicación. Las 42 pruebas originales siguen pasando sin modificaciones en sus aserciones.

| Archivo | Cambio | Motivo |
|---|---|---|
| `app/services/model_service.py` | `zip(..., strict=True)` en tres lugares; variable `l` → `lbl` | Ruff B905/E741: detectar longitudes distintas y evitar un nombre ambiguo |
| `app/math_engine/__init__.py` | orden de *imports* y de `__all__` | Ruff I001/RUF022 |
| `finite_differences.py`, `gradient_descent.py`, `model_service.py` | `Callable` importado de `collections.abc` | Ruff UP035 (forma moderna equivalente) |
| `app/api/schemas.py` | anotación de retorno sin comillas | Ruff UP037 |
| `tests/conftest.py` | `noqa` innecesarios eliminados; asignación de marcadores por módulo | Ruff RUF100; clasificación de pruebas |
