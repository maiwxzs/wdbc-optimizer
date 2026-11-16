# API REST

Base: `http://localhost:8000`. Con la API en marcha, la documentación interactiva está en `/docs` (Swagger UI) y en `/redoc`, generadas por FastAPI a partir de los esquemas de `app/api/schemas.py`.

Todas las respuestas son JSON estándar: los valores no finitos llegan como `null`. Los errores de validación devuelven **422** con un campo `detail`.

## Endpoints

| Método | Ruta | Cuerpo | Respuesta (claves principales) |
|---|---|---|---|
| GET | `/api/health` | — | `{"status": "ok"}` |
| GET | `/api/dataset/summary` | — | `n_samples`, `n_train`, `n_test`, `n_features`, `class_counts`, `label_convention`, `pca_explained_variance`, `features[]` (estadísticos por característica y clase) |
| GET | `/api/dataset/pca` | — | `train[]` (`x`, `y`, `label`), `test[]` (con `proba` y `pred` del modelo activo), `explained_variance` |
| GET | `/api/model` | — | `summary` (configuración, parada, $L$, …), `metrics.train` / `metrics.test`, `weights[]`, `bias` |
| POST | `/api/train` | `TrainRequest` | resumen de optimización + `history`, `metrics`, `roc_test`, `weights`, `active_updated` |
| POST | `/api/train/compare` | `CompareRequest` | `lipschitz_L`, `alpha_critical`, `max_iter`, `runs[]` |
| POST | `/api/predict` | `PredictRequest` | `probability_malignant`, `logit`, `prediction`, `prediction_label`, `risk_band`, `true_label`, `bias`, `pca`, `raw_features`, `contributions[]` |
| POST | `/api/audit/gradient-check` | `GradientCheckRequest` | `relative_error`, `verdict`, `message`, `components[31]`, normas |
| POST | `/api/audit/step-sweep` | `StepSweepRequest` | ver `step_size_sweep` + `verdict_at_best`, `point`, `loss_at_point` |
| GET | `/api/audit/float` | — | `z`, curvas de sigmoide y pérdida, `table[]` IEEE, `constants`, `notes[]` |

## Esquemas de petición

### `TrainRequest`

| Campo | Tipo | Defecto | Restricción |
|---|---|---|---|
| `alpha` | float | 1.0 | $0<\alpha\le10^{12}$ |
| `max_iter` | int | 8000 | 1 a 100 000 |
| `tol_grad` | float | $10^{-5}$ | $\ge0$ |
| `tol_loss` | float | $10^{-12}$ | $\ge0$ |
| `l2` | float | $10^{-3}$ | 0 a 100 |
| `mode` | `"logits"` \| `"clipped"` \| `"naive"` | `"logits"` | — |
| `init` | `"zeros"` \| `"random"` \| `"trained"` | `"zeros"` | — |
| `seed` | int | 0 | — |
| `set_active` | bool | false | si es `true` y no diverge, se convierte en el modelo activo |
| `max_points` | int | 1500 | 50 a 10 000 (submuestreo del historial) |

### `CompareRequest`

`scenarios` (lista opcional de `{label, alpha, mode, l2}`; si se omite, se usan los seis predefinidos) y `max_iter` (10 a 20 000; 3000 por defecto).

### `PredictRequest`

Exactamente uno de los dos: `sample_index` (entero $\ge0$, índice en el conjunto de prueba) o `features` (lista de **30** valores crudos, sin estandarizar).

### `GradientCheckRequest`

`point` (`zeros`/`random`/`trained`, por defecto `random`), `h` ($0<h\le1$, $10^{-5}$), `seed`, `l2`, `mode`.

### `StepSweepRequest`

`point`, `exp_min` ($-20$ a $-1$; $-16$), `exp_max` ($-10$ a $1$; $0$), `n` (5 a 200; 49), `seed`, `l2`. Requiere `exp_min < exp_max`.

## Ejemplos

```bash
curl -s localhost:8000/api/health
curl -s -X POST localhost:8000/api/train -H "Content-Type: application/json" \
     -d '{"alpha": 5, "max_iter": 3000}' | python -m json.tool | head -30
curl -s -X POST localhost:8000/api/predict -H "Content-Type: application/json" \
     -d '{"sample_index": 3}'
```

## Configuración por variables de entorno

| Variable | Dónde | Defecto | Uso |
|---|---|---|---|
| `CORS_ORIGINS` | backend | `http://localhost:5173,http://127.0.0.1:5173` | orígenes permitidos, separados por comas |
| `VITE_API_URL` | frontend (en la compilación) | `""` (mismo origen y *proxy* de Vite) | URL base de la API |
