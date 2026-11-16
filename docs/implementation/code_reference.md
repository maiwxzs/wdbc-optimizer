# Referencia de código

Firmas públicas del paquete `backend/app`, extraídas del código con `inspect`. Los nombres que empiezan por `_` son internos.

## `app.math_engine.activations`

| Símbolo | Firma | Descripción |
|---|---|---|
| `EXP_OVERFLOW_THRESHOLD` | `float` | $\ln(\text{DBL\_MAX})\approx709.78$ |
| `sigmoid` | `(z: ndarray \| float) -> ndarray \| float` | sigmoide estable por tramos; conserva la forma; devuelve `float` si la entrada es escalar |
| `naive_sigmoid` | `(z) -> ndarray \| float` | $1/(1+e^{-z})$ sin protección (solo para auditoría) |
| `log_sigmoid` | `(z) -> ndarray \| float` | $\log\sigma(z)=-\operatorname{logaddexp}(0,-z)$ |
| `count_exp_overflows` | `(z: ndarray) -> int` | número de elementos con $-z>709.78$ |

## `app.math_engine.losses`

| Símbolo | Firma | Descripción |
|---|---|---|
| `EPSILON` | `float = 1e-15` | margen del *clip* |
| `bce_from_logits` | `(y, z) -> float` | BCE media vía Log-Sum-Exp |
| `bce_per_sample_from_logits` | `(y, z) -> ndarray` | BCE por muestra vía Log-Sum-Exp |
| `binary_cross_entropy` | `(y, y_hat, eps=1e-15) -> float` | BCE con `clip(y_hat, eps, 1-eps)` |
| `naive_binary_cross_entropy` | `(y, y_hat) -> float` | BCE sin protección; puede dar `inf`/`NaN` |
| `l2_penalty` | `(w, lam) -> float` | $\frac\lambda2\lVert\boldsymbol w\rVert^2$ (0 si $\lambda\le0$) |

## `app.math_engine.gradients`

| Símbolo | Firma | Descripción |
|---|---|---|
| `NumericMode` | `Literal["logits", "clipped", "naive"]` | formulación numérica |
| `logistic_gradient` | `(X, y, w, b, l2=0.0) -> tuple[ndarray, float]` | $(\nabla_{\boldsymbol w}J,\partial_bJ)$ en forma separada |
| `logistic_loss` | `(X, y, w, b, l2=0.0) -> float` | $J$ en forma separada (Log-Sum-Exp) |
| `pack` | `(w, b) -> ndarray` | $[\boldsymbol w;b]$ |
| `unpack` | `(theta) -> tuple[ndarray, float]` | inversa de `pack` |
| `LogisticObjective` | `@dataclass(X, y, l2=0.0, mode="logits", eps=1e-15)` | objetivo empaquetado |
| `  .m`, `.dim` | propiedades | $m$ y $n=d+1$ |
| `  .logits(theta)` | `-> ndarray` | $X\boldsymbol w+b$ |
| `  .loss(theta)` | `-> float` | $J(\boldsymbol\theta)$ según `mode` |
| `  .grad(theta)` | `-> ndarray` | $\nabla J(\boldsymbol\theta)$ |
| `  .loss_and_grad(theta)` | `-> tuple[float, ndarray]` | ambos |
| `  .lipschitz_constant()` | `-> float` | $L=\lambda_{\max}(\tilde X^\top\tilde X)/(4m)+\lambda$ |
| `  .overflow_events`, `.n_loss_evals`, `.n_grad_evals` | `int` | contadores de telemetría |

## `app.math_engine.finite_differences`

| Símbolo | Firma | Descripción |
|---|---|---|
| `EPS_MACHINE` | `float` | $2^{-52}$ |
| `central_difference_gradient` | `(f, theta, h=1e-5) -> ndarray` | diferencia central por componente |
| `forward_difference_gradient` | `(f, theta, h=1e-8) -> ndarray` | diferencia progresiva |
| `relative_error` | `(g_analytic, g_numeric) -> float` | $\lVert g_a-g_n\rVert/(\lVert g_a\rVert+\lVert g_n\rVert+10^{-15})$ |
| `classify_relative_error` | `(err) -> tuple[str, str]` | veredicto y mensaje |
| `gradient_check` | `(loss_fn, grad_fn, theta, h=1e-5) -> dict` | claves: `h`, `analytic`, `numeric`, `abs_error`, `component_rel_error`, `relative_error`, `max_abs_error`, `analytic_norm`, `numeric_norm`, `verdict`, `message` |
| `step_size_sweep` | `(loss_fn, grad_fn, theta, hs) -> dict` | claves: `h`, `central_error`, `forward_error`, `central_model`, `forward_model`, `central_best`, `forward_best`, `central_h_opt_fit`, `forward_h_opt_fit`, `central_h_opt_theory`, `forward_h_opt_theory`, `eps_machine` |

## `app.optimization.gradient_descent`

| Símbolo | Firma | Descripción |
|---|---|---|
| `StopReason` | `Enum(str)`: `GRAD_NORM`, `LOSS_CHANGE`, `MAX_ITER`, `DIVERGENCE` | motivo de parada |
| `STOP_MESSAGES` | `dict[StopReason, str]` | mensajes en español |
| `GDConfig` | `@dataclass(alpha=0.5, max_iter=5000, tol_grad=1e-5, tol_loss=1e-12, divergence_threshold=1e8, record_every=1)` | `.validate()` lanza `ValueError` |
| `GDHistory` | `@dataclass` con listas `iteration`, `loss`, `grad_norm`, `loss_change`, `weight_norm` | telemetría |
| `GDResult` | `@dataclass(theta, stop_reason, iterations, history, final_loss, final_grad_norm, elapsed_ms, numeric_flags, divergence_detail=None)` | propiedades `w`, `b`, `converged`, `message` |
| `gradient_descent` | `(objective, theta0, config=None, callback=None) -> GDResult` | optimizador |

## `app.evaluation.metrics`

| Símbolo | Firma |
|---|---|
| `confusion` | `(y_true, y_pred) -> dict[str, int]` |
| `classification_report` | `(y_true, proba, threshold=0.5) -> dict` |
| `roc_auc` | `(y_true, scores) -> float` |
| `roc_curve` | `(y_true, scores, n_points=101) -> dict[str, list[float]]` |

## `app.data.loader`

| Símbolo | Firma |
|---|---|
| `load_wdbc` | `(test_size=0.2, random_state=42) -> WDBCDataset` (con `lru_cache`) |
| `WDBCDataset` | *dataclass* congelada: `feature_names`, `feature_names_es`, `X_train_raw`, `X_test_raw`, `X_train`, `X_test`, `y_train`, `y_test`, `idx_train`, `idx_test`, `scaler`, `pca`, `pca_train`, `pca_test`; `.n_features`, `.standardize(X_raw)`, `.project(X_std)` |
| `translate_feature` | `(name: str) -> str` |

## `app.services`

| Símbolo | Firma |
|---|---|
| `serialization.to_jsonable` | `(obj) -> Any` |
| `serialization.downsample_indices` | `(n, max_points) -> ndarray` |
| `model_service.TrainRequestDTO` | `@dataclass(alpha=1.0, max_iter=8000, tol_grad=1e-5, tol_loss=1e-12, l2=1e-3, mode="logits", init="zeros", seed=0, set_active=False, max_points=1500)` |
| `model_service.ModelService` | `(dataset=None)`; métodos: `train`, `compare`, `predict`, `model_info`, `dataset_summary`, `pca_points`, `gradient_check`, `step_sweep`, `float_audit`, `warmup` |
| `model_service.get_service` | `() -> ModelService` (singleton seguro entre hilos) |

## Herramientas fuera del paquete

| Archivo | Uso |
|---|---|
| `scripts/reproduce_experiments.py` | `python scripts/reproduce_experiments.py [--output DIR] [--no-figures]` |
| `examples/basic_usage.py` | `python examples/basic_usage.py` |
