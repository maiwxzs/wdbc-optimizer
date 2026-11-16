# Diseño de módulos

## Estructura del backend

```text
backend/app/
├── main.py                      # FastAPI: rutas, CORS, lifespan, threadpool
├── api/schemas.py               # Pydantic v2: TrainRequest, CompareRequest, PredictRequest, ...
├── data/loader.py               # WDBCDataset, load_wdbc(), translate_feature()
├── math_engine/
│   ├── __init__.py              # API pública del núcleo
│   ├── activations.py           # sigmoid, naive_sigmoid, log_sigmoid, count_exp_overflows
│   ├── losses.py                # bce_from_logits, binary_cross_entropy, naive_..., l2_penalty
│   ├── gradients.py             # LogisticObjective, logistic_gradient, logistic_loss, pack/unpack
│   └── finite_differences.py    # diferencias central/progresiva, gradient_check, step_size_sweep
├── optimization/gradient_descent.py  # GDConfig, GDHistory, GDResult, StopReason, gradient_descent
├── evaluation/metrics.py        # confusion, classification_report, roc_auc, roc_curve
└── services/
    ├── model_service.py         # ModelService (orquestación + estado) y get_service()
    └── serialization.py         # to_jsonable, downsample_indices
```

## Responsabilidades

| Módulo | Responsabilidad única | Entradas → salidas |
|---|---|---|
| `activations` | funciones de activación estables e ingenuas | `z` → `σ(z)`, `log σ(z)` |
| `losses` | pérdidas BCE en tres formulaciones + L2 | `(y, z o p)` → escalar |
| `gradients` | objetivo $J(\boldsymbol\theta)$ y su gradiente analítico; constante $L$ | `θ` → `J`, `∇J` |
| `finite_differences` | derivación numérica y auditoría del gradiente | `(f, θ, h)` → gradiente aproximado, errores |
| `gradient_descent` | iteración, criterios de parada, telemetría | `(objetivo, θ⁰, config)` → `GDResult` |
| `metrics` | evaluación del clasificador | `(y, p)` → métricas |
| `loader` | datos y preprocesamiento | — → `WDBCDataset` |
| `model_service` | componer lo anterior para cada caso de uso de la API | DTO → dict JSON-safe |
| `serialization` | frontera NumPy → JSON | objetos de NumPy → tipos nativos |

## Interfaz entre el optimizador y el objetivo

`gradient_descent` solo necesita del objetivo `loss_and_grad(θ)`, los contadores `overflow_events`, `n_loss_evals` y `n_grad_evals`, y que `θ` sea un vector plano (de ahí `pack`/`unpack`). Gracias a ese empaquetado $\boldsymbol\theta=[\boldsymbol w;b]$, el mismo objetivo sirve al optimizador, a las diferencias finitas y al método de Newton de las pruebas.

## Diagrama UML de clases

```mermaid
%% Diagrama UML de clases del backend (clases y dataclasses reales)
classDiagram
    class LogisticObjective {
        +ndarray X
        +ndarray y
        +float l2
        +NumericMode mode
        +float eps
        +int overflow_events
        +int n_loss_evals
        +int n_grad_evals
        +m() int
        +dim() int
        +logits(theta) ndarray
        +loss(theta) float
        +grad(theta) ndarray
        +loss_and_grad(theta) tuple
        +lipschitz_constant() float
        -_activate(z) ndarray
    }
    class GDConfig {
        +float alpha = 0.5
        +int max_iter = 5000
        +float tol_grad = 1e-5
        +float tol_loss = 1e-12
        +float divergence_threshold = 1e8
        +int record_every = 1
        +validate() None
    }
    class GDHistory {
        +list~int~ iteration
        +list~float~ loss
        +list~float~ grad_norm
        +list~float~ loss_change
        +list~float~ weight_norm
        +append(k, J, gn, dJ, wn) None
    }
    class GDResult {
        +ndarray theta
        +StopReason stop_reason
        +int iterations
        +GDHistory history
        +float final_loss
        +float final_grad_norm
        +float elapsed_ms
        +dict numeric_flags
        +str divergence_detail
        +w() ndarray
        +b() float
        +converged() bool
        +message() str
    }
    class StopReason {
        <<enumeration>>
        GRAD_NORM
        LOSS_CHANGE
        MAX_ITER
        DIVERGENCE
    }
    class WDBCDataset {
        <<frozen dataclass>>
        +list feature_names
        +ndarray X_train / X_test
        +ndarray y_train / y_test
        +StandardScaler scaler
        +PCA pca
        +n_features() int
        +standardize(X_raw) ndarray
        +project(X_std) ndarray
    }
    class ModelService {
        +WDBCDataset ds
        -ActiveModel _active
        -int _active_version
        -OrderedDict _cache
        +train(TrainRequestDTO) dict
        +compare(scenarios, max_iter) dict
        +predict(sample_index, features) dict
        +model_info() dict
        +dataset_summary() dict
        +pca_points() dict
        +gradient_check(point, h, seed, l2, mode) dict
        +step_sweep(point, exp_min, exp_max, n, seed, l2) dict
        +float_audit() dict
        +warmup() None
    }
    class ActiveModel {
        +ndarray theta
        +dict config
        +dict result_summary
        +dict test_report
        +dict train_report
    }
    class TrainRequestDTO {
        +float alpha
        +int max_iter
        +float tol_grad
        +float tol_loss
        +float l2
        +str mode
        +str init
        +int seed
        +bool set_active
        +int max_points
    }
    GDResult *-- GDHistory
    GDResult --> StopReason
    ModelService *-- WDBCDataset
    ModelService o-- ActiveModel
    ModelService ..> TrainRequestDTO : recibe
    ModelService ..> LogisticObjective : crea
    ModelService ..> GDConfig : crea
    ModelService ..> GDResult : consume
    GDResult ..> LogisticObjective : producido optimizando
```

*Fuente: [`docs/diagrams/class_diagram.mmd`](../diagrams/class_diagram.mmd).*

El núcleo es mayoritariamente **funcional**: funciones puras sobre arreglos. Las clases son *dataclasses* de configuración y resultado, más dos clases con comportamiento: `LogisticObjective`, que encapsula los datos y la formulación numérica, y `ModelService`, que encapsula el estado de la aplicación.

## Dependencias

```mermaid
%% Dependencias entre módulos Python (imports reales)
flowchart LR
    main["app.main"] --> schemas["app.api.schemas"]
    main --> svc["app.services.model_service"]
    svc --> ser["app.services.serialization"]
    svc --> loader["app.data.loader"]
    svc --> metrics["app.evaluation.metrics"]
    svc --> me["app.math_engine"]
    svc --> fd["app.math_engine.finite_differences"]
    svc --> opt["app.optimization"]
    opt --> gd["app.optimization.gradient_descent"]
    gd --> grad["app.math_engine.gradients"]
    me --> act["app.math_engine.activations"]
    me --> loss["app.math_engine.losses"]
    me --> grad
    me --> fd
    grad --> act
    grad --> loss
    loader -. "load_breast_cancer, train_test_split,<br/>StandardScaler, PCA" .-> sklearn[("scikit-learn")]
    main -.-> fastapi[("FastAPI / Starlette")]
    schemas -.-> pydantic[("Pydantic v2")]
    act -.-> numpy[("NumPy")]
    loss -.-> numpy
    grad -.-> numpy
    fd -.-> numpy
    gd -.-> numpy
    metrics -.-> numpy
```

*Fuente: [`docs/diagrams/module_dependencies.mmd`](../diagrams/module_dependencies.mmd).*

Las líneas punteadas son dependencias externas. Solo `app.data.loader` toca scikit-learn.
