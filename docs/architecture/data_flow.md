# Flujo de datos

## 1. Flujo de ejecución (arranque y petición de entrenamiento)

```mermaid
%% Flujo de ejecución: arranque de la API y atención de POST /api/train
sequenceDiagram
    autonumber
    participant U as Usuario (navegador)
    participant F as Frontend (React)
    participant A as FastAPI (main.py)
    participant S as ModelService
    participant G as gradient_descent
    participant O as LogisticObjective

    Note over A,S: Arranque (lifespan)
    A->>S: get_service() en threadpool
    S->>S: load_wdbc() · split 80/20 · StandardScaler · PCA
    S->>G: train(α=1, set_active=True)
    G->>O: loss_and_grad(θ) × 4184
    G-->>S: GDResult (grad_norm, k=4183)
    S-->>A: modelo activo listo
    A-)S: warmup() en segundo plano (compare, sweep, float_audit, pca)

    Note over U,A: Petición del usuario (pestaña Telemetría)
    U->>F: ajusta α, tolerancias, λ, modo
    F->>A: POST /api/train {alpha, max_iter, ...}
    A->>A: TrainRequest (Pydantic) → TrainRequestDTO
    A->>S: train(dto) en threadpool
    alt resultado en caché
        S-->>A: dict cacheado
    else
        S->>G: gradient_descent(obj, θ⁰, GDConfig)
        loop k = 0 … k_max
            G->>O: loss_and_grad(θ)
            G->>G: comprobar divergencia → ‖∇J‖ → |ΔJ| → k_max
            G->>G: θ ← θ − α∇J
        end
        G-->>S: GDResult
        S->>S: métricas, ROC, pesos, L, submuestreo
        S->>S: to_jsonable (NaN/Inf → null)
    end
    S-->>A: dict JSON-safe
    A-->>F: 200 OK (o 422 si ValueError)
    F-->>U: curvas J, ‖∇J‖, |ΔJ| y motivo de parada
```

*Fuente: [`docs/diagrams/execution_flow.mmd`](../diagrams/execution_flow.mmd).*

**Explicación.**

1. **Arranque.** El `lifespan` de FastAPI crea el `ModelService` singleton, que carga los datos y entrena el modelo por defecto ($\alpha=1$; 4183 iteraciones). Después lanza `warmup()` en un *executor* para precalcular en segundo plano lo que piden las pestañas al abrirse, sin retrasar el arranque.
2. **Petición.** Pydantic valida el cuerpo y lo responde con 422 si está fuera de rango. El DTO pasa al servicio en un hilo. Si la misma configuración ya se calculó (y no promueve el modelo), se devuelve desde la caché.
3. **Cálculo.** El optimizador itera hasta un criterio de parada. El servicio añade métricas, la curva ROC, los pesos ordenados, $L$ y $\alpha/(2/L)$, y submuestrea el historial a `max_points`.
4. **Serialización.** `to_jsonable` convierte los tipos de NumPy a tipos nativos y los no finitos a `null`. La respuesta es siempre JSON estándar.

## 2. Proceso de entrenamiento

```mermaid
%% Proceso de entrenamiento (gradient_descent) tal como está implementado
flowchart TD
    A["Preparación de datos<br/>X estandarizada (455×30), y ∈ {0,1}"] --> B["Objetivo J(θ)<br/>LogisticObjective(X, y, λ, modo)"]
    B --> C["Inicialización θ⁽⁰⁾<br/>zeros · random(seed) · trained"]
    C --> D["k ← 0"]
    D --> E["Predicción<br/>z = Xw + b,  p = σ(z)"]
    E --> F["Pérdida<br/>J = mean(logaddexp(0,z) − y·z) + λ/2‖w‖²"]
    F --> G["Gradiente<br/>∇w = Xᵀ(p−y)/m + λw,  ∂b = mean(p−y)"]
    G --> H{"¿J no finito, J > 1e8<br/>o ∇J/θ no finitos?"}
    H -- sí --> X1["Parada: DIVERGENCE<br/>+ diagnóstico"]
    H -- no --> I{"‖∇J‖₂ < tol_grad?"}
    I -- sí --> X2["Parada: GRAD_NORM"]
    I -- no --> J{"k > 0 y |ΔJ| < tol_loss?"}
    J -- sí --> X3["Parada: LOSS_CHANGE"]
    J -- no --> K{"k = k_max?"}
    K -- sí --> X4["Parada: MAX_ITER"]
    K -- no --> L["Registrar telemetría<br/>(k, J, ‖∇J‖, |ΔJ|, ‖w‖)"]
    L --> M["Actualización<br/>θ ← θ − α∇J"]
    M --> N["k ← k + 1"]
    N --> E
    X1 --> R["GDResult: θ, motivo, historial,<br/>banderas numéricas, tiempo"]
    X2 --> R
    X3 --> R
    X4 --> R
```

*Fuente: [`docs/diagrams/training_pipeline.mmd`](../diagrams/training_pipeline.mmd).*

## 3. Transformaciones de los datos

| Paso | Entrada | Salida | Dónde |
|---|---|---|---|
| Carga | `load_breast_cancer()` | `X` (569×30), `target` | `loader.load_wdbc` |
| Etiquetas | `target` (0 = maligno) | `y = 1 − target` (1 = maligno) | ídem |
| Partición | `X, y, idx` | train 455 / test 114, estratificado | ídem |
| Escalado | `X_train_raw` | `X_train` con media 0 y desv. 1; `X_test` con los mismos parámetros | ídem |
| PCA | `X_train` estandarizada | `pca_train`, `pca_test` (2D, solo para gráficos) | ídem |
| Logits | `X`, `θ` | `z = X w + b` | `LogisticObjective.logits` |
| Probabilidades | `z` | `p = σ(z)` | `sigmoid` |
| Predicción en tiempo real | 30 valores crudos | `standardize` → `w·x̃` → `z` → `p` → banda de riesgo | `ModelService.predict` |

## 4. Estado

| Estado | Ámbito | Ciclo de vida |
|---|---|---|
| `WDBCDataset` | proceso (`lru_cache`) | inmutable (`frozen=True`) |
| `ModelService` | proceso (singleton `get_service`) | se crea al arrancar |
| Modelo activo (`ActiveModel`) | proceso | se reemplaza con `set_active=true` si el entrenamiento no diverge |
| Caché de resultados | proceso | LRU de 64 entradas; la clave incluye la versión del modelo activo |
| `LogisticObjective` | por petición | se descarta tras responder |
