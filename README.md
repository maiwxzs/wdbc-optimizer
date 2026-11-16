# 🩺 WDBC · Motor de Optimización Numérica & CDSS Oncológico

Plataforma web full-stack para el diagnóstico asistido de cáncer de mama (Wisconsin Diagnostic Breast Cancer) y la **auditoría rigurosa de un optimizador**. La regresión logística no se presenta como "un modelo de ML", sino como la solución numérica de un problema de optimización convexa sin restricciones:

$$\min_{\boldsymbol w\in\mathbb R^{30},\,b\in\mathbb R}\; J(\boldsymbol w,b)= \frac1m\sum_{i=1}^m\Big[\log\!\big(1+e^{z_i}\big)-y_i z_i\Big]+\frac\lambda2\lVert\boldsymbol w\rVert_2^2,\qquad z_i=\boldsymbol w^\top\tilde{\boldsymbol x}_i+b$$

> ⚠️ Herramienta académica. No es un dispositivo médico ni sustituye el criterio clínico.

---

## 1. Puesta en marcha

### Backend (Python 3.10+)
```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                                                  # 42 pruebas
uvicorn app.main:app --reload --port 8000               # docs: http://localhost:8000/docs
```

### Frontend (Node 18+)
```bash
cd frontend
npm install
npm run dev                                             # http://localhost:5173
```
### ⚡ Modo presentación (recomendado para la sustentación)
`npm run dev` es el modo de desarrollo: sirve cientos de módulos sin comprimir y es notablemente más lento. Para la presentación, con el backend corriendo:
```bash
cd frontend
npm run presentacion        # compila, optimiza y abre http://localhost:4173
```
En Windows/PowerShell, si `npm` está bloqueado: `npm.cmd run presentacion`.

Vite redirige `/api/*` a `http://127.0.0.1:8000`. Para otro host: `VITE_API_URL=https://mi-api npm run dev`, y en el backend `CORS_ORIGINS=https://mi-front`.

**Equipo:** Maicol Beltran Aguilar (*Lead Numerical Engineer*) y Brayan Fernandez (*Frontend Architect*). Se edita en `frontend/src/config/team.ts`.

### Optimizaciones de rendimiento aplicadas
- **Backend:** caché de resultados deterministas (escenarios, entrenamiento, curva en V, laboratorio flotante) y precálculo en segundo plano al arrancar: Telemetría y Auditoría pasan de ~1.4 s a < 1 ms.
- **Carga:** ECharts modular (solo los 5 tipos de gráfico usados, en vez del paquete completo), pestañas y KaTeX con carga diferida, librerías en chunks cacheables, fuentes sin bloquear el render.
- **60 fps:** sin `backdrop-filter` en los paneles (solo en barra y desplegable), fondo en una capa fija en vez de `background-attachment: fixed`, borde rotatorio animado con `transform` (GPU) en lugar de repintar el degradado, sin sombras por punto en los diagramas de dispersión, onda del paciente en capa de canvas propia, submuestreo LTTB en curvas largas, y sin `StrictMode` (duplicaba las peticiones en desarrollo).

---

## 2. Arquitectura

```text
backend/app/
├── math_engine/              ← NumPy puro (prohibido sklearn: lo verifica tests/test_architecture.py)
│   ├── activations.py        σ(z) estable por tramos, σ ingenua (auditoría), log σ vía LSE
│   ├── losses.py             BCE Log-Sum-Exp · BCE con clip ε=1e-15 · BCE ingenua
│   ├── gradients.py          ∇J analítico vectorizado + LogisticObjective(θ=[w;b]) + constante L
│   └── finite_differences.py diferencias central/progresiva, error relativo, barrido en V
├── optimization/
│   └── gradient_descent.py   GD con telemetría y 4 condiciones de parada
├── evaluation/metrics.py     matriz de confusión, sensibilidad, especificidad, AUC (Mann-Whitney)
├── data/loader.py            ÚNICO uso de sklearn: load_breast_cancer, train_test_split, StandardScaler, PCA(2)
├── services/model_service.py orquestación + estado del modelo activo
├── api/schemas.py            validación Pydantic v2
└── main.py                   FastAPI (async + run_in_threadpool para el cálculo CPU-bound)

frontend/src/
├── App.tsx                   AnimatePresence mode="wait" (transición tipo diapositiva)
├── components/CreatorsCard   firma con borde conic-gradient rotatorio, hover 1.03x, desplegable
├── components/charts/        ECharts: pérdida, gauge elástico, curva V con caída en rebote, etc.
└── tabs/                     Clínico · Telemetría · Auditoría
```

**Convención de etiquetas:** sklearn codifica `0 = maligno`. El cargador re-codifica `y = 1 − target`, de modo que **y = 1 ⇔ maligno** y σ(z) se lee directamente como P(maligno | x). El `StandardScaler` se ajusta **solo con train** (sin fuga de datos). Split estratificado 80/20, semilla 42.

---

## 3. Especificación numérica implementada

| Componente | Implementación | Por qué |
|---|---|---|
| Sigmoide | `e = exp(−|z|)`; `1/(1+e)` si z ≥ 0, `e/(1+e)` si z < 0 | Argumento de `exp` siempre ≤ 0 → nunca desborda. Un solo `exp` por elemento (con `np.where` se evaluarían ambas ramas y saltarían warnings). |
| Pérdida (motor) | `logaddexp(0, z) − y·z` | Algebraicamente idéntica a la BCE y exacta en float64 para cualquier z. |
| Pérdida (alternativa) | `clip(ŷ, ε, 1−ε)`, ε = 1e-15 | Seleccionable (`mode="clipped"`); evita log 0 pero satura en ≈ 34.54. |
| Gradiente | `Xᵀ(ŷ − y)/m + λw`, `Σ(ŷ − y)/m` | Exacto, vectorizado. |
| Diferencias finitas | central `[J(θ+he_j) − J(θ−he_j)]/2h` y progresiva | Error relativo `‖g_a − g_n‖/(‖g_a‖+‖g_n‖+1e-15)`. |
| Parada | divergencia → ‖∇J‖ → \|ΔJ\| → k_max | La divergencia se evalúa primero: `inf − inf = NaN` no debe pasar por "convergencia". Nunca lanza excepción; devuelve el estado. |
| JSON | NaN/±Inf → `null` (series) o `"NaN"`, `"+inf"` (tabla IEEE) | JSON estándar no admite no finitos; el servidor nunca cae. |

`pytest.ini` convierte cualquier `RuntimeWarning` en error: si la ruta estable emitiera un overflow, las pruebas fallarían.

---

## 4. Resultados medidos (semilla 42, λ = 1e-3)

| Magnitud | Valor |
|---|---|
| Constante de suavidad L = λ_max(X̃ᵀX̃)/(4m) + λ | 3.3455 |
| α crítico (cota suficiente) 2/L | 0.598 |
| Modelo por defecto (α = 1) | converge por ‖∇J‖ < 1e-5 en 4 183 iteraciones (~0.26 s) |
| Test (114 pacientes) | exactitud 97.4 %, sensibilidad 95.2 %, especificidad 98.6 %, AUC 0.993 |
| Chequeo de gradiente (θ aleatorio, h = 1e-5) | error relativo ≈ 1.2e-11 → *excelente* |
| h* central (empírico / teoría ε^⅓) | 2.2e-5 / 6.1e-6 |
| h* progresiva (empírico / teoría √ε) | 4.6e-8 / 1.5e-8 |

**Escenarios comparativos (k_max = 3000):**

| Escenario | Parada | Observación |
|---|---|---|
| α = 0.05 | max_iter | estable pero lento |
| α = 1/L | max_iter | descenso monótono garantizado |
| α = 5 (≈ 8× α_crít) | **grad_norm en 837 it.** | 2/L es suficiente, no necesario: la curvatura local es < L |
| α = 100 | max_iter, J ≈ 18.8 | régimen oscilatorio |
| α = 1e5 | **divergence** (J ≈ 9.7e10 > 1e8) | |
| α = 1, formulación ingenua | **divergence** (NaN en k = 44) | σ(z) redondea a 1.0 para z ≳ 36.7 → `0·log 0 = NaN`, sin un solo overflow |

### Hallazgos que conviene destacar en la sustentación
1. **Sin regularización el problema está mal planteado.** El conjunto de entrenamiento es casi linealmente separable: sin L2 no existe minimizador finito (J → 0 con ‖w‖ → ∞) y la exactitud en test *baja* con α grandes. Con λ = 1e-3 el objetivo es fuertemente convexo y tiene minimizador único.
2. **El NaN de la BCE ingenua llega antes que el overflow:** basta la saturación σ = 1.0 (z ≈ 37), mucho antes de `exp` > DBL_MAX (z ≈ −709.8).
3. **El clip no es inocuo:** 1 − 1e-15 no es representable en float64 (se redondea a 1 − 9.992e-16), y la pérdida por muestra queda acotada en ≈ 34.54, lo que sesga muestras muy mal clasificadas. La forma Log-Sum-Exp no tiene ninguno de los dos problemas.
4. **El chequeo de gradiente en el óptimo da "aceptable" (≈3e-7), no "excelente":** como ‖∇J(θ*)‖ ≈ 1e-5, el ruido de redondeo ε·J/h domina el numerador. No es un error del gradiente; por eso la auditoría se hace por defecto en un θ aleatorio.

---

## 5. API REST

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/health` | estado |
| GET | `/api/dataset/summary` | tamaños, clases, estadísticas por característica |
| GET | `/api/dataset/pca` | proyección PCA 2D (train + test con probabilidades) |
| GET | `/api/model` | modelo activo: resumen de optimización, métricas, pesos |
| POST | `/api/train` | `{alpha, max_iter, tol_grad, tol_loss, l2, mode, init, seed, set_active}` → historia + métricas |
| POST | `/api/train/compare` | barrido de escenarios convergentes / oscilatorios / divergentes |
| POST | `/api/predict` | `{sample_index}` o `{features: [30 valores crudos]}` → p, z, contribuciones wⱼx̃ⱼ |
| POST | `/api/audit/gradient-check` | `{point, h, seed}` → analítico vs numérico por componente |
| POST | `/api/audit/step-sweep` | `{point, exp_min, exp_max, n}` → curva en V + modelo ajustado |
| GET | `/api/audit/float` | laboratorio IEEE-754: sigmoide/pérdida ingenua vs clip vs LSE |

---

## 6. Pestañas

- **📊 Panel Clínico & Simulación** — KPIs del modelo activo, selector de paciente (aleatorio benigno/maligno, caso frontera, errores del modelo), sliders "what-if" de los 10 rasgos medios con marcas de la media de cada clase, gauge radial con aguja elástica y anillo de pulso rojo si p ≥ 50 %, explicabilidad del logit, mapa PCA clicable y matriz de confusión.
- **⚡ Telemetría del Motor Numérico** — control de α (escala log, con razón α/α_crít), tolerancias, λ, formulación numérica e inicialización; curvas J(θ⁽ᵏ⁾), ‖∇J‖₂ y |ΔJ| con trazado animado; condiciones de parada con la activa resaltada; barrido comparativo de 6 escenarios. Opción para promover el resultado al panel clínico.
- **🔬 Auditoría Matemática** — chequeo de gradiente en 31 parámetros con veredicto, curva en V log-log (los puntos caen rebotando sobre su valor), modelo de error total con h* teórico vs empírico, y laboratorio de punto flotante con tabla de valores especiales IEEE-754.

---

## 7. Animaciones (Framer Motion + ECharts)
- Pestañas: salida `x: −30, opacity 0, blur 4px` (0.25 s); entrada `x: 30 → 0` con `spring(260, 20)`, opacidad y desenfoque en 0.35 s.
- Tarjetas en cascada: `staggerChildren: 0.08`, `y: 20 → 0`.
- Pérdida: `animationDuration: 2000`, `cubicInOut` (la curva se dibuja de izquierda a derecha en cada ejecución).
- Gauge: `elasticOut` en entrada y actualización; color interpolado verde → ámbar → rojo.
- Curva en V: puntos que parten del techo del eje y caen con `bounceOut`, escalonados 28 ms por índice.
