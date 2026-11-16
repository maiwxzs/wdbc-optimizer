# Frontend

Aplicación de una sola página en `frontend/`. **No realiza ningún cálculo del modelo**: todas las cifras proceden de la API. Su función es visualizar y permitir experimentar.

## Tecnologías

React 18 · TypeScript 5 · Vite 5 · Tailwind CSS 3 · ECharts 5 (importación modular) · Framer Motion 11 · KaTeX · lucide-react. Las versiones exactas están en `frontend/package.json` y `package-lock.json`.

## Estructura

```text
frontend/src/
├── App.tsx                 # pestañas con React.lazy + AnimatePresence
├── main.tsx                # punto de entrada (sin StrictMode)
├── config/team.ts          # datos del equipo y del proyecto (editar aquí)
├── lib/
│   ├── api.ts              # cliente fetch tipado → /api/*
│   ├── types.ts            # tipos de las respuestas (Num = number | null)
│   ├── echarts.ts          # registro modular de los gráficos usados
│   ├── chartTheme.ts, format.ts, motion.ts
├── hooks/useAsync.ts       # peticiones con descarte de respuestas obsoletas
├── components/
│   ├── TabNav.tsx, CreatorsCard.tsx, TeX.tsx, ui.tsx
│   └── charts/             # EChart, LogLineChart, RiskGauge, VCurveChart,
│                           # GradientCompareChart, FloatChart, PCAScatter, ContributionsChart
└── tabs/
    ├── ClinicalTab.tsx     # usa /dataset/summary, /model, /dataset/pca, /predict
    ├── TelemetryTab.tsx    # usa /train, /train/compare
    └── AuditTab.tsx        # usa /audit/gradient-check, /audit/step-sweep, /audit/float
```

## Pestañas

| Pestaña | Qué muestra | Endpoints |
|---|---|---|
| **Panel Clínico & Simulación** | KPIs del modelo activo, selector de paciente, *sliders* "qué pasaría si" sobre los 10 rasgos medios, indicador de riesgo, contribuciones al logit, mapa PCA y matriz de confusión | `summary`, `model`, `pca`, `predict` (por índice y por características) |
| **Telemetría del Motor Numérico** | controles de $\alpha$ (escala logarítmica, con $\alpha/\alpha_{\text{crít}}$), tolerancias, $\lambda$, modo e inicialización; curvas $J$, $\lVert\nabla J\rVert$, $\lvert\Delta J\rvert$; criterios de parada; comparación de 6 escenarios | `train`, `compare` |
| **Auditoría Matemática** | chequeo de gradiente en 31 componentes, curva en V, $h^\star$ teórico frente al empírico, laboratorio IEEE-754 | `gradient-check`, `step-sweep`, `float` |

## Rendimiento (según el README original)

Carga diferida de las pestañas y de KaTeX, ECharts modular, *chunks* de bibliotecas separados (`vite.config.ts → manualChunks`), submuestreo LTTB en las curvas largas (`LogLineChart`) y modo de presentación (`npm run presentacion`), que sirve la compilación optimizada.

## Verificación

- `npm run typecheck` (`tsc --noEmit`) y `npm run build` se ejecutan en la CI.
- **No hay pruebas automatizadas de componentes ni de extremo a extremo** (ver [limitaciones](../introduction/scope_and_limitations.md)).
- Al compilar, Vite avisa de que el *chunk* de ECharts supera los 500 kB. Ya está aislado en su propio archivo cacheable.
