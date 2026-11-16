import { motion } from "framer-motion";
import { Crosshair, Dice5, Gauge, HeartPulse, RotateCcw, ScanLine, Shuffle, SlidersHorizontal, Target, TriangleAlert } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ContributionsChart } from "../components/charts/ContributionsChart";
import { PCAScatter } from "../components/charts/PCAScatter";
import { RiskGauge, riskColor } from "../components/charts/RiskGauge";
import { Badge, ErrorBox, GlassCard, LoadingBlock, MetricTile } from "../components/ui";
import { TeX } from "../components/TeX";
import { useAsync } from "../hooks/useAsync";
import { api } from "../lib/api";
import { COLORS, STOP_META, pct, sci } from "../lib/format";
import { cardItem, staggerContainer } from "../lib/motion";
import type { Confusion, FeatureStat, PCAPoints, Prediction } from "../lib/types";

const pid = (i: number) => `P-${String(i + 1).padStart(3, "0")}`;

export function ClinicalTab({ modelVersion }: { modelVersion: number }) {
  const summary = useAsync(() => api.summary(), []);
  const model = useAsync(() => api.model(), [modelVersion]);
  const pca = useAsync(() => api.pca(), [modelVersion]);

  const [selected, setSelected] = useState<number | null>(null);
  const [features, setFeatures] = useState<number[] | null>(null);
  const [modified, setModified] = useState(false);
  const [pred, setPred] = useState<Prediction | null>(null);
  const [predErr, setPredErr] = useState<string | null>(null);
  const reqId = useRef(0);

  // Paciente inicial: el primer maligno del conjunto de prueba
  useEffect(() => {
    if (selected === null && pca.data) setSelected(pca.data.test.find((p) => p.label === 1)?.i ?? 0);
  }, [pca.data, selected]);

  // Predicción del paciente seleccionado (valores reales del conjunto de prueba)
  useEffect(() => {
    if (selected === null) return;
    const id = ++reqId.current;
    api
      .predictSample(selected)
      .then((p) => {
        if (id !== reqId.current) return;
        setPred(p);
        setFeatures(p.raw_features);
        setModified(false);
        setPredErr(null);
      })
      .catch((e) => id === reqId.current && setPredErr(String(e.message ?? e)));
  }, [selected, modelVersion]);

  // Simulación "what-if": re-predicción con debounce al mover los sliders
  useEffect(() => {
    if (!modified || !features) return;
    const id = ++reqId.current;
    const t = window.setTimeout(() => {
      api
        .predictFeatures(features)
        .then((p) => id === reqId.current && (setPred(p), setPredErr(null)))
        .catch((e) => id === reqId.current && setPredErr(String(e.message ?? e)));
    }, 220);
    return () => window.clearTimeout(t);
  }, [features, modified]);

  const meanFeatures = useMemo(() => summary.data?.features.filter((f) => f.group === "mean") ?? [], [summary.data]);

  const pick = useCallback(
    (kind: "benign" | "malignant" | "border" | "error") => {
      const t = pca.data?.test ?? [];
      let pool = t;
      if (kind === "benign") pool = t.filter((p) => p.label === 0);
      if (kind === "malignant") pool = t.filter((p) => p.label === 1);
      if (kind === "error") pool = t.filter((p) => p.pred !== p.label);
      if (kind === "border") {
        const best = [...t].sort((a, b) => Math.abs(a.proba - 0.5) - Math.abs(b.proba - 0.5))[0];
        if (best) setSelected(best.i);
        return;
      }
      if (pool.length) setSelected(pool[Math.floor(Math.random() * pool.length)].i);
    },
    [pca.data],
  );

  const err = summary.error || model.error || pca.error;
  if (err) return <ErrorBox message={err} onRetry={() => (summary.run(), model.run(), pca.run())} />;

  const test = model.data?.metrics.test;
  const s = model.data?.summary;
  const nErrors = pca.data?.test.filter((p) => p.pred !== p.label).length ?? 0;

  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="grid grid-cols-12 gap-4">
      {/* ---------- KPIs del modelo activo ---------- */}
      <div className="col-span-12 grid grid-cols-2 gap-3 md:grid-cols-5">
        <MetricTile label="Exactitud (test)" value={test?.accuracy} format={(v) => pct(v)} color={COLORS.engine} icon={<Target className="h-4 w-4" />} hint={`${summary.data?.n_test ?? "—"} pacientes no vistos`} />
        <MetricTile label="Sensibilidad" value={test?.sensitivity} format={(v) => pct(v)} color={COLORS.malignant} icon={<HeartPulse className="h-4 w-4" />} hint="malignos detectados (TP / P)" />
        <MetricTile label="Especificidad" value={test?.specificity} format={(v) => pct(v)} color={COLORS.benign} icon={<ScanLine className="h-4 w-4" />} hint="benignos descartados (TN / N)" />
        <MetricTile label="AUC-ROC" value={test?.auc} format={(v) => v.toFixed(4)} color={COLORS.accent} icon={<Gauge className="h-4 w-4" />} hint="Mann-Whitney U (NumPy)" />
        <MetricTile label="F1-score" value={test?.f1} format={(v) => v.toFixed(4)} color="#3B6E8F" icon={<Crosshair className="h-4 w-4" />} hint={`umbral de decisión p ≥ 0.5`} />
      </div>

      {/* ---------- Selector de paciente + simulación ---------- */}
      <GlassCard
        className="col-span-12 lg:col-span-4"
        title="Simulación de diagnóstico"
        subtitle="Seleccione un paciente del conjunto de prueba y modifique sus rasgos morfológicos"
        icon={<SlidersHorizontal className="h-4 w-4" />}
      >
        {!pca.data || !summary.data ? (
          <LoadingBlock />
        ) : (
          <>
            <select className="field" value={selected ?? 0} onChange={(e) => setSelected(Number(e.target.value))}>
              {pca.data.test.map((p) => (
                <option key={p.i} value={p.i}>
                  {pid(p.i)} · real: {p.label ? "Maligno" : "Benigno"} · p̂ = {pct(p.proba)}
                  {p.pred !== p.label ? "  ⚠" : ""}
                </option>
              ))}
            </select>
            <div className="mt-2.5 grid grid-cols-2 gap-2">
              <button className="btn-ghost !px-2 !py-1.5 text-xs" onClick={() => pick("benign")}>
                <Dice5 className="h-3.5 w-3.5 text-benign" /> Benigno aleatorio
              </button>
              <button className="btn-ghost !px-2 !py-1.5 text-xs" onClick={() => pick("malignant")}>
                <Dice5 className="h-3.5 w-3.5 text-malignant" /> Maligno aleatorio
              </button>
              <button className="btn-ghost !px-2 !py-1.5 text-xs" onClick={() => pick("border")}>
                <Shuffle className="h-3.5 w-3.5 text-amber-600" /> Caso frontera
              </button>
              <button className="btn-ghost !px-2 !py-1.5 text-xs" disabled={!nErrors} onClick={() => pick("error")}>
                <TriangleAlert className="h-3.5 w-3.5 text-accent" /> Error del modelo ({nErrors})
              </button>
            </div>

            <div className="mt-4 flex items-center justify-between">
              <span className="label">Rasgos (media del núcleo celular)</span>
              {modified && (
                <button className="flex items-center gap-1 text-[11px] text-engine hover:text-engine" onClick={() => selected !== null && api.predictSample(selected).then((p) => (setPred(p), setFeatures(p.raw_features), setModified(false)))}>
                  <RotateCcw className="h-3 w-3" /> restaurar
                </button>
              )}
            </div>
            <div className="mt-2 max-h-[400px] space-y-3 overflow-y-auto pr-1">
              {features &&
                meanFeatures.map((f) => (
                  <FeatureSlider
                    key={f.index}
                    stat={f}
                    value={features[f.index]}
                    onChange={(v) => {
                      setFeatures((prev) => (prev ? prev.map((x, j) => (j === f.index ? v : x)) : prev));
                      setModified(true);
                    }}
                  />
                ))}
            </div>
            <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
              Los 20 rasgos restantes (error estándar y "peor") se mantienen en los valores del paciente. Las marcas{" "}
              <span className="text-benign">▲</span>/<span className="text-malignant">▲</span> indican la media de cada clase en entrenamiento.
            </p>
          </>
        )}
      </GlassCard>

      {/* ---------- Gauge de riesgo ---------- */}
      <GlassCard
        className="col-span-12 md:col-span-6 lg:col-span-4"
        glow={pred && pred.probability_malignant >= 0.5 ? "red" : "green"}
        title="Riesgo estimado de malignidad"
        subtitle={selected !== null ? `${pid(selected)}${modified ? " · escenario simulado" : ""}` : undefined}
        icon={<Gauge className="h-4 w-4" />}
      >
        {predErr && <ErrorBox message={predErr} />}
        {!pred ? (
          <LoadingBlock height={300} />
        ) : (
          <>
            <RiskGauge probability={pred.probability_malignant} />
            <div className="-mt-2 flex flex-wrap items-center justify-center gap-2">
              <Badge color={riskColor(pred.probability_malignant)} pulse={pred.probability_malignant >= 0.5}>
                {pred.prediction_label} · {pred.risk_band.label}
              </Badge>
              {pred.true_label !== null && !modified && (
                <Badge color={pred.true_label === pred.prediction ? COLORS.benign : COLORS.accent}>
                  Diagnóstico real: {pred.true_label ? "Maligno" : "Benigno"} {pred.true_label === pred.prediction ? "✓" : "✗"}
                </Badge>
              )}
            </div>
            <div className="mt-4 rounded-xl border border-edge bg-night/60 p-3 text-center">
              <TeX block>{String.raw`\hat p=\sigma\!\left(\boldsymbol w^{\top}\tilde{\boldsymbol x}+b\right)=\sigma(${pred.logit.toFixed(3)})=${pred.probability_malignant.toFixed(4)}`}</TeX>
              <p className="mt-1 text-[11px] text-slate-500">
                x̃ = características estandarizadas (z-score con μ, σ de entrenamiento) · b = {sci(pred.bias)}
              </p>
            </div>
          </>
        )}
      </GlassCard>

      {/* ---------- Contribuciones ---------- */}
      <GlassCard
        className="col-span-12 md:col-span-6 lg:col-span-4"
        title="Explicabilidad del logit"
        subtitle="Contribución aditiva wⱼ·x̃ⱼ de los 10 rasgos más influyentes"
        icon={<ScanLine className="h-4 w-4" />}
        glow="violet"
      >
        {pred ? <ContributionsChart items={pred.contributions} height={380} /> : <LoadingBlock height={380} />}
        <div className="mt-1 flex justify-center gap-4 text-[11px] text-slate-500">
          <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm bg-malignant" /> empuja a maligno</span>
          <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm bg-benign" /> empuja a benigno</span>
        </div>
      </GlassCard>

      {/* ---------- PCA ---------- */}
      <GlassCard
        className="col-span-12 lg:col-span-7"
        title="Espacio de pacientes (proyección PCA 2D)"
        subtitle="Clic en un punto para cargar ese paciente · el modelo decide en ℝ³⁰, PCA es solo visual"
        icon={<Crosshair className="h-4 w-4" />}
      >
        {pca.data && selected !== null ? (
          <PCAScatter data={pca.data as PCAPoints} selected={selected} current={modified ? pred?.pca : null} onSelect={setSelected} height={360} />
        ) : (
          <LoadingBlock height={360} />
        )}
      </GlassCard>

      {/* ---------- Matriz de confusión + modelo ---------- */}
      <motion.div variants={cardItem} className="col-span-12 grid gap-4 lg:col-span-5">
        <GlassCard title="Matriz de confusión (test)" subtitle="Clase positiva = maligno" icon={<Target className="h-4 w-4" />}>
          {test ? <ConfusionMatrix c={test.confusion} /> : <LoadingBlock height={160} />}
        </GlassCard>
        <GlassCard title="Modelo activo" subtitle="Solución numérica de min J(w, b)" icon={<HeartPulse className="h-4 w-4" />} glow="violet">
          {s ? (
            <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
              <KV k="Parada" v={<span style={{ color: STOP_META[s.stop_reason].color }}>{STOP_META[s.stop_reason].label}</span>} />
              <KV k="Iteraciones" v={s.iterations.toLocaleString("es")} />
              <KV k="α · λ (L2)" v={`${sci(s.config.alpha)} · ${sci(s.config.l2)}`} />
              <KV k="J(θ*)" v={sci(s.final_loss, 5)} />
              <KV k="‖∇J(θ*)‖₂" v={sci(s.final_grad_norm)} />
              <KV k="Formulación" v={s.config.mode} />
            </div>
          ) : (
            <LoadingBlock height={100} />
          )}
        </GlassCard>
      </motion.div>
    </motion.div>
  );
}

function KV({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className="flex flex-col rounded-lg border border-edge/70 bg-night/50 px-2.5 py-1.5">
      <span className="text-[10px] uppercase tracking-wider text-slate-500">{k}</span>
      <span className="num mt-0.5 text-slate-900">{v}</span>
    </div>
  );
}

function FeatureSlider({ stat, value, onChange }: { stat: FeatureStat; value: number; onChange: (v: number) => void }) {
  const span = stat.max - stat.min || 1;
  const pos = (v: number) => `${((v - stat.min) / span) * 100}%`;
  const z = (value - stat.mean) / (stat.std || 1);
  return (
    <div>
      <div className="flex items-baseline justify-between text-xs">
        <span className="text-slate-700">{stat.name_es.replace(" (media)", "")}</span>
        <span className="num text-slate-900">
          {sci(value)} <span className={`ml-1 text-[10px] ${Math.abs(z) > 2 ? "text-amber-600" : "text-slate-500"}`}>z={z >= 0 ? "+" : ""}{z.toFixed(2)}</span>
        </span>
      </div>
      <div className="relative mt-1">
        <input type="range" min={stat.min} max={stat.max} step={span / 400} value={value} onChange={(e) => onChange(Number(e.target.value))} />
        <span className="pointer-events-none absolute -bottom-2.5 -translate-x-1/2 text-[8px] text-benign" style={{ left: pos(stat.mean_benign) }}>▲</span>
        <span className="pointer-events-none absolute -bottom-2.5 -translate-x-1/2 text-[8px] text-malignant" style={{ left: pos(stat.mean_malignant) }}>▲</span>
      </div>
    </div>
  );
}

function ConfusionMatrix({ c }: { c: Confusion }) {
  const total = c.tp + c.tn + c.fp + c.fn;
  const cells = [
    { k: "VP", v: c.tp, color: COLORS.malignant, desc: "Maligno → Maligno" },
    { k: "FN", v: c.fn, color: COLORS.accent, desc: "Maligno → Benigno" },
    { k: "FP", v: c.fp, color: COLORS.amber, desc: "Benigno → Maligno" },
    { k: "VN", v: c.tn, color: COLORS.benign, desc: "Benigno → Benigno" },
  ];
  return (
    <div className="grid grid-cols-[auto_1fr_1fr] gap-2 text-xs">
      <span />
      <span className="label text-center">Pred. maligno</span>
      <span className="label text-center">Pred. benigno</span>
      {[0, 1].map((row) => (
        <div key={row} className="contents">
          <span className="label flex items-center pr-1">{row === 0 ? "Real maligno" : "Real benigno"}</span>
          {cells.slice(row * 2, row * 2 + 2).map((cell, i) => (
            <motion.div
              key={cell.k}
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.15 + (row * 2 + i) * 0.08, type: "spring", stiffness: 260, damping: 18 }}
              className="rounded-xl border p-3 text-center"
              style={{ borderColor: `${cell.color}55`, background: `${cell.color}${Math.round(20 + (cell.v / total) * 90).toString(16).padStart(2, "0")}` }}
              title={cell.desc}
            >
              <div className="num text-2xl font-bold" style={{ color: cell.color }}>{cell.v}</div>
              <div className="text-[10px] text-slate-500">{cell.k} · {cell.desc}</div>
            </motion.div>
          ))}
        </div>
      ))}
    </div>
  );
}
