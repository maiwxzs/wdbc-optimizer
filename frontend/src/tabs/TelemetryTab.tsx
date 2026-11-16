import { AnimatePresence, motion } from "framer-motion";
import { Activity, AlertOctagon, CheckCircle2, Clock, Cpu, Flame, GitCompareArrows, Hourglass, Play, Rocket, Sigma, Timer, Waves } from "lucide-react";
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { LogLineChart, type LineSeries } from "../components/charts/LogLineChart";
import { Badge, ErrorBox, Field, GlassCard, LoadingBlock, MetricTile, Segmented } from "../components/ui";
import { TeX } from "../components/TeX";
import { useAsync } from "../hooks/useAsync";
import { api } from "../lib/api";
import { COLORS, STOP_META, fromLog, pct, sci, toLog } from "../lib/format";
import { staggerContainer } from "../lib/motion";
import type { InitMode, NumericMode, StopReason, TrainResult } from "../lib/types";

const SCENARIO_COLORS = [COLORS.faint, COLORS.engine, COLORS.benign, COLORS.amber, COLORS.malignant, COLORS.accent];

const PRESETS: { label: string; alpha: number; mode: NumericMode; hint: string }[] = [
  { label: "Estable", alpha: 1, mode: "logits", hint: "α = 1 · LSE" },
  { label: "Rápido", alpha: 5, mode: "logits", hint: "α = 5 > 2/L" },
  { label: "Oscilante", alpha: 100, mode: "logits", hint: "α = 100" },
  { label: "Divergente", alpha: 1e5, mode: "logits", hint: "J > 1e8" },
  { label: "Sin estabilizar", alpha: 1000, mode: "naive", hint: "NaN / overflow" },
];

export function TelemetryTab({ onModelPromoted }: { onModelPromoted: () => void }) {
  const [alpha, setAlpha] = useState(1);
  const [maxIter, setMaxIter] = useState(8000);
  const [tolGradExp, setTolGradExp] = useState(-5);
  const [tolLossExp, setTolLossExp] = useState(-12);
  const [l2, setL2] = useState(1e-3);
  const [mode, setMode] = useState<NumericMode>("logits");
  const [init, setInit] = useState<InitMode>("zeros");
  const [promote, setPromote] = useState(false);

  const [result, setResult] = useState<TrainResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [runId, setRunId] = useState(0);

  const compare = useAsync(() => api.compare(3000), []);

  const run = useCallback(
    async (override?: { alpha: number; mode: NumericMode }) => {
      setRunning(true);
      setError(null);
      try {
        const r = await api.train({
          alpha: override?.alpha ?? alpha,
          mode: override?.mode ?? mode,
          max_iter: maxIter,
          tol_grad: fromLog(tolGradExp),
          tol_loss: fromLog(tolLossExp),
          l2,
          init,
          set_active: promote,
          max_points: 800,
        });
        setResult(r);
        setRunId((k) => k + 1);
        if (r.active_updated) onModelPromoted();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setRunning(false);
      }
    },
    [alpha, mode, maxIter, tolGradExp, tolLossExp, l2, init, promote, onModelPromoted],
  );

  // Ejecución inicial con la configuración por defecto
  useEffect(() => {
    void run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const alphaCrit = result?.alpha_critical ?? compare.data?.alpha_critical ?? null;
  const ratio = alphaCrit ? alpha / alphaCrit : null;

  const lossSeries = useMemo<LineSeries[]>(
    () => (result ? [{ name: "J(θ⁽ᵏ⁾)", color: COLORS.engine, x: result.history.iteration, y: result.history.loss, area: true }] : []),
    [result],
  );
  const gradSeries = useMemo<LineSeries[]>(
    () =>
      result
        ? [
            { name: "‖∇J‖₂", color: COLORS.accent, x: result.history.iteration, y: result.history.grad_norm },
            { name: "|ΔJ|", color: COLORS.amber, x: result.history.iteration, y: result.history.loss_change, width: 1.4 },
          ]
        : [],
    [result],
  );
  const compareSeries = useMemo<LineSeries[]>(
    () =>
      compare.data?.runs.map((r, i) => ({
        name: r.label,
        color: SCENARIO_COLORS[i % SCENARIO_COLORS.length],
        x: r.history.iteration,
        y: r.history.loss,
      })) ?? [],
    [compare.data],
  );

  const stop = result ? STOP_META[result.stop_reason] : null;

  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="grid grid-cols-12 gap-4">
      {/* ------------------------------------------------ Panel de control */}
      <div className="col-span-12 space-y-4 lg:col-span-4">
        <GlassCard title="Configuración del optimizador" subtitle="Gradiente descendente de paso fijo · NumPy" icon={<Cpu className="h-4 w-4" />}>
          <div className="space-y-4">
            <Field
              label={`Tasa de aprendizaje α = ${sci(alpha)}`}
              hint={
                ratio !== null && (
                  <span>
                    α / α<sub>crít</sub> ={" "}
                    <b style={{ color: ratio < 1 ? COLORS.benign : ratio < 200 ? COLORS.amber : COLORS.malignant }}>{sci(ratio)}</b> · α
                    <sub>crít</sub> = 2/L = {sci(alphaCrit)}
                  </span>
                )
              }
            >
              <input type="range" min={-3} max={6} step={0.05} value={toLog(alpha)} onChange={(e) => setAlpha(Number(fromLog(Number(e.target.value)).toPrecision(3)))} />
            </Field>

            <div className="flex flex-wrap gap-1.5">
              {PRESETS.map((p) => (
                <button
                  key={p.label}
                  className="rounded-lg border border-edge bg-night/60 px-2 py-1 text-[11px] text-slate-700 transition hover:border-engine/60 hover:text-engine"
                  title={p.hint}
                  onClick={() => {
                    setAlpha(p.alpha);
                    setMode(p.mode);
                    void run({ alpha: p.alpha, mode: p.mode });
                  }}
                >
                  {p.label}
                </button>
              ))}
            </div>

            <div className="grid grid-cols-2 gap-3">
              <Field label="max_iter">
                <input className="field" type="number" min={1} max={100000} value={maxIter} onChange={(e) => setMaxIter(Math.max(1, Number(e.target.value)))} />
              </Field>
              <Field label="λ (L2)">
                <select className="field" value={l2} onChange={(e) => setL2(Number(e.target.value))}>
                  {[0, 1e-4, 1e-3, 1e-2, 1e-1].map((v) => (
                    <option key={v} value={v}>{v === 0 ? "0 (sin reg.)" : v.toExponential(0)}</option>
                  ))}
                </select>
              </Field>
              <Field label={`tol ‖∇J‖ = 1e${tolGradExp}`}>
                <input type="range" min={-10} max={-2} step={1} value={tolGradExp} onChange={(e) => setTolGradExp(Number(e.target.value))} />
              </Field>
              <Field label={`tol |ΔJ| = 1e${tolLossExp}`}>
                <input type="range" min={-16} max={-4} step={1} value={tolLossExp} onChange={(e) => setTolLossExp(Number(e.target.value))} />
              </Field>
            </div>

            <Field label="Formulación numérica de J">
              <Segmented<NumericMode>
                value={mode}
                onChange={setMode}
                options={[
                  { value: "logits", label: "Log-Sum-Exp" },
                  { value: "clipped", label: "Clip ε" },
                  { value: "naive", label: "Ingenua" },
                ]}
              />
            </Field>
            <Field label="Inicialización θ⁽⁰⁾">
              <Segmented<InitMode>
                value={init}
                onChange={setInit}
                options={[
                  { value: "zeros", label: "Ceros" },
                  { value: "random", label: "𝒩(0, 0.5²)" },
                  { value: "trained", label: "Modelo activo" },
                ]}
              />
            </Field>

            <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-700">
              <input type="checkbox" checked={promote} onChange={(e) => setPromote(e.target.checked)} className="accent-[#1F5A8A]" />
              Promover el resultado al Panel Clínico (si no diverge)
            </label>

            <motion.button whileTap={{ scale: 0.97 }} className="btn-primary w-full" disabled={running} onClick={() => void run()}>
              {running ? <Hourglass className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {running ? "Optimizando…" : "Ejecutar optimización"}
            </motion.button>
          </div>
        </GlassCard>

        <GlassCard title="Condiciones de parada" subtitle="Evaluadas en cada iteración k, en este orden" icon={<Sigma className="h-4 w-4" />} glow="violet">
          <TeX block>{String.raw`\boldsymbol\theta^{(k+1)}=\boldsymbol\theta^{(k)}-\alpha\,\nabla J\big(\boldsymbol\theta^{(k)}\big),\qquad \boldsymbol\theta=[\boldsymbol w;\,b]`}</TeX>
          <ul className="mt-3 space-y-2">
            <StopItem active={result?.stop_reason} reason="divergence" icon={<Flame className="h-3.5 w-3.5" />} tex={String.raw`J^{(k)}\in\{\mathrm{NaN},\pm\infty\}\ \lor\ J^{(k)}>10^{8}`} label="Fallo numérico" />
            <StopItem active={result?.stop_reason} reason="grad_norm" icon={<CheckCircle2 className="h-3.5 w-3.5" />} tex={String.raw`\|\nabla J^{(k)}\|_2<\mathrm{tol}_{\nabla}`} label="Norma del gradiente" />
            <StopItem active={result?.stop_reason} reason="loss_change" icon={<Waves className="h-3.5 w-3.5" />} tex={String.raw`|J^{(k)}-J^{(k-1)}|<\mathrm{tol}_{J}`} label="Cambio de pérdida" />
            <StopItem active={result?.stop_reason} reason="max_iter" icon={<Clock className="h-3.5 w-3.5" />} tex={String.raw`k=k_{\max}`} label="Límite de iteraciones" />
          </ul>
          <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
            Convergencia garantizada si <TeX>{String.raw`0<\alpha<2/L`}</TeX> con <TeX>{String.raw`L=\tfrac{1}{4m}\lambda_{\max}(\tilde X^{\top}\tilde X)+\lambda`}</TeX>. Es una cota
            suficiente, no necesaria: lejos del peor caso σ′ = 1/4, la curvatura local es menor y pasos mayores también convergen.
          </p>
        </GlassCard>
      </div>

      {/* ------------------------------------------------ Resultados */}
      <div className="col-span-12 space-y-4 lg:col-span-8">
        {error && <ErrorBox message={error} onRetry={() => void run()} />}

        <AnimatePresence mode="wait">
          {result && stop && (
            <motion.div
              key={`banner-${runId}`}
              initial={{ opacity: 0, y: -10, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8 }}
              transition={{ type: "spring", stiffness: 260, damping: 22 }}
              className={`glass flex flex-col gap-2 p-4 md:flex-row md:items-center md:justify-between ${stop.tone === "bad" ? "shadow-neon-red" : stop.tone === "ok" ? "shadow-neon-green" : ""}`}
            >
              <div className="flex items-start gap-3">
                <div className="rounded-xl p-2" style={{ background: `${stop.color}22`, color: stop.color }}>
                  {stop.tone === "bad" ? <AlertOctagon className="h-5 w-5" /> : stop.tone === "ok" ? <Rocket className="h-5 w-5" /> : <Timer className="h-5 w-5" />}
                </div>
                <div>
                  <div className="font-semibold" style={{ color: stop.color }}>{stop.label}</div>
                  <div className="text-xs text-slate-500">{result.message}</div>
                  {result.divergence_detail && <div className="mt-1 font-mono text-xs text-red-700">↳ {result.divergence_detail}</div>}
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <Badge color={COLORS.engine}>α = {sci(result.config.alpha)}</Badge>
                <Badge color={COLORS.accent}>{result.config.mode}</Badge>
                {result.active_updated && <Badge color={COLORS.benign} pulse>Modelo clínico actualizado</Badge>}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <motion.div variants={staggerContainer} className="grid grid-cols-2 gap-3 md:grid-cols-5">
          <MetricTile label="Iteraciones" value={result?.iterations} format={(v) => Math.round(v).toLocaleString("es")} icon={<Activity className="h-4 w-4" />} />
          <MetricTile label="J final" value={result?.final_loss} format={(v) => sci(v, 4)} color={COLORS.engine} hint={result && result.final_loss === null ? "NaN / Inf" : "BCE + (λ/2)‖w‖²"} />
          <MetricTile label="‖∇J‖₂ final" value={result?.final_grad_norm} format={(v) => sci(v, 2)} color={COLORS.accent} />
          <MetricTile label="Exactitud test" value={result?.metrics.test.accuracy} format={(v) => pct(v)} color={COLORS.benign} />
          <MetricTile
            label="Overflow exp()"
            value={result?.numeric_flags.exp_overflow_events}
            format={(v) => Math.round(v).toLocaleString("es")}
            color={result && result.numeric_flags.exp_overflow_events > 0 ? COLORS.malignant : COLORS.faint}
            hint={result ? `${sci(result.elapsed_ms, 2)} ms · ${result.numeric_flags.gradient_evaluations} ∇J` : undefined}
          />
        </motion.div>

        <GlassCard title="Función objetivo J(θ⁽ᵏ⁾)" subtitle="Escala logarítmica · rueda del ratón para zoom" icon={<Activity className="h-4 w-4" />}>
          {running && !result ? (
            <LoadingBlock height={290} />
          ) : result ? (
            <LogLineChart key={`loss-${runId}`} series={lossSeries} height={290} yName="J" legend={false} />
          ) : null}
        </GlassCard>

        <GlassCard title="Norma del gradiente y cambio de pérdida" subtitle="Criterios de convergencia de primer orden y de estancamiento" icon={<Waves className="h-4 w-4" />} glow="violet">
          {result ? (
            <LogLineChart
              key={`grad-${runId}`}
              series={gradSeries}
              height={270}
              hLines={[
                { value: result.config.tol_grad, label: `tol∇ = ${sci(result.config.tol_grad, 0)}`, color: COLORS.accent },
                { value: result.config.tol_loss, label: `tolJ = ${sci(result.config.tol_loss, 0)}`, color: COLORS.amber },
              ]}
            />
          ) : (
            <LoadingBlock height={270} />
          )}
        </GlassCard>
      </div>

      {/* ------------------------------------------------ Comparativa */}
      <GlassCard
        className="col-span-12"
        title="Convergencia vs inestabilidad: barrido de escenarios"
        subtitle="Mismo problema, distintos α y formulaciones numéricas · θ⁽⁰⁾ = 0 · k_max = 3000"
        icon={<GitCompareArrows className="h-4 w-4" />}
        actions={
          <button className="btn-ghost !py-1.5 text-xs" disabled={compare.loading} onClick={() => void compare.run()}>
            <GitCompareArrows className="h-3.5 w-3.5" /> Re-ejecutar
          </button>
        }
      >
        {compare.error ? (
          <ErrorBox message={compare.error} />
        ) : !compare.data ? (
          <LoadingBlock height={340} />
        ) : (
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-12 xl:col-span-8">
              <LogLineChart series={compareSeries} height={360} yName="J" />
            </div>
            <div className="col-span-12 space-y-2 xl:col-span-4">
              {compare.data.runs.map((r, i) => (
                <motion.div
                  key={r.label}
                  initial={{ opacity: 0, x: 16 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.1 + i * 0.07 }}
                  className="rounded-xl border border-edge bg-night/50 px-3 py-2"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="flex items-center gap-2 text-xs font-semibold text-slate-800">
                      <span className="h-2 w-2 rounded-full" style={{ background: SCENARIO_COLORS[i % SCENARIO_COLORS.length] }} />
                      {r.label}
                    </span>
                    <Badge color={STOP_META[r.stop_reason].color}>{r.stop_reason}</Badge>
                  </div>
                  <div className="num mt-1 flex justify-between text-[11px] text-slate-500">
                    <span>k = {r.iterations}</span>
                    <span>J = {sci(r.final_loss, 3)}</span>
                    <span>acc = {pct(r.test_accuracy)}</span>
                    <span>α/α꜀ = {sci(r.alpha_over_critical, 2)}</span>
                  </div>
                  {r.divergence_detail && <div className="mt-0.5 truncate font-mono text-[10px] text-red-700/80" title={r.divergence_detail}>{r.divergence_detail}</div>}
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </GlassCard>
    </motion.div>
  );
}

function StopItem({ active, reason, icon, tex, label }: { active?: StopReason; reason: StopReason; icon: ReactNode; tex: string; label: string }) {
  const on = active === reason;
  const color = STOP_META[reason].color;
  return (
    <motion.li
      animate={{ borderColor: on ? `${color}aa` : COLORS.edge, backgroundColor: on ? `${color}10` : "#FFFFFF" }}
      transition={{ duration: 0.4 }}
      className="flex items-center gap-3 rounded-xl border px-3 py-2"
    >
      <span style={{ color: on ? color : COLORS.faint }}>{icon}</span>
      <div className="flex-1">
        <div className={`text-[11px] font-semibold ${on ? "text-slate-900" : "text-slate-500"}`}>{label}</div>
        <div className="text-xs"><TeX>{tex}</TeX></div>
      </div>
      {on && <span className="text-[10px] font-bold uppercase tracking-wider" style={{ color }}>activa</span>}
    </motion.li>
  );
}
