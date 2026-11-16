import { motion } from "framer-motion";
import { Binary, FlaskConical, Microscope, Play, Ruler, TriangleAlert } from "lucide-react";
import { useState } from "react";
import { FloatChart } from "../components/charts/FloatChart";
import { GradientCompareChart } from "../components/charts/GradientCompareChart";
import { VCurveChart } from "../components/charts/VCurveChart";
import { Badge, ErrorBox, Field, GlassCard, LoadingBlock, MetricTile, Segmented } from "../components/ui";
import { TeX } from "../components/TeX";
import { useAsync } from "../hooks/useAsync";
import { api } from "../lib/api";
import { COLORS, ieee, sci } from "../lib/format";
import { staggerContainer } from "../lib/motion";
import type { GradientCheck, InitMode, IeeeValue } from "../lib/types";

const VERDICT_COLOR: Record<GradientCheck["verdict"], string> = {
  excelente: COLORS.benign,
  aceptable: COLORS.engine,
  sospechoso: COLORS.amber,
  incorrecto: COLORS.malignant,
  fallo: COLORS.malignant,
};

const POINT_OPTS: { value: InitMode; label: string }[] = [
  { value: "random", label: "θ aleatorio" },
  { value: "zeros", label: "θ = 0" },
  { value: "trained", label: "θ* (óptimo)" },
];

export function AuditTab() {
  const [point, setPoint] = useState<InitMode>("random");
  const [hExp, setHExp] = useState(-5);
  const [seed, setSeed] = useState(0);
  const [sweepPoint, setSweepPoint] = useState<InitMode>("random");

  const gc = useAsync(() => api.gradientCheck({ point, h: Math.pow(10, hExp), seed }), [point, hExp, seed]);
  const sweep = useAsync(() => api.stepSweep({ point: sweepPoint, exp_min: -16, exp_max: 0, n: 49, seed: 0 }), [sweepPoint]);
  const fa = useAsync(() => api.floatAudit(), []);

  const g = gc.data;
  const vColor = g ? VERDICT_COLOR[g.verdict] : COLORS.faint;

  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="grid grid-cols-12 gap-4">
      {/* ------------------------------------------- Chequeo de gradiente */}
      <GlassCard
        className="col-span-12 lg:col-span-4"
        title="Chequeo de gradiente"
        subtitle="Analítico vs diferencias finitas centrales en los 31 parámetros"
        icon={<Microscope className="h-4 w-4" />}
      >
        <div className="space-y-4">
          <TeX block>{String.raw`\frac{\partial J}{\partial \theta_j}\approx\frac{J(\boldsymbol\theta+h\mathbf e_j)-J(\boldsymbol\theta-h\mathbf e_j)}{2h}`}</TeX>
          <Field label="Punto de evaluación θ">
            <Segmented<InitMode> value={point} onChange={setPoint} options={POINT_OPTS} />
          </Field>
          <div className="grid grid-cols-2 gap-3">
            <Field label={`Paso h = 1e${hExp}`}>
              <input type="range" min={-12} max={-1} step={1} value={hExp} onChange={(e) => setHExp(Number(e.target.value))} />
            </Field>
            <Field label="Semilla">
              <div className="flex gap-2">
                <input className="field" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} disabled={point !== "random"} />
                <button className="btn-ghost !px-2" title="Nueva semilla" disabled={point !== "random"} onClick={() => setSeed(Math.floor(Math.random() * 1e4))}>
                  <Play className="h-3.5 w-3.5" />
                </button>
              </div>
            </Field>
          </div>

          {gc.error ? (
            <ErrorBox message={gc.error} />
          ) : !g ? (
            <LoadingBlock height={150} />
          ) : (
            <motion.div key={`${g.point}-${g.h}-${seed}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="rounded-xl border p-4" style={{ borderColor: `${vColor}55`, background: `${vColor}0a` }}>
              <div className="label">Error relativo</div>
              <div className="num mt-1 text-3xl font-bold" style={{ color: vColor }}>
                {sci(g.relative_error, 3)}
              </div>
              <div className="mt-2 flex items-center gap-2">
                <Badge color={vColor} pulse>{g.verdict.toUpperCase()}</Badge>
                <span className="text-[11px] text-slate-500">{g.message}</span>
              </div>
              <div className="mt-3">
                <TeX block>{String.raw`\varepsilon_{\text{rel}}=\frac{\|\nabla_{\!a}-\nabla_{\!n}\|_2}{\|\nabla_{\!a}\|_2+\|\nabla_{\!n}\|_2+10^{-15}}`}</TeX>
              </div>
              <div className="num mt-2 grid grid-cols-2 gap-2 text-[11px] text-slate-500">
                <span>‖∇ₐ‖₂ = {sci(g.analytic_norm)}</span>
                <span>‖∇ₙ‖₂ = {sci(g.numeric_norm)}</span>
                <span>máx |Δ| = {sci(g.max_abs_error, 2)}</span>
                <span>J(θ) = {sci(g.loss_at_point, 5)}</span>
              </div>
              {g.point === "trained" && (
                <p className="mt-2 flex gap-1.5 text-[11px] text-amber-700/90">
                  <TriangleAlert className="h-3.5 w-3.5 shrink-0" />
                  En el óptimo ‖∇J‖ ≈ tol: el numerador queda dominado por el ruido de redondeo ε·J/h y el error relativo se infla. No indica un gradiente incorrecto.
                </p>
              )}
            </motion.div>
          )}
        </div>
      </GlassCard>

      <GlassCard
        className="col-span-12 lg:col-span-8"
        title="Gradiente por componente"
        subtitle="w₁…w₃₀ (características estandarizadas) y sesgo b · error absoluto en escala log"
        icon={<Binary className="h-4 w-4" />}
        glow="violet"
      >
        {g ? <GradientCompareChart data={g} height={430} /> : <LoadingBlock height={430} />}
      </GlassCard>

      {/* ------------------------------------------- Curva en V */}
      <GlassCard
        className="col-span-12 lg:col-span-8"
        title="Curva en «V»: truncamiento vs redondeo"
        subtitle="Error relativo del gradiente numérico frente al paso h (log-log, 49 pasos de 1e-16 a 1)"
        icon={<Ruler className="h-4 w-4" />}
        actions={<Segmented<InitMode> value={sweepPoint} onChange={setSweepPoint} options={POINT_OPTS} />}
      >
        {sweep.error ? <ErrorBox message={sweep.error} /> : sweep.data ? <VCurveChart data={sweep.data} height={400} /> : <LoadingBlock height={400} />}
      </GlassCard>

      <div className="col-span-12 space-y-4 lg:col-span-4">
        <motion.div variants={staggerContainer} className="grid grid-cols-2 gap-3">
          <MetricTile label="h* central (emp.)" value={sweep.data?.central_best.h} format={(v) => sci(v, 1)} color={COLORS.engine} hint={`teoría ε^⅓ = ${sci(sweep.data?.central_h_opt_theory, 1)}`} />
          <MetricTile label="error mín. central" value={sweep.data?.central_best.error} format={(v) => sci(v, 1)} color={COLORS.engine} />
          <MetricTile label="h* progresiva (emp.)" value={sweep.data?.forward_best.h} format={(v) => sci(v, 1)} color={COLORS.accent} hint={`teoría √ε = ${sci(sweep.data?.forward_h_opt_theory, 1)}`} />
          <MetricTile label="error mín. progresiva" value={sweep.data?.forward_best.error} format={(v) => sci(v, 1)} color={COLORS.accent} />
        </motion.div>
        <GlassCard title="Modelo de error total" icon={<FlaskConical className="h-4 w-4" />} glow="violet">
          <TeX block>{String.raw`E_{\text{central}}(h)\approx\underbrace{\tfrac{h^{2}}{6}|J'''|}_{\text{truncamiento}}+\underbrace{\tfrac{\varepsilon_{\text{mach}}|J|}{h}}_{\text{redondeo}}`}</TeX>
          <TeX block>{String.raw`E_{\text{prog}}(h)\approx\tfrac{h}{2}|J''|+\tfrac{2\varepsilon_{\text{mach}}|J|}{h}`}</TeX>
          <TeX block>{String.raw`h^{*}_{\text{central}}\sim\varepsilon_{\text{mach}}^{1/3},\qquad h^{*}_{\text{prog}}\sim\varepsilon_{\text{mach}}^{1/2}`}</TeX>
          <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
            Rama izquierda (pendiente −1): la resta J(θ+h)−J(θ−h) sufre <b className="text-slate-800">cancelación catastrófica</b>. Rama derecha: domina el error de
            truncamiento de Taylor (pendiente +2 central, +1 progresiva). Con ε<sub>mach</sub> = {sci(sweep.data?.eps_machine, 3)}, las líneas discontinuas son el modelo ajustado a
            cada rama.
          </p>
        </GlassCard>
      </div>

      {/* ------------------------------------------- Error flotante */}
      <GlassCard
        className="col-span-12 lg:col-span-7"
        title="Laboratorio de error de punto flotante"
        subtitle="Pérdida por muestra con y = 1 · los huecos de la curva roja son NaN/±Inf"
        icon={<TriangleAlert className="h-4 w-4" />}
        glow="red"
      >
        {fa.error ? <ErrorBox message={fa.error} /> : fa.data ? <FloatChart data={fa.data} height={380} /> : <LoadingBlock height={380} />}
        <div className="mt-2 grid gap-2 md:grid-cols-3">
          <Formula color={COLORS.benign} title="Log-Sum-Exp (motor)" tex={String.raw`\ell=\log(1+e^{z})-yz`} />
          <Formula color={COLORS.engine} title="Acotada" tex={String.raw`\hat y\leftarrow\mathrm{clip}(\hat y,\varepsilon,1-\varepsilon)`} />
          <Formula color={COLORS.malignant} title="Ingenua" tex={String.raw`\sigma(z)=\tfrac{1}{1+e^{-z}}`} />
        </div>
      </GlassCard>

      <GlassCard className="col-span-12 lg:col-span-5" title="Valores especiales IEEE-754" subtitle="Sondeo de la sigmoide y la pérdida en puntos críticos" icon={<Binary className="h-4 w-4" />}>
        {!fa.data ? (
          <LoadingBlock height={380} />
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-[11px]">
                <thead>
                  <tr className="text-left text-slate-500">
                    <th className="py-1.5 pr-2 font-medium">z</th>
                    <th className="pr-2 font-medium">σ ingenua</th>
                    <th className="pr-2 font-medium">σ estable</th>
                    <th className="pr-2 font-medium">ℓ₁ ingenua</th>
                    <th className="pr-2 font-medium">ℓ₁ clip</th>
                    <th className="font-medium">ℓ₁ exacta</th>
                  </tr>
                </thead>
                <tbody className="num">
                  {fa.data.table.map((r, i) => (
                    <motion.tr key={r.z} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 + i * 0.04 }} className="border-t border-edge/70">
                      <td className="py-1.5 pr-2 text-slate-700">{r.z}</td>
                      <Cell v={r.sigmoid_naive} />
                      <Cell v={r.sigmoid_stable} />
                      <Cell v={r.loss_y1_naive} />
                      <Cell v={r.loss_y1_clip} warn={typeof r.loss_y1_clip === "number" && typeof r.loss_y1_exact === "number" && Math.abs(r.loss_y1_clip - r.loss_y1_exact) / Math.max(r.loss_y1_exact, 1e-300) > 1e-3} />
                      <Cell v={r.loss_y1_exact} good />
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
            <ul className="mt-3 space-y-1.5">
              {fa.data.notes.map((n) => (
                <li key={n} className="flex gap-2 text-[11px] leading-relaxed text-slate-500">
                  <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-accent" /> {n}
                </li>
              ))}
            </ul>
          </>
        )}
      </GlassCard>
    </motion.div>
  );
}

function Cell({ v, good, warn }: { v: IeeeValue; good?: boolean; warn?: boolean }) {
  const special = typeof v !== "number";
  const exact0or1 = v === 0 || v === 1;
  const color = special ? COLORS.malignant : warn ? COLORS.amber : exact0or1 && !good ? COLORS.amber : good ? COLORS.benign : COLORS.text;
  return (
    <td className="pr-2" style={{ color, fontWeight: special ? 700 : 400 }} title={exact0or1 ? "Valor saturado: redondeado exactamente" : undefined}>
      {ieee(v)}
    </td>
  );
}

function Formula({ color, title, tex }: { color: string; title: string; tex: string }) {
  return (
    <div className="rounded-xl border bg-night/50 px-3 py-2" style={{ borderColor: `${color}55` }}>
      <div className="text-[10px] font-semibold uppercase tracking-wider" style={{ color }}>{title}</div>
      <div className="text-xs"><TeX>{tex}</TeX></div>
    </div>
  );
}
