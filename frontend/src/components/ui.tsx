import { animate, motion, useMotionValue, useTransform } from "framer-motion";
import { AlertTriangle, Loader2 } from "lucide-react";
import { useEffect, useId, type ReactNode } from "react";
import { cardItem } from "../lib/motion";

/* ----------------------------------------------------------- GlassCard */
interface GlassCardProps {
  title?: ReactNode;
  icon?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  className?: string;
  glow?: "cyan" | "violet" | "red" | "green" | "none";
  children: ReactNode;
}

const GLOW: Record<NonNullable<GlassCardProps["glow"]>, string> = {
  cyan: "",
  violet: "",
  red: "shadow-neon-red",
  green: "shadow-neon-green",
  none: "",
};

export function GlassCard({ title, icon, subtitle, actions, className = "", glow = "cyan", children }: GlassCardProps) {
  return (
    <motion.section variants={cardItem} className={`glass relative p-5 transition-shadow duration-300 ${GLOW[glow]} ${className}`}>
      {(title || actions) && (
        <header className="mb-4 flex items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            {icon && <div className="mt-0.5 rounded-lg border border-edge bg-night/60 p-2 text-engine">{icon}</div>}
            <div>
              {title && <h3 className="text-[15px] font-semibold text-slate-900">{title}</h3>}
              {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
            </div>
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      {children}
    </motion.section>
  );
}

/* ------------------------------------------------------ AnimatedNumber */
export function AnimatedNumber({ value, format, duration = 1.1 }: { value: number; format: (v: number) => string; duration?: number }) {
  const mv = useMotionValue(0);
  const text = useTransform(mv, (v) => format(v));
  useEffect(() => {
    if (!Number.isFinite(value)) return;
    const ctrl = animate(mv, value, { duration, ease: [0.16, 1, 0.3, 1] });
    return () => ctrl.stop();
  }, [value, duration, mv]);
  if (!Number.isFinite(value)) return <span>—</span>;
  return <motion.span>{text}</motion.span>;
}

/* ---------------------------------------------------------- MetricTile */
interface MetricTileProps {
  label: string;
  value: number | null | undefined;
  format: (v: number) => string;
  hint?: string;
  color?: string;
  icon?: ReactNode;
}

export function MetricTile({ label, value, format, hint, color = "#1F5A8A", icon }: MetricTileProps) {
  return (
    <motion.div variants={cardItem} className="glass relative overflow-hidden px-4 py-3.5">
      <div className="absolute inset-y-0 left-0 w-[3px]" style={{ background: color }} />
      <div className="flex items-center justify-between">
        <span className="label">{label}</span>
        {icon && <span style={{ color }}>{icon}</span>}
      </div>
      <div className="num mt-1.5 text-2xl font-semibold" style={{ color }}>
        {value === null || value === undefined ? "—" : <AnimatedNumber value={value} format={format} />}
      </div>
      {hint && <div className="mt-0.5 text-[11px] text-slate-500">{hint}</div>}
    </motion.div>
  );
}

/* --------------------------------------------------------------- Badge */
export function Badge({ color, children, pulse = false }: { color: string; children: ReactNode; pulse?: boolean }) {
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold"
      style={{ color, borderColor: `${color}55`, background: `${color}0f` }}
    >
      <span className="relative flex h-1.5 w-1.5">
        {pulse && <span className="soft-dot absolute -inset-1 inline-flex rounded-full" style={{ background: color }} />}
        <span className="relative inline-flex h-1.5 w-1.5 rounded-full" style={{ background: color }} />
      </span>
      {children}
    </span>
  );
}

/* --------------------------------------------------- Loading / errores */
export function LoadingBlock({ label = "Calculando…", height = 240 }: { label?: string; height?: number }) {
  return (
    <div className="flex items-center justify-center gap-2 text-sm text-slate-500" style={{ height }}>
      <Loader2 className="h-4 w-4 animate-spin text-engine" /> {label}
    </div>
  );
}

export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-malignant/40 bg-malignant/10 p-4 text-sm text-red-800">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="flex-1">
        <p className="font-semibold">No se pudo contactar con el motor numérico</p>
        <p className="mt-1 font-mono text-xs text-red-700/80">{message}</p>
        <p className="mt-1 text-xs text-red-700/70">¿Está corriendo el backend? → uvicorn app.main:app --port 8000</p>
      </div>
      {onRetry && (
        <button className="btn-ghost !py-1 text-xs" onClick={onRetry}>
          Reintentar
        </button>
      )}
    </div>
  );
}

/* --------------------------------------------------------- Form fields */
export function Field({ label, hint, children }: { label: string; hint?: ReactNode; children: ReactNode }) {
  return (
    <div className="block">
      <span className="label">{label}</span>
      <div className="mt-1.5">{children}</div>
      {hint && <span className="mt-1 block text-[11px] text-slate-500">{hint}</span>}
    </div>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  const pillId = useId(); // layoutId único por instancia: evita que la píldora salte entre controles
  return (
    <div className="relative flex rounded-md border border-edge bg-night p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={`relative z-10 flex-1 rounded-md px-2 py-1.5 text-xs font-medium transition ${
            value === o.value ? "text-engine" : "text-slate-500 hover:text-slate-800"
          }`}
        >
          {value === o.value && (
            <motion.span
              layoutId={`seg-${pillId}`}
              className="absolute inset-0 -z-10 rounded-md border border-engine/40 bg-white shadow-sm"
              transition={{ duration: 0.18, ease: "easeOut" }}
            />
          )}
          {o.label}
        </button>
      ))}
    </div>
  );
}
