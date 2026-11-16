import type { IeeeValue, Num, StopReason } from "./types";

export const COLORS = {
  // Paleta clara y sobria (tinta oscura sobre papel, acentos desaturados).
  night: "#FFFFFF",
  edge: "#DDE1E6",
  benign: "#2E7D5B",
  malignant: "#B23A2E",
  engine: "#1F5A8A",
  accent: "#5E4B8B",
  amber: "#B7791F",
  text: "#1F2933",
  muted: "#5F6B76",
  faint: "#9AA5B1",
};

export function sci(v: Num | undefined, digits = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e-3 && a < 1e4) return v.toPrecision(digits + 1).replace(/\.?0+$/, "");
  return v.toExponential(digits);
}

export function pct(v: Num | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function ieee(v: IeeeValue): string {
  return typeof v === "number" ? sci(v, 4) : v;
}

export const STOP_META: Record<StopReason, { label: string; color: string; tone: "ok" | "warn" | "bad" }> = {
  grad_norm: { label: "Convergió · ‖∇J‖₂ < tol", color: COLORS.benign, tone: "ok" },
  loss_change: { label: "Convergió · |ΔJ| < tol", color: COLORS.benign, tone: "ok" },
  max_iter: { label: "Límite de iteraciones", color: COLORS.amber, tone: "warn" },
  divergence: { label: "Divergencia / fallo numérico", color: COLORS.malignant, tone: "bad" },
};

/** Log10 → valor para sliders en escala logarítmica. */
export const fromLog = (x: number) => Math.pow(10, x);
export const toLog = (x: number) => Math.log10(x);
