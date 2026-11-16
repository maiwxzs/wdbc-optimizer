import { COLORS } from "./format";

/** Opciones ECharts comunes al tema claro y sobrio. */
export const baseChart = {
  backgroundColor: "transparent",
  textStyle: { fontFamily: "Inter, system-ui, sans-serif", color: COLORS.muted },
  grid: { left: 64, right: 24, top: 36, bottom: 48, containLabel: false },
  tooltip: {
    trigger: "axis" as const,
    backgroundColor: "rgba(255,255,255,0.97)",
    borderColor: COLORS.edge,
    textStyle: { color: COLORS.text, fontFamily: "'JetBrains Mono', monospace", fontSize: 12 },
    extraCssText: "border-radius: 6px; box-shadow: 0 2px 8px rgba(16,24,40,.08);",
  },
  legend: { textStyle: { color: COLORS.muted }, top: 0, icon: "roundRect", itemWidth: 14, itemHeight: 6 },
};

export const axisStyle = {
  axisLine: { lineStyle: { color: COLORS.edge } },
  axisTick: { lineStyle: { color: COLORS.edge } },
  axisLabel: { color: COLORS.muted, fontFamily: "'JetBrains Mono', monospace", fontSize: 11 },
  splitLine: { lineStyle: { color: "#E7EAEE", type: "dashed" as const } },
  nameTextStyle: { color: COLORS.muted, fontSize: 11 },
};

/** Etiqueta de eje logarítmico compacta: 1e-6, 1e+3… */
export const logLabel = (v: number) => (v === 0 ? "0" : v.toExponential(0).replace("e+0", "e").replace("e-0", "e-"));

/** Gradiente vertical para áreas bajo líneas. */
export const areaGradient = (hex: string) => ({
  type: "linear" as const,
  x: 0,
  y: 0,
  x2: 0,
  y2: 1,
  colorStops: [
    { offset: 0, color: hex + "22" },
    { offset: 1, color: hex + "00" },
  ],
});

/** Línea sólida, sin halo (se conserva el nombre por compatibilidad). */
export const neonLine = (hex: string, width = 2) => ({
  width,
  color: hex,
});
