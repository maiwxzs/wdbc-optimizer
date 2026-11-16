import type { EChartsOption } from "echarts";
import { EChart } from "./EChart";
import { AnimatePresence, motion } from "framer-motion";
import { useMemo } from "react";
import { COLORS } from "../../lib/format";

/** Mezcla lineal de dos colores hex (verde → rojo según la probabilidad). */
function mix(a: string, b: string, t: number) {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  const c = pa.map((v, i) => Math.round(v + (pb[i] - v) * t));
  return `#${c.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}

export function riskColor(p: number) {
  // Verde puro hasta 0.3, transición a ámbar en 0.5, rojo carmesí desde 0.7
  if (p < 0.3) return COLORS.benign;
  if (p < 0.5) return mix(COLORS.benign, COLORS.amber, (p - 0.3) / 0.2);
  if (p < 0.7) return mix(COLORS.amber, COLORS.malignant, (p - 0.5) / 0.2);
  return COLORS.malignant;
}

/** Gauge radial clínico: aguja elástica + anillo de pulso neón si p ≥ 0.5. */
export function RiskGauge({ probability, height = 300 }: { probability: number; height?: number }) {
  const color = riskColor(probability);
  const alert = probability >= 0.5;

  const option = useMemo<EChartsOption>(
    () => ({
      backgroundColor: "transparent",
      series: [
        {
          type: "gauge",
          startAngle: 215,
          endAngle: -35,
          min: 0,
          max: 100,
          splitNumber: 10,
          radius: "90%",
          center: ["50%", "55%"],
          progress: {
            show: true,
            width: 14,
            roundCap: true,
            itemStyle: { color },
          },
          axisLine: {
            roundCap: true,
            lineStyle: {
              width: 14,
              color: [
                [0.3, "rgba(46,125,91,0.14)"],
                [0.5, "rgba(183,121,31,0.14)"],
                [1, "rgba(178,58,46,0.14)"],
              ],
            },
          },
          pointer: {
            length: "64%",
            width: 6,
            offsetCenter: [0, "4%"],
            itemStyle: { color },
          },
          anchor: { show: true, size: 18, showAbove: true, itemStyle: { color: COLORS.night, borderColor: color, borderWidth: 3 } },
          axisTick: { distance: -24, length: 5, lineStyle: { color: "#B8C0C9", width: 1 } },
          splitLine: { distance: -28, length: 12, lineStyle: { color: "#9AA5B1", width: 2 } },
          axisLabel: {
            distance: -6,
            color: COLORS.muted,
            fontSize: 10,
            fontFamily: "'JetBrains Mono', monospace",
            formatter: (v: number) => (v % 20 === 0 ? `${v}` : ""),
          },
          title: { offsetCenter: [0, "74%"], color: COLORS.muted, fontSize: 12 },
          detail: {
            valueAnimation: true,
            offsetCenter: [0, "44%"],
            fontSize: 32,
            fontWeight: 700,
            fontFamily: "'JetBrains Mono', monospace",
            color,
            formatter: (v: number) => `${v.toFixed(1)}%`,
          },
          data: [{ value: Math.round(probability * 1000) / 10, name: "P(maligno | x)" }],
          animationDuration: 1800,
          animationEasing: "elasticOut",
          animationDurationUpdate: 1800,
          animationEasingUpdate: "elasticOut",
        },
      ],
    }) as EChartsOption,
    [probability, color],
  );

  return (
    <div className="relative" style={{ height }}>
      <AnimatePresence>
        {alert && (
          <motion.div
            key="ring"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="pointer-events-none absolute left-1/2 top-[55%] -translate-x-1/2 -translate-y-1/2"
            style={{ width: height * 0.82, height: height * 0.82 }}
          >
            <div className="pulse-ring absolute inset-0 rounded-full border-2" style={{ borderColor: COLORS.malignant }} />
            <div
              className="pulse-ring absolute inset-0 rounded-full border"
              style={{ borderColor: COLORS.malignant, animationDelay: "0.7s" }}
            />
          </motion.div>
        )}
      </AnimatePresence>
      <EChart option={option} style={{ height, width: "100%" }} />
    </div>
  );
}
