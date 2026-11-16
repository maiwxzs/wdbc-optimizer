import type { EChartsOption } from "echarts";
import { EChart } from "./EChart";
import { useEffect, useMemo, useState } from "react";
import { axisStyle, baseChart, logLabel } from "../../lib/chartTheme";
import { COLORS, sci } from "../../lib/format";
import type { Num, StepSweep } from "../../lib/types";

const TOP = 10; // techo del eje log desde el que "caen" los puntos

/**
 * Curva en "V" del error de diferencias finitas (log-log). Los puntos se
 * muestran primero en el techo del eje y caen rebotando (bounceOut) sobre su
 * valor real, escalonados por índice.
 */
export function VCurveChart({ data, height = 380 }: { data: StepSweep; height?: number }) {
  const [landed, setLanded] = useState(false);
  useEffect(() => {
    setLanded(false);
    const t = window.setTimeout(() => setLanded(true), 120);
    return () => window.clearTimeout(t);
  }, [data]);

  const option = useMemo<EChartsOption>(() => {
    const pts = (ys: Num[]) =>
      data.h.map((h, i) => {
        const y = ys[i];
        const ok = y !== null && Number.isFinite(y) && y > 0;
        return [h, landed ? (ok ? y : null) : TOP];
      });
    const line = (ys: Num[]) => data.h.map((h, i) => [h, ys[i] !== null && (ys[i] as number) > 0 ? ys[i] : null]);
    const vline = (x: number, label: string, color: string) => ({
      xAxis: x,
      label: { formatter: label, color, fontSize: 10, position: "insideEndTop" as const },
      lineStyle: { color, type: "dashed" as const, width: 1.2 },
    });

    return {
      ...baseChart,
      grid: { left: 64, right: 24, top: 44, bottom: 52 },
      legend: { ...baseChart.legend, data: ["Central O(h²)", "Progresiva O(h)", "Modelo central", "Modelo progresiva"] },
      tooltip: {
        ...baseChart.tooltip,
        trigger: "item",
        formatter: (p: unknown) => {
          const { seriesName, value } = p as { seriesName: string; value: [number, number] };
          return `${seriesName}<br/>h = ${sci(value[0], 2)}<br/>error rel. = ${sci(value[1], 3)}`;
        },
      },
      xAxis: { type: "log", name: "paso h", nameLocation: "middle", nameGap: 30, ...axisStyle, axisLabel: { ...axisStyle.axisLabel, formatter: logLabel } },
      yAxis: { type: "log", name: "error relativo", max: TOP, ...axisStyle, axisLabel: { ...axisStyle.axisLabel, formatter: logLabel } },
      animationDuration: 600,
      animationDurationUpdate: 1100,
      animationEasingUpdate: "bounceOut",
      animationDelayUpdate: (i: number) => i * 28,
      series: [
        {
          id: "central",
          name: "Central O(h²)",
          type: "scatter",
          symbolSize: 8,
          data: pts(data.central_error),
          itemStyle: { color: COLORS.engine },
          markLine: {
            silent: true,
            symbol: "none",
            animation: false,
            data: [
              vline(data.central_h_opt_theory, "ε^{1/3}", COLORS.engine),
              vline(data.forward_h_opt_theory, "√ε", COLORS.accent),
            ],
          },
        },
        {
          id: "forward",
          name: "Progresiva O(h)",
          type: "scatter",
          symbol: "triangle",
          symbolSize: 8,
          data: pts(data.forward_error),
          itemStyle: { color: COLORS.accent },
        },
        {
          id: "cmodel",
          name: "Modelo central",
          type: "line",
          showSymbol: false,
          data: landed ? line(data.central_model) : [],
          lineStyle: { color: COLORS.engine, type: "dashed", width: 1.4, opacity: 0.7 },
          itemStyle: { color: COLORS.engine },
        },
        {
          id: "fmodel",
          name: "Modelo progresiva",
          type: "line",
          showSymbol: false,
          data: landed ? line(data.forward_model) : [],
          lineStyle: { color: COLORS.accent, type: "dashed", width: 1.4, opacity: 0.7 },
          itemStyle: { color: COLORS.accent },
        },
      ],
    } as EChartsOption;
  }, [data, landed]);

  return <EChart option={option} style={{ height, width: "100%" }} />;
}
