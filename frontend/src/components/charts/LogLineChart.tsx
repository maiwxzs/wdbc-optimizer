import type { EChartsOption } from "echarts";
import { EChart } from "./EChart";
import { useMemo } from "react";
import { areaGradient, axisStyle, baseChart, logLabel, neonLine } from "../../lib/chartTheme";
import { COLORS, sci } from "../../lib/format";
import type { Num } from "../../lib/types";

export interface LineSeries {
  name: string;
  color: string;
  x: number[];
  y: Num[];
  area?: boolean;
  dashed?: boolean;
  width?: number;
}

export interface HLine {
  value: number;
  label: string;
  color?: string;
}

interface Props {
  series: LineSeries[];
  height?: number;
  yLog?: boolean;
  xLog?: boolean;
  xName?: string;
  yName?: string;
  hLines?: HLine[];
  /** Duración del trazado inicial (la curva se "dibuja" de izquierda a derecha). */
  drawMs?: number;
  legend?: boolean;
}

/**
 * Gráfico de líneas para telemetría numérica. En escala log los valores ≤ 0 o
 * no finitos se omiten (null) — ECharts deja un hueco, lo cual es deliberado:
 * un hueco en la curva ES la evidencia visual de un NaN/Inf.
 */
export function LogLineChart({
  series,
  height = 280,
  yLog = true,
  xLog = false,
  xName = "Iteración k",
  yName,
  hLines = [],
  drawMs = 2000,
  legend = true,
}: Props) {
  const option = useMemo<EChartsOption>(() => {
    const clean = (v: Num) => (v === null || !Number.isFinite(v) || (yLog && v <= 0) ? null : v);
    return {
      ...baseChart,
      legend: legend ? { ...baseChart.legend, data: series.map((s) => s.name) } : undefined,
      grid: { ...baseChart.grid, top: legend ? 40 : 20 },
      tooltip: {
        ...baseChart.tooltip,
        valueFormatter: (v) => sci(v as number, 4),
      },
      xAxis: {
        type: xLog ? "log" : "value",
        name: xName,
        nameLocation: "middle",
        nameGap: 30,
        ...axisStyle,
        axisLabel: { ...axisStyle.axisLabel, formatter: xLog ? logLabel : undefined },
        splitLine: { show: false },
        min: xLog ? undefined : "dataMin",
        max: xLog ? undefined : "dataMax",
      },
      yAxis: {
        type: yLog ? "log" : "value",
        name: yName,
        nameLocation: "end",
        ...axisStyle,
        axisLabel: { ...axisStyle.axisLabel, formatter: yLog ? logLabel : undefined },
      },
      dataZoom: [{ type: "inside", xAxisIndex: 0 }],
      animationDuration: drawMs,
      animationEasing: "cubicInOut",
      series: series.map((s, i) => ({
        name: s.name,
        type: "line",
        showSymbol: false,
        smooth: false,
        sampling: "lttb", // submuestreo que preserva la forma: menos segmentos que dibujar
        connectNulls: false,
        data: s.x.map((x, j) => [x, clean(s.y[j] ?? null)]),
        lineStyle: { ...neonLine(s.color, s.width ?? 2.2), type: s.dashed ? "dashed" : "solid" },
        itemStyle: { color: s.color },
        areaStyle: s.area ? { color: areaGradient(s.color), origin: "start" } : undefined,
        emphasis: { focus: "series" },
        markLine:
          i === 0 && hLines.length
            ? {
                silent: true,
                symbol: "none",
                animation: false,
                data: hLines.map((h) => ({
                  yAxis: h.value,
                  label: { formatter: h.label, color: h.color ?? COLORS.accent, position: "insideEndTop", fontSize: 10 },
                  lineStyle: { color: h.color ?? COLORS.accent, type: "dashed", width: 1.2 },
                })),
              }
            : undefined,
      })),
    } as EChartsOption;
  }, [series, yLog, xLog, xName, yName, hLines, drawMs, legend]);

  return <EChart option={option} style={{ height, width: "100%" }} notMerge />;
}
