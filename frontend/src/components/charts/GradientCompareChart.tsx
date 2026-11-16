import type { EChartsOption } from "echarts";
import { EChart } from "./EChart";
import { useMemo } from "react";
import { axisStyle, baseChart, logLabel } from "../../lib/chartTheme";
import { COLORS, sci } from "../../lib/format";
import type { GradientCheck } from "../../lib/types";

/** ∇ analítico (barras) vs ∇ numérico (diamantes) y error absoluto (log) por componente. */
export function GradientCompareChart({ data, height = 400 }: { data: GradientCheck; height?: number }) {
  const option = useMemo<EChartsOption>(() => {
    const labels = data.components.map((_, i) => (i === data.components.length - 1 ? "b" : `w${i + 1}`));
    return {
      ...baseChart,
      legend: { ...baseChart.legend, data: ["∇ analítico", "∇ numérico (central)", "|error|"] },
      tooltip: {
        ...baseChart.tooltip,
        formatter: (ps: unknown) => {
          const arr = ps as { dataIndex: number }[];
          const c = data.components[arr[0].dataIndex];
          return `<b>${c.name}</b><br/>analítico: ${sci(c.analytic, 6)}<br/>numérico: ${sci(c.numeric, 6)}<br/>|Δ|: ${sci(c.abs_error, 2)}`;
        },
      },
      grid: [
        { left: 64, right: 20, top: 40, height: "46%" },
        { left: 64, right: 20, top: "68%", bottom: 36 },
      ],
      xAxis: [
        { type: "category", data: labels, gridIndex: 0, ...axisStyle, axisLabel: { show: false }, splitLine: { show: false } },
        { type: "category", data: labels, gridIndex: 1, ...axisStyle, axisLabel: { ...axisStyle.axisLabel, fontSize: 9, interval: 1 }, splitLine: { show: false } },
      ],
      yAxis: [
        { type: "value", gridIndex: 0, name: "∂J/∂θⱼ", ...axisStyle },
        { type: "log", gridIndex: 1, name: "|Δ|", ...axisStyle, axisLabel: { ...axisStyle.axisLabel, formatter: logLabel } },
      ],
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      animationDuration: 1100,
      animationDelay: (i: number) => i * 25,
      series: [
        {
          name: "∇ analítico",
          type: "bar",
          xAxisIndex: 0,
          yAxisIndex: 0,
          barWidth: "55%",
          data: data.components.map((c) => c.analytic),
          itemStyle: { color: `${COLORS.engine}aa`, borderRadius: 3 },
        },
        {
          name: "∇ numérico (central)",
          type: "scatter",
          xAxisIndex: 0,
          yAxisIndex: 0,
          symbol: "diamond",
          symbolSize: 9,
          data: data.components.map((c) => c.numeric),
          itemStyle: { color: COLORS.accent, borderColor: "#fff", borderWidth: 0.6 },
        },
        {
          name: "|error|",
          type: "bar",
          xAxisIndex: 1,
          yAxisIndex: 1,
          barWidth: "55%",
          data: data.components.map((c) => (c.abs_error > 0 ? c.abs_error : null)),
          itemStyle: { color: `${COLORS.amber}cc`, borderRadius: [3, 3, 0, 0] },
        },
      ],
    } as EChartsOption;
  }, [data]);
  return <EChart option={option} style={{ height, width: "100%" }} notMerge />;
}
