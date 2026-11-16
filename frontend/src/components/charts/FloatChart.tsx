import type { EChartsOption } from "echarts";
import { EChart } from "./EChart";
import { useMemo } from "react";
import { axisStyle, baseChart, logLabel } from "../../lib/chartTheme";
import { COLORS, sci } from "../../lib/format";
import type { FloatAudit, Num } from "../../lib/types";

/** Pérdida por muestra (y = 1) frente a z: exacta (LSE), acotada (clip) e ingenua. */
export function FloatChart({ data, height = 360 }: { data: FloatAudit; height?: number }) {
  const option = useMemo<EChartsOption>(() => {
    const pairs = (ys: Num[]) => data.z.map((z, i) => [z, ys[i] !== null && (ys[i] as number) > 0 ? ys[i] : null]);
    const c = data.constants;
    return {
      ...baseChart,
      grid: { left: 64, right: 24, top: 44, bottom: 50 },
      legend: { ...baseChart.legend, data: ["Exacta · Log-Sum-Exp", "Acotada · clip ε=1e-15", "Ingenua · −log σ(z)"] },
      tooltip: { ...baseChart.tooltip, valueFormatter: (v) => sci(v as number, 4) },
      xAxis: { type: "value", name: "logit z", nameLocation: "middle", nameGap: 28, min: -800, max: 60, ...axisStyle, splitLine: { show: false } },
      yAxis: { type: "log", name: "ℓ(z, y=1)", ...axisStyle, axisLabel: { ...axisStyle.axisLabel, formatter: logLabel } },
      dataZoom: [{ type: "inside" }],
      animationDuration: 2000,
      animationEasing: "cubicInOut",
      series: [
        {
          name: "Exacta · Log-Sum-Exp",
          type: "line",
          showSymbol: false,
          data: pairs(data.loss_y1.exact),
          lineStyle: { color: COLORS.benign, width: 2.5 },
          itemStyle: { color: COLORS.benign },
          markLine: {
            silent: true,
            symbol: "none",
            data: [
              { xAxis: -c.exp_overflow_threshold, label: { formatter: "exp overflow (−709.8)", color: COLORS.malignant, fontSize: 10 }, lineStyle: { color: COLORS.malignant, type: "dashed" } },
              { xAxis: c.sigmoid_saturation_z, label: { formatter: "σ→1.0 (z≈36.7)", color: COLORS.amber, fontSize: 10 }, lineStyle: { color: COLORS.amber, type: "dashed" } },
            ],
          },
        },
        {
          name: "Acotada · clip ε=1e-15",
          type: "line",
          showSymbol: false,
          data: pairs(data.loss_y1.clipped),
          lineStyle: { color: COLORS.engine, width: 2, type: "dashed" },
          itemStyle: { color: COLORS.engine },
        },
        {
          name: "Ingenua · −log σ(z)",
          type: "line",
          showSymbol: false,
          connectNulls: false,
          data: pairs(data.loss_y1.naive),
          lineStyle: { color: COLORS.malignant, width: 2 },
          itemStyle: { color: COLORS.malignant },
        },
      ],
    } as EChartsOption;
  }, [data]);
  return <EChart option={option} style={{ height, width: "100%" }} />;
}
