import type { EChartsOption } from "echarts";
import { EChart } from "./EChart";
import { useMemo } from "react";
import { axisStyle, baseChart } from "../../lib/chartTheme";
import { COLORS, sci } from "../../lib/format";
import type { Contribution } from "../../lib/types";

/** Contribuciones aditivas al logit: z = Σ w_j x̃_j + b. Rojo empuja a maligno. */
export function ContributionsChart({ items, top = 10, height = 320 }: { items: Contribution[]; top?: number; height?: number }) {
  const option = useMemo<EChartsOption>(() => {
    const rows = items.slice(0, top).reverse();
    return {
      ...baseChart,
      grid: { left: 150, right: 30, top: 10, bottom: 30 },
      tooltip: {
        ...baseChart.tooltip,
        trigger: "item",
        formatter: (p: unknown) => {
          const { dataIndex } = p as { dataIndex: number };
          const r = rows[dataIndex];
          return `<b>${r.name}</b><br/>valor: ${sci(r.raw_value)}<br/>x̃ (z-score): ${sci(r.std_value)}<br/>w: ${sci(r.weight)}<br/>w·x̃: <b>${sci(r.contribution)}</b>`;
        },
      },
      xAxis: { type: "value", name: "w_j · x̃_j", nameLocation: "middle", nameGap: 22, ...axisStyle },
      yAxis: {
        type: "category",
        data: rows.map((r) => r.name),
        ...axisStyle,
        axisLabel: { ...axisStyle.axisLabel, fontFamily: "Inter", fontSize: 11, color: COLORS.text },
        splitLine: { show: false },
      },
      animationDuration: 900,
      animationEasing: "cubicOut",
      animationDelay: (i: number) => i * 45,
      series: [
        {
          type: "bar",
          barWidth: "58%",
          data: rows.map((r) => ({
            value: r.contribution,
            itemStyle: {
              color: r.contribution >= 0 ? COLORS.malignant : COLORS.benign,
              borderRadius: r.contribution >= 0 ? [0, 6, 6, 0] : [6, 0, 0, 6],
              shadowBlur: 0,
              shadowColor: r.contribution >= 0 ? `${COLORS.malignant}88` : `${COLORS.benign}88`,
            },
          })),
        },
      ],
    } as EChartsOption;
  }, [items, top]);
  return <EChart option={option} style={{ height, width: "100%" }} />;
}
