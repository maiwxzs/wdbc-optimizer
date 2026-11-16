import type { EChartsOption } from "echarts";
import { EChart } from "./EChart";
import { useMemo } from "react";
import { axisStyle, baseChart } from "../../lib/chartTheme";
import { COLORS, pct } from "../../lib/format";
import type { PCAPoints } from "../../lib/types";

interface Props {
  data: PCAPoints;
  selected: number;
  current?: { x: number; y: number } | null;
  onSelect: (i: number) => void;
  height?: number;
}

/** Proyección PCA 2D (solo visualización; el modelo opera en ℝ^30). */
export function PCAScatter({ data, selected, current, onSelect, height = 340 }: Props) {
  const option = useMemo<EChartsOption>(() => {
    const train = (lbl: 0 | 1) => data.train.filter((p) => p.label === lbl).map((p) => [p.x, p.y]);
    const test = (lbl: 0 | 1) =>
      data.test.filter((p) => p.label === lbl).map((p) => ({ value: [p.x, p.y], i: p.i, proba: p.proba, wrong: p.pred !== p.label }));
    const wrong = data.test.filter((p) => p.pred !== p.label).map((p) => ({ value: [p.x, p.y], i: p.i, proba: p.proba }));
    const sel = data.test.find((p) => p.i === selected);
    const point = current ?? (sel ? { x: sel.x, y: sel.y } : null);
    const ev = data.explained_variance;
    return {
      ...baseChart,
      grid: { left: 50, right: 20, top: 40, bottom: 44 },
      legend: { ...baseChart.legend, data: ["Benigno (test)", "Maligno (test)", "Error de clasificación", "Paciente"] },
      tooltip: {
        ...baseChart.tooltip,
        trigger: "item",
        formatter: (p: unknown) => {
          const d = (p as { data: { i?: number; proba?: number } }).data;
          return d && d.i !== undefined ? `Paciente P-${String(d.i + 1).padStart(3, "0")}<br/>P(maligno) = ${pct(d.proba ?? null)}` : "Entrenamiento";
        },
      },
      xAxis: { type: "value", name: `PC1 (${pct(ev[0])})`, nameLocation: "middle", nameGap: 26, ...axisStyle, scale: true },
      yAxis: { type: "value", name: `PC2 (${pct(ev[1])})`, ...axisStyle, scale: true },
      animationDuration: 900,
      series: [
        { name: "train-b", type: "scatter", symbolSize: 4, data: train(0), itemStyle: { color: COLORS.benign, opacity: 0.16 }, silent: true },
        { name: "train-m", type: "scatter", symbolSize: 4, data: train(1), itemStyle: { color: COLORS.malignant, opacity: 0.16 }, silent: true },
        { name: "Benigno (test)", type: "scatter", symbolSize: 8, data: test(0), itemStyle: { color: COLORS.benign, opacity: 0.9 } },
        { name: "Maligno (test)", type: "scatter", symbolSize: 8, data: test(1), itemStyle: { color: COLORS.malignant, opacity: 0.9 } },
        {
          name: "Error de clasificación",
          type: "scatter",
          symbol: "circle",
          symbolSize: 15,
          data: wrong,
          itemStyle: { color: "transparent", borderColor: COLORS.accent, borderWidth: 2 },
        },
        {
          name: "Paciente",
          type: "effectScatter",
          symbol: "diamond",
          symbolSize: 18,
          rippleEffect: { brushType: "stroke", scale: 2.2, period: 4 },
          data: point ? [[point.x, point.y]] : [],
          itemStyle: { color: COLORS.engine },
          // Capa de canvas propia: la onda animada no obliga a redibujar los ~570 puntos en cada fotograma
          zlevel: 1,
          z: 10,
        },
      ],
    } as EChartsOption;
  }, [data, selected, current]);

  const onEvents = useMemo(
    () => ({
      click: (p: { data?: { i?: number } }) => {
        if (p.data && typeof p.data.i === "number") onSelect(p.data.i);
      },
    }),
    [onSelect],
  );

  return <EChart option={option} style={{ height, width: "100%" }} onEvents={onEvents} />;
}
