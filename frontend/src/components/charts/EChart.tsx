import type { EChartsOption } from "echarts";
import ReactEChartsCore from "echarts-for-react/lib/core";
import type { CSSProperties } from "react";
import { echarts } from "../../lib/echarts";

interface Props {
  option: EChartsOption;
  style?: CSSProperties;
  notMerge?: boolean;
  onEvents?: Record<string, (params: any) => void>;
}

/** Envoltorio único: ECharts modular + lazyUpdate (agrupa actualizaciones por fotograma). */
export function EChart({ option, style, notMerge = false, onEvents }: Props) {
  return (
    <ReactEChartsCore
      echarts={echarts}
      option={option}
      style={style}
      notMerge={notMerge}
      lazyUpdate
      onEvents={onEvents}
      opts={{ renderer: "canvas" }}
    />
  );
}
