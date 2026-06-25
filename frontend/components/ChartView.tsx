import dynamic from "next/dynamic";
import { Empty, Typography } from "antd";
import type { EChartsOption } from "echarts";

const ReactECharts = dynamic(() => import("echarts-for-react"), { ssr: false });
const { Text } = Typography;

interface ChartViewProps {
  type: string | null | undefined;
  data: Record<string, unknown> | null | undefined;
  analysisType?: string;
}

// 分析类型 → 推荐图类型映射
const ANALYSIS_CHART_MAP: Record<string, string> = {
  trend: "line",
  time_series: "line",
  rank: "bar",
  comparison: "bar",
  proportion: "pie",
  share: "pie",
  correlation: "scatter",
  relation: "scatter",
  multi_dim: "radar",
  comprehensive: "radar",
  anomaly: "scatter_line",
  outlier: "scatter_line",
};

// 数据形状驱动的降级策略
function detectType(
  analysisType: string | undefined,
  chartData: Record<string, unknown>,
): string {
  if (analysisType) {
    const prefix = analysisType.toLowerCase().split("_")[0];
    const recommended =
      ANALYSIS_CHART_MAP[analysisType] || ANALYSIS_CHART_MAP[prefix];
    if (recommended) {
      // 雷达图降级：无 radar.indicator 时转柱状
      if (recommended === "radar") {
        const hasRadar = chartData["radar"] != null;
        const series = chartData["series"];
        const hasRadarSeries =
          Array.isArray(series) &&
          (series as Record<string, unknown>[]).some(
            (s) => s?.type === "radar",
          );
        if (!hasRadar && !hasRadarSeries) return "bar";
      }
      // 饼图降级：数据项>8 时转柱状
      if (recommended === "pie") {
        const ds = chartData["dataset"];
        const dsLen =
          Array.isArray(ds) && typeof ds[0] === "object" ? ds.length : 0;
        const series = chartData["series"];
        let maxCat = 0;
        if (dsLen > 0) {
          maxCat = dsLen;
        } else if (Array.isArray(series)) {
          maxCat = (series as Record<string, unknown>[]).reduce(
            (acc, s) =>
              Math.max(
                acc,
                Array.isArray(s.data) ? (s.data as unknown[]).length : 0,
              ),
            0,
          );
        }
        if (maxCat > 8) return "bar";
      }
      // 折线图降级：类别≤5 时转柱状
      if (recommended === "line") {
        const xAxis = chartData["xAxis"];
        const cats =
          xAxis != null && typeof xAxis === "object" && "data" in xAxis
            ? (xAxis as { data?: unknown[] }).data
            : undefined;
        const catCount = Array.isArray(cats) ? cats.length : 0;
        if (catCount > 0 && catCount <= 5) return "bar";
      }
      return recommended;
    }
  }

  // 空降级：从数据形状推断
  const series = chartData["series"];
  if (Array.isArray(series)) {
    const seriesArr = series as Record<string, unknown>[];
    if (seriesArr.length > 1) return "bar";
    const first = seriesArr[0];
    if (first?.type && typeof first.type === "string") return first.type;
  }
  return "bar";
}

// 每种图表的 option 骨架
function buildOption(type: string): EChartsOption {
  const base: EChartsOption = {
    legend: { type: "scroll", bottom: 0 },
  };

  switch (type) {
    case "line":
      return {
        ...base,
        tooltip: { trigger: "axis" },
        grid: { left: "3%", right: "4%", bottom: "15%", containLabel: true },
      };
    case "bar":
      return {
        ...base,
        tooltip: { trigger: "axis" },
        grid: { left: "3%", right: "4%", bottom: "15%", containLabel: true },
        xAxis: { type: "category", axisLabel: { rotate: 45 } },
        yAxis: { type: "value" },
      };
    case "pie":
      return {
        ...base,
        tooltip: { trigger: "item", formatter: "{b}: {c} ({d}%)" },
        series: [
          {
            type: "pie" as const,
            radius: ["40%", "70%"],
            center: ["50%", "50%"],
          },
        ],
      };
    case "scatter":
      return {
        ...base,
        tooltip: { trigger: "item" },
        grid: { left: "3%", right: "4%", bottom: "15%", containLabel: true },
        xAxis: { type: "value" },
        yAxis: { type: "value" },
      };
    case "scatter_line":
      return {
        ...base,
        tooltip: { trigger: "axis" },
        grid: { left: "3%", right: "4%", bottom: "15%", containLabel: true },
      };
    case "radar":
      return {
        ...base,
        tooltip: { trigger: "item" },
        radar: { center: ["50%", "50%"], radius: "60%" },
      };
    default:
      return {
        tooltip: { trigger: "axis" },
        grid: { left: "3%", right: "4%", bottom: "15%", containLabel: true },
      };
  }
}

/** 构建最终的 ECharts option（合并骨架 + 数据），并应用后处理。 */
function buildMergedOption(
  chartType: string,
  chartData: Record<string, unknown>,
): EChartsOption {
  const merged = {
    ...buildOption(chartType),
    ...chartData,
  } as EChartsOption;

  // scatter_line 特殊处理：保证 series 中有 smooth + 圆点
  if (chartType === "scatter_line" && Array.isArray(merged.series)) {
    const raw = merged.series as Record<string, unknown>[];
    merged.series = raw.map((s) => ({
      ...s,
      type: "line",
      smooth: true,
      symbol: "circle",
      symbolSize: 6,
    })) as typeof merged.series;
  }

  // 水平柱状图：bar 且 label 过多时翻转轴
  if (chartType === "bar") {
    const opt = merged as Record<string, unknown>;
    const xAxis = opt["xAxis"] as Record<string, unknown> | undefined;
    const yAxis = opt["yAxis"] as Record<string, unknown> | undefined;
    const xLabels = xAxis?.data;
    const labelLen = Array.isArray(xLabels) ? xLabels.length : 0;
    if (labelLen > 6) {
      opt["xAxis"] = yAxis;
      opt["yAxis"] = xAxis;
    }
  }

  return merged;
}

/** 从 chartData 中提取系列名列表 */
function getSeriesNames(chartData: Record<string, unknown>): string[] {
  const series = chartData["series"];
  if (!Array.isArray(series)) return [];
  return (series as Record<string, unknown>[])
    .map((s) => s.name as string | undefined)
    .filter(Boolean) as string[];
}

/** 渲染单个 ECharts 图表 */
function SingleChart({
  chartType,
  chartData,
  idx,
  noLegend,
}: {
  chartType: string;
  chartData: Record<string, unknown>;
  idx?: number;
  noLegend?: boolean;
}) {
  const option = buildMergedOption(chartType, chartData);

  // 子图可去掉 legend 节约空间，或仅保留 legend.data 子集
  if (noLegend && option.legend) {
    option.legend = undefined;
  }

  return (
    <div style={{ marginBottom: 16 }}>
      <ReactECharts
        option={option}
        style={{ height: 380, width: "100%" }}
        notMerge={true}
        lazyUpdate={true}
      />
    </div>
  );
}

export default function ChartView({ type, data, analysisType }: ChartViewProps) {
  const chartData = data ?? {};

  if (!chartData || Object.keys(chartData).length === 0) {
    return <Empty description="暂无图表数据" />;
  }

  // ====== 多图表模式：量级差异拆分为多个子图 ======
  const multiCharts = chartData["charts"];
  if (Array.isArray(multiCharts) && multiCharts.length > 1) {
    const intro = chartData["intro"] as string | undefined;
    return (
      <div>
        {intro && (
          <div
            style={{
              padding: "6px 0 2px",
              fontSize: 13,
              color: "#888",
              textAlign: "center",
            }}
          >
            <Text type="secondary">{intro}</Text>
          </div>
        )}
        {(multiCharts as Record<string, unknown>[]).map((subChart, idx) => {
          const subChartType = type || detectType(analysisType, subChart);
          const names = getSeriesNames(subChart);
          return (
            <div key={idx} style={{ marginTop: idx > 0 ? 8 : 0 }}>
              <div
                style={{
                  padding: "4px 0 0 8px",
                  fontWeight: 500,
                  fontSize: 13,
                  color: "#555",
                }}
              >
                图 {idx + 1}：{names.join("、")}
              </div>
              <SingleChart
                chartType={subChartType}
                chartData={subChart}
                idx={idx}
              />
            </div>
          );
        })}
      </div>
    );
  }

  // ====== 单图表模式 ======
  const chartType = type || detectType(analysisType, chartData);
  return <SingleChart chartType={chartType} chartData={chartData} />;
}
