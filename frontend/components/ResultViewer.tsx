import { useMemo, useState, useEffect } from "react";
import { Tabs, Spin, Alert, Empty } from "antd";
import type { QueryResponseData, DebugTraces } from "../lib/types";
import ChartView from "./ChartView";
import TableView from "./TableView";
import InsightView from "./InsightView";
import DebugTracePanel from "./DebugTracePanel";

interface ResultViewerProps {
  data: QueryResponseData | null;
  debugTraces: DebugTraces | null;
  loading: boolean;
  error: string | null;
}

export default function ResultViewer({
  data,
  debugTraces,
  loading,
  error,
}: ResultViewerProps) {
  // 检测默认 tab — 始终默认图表 tab
  const defaultTab = useMemo(() => {
    if (!data) return "chart";
    return "chart";
  }, [data]);

  const [activeTab, setActiveTab] = useState(defaultTab);

  // 当 data 改变时重置 tab
  useEffect(() => {
    setActiveTab(defaultTab);
  }, [defaultTab]);

  return (
    <div
      style={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        background: "#fff",
      }}
    >
      {/*  loading / error / empty 状态 */}
      {loading && (
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            flex: 1,
          }}
        >
          <Spin size="large" tip="正在分析...">
            <div style={{ padding: 50 }} />
          </Spin>
        </div>
      )}

      {!loading && error && (
        <div style={{ padding: 24 }}>
          <Alert type="error" message="处理出错" description={error} />
        </div>
      )}

      {!loading && !error && !data && (
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            flex: 1,
          }}
        >
          <Empty description="请在左侧输入数据分析问题" />
        </div>
      )}

      {/* 正常结果展示 */}
      {!loading && !error && data && (
        <>
          {/* Tab 栏 */}
          <Tabs
            activeKey={activeTab}
            onChange={setActiveTab}
            style={{ padding: "0 16px", marginBottom: 0 }}
            items={[
              {
                key: "chart",
                label: "图表",
                disabled: false,
              },
              {
                key: "table",
                label: "数据表",
                disabled: !data.table_data,
              },
              {
                key: "insight",
                label: "智能解读",
                disabled: !data.interpretation,
              },
            ]}
          />

          {/* 内容区 */}
          <div style={{ flex: 1, overflow: "auto", padding: "0 16px 16px" }}>
            {data.data_warning && (
              <Alert
                type="warning"
                message="部分数据可能存在异常，请结合原始数据验证"
                showIcon
                style={{ marginBottom: 12 }}
              />
            )}

            {activeTab === "chart" && data.chart_data && (
              <ChartView
                type={data.chart_type}
                data={data.chart_data}
                analysisType={data.analysis_type}
              />
            )}
            {activeTab === "chart" && !data.chart_data && (
              <Empty description="暂无图表数据" />
            )}
            {activeTab === "table" && data.table_data && (
              <TableView
                columns={data.table_columns}
                data={data.table_data}
              />
            )}
            {activeTab === "insight" && data.interpretation && (
              <InsightView content={data.interpretation} />
            )}
          </div>

          {/* Trace 面板 */}
          <DebugTracePanel traces={debugTraces} />
        </>
      )}
    </div>
  );
}
