import { Tag, Typography, Timeline, Collapse, Space, Row, Col, Card } from "antd";
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  FieldTimeOutlined,
  BulbOutlined,
} from "@ant-design/icons";
import type { DebugTraces, ReplanDetail } from "../lib/types";

const { Text, Paragraph } = Typography;

interface DebugTracePanelProps {
  traces: DebugTraces | null;
}

const statusColor = (status: string): string =>
  ({
    success: "green",
    failed: "red",
    replanned: "orange",
    cached: "blue",
    passed: "green",
  })[status] || "gray";

const statusTagColor = (status: string): string =>
  ({
    success: "success",
    failed: "error",
    replanned: "warning",
    cached: "processing",
    passed: "success",
  })[status] || "default";

const statusIcon = (status: string) => {
  switch (status) {
    case "success":
    case "passed":
      return <CheckCircleOutlined style={{ color: "#52c41a" }} />;
    case "failed":
      return <CloseCircleOutlined style={{ color: "#ff4d4f" }} />;
    case "replanned":
      return <SyncOutlined style={{ color: "#fa8c16" }} />;
    case "cached":
      return <FieldTimeOutlined style={{ color: "#1677ff" }} />;
    default:
      return <BulbOutlined style={{ color: "#8c8c8c" }} />;
  }
};

/** 三段式 CoT 渲染 */
function ReplanBlock({ replan }: { replan: ReplanDetail | null | undefined }) {
  if (!replan) return null;

  const sections = [
    {
      title: "错误根因分析",
      key: "error_analysis" as const,
      color: "#fff2f0",
      border: "#ff4d4f",
    },
    {
      title: "调整后计划",
      key: "revised_plan" as const,
      color: "#e6f4ff",
      border: "#1677ff",
    },
    {
      title: "新行动执行",
      key: "action_execution" as const,
      color: "#fafafa",
      border: "#d9d9d9",
    },
  ];

  return (
    <div style={{ marginTop: 8 }}>
      <Text strong style={{ fontSize: 12, display: "block", marginBottom: 4 }}>
        结构化重规划 (Structured Re-plan):
      </Text>
      {sections.map(({ title, key, color, border }) => {
        const content = replan[key];
        if (!content) return null;
        return (
          <Card
            key={key}
            size="small"
            style={{
              marginBottom: 4,
              background: color,
              borderLeft: `3px solid ${border}`,
              fontSize: 12,
            }}
          >
            <Text strong style={{ fontSize: 12, display: "block", marginBottom: 2 }}>
              {title}
            </Text>
            <Paragraph style={{ fontSize: 12, margin: 0, whiteSpace: "pre-wrap" }}>
              {content}
            </Paragraph>
          </Card>
        );
      })}
    </div>
  );
}

export default function DebugTracePanel({ traces }: DebugTracePanelProps) {
  if (!traces || traces.steps.length === 0) {
    return <Text type="secondary">无追踪数据</Text>;
  }

  const totalSteps = traces.steps.length;
  const successCount = traces.steps.filter(
    (s) => s.status === "success" || s.status === "passed",
  ).length;

  return (
    <Collapse
      ghost
      items={[
        {
          key: "trace",
          label: (
            <Space>
              <Text strong style={{ fontSize: 14 }}>
                思考过程
              </Text>
              <Tag>{totalSteps} 步</Tag>
              <Tag color="success">{successCount} 成功</Tag>
              <Text type="secondary" style={{ fontSize: 12 }}>
                总耗时: {traces.total_latency_ms.toFixed(0)}ms
              </Text>
            </Space>
          ),
          children: (
            <div>
              <Row justify="space-between" style={{ marginBottom: 12 }}>
                <Col>
                  <Text code style={{ fontSize: 11 }}>{traces.trace_id}</Text>
                </Col>
              </Row>
              <Timeline>
                {traces.steps.map((step, i) => (
                  <Timeline.Item key={i} color={statusColor(step.status)} dot={statusIcon(step.status)}>
                    <Space>
                      <Tag color={statusTagColor(step.status)}>
                        {step.status}
                      </Tag>
                      <Text strong>{step.stage}</Text>
                      <Text type="secondary">{step.latency_ms.toFixed(1)}ms</Text>
                      <Text type="secondary" style={{ fontSize: 11 }}>
                        步骤 {i + 1}/{totalSteps}
                      </Text>
                    </Space>

                    {step.detail && (
                      <div style={{ marginTop: 4 }}>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          {step.detail}
                        </Text>
                      </div>
                    )}

                    {/* 结构化 Re-plan 三段式 CoT */}
                    <ReplanBlock replan={step.replan} />

                    {step.score != null && (
                      <div style={{ marginTop: 4 }}>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          得分: {step.score.toFixed(3)}
                        </Text>
                      </div>
                    )}
                  </Timeline.Item>
                ))}
              </Timeline>
            </div>
          ),
        },
      ]}
    />
  );
}
