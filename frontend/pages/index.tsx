import { useState, useCallback, useEffect, useRef } from "react";
import { Layout, Typography, Row, Col, message } from "antd";
import Head from "next/head";
import ChatPanel from "../components/ChatPanel";
import ResultViewer from "../components/ResultViewer";
import DataSetSelector from "../components/DataSetSelector";
import { sendQueryStream, fetchSchema } from "../lib/api";
import type { Message, SchemaTable, QueryResponse } from "../lib/types";

const { Header, Content } = Layout;
const { Title } = Typography;

let messageCounter = 0;
function nextMsgId(): string {
  messageCounter += 1;
  return `msg_${Date.now()}_${messageCounter}`;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(
    null,
  );
  const [isLoading, setIsLoading] = useState(false);
  const [schemaTables, setSchemaTables] = useState<SchemaTable[]>([]);

  // 实时计时器 refs（在 SSE 事件间隔也能每秒更新耗时）
  const stageRef = useRef("初始化");
  const startTimeRef = useRef(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // 初始化时加载 schema
  useEffect(() => {
    fetchSchema()
      .then((res) => setSchemaTables(res.tables))
      .catch(() => {
        // 非关键路径，静默失败
      });
  }, []);

  // 当前选中的消息数据
  const selectedMessage = messages.find(
    (m) => m.id === selectedMessageId,
  );
  const currentData = selectedMessage?.response ?? null;
  const currentDebugTraces = selectedMessage?.debugTraces ?? null;
  const currentError = selectedMessage?.error ?? null;

  const handleSend = useCallback(
    async (query: string) => {
      const userMsg: Message = {
        id: nextMsgId(),
        role: "user",
        content: query,
        timestamp: Date.now(),
      };

      const loadingMsg: Message = {
        id: nextMsgId(),
        role: "assistant",
        content: "正在分析...",
        timestamp: Date.now(),
        loading: true,
      };

      setMessages((prev) => [...prev, userMsg, loadingMsg]);
      setSelectedMessageId(loadingMsg.id);
      setIsLoading(true);

      // 实时计时器：独立于 SSE 事件，每 1 秒更新耗时
      stageRef.current = "初始化";
      startTimeRef.current = Date.now();
      intervalRef.current = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTimeRef.current) / 1000);
        const content = `正在分析 (${stageRef.current}) — 已耗时 ${elapsed}秒`;
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingMsg.id ? { ...m, content } : m,
          ),
        );
      }, 1000);

      try {
        // SSE 流式查询 — 分阶段更新 loading 显示
        const response: QueryResponse = await sendQueryStream(
          query,
          (stage, elapsedMs) => {
            stageRef.current = stage;
            const seconds = Math.floor(elapsedMs / 1000);
            startTimeRef.current = Date.now() - seconds * 1000; // 与后端时间对齐
            const content = `正在分析 (${stage}) — 已耗时 ${seconds}秒`;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === loadingMsg.id ? { ...m, content } : m,
              ),
            );
          },
        );

        // 最终结果
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingMsg.id
              ? {
                  ...m,
                  loading: false,
                  content:
                    response.data?.interpretation?.slice(0, 100) ||
                    response.data?.message ||
                    (response.status === "success"
                      ? "分析完成"
                      : response.data?.message || "处理完成"),
                  response: response.data,
                  debugTraces: response.debug_traces,
                }
              : m,
          ),
        );

        if (response.status === "success" && response.data) {
          setSelectedMessageId(loadingMsg.id);
        } else if (response.status === "clarify") {
          message.info(response.data?.message || "需要更多信息");
        } else if (response.status === "error") {
          message.error(response.data?.message || "处理出错");
        }
      } catch (err: unknown) {
        const errMsg =
          err instanceof Error ? err.message : "请求失败，请检查后端服务";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingMsg.id
              ? {
                  ...m,
                  loading: false,
                  content: "处理失败",
                  error: errMsg,
                }
              : m,
          ),
        );
        message.error(errMsg);
      } finally {
        // 停止实时计时器
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
        setIsLoading(false);
      }
    },
    [],
  );

  return (
    <>
      <Head>
        <title>ChatBI 智能问数</title>
        <meta name="description" content="智能问数 ChatBI 系统" />
      </Head>
      <Layout style={{ height: "100vh", overflow: "hidden" }}>
        {/* 顶栏 */}
        <Header
          style={{
            background: "#001529",
            padding: "0 24px",
            display: "flex",
            alignItems: "center",
            height: 56,
            lineHeight: "56px",
          }}
        >
          <Row
            justify="space-between"
            align="middle"
            style={{ width: "100%" }}
          >
            <Col>
              <Title
                level={4}
                style={{ color: "#fff", margin: 0, fontWeight: 600 }}
              >
                ChatBI 智能问数
              </Title>
            </Col>
            <Col>
              <DataSetSelector
                tables={schemaTables}
                onTablesLoaded={setSchemaTables}
              />
            </Col>
          </Row>
        </Header>

        {/* 主体 */}
        <Content style={{ display: "flex", height: "calc(100vh - 56px)" }}>
          {/* 左侧对话面板 */}
          <div
            style={{
              width: "35%",
              minWidth: 320,
              borderRight: "1px solid #f0f0f0",
              overflow: "hidden",
            }}
          >
            <ChatPanel
              messages={messages}
              onSend={handleSend}
              isLoading={isLoading}
              onSelectMessage={setSelectedMessageId}
              selectedMessageId={selectedMessageId}
            />
          </div>

          {/* 右侧结果展示 */}
          <div style={{ flex: 1, overflow: "hidden" }}>
            <ResultViewer
              data={currentData}
              debugTraces={currentDebugTraces}
              loading={!!(isLoading && selectedMessage?.loading)}
              error={currentError}
            />
          </div>
        </Content>
      </Layout>
    </>
  );
}
