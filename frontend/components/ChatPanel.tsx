import { useState, useRef, useEffect, useCallback } from "react";
import {
  Input,
  Button,
  Typography,
  Spin,
  Tag,
  Avatar,
  Space,
  Alert,
  List,
} from "antd";
import {
  SendOutlined,
  UserOutlined,
  RobotOutlined,
} from "@ant-design/icons";
import type { Message } from "../lib/types";

const { Text, Paragraph } = Typography;
const { TextArea } = Input;

interface ChatPanelProps {
  messages: Message[];
  onSend: (query: string) => void;
  isLoading: boolean;
  onSelectMessage: (messageId: string | null) => void;
  selectedMessageId: string | null;
}

export default function ChatPanel({
  messages,
  onSend,
  isLoading,
  onSelectMessage,
  selectedMessageId,
}: ChatPanelProps) {
  const [inputValue, setInputValue] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  // 自动滚动到底部
  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  const handleSend = useCallback(() => {
    const trimmed = inputValue.trim();
    if (!trimmed || isLoading) return;
    setInputValue("");
    onSend(trimmed);
  }, [inputValue, isLoading, onSend]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        background: "#fff",
      }}
    >
      {/* 标题 */}
      <div
        style={{
          padding: "12px 16px",
          borderBottom: "1px solid #f0f0f0",
          background: "#fafafa",
        }}
      >
        <Text strong style={{ fontSize: 15 }}>
          对话
        </Text>
      </div>

      {/* 消息列表 */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "12px 16px",
        }}
      >
        {messages.length === 0 ? (
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              height: "100%",
              color: "#999",
            }}
          >
            <Text type="secondary">输入数据分析问题开始对话</Text>
          </div>
        ) : (
          <List
            dataSource={messages}
            split={false}
            renderItem={(msg) => (
              <List.Item
                style={{
                  padding: "8px 0",
                  cursor:
                    msg.role === "assistant" && msg.response
                      ? "pointer"
                      : "default",
                  background:
                    msg.id === selectedMessageId
                      ? "#f0f5ff"
                      : "transparent",
                  borderRadius: 6,
                  paddingLeft: 8,
                  paddingRight: 8,
                }}
                onClick={() => {
                  if (msg.role === "assistant" && msg.response) {
                    onSelectMessage(msg.id);
                  }
                }}
              >
                <Space align="start" size={12}>
                  <Avatar
                    icon={
                      msg.role === "user" ? (
                        <UserOutlined />
                      ) : (
                        <RobotOutlined />
                      )
                    }
                    style={{
                      backgroundColor:
                        msg.role === "user" ? "#1677ff" : "#52c41a",
                      marginTop: 4,
                    }}
                    size={28}
                  />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <Paragraph
                      style={{
                        marginBottom: 4,
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-word",
                      }}
                    >
                      {msg.content}
                    </Paragraph>
                    {msg.loading && (
                      <Spin size="small" style={{ marginTop: 4 }} />
                    )}
                    {msg.response && (
                      <Tag color="blue" style={{ marginTop: 4 }}>
                        查看结果
                      </Tag>
                    )}
                    {msg.error && (
                      <Alert
                        type="error"
                        message={msg.error}
                        style={{ marginTop: 4, fontSize: 12 }}
                      />
                    )}
                  </div>
                </Space>
              </List.Item>
            )}
          />
        )}
        <div ref={scrollRef} />
      </div>

      {/* 输入区 */}
      <div
        style={{
          borderTop: "1px solid #f0f0f0",
          padding: "12px 16px",
          background: "#fafafa",
        }}
      >
        <TextArea
          rows={3}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入数据分析问题..."
          disabled={isLoading}
          style={{ resize: "none" }}
        />
        <div
          style={{
            display: "flex",
            justifyContent: "flex-end",
            marginTop: 8,
          }}
        >
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleSend}
            loading={isLoading}
          >
            发送
          </Button>
        </div>
      </div>
    </div>
  );
}
