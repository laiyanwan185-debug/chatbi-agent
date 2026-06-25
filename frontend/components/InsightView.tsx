import { Typography, Empty, Button, message } from "antd";
import { CopyOutlined } from "@ant-design/icons";
import ReactMarkdown from "react-markdown";

interface InsightViewProps {
  content: string | null | undefined;
}

export default function InsightView({ content }: InsightViewProps) {
  if (!content || !content.trim()) {
    return <Empty description="暂无解读内容" />;
  }

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      message.success("已复制到剪贴板");
    } catch {
      message.error("复制失败，请手动选择文本复制");
    }
  };

  return (
    <div style={{ padding: "8px 0", lineHeight: 1.8 }}>
      <div style={{ textAlign: "right", marginBottom: 8 }}>
        <Button
          size="small"
          icon={<CopyOutlined />}
          onClick={handleCopy}
        >
          复制内容
        </Button>
      </div>
      <ReactMarkdown
        components={{
          h1: ({ children }) => (
            <Typography.Title level={3}>{children}</Typography.Title>
          ),
          h2: ({ children }) => (
            <Typography.Title level={4}>{children}</Typography.Title>
          ),
          h3: ({ children }) => (
            <Typography.Title level={5}>{children}</Typography.Title>
          ),
          p: ({ children }) => (
            <Typography.Paragraph>{children}</Typography.Paragraph>
          ),
          table: ({ children }) => (
            <table
              style={{
                borderCollapse: "collapse",
                width: "100%",
                fontSize: 13,
              }}
            >
              {children}
            </table>
          ),
          th: ({ children }) => (
            <th
              style={{
                border: "1px solid #d9d9d9",
                padding: "6px 10px",
                background: "#fafafa",
                fontWeight: 600,
              }}
            >
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td style={{ border: "1px solid #d9d9d9", padding: "4px 10px" }}>
              {children}
            </td>
          ),
          blockquote: ({ children }) => (
            <blockquote
              style={{
                borderLeft: "4px solid #faad14",
                padding: "8px 16px",
                margin: "12px 0",
                background: "#fffbe6",
                borderRadius: 4,
              }}
            >
              {children}
            </blockquote>
          ),
          code: ({ className, children, ...props }) => {
            const isInline = !className;
            if (isInline) {
              return (
                <code
                  style={{
                    background: "#f5f5f5",
                    padding: "2px 6px",
                    borderRadius: 3,
                    fontSize: "0.9em",
                  }}
                  {...props}
                >
                  {children}
                </code>
              );
            }
            return (
              <div style={{ position: "relative" }}>
                <Button
                  size="small"
                  icon={<CopyOutlined />}
                  style={{ position: "absolute", right: 8, top: 8, zIndex: 1 }}
                  onClick={() => {
                    const codeText = String(children).replace(/\n$/, "");
                    navigator.clipboard.writeText(codeText).then(
                      () => message.success("代码已复制"),
                      () => message.error("复制失败"),
                    );
                  }}
                />
                <pre
                  style={{
                    background: "#1e1e1e",
                    color: "#d4d4d4",
                    padding: 16,
                    borderRadius: 6,
                    overflow: "auto",
                    fontSize: 13,
                  }}
                >
                  <code className={className} {...props}>
                    {children}
                  </code>
                </pre>
              </div>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
