/**
 * ImportTableModal — 数据表导入弹窗组件
 *
 * 支持 CSV/Excel 文件上传，显示预览信息（列名、行数、关联关系），
 * 确认后调用后端 import API。
 */
import { useState } from "react";
import {
  Modal,
  Upload,
  Input,
  Button,
  Descriptions,
  Steps,
  Tag,
  message,
  Spin,
} from "antd";
import { InboxOutlined, CheckCircleOutlined } from "@ant-design/icons";
import type { UploadFile } from "antd/es/upload/interface";
import { uploadTable } from "../lib/api";
import type { ImportResponse } from "../lib/api";

const { Dragger } = Upload;
const { TextArea } = Input;

interface ImportTableModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

type ImportStep = "upload" | "preview" | "done";

export default function ImportTableModal({
  open,
  onClose,
  onSuccess,
}: ImportTableModalProps) {
  const [step, setStep] = useState<ImportStep>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [tableName, setTableName] = useState("");
  const [tableComment, setTableComment] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ImportResponse | null>(null);

  const reset = () => {
    setStep("upload");
    setFile(null);
    setTableName("");
    setTableComment("");
    setLoading(false);
    setResult(null);
  };

  const handleImport = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const res = await uploadTable(file, tableName || undefined, tableComment || undefined);
      setResult(res);
      setStep("done");
      message.success(`表 "${res.table_name}" 导入成功（${res.row_count} 行）`);
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : "导入失败";
      message.error(errMsg);
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    if (loading) return; // 导入进行中不允许关闭
    if (step === "done") {
      onSuccess();
    }
    reset();
    onClose();
  };

  const getStepIndex = () => {
    switch (step) {
      case "upload": return 0;
      case "preview": return 1;
      case "done": return 2;
    }
  };

  return (
    <Modal
      title="导入数据表"
      open={open}
      onCancel={handleClose}
      width={640}
      maskClosable={!loading}
      keyboard={!loading}
      footer={
        step === "done"
          ? [
              <Button key="close" type="primary" onClick={handleClose}>
                完成
              </Button>,
            ]
          : [
              <Button key="cancel" onClick={handleClose}>
                取消
              </Button>,
              <Button
                key="import"
                type="primary"
                loading={loading}
                disabled={!file}
                onClick={handleImport}
              >
                确认导入
              </Button>,
            ]
      }
    >
      <Steps
        current={getStepIndex()}
        size="small"
        style={{ marginBottom: 24 }}
        items={[
          { title: "上传文件" },
          { title: "预览确认" },
          { title: "完成" },
        ]}
      />

      {/* Step 1: Upload */}
      {step === "upload" && (
        <Dragger
          accept=".csv,.xlsx,.xls"
          beforeUpload={(file) => {
            const MAX_SIZE = 50 * 1024 * 1024;
            if (file.size > MAX_SIZE) {
              message.error("文件不能超过 50MB");
              return Upload.LIST_IGNORE;
            }
            return false; // 阻止自动上传
          }}
          onChange={(info) => {
            const origin = info.file.originFileObj;
            if (!origin) return;
            setFile(origin);
            const name = origin.name.replace(/\.[^.]+$/, "").replace(/[^\w]/g, "_").toLowerCase();
            setTableName(name || "imported_table");
            setStep("preview");
          }}
          showUploadList={false}
          maxCount={1}
        >
          <p className="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
          <p className="ant-upload-hint">
            支持 CSV (.csv)、Excel (.xlsx/.xls) 格式
          </p>
        </Dragger>
      )}

      {/* Step 2: Preview */}
      {step === "preview" && file && (
        <Spin spinning={loading}>
          <Descriptions column={1} size="small" bordered>
            <Descriptions.Item label="文件名">
              {file.name}
            </Descriptions.Item>
            <Descriptions.Item label="文件大小">
              {(file.size / 1024).toFixed(1)} KB
            </Descriptions.Item>
            <Descriptions.Item label="表名">
              <Input
                placeholder="默认从文件名推断"
                value={tableName}
                onChange={(e) => setTableName(e.target.value)}
                style={{ width: "100%" }}
              />
            </Descriptions.Item>
            <Descriptions.Item label="表描述（可选）">
              <TextArea
                placeholder="简要描述此数据表的内容和用途"
                value={tableComment}
                onChange={(e) => setTableComment(e.target.value)}
                rows={2}
              />
            </Descriptions.Item>
          </Descriptions>
          <div style={{ marginTop: 16, color: "#888", fontSize: 12 }}>
            <p>导入后将自动执行：</p>
            <ul style={{ paddingLeft: 20, margin: "4px 0" }}>
              <li>创建数据表并写入数据</li>
              <li>自动注册列类型为业务指标</li>
              <li>自动检测与现有表的关联关系</li>
              <li>更新智能体的表结构感知</li>
            </ul>
          </div>
        </Spin>
      )}

      {/* Step 3: Done */}
      {step === "done" && result && (
        <>
          <div style={{ textAlign: "center", margin: "16px 0" }}>
            <CheckCircleOutlined style={{ fontSize: 48, color: "#52c41a" }} />
          </div>
          <Descriptions column={1} size="small" bordered>
            <Descriptions.Item label="表名">
              <Tag color="blue">{result.table_name}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="数据行数">
              {result.row_count.toLocaleString()} 行
            </Descriptions.Item>
            <Descriptions.Item label="数据列数">
              {result.column_count} 列
            </Descriptions.Item>
            <Descriptions.Item label="自动生成指标">
              {result.indicator_count} 个
            </Descriptions.Item>
            {result.time_column && (
              <Descriptions.Item label="时间列">
                {result.time_column}
              </Descriptions.Item>
            )}
            {result.join_relationships.length > 0 && (
              <Descriptions.Item label="关联关系">
                {result.join_relationships.map((j, i) => (
                  <Tag key={i} color="green" style={{ marginBottom: 4 }}>
                    {j.on.join(", ")} → {j.target}
                  </Tag>
                ))}
              </Descriptions.Item>
            )}
          </Descriptions>
          <div style={{ marginTop: 16, color: "#666", fontSize: 13 }}>
            现在可以在聊天中提问涉及此表的问题了。
          </div>
        </>
      )}
    </Modal>
  );
}
