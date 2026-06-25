import { useState, useEffect } from "react";
import { Select, Modal, Descriptions, Tag, Spin, Button, Popconfirm, message } from "antd";
import { InfoCircleOutlined, PlusOutlined, DeleteOutlined } from "@ant-design/icons";
import { fetchSchema, fetchImportedTables, deleteImportedTable } from "../lib/api";
import type { SchemaTable, SchemaColumn } from "../lib/types";
import ImportTableModal from "./ImportTableModal";

interface DataSetSelectorProps {
  tables: SchemaTable[];
  onTablesLoaded?: (tables: SchemaTable[]) => void;
}

export default function DataSetSelector({
  tables,
  onTablesLoaded,
}: DataSetSelectorProps) {
  const [localTables, setLocalTables] = useState<SchemaTable[]>(tables);
  const [loading, setLoading] = useState(false);
  const [selectedTable, setSelectedTable] = useState<SchemaTable | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [importedTables, setImportedTables] = useState<string[]>([]);

  // 首次加载 schema
  useEffect(() => {
    if (tables.length > 0) return; // 父组件已传入
    setLoading(true);
    fetchSchema()
      .then((res) => {
        setLocalTables(res.tables);
        onTablesLoaded?.(res.tables);
      })
      .catch(() => {
        // 静默失败，等待手动重试
      })
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tables.length]);

  // 加载已导入表列表
  useEffect(() => {
    fetchImportedTables()
      .then(setImportedTables)
      .catch(() => {});
  }, []);

  const handleSelect = (tableName: string | undefined) => {
    if (!tableName) {
      setSelectedTable(null);
      return;
    }

    // 特殊选项：导入新表
    if (tableName === "__import__") {
      setImportModalOpen(true);
      return;
    }

    const table = localTables.find((t) => t.name === tableName) || null;
    setSelectedTable(table);
    setModalOpen(true);
  };

  const handleImportSuccess = async () => {
    // 刷新 schema 列表
    try {
      const res = await fetchSchema();
      setLocalTables(res.tables);
      onTablesLoaded?.(res.tables);
    } catch {
      // 静默失败
    }
    // 刷新已导入表列表
    try {
      const imported = await fetchImportedTables();
      setImportedTables(imported);
    } catch {
      // 静默失败
    }
  };

  const handleDeleteTable = async () => {
    if (!selectedTable) return;
    try {
      await deleteImportedTable(selectedTable.name);
      message.success(`表 "${selectedTable.name}" 已删除`);
      setModalOpen(false);
      setSelectedTable(null);
      // 刷新 schema
      const res = await fetchSchema();
      setLocalTables(res.tables);
      onTablesLoaded?.(res.tables);
      // 刷新已导入表列表
      const imported = await fetchImportedTables();
      setImportedTables(imported);
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : "删除失败";
      message.error(errMsg);
    }
  };

  const isImported = (name: string) => importedTables.includes(name);

  const renderColumnTag = (col: SchemaColumn) => {
    const tags: string[] = [];
    if (col.is_pk) tags.push("PK");
    if (!col.nullable) tags.push("NN");
    return (
      <span>
        <code>{col.name}</code>
        <span style={{ color: "#999", marginLeft: 4, fontSize: 12 }}>
          {col.type}
        </span>
        {tags.map((t) => (
          <Tag key={t} color="blue" style={{ marginLeft: 4, fontSize: 10 }}>
            {t}
          </Tag>
        ))}
        {col.comment && (
          <span style={{ color: "#666", marginLeft: 6, fontSize: 12 }}>
            — {col.comment}
          </span>
        )}
      </span>
    );
  };

  // 构建 Select 选项
  const options = [
    ...localTables.map((t) => ({
      label: isImported(t.name) ? `📥 ${t.name}${t.comment ? ` (${t.comment})` : ""}` : `${t.name}${t.comment ? ` (${t.comment})` : ""}`,
      value: t.name,
    })),
    {
      label: (
        <span style={{ color: "#1677ff" }}>
          <PlusOutlined style={{ marginRight: 4 }} />
          导入新数据表
        </span>
      ),
      value: "__import__",
    },
  ];

  return (
    <>
      <Select
        style={{ width: 280 }}
        placeholder={loading ? <Spin size="small" /> : "查看数据表结构"}
        allowClear
        showSearch
        value={undefined}
        onChange={handleSelect}
        options={options}
        suffixIcon={<InfoCircleOutlined />}
        filterOption={(input, option) => {
          // 导入选项始终显示
          if (option?.value === "__import__") return true;
          // 安全类型检查
          const label = typeof option?.label === "string" ? option.label : "";
          return label.toLowerCase().includes(input.toLowerCase());
        }}
      />

      {/* 表结构详情弹窗 */}
      <Modal
        title={`表结构: ${selectedTable?.name || ""}`}
        open={modalOpen}
        onCancel={() => setModalOpen(false)}
        footer={
          selectedTable && isImported(selectedTable.name) ? (
            <Popconfirm
              title="确认删除此表？"
              description="删除后数据表和所有关联元数据将被清除"
              onConfirm={handleDeleteTable}
              okText="确认删除"
              cancelText="取消"
              okButtonProps={{ danger: true }}
            >
              <Button danger icon={<DeleteOutlined />}>
                删除此表
              </Button>
            </Popconfirm>
          ) : null
        }
        width={600}
      >
        {selectedTable && (
          <>
            <Descriptions column={1} size="small" bordered>
              <Descriptions.Item label="表名">
                {selectedTable.name}
                {isImported(selectedTable.name) && (
                  <Tag color="green" style={{ marginLeft: 8 }}>已导入</Tag>
                )}
              </Descriptions.Item>
              <Descriptions.Item label="描述">
                {selectedTable.comment || "-"}
              </Descriptions.Item>
              <Descriptions.Item label="主键">
                {selectedTable.primary_keys.join(", ") || "-"}
              </Descriptions.Item>
              <Descriptions.Item label="外键">
                {selectedTable.foreign_keys.length > 0
                  ? selectedTable.foreign_keys.map(
                      (fk) =>
                        `${fk.column} → ${fk.ref_table}.${fk.ref_column}`,
                    ).join("; ")
                  : "-"}
              </Descriptions.Item>
            </Descriptions>
            <div style={{ marginTop: 16 }}>
              <strong>字段列表</strong>
              <div
                style={{
                  marginTop: 8,
                  maxHeight: 300,
                  overflowY: "auto",
                }}
              >
                {selectedTable.columns.map((col) => (
                  <div
                    key={col.name}
                    style={{
                      padding: "6px 8px",
                      borderBottom: "1px solid #f0f0f0",
                    }}
                  >
                    {renderColumnTag(col)}
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </Modal>

      {/* 导入弹窗 */}
      <ImportTableModal
        open={importModalOpen}
        onClose={() => setImportModalOpen(false)}
        onSuccess={handleImportSuccess}
      />
    </>
  );
}
