import { useMemo, useState } from "react";
import { Table, Button, Input, Space, Empty } from "antd";
import { DownloadOutlined, SearchOutlined } from "@ant-design/icons";
import type { ColumnsType } from "antd/es/table";

interface TableViewProps {
  columns: string[] | null | undefined;
  data: Record<string, unknown>[] | null;
}

export default function TableView({ columns, data }: TableViewProps) {
  const [searchText, setSearchText] = useState("");

  const colList = useMemo(() => columns ?? [], [columns]);

  // CSV 导出
  const handleExportCSV = () => {
    if (!data || data.length === 0 || colList.length === 0) return;

    const BOM = "﻿";
    const header = colList.map((c) => `"${c.replace(/"/g, '""')}"`).join(",");
    const rows = data.map((row) =>
      colList.map((col) => {
        const val = row[col];
        if (val == null) return "";
        const str = String(val);
        return str.includes(",") || str.includes('"') || str.includes("\n")
          ? `"${str.replace(/"/g, '""')}"`
          : str;
      }).join(",")
    );

    const csv = BOM + [header, ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `export_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // 过滤后的数据
  const filteredData = useMemo(() => {
    if (!data) return [];
    if (!searchText.trim()) return data;
    const lower = searchText.toLowerCase();
    return data.filter((row) =>
      colList.some((col) => {
        const val = row[col];
        return val != null && String(val).toLowerCase().includes(lower);
      })
    );
  }, [data, searchText, colList]);

  const antColumns: ColumnsType<Record<string, unknown>> = useMemo(() => {
    return colList.map((col) => ({
      title: col,
      dataIndex: col,
      key: col,
      sorter: (a: Record<string, unknown>, b: Record<string, unknown>) => {
        const va = a[col];
        const vb = b[col];
        if (typeof va === "number" && typeof vb === "number") return va - vb;
        return String(va ?? "").localeCompare(String(vb ?? ""));
      },
      render: (val: unknown) =>
        val === null || val === undefined ? "-" : String(val),
      ellipsis: true,
    }));
  }, [colList]);

  if (!data || data.length === 0) {
    return <Empty description="暂无数据" />;
  }

  return (
    <div>
      <Space style={{ marginBottom: 12, width: "100%", justifyContent: "space-between" }}>
        <Input
          placeholder="搜索全部列..."
          prefix={<SearchOutlined />}
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          allowClear
          style={{ width: 280 }}
        />
        <Button icon={<DownloadOutlined />} onClick={handleExportCSV}>
          导出 CSV
        </Button>
      </Space>
      <Table
        dataSource={filteredData.map((row, i) => ({ ...row, _key: i }))}
        columns={antColumns}
        rowKey="_key"
        pagination={{
          pageSize: 20,
          showSizeChanger: true,
          pageSizeOptions: ["10", "20", "50", "100"],
          showTotal: (total) => `共 ${total} 条`,
        }}
        scroll={{ x: "max-content", y: 420 }}
        size="small"
        locale={{ emptyText: <Empty description="暂无数据" /> }}
      />
    </div>
  );
}
