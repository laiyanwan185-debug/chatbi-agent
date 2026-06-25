/* API 客户端 — Axios 实例 + 请求函数 */

import axios from "axios";
import type { QueryResponse, SchemaResponse, DebugTraces } from "./types";

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8004/api";

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 120000,        // 120s — 管线执行最大超时
  headers: { "Content-Type": "application/json" },
});

export async function sendQuery(query: string): Promise<QueryResponse> {
  const { data } = await api.post<QueryResponse>("/query", { query });
  return data;
}

/**
 * SSE 流式查询：分阶段接收进度事件，最后返回完整结果。
 *
 * @param query - 用户问句
 * @param onStage - 每阶段回调 (stage, elapsedMs)
 * @returns 最终 QueryResponse
 */
export async function sendQueryStream(
  query: string,
  onStage?: (stage: string, elapsedMs: number) => void,
): Promise<QueryResponse> {
  const response = await fetch(`${BASE_URL}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: 请求失败`);
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // 解析 SSE 事件 (data: {...}\n\n)
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || ""; // 最后一个可能不完整

    for (const part of parts) {
      const match = part.match(/^data: (.+)/);
      if (!match) continue;

      try {
        const event = JSON.parse(match[1]);

        if (event.type === "stage" && onStage) {
          onStage(event.stage, event.elapsed_ms);
        } else if (event.type === "result") {
          return {
            status: event.status,
            data: event.data ?? null,
            debug_traces: event.debug_traces ?? null,
          } as QueryResponse;
        }
      } catch {
        // 忽略解析失败的事件
      }
    }
  }

  throw new Error("流式响应意外结束");
}

export async function fetchSchema(): Promise<SchemaResponse> {
  const { data } = await api.get<SchemaResponse>("/schema");
  return data;
}

export async function fetchTrace(traceId: string): Promise<DebugTraces> {
  const { data } = await api.get<DebugTraces>(`/trace/${traceId}`);
  return data;
}

// ── 数据表导入 API ──

export interface ImportResponse {
  status: string;
  table_name: string;
  row_count: number;
  column_count: number;
  columns: string[];
  join_relationships: { target: string; type: string; on: string[] }[];
  indicator_count: number;
  time_column: string | null;
}

export async function uploadTable(
  file: File,
  tableName?: string,
  tableComment?: string,
): Promise<ImportResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (tableName) formData.append("table_name", tableName);
  if (tableComment) formData.append("table_comment", tableComment);
  // 不手动设置 Content-Type，让 Axios 自动检测 FormData 并添加正确的 boundary
  const { data } = await api.post<ImportResponse>("/tables/upload", formData, {
    timeout: 120000,
  });
  return data;
}

export async function fetchImportedTables(): Promise<string[]> {
  const { data } = await api.get<{ tables: string[] }>("/tables/imported");
  return data.tables;
}

export async function deleteImportedTable(tableName: string): Promise<void> {
  await api.delete(`/tables/${encodeURIComponent(tableName)}`);
}

export default api;
