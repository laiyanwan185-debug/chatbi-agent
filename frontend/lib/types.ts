/* API 契约类型定义 — 与 backend/app/api/routes.py 同步 */

export interface TraceStep {
  stage: string;
  status: string;        // "success" | "failed" | "replanned" | "cached"
  latency_ms: number;
  detail: string;
  replan?: ReplanDetail | null;
  score?: number | null;
}

export interface ReplanDetail {
  error_analysis: string;
  revised_plan: string;
  action_execution: string;
}

export interface DebugTraces {
  trace_id: string;
  total_latency_ms: number;
  steps: TraceStep[];
}

export interface DagSummary {
  total_latency_ms: number;
  dag_status: string;
  node_count: number;
}

export interface QueryResponseData {
  query?: string;
  interpretation?: string;
  visualization_mode?: "chart" | "table" | "mixed";
  chart_type?: string | null;
  chart_data?: Record<string, unknown> | null;
  table_data?: Record<string, unknown>[] | null;
  table_columns?: string[] | null;
  analysis_type?: string;
  indicators?: string[];
  data_warning?: boolean;
  dag_summary?: DagSummary;
  /** non-success 时返回 */
  message?: string;
  suggestions?: string[];
}

export interface QueryResponse {
  status: string;         // "success" | "clarify" | "error"
  data: QueryResponseData | null;
  debug_traces: DebugTraces | null;
}

export interface SchemaColumn {
  name: string;
  type: string;
  comment: string | null;
  is_pk: boolean;
  nullable: boolean;
}

export interface SchemaForeignKey {
  column: string;
  ref_table: string;
  ref_column: string;
}

export interface SchemaTable {
  name: string;
  comment: string | null;
  columns: SchemaColumn[];
  primary_keys: string[];
  foreign_keys: SchemaForeignKey[];
}

export interface SchemaResponse {
  tables: SchemaTable[];
}

/* 前端内部消息类型 */
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  response?: QueryResponseData | null;
  debugTraces?: DebugTraces | null;
  loading?: boolean;
  error?: string | null;
}
