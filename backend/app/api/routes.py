"""API 路由 — 全链路管线 + 元数据接口。

路由：
  POST /api/query       全链路管线（含 LLM 解读 + 强制图表）
  POST /api/query/stream SSE 流式版（分阶段推送进度事件）
  POST /api/chat        POST /api/query 的别名（向后兼容）
  GET  /api/health      健康检查
  GET  /api/schema      表结构元数据
  GET  /api/tables/{name}  单表详情
  GET  /api/trace/{id}      Trace 详情（预留）
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from typing import Any, AsyncGenerator

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.engine.trace_logger import start_trace, end_trace, add_step

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


# =============================================================================
# 请求 / 响应模型
# =============================================================================

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求。"""
    query: str = Field(..., min_length=1, max_length=1000, description="用户问句")


class ChatResponse(BaseModel):
    """聊天响应。"""
    status: str = Field(..., description="响应状态: success / clarify / error")
    data: dict[str, Any] | None = Field(None, description="结果数据")
    debug_traces: dict[str, Any] | None = Field(None, description="全链路追踪")


# =============================================================================
# 路由
# =============================================================================


@router.post("/query", response_model=ChatResponse)
async def query(request: ChatRequest, fastapi_request: Request) -> dict[str, Any]:
    """全链路查询端点。"""
    trace_id = start_trace()
    logger.info("Query request: trace=%s, query='%s'", trace_id, request.query[:80])

    response: ChatResponse | None = None
    try:
        pool = getattr(fastapi_request.app.state, "pool", None)
        response = await _execute_pipeline(request.query, pool)
        return response
    except Exception as exc:
        logger.exception("Query processing failed: trace=%s", trace_id)
        traces = end_trace()
        response = ChatResponse(
            status="error",
            data={"message": f"系统处理异常: {exc}"},
            debug_traces=traces,
        )
        return response
    finally:
        # 持久化 trace 到 DiskCache（无论成功/失败均执行）
        if response is not None and response.debug_traces is not None:
            try:
                ts = getattr(fastapi_request.app.state, "trace_storage", None)
                if ts is not None:
                    ts.save(trace_id, response.debug_traces)
            except Exception:
                logger.warning("Failed to persist trace %s", trace_id, exc_info=True)


@router.post("/chat", response_model=ChatResponse, include_in_schema=False)
async def chat(request: ChatRequest, fastapi_request: Request) -> dict[str, Any]:
    """POST /api/query 的别名（向后兼容）。"""
    return await query(request, fastapi_request)


@router.post("/query/stream")
async def query_stream(request: ChatRequest, fastapi_request: Request) -> StreamingResponse:
    """SSE 流式查询：分阶段推送进度事件，最后推送完整结果。"""
    trace_id = start_trace()
    logger.info("Stream query: trace=%s, query='%s'", trace_id, request.query[:80])
    pool = getattr(fastapi_request.app.state, "pool", None)
    return StreamingResponse(
        _pipeline_stream(request.query, pool, trace_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """健康检查。"""
    pool = getattr(request.app.state, "pool", None)
    if pool is not None:
        try:
            result = await pool.health()
            return result
        except Exception:
            return {"status": "unhealthy", "error": "DB pool health check failed"}
    return {"status": "ok", "note": "no db pool"}


@router.get("/schema")
async def get_schema(request: Request) -> dict[str, Any]:
    """返回所有表结构元数据。"""
    schema = getattr(request.app.state, "schema", None)
    if schema is None or schema._schema is None:
        raise HTTPException(status_code=503, detail="Schema not loaded")

    tables: list[dict[str, Any]] = []
    for name, tbl in schema._schema.tables.items():
        tables.append({
            "name": name,
            "comment": tbl.comment,
            "columns": [
                {
                    "name": c.name,
                    "type": c.dtype,
                    "comment": c.comment,
                    "is_pk": c.is_pk,
                    "nullable": c.nullable,
                }
                for c in tbl.columns
            ],
            "primary_keys": tbl.primary_keys,
            "foreign_keys": [
                {"column": fk.column, "ref_table": fk.ref_table, "ref_column": fk.ref_column}
                for fk in tbl.foreign_keys
            ],
        })

    return {"tables": tables}


@router.get("/tables/{table_name:path}")
async def get_table_detail(table_name: str, request: Request) -> dict[str, Any]:
    """返回单表详情。"""
    schema = getattr(request.app.state, "schema", None)
    if schema is None or schema._schema is None:
        raise HTTPException(status_code=503, detail="Schema not loaded")

    tbl = schema._schema.tables.get(table_name)
    if tbl is None:
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")

    return {
        "name": tbl.name,
        "comment": tbl.comment,
        "columns": [
            {
                "name": c.name,
                "type": c.dtype,
                "comment": c.comment,
                "is_pk": c.is_pk,
                "nullable": c.nullable,
            }
            for c in tbl.columns
        ],
        "primary_keys": tbl.primary_keys,
        "foreign_keys": [
            {"column": fk.column, "ref_table": fk.ref_table, "ref_column": fk.ref_column}
            for fk in tbl.foreign_keys
        ],
    }


@router.get("/trace/{trace_id}")
async def get_trace(trace_id: str, request: Request) -> dict[str, Any]:
    """返回指定 trace_id 的全链路追踪详情。"""
    ts = getattr(request.app.state, "trace_storage", None)
    if ts is None:
        raise HTTPException(status_code=503, detail="Trace storage not available")
    data = ts.load(trace_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found or expired")
    return data


# =============================================================================
# 管线执行
# =============================================================================


async def _execute_pipeline(query_text: str, pool: Any = None) -> ChatResponse:
    """执行全链路查询管线。"""
    # ── 延迟导入（避免 lifespan 时机问题）──
    from app.engine.parser import parser_engine
    from app.engine.feedback_gate import feedback_gate
    from app.engine.orchestrator import orchestrator
    from app.engine.registry import registry
    from app.evaluator.metrics import FiveDimEvaluator

    # ── Step 1: Parse ──
    t0 = time.monotonic()
    parse_result = await parser_engine.parse(query_text)
    t1 = time.monotonic()
    add_step("parser", "success", (t1 - t0) * 1000,
             f"status={parse_result.get('status')}")

    # ── Step 2: Gate① 置信度校验 ──
    gate1 = feedback_gate.check_confidence(parse_result)
    add_step("gate.confidence", "passed" if gate1.passed else "failed",
             _step_latency(gate1), gate1.reason, score=round(gate1.score, 3))

    if not gate1.passed:
        status = parse_result.get("status", "error")
        if status in ("reply", "clarify", "greeting"):
            traces = end_trace()
            return ChatResponse(
                status=status,
                data={"message": parse_result.get("message", parse_result.get("detail", ""))},
                debug_traces=traces,
            )
        traces = end_trace()
        return ChatResponse(
            status="clarify",
            data={"message": gate1.reason, "suggestions": gate1.suggestions},
            debug_traces=traces,
        )

    # ── Step 3: Orchestrator 构建 DAG ──
    t0 = time.monotonic()
    dag_plan = orchestrator.build_plan(parse_result, registry)
    t1 = time.monotonic()
    add_step("orchestrator.build", "success", (t1 - t0) * 1000,
             f"{dag_plan.size} nodes, {len(dag_plan.level_groups)} levels")

    # ── Step 4: Gate② 计划校验 ──
    req_ctx: dict[str, Any] = {"permissions": ["user"]}
    gate2 = feedback_gate.check_plan(dag_plan, req_ctx, registry)
    add_step("gate.plan", "passed" if gate2.passed else "failed",
             _step_latency(gate2), gate2.reason, score=round(gate2.score, 3))

    if not gate2.passed:
        traces = end_trace()
        return ChatResponse(
            status="error",
            data={"message": f"DAG 计划校验不通过: {gate2.reason}",
                  "suggestions": gate2.suggestions},
            debug_traces=traces,
        )

    # ── Step 5: Executor 执行 ──
    t0 = time.monotonic()
    exec_result = await orchestrator.execute(dag_plan, req_ctx, pool, registry)
    t1 = time.monotonic()
    exec_status = "success" if exec_result.dag_status == "full" else "partial"
    add_step("executor", exec_status, (t1 - t0) * 1000,
             f"dag_status={exec_result.dag_status}, "
             f"nodes={len(exec_result.nodes_execution)}, "
             f"rows={_df_rows(exec_result.final_data)}")

    # ── Step 6: Gate③ 约束 + 结果校验 ──
    evaluator = FiveDimEvaluator()
    exec_dict = {
        "final_data": exec_result.final_data,
        "success": exec_result.success,
        "dag_status": exec_result.dag_status,
        "data_warning": exec_result.data_warning,
        "total_latency_ms": exec_result.total_latency_ms,
        "nodes_execution": [
            {"node_id": ne.node_id, "status": ne.status.value,
             "latency_ms": ne.latency_ms, "error": ne.error}
            for ne in exec_result.nodes_execution
        ],
    }
    gate3 = await feedback_gate.check_constraint(parse_result, dag_plan, exec_dict, evaluator)
    add_step("gate.constraint", "passed" if gate3.passed else "failed",
             _step_latency(gate3), gate3.reason, score=round(gate3.score, 3))

    # ── Step 7: LLM 智能解读（替换旧 fallback）──
    interpretation = await _llm_interpret(query_text, parse_result, exec_result)
    add_step("interpreter", "success", 0, "LLM interpreter")

    # ── Step 8: Gate④ 输出校验 ──
    gate4 = feedback_gate.check_output(interpretation, None)
    add_step("gate.output", "passed" if gate4.passed else "failed",
             _step_latency(gate4), gate4.reason, score=round(gate4.score, 3))

    # ── Step 9: 构建响应 ──
    response_data = _build_chart_response(query_text, parse_result, exec_result, interpretation)
    response_data["data_warning"] = exec_result.data_warning

    traces = end_trace()
    return ChatResponse(status="success", data=response_data, debug_traces=traces)


async def _pipeline_stream(
    query_text: str,
    pool: Any,
    trace_id: str,
) -> AsyncGenerator[str, None]:
    """SSE 流式管线：分阶段推送进度事件，最后推送完整结果。

    Yields SSE-format strings:
      data: {"type":"stage","stage":"解析","elapsed_ms":1520,"status":"success"}
      data: {"type":"result","status":"success","data":{...}}
    """
    import json

    def _sse(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    t_start = time.monotonic()

    # ── 延迟导入 ──
    from app.engine.parser import parser_engine
    from app.engine.feedback_gate import feedback_gate
    from app.engine.orchestrator import orchestrator
    from app.engine.registry import registry
    from app.evaluator.metrics import FiveDimEvaluator

    try:
        # ── Step 1: Parse ──
        t0 = time.monotonic()
        parse_result = await parser_engine.parse(query_text)
        t1 = time.monotonic()
        elapsed = int((t1 - t_start) * 1000)
        add_step("parser", "success", (t1 - t0) * 1000,
                 f"status={parse_result.get('status')}")
        yield _sse({"type": "stage", "stage": "语义解析", "elapsed_ms": elapsed, "status": "success"})

        # ── Step 2: Gate① ──
        gate1 = feedback_gate.check_confidence(parse_result)
        add_step("gate.confidence", "passed" if gate1.passed else "failed",
                 _step_latency(gate1), gate1.reason, score=round(gate1.score, 3))

        if not gate1.passed:
            status = parse_result.get("status", "error")
            if status in ("reply", "clarify", "greeting"):
                traces = end_trace()
                yield _sse({"type": "result", "status": status,
                            "data": {"message": parse_result.get("message", parse_result.get("detail", ""))},
                            "debug_traces": traces})
                return
            traces = end_trace()
            yield _sse({"type": "result", "status": "clarify",
                        "data": {"message": gate1.reason, "suggestions": gate1.suggestions},
                        "debug_traces": traces})
            return

        yield _sse({"type": "stage", "stage": "校验通过", "elapsed_ms": int((time.monotonic() - t_start) * 1000),
                    "status": "success"})

        # ── Step 3: Build DAG ──
        t0 = time.monotonic()
        dag_plan = orchestrator.build_plan(parse_result, registry)
        t1 = time.monotonic()
        add_step("orchestrator.build", "success", (t1 - t0) * 1000,
                 f"{dag_plan.size} nodes, {len(dag_plan.level_groups)} levels")
        yield _sse({"type": "stage", "stage": "构建执行计划", "elapsed_ms": int((t1 - t_start) * 1000),
                    "status": "success"})

        # ── Step 4: Gate② ──
        req_ctx = {"permissions": ["user"]}
        gate2 = feedback_gate.check_plan(dag_plan, req_ctx, registry)
        add_step("gate.plan", "passed" if gate2.passed else "failed",
                 _step_latency(gate2), gate2.reason, score=round(gate2.score, 3))
        if not gate2.passed:
            traces = end_trace()
            yield _sse({"type": "result", "status": "error",
                        "data": {"message": f"DAG 计划校验不通过: {gate2.reason}",
                                  "suggestions": gate2.suggestions},
                        "debug_traces": traces})
            return

        # ── Step 5: Execute ──
        t0 = time.monotonic()
        yield _sse({"type": "stage", "stage": "执行数据查询", "elapsed_ms": int((t0 - t_start) * 1000),
                    "status": "running"})
        exec_result = await orchestrator.execute(dag_plan, req_ctx, pool, registry)
        t1 = time.monotonic()
        exec_status = "success" if exec_result.dag_status == "full" else "partial"
        add_step("executor", exec_status, (t1 - t0) * 1000,
                 f"dag_status={exec_result.dag_status}, "
                 f"nodes={len(exec_result.nodes_execution)}, "
                 f"rows={_df_rows(exec_result.final_data)}")
        yield _sse({"type": "stage", "stage": "数据查询完成", "elapsed_ms": int((t1 - t_start) * 1000),
                    "status": exec_status})

        # ── Step 6: Gate③ ──
        evaluator = FiveDimEvaluator()
        exec_dict = {
            "final_data": exec_result.final_data,
            "success": exec_result.success,
            "dag_status": exec_result.dag_status,
            "data_warning": exec_result.data_warning,
            "total_latency_ms": exec_result.total_latency_ms,
            "nodes_execution": [
                {"node_id": ne.node_id, "status": ne.status.value,
                 "latency_ms": ne.latency_ms, "error": ne.error}
                for ne in exec_result.nodes_execution
            ],
        }
        gate3 = await feedback_gate.check_constraint(parse_result, dag_plan, exec_dict, evaluator)
        add_step("gate.constraint", "passed" if gate3.passed else "failed",
                 _step_latency(gate3), gate3.reason, score=round(gate3.score, 3))

        # ── Step 7: LLM 解读 ──
        yield _sse({"type": "stage", "stage": "生成智能解读", "elapsed_ms": int((time.monotonic() - t_start) * 1000),
                    "status": "running"})
        interpretation = await _llm_interpret(query_text, parse_result, exec_result)
        add_step("interpreter", "success", 0, "LLM interpreter")

        # ── Step 8: Gate④ ──
        gate4 = feedback_gate.check_output(interpretation, None)
        add_step("gate.output", "passed" if gate4.passed else "failed",
                 _step_latency(gate4), gate4.reason, score=round(gate4.score, 3))

        # ── Step 9: Build response ──
        response_data = _build_chart_response(query_text, parse_result, exec_result, interpretation)
        response_data["data_warning"] = exec_result.data_warning

        traces = end_trace()
        yield _sse({"type": "result", "status": "success",
                    "data": response_data, "debug_traces": traces})

    except Exception as exc:
        logger.exception("Stream pipeline failed: trace=%s", trace_id)
        traces = end_trace()
        yield _sse({"type": "result", "status": "error",
                    "data": {"message": f"系统处理异常: {exc}"},
                    "debug_traces": traces})


# =============================================================================
# Helper utilities
# =============================================================================

ANALYSIS_TYPE_LABELS = {
    "trend":       "趋势分析",
    "rank":        "排行对比",
    "detail":      "维度明细",
    "correlation": "相关性分析",
    "anomaly":     "异常检测",
    "composite":   "综合评价",
    "advanced":    "高级分析",
}


def _step_latency(gate_result: Any) -> float:
    """从 GateResult 估算耗时（闸门同步执行，近似 0）。"""
    return 0.0


def _df_rows(data: Any) -> int:
    """获取 DataFrame 行数。"""
    if isinstance(data, pd.DataFrame):
        return len(data)
    if isinstance(data, dict):
        return len(data)
    return 0


def _resolve_display_data(exec_result: Any) -> pd.DataFrame | None:
    """从执行结果中解析出可展示的 DataFrame。"""
    if isinstance(exec_result.raw_data, pd.DataFrame) and not exec_result.raw_data.empty:
        return exec_result.raw_data
    if isinstance(exec_result.final_data, pd.DataFrame) and not exec_result.final_data.empty:
        return exec_result.final_data
    if isinstance(exec_result.final_data, dict) and exec_result.final_data:
        for df in exec_result.final_data.values():
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
    return None


# =============================================================================
# LLM 智能解读
# =============================================================================

_INTERPRET_PROMPT = """你是一个资深数据分析师。请基于以下查询结果，拆解用户问题中的子问题并逐一回答。

========== 用户原始问题 ==========
{question}

========== 系统执行摘要 ==========
分析类型：{analysis_type_label}
查询指标：{indicators}
时间范围：{time_range}
执行算法：{algorithms}
DAG 状态：{dag_status}

========== 分析中间结果 ==========
{analysis_results}

========== 查询数据预览 ==========
{data_preview}

========== 输出要求 ==========
请按以下结构输出纯文字解读（不要加任何 markdown 符号 # * _ ` > 等）：

【问题拆解与逐条回答】

第一步：识别用户问题中包含哪些具体的分析诉求/子问题。
第二步：逐一回答每个子问题，格式如下：

① <子问题1> → ✓ 基于现有数据分析结论：<引用数据给出具体结论>
② <子问题2> → ✗ 系统能力说明：<诚实说明为什么无法回答，缺什么数据或分析能力，给出建议>
③ ...

【核心数据结论】
基于可执行的分析，给出 2-3 句最核心的发现，引用具体数值。

【详细分析】
对每个可回答的子问题展开说明，引用数据和分析结果中的具体数值。

【业务建议】
基于分析结论给出可操作的业务建议，每一条对应一个具体行动方向。

注意：
- 诚实！系统分析了什么就写什么，没做的分析不能假装做了
- 每个子问题都要有明确结论：可回答就引用数据，不可回答就说明原因和建议
- 如果有具体数值，必须引用数值支撑结论
- 不要输出 "根据提供的数据" 等废话，直接写结论"""


def _build_analysis_summary(exec_result: Any) -> str:
    """从 ExecutionResult 构建分析中间结果摘要字符串。"""
    parts = []

    # DAG 执行摘要
    dag_status_label = {"full": "完整执行", "partial": "部分执行", "failed": "执行失败"}
    parts.append(f"DAG状态：{dag_status_label.get(getattr(exec_result, 'dag_status', ''), getattr(exec_result, 'dag_status', '未知'))}")

    # 节点执行列表
    node_lines = []
    for ne in getattr(exec_result, "nodes_execution", []):
        status_icon = {"success": "OK", "failed": "FAIL", "skipped": "SKIP", "pending": "PEND"}
        icon = status_icon.get(ne.status.value if hasattr(ne.status, 'value') else str(ne.status), "?")
        err = f" error={ne.error[:100]}" if ne.error else ""
        node_lines.append(f"  [{icon}] {ne.node_id} ({ne.latency_ms}ms){err}")
    if node_lines:
        parts.append("节点执行：")
        parts.extend(node_lines)

    # 分析节点中间结果
    analysis_results = getattr(exec_result, "analysis_results", None)
    if analysis_results:
        parts.append("")
        parts.append("各算法分析结果：")
        for node_id, result in analysis_results.items():
            parts.append(f"  ── {node_id} ──")
            if isinstance(result, dict):
                for k, v in result.items():
                    parts.append(f"    {k}: {v}")
            elif isinstance(result, str):
                parts.append(f"    {result[:200]}")

    return "\n".join(parts)


async def _llm_interpret(
    query_text: str,
    parse_result: dict[str, Any],
    exec_result: Any,
) -> str:
    """LLM 智能解读，产出针对用户子问题的逐条回答。"""
    from app.core.llm import chat_with_model

    execution_plan = parse_result.get("execution_plan", {})
    indicators = execution_plan.get("indicators", [])
    analysis_type = execution_plan.get("analysis_type", "detail")
    type_label = ANALYSIS_TYPE_LABELS.get(analysis_type, analysis_type)
    time_range = execution_plan.get("time_range", {})

    # 构建执行摘要
    algorithms = ", ".join(
        getattr(exec_result, "_algorithm_names", [])
    ) or type_label
    dag_status = getattr(exec_result, "dag_status", "unknown")
    dag_status_label = {"full": "完整执行", "partial": "部分执行", "failed": "执行失败", "unknown": "未知"}

    # 分析中间结果
    analysis_results_str = _build_analysis_summary(exec_result)

    # 数据预览
    display_df = _resolve_display_data(exec_result)
    if display_df is not None and not display_df.empty:
        data_preview = display_df.head(10).to_markdown(index=False, numalign="left")
    else:
        data_preview = "（无有效数据返回）"

    tr_parts = []
    if time_range.get("start"):
        tr_parts.append(str(time_range["start"]))
    if time_range.get("end"):
        tr_parts.append(str(time_range["end"]))

    prompt = _INTERPRET_PROMPT.format(
        question=query_text,
        indicators="、".join(indicators) if indicators else "未指定",
        analysis_type_label=type_label,
        time_range=" ~ ".join(tr_parts) if tr_parts else "未指定",
        algorithms=algorithms,
        dag_status=dag_status_label.get(dag_status, dag_status),
        analysis_results=analysis_results_str,
        data_preview=data_preview,
    )

    try:
        resp = await chat_with_model("openai", [
            {"role": "user", "content": prompt},
        ], temperature=0.3, max_tokens=2048)
        raw = resp.strip() if resp else ""
        if raw:
            # 清洗 markdown 符号
            raw = _clean_markdown(raw)
            return raw
    except Exception as exc:
        logger.warning("LLM 解读失败，降级到规则: %s", exc)

    # 降级：规则版
    return _fallback_interpret(query_text, parse_result, exec_result)


def _clean_markdown(text: str) -> str:
    """剔除 markdown 符号。"""
    text = re.sub(r'[#*_~`>]', '', text)
    # 多余空行合并
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _fallback_interpret(
    query_text: str,
    parse_result: dict[str, Any],
    exec_result: Any,
) -> str:
    """备用解读：将执行摘要 + 数据转为结构化纯文字。"""
    execution_plan = parse_result.get("execution_plan", {})
    lines: list[str] = []

    # 用户问题
    lines.append(f"用户问题：{query_text}")
    lines.append("")

    indicators = execution_plan.get("indicators", [])
    analysis_type = execution_plan.get("analysis_type", "unknown")
    type_label = ANALYSIS_TYPE_LABELS.get(analysis_type, analysis_type)
    time_range = execution_plan.get("time_range", {})

    dag_status = getattr(exec_result, "dag_status", "")
    dag_status_label = {"full": "完整执行", "partial": "部分执行", "failed": "执行失败"}
    status_text = dag_status_label.get(dag_status, dag_status)

    lines.append(f"【执行摘要】")
    lines.append(f"分析类型：{type_label}")
    lines.append(f"DAG 状态：{status_text}")
    if indicators:
        lines.append(f"查询指标：{'、'.join(indicators)}")
    if time_range.get("start") or time_range.get("end"):
        tr = f"时间范围：{time_range.get('start', '')} ~ {time_range.get('end', '')}"
        lines.append(tr)
    lines.append("")

    # 分析中间结果
    analysis_results = getattr(exec_result, "analysis_results", None)
    if analysis_results:
        lines.append("【分析结果】")
        for node_id, result in analysis_results.items():
            algo_name = node_id.replace("analysis_", "")
            if isinstance(result, dict):
                parts = [f"  {algo_name}: {', '.join(f'{k}={v}' for k, v in result.items())}"]
                lines.extend(parts)
            elif isinstance(result, str):
                lines.append(f"  {algo_name}: {result[:200]}")
        lines.append("")

    display_df = _resolve_display_data(exec_result)

    if display_df is not None and not display_df.empty:
        lines.append(f"【数据摘要】")
        lines.append(f"数据行数：{len(display_df)} 行，{len(display_df.columns)} 列")
        lines.append("")
        lines.append("数据预览：")
        lines.append(display_df.head(10).to_markdown(index=False, numalign="left"))
    else:
        lines.append("查询未返回有效数据")

    if exec_result.data_warning:
        lines.append("")
        lines.append("注意：部分节点执行异常，结果可能不完整")

    return "\n".join(lines)


# =============================================================================
# 图表生成
# =============================================================================


def _build_chart_response(
    query_text: str,
    parse_result: dict[str, Any],
    exec_result: Any,
    interpretation: str,
) -> dict[str, Any]:
    """构建前端渲染所需的数据结构。

    data 结构：
      query, interpretation, visualization_mode, chart_type, chart_data,
      table_data, table_columns, analysis_type, indicators, dag_summary
    """
    execution_plan = parse_result.get("execution_plan", {})
    analysis_type = execution_plan.get("analysis_type", "detail")

    # 图表类型映射（detail 也映射为 bar）
    from app.evaluator.metrics import CHART_REQUIREMENTS
    chart_req = CHART_REQUIREMENTS.get(analysis_type, {})
    chart_type = chart_req.get("chart", "bar")
    if chart_type == "table":
        chart_type = "bar"

    # 解析展示数据
    display_df = _resolve_display_data(exec_result)

    table_data: list[dict[str, Any]] | None = None
    table_columns: list[str] | None = None
    chart_data: dict[str, Any] | None = None

    if display_df is not None and not display_df.empty:
        from app.engine.sandbox import clean_dataframe
        cleaned = clean_dataframe(display_df)
        table_data = cleaned.to_dict(orient="records")
        table_columns = cleaned.columns.tolist()

        # 尝试标准图表
        chart_data = _df_to_echarts(cleaned, chart_type)

    # 保底：任何情况都生成一个图表
    if not chart_data or not chart_data.get("series"):
        chart_data = _force_universal_chart(display_df)

    indicators = execution_plan.get("indicators", [])

    return {
        "query": query_text,
        "interpretation": interpretation,
        "visualization_mode": "chart",
        "chart_type": chart_type,
        "chart_data": chart_data,
        "table_data": table_data,
        "table_columns": table_columns,
        "analysis_type": analysis_type,
        "indicators": indicators,
        "data_warning": exec_result.data_warning,
        "dag_summary": {
            "total_latency_ms": exec_result.total_latency_ms,
            "dag_status": exec_result.dag_status,
            "node_count": len(exec_result.nodes_execution),
        },
    }


def _force_universal_chart(df: pd.DataFrame | None) -> dict[str, Any]:
    """万能保底图表：任何数据都能生成一个 bar 图。"""
    if df is not None and not df.empty:
        cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        # 场景1：有分类列 + 数值列 → 分类为 X，数值为系列
        if len(cols) >= 2 and numeric_cols:
            x_col = _pick_x_col(df, cols, numeric_cols) or cols[0]
            y_cols = [c for c in numeric_cols if c != x_col] or numeric_cols[:1]
            x_data = df[x_col].astype(str).tolist()

            scale_groups = _split_y_cols_by_scale(df, y_cols)

            def _make_one(g_cols):
                return {
                    "xAxis": {"type": "category", "data": x_data},
                    "yAxis": {"type": "value"},
                    "series": [
                        {"name": c, "type": "bar", "data": _to_numeric_list(df[c])}
                        for c in g_cols
                    ],
                }

            if len(scale_groups) == 1:
                return _make_one(scale_groups[0])

            return {
                "charts": [_make_one(g) for g in scale_groups],
                "intro": _MULTI_CHART_INTRO,
            }

        # 场景2：只有一列且为文本 → 做频次统计
        if len(cols) == 1 and not numeric_cols:
            freq = df[cols[0]].value_counts().head(20)
            return {
                "xAxis": {"type": "category", "data": freq.index.astype(str).tolist()},
                "yAxis": {"type": "value"},
                "series": [{
                    "name": f"{cols[0]} 分布",
                    "type": "bar",
                    "data": freq.values.tolist(),
                }],
            }

        # 场景3：单列纯数值或混合型 → 行号为 X，该列为 Y
        col = cols[0]
        return {
            "xAxis": {"type": "category", "data": [str(i + 1) for i in range(len(df))]},
            "yAxis": {"type": "value"},
            "series": [
                {"name": col, "type": "bar",
                 "data": _to_numeric_list(df[col]) if col in numeric_cols
                         else [float(i + 1) for i in range(len(df))]},
            ],
        }

    # 无数据
    return {
        "xAxis": {"type": "category", "data": ["暂无数据"]},
        "yAxis": {"type": "value"},
        "series": [{"name": "数据", "type": "bar", "data": []}],
    }


# 时间/维度列名集合（即使为数值类型也优先作 X 轴）
_TIME_DIM_COL_NAMES = frozenset({
    "统计年份", "year", "年份", "季度", "统计季度", "月份", "月",
    "period", "time", "date", "日期", "年度", "month",
})


def _pick_x_col(df: pd.DataFrame, cols: list[str], numeric_cols: list[str]) -> str | None:
    """为折线/柱状图选择最合适的 X 轴列。

    优先级：
      1. 非数值列（文本维度列）
      2. 时间/维度列名匹配（如 统计年份，即使为 INTEGER 也作 X 轴）
      3. 低基数数值列（唯一值 ≤ 30，适合作为维度轴）
    """
    # 1. 非数值列
    x = next((c for c in cols if c not in numeric_cols), None)
    if x:
        return x
    # 2. 时间列名匹配（如 统计年份、季度）
    x = next((c for c in cols if c in _TIME_DIM_COL_NAMES), None)
    if x:
        return x
    # 3. 低基数数值列（≤30 个唯一值，适合作维度轴）
    for c in cols:
        if c in numeric_cols and df[c].nunique(dropna=False) <= 30:
            return c
    return None


def _compute_max_scale(col: str, df: pd.DataFrame) -> float:
    """计算列的最大量级（log10）。"""
    vals = df[col].dropna().abs()
    mx = float(vals.max()) if len(vals) > 0 else 0.0
    return math.floor(math.log10(mx)) if mx > 0 else 0.0


def _split_y_cols_by_scale(
    df: pd.DataFrame,
    y_cols: list[str],
    *,
    max_scale_diff: int = 1,
) -> list[list[str]]:
    """按数值量级分组 Y 列，避免单图中大值列遮蔽小值列。

    Args:
        df: 源数据
        y_cols: 待分组的数值列名列表
        max_scale_diff: 同组允许的最大量级差（默认 1 = 同一数量级）

    Returns:
        分组列表，如 [["GDP", "固定资产投资"], ["城镇化率", "失业率"]]
    """
    if len(y_cols) <= 1:
        return [y_cols]

    scales = {c: _compute_max_scale(c, df) for c in y_cols}
    sorted_cols = sorted(y_cols, key=lambda c: scales[c], reverse=True)

    groups: list[list[str]] = [[sorted_cols[0]]]
    for c in sorted_cols[1:]:
        if scales[groups[-1][0]] - scales[c] <= max_scale_diff:
            groups[-1].append(c)
        else:
            groups.append([c])
    return groups


_MULTI_CHART_INTRO = (
    "因各指标数值量级差异较大，已拆分为下图分别展示"
)


def _make_single_chart(x_data: list[str], group_cols: list[str], chart_type: str, df: pd.DataFrame) -> dict[str, Any]:
    """生成单个 ECharts option 字典。"""
    return {
        "xAxis": {"type": "category", "data": x_data},
        "yAxis": {"type": "value"},
        "series": [
            {"name": c, "type": chart_type, "data": _to_numeric_list(df[c])}
            for c in group_cols
        ],
    }


def _df_to_echarts(df: pd.DataFrame, chart_type: str) -> dict[str, Any]:
    """DataFrame → ECharts option 格式。

    当 bar/line 的多系列量级差 >1 个数量级时，自动拆分为多个子图表。
    """
    cols = df.columns.tolist()
    if not cols:
        return {}

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if chart_type in ("line", "bar"):
        x_col = _pick_x_col(df, cols, numeric_cols)
        if x_col:
            x_data = df[x_col].astype(str).tolist()
            y_cols = [c for c in numeric_cols if c != x_col] or numeric_cols[:1]
        else:
            x_data = [str(i + 1) for i in range(len(df))]
            y_cols = numeric_cols[:1]
        if not y_cols:
            return {}

        scale_groups = _split_y_cols_by_scale(df, y_cols)

        if len(scale_groups) == 1:
            return _make_single_chart(x_data, scale_groups[0], chart_type, df)

        return {
            "charts": [_make_single_chart(x_data, g, chart_type, df) for g in scale_groups],
            "intro": _MULTI_CHART_INTRO,
        }

    if chart_type == "scatter":
        num_cols = [c for c in cols if c in numeric_cols]
        if len(num_cols) >= 2:
            return {
                "xAxis": {"type": "value"},
                "yAxis": {"type": "value"},
                "series": [{
                    "type": "scatter",
                    "data": [[float(row[c]) for c in num_cols[:2]]
                             for _, row in df.iterrows()],
                }],
            }
        return {}

    if chart_type == "pie":
        if len(cols) >= 2:
            x_col, y_col = cols[0], cols[1] if cols[1] in numeric_cols else (numeric_cols[0] if numeric_cols else cols[1])
            return {
                "series": [{
                    "type": "pie",
                    "data": [
                        {"name": str(row[x_col]), "value": float(row[y_col])}
                        for _, row in df.iterrows()
                    ],
                }],
            }
        return {}

    if chart_type == "radar":
        num_cols = [c for c in cols if c in numeric_cols]
        if len(num_cols) >= 2:
            indicators_list = [{"name": c, "max": float(df[c].max())} for c in num_cols]
            # 每个省份生成一个独立 series，ECharts 自动分配不同颜色
            entity_names = df[cols[0]].astype(str).tolist()
            series = [
                {
                    "name": name,
                    "type": "radar",
                    "data": [{"value": [float(row[c]) for c in num_cols]}],
                }
                for name, (_, row) in zip(entity_names, df.iterrows())
            ]
            return {
                "radar": {"indicator": indicators_list},
                "legend": {"type": "scroll", "bottom": 0, "data": entity_names},
                "series": series,
            }
        return {}

    if chart_type == "scatter_line":
        x_col = _pick_x_col(df, cols, numeric_cols) or cols[0]
        y_cols = [c for c in numeric_cols if c != x_col] or numeric_cols[:1]
        return {
            "xAxis": {"type": "category", "data": df[x_col].astype(str).tolist()},
            "yAxis": {"type": "value"},
            "series": [
                {
                    "name": c,
                    "type": "line",
                    "data": _to_numeric_list(df[c]),
                    "smooth": True,
                    "symbol": "circle",
                    "symbolSize": 6,
                }
                for c in y_cols
            ],
        }

    return {}


def _to_numeric_list(series: pd.Series) -> list[float | None]:
    """Series 转为数字列表，非数值型转为 None。"""
    result: list[float | None] = []
    for v in series:
        try:
            result.append(float(v))
        except (ValueError, TypeError):
            result.append(None)
    return result


# =============================================================================
# 数据表导入相关模型
# =============================================================================


class ImportResponse(BaseModel):
    """导入响应。"""
    status: str = "success"
    table_name: str = ""
    row_count: int = 0
    column_count: int = 0
    columns: list[str] = Field(default_factory=list)
    join_relationships: list[dict[str, Any]] = Field(default_factory=list)
    indicator_count: int = 0
    time_column: str | None = None


class ImportErrorResponse(BaseModel):
    """导入错误响应。"""
    status: str = "error"
    error: str = ""


# =============================================================================
# 数据表导入路由
# =============================================================================


# 使用单独的 router 端点处理文件上传
from fastapi import UploadFile, File, Form


@router.post("/tables/upload")
async def upload_table(
    request: Request,
    file: UploadFile = File(...),
    table_name: str | None = Form(None),
    table_comment: str | None = Form(None),
) -> dict[str, Any]:
    """上传文件并导入为数据表。"""
    from app.engine.table_importer import ImportError_

    importer = getattr(request.app.state, "table_importer", None)
    if importer is None:
        raise HTTPException(500, "TableImporter 未初始化")

    # 文件大小限制（50MB）
    MAX_UPLOAD_SIZE = 50 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, f"文件大小超过限制（最大 {MAX_UPLOAD_SIZE // 1024 // 1024}MB）")

    filename = file.filename or "upload.csv"

    try:
        result = await importer.import_file(
            file_content=content,
            filename=filename,
            table_name=table_name,
            table_comment=table_comment,
        )
        return {
            "status": "success",
            "table_name": result.table_name,
            "row_count": result.row_count,
            "column_count": result.column_count,
            "columns": result.columns,
            "join_relationships": [j.model_dump() for j in result.join_relationships],
            "indicator_count": result.indicator_count,
            "time_column": result.time_column,
        }
    except ImportError_ as e:
        # 用户可理解的校验错误
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Table import failed: %s", e)
        raise HTTPException(500, "导入失败，请稍后重试")


@router.get("/tables/imported")
async def get_imported_tables(request: Request) -> dict[str, Any]:
    """获取所有已导入的表名列表。"""
    importer = getattr(request.app.state, "table_importer", None)
    if importer is None:
        return {"tables": []}
    return {"tables": importer.get_imported_tables()}


@router.delete("/tables/{name}")
async def delete_imported_table(name: str, request: Request) -> dict[str, Any]:
    """删除已导入的数据表及其所有关联元数据。"""
    importer = getattr(request.app.state, "table_importer", None)
    if importer is None:
        raise HTTPException(500, "TableImporter 未初始化")

    if name not in importer.get_imported_tables():
        raise HTTPException(404, f"表 '{name}' 不是通过导入创建的")

    try:
        await importer.delete_table(name)
        return {"status": "deleted", "table": name}
    except Exception as e:
        logger.exception("Table delete failed: %s", e)
        raise HTTPException(500, "删除失败，请稍后重试")
