"""全链路可观测 Trace — Trace_ID + @traceable 装饰器 + ContextVar 上下文管理。

职责边界：
  - TraceContext: 全链路追踪上下文（trace_id + steps + 时间线）
  - @traceable(stage_name): 自动记录函数执行时长的装饰器
  - start_trace() / end_trace(): 全局 ContextVar 生命周期管理

调用方：
  - routes.py: 每请求先 start_trace()，响应注入 debug_traces
  - Engine 各层: @traceable 标记函数，自动记录步骤
"""

from __future__ import annotations

import functools
import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# TraceStep — 单步记录
# =============================================================================


@dataclass
class TraceStep:
    """单步骤追踪记录。

    Attributes:
        stage:      阶段名（如 "parser" / "gate.confidence" / "executor"）。
        status:     状态（"success" / "failed" / "replanned" / "cached"）。
        latency_ms: 耗时（毫秒）。
        detail:     简短描述（如 "Gate③ passed: score=0.92"）。
        replan:     结构化 Re-plan 三段式 CoT（仅 replanned 时有值）。
            dict 格式：{"error_analysis": str, "revised_plan": str, "action_execution": str}。
        score:      闸门分数（可选）。
    """
    stage: str
    status: str
    latency_ms: float
    detail: str = ""
    replan: dict | None = None
    score: float | None = None


# =============================================================================
# TraceContext — 全链路追踪上下文
# =============================================================================


@dataclass
class TraceContext:
    """单个请求的全链路追踪上下文。

    通过 ContextVar 隐式传递，不污染函数签名。
    """
    trace_id: str
    steps: list[TraceStep] = field(default_factory=list)
    start_time: float = field(default_factory=time.monotonic)
    active: bool = True

    def add_step(
        self,
        stage: str,
        status: str,
        latency_ms: float,
        detail: str = "",
        replan: dict | None = None,
        score: float | None = None,
    ) -> None:
        """记录一步追踪信息。"""
        if not self.active:
            logger.debug("Trace %s 已结束，忽略 step: %s", self.trace_id, stage)
            return
        # 序列化 replan dict 为字符串存储
        replan_str = None
        if replan is not None:
            replan_str = {
                "error_analysis": replan.get("error_analysis", "")[:200],
                "revised_plan": replan.get("revised_plan", "")[:200],
                "action_execution": replan.get("action_execution", "")[:200],
            }
        self.steps.append(TraceStep(
            stage=stage, status=status, latency_ms=latency_ms,
            detail=detail[:200],
            replan=replan_str,
            score=score,
        ))

    def to_dict(self) -> dict[str, Any]:
        """序列化为 API 响应可用的字典。"""
        total_ms = round((time.monotonic() - self.start_time) * 1000, 2)
        return {
            "trace_id": self.trace_id,
            "total_latency_ms": total_ms,
            "steps": [
                {
                    "stage": s.stage,
                    "status": s.status,
                    "latency_ms": round(s.latency_ms, 1),
                    "detail": s.detail,
                    "replan": s.replan,
                    "score": s.score,
                }
                for s in self.steps
            ],
        }


# =============================================================================
# ContextVar — 隐式上下文传递
# =============================================================================

_trace_var: ContextVar[TraceContext | None] = ContextVar("trace", default=None)


def start_trace() -> str:
    """开启新追踪，返回 trace_id。

    应在每个请求入口处调用（如 routes.py 的 chat endpoint）。
    """
    trace_id = uuid.uuid4().hex[:12]
    ctx = TraceContext(trace_id=trace_id)
    _trace_var.set(ctx)
    logger.debug("Trace started: %s", trace_id)
    return trace_id


def get_trace() -> TraceContext | None:
    """获取当前协程的活跃 TraceContext。"""
    return _trace_var.get()


def end_trace() -> dict[str, Any] | None:
    """关闭当前追踪，返回 debug_traces 字典。

    Returns:
        序列化后的 trace dict（含 trace_id, total_latency_ms, steps），
        若无活跃 trace 则返回 None。
    """
    ctx = _trace_var.get()
    if ctx is None:
        return None
    ctx.active = False
    result = ctx.to_dict()
    _trace_var.set(None)
    logger.debug("Trace ended: %s (%d steps, %.0f ms)",
                 ctx.trace_id, len(ctx.steps), result["total_latency_ms"])
    return result


def add_step(
    stage: str,
    status: str,
    latency_ms: float,
    detail: str = "",
    replan: dict | None = None,
    score: float | None = None,
) -> None:
    """向当前活跃 trace 添加一步记录。

    若无活跃 trace（ContextVar 为 None），静默忽略。
    """
    ctx = _trace_var.get()
    if ctx is not None:
        ctx.add_step(stage, status, latency_ms, detail, replan, score)


# =============================================================================
# @traceable 装饰器
# =============================================================================


def traceable(stage_name: str):
    """装饰器：自动记录被装饰函数的执行耗时和状态。

    仅在 TRACE_ENABLED=True 且有活跃 TraceContext 时记录。

    用法：
        @traceable("parser")
        async def parse(self, query: str) -> dict:
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            from config import settings

            # 快速路径：Trace 未启用或无活跃上下文
            if not settings.TRACE_ENABLED or _trace_var.get() is None:
                return await func(*args, **kwargs)

            start = time.monotonic()
            try:
                result = await func(*args, **kwargs)
                elapsed = (time.monotonic() - start) * 1000
                _log_step(stage_name, "success", elapsed, result)
                return result
            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                ctx = _trace_var.get()
                if ctx:
                    ctx.add_step(stage_name, "failed", elapsed,
                                 detail=str(exc)[:200])
                raise
        return wrapper
    return decorator


def _log_step(stage: str, status: str, latency_ms: float, result: Any) -> None:
    """内部工具：根据返回值类型生成 detail。"""
    detail = ""
    if isinstance(result, dict):
        status_val = result.get("status", "") if result else ""
        if status_val:
            detail = f"status={status_val}"
        score = result.get("score")
        if score is not None:
            detail += f" score={score}"
    elif hasattr(result, "passed") and hasattr(result, "score"):
        detail = f"passed={result.passed} score={result.score}"

    ctx = _trace_var.get()
    if ctx:
        ctx.add_step(stage, status, latency_ms, detail=detail[:200])
