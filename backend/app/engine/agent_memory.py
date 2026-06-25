"""
Agent Memory — 三层结构：session_buffer + summary + long_term
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class ConversationTurn:
    """单轮对话记录。"""
    role: str          # user | assistant | system
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UserPreference:
    """跨会话用户偏好（可选，Day 13 补充）。"""
    preferred_chart_type: str | None = None
    region_focus: list[str] = field(default_factory=list)
    time_focus: list[str] = field(default_factory=list)


class AgentMemory:
    """三层 Agent 记忆系统。

    - session_buffer: 最近 N 轮原始对话历史
    - summary: buffer 超限时 LLM 压缩的会话摘要
    - long_term: 跨会话用户偏好（可选）
    """

    def __init__(self, max_buffer_turns: int = 10) -> None:
        self._max_buffer_turns = max_buffer_turns
        self._session_buffer: list[ConversationTurn] = []
        self._summary: str = ""
        self._long_term: UserPreference = UserPreference()

    # ── session_buffer ──

    def add_turn(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """添加一轮对话到 buffer。"""
        self._session_buffer.append(ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {},
        ))
        # 自动裁剪：超出 max_buffer_turns 时触发压缩
        if len(self._session_buffer) > self._max_buffer_turns:
            self._auto_summarize()

    def get_recent_turns(self, n: int = 3) -> list[ConversationTurn]:
        """获取最近 N 轮对话。"""
        return self._session_buffer[-n:]

    def get_all_turns(self) -> list[ConversationTurn]:
        return list(self._session_buffer)

    # ── summary ──

    def update_summary(self, summary: str) -> None:
        """外部设置 LLM 压缩后的摘要。"""
        self._summary = summary
        # 设置摘要后清理 buffer（保留最近 3 轮用于上下文指代）
        self._session_buffer = self._session_buffer[-3:]

    def get_summary(self) -> str:
        return self._summary

    def _auto_summarize(self) -> None:
        """自动触发摘要（标记需要压缩，由 summarize() 方法实际执行）。"""
        self._summary_dirty = True  # type: ignore[has-type]

    @property
    def needs_summary(self) -> bool:
        return getattr(self, "_summary_dirty", False)

    # ── LLM 压缩摘要 ──

    async def summarize(
        self,
        llm_judge: Callable[[str], Awaitable[str]],
    ) -> str:
        """调用 LLM 压缩 session_buffer 为摘要，然后更新 summary。

        Args:
            llm_judge: 异步 LLM 回调，接收 prompt 返回文本。

        Returns:
            生成的摘要文本。

        Raises:
            Exception: LLM 调用失败后，回退到截断 buffer 头部。
        """
        if not self._session_buffer:
            return self._summary

        turns_text = "\n".join(
            f"{t.role}: {t.content[:500]}" for t in self._session_buffer
        )
        prompt = SUMMARY_PROMPT.format(turns=turns_text)
        try:
            summary = await llm_judge(prompt)
            summary = summary.strip()
        except Exception as exc:
            logger.warning("AgentMemory summarize LLM 调用失败，使用截断摘要: %s", exc)
            # 回退：取 buffer 头部 N 条做简陋摘要
            head = self._session_buffer[:3]
            summary = " | ".join(f"{t.role}: {t.content[:100]}" for t in head)

        self.update_summary(summary)
        return summary

    def record_replan_turn(
        self,
        gate_type: str,
        gate_result: Any,
        error_analysis: str,
        revised_plan: str,
    ) -> None:
        """记录一轮重规划的上下文到 session_buffer。

        Args:
            gate_type: 闸门类型 ("confidence" / "plan" / "constraint" / "output")。
            gate_result: GateResult 实例。
            error_analysis: CoT 的错误根因分析。
            revised_plan: CoT 的调整后计划。
        """
        content = (
            f"[Re-plan:{gate_type}] score={gate_result.score:.4f}, "
            f"reason={gate_result.reason}\n"
            f"Error Analysis: {error_analysis}\n"
            f"Revised Plan: {revised_plan}"
        )
        self.add_turn("system", content, metadata={"replan": True, "gate_type": gate_type})

    # ── long_term ──

    def update_preference(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self._long_term, key):
                setattr(self._long_term, key, value)

    def get_preference(self) -> UserPreference:
        return self._long_term

    # ── 注入 Parser 上下文 ──

    def to_llm_context(self) -> str:
        """序列化为 LLM 上下文（summary + 最近 3 轮）。"""
        parts: list[str] = []
        if self._summary:
            parts.append(f"[会话摘要]\n{self._summary}")
        recent = self.get_recent_turns(3)
        if recent:
            turns = "\n".join(
                f"{t.role}: {t.content[:200]}" for t in recent
            )
            parts.append(f"[最近对话]\n{turns}")
        return "\n\n".join(parts)

    def clear(self) -> None:
        self._session_buffer.clear()
        self._summary = ""

    @property
    def turn_count(self) -> int:
        return len(self._session_buffer)


# =============================================================================
# LLM Prompt 模板
# =============================================================================

SUMMARY_PROMPT = """你是一个对话摘要专家。请将以下多轮对话内容压缩为一段简洁的摘要（50字以内），
保留核心的业务目标、已确定的指标、时间范围和用户偏好。

对话内容：
{turns}

摘要（50字以内）："""


logger = logging.getLogger(__name__)
