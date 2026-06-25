"""Structured Re-plan Engine — 三段式 CoT Prompt + 解析器。

职责边界：
  - ReplanAction: CoT 输出的结构化数据模型（error_analysis / revised_plan / action）
  - ReplanEngine: 构建 CoT Prompt、调用 LLM、解析响应

调用方：feedback_gate.py 的 retry 循环。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# ReplanAction — CoT 结构化输出
# =============================================================================


@dataclass
class ReplanAction:
    """三段式 CoT 的结构化输出。

    Attributes:
        error_analysis: 错误根因分析。
        revised_plan:   调整后的新计划。
        action_type:    行动类型。
        action_params:  行动参数字典。
    """

    error_analysis: str
    revised_plan: str
    action_type: str  # fix_sql | clean_data | fix_dag | fix_algorithm | re_interpret
    action_params: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# CoT Prompt 模板
# =============================================================================

# ── Gate② 计划校验用 ──
COT_PROMPT_PLAN = """你是一个数据分析系统的故障诊断专家。反馈闸门 {gate_type} 第 {step} 次校验未通过。

## 校验结果
  评分: {score}
  原因: {reason}
  建议: {suggestions}

## 对话上下文
{agent_memory_context}

## 故障分析任务
请按以下三段式结构分析并给出修正方案：

[Error Analysis]
分析 DAG 计划失败的根因（算法不存在/依赖节点缺失/权限不足？）。

[Revised Plan]
针对上述根因，写出具体的调整方案（替换算法/修改节点/调整权限？）。

[Action]
只输出以下 JSON 格式（不要其他文字）：

{{"action_type": "fix_dag|fix_algorithm", "action_params": {{"node_id": "analysis_1", "new_algorithm": "pearson"}}}}

可用的 action_type:
  - fix_dag: 修改 DAG 结构
  - fix_algorithm: 替换 AnalysisNode 的 algorithm_name
"""

# ── Gate③ 约束校验用 ──
COT_PROMPT_CONSTRAINT = """你是一个数据分析系统的故障诊断专家。反馈闸门 {gate_type} 第 {step} 次校验未通过。

## 校验结果
  评分: {score}
  原因: {reason}
  建议: {suggestions}

## 对话上下文
{agent_memory_context}

## 故障分析任务
请按以下三段式结构分析并给出修正方案：

[Error Analysis]
分析约束校验失败的根因（SQL 语法错误/字段不存在/数据质量问题/算法执行异常？）。

[Revised Plan]
针对上述根因，写出具体的调整方案（修正 SQL/清洗数据/替换算法？）。

[Action]
只输出以下 JSON 格式（不要其他文字）：

{{"action_type": "fix_sql|clean_data", "action_params": {{"patched_sql": "SELECT ...", "reason": "..."}}}}

可用的 action_type:
  - fix_sql: 修正 SQL 查询语句（action_params 需含 patched_sql 字段）
  - clean_data: 清洗数据后重试（action_params 需含 strategy: drop/fill）
"""

# ── Gate④ 输出校验用 ──
COT_PROMPT_OUTPUT = """你是一个数据分析系统的故障诊断专家。反馈闸门 {gate_type} 第 {step} 次校验未通过。

## 校验结果
  评分: {score}
  原因: {reason}
  建议: {suggestions}

## 对话上下文
{agent_memory_context}

## 故障分析任务
请按以下三段式结构分析并给出修正方案：

[Error Analysis]
分析输出校验失败的根因（解读为空/结构不完整/数据质量警告？）。

[Revised Plan]
针对上述根因，写出具体的调整方案（补充数据点/调整解读结构/标记警告？）。

[Action]
只输出以下 JSON 格式（不要其他文字）：

{{"action_type": "re_interpret", "action_params": {{"focus": "关键发现", "style": "concise"}}}}
"""

COT_PROMPT_TEMPLATES: dict[str, str] = {
    "plan": COT_PROMPT_PLAN,
    "constraint": COT_PROMPT_CONSTRAINT,
    "output": COT_PROMPT_OUTPUT,
}


# =============================================================================
# ReplanEngine
# =============================================================================


class ReplanEngine:
    """Structured Re-plan 引擎。

    使用方式：
        action = await ReplanEngine.replan(
            gate_result, gate_type="constraint",
            step=0, agent_memory=mem, llm_judge=my_llm,
        )
    """

    @staticmethod
    def build_cot_prompt(
        gate_result: Any,
        gate_type: str,
        step: int,
        agent_memory: Any = None,
    ) -> str:
        """构建三段式 CoT Prompt。

        Args:
            gate_result: GateResult 实例。
            gate_type: 闸门类型 ("plan" / "constraint" / "output")。
            step: 当前重试步数（0-based）。
            agent_memory: AgentMemory 实例（可选）。

        Returns:
            CoT Prompt 字符串。
        """
        template = COT_PROMPT_TEMPLATES.get(gate_type, COT_PROMPT_CONSTRAINT)
        context = ""
        if agent_memory is not None:
            context = agent_memory.to_llm_context()
        suggestions_str = "\n".join(
            f"  - {s}" for s in (gate_result.suggestions or [])
        ) or "  无"

        return template.format(
            gate_type=gate_type,
            step=step + 1,
            score=gate_result.score,
            reason=gate_result.reason,
            suggestions=suggestions_str,
            agent_memory_context=context,
        )

    @staticmethod
    async def call_llm(
        llm_judge: Callable[[str], Awaitable[str]],
        prompt: str,
    ) -> str:
        """调用 LLM 获取 CoT 响应。"""
        return await llm_judge(prompt)

    @staticmethod
    async def replan(
        gate_result: Any,
        gate_type: str,
        step: int,
        agent_memory: Any,
        llm_judge: Callable[[str], Awaitable[str]],
    ) -> ReplanAction | None:
        """一站式 CoT 重规划：构建 prompt → 调用 LLM → 解析。

        Returns:
            ReplanAction，解析失败时返回 None。
        """
        prompt = ReplanEngine.build_cot_prompt(
            gate_result, gate_type, step, agent_memory,
        )
        try:
            text = await llm_judge(prompt)
            return ReplanEngine.parse_response(text)
        except Exception as exc:
            logger.warning("CoT replan 调用或解析异常: %s", exc)
            return None

    @staticmethod
    def parse_response(text: str) -> ReplanAction:
        """解析 LLM 输出的三段式 CoT 文本。

        提取 [Error Analysis]、[Revised Plan]、[Action] 三部分，
        其中 Action 部分必须为 JSON 格式。

        Args:
            text: LLM 原始输出。

        Returns:
            ReplanAction。

        Raises:
            ValueError: 无法解析 Action JSON。
        """
        text = text.strip()

        # 提取 [Error Analysis]
        error_analysis = _extract_section(text, "Error Analysis")
        # 提取 [Revised Plan]
        revised_plan = _extract_section(text, "Revised Plan")
        # 提取 Action JSON
        action_json = _extract_section(text, "Action")

        if not error_analysis:
            error_analysis = "（LLM 未输出 Error Analysis 部分）"
        if not revised_plan:
            revised_plan = "（LLM 未输出 Revised Plan 部分）"

        # 解析 Action JSON
        action_type, action_params = _parse_action_json(action_json or text)

        return ReplanAction(
            error_analysis=error_analysis,
            revised_plan=revised_plan,
            action_type=action_type,
            action_params=action_params,
        )


# =============================================================================
# 内部工具
# =============================================================================


def _extract_section(text: str, section_name: str) -> str:
    """从 CoT 输出中提取指定段落的文本。

    匹配格式：[Error Analysis] ... [Revised Plan] 中的 ... 部分。
    """
    # 匹配 [section_name] 到下一个 [xxx] 或文本结尾
    pattern = re.compile(
        rf"\[{re.escape(section_name)}\](.*?)(?=\[|\Z)", re.DOTALL,
    )
    m = pattern.search(text)
    if m:
        return m.group(1).strip()
    return ""


def _parse_action_json(text: str) -> tuple[str, dict]:
    """从文本中提取 Action JSON，解析出 action_type 和 action_params。

    Args:
        text: 可能包含 JSON 的文本。

    Returns:
        (action_type, action_params) 元组。解析失败时 action_type="unknown"。
    """
    # 尝试直接解析整个文本为 JSON
    cleaned = text.strip()
    # 查找第一个 { 并提取平衡括号的内容
    start = cleaned.find("{")
    if start >= 0:
        try:
            obj = _parse_balanced_json(cleaned, start)
            if obj is not None:
                action_type = obj.get("action_type", "unknown")
                action_params = obj.get("action_params", {})
                if not isinstance(action_params, dict):
                    action_params = {}
                return str(action_type), action_params
        except Exception:
            pass

    # 正则回退：查找扁平 action_type + action_params
    flat_pattern = re.compile(
        r'"action_type"\s*:\s*"(\w+)".*?"action_params"\s*:\s*(\{[^{}]*\})',
        re.DOTALL,
    )
    m = flat_pattern.search(text)
    if m:
        action_type = m.group(1)
        try:
            action_params = json.loads(m.group(2))
        except json.JSONDecodeError:
            action_params = {}
        return action_type, action_params

    logger.warning("无法从 CoT 响应中解析 Action JSON: %.100s", text)
    return "unknown", {}


def _parse_balanced_json(text: str, start: int) -> dict | None:
    """从 start 位置开始，提取并解析平衡括号的 JSON 对象。

    处理嵌套 {} 的 JSON（如 {"a": {"b": "c"}}）。
    """
    depth = 0
    for end in range(start, len(text)):
        ch = text[end]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start: end + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None
