"""五维评估矩阵 — LLM-as-Judge + 规则混合评估。

用法:
    from app.evaluator.metrics import FiveDimEvaluator

    evaluator = FiveDimEvaluator()
    report = await evaluator.evaluate(
        question="2023年GDP排名",
        parse_result={...},
        execution_result={...},
        llm_judge=my_llm_call,       # 可选，不传则规则降级
    )
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================

# 分析类型 → 推荐图表类型 → 数据列要求
CHART_REQUIREMENTS: dict[str, dict] = {
    "trend":       {"chart": "line",   "need_time": True,  "need_numeric": True,  "min_cols": 2},
    "rank":        {"chart": "bar",    "need_time": False, "need_numeric": True,  "min_cols": 2},
    "correlation": {"chart": "scatter","need_time": False, "need_numeric": True,  "min_cols": 2},
    "anomaly":     {"chart": "scatter_line", "need_time": False, "need_numeric": True, "min_cols": 2},
    "composite":   {"chart": "radar",  "need_time": False, "need_numeric": True,  "min_cols": 3},
    "advanced":    {"chart": "scatter","need_time": False, "need_numeric": True,  "min_cols": 2},
    "detail":      {"chart": "table",  "need_time": False, "need_numeric": False, "min_cols": 0},
}


# =============================================================================
# 数据结构
# =============================================================================


@dataclass
class EvalDimensionScore:
    """单维度评估结果。"""
    name: str          # 维度名
    score: float       # 0.0 ~ 1.0
    method: str        # "rule" | "llm"
    detail: str        # 评语


@dataclass
class EvalReport:
    """五维评估报告。"""
    dimensions: dict[str, EvalDimensionScore]
    overall_score: float
    passed: bool
    data_warning: bool = False
    summary: str = ""


# =============================================================================
# 评估器
# =============================================================================


class FiveDimEvaluator:
    """五维评估矩阵。

    五个维度及权重（可从 config 传入）:
      correctness(30%)    — 结果正确性
      completeness(20%)   — 分析完整性
      consistency(20%)    — 逻辑一致性
      interpretability(15%) — 可解释性
      display_fitness(15%)  — 展示适配度

    传入 llm_judge 时用 LLM 评估 completeness / interpretability，
    不传入则规则降级。
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {
            "correctness": 0.30,
            "completeness": 0.20,
            "consistency": 0.20,
            "interpretability": 0.15,
            "display_fitness": 0.15,
        }
        # 缓存权重和，后续不需要重复计算
        self._weight_sum = sum(self.weights.values())

    async def evaluate(
        self,
        question: str,
        parse_result: dict[str, Any],
        execution_result: dict[str, Any],
        interpretation: str | None = None,
        llm_judge: Callable[[str], Awaitable[str]] | None = None,
    ) -> EvalReport:
        """执行五维评估。

        Args:
            question: 原始问句。
            parse_result: Parser 输出。
            execution_result: Executor 输出（含 final_data, dag_status, data_warning 等）。
            interpretation: 解读层输出文本（Gate③ 调用时为 None）。
            llm_judge: 异步 LLM 回调，接收 prompt 返回文本。

        Returns:
            EvalReport。
        """
        dims: dict[str, EvalDimensionScore] = {}

        # 1. 正确性 — 规则
        dims["correctness"] = self._eval_correctness(execution_result)

        # 3. 一致性 — 规则（无外部依赖，先执行）
        dims["consistency"] = self._eval_consistency(execution_result)

        # 5. 展示适配度 — 规则（无外部依赖）
        dims["display_fitness"] = self._eval_display_fitness(
            parse_result, execution_result,
        )

        # 2+4. 完整性 + 可解释性 — LLM 并行执行
        if llm_judge:
            # completeness + interpretability 并行执行
            comp_task = self._eval_completeness_llm(
                question, parse_result, execution_result, llm_judge,
            )
            interp_task: Awaitable | None = None
            if interpretation:
                interp_task = self._eval_interpretability_llm(interpretation, llm_judge)

            if interp_task:
                comp_result, interp_result = await asyncio.gather(comp_task, interp_task)
                dims["completeness"] = comp_result
                dims["interpretability"] = interp_result
            else:
                dims["completeness"] = await comp_task
                dims["interpretability"] = self._eval_interpretability_rule(
                    interpretation or "",
                )
        else:
            dims["completeness"] = self._eval_completeness_rule(
                question, parse_result, execution_result,
            )
            dims["interpretability"] = self._eval_interpretability_rule(
                interpretation or "",
            )

        # 加权总分
        overall = sum(
            dims[k].score * self.weights.get(k, 0)
            for k in dims
        ) / self._weight_sum

        from config import settings
        threshold = settings.EVAL_PASS_THRESHOLD
        data_warning = execution_result.get("data_warning", False)

        return EvalReport(
            dimensions=dims,
            overall_score=round(overall, 4),
            passed=overall >= threshold,
            data_warning=data_warning,
        )

    # ── 1. 正确性（规则） ──────────────────────────────────────────

    def _eval_correctness(self, result: dict[str, Any]) -> EvalDimensionScore:
        """评估结果正确性。"""
        penalties = 0.0
        details: list[str] = []

        final_data = self._resolve_final_data(result)
        if final_data is None:
            return EvalDimensionScore(
                "correctness", 0.0, "rule", "执行结果为空",
            )

        # DAG 状态
        dag_status = result.get("dag_status", "full")
        if dag_status == "failed":
            return EvalDimensionScore(
                "correctness", 0.0, "rule", "DAG 执行失败",
            )
        if dag_status == "partial":
            penalties += 0.2
            details.append("部分节点执行失败")

        # data_warning
        if result.get("data_warning"):
            penalties += 0.2
            details.append("数据警告标记")

        # DataFrame 数值检查（仍未清洗的异常值）
        if isinstance(final_data, pd.DataFrame):
            try:
                numeric_cols = final_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    numeric_data = final_data[numeric_cols]
                    nan_count = int(numeric_data.isna().sum().sum())
                    inf_count = int(
                        np.isinf(numeric_data.values).sum()
                        if hasattr(numeric_data, "values")
                        else 0
                    )
                    if nan_count > 0:
                        penalties += 0.15
                        details.append(f"发现 {nan_count} 个 NaN")
                    if inf_count > 0:
                        penalties += 0.15
                        details.append(f"发现 {inf_count} 个 Infinity")
                # 数据行数过少扣分
                if len(final_data) < 3:
                    penalties += 0.05
                    details.append(f"数据行数过少({len(final_data)}行)")
            except Exception:
                pass

        score = max(0.0, 1.0 - penalties)
        detail = "; ".join(details) if details else "数据完整，无异常"
        return EvalDimensionScore("correctness", score, "rule", detail)

    # ── 2. 完整性（LLM + 规则降级） ──────────────────────────────

    async def _eval_completeness_llm(
        self,
        question: str,
        parse_result: dict[str, Any],
        execution_result: dict[str, Any],
        llm_judge: Callable[[str], Awaitable[str]],
    ) -> EvalDimensionScore:
        """LLM 评估完整性。"""
        prompt = _PROMPT_COMPLETENESS.format(
            question=question,
            analysis_type=parse_result.get("execution_plan", {}).get("analysis_type", "unknown"),
            indicators=parse_result.get("execution_plan", {}).get("indicators", []),
            tables=parse_result.get("execution_plan", {}).get("tables", []),
            time_range=parse_result.get("execution_plan", {}).get("time_range", {}),
            data_shape=_describe_data_shape(execution_result.get("final_data")),
        )
        try:
            text = await llm_judge(prompt)
            score, detail = _parse_llm_score(text)
            return EvalDimensionScore("completeness", score, "llm", detail)
        except Exception as exc:
            logger.warning("LLM 完整性评估失败，降级到规则: %s", exc)
            return self._eval_completeness_rule(question, parse_result, execution_result)

    def _eval_completeness_rule(
        self,
        question: str,
        parse_result: dict[str, Any],
        execution_result: dict[str, Any],
    ) -> EvalDimensionScore:
        """规则降级：检查 execution_plan 字段的解析完整度。"""
        plan = parse_result.get("execution_plan", {})
        present = 0
        total = 0

        for field in ("indicators", "tables", "time_range", "analysis_type"):
            total += 1
            val = plan.get(field)
            if isinstance(val, list) and len(val) > 0:
                present += 1
            elif isinstance(val, dict) and ("start" in val or "end" in val):
                present += 1
            elif isinstance(val, str) and val.strip():
                present += 1

        # 结果数据不为空 +0.2
        bonus = 0.0
        if execution_result.get("final_data") is not None:
            bonus = 0.2

        # 列名匹配度：检查 indicators 是否在 final_data 的列中
        col_match_bonus = 0.0
        final_data = execution_result.get("final_data")
        indicators = plan.get("indicators", [])
        if isinstance(final_data, pd.DataFrame) and indicators:
            data_cols_lower = {c.lower().strip('" ') for c in final_data.columns}
            matched = sum(1 for ind in indicators if ind.lower().strip('" ') in data_cols_lower)
            if indicators:
                col_match_bonus = 0.1 * (matched / len(indicators))

        score = min(1.0, (present / max(total, 1)) + bonus + col_match_bonus)
        return EvalDimensionScore(
            "completeness", score, "rule",
            f"解析字段完整度: {present}/{total}",
        )

    # ── 3. 一致性（规则） ──────────────────────────────────────────

    def _eval_consistency(self, result: dict[str, Any]) -> EvalDimensionScore:
        """评估逻辑一致性。"""
        issues: list[str] = []

        # data_warning 标记
        if result.get("data_warning"):
            issues.append("数据警告")

        # dag_status
        dag_status = result.get("dag_status", "full")
        if dag_status == "partial":
            issues.append("部分节点跳过")
        elif dag_status == "failed":
            return EvalDimensionScore("consistency", 0.0, "rule", "DAG 执行失败")

        # 百分比列求和 ≈ 100%
        final_data = result.get("final_data")
        if isinstance(final_data, pd.DataFrame) and not final_data.empty:
            for col in final_data.columns:
                col_lower = col.lower()
                if any(kw in col_lower for kw in ("占比", "比例", "percentage", "ratio", "pct")):
                    try:
                        total = final_data[col].sum()
                        if not (99.0 <= total <= 101.0):
                            issues.append(f"'{col}' 之和={total:.2f}%，超出 [99,101]")
                    except Exception:
                        pass

        score = max(0.0, 1.0 - len(issues) * 0.25)
        detail = "; ".join(issues) if issues else "逻辑一致"
        return EvalDimensionScore("consistency", score, "rule", detail)

    # ── 4. 可解释性（LLM + 规则降级） ─────────────────────────────

    async def _eval_interpretability_llm(
        self,
        interpretation: str,
        llm_judge: Callable[[str], Awaitable[str]],
    ) -> EvalDimensionScore:
        """LLM 评估可解释性。"""
        prompt = _PROMPT_INTERPRETABILITY.format(text=interpretation[:2000])
        try:
            text = await llm_judge(prompt)
            score, detail = _parse_llm_score(text)
            return EvalDimensionScore("interpretability", score, "llm", detail)
        except Exception as exc:
            logger.warning("LLM 可解释性评估失败，降级到规则: %s", exc)
            return self._eval_interpretability_rule(interpretation)

    def _eval_interpretability_rule(self, text: str) -> EvalDimensionScore:
        """规则降级：文本长度 + 基础结构检查 + 关键词结构检查。"""
        if not text or not text.strip():
            return EvalDimensionScore("interpretability", 0.0, "rule", "解读文本为空")

        details: list[str] = []
        penalties = 0.0
        length = len(text.strip())

        if length < 50:
            penalties += 0.5
            details.append("解读过短")
        elif length < 100:
            penalties += 0.2
            details.append("解读偏短")

        # 基本结构：句号/换行表示有分段
        sentences = text.count("。") + text.count("\n")
        if sentences < 2:
            penalties += 0.2
            details.append("缺少段落结构")

        # 关键词结构检查：检查是否含关键结构词（数据描述/结论/发现）
        structure_keywords = ["数据描述", "结论", "发现", "分析", "趋势", "排名", "占比", "相关", "异常"]
        found_keywords = sum(1 for kw in structure_keywords if kw in text)
        if found_keywords < 2:
            penalties += 0.15
            details.append(f"缺少关键结构词")

        score = max(0.0, 1.0 - penalties)
        detail = "; ".join(details) if details else "结构完整"
        return EvalDimensionScore("interpretability", score, "rule", detail)

    @staticmethod
    def _resolve_final_data(execution_result: dict[str, Any]) -> pd.DataFrame | None:
        """从执行结果中解析出代表性 DataFrame。

        处理 final_data 可能为 dict[str, DataFrame]（merge collect 策略）的情况。
        从 dict 中选取行数最大的 DataFrame 作为代表。
        """
        data = execution_result.get("final_data")
        if data is None:
            return None
        if isinstance(data, pd.DataFrame):
            return data if not data.empty else None
        if isinstance(data, dict):
            best = None
            best_rows = 0
            for df in data.values():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    if len(df) > best_rows:
                        best = df
                        best_rows = len(df)
            return best
        return None

    # ── 5. 展示适配度（规则） ──────────────────────────────────────

    def _eval_display_fitness(
        self,
        parse_result: dict[str, Any],
        execution_result: dict[str, Any],
    ) -> EvalDimensionScore:
        """评估数据与推荐图表的适配度。"""
        plan = parse_result.get("execution_plan", {})
        analysis_type = plan.get("analysis_type", "detail")
        req = CHART_REQUIREMENTS.get(analysis_type)

        if req is None:
            return EvalDimensionScore("display_fitness", 0.5, "rule", f"未知分析类型 '{analysis_type}'")

        final_data = self._resolve_final_data(execution_result)
        if final_data is None:
            return EvalDimensionScore(
                "display_fitness", 0.0, "rule",
                f"数据为空，无法适配 {req['chart']} 图表",
            )

        issues: list[str] = []
        numeric_cols = final_data.select_dtypes(include=[np.number]).columns.tolist()

        # 列数检查
        if len(final_data.columns) < req["min_cols"]:
            issues.append(f"数据列数({len(final_data.columns)})不足，{req['chart']} 至少需 {req['min_cols']} 列")

        # 行数检查：结果过少影响展示效果
        # 注意：某些分析类型天然产出少行结果（如 CAGR 只返回 1 行增长率）
        row_count = len(final_data)
        if row_count < 2:
            # 仅对 detail（明细查询）和 rank（排名）严格要求行数
            if analysis_type in ("detail", "rank"):
                issues.append(f"结果行数过少({row_count}行)，展示效果差")
        elif row_count < 4 and analysis_type not in (
            "detail", "correlation", "anomaly", "rank",
            "trend", "composite", "cross_domain", "multi_dim",
        ):
            issues.append(f"结果行数偏少({row_count}行)")

        # 数值列检查
        if req["need_numeric"] and len(numeric_cols) < 1:
            issues.append(f"缺少数值列，无法绘制 {req['chart']}")

        # 时间列检查
        if req["need_time"]:
            time_cols = final_data.select_dtypes(
                include=["datetime64", "object"],
            ).columns.tolist()
            # 宽松检查：有任意非数值列即可视为时间/类别列
            non_numeric = [c for c in final_data.columns if c not in numeric_cols]
            if len(non_numeric) < 1:
                issues.append(f"缺少时间/类别列，无法绘制 {req['chart']}")

        if issues:
            score = max(0.0, 1.0 - len(issues) * 0.3)
            return EvalDimensionScore(
                "display_fitness", score, "rule",
                f"推荐 {req['chart']}; {'; '.join(issues)}",
            )

        return EvalDimensionScore(
            "display_fitness", 1.0, "rule",
            f"数据适配 {req['chart']} 图表 ({len(numeric_cols)} 个数值列)",
        )


# =============================================================================
# 内部工具
# =============================================================================


def _describe_data_shape(data: Any) -> str:
    """描述数据形状（用于 LLM prompt）。"""
    if isinstance(data, pd.DataFrame):
        return f"DataFrame(rows={len(data)}, cols={len(data.columns)}, columns={list(data.columns)})"
    if isinstance(data, dict):
        parts = []
        for k, v in data.items():
            if isinstance(v, pd.DataFrame):
                parts.append(f"{k}({len(v)}x{len(v.columns)})")
            else:
                parts.append(k)
        return f"dict({', '.join(parts)})"
    return str(type(data).__name__)


def _parse_llm_score(text: str) -> tuple[float, str]:
    """从 LLM 响应中解析 `{"score": 0.85, "detail": "..."}`。

    先尝试 JSON 解析；失败则用正则回退提取分数。
    """
    import json
    import re

    text = text.strip()

    # 尝试 JSON
    try:
        obj = json.loads(text)
        score = float(obj.get("score", 0.5))
        detail = str(obj.get("detail", ""))
        return max(0.0, min(1.0, score)), detail
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 正则回退
    m = re.search(r'"score"\s*:\s*([0-9.]+)', text)
    if m:
        score = float(m.group(1))
        return max(0.0, min(1.0, score)), text.strip()[:200]

    # 找不到分数，默认 0.5
    return 0.5, text.strip()[:200]


# =============================================================================
# LLM Prompt 模板
# =============================================================================

_PROMPT_COMPLETENESS = """你是一个数据分析评估专家。请评估分析结果是否完整覆盖了用户问题中的所有分析维度。

用户问题: {question}

系统解析:
  分析类型: {analysis_type}
  指标: {indicators}
  数据表: {tables}
  时间范围: {time_range}

结果数据概览: {data_shape}

评分标准:
  - 1.0: 完全覆盖所有分析维度，结果数据完整
  - 0.8: 覆盖了用户问题中的主要分析要求，数据有少量缺失
  - 0.6: 覆盖了部分分析要求，但有一些明显的遗漏
  - 0.4: 只覆盖了小部分分析要求，遗漏较多
  - 0.2: 几乎没有覆盖任何分析要求
  - 0.0: 完全没有覆盖

请以JSON格式回复: {{"score": 0.85, "detail": "覆盖了XX和XX维度，缺少XX"}}
只输出JSON，不要其他文字。"""

_PROMPT_INTERPRETABILITY = """你是一个数据分析报告质量评估专家。请评估以下分析解读的质量。

分析解读:
{text}

评分标准（从可解释性角度）:
  - 1.0: 结构清晰(标题+分段)，有分析说明和数据描述，关键信息突出
  - 0.8: 结构基本完整，有分析说明，表述清楚
  - 0.6: 有一定的结构和说明，但不够详细
  - 0.4: 结构混乱或过于简短，难以理解
  - 0.2: 内容极少，几乎没有有用信息
  - 0.0: 内容为空或完全不可读

请以JSON格式回复: {{"score": 0.85, "detail": "结构清晰，覆盖了关键发现"}}
只输出JSON，不要其他文字。"""
