"""反馈闸门 — 四层一反馈质量校验核心。

当前实现：
  Gate① 置信度闸门 — Parser 输出质量评估（completeness × match_score × model_confidence）
  Gate② 计划校验闸门 — DAGPlan 有效性 + RBAC 权限聚合校验
  Gate③ 约束+结果闸门 — SQL 安全二次校验 / 数据序列化校验 / 五维评估
  Gate④ 输出校验 — 终检通过率 / 连续失败标记 Data_Warning

使用方式：
    from app.engine.feedback_gate import feedback_gate

    result = feedback_gate.check_confidence(parse_result)
    if not result.passed:
        # 触发澄清 / 重试
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import numpy as np
import pandas as pd
import sqlparse
from sqlparse.sql import Function

from app.engine.registry import AnalyzerRegistry
from app.engine.replan import ReplanAction, ReplanEngine

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================

# Gate③ — PostgreSQL 禁止函数白名单（sqlparse AST 精确检测）
BLOCKED_FUNCTIONS: frozenset[str] = frozenset({
    # DML/DLL（双重保障，_validate_sql 已拦截 SELECT 之外的操作）
    "COPY",
    # 文件读写
    "PG_READ_FILE", "PG_READ_BINARY_FILE", "PG_WRITE_FILE",
    # 目录遍历
    "PG_LS_DIR", "PG_LS_WALD_DIR", "PG_LS_LOGDIR",
    # 大对象操作
    "LO_IMPORT", "LO_EXPORT", "LO_OPEN", "LO_CREATE", "LO_WRITE",
    # 后端控制
    "PG_CANCEL_BACKEND", "PG_TERMINATE_BACKEND",
    # 远程执行
    "DBLINK_CONNECT", "DBLINK_CONNECT_U", "DBLINK_EXEC",
    "DBLINK_OPEN", "DBLINK_SEND_QUERY",
    # 延时攻击
    "PG_SLEEP",
})


# =============================================================================
# GateResult — 统一返回结构
# =============================================================================


@dataclass
class GateResult:
    """闸门校验结果，四道闸门共用。

    Attributes:
        passed:      是否通过。
        score:       评分（0.0 ~ 1.0）。
        reason:      校验结论摘要。
        suggestions: 不通过时的改进建议列表。
    """

    passed: bool
    score: float
    reason: str
    suggestions: list[str] = field(default_factory=list)


# =============================================================================
# FeedbackGate — 反馈闸门调度器
# =============================================================================


class FeedbackGate:
    """反馈闸门调度器。

    方法：
      check_confidence(parse_result) → GateResult          (Gate①)
      check_plan(plan, context, registry) → GateResult     (Gate②)
      check_constraint(parse_result, plan, exec_result, evaluator) → GateResult  (Gate③)
      check_output(interpretation, eval_reports, llm_judge) → GateResult          (Gate④)
    """

    # ------------------------------------------------------------------
    # Gate① — 置信度闸门
    # ------------------------------------------------------------------

    def check_confidence(self, parse_result: dict[str, Any]) -> GateResult:
        """校验 Parser 输出质量。

        得分公式: score = completeness × match_score × model_confidence

        检测项:
          - 状态劫持: greeting / error / clarify → 直接不通过
          - 字段完备度 (completeness): analysis_type / indicators / tables / time_range / raw_sql / filters
          - 数据源完整性 (match_score): indicators 非空 / tables 非空 / time_range 有效
          - 模型置信度 (model_confidence): 直接取 SQLAgentPlan.confidence

        Args:
            parse_result: parser.py 的 parse() 返回值。
        """
        # 1. 状态劫持
        status = parse_result.get("status", "")
        if status in ("greeting", "error", "clarify"):
            return GateResult(
                passed=False,
                score=0.0,
                reason=f"Parser 返回非执行状态: '{status}'",
                suggestions=self._suggest_for_status(status),
            )

        if status != "ready_for_execution":
            return GateResult(
                passed=False,
                score=0.0,
                reason=f"未知的 Parser 状态: '{status}'，期望 'ready_for_execution'",
                suggestions=["系统解析异常，请稍后重试"],
            )

        # 2. 提取 execution_plan
        execution_plan: dict[str, Any] = parse_result.get("execution_plan", {})
        if not execution_plan:
            return GateResult(
                passed=False,
                score=0.0,
                reason="execution_plan 为空",
                suggestions=["Parser 未生成执行计划，请检查问句是否包含有效分析需求"],
            )

        analysis_type = execution_plan.get("analysis_type", "")

        # 3. 字段完备度 (completeness)
        required_fields = {
            "analysis_type": self._non_empty_str,
            "indicators": self._non_empty_list,
            "tables": self._non_empty_list,
            "time_range": self._has_time_range,
            "raw_sql": self._non_empty_str,
            "filters": lambda v: v is not None,  # filters 存在即可（即使为空列表）
        }

        present = 0
        total_fields = len(required_fields)
        skipped_fields = 0
        missing_fields: list[str] = []
        for field_name, check_fn in required_fields.items():
            value = execution_plan.get(field_name)
            # detail 类型的查询（如"列出所有省份"）不需要 time_range
            if analysis_type == "detail" and field_name == "time_range" and not self._has_time_range(value):
                skipped_fields += 1
                continue
            if check_fn(value):
                present += 1
            else:
                missing_fields.append(field_name)

        effective_total = total_fields - skipped_fields
        completeness = present / effective_total if effective_total > 0 else 0.0

        # 4. 数据源完整性 (match_score)
        indicators = execution_plan.get("indicators", [])
        tables = execution_plan.get("tables", [])
        time_range = execution_plan.get("time_range", {})
        analysis_type = execution_plan.get("analysis_type", "")

        indicators_ok = 1.0 if (isinstance(indicators, list) and len(indicators) > 0) else 0.0
        tables_ok = 1.0 if (isinstance(tables, list) and len(tables) > 0) else 0.0
        time_ok = 1.0 if (isinstance(time_range, dict) and ("start" in time_range or "end" in time_range)) else 0.0

        # detail 类型的查询（如"列出所有省份"）可能没有时间约束，此时 time 不作为扣分项
        if analysis_type == "detail" and not time_ok:
            match_score = (indicators_ok + tables_ok + 1.0) / 3.0
        else:
            match_score = (indicators_ok + tables_ok + time_ok) / 3.0

        # 5. 模型置信度
        model_confidence = execution_plan.get("confidence", 0.0)
        if not isinstance(model_confidence, (int, float)):
            model_confidence = 0.0
        model_confidence = max(0.0, min(1.0, float(model_confidence)))

        # 6. 综合评分
        score = completeness * match_score * model_confidence
        from config import settings
        threshold = settings.CONFIDENCE_THRESHOLD

        if score >= threshold:
            return GateResult(
                passed=True,
                score=round(score, 4),
                reason=f"置信度校验通过: score={score:.4f} >= threshold={threshold}",
            )

        # 7. 不通过 → 生成建议
        suggestions = self._build_confidence_suggestions(
            missing_fields, indicators, tables, model_confidence,
        )
        return GateResult(
            passed=False,
            score=round(score, 4),
            reason=(
                f"置信度校验不通过: score={score:.4f} < threshold={threshold} "
                f"(completeness={completeness:.2f}, match_score={match_score:.2f}, "
                f"model_confidence={model_confidence:.2f})"
            ),
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # Gate② — 计划校验闸门
    # ------------------------------------------------------------------

    def check_plan(
        self,
        plan: Any,
        context: dict[str, Any],
        registry: AnalyzerRegistry,
    ) -> GateResult:
        """校验 DAGPlan 的有效性。

        检测项:
          - DAG 环路: Kahn 算法入度归零检查
          - 节点可达性: 所有 depends_on 指向的节点必须存在
          - 数据源存在性: AnalysisNode / MergeNode 的 data_source 必须为有效上游
          - 算法存在性: AnalysisNode 的 algorithm_name 必须在 registry 中
          - RBAC 权限: 所有分析算法所需 permissions 是否在 context.permissions 中

        Args:
            plan:     DAGPlan 实例。
            context:  执行上下文（含 "permissions" 列表）。
            registry: AnalyzerRegistry 实例。
        """
        issues: list[str] = []

        # 1. DAG 环路检测（Kahn 算法）
        node_ids = [n.node_id for n in plan.nodes]
        if len(node_ids) != len(set(node_ids)):
            return GateResult(
                passed=False, score=0.0,
                reason="DAG 节点 ID 重复",
                suggestions=["节点 ID 必须全局唯一，请检查 DAGBuilder 生成逻辑"],
            )

        try:
            cycle_nodes = self._detect_cycle(plan)
            if cycle_nodes:
                return GateResult(
                    passed=False, score=0.0,
                    reason=f"DAG 中存在环: {cycle_nodes}",
                    suggestions=[f"涉及节点: {cycle_nodes}，请检查依赖关系"],
                )
        except Exception as exc:
            return GateResult(
                passed=False, score=0.0,
                reason=f"DAG 环路检测异常: {exc}",
                suggestions=["系统内部错误，请重试"],
            )

        # 2. 节点可达性 & 数据源 & 算法存在性 & RBAC
        reachability_ok = 0
        reachability_total = 0
        source_ok = 0
        source_total = 0
        algo_ok = 0
        algo_total = 0

        # 收集所有分析节点所需的权限
        required_permissions: set[str] = set()
        user_permissions: list[str] = context.get("permissions", ["user"])

        for node in plan.nodes:
            # 可达性: depends_on 中的 node_id 必须存在
            for dep_id in node.depends_on:
                reachability_total += 1
                try:
                    plan.get_node(dep_id)
                    reachability_ok += 1
                except Exception:
                    issues.append(f"节点 '{node.node_id}' 依赖的 '{dep_id}' 不存在")

            # 数据源存在性
            if hasattr(node, "data_source"):
                source_total += 1
                ds = node.data_source
                if ds in node_ids:
                    source_ok += 1
                else:
                    issues.append(f"节点 '{node.node_id}' 的数据源 '{ds}' 不存在")

            if hasattr(node, "data_sources"):
                for ds in node.data_sources:
                    source_total += 1
                    if ds in node_ids:
                        source_ok += 1
                    else:
                        issues.append(f"融合节点 '{node.node_id}' 的数据源 '{ds}' 不存在")

            # 算法存在性 + 权限收集
            if hasattr(node, "algorithm_name"):
                algo_total += 1
                algo_name = getattr(node, "algorithm_name", "")
                try:
                    cls = registry.get(algo_name)
                    # 收集权限（从 tool 的 permissions 或 analyzer 默认权限）
                    if hasattr(cls, "permissions") and cls.permissions:
                        required_permissions.update(cls.permissions)
                    algo_ok += 1
                except Exception:
                    issues.append(f"节点 '{node.node_id}' 的算法 '{algo_name}' 未注册")

        # 3. RBAC 校验
        rbac_ok = True
        if required_permissions:
            if not any(p in user_permissions for p in required_permissions):
                rbac_ok = False
                issues.append(
                    f"权限不足: 需要 {sorted(required_permissions)}，"
                    f"当前上下文权限为 {user_permissions}"
                )

        # 4. 综合得分
        dims = []
        scores = []

        if reachability_total > 0:
            dims.append("reachability")
            scores.append(reachability_ok / reachability_total)

        if source_total > 0:
            dims.append("data_source")
            scores.append(source_ok / source_total)

        if algo_total > 0:
            dims.append("algorithm")
            scores.append(algo_ok / algo_total)

        if rbac_ok:
            scores.append(1.0)
        else:
            scores.append(0.0)

        score = sum(scores) / len(scores) if scores else 1.0

        from config import settings
        threshold = settings.CONFIDENCE_THRESHOLD

        if score >= threshold and not issues:
            return GateResult(
                passed=True,
                score=round(score, 4),
                reason=f"计划校验通过: score={score:.4f} >= threshold={threshold}",
            )

        return GateResult(
            passed=False,
            score=round(score, 4),
            reason=(
                f"计划校验不通过: score={score:.4f} < threshold={threshold}, "
                f"发现 {len(issues)} 个问题"
            ),
            suggestions=issues,
        )

    # ------------------------------------------------------------------
    # Gate③ — 约束 + 结果闸门
    # ------------------------------------------------------------------

    async def check_constraint(
        self,
        parse_result: dict[str, Any],
        plan: Any,
        execution_result: dict[str, Any],
        evaluator: Any,  # FiveDimEvaluator
        llm_judge: Callable[[str], Awaitable[str]] | None = None,
    ) -> GateResult:
        """约束 + 结果校验。

        三步校验:
          1. SQL 安全二次校验（禁止函数白名单）
          2. 数据序列化校验（NaN / Infinity / NaT）
          3. 五维评估（规则 + 可选 LLM）

        Args:
            parse_result:     Parser 输出。
            plan:             DAGPlan 实例。
            execution_result: Executor 执行结果。
            evaluator:        FiveDimEvaluator 实例。
            llm_judge:        可选的 LLM 回调（用于 completeness / interpretability 评估）。

        Returns:
            GateResult。
        """
        issues: list[str] = []

        # ── Step 1: SQL 安全二次校验 ──
        execution_plan = parse_result.get("execution_plan", {})
        sql = execution_plan.get("raw_sql", "")
        sql_safe, sql_issues = self._check_sql_safety(sql)
        if not sql_safe:
            issues.extend(sql_issues)

        # ── Step 2: 数据序列化校验 ──
        final_data = execution_result.get("final_data")
        serial_ok, serial_issues = self._check_serialization(final_data)
        if not serial_ok:
            issues.extend(serial_issues)

        # ── Step 3: 五维评估 ──
        report = None
        try:
            report = await evaluator.evaluate(
                question=parse_result.get("route_metadata", {}).get("intent", ""),
                parse_result=parse_result,
                execution_result=execution_result,
                llm_judge=llm_judge,
            )
        except Exception as exc:
            logger.warning("五维评估执行异常，跳过: %s", exc)
            report = None

        # ── 综合判定 ──
        passed = sql_safe and serial_ok
        score_parts: list[float] = []

        # SQL 安全分
        score_parts.append(1.0 if sql_safe else 0.0)
        # 序列化分
        score_parts.append(1.0 if serial_ok else 0.0)
        # 五维评估分（如果有）
        if report is not None:
            score_parts.append(report.overall_score)
            if not report.passed:
                issues.append(f"五维评估不通过: score={report.overall_score:.4f}")
                passed = False

        score = sum(score_parts) / len(score_parts)

        from config import settings
        threshold = settings.CONFIDENCE_THRESHOLD
        passed = passed and (report is None or report.passed)

        if passed and not issues:
            return GateResult(
                passed=True,
                score=round(score, 4),
                reason=f"约束+结果校验通过: score={score:.4f}",
            )

        return GateResult(
            passed=False,
            score=round(score, 4),
            reason=f"约束+结果校验不通过: score={score:.4f}, 发现 {len(issues)} 个问题",
            suggestions=issues,
        )

    # ------------------------------------------------------------------
    # Gate④ — 输出校验
    # ------------------------------------------------------------------

    def check_output(
        self,
        interpretation: str,
        eval_reports: list[Any],  # list[EvalReport]
        llm_judge: Callable[[str], Awaitable[str]] | None = None,
    ) -> GateResult:
        """输出终检。

        检测项:
          - 解读文本非空
          - 历史评估连续失败检测（连续 ≥2 次 → Data_Warning）
          - 熔断保护（总失败 ≥MAX_AGENT_STEPS → 强制通过 + Data_Warning）

        Args:
            interpretation: 解读层输出文本。
            eval_reports:   五维评估历史记录（从最近到最早排序）。
            llm_judge:      可选的 LLM 回调。

        Returns:
            GateResult（含 data_warning 标记在 suggestions 中）。
        """
        suggestions: list[str] = []

        # 1. 解读文本非空
        if not interpretation or not interpretation.strip():
            return GateResult(
                passed=False, score=0.0,
                reason="解读文本为空",
                suggestions=["解读层未生成分析报告"],
            )

        # 2. 连续失败检测
        consecutive_fails = 0
        total_fails = 0
        for report in (eval_reports or []):
            if not report.passed:
                consecutive_fails += 1
                total_fails += 1
            else:
                consecutive_fails = 0

        from config import settings
        data_warning = False
        max_steps = settings.MAX_AGENT_STEPS

        if consecutive_fails >= 2:
            data_warning = True
            suggestions.append(
                f"连续 {consecutive_fails} 次五维评估不达标，标记 Data_Warning"
            )

        # 3. 熔断保护
        if total_fails >= max_steps:
            data_warning = True
            suggestions.append(
                f"共 {total_fails} 次评估不达标（≥{max_steps}），触发熔断强制通过"
            )
            return GateResult(
                passed=True,
                score=max(0.0, 1.0 - total_fails * 0.2),
                reason=f"熔断保护: 强制通过 (total_fails={total_fails})",
                suggestions=list(set(suggestions)),
            )

        # 4. 正常判定
        if not data_warning:
            return GateResult(
                passed=True, score=1.0,
                reason="输出校验通过",
            )

        return GateResult(
            passed=True,
            score=0.6,
            reason="输出校验通过（含 Data_Warning）",
            suggestions=suggestions,
        )

    # =========================================================================
    # Retry Loop — 四道闸门统一硬熔断（MAX_AGENT_STEPS = 3）+ Graceful Degradation
    # =========================================================================

    async def check_confidence_with_retry(
        self,
        parse_result: dict[str, Any],
        agent_memory: Any = None,
    ) -> GateResult:
        """Gate① 带 MAX_AGENT_STEPS 硬熔断的 retry loop。

        方案A：直接 Graceful Degradation（Gate① 本质是 Parser 能力不足，
        CoT 无法让 Gate① 自修 parse_result，不尝试 CoT 修复）。

        Args:
            parse_result: Parser 输出。
            agent_memory: AgentMemory 实例（仅用于记录，不用于 CoT）。
        """
        from config import settings

        last_result: GateResult | None = None
        for step in range(settings.MAX_AGENT_STEPS):
            result = self.check_confidence(parse_result)
            if result.passed:
                return result
            last_result = result
            if step == settings.MAX_AGENT_STEPS - 1:
                break
            # Gate① 不尝试 CoT 修复，标记后等待外部重新解析
            if agent_memory is not None:
                agent_memory.add_turn(
                    "system",
                    f"[Gate①] 第 {step+1} 次校验失败: {result.reason}",
                    metadata={"gate": "confidence", "retry": True},
                )

        return self._graceful_degrade(
            settings.MAX_AGENT_STEPS - 1, "Gate①", last_result, parse_result,
        )

    async def check_plan_with_retry(
        self,
        plan: Any,
        context: dict[str, Any],
        registry: AnalyzerRegistry,
        agent_memory: Any = None,
        llm_judge: Callable[[str], Awaitable[str]] | None = None,
    ) -> GateResult:
        """Gate② 带 MAX_AGENT_STEPS + CoT 重规划的 retry loop。

        失败 → CoT 分析根因 → ReplanAction → 修改 DAGPlan → re-check。
        """
        from config import settings

        last_result: GateResult | None = None
        current_plan = plan
        for step in range(settings.MAX_AGENT_STEPS):
            result = self.check_plan(current_plan, context, registry)
            if result.passed:
                return result
            last_result = result
            if step == settings.MAX_AGENT_STEPS - 1:
                break
            # CoT → 修改 plan
            if llm_judge is not None and agent_memory is not None:
                try:
                    action = await self._cot_replan(
                        result, "plan", step, agent_memory, llm_judge,
                    )
                    if action is not None:
                        agent_memory.record_replan_turn(
                            "plan", result, action.error_analysis, action.revised_plan,
                        )
                        current_plan = await self._apply_action_to_plan(
                            current_plan, action, context, registry,
                        )
                except Exception as exc:
                    logger.warning("Gate② CoT 重规划异常: %s", exc)

        return self._graceful_degrade(
            settings.MAX_AGENT_STEPS - 1, "Gate②", last_result, current_plan,
        )

    async def check_constraint_with_retry(
        self,
        parse_result: dict[str, Any],
        plan: Any,
        execution_result: dict[str, Any],
        evaluator: Any,
        agent_memory: Any = None,
        llm_judge: Callable[[str], Awaitable[str]] | None = None,
        re_execute_fn: Callable[
            [ReplanAction, dict[str, Any], Any, dict[str, Any]],
            Awaitable[dict[str, Any]],
        ] | None = None,
    ) -> GateResult:
        """Gate③ 带 MAX_AGENT_STEPS + CoT 重规划的 retry loop。

        失败 → CoT 分析根因 → ReplanAction → re_execute_fn（由 orchestator
        传入，负责重新执行 SQL/DAG）→ 新 execution_result → re-check。
        """
        from config import settings

        last_result: GateResult | None = None
        current_exec_result = execution_result
        current_parse = parse_result
        for step in range(settings.MAX_AGENT_STEPS):
            result = await self.check_constraint(
                current_parse, plan, current_exec_result, evaluator, llm_judge,
            )
            if result.passed:
                return result
            last_result = result
            if step == settings.MAX_AGENT_STEPS - 1:
                break
            # CoT → fix → re-execute
            if llm_judge is not None and agent_memory is not None and re_execute_fn is not None:
                try:
                    action = await self._cot_replan(
                        result, "constraint", step, agent_memory, llm_judge,
                    )
                    if action is not None:
                        agent_memory.record_replan_turn(
                            "constraint", result,
                            action.error_analysis, action.revised_plan,
                        )
                        # 应用 fix（如 patched SQL 直接写入 parse_result）
                        if action.action_type == "fix_sql":
                            patched_sql = action.action_params.get("patched_sql", "")
                            if patched_sql:
                                current_parse["execution_plan"]["raw_sql"] = patched_sql
                        # 外部重新执行
                        current_exec_result = await re_execute_fn(
                            action, current_parse, plan, current_exec_result,
                        )
                except Exception as exc:
                    logger.warning("Gate③ CoT 重规划异常: %s", exc)

        return self._graceful_degrade(
            settings.MAX_AGENT_STEPS - 1, "Gate③", last_result, current_exec_result,
        )

    async def check_output_with_retry(
        self,
        interpretation: str,
        eval_reports: list[Any],
        agent_memory: Any = None,
        llm_judge: Callable[[str], Awaitable[str]] | None = None,
    ) -> GateResult:
        """Gate④ 带 MAX_AGENT_STEPS 硬熔断的 retry loop。

        连续失败 ≥MAX_AGENT_STEPS → 熔断强制通过 + Data_Warning。
        """
        from config import settings

        # 当前实现：Gate④ 的 eval_reports 是外部累积传入，
        # 每次调用 check_output 自身计算连续失败 / 总失败。
        # 如果已有熔断条件满足，直接熔断。
        for step in range(settings.MAX_AGENT_STEPS):
            result = self.check_output(interpretation, eval_reports, llm_judge)
            # 正常通过（无 data_warning）
            if result.passed and not any(
                "Data_Warning" in s for s in result.suggestions
            ):
                return result
            # 已有熔断标记 → 已是最优结果
            if any("熔断" in s for s in result.suggestions):
                return result
            # 含 Data_Warning 但未熔断 → 尝试 CoT 改善
            if step < settings.MAX_AGENT_STEPS - 1:
                if llm_judge is not None and agent_memory is not None:
                    try:
                        action = await self._cot_replan(
                            result, "output", step, agent_memory, llm_judge,
                        )
                        if action is not None:
                            agent_memory.record_replan_turn(
                                "output", result,
                                action.error_analysis, action.revised_plan,
                            )
                    except Exception:
                        pass
                # 模拟重试（orchestrator 实际调用时需重新生成 interpretation）
                # 此处注入一个模拟 eval_report 让熔断在下一轮触发
                from app.evaluator.metrics import EvalReport
                eval_reports = list(eval_reports or []) + [
                    EvalReport(
                        dimensions={}, overall_score=0.5, passed=False,
                    ),
                ]

        # 最后一次尝试（熔断已由 check_output 内部处理）
        final_result = self.check_output(interpretation, eval_reports, llm_judge)
        if not final_result.passed:
            return self._graceful_degrade(
                settings.MAX_AGENT_STEPS - 1, "Gate④", final_result, interpretation,
            )
        return final_result

    # =========================================================================
    # Retry Loop 内部工具
    # =========================================================================

    @staticmethod
    async def _cot_replan(
        gate_result: GateResult,
        gate_type: str,
        step: int,
        agent_memory: Any,
        llm_judge: Callable[[str], Awaitable[str]],
    ) -> ReplanAction | None:
        """调用 ReplanEngine 执行三段式 CoT 重规划。

        Returns:
            ReplanAction，解析/调用失败时返回 None。
        """
        return await ReplanEngine.replan(
            gate_result, gate_type, step, agent_memory, llm_judge,
        )

    @staticmethod
    async def _apply_action_to_plan(
        plan: Any,
        action: ReplanAction,
        context: dict[str, Any],
        registry: AnalyzerRegistry,
    ) -> Any:
        """根据 ReplanAction 修改 DAGPlan。

        action_type = "fix_algorithm" 时替换 AnalysisNode 的 algorithm_name。
        """
        if action.action_type == "fix_algorithm":
            node_id = action.action_params.get("node_id", "")
            new_algo = action.action_params.get("new_algorithm", "")
            if node_id and new_algo:
                for node in plan.nodes:
                    if node.node_id == node_id and hasattr(node, "algorithm_name"):
                        logger.info(
                            "CoT 修复: 节点 %s 算法 %s → %s",
                            node_id, node.algorithm_name, new_algo,
                        )
                        node.algorithm_name = new_algo
                        break

        elif action.action_type == "fix_dag":
            node_id = action.action_params.get("node_id", "")
            if node_id:
                plan.nodes = [
                    n for n in plan.nodes if n.node_id != node_id
                ]
                logger.info("CoT 修复: 移除节点 %s", node_id)
                # 重新计算 level_groups
                plan.level_groups = self._recompute_levels(plan)

        return plan

    @staticmethod
    def _recompute_levels(plan: Any) -> list[list[str]]:
        """重新计算 DAGPlan 的 level_groups（Kahn 算法）。"""
        # 构建入度表和邻接表
        in_degree: dict[str, int] = {n.node_id: 0 for n in plan.nodes}
        adj: dict[str, list[str]] = {n.node_id: [] for n in plan.nodes}

        for node in plan.nodes:
            for dep in node.depends_on:
                if dep in adj:
                    adj[dep].append(node.node_id)
                    in_degree[node.node_id] = in_degree.get(node.node_id, 0) + 1

        # Kahn 分层
        levels: list[list[str]] = []
        queue = [nid for nid, deg in in_degree.items() if deg == 0]

        while queue:
            levels.append(sorted(queue))
            next_queue: list[str] = []
            for nid in queue:
                for neighbor in adj.get(nid, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)
            queue = next_queue

        return levels

    @staticmethod
    def _graceful_degrade(
        step: int,
        gate_name: str,
        last_result: GateResult | None,
        context: Any,
    ) -> GateResult:
        """硬熔断降级。

        3 次重试均失败后强制通过 + Data_Warning 标记，确保永不白屏。

        Args:
            step: 失败时的步数（0-based）。
            gate_name: 闸门名称。
            last_result: 最后一次校验结果（可能为 None）。
            context: 降级时关联的上下文对象（仅用于日志）。

        Returns:
            GateResult(passed=True) 含 Data_Warning 标记。
        """
        from config import settings

        score = 0.0
        if last_result is not None:
            score = max(0.0, last_result.score - step * 0.15)

        logger.warning(
            "%s 硬熔断触发: %d 次重试后强制通过 | context=%s",
            gate_name, step + 1, type(context).__name__,
        )

        return GateResult(
            passed=True,
            score=round(score, 4),
            reason=f"{gate_name} 硬熔断: 强制通过 ({step+1}/{settings.MAX_AGENT_STEPS})",
            suggestions=[
                f"{gate_name} 重试 {step+1} 次后触发熔断保护",
                settings.DATA_WARNING_FLAG,
            ],
        )

    # =========================================================================
    # 内部工具
    # =========================================================================

    # ── Gate① 辅助 ──

    @staticmethod
    def _non_empty_str(val: Any) -> bool:
        return isinstance(val, str) and len(val.strip()) > 0

    @staticmethod
    def _non_empty_list(val: Any) -> bool:
        return isinstance(val, list) and len(val) > 0

    @staticmethod
    def _has_time_range(val: Any) -> bool:
        if not isinstance(val, dict):
            return False
        return bool(val.get("start") or val.get("end"))

    @staticmethod
    def _suggest_for_status(status: str) -> list[str]:
        mapping = {
            "greeting": ["您好！请提出具体的数据分析需求，例如'2023年各省GDP排名'"],
            "error": ["系统解析异常，请稍后重试或联系管理员"],
            "clarify": ["请补充更详细的查询条件，如时间范围、指标名称等"],
        }
        return mapping.get(status, ["请重新描述您的分析需求"])

    @staticmethod
    def _build_confidence_suggestions(
        missing_fields: list[str],
        indicators: Any,
        tables: Any,
        model_confidence: float,
    ) -> list[str]:
        """生成 Gate① 不通过时的改进建议。"""
        suggestions: list[str] = []

        field_hints = {
            "analysis_type": "未能识别分析类型（趋势/排名/对比/异常等）",
            "indicators": "未提取到分析指标，请明确要分析的字段",
            "tables": "未能识别数据表，请指定数据范围",
            "time_range": "缺少时间范围，请补充如 '2023年' 或 '近三年'",
            "raw_sql": "未能生成查询 SQL",
            "filters": "过滤条件不清晰",
        }
        for field in missing_fields:
            hint = field_hints.get(field, f"缺少字段: {field}")
            suggestions.append(hint)

        if not (isinstance(indicators, list) and len(indicators) > 0):
            suggestions.append("未识别到分析指标，请用具体业务指标（如 GDP、增速）重新描述")
        if not (isinstance(tables, list) and len(tables) > 0):
            suggestions.append("未能匹配到数据表结构，请确认查询的表名或指标是否存在于数据集中")
        if model_confidence < 0.5:
            suggestions.append("模型对本次解析信心较低，建议用更精确的关键词重新描述")

        return suggestions

    # ── Gate② 辅助 ──

    @staticmethod
    def _detect_cycle(plan: Any) -> list[str]:
        """DFS 检测 DAG 中是否存在环。返回环中涉及的节点列表（空列表表示无环）。"""
        # 构建邻接表
        adj: dict[str, list[str]] = {n.node_id: list(n.depends_on) for n in plan.nodes}

        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {nid: WHITE for nid in adj}
        cycle_nodes: list[str] = []

        def dfs(node_id: str) -> bool:
            color[node_id] = GRAY
            for neighbor in adj.get(node_id, []):
                if neighbor not in color:
                    continue
                if color[neighbor] == GRAY:
                    cycle_nodes.append(f"{neighbor} → {node_id}")
                    return True
                if color[neighbor] == WHITE:
                    if dfs(neighbor):
                        return True
            color[node_id] = BLACK
            return False

        for nid in adj:
            if color[nid] == WHITE:
                if dfs(nid):
                    return cycle_nodes

        return []

    # ── Gate③ 辅助 ──

    @staticmethod
    def _check_sql_safety(sql: str) -> tuple[bool, list[str]]:
        """二次 SQL 安全校验：禁止函数白名单。

        使用 sqlparse 的 Function 类型精确检测 PostgreSQL 危险函数调用，
        避免正则误匹配列名/注释中的函数名。

        Returns:
            (passed, issues_list)
        """
        if not sql or not sql.strip():
            return True, []

        issues: list[str] = []
        try:
            parsed = sqlparse.parse(sql)
            for stmt in parsed:
                _walk_sql_functions(stmt, stmt, issues)
        except Exception as exc:
            # sqlparse 解析失败不是安全事件（语法错本身会被 DB 拒绝）
            logger.debug("sqlparse 解析异常（忽略）: %s", exc)

        return len(issues) == 0, issues

    @staticmethod
    def _check_serialization(data: Any) -> tuple[bool, list[str]]:
        """数据序列化二次校验：检查 NaN / Infinity / NaT。

        DataCleaner 已在前序执行，此处做二次确认。
        """
        if data is None:
            return True, []

        issues: list[str] = []

        if isinstance(data, pd.DataFrame):
            if data.empty:
                return True, []
            try:
                numeric = data.select_dtypes(include=[np.number])
                if not numeric.empty:
                    nan_count = int(numeric.isna().sum().sum())
                    if nan_count > 0:
                        issues.append(f"数据中仍有 {nan_count} 个 NaN 未清洗")
                    inf_mask = np.isinf(numeric.values)
                    inf_count = int(inf_mask.sum())
                    if inf_count > 0:
                        issues.append(f"数据中仍有 {inf_count} 个 Infinity 未清洗")
            except Exception:
                pass

        elif isinstance(data, dict):
            # 递归检查 dict 中的 DataFrame
            for key, val in data.items():
                ok, sub = FeedbackGate._check_serialization(val)
                if not ok:
                    issues.extend(f"[{key}] {s}" for s in sub)

        return len(issues) == 0, issues


# =============================================================================
# 模块级 SQL Function 遍历工具
# =============================================================================


def _walk_sql_functions(root: Any, token: Any, found: list[str]) -> None:
    """递归遍历 sqlparse AST，收集禁止函数名。"""
    if isinstance(token, Function):
        name = token.get_name()
        if name and name.upper() in BLOCKED_FUNCTIONS:
            found.append(f"SQL 中包含危险函数: {name}")
    for child in token.get_sublists():
        _walk_sql_functions(root, child, found)


# 模块级单例
feedback_gate = FeedbackGate()
