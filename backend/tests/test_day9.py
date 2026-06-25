"""Day 9 验证 — Structured Re-plan + Agent Memory + 硬熔断。

测试项:
  AgentMemory: summarize() / record_replan_turn() / to_llm_context()
  ReplanEngine: build_cot_prompt() / parse_response() / full replan()
  Gate① retry: MAX_AGENT_STEPS 直接 GD
  Gate② retry: CoT → replan → re-check
  Gate③ retry: CoT → re_execute_fn → re-check
  Gate④ retry: 硬熔断 / Data_Warning
  Graceful Degradation: _graceful_degrade() 返回 Data_Warning
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.engine.agent_memory import AgentMemory, ConversationTurn
from app.engine.replan import ReplanAction, ReplanEngine
from app.engine.feedback_gate import FeedbackGate, GateResult
from app.engine.registry import AnalyzerRegistry
from app.evaluator.metrics import FiveDimEvaluator, EvalReport


# ═══════════════════════════════════════════════════════════════
# Mock
# ═══════════════════════════════════════════════════════════════

MOCK_COT_RESPONSE = """[Error Analysis]
SQL中字段字段名 gdp_val 不存在，正确的字段名为 gdp。

[Revised Plan]
将 SQL 中的 gdp_val 替换为 gdp 字段，重新执行查询。

[Action]
{"action_type": "fix_sql", "action_params": {"patched_sql": "SELECT province, gdp FROM macro_economy WHERE year='2023'", "reason": "修正字段名"}}
"""

MOCK_COT_PLAN_RESPONSE = """[Error Analysis]
算法 'unknown_algo' 未在 registry 中注册，导致 Gate② 计划校验失败。

[Revised Plan]
将该节点的算法替换为已注册的 'pearson' 算法。

[Action]
{"action_type": "fix_algorithm", "action_params": {"node_id": "analysis_1", "new_algorithm": "pearson"}}
"""

MOCK_COT_OUTPUT_RESPONSE = """[Error Analysis]
解读文本为空或结构不完整，无法通过输出校验。

[Revised Plan]
补充数据关键发现和结论，结构化为分段报告。

[Action]
{"action_type": "re_interpret", "action_params": {"focus": "关键发现", "style": "structured"}}
"""

MOCK_SUMMARY_RESPONSE = "用户查询2023年GDP数据，分析各省排名。"
MOCK_COT_ERROR_RESPONSE = "这不是一个有效的 CoT 响应。"


async def _mock_llm(prompt: str) -> str:
    """默认 mock LLM：根据 prompt 中闸门类型返回对应的固定响应。"""
    if "反馈闸门 plan" in prompt:
        return MOCK_COT_PLAN_RESPONSE
    if "反馈闸门 output" in prompt:
        return MOCK_COT_OUTPUT_RESPONSE
    return MOCK_COT_RESPONSE


async def _mock_llm_summary(prompt: str) -> str:
    return MOCK_SUMMARY_RESPONSE


async def _mock_llm_error(prompt: str) -> str:
    raise RuntimeError("LLM 调用失败")


async def _mock_llm_summary(prompt: str) -> str:
    return MOCK_SUMMARY_RESPONSE


async def _mock_llm_error(prompt: str) -> str:
    raise RuntimeError("LLM 调用失败")


def _make_dag_plan():
    """创建简单的 DAGPlan（含 1 SQLNode + 1 AnalysisNode）。"""
    from app.engine.orchestrator import DAGPlan, AnalysisNode, SQLNode
    nodes = [
        SQLNode("sql_1", "SQL", "SELECT * FROM macro_economy", depends_on=[]),
        AnalysisNode(
            "analysis_1", "Analysis", "pearson", "sql_1",
            params={}, depends_on=["sql_1"],
        ),
    ]
    plan = DAGPlan(nodes)
    plan.level_groups = [["sql_1"], ["analysis_1"]]
    return plan


def _make_default_context() -> dict:
    return {"permissions": ["user"]}


def _make_parse_result(
    sql: str = "SELECT * FROM macro_economy WHERE year='2023'",
    analysis_type: str = "trend",
) -> dict:
    return {
        "status": "ready_for_execution",
        "route_metadata": {"intent": "2023年GDP趋势"},
        "execution_plan": {
            "analysis_type": analysis_type,
            "indicators": ["gdp", "gdp_growth_rate"],
            "tables": ["macro_economy"],
            "time_range": {"start": "2020", "end": "2023"},
            "raw_sql": sql,
            "filters": [],
            "confidence": 0.9,
        },
    }


def _make_execution_result(
    dag_status: str = "full",
    data_warning: bool = False,
    final_data: pd.DataFrame | None = None,
) -> dict:
    if final_data is None:
        final_data = pd.DataFrame({"province": ["A", "B"], "gdp": [100.0, 200.0]})
    return {
        "success": True,
        "dag_status": dag_status,
        "data_warning": data_warning,
        "final_data": final_data,
        "total_latency_ms": 500,
        "nodes_execution": [],
    }


async def _mock_re_execute(
    action: ReplanAction,
    parse_result: dict,
    plan: any,
    execution_result: dict,
) -> dict:
    """模拟 Gate③ 的 re_execute_fn。"""
    if action.action_type == "fix_sql":
        parse_result["execution_plan"]["raw_sql"] = (
            action.action_params.get("patched_sql", "")
        )
    # 返回干净的 execution_result
    return _make_execution_result()


# ═══════════════════════════════════════════════════════════════
# 1. AgentMemory
# ═══════════════════════════════════════════════════════════════

def test_memory_basic():
    """session_buffer 基础操作。"""
    mem = AgentMemory(max_buffer_turns=10)
    mem.add_turn("user", "2023年GDP排名")
    mem.add_turn("assistant", "已查询到GDP数据")
    assert mem.turn_count == 2
    assert mem.get_recent_turns(1)[0].content == "已查询到GDP数据"
    print("  [PASS] session_buffer 基础操作")


def test_memory_get_recent_turns():
    """获取最近 N 轮。"""
    mem = AgentMemory()
    for i in range(5):
        mem.add_turn("user", f"问句{i}")
    recent = mem.get_recent_turns(3)
    assert len(recent) == 3
    assert recent[-1].content == "问句4"
    print("  [PASS] get_recent_turns")


def test_memory_auto_summarize():
    """超过 max_buffer_turns 时标记 needs_summary。"""
    mem = AgentMemory(max_buffer_turns=3)
    assert not mem.needs_summary
    for i in range(4):
        mem.add_turn("user", f"问句{i}")
    assert mem.needs_summary, "超过 max 应标记需要摘要"
    print("  [PASS] 自动标记 needs_summary")


def test_memory_update_summary():
    """update_summary 后清空 buffer 保留最近 3 轮。"""
    mem = AgentMemory(max_buffer_turns=10)
    for i in range(8):
        mem.add_turn("user", f"问句{i}")
    mem.update_summary("测试摘要")
    assert mem.get_summary() == "测试摘要"
    assert mem.turn_count <= 3, "设置摘要后应裁剪 buffer 到 3 轮"
    print(f"  [PASS] update_summary: 摘要={mem.get_summary()}, buffer={mem.turn_count}")


def test_memory_record_replan_turn():
    """record_replan_turn 写入 buffer。"""
    mem = AgentMemory()
    result = GateResult(passed=False, score=0.6, reason="字段缺失")
    mem.record_replan_turn("constraint", result, "字段gdp_val缺失", "改用gdp字段")
    assert mem.turn_count == 1
    recent = mem.get_recent_turns(1)
    assert "[Re-plan:constraint]" in recent[0].content
    print(f"  [PASS] record_replan_turn: {recent[0].content[:80]}")


async def test_memory_summarize_success():
    """summarize() LLM 调用成功。"""
    mem = AgentMemory()
    mem.add_turn("user", "2023年GDP排名前5")
    mem.add_turn("assistant", "已获取排名数据")
    summary = await mem.summarize(_mock_llm_summary)
    assert summary == MOCK_SUMMARY_RESPONSE
    assert mem.get_summary() == MOCK_SUMMARY_RESPONSE
    print(f"  [PASS] summarize 成功: {summary}")


async def test_memory_summarize_fallback():
    """summarize() LLM 失败时回退到截断摘要。"""
    mem = AgentMemory()
    mem.add_turn("user", "2023年GDP排名")
    summary = await mem.summarize(_mock_llm_error)
    assert len(summary) > 0, "LLM 失败后应有回退摘要"
    assert mem.get_summary() == summary
    print(f"  [PASS] summarize fallback: {summary[:80]}")


def test_memory_to_llm_context():
    """to_llm_context() 含摘要 + 最近 3 轮。"""
    mem = AgentMemory()
    mem.add_turn("user", "2023年GDP")
    mem.add_turn("assistant", "已查询")
    mem.update_summary("用户查询GDP数据")
    ctx = mem.to_llm_context()
    assert "2023年GDP" in ctx
    assert "用户查询GDP数据" in ctx
    print(f"  [PASS] to_llm_context:\n{ctx[:120]}")


def test_memory_to_llm_context_no_summary():
    """无摘要时 to_llm_context 不含摘要部分。"""
    mem = AgentMemory()
    mem.add_turn("user", "问句")
    ctx = mem.to_llm_context()
    assert "[会话摘要]" not in ctx
    print(f"  [PASS] to_llm_context 无摘要: {ctx[:80]}")


# ═══════════════════════════════════════════════════════════════
# 2. ReplanEngine
# ═══════════════════════════════════════════════════════════════

def test_replan_build_cot_prompt():
    """构建 CoT Prompt 含三段式结构。"""
    result = GateResult(
        passed=False, score=0.6,
        reason="字段缺失",
        suggestions=["gdp 字段不存在", "请重试"],
    )
    prompt = ReplanEngine.build_cot_prompt(result, "constraint", 0)
    assert "[Error Analysis]" in prompt
    assert "[Revised Plan]" in prompt
    assert "[Action]" in prompt
    assert "0.6" in prompt
    assert "字段缺失" in prompt
    print(f"  [PASS] CoT prompt 构建成功 ({len(prompt)} chars)")


def test_replan_parse_response():
    """解析有效三段式 CoT 响应。"""
    action = ReplanEngine.parse_response(MOCK_COT_RESPONSE)
    assert action.error_analysis == "SQL中字段字段名 gdp_val 不存在，正确的字段名为 gdp。"
    assert "gdp_val" in action.revised_plan
    assert action.action_type == "fix_sql"
    assert action.action_params["patched_sql"] == "SELECT province, gdp FROM macro_economy WHERE year='2023'"
    print(f"  [PASS] parse_response: type={action.action_type}, error={action.error_analysis[:50]}")


def test_replan_parse_plan_response():
    """解析 Gate② CoT 响应。"""
    action = ReplanEngine.parse_response(MOCK_COT_PLAN_RESPONSE)
    assert action.action_type == "fix_algorithm"
    assert action.action_params["node_id"] == "analysis_1"
    assert action.action_params["new_algorithm"] == "pearson"
    print(f"  [PASS] parse_response (plan): node={action.action_params.get('node_id')}")


def test_replan_parse_invalid():
    """无效 CoT 响应 → action_type='unknown'。"""
    action = ReplanEngine.parse_response(MOCK_COT_ERROR_RESPONSE)
    assert action.action_type == "unknown"
    print(f"  [PASS] parse_response (invalid): type={action.action_type}")


def test_replan_parse_minimal():
    """最小有效 CoT 响应。"""
    text = """[Error Analysis]
字段缺失

[Revised Plan]
补充字段

[Action]
{"action_type": "fix_sql", "action_params": {"patched_sql": "SELECT gdp FROM t"}}
"""
    action = ReplanEngine.parse_response(text)
    assert action.error_analysis == "字段缺失"
    assert action.action_type == "fix_sql"
    print(f"  [PASS] parse_response (minimal): type={action.action_type}")


async def test_replan_full():
    """ReplanEngine.replan() 一站式调用。"""
    result = GateResult(
        passed=False, score=0.6,
        reason="SQL 字段错误",
        suggestions=["gdp_val 不存在"],
    )
    mem = AgentMemory()
    mem.add_turn("user", "2023年GDP排名")
    mem.add_turn("assistant", "SQL执行出错")

    action = await ReplanEngine.replan(result, "constraint", 0, mem, _mock_llm)
    assert action is not None, "CoT replan 应返回有效 action"
    # 调试：对比直接 parse 和经过 replan 的结果
    direct = ReplanEngine.parse_response(MOCK_COT_RESPONSE)
    print(f"  debug: direct.type='{direct.action_type}', "
          f"replan.type='{action.action_type}', type={type(action.action_type).__name__}")
    assert action.action_type == "fix_sql", (
        f"预期 fix_sql, 实为 '{action.action_type}'"
    )
    assert "gdp" in action.action_params.get("patched_sql", "")
    print(f"  [PASS] replan full: type={action.action_type}, plan={action.revised_plan[:50]}")


# ═══════════════════════════════════════════════════════════════
# 3. Graceful Degradation
# ═══════════════════════════════════════════════════════════════

def test_graceful_degrade():
    """硬熔断降级返回 Data_Warning。"""
    last = GateResult(passed=False, score=0.6, reason="test fail")
    result = FeedbackGate._graceful_degrade(2, "Gate③", last, {})
    assert result.passed, "熔断应强制通过"
    assert any("Data_Warning" in s for s in result.suggestions), "应有 Data_Warning"
    assert "熔断" in result.reason
    print(f"  [PASS] graceful_degrade: {result.reason}")


def test_graceful_degrade_none_result():
    """last_result 为 None 时降级不崩溃。"""
    result = FeedbackGate._graceful_degrade(0, "Gate①", None, {})
    assert result.passed
    assert any("Data_Warning" in s for s in result.suggestions)
    print(f"  [PASS] graceful_degrade (None result): {result.reason}")


# ═══════════════════════════════════════════════════════════════
# 4. Gate① Retry — 直接 Graceful Degradation
# ═══════════════════════════════════════════════════════════════

async def test_gate1_pass_first():
    """Gate① 首次通过 → 直接返回。"""
    gate = FeedbackGate()
    result = await gate.check_confidence_with_retry(
        _make_parse_result(),
    )
    assert result.passed, "高置信度应通过"
    print(f"  [PASS] Gate① 首次通过: score={result.score:.4f}")


async def test_gate1_graceful_degrade_after_max():
    """Gate① 3 次失败 → 熔断降级。"""
    gate = FeedbackGate()
    # 构建一个低置信度的 parse_result
    bad_parse = {
        "status": "ready_for_execution",
        "route_metadata": {"intent": ""},
        "execution_plan": {
            "analysis_type": "",
            "indicators": [],
            "tables": [],
            "time_range": {},
            "raw_sql": "",
            "filters": None,
            "confidence": 0.1,
        },
    }
    result = await gate.check_confidence_with_retry(bad_parse)
    assert result.passed, "熔断应强制通过"
    assert any("Data_Warning" in s for s in result.suggestions), "应有 Data_Warning"
    print(f"  [PASS] Gate① 熔断降级: {result.reason}")


async def test_gate1_greeting_status():
    """问候状态 → 直接 GD。"""
    gate = FeedbackGate()
    parse_result = {
        "status": "greeting",
        "route_metadata": {"intent": ""},
        "execution_plan": {},
    }
    result = await gate.check_confidence_with_retry(parse_result)
    assert result.passed, "熔断应强制通过"
    assert any("Data_Warning" in s for s in result.suggestions)
    print(f"  [PASS] Gate① 问候状态熔断: {result.reason}")


# ═══════════════════════════════════════════════════════════════
# 5. Gate② Retry — CoT → Replan → Re-check
# ═══════════════════════════════════════════════════════════════

async def test_gate2_pass_first():
    """Gate② 首次通过。"""
    gate = FeedbackGate()
    plan = _make_dag_plan()
    reg = AnalyzerRegistry()
    reg.auto_discover()
    result = await gate.check_plan_with_retry(
        plan, _make_default_context(), reg,
    )
    assert result.passed, "正常 DAG 应通过"
    print(f"  [PASS] Gate② 首次通过: score={result.score:.4f}")


async def test_gate2_cot_replan():
    """Gate② CoT → 修复 plan → 重试通过。"""
    gate = FeedbackGate()
    reg = AnalyzerRegistry()
    reg.auto_discover()

    # 构建一个有问题的 plan（algorithm_name 不存在）
    from app.engine.orchestrator import DAGPlan, AnalysisNode, SQLNode
    nodes = [
        SQLNode("sql_1", "SQL", "SELECT * FROM t", depends_on=[]),
        AnalysisNode(
            "analysis_1", "Analysis", "unknown_algo", "sql_1",
            params={}, depends_on=["sql_1"],
        ),
    ]
    bad_plan = DAGPlan(nodes)
    bad_plan.level_groups = [["sql_1"], ["analysis_1"]]

    mem = AgentMemory()
    mem.add_turn("user", "2023年GDP趋势")

    result = await gate.check_plan_with_retry(
        bad_plan, _make_default_context(), reg,
        agent_memory=mem, llm_judge=_mock_llm,
    )
    # 应该修复并通过（用 pearson 替换 unknown_algo）
    assert result.passed, "CoT 修复后应通过"
    print(f"  [PASS] Gate② CoT replan: score={result.score:.4f}, reason={result.reason[:60]}")


async def test_gate2_graceful_degrade():
    """Gate② 3 次 CoT 失败 → 熔断降级。"""
    gate = FeedbackGate()
    reg = AnalyzerRegistry()
    reg.auto_discover()

    from app.engine.orchestrator import DAGPlan, AnalysisNode, SQLNode
    nodes = [
        SQLNode("sql_1", "SQL", "SELECT * FROM t", depends_on=[]),
        AnalysisNode(
            "analysis_1", "Analysis", "nonexistent_algo_1", "sql_1",
            params={}, depends_on=["sql_1"],
        ),
        AnalysisNode(
            "analysis_2", "Analysis", "nonexistent_algo_2", "sql_1",
            params={}, depends_on=["sql_1"],
        ),
    ]
    bad_plan = DAGPlan(nodes)
    bad_plan.level_groups = [["sql_1"], ["analysis_1", "analysis_2"]]

    mem = AgentMemory()
    mem.add_turn("user", "测试")

    result = await gate.check_plan_with_retry(
        bad_plan, _make_default_context(), reg,
        agent_memory=mem, llm_judge=_mock_llm,
    )
    assert result.passed, "熔断应强制通过"
    assert any("Data_Warning" in s for s in result.suggestions), "应有 Data_Warning"
    print(f"  [PASS] Gate② 熔断降级: {result.reason}")


# ═══════════════════════════════════════════════════════════════
# 6. Gate③ Retry — CoT → Re-execute → Re-check
# ═══════════════════════════════════════════════════════════════

async def test_gate3_pass_first():
    """Gate③ 首次通过。"""
    gate = FeedbackGate()
    ev = FiveDimEvaluator()
    result = await gate.check_constraint_with_retry(
        _make_parse_result(), _make_dag_plan(),
        _make_execution_result(), ev,
    )
    assert result.passed
    print(f"  [PASS] Gate③ 首次通过: score={result.score:.4f}")


async def test_gate3_cot_replan():
    """Gate③ CoT → fix_sql → re-execute → 通过。"""
    gate = FeedbackGate()
    ev = FiveDimEvaluator()

    # 构建含危险 SQL 的 parse_result
    parse_result = _make_parse_result(sql="SELECT pg_sleep(10)")
    plan = _make_dag_plan()
    exec_result = _make_execution_result()
    mem = AgentMemory()
    mem.add_turn("user", "2023年GDP排名")

    result = await gate.check_constraint_with_retry(
        parse_result, plan, exec_result, ev,
        agent_memory=mem, llm_judge=_mock_llm,
        re_execute_fn=_mock_re_execute,
    )
    assert result.passed, "CoT 修复后应通过"
    print(f"  [PASS] Gate③ CoT replan: score={result.score:.4f}")


async def test_gate3_graceful_degrade():
    """Gate③ 多次失败 → 熔断降级。"""
    gate = FeedbackGate()
    ev = FiveDimEvaluator()

    # 构建含 NaN 的数据，每次都会序列化校验失败
    dirty_df = pd.DataFrame({"x": [1.0, float("nan")]})
    parse_result = _make_parse_result()
    plan = _make_dag_plan()
    exec_result = _make_execution_result(final_data=dirty_df)
    mem = AgentMemory()
    mem.add_turn("user", "测试")

    # 用 invalid CoT（不产生有效 action），不传 re_execute_fn 防止自动修复
    async def _mock_bad_llm(prompt: str) -> str:
        return "无法解析"

    result = await gate.check_constraint_with_retry(
        parse_result, plan, exec_result, ev,
        agent_memory=mem, llm_judge=_mock_bad_llm,
    )
    assert result.passed, "熔断应强制通过"
    assert any("Data_Warning" in s for s in result.suggestions), "应有 Data_Warning"
    print(f"  [PASS] Gate③ 熔断降级: {result.reason}")


# ═══════════════════════════════════════════════════════════════
# 7. Gate④ Retry — 硬熔断 + Data_Warning
# ═══════════════════════════════════════════════════════════════

async def test_gate4_pass_first():
    """Gate④ 首次通过。"""
    gate = FeedbackGate()
    result = await gate.check_output_with_retry(
        "2023年GDP排名分析。广东第一。", [],
    )
    assert result.passed
    print(f"  [PASS] Gate④ 首次通过")


async def test_gate4_empty_interpretation():
    """空解读 → 熔断后带 Data_Warning 通过。"""
    gate = FeedbackGate()
    result = await gate.check_output_with_retry(
        "", [],
        llm_judge=_mock_llm,
    )
    assert result.passed, "熔断应强制通过"
    assert any("Data_Warning" in s for s in result.suggestions)
    print(f"  [PASS] Gate④ 空解读熔断: {result.reason}")


async def test_gate4_meltdown():
    """连续多次失败 → 熔断强制通过。"""
    gate = FeedbackGate()
    reports = [
        EvalReport(dimensions={}, overall_score=0.3, passed=False),
        EvalReport(dimensions={}, overall_score=0.4, passed=False),
        EvalReport(dimensions={}, overall_score=0.5, passed=False),
    ]
    result = await gate.check_output_with_retry(
        "正常解读文本。结构完整。", reports,
        llm_judge=_mock_llm,
    )
    assert result.passed, "熔断应强制通过"
    assert any("Data_Warning" in s for s in result.suggestions)
    print(f"  [PASS] Gate④ 熔断: {result.reason}")


# ═══════════════════════════════════════════════════════════════
# 8. AgentMemory to_llm_context with Replan
# ═══════════════════════════════════════════════════════════════

def test_memory_context_after_replan():
    """record_replan_turn 后 to_llm_context 包含重规划上下文。"""
    mem = AgentMemory()
    mem.add_turn("user", "2023年GDP排名")
    result = GateResult(passed=False, score=0.6, reason="字段缺失")
    mem.record_replan_turn("constraint", result, "字段缺失", "改用正确字段")
    ctx = mem.to_llm_context()
    assert "[Re-plan:constraint]" in ctx
    assert "字段缺失" in ctx
    print(f"  [PASS] context after replan: {ctx[:100]}")


# ═══════════════════════════════════════════════════════════════
# 执行
# ═══════════════════════════════════════════════════════════════

def run():
    import asyncio
    print("=" * 60)
    print("Day 9 — Structured Re-plan + Agent Memory + 硬熔断")
    print("=" * 60)

    print("\n=== 1. AgentMemory ===")
    test_memory_basic()
    test_memory_get_recent_turns()
    test_memory_auto_summarize()
    test_memory_update_summary()
    test_memory_record_replan_turn()
    asyncio.run(test_memory_summarize_success())
    asyncio.run(test_memory_summarize_fallback())
    test_memory_to_llm_context()
    test_memory_to_llm_context_no_summary()
    test_memory_context_after_replan()

    print("\n=== 2. ReplanEngine ===")
    test_replan_build_cot_prompt()
    test_replan_parse_response()
    test_replan_parse_plan_response()
    test_replan_parse_invalid()
    test_replan_parse_minimal()
    asyncio.run(test_replan_full())

    print("\n=== 3. Graceful Degradation ===")
    test_graceful_degrade()
    test_graceful_degrade_none_result()

    print("\n=== 4. Gate① Retry ===")
    asyncio.run(test_gate1_pass_first())
    asyncio.run(test_gate1_graceful_degrade_after_max())
    asyncio.run(test_gate1_greeting_status())

    print("\n=== 5. Gate② Retry ===")
    asyncio.run(test_gate2_pass_first())
    asyncio.run(test_gate2_cot_replan())
    asyncio.run(test_gate2_graceful_degrade())

    print("\n=== 6. Gate③ Retry ===")
    asyncio.run(test_gate3_pass_first())
    asyncio.run(test_gate3_cot_replan())
    asyncio.run(test_gate3_graceful_degrade())

    print("\n=== 7. Gate④ Retry ===")
    asyncio.run(test_gate4_pass_first())
    asyncio.run(test_gate4_empty_interpretation())
    asyncio.run(test_gate4_meltdown())

    print("\n" + "=" * 60)
    print("全部通过")
    print("=" * 60)


if __name__ == "__main__":
    run()
