"""feedback_gate.py Day 8 验证 — Gate③ + Gate④ + 五维评估。

测试项:
  Gate③: SQL安全 / 数据序列化 / 五维评估（规则+LLM）
  Gate④: 正常通过 / 连续失败标记 Data_Warning / 熔断保护
  评估器: 正确性/完整性/一致性/可解释性/展示适配度
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.engine.feedback_gate import FeedbackGate, GateResult, BLOCKED_FUNCTIONS
from app.engine.orchestrator import SQLNode, AnalysisNode, DAGPlan
from app.evaluator.metrics import FiveDimEvaluator, EvalReport, EvalDimensionScore


# ═══════════════════════════════════════
# Helper
# ═══════════════════════════════════════

def _make_parse_result(
    sql: str = "SELECT * FROM macro_economy WHERE year='2023'",
    intent: str = "query_data",
    analysis_type: str = "trend",
    indicators: list[str] | None = None,
) -> dict:
    return {
        "status": "ready_for_execution",
        "route_metadata": {"intent": intent},
        "execution_plan": {
            "analysis_type": analysis_type,
            "indicators": indicators or ["gdp", "gdp_growth_rate"],
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


def _make_simple_plan() -> DAGPlan:
    nodes = [
        SQLNode("sql_1", "SQL", "SELECT * FROM t", depends_on=[]),
        AnalysisNode("analysis_1", "Analysis", "pearson", "sql_1",
                     params={}, depends_on=["sql_1"]),
    ]
    plan = DAGPlan(nodes)
    plan.level_groups = [["sql_1"], ["analysis_1"]]
    return plan


# ═══════════════════════════════════════
# 1. 评估器（门③内部调用）
# ═══════════════════════════════════════

def test_eval_correctness_full():
    """完整数据 → 正确性 1.0。"""
    ev = FiveDimEvaluator()
    score = ev._eval_correctness(_make_execution_result())
    assert score.score == 1.0, f"应为 1.0: {score.score}"
    print(f"  [PASS] 正确性满分: {score.score}")


def test_eval_correctness_empty():
    """空结果 → 正确性 0.0。"""
    ev = FiveDimEvaluator()
    score = ev._eval_correctness({"final_data": pd.DataFrame()})
    assert score.score == 0.0, f"应为 0.0: {score.score}"
    print(f"  [PASS] 空结果正确性: {score.score}")


def test_eval_correctness_partial():
    """部分执行 + data_warning → 正确性降级。"""
    ev = FiveDimEvaluator()
    score = ev._eval_correctness(_make_execution_result(
        dag_status="partial", data_warning=True,
    ))
    assert score.score < 1.0, f"部分执行应降级: {score.score}"
    print(f"  [PASS] 部分执行正确性: {score.score}")


def test_eval_correctness_dirty_data():
    """含 NaN 的数据 → 正确性降级。"""
    ev = FiveDimEvaluator()
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    score = ev._eval_correctness(_make_execution_result(final_data=df))
    assert score.score < 1.0, f"含 NaN 应降级: {score.score}"
    print(f"  [PASS] NaN 数据正确性: {score.score}")


def test_eval_consistency_pct_sum():
    """占比列之和 ≈ 100% → 正常。"""
    ev = FiveDimEvaluator()
    df = pd.DataFrame({"province": ["A", "B"], "占比": [40.0, 60.0]})
    score = ev._eval_consistency(_make_execution_result(final_data=df))
    assert score.score == 1.0, f"应满分: {score.score}"
    print(f"  [PASS] 占比一致性: {score.score}")


def test_eval_consistency_pct_bad():
    """占比列之和 ≠ 100% → 降级。"""
    ev = FiveDimEvaluator()
    df = pd.DataFrame({"province": ["A", "B"], "占比": [30.0, 50.0]})  # sum=80
    score = ev._eval_consistency(_make_execution_result(final_data=df))
    assert score.score < 1.0, f"应降级: {score.score}"
    print(f"  [PASS] 占比不一致降级: {score.score}")


def test_eval_completeness_rule():
    """完整性规则降级：字段完整 → 高分。"""
    ev = FiveDimEvaluator()
    parse_result = _make_parse_result()
    score = ev._eval_completeness_rule("test", parse_result, _make_execution_result())
    assert score.score >= 0.5, f"完整解析应有分: {score.score}"
    print(f"  [PASS] 完整性规则: {score.score}")


def test_eval_interpretability_rule():
    """可解释性规则降级：短文本 → 低分。"""
    ev = FiveDimEvaluator()
    score = ev._eval_interpretability_rule("短文本")
    assert score.score < 0.6, f"短文本应低分: {score.score}"
    print(f"  [PASS] 短文本可解释性: {score.score}")


def test_eval_interpretability_rule_long():
    """长文本 → 高分。"""
    ev = FiveDimEvaluator()
    text = (
        "2023年GDP排名分析显示，广东省以13.57万亿元位居全国第一。\n"
        "江苏省以12.82万亿元紧随其后，排名第二。\n"
        "山东省以9.21万亿元排名第三，浙江和河南分列第四、五位。\n"
        "从增速来看，中西部省份增速普遍高于东部沿海地区。\n"
        "建议关注产业升级和区域协调发展。"
    )
    score = ev._eval_interpretability_rule(text)
    assert score.score >= 0.6, f"完整文本应有分: {score.score}"
    print(f"  [PASS] 长文本可解释性: {score.score}")


def test_eval_display_fitness():
    """trend 类型 → 折线图需时间列。"""
    ev = FiveDimEvaluator()
    parse_result = _make_parse_result(analysis_type="trend")
    df = pd.DataFrame({"year": ["2020", "2021"], "gdp": [100, 200]})
    score = ev._eval_display_fitness(parse_result, _make_execution_result(final_data=df))
    assert score.score > 0, f"有数据应有分: {score.score}"
    print(f"  [PASS] 展示适配度: {score.score} ({score.detail})")


def test_eval_display_fitness_empty():
    """空数据 → 展示适配度 0。"""
    ev = FiveDimEvaluator()
    parse_result = _make_parse_result(analysis_type="trend")
    score = ev._eval_display_fitness(parse_result, _make_execution_result(
        final_data=pd.DataFrame(),
    ))
    assert score.score == 0.0, f"空数据应为 0: {score.score}"
    print(f"  [PASS] 空数据展示适配度: {score.score}")


def test_eval_parse_llm_score():
    """LLM JSON 解析工具函数。"""
    from app.evaluator.metrics import _parse_llm_score
    score, detail = _parse_llm_score('{"score": 0.85, "detail": "good"}')
    assert abs(score - 0.85) < 0.01, f"解析失败: {score}"
    print(f"  [PASS] LLM 分数解析: {score}")


# ═══════════════════════════════════════
# 2. Gate③ — 约束+结果闸门
# ═══════════════════════════════════════

def test_gate3_sql_safe():
    """安全 SQL → 通过。"""
    ok, issues = FeedbackGate._check_sql_safety("SELECT * FROM macro_economy")
    assert ok, f"安全 SQL 应通过: {issues}"
    print("  [PASS] SQL 安全校验通过")


def test_gate3_sql_blocked():
    """含禁止函数的 SQL → 拦截。"""
    # 遍历几个有代表性的禁止函数
    blocked_samples = [
        "SELECT * FROM t WHERE pg_sleep(10)",
        "SELECT lo_import('/etc/passwd')",
        "SELECT * FROM dblink_exec('connstr', 'DROP TABLE x')",
        "SELECT pg_read_file('/etc/passwd')",
    ]
    for sql in blocked_samples:
        ok, issues = FeedbackGate._check_sql_safety(sql)
        assert not ok, f"应拦截: {sql[:50]}"
    print(f"  [PASS] SQL 禁止函数拦截: {len(blocked_samples)} 条")


def test_gate3_sql_all_blocked_functions():
    """每个 BLOCKED_FUNCTIONS 的函数都被检测到。"""
    for func in sorted(BLOCKED_FUNCTIONS):
        sql = f"SELECT {func.lower()}(1)"
        ok, issues = FeedbackGate._check_sql_safety(sql)
        assert not ok, f"应拦截 {func}: {issues}"
    print(f"  [PASS] 全部 {len(BLOCKED_FUNCTIONS)} 个禁止函数检测")


def test_gate3_serialization_clean():
    """干净 DataFrame → 通过。"""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    ok, issues = FeedbackGate._check_serialization(df)
    assert ok, f"干净数据应通过: {issues}"
    print("  [PASS] 序列化校验通过")


def test_gate3_serialization_nan():
    """含 NaN → 不通过。"""
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    ok, issues = FeedbackGate._check_serialization(df)
    assert not ok, "NaN 应被检测"
    print(f"  [PASS] NaN 检测: {issues}")


def test_gate3_serialization_inf():
    """含 Infinity → 不通过。"""
    df = pd.DataFrame({"x": [1.0, np.inf, 3.0]})
    ok, issues = FeedbackGate._check_serialization(df)
    assert not ok, "Infinity 应被检测"
    print(f"  [PASS] Infinity 检测: {issues}")


def test_gate3_serialization_dict():
    """嵌套 dict 中的 DataFrame → 递归检测。"""
    df = pd.DataFrame({"x": [np.nan]})
    ok, issues = FeedbackGate._check_serialization({"a": df})
    assert not ok, "嵌套 NaN 应被检测"
    print(f"  [PASS] 嵌套序列化检测: {issues}")


def test_gate3_constraint_pass():
    """完整管线 → 通过。"""
    gate = FeedbackGate()
    ev = FiveDimEvaluator()
    parse_result = _make_parse_result()
    plan = _make_simple_plan()
    exec_result = _make_execution_result()
    result = gate.check_constraint(parse_result, plan, exec_result, ev)
    assert result.passed, f"应通过: {result.reason}"
    print(f"  [PASS] Gate③ 完整通过: score={result.score:.4f}")


def test_gate3_constraint_blocked_sql():
    """含危险 SQL → 不通过。"""
    gate = FeedbackGate()
    ev = FiveDimEvaluator()
    parse_result = _make_parse_result(sql="SELECT pg_sleep(10)")
    plan = _make_simple_plan()
    exec_result = _make_execution_result()
    result = gate.check_constraint(parse_result, plan, exec_result, ev)
    assert not result.passed, "危险 SQL 应不通过"
    assert any("pg_sleep" in s for s in result.suggestions)
    print(f"  [PASS] Gate③ 危险 SQL 拦截: {result.suggestions}")


def test_gate3_constraint_dirty_data():
    """脏数据 → 序列化校验不通过。"""
    gate = FeedbackGate()
    ev = FiveDimEvaluator()
    parse_result = _make_parse_result()
    plan = _make_simple_plan()
    df = pd.DataFrame({"x": [1.0, np.nan]})
    exec_result = _make_execution_result(final_data=df)
    result = gate.check_constraint(parse_result, plan, exec_result, ev)
    assert not result.passed, "脏数据应不通过"
    assert any("NaN" in s for s in result.suggestions)
    print(f"  [PASS] Gate③ 脏数据拦截: {result.suggestions}")


# ═══════════════════════════════════════
# 3. Gate④ — 输出校验
# ═══════════════════════════════════════

def test_gate4_pass():
    """正常解读 → 通过。"""
    gate = FeedbackGate()
    result = gate.check_output(
        "2023年GDP排名分析。广东第一。江苏第二。",
        eval_reports=[EvalReport(
            dimensions={}, overall_score=0.9, passed=True,
        )],
    )
    assert result.passed, f"应通过: {result.reason}"
    assert result.score == 1.0, f"分数应 1.0: {result.score}"
    print("  [PASS] Gate④ 通过")


def test_gate4_empty():
    """空解读 → 不通过。"""
    gate = FeedbackGate()
    result = gate.check_output("", eval_reports=[])
    assert not result.passed, "空解读应不通过"
    print("  [PASS] Gate④ 空解读拦截")


def test_gate4_consecutive_failure():
    """连续 2 次不通过 → Data_Warning。"""
    gate = FeedbackGate()
    reports = [
        EvalReport(dimensions={}, overall_score=0.6, passed=False),
        EvalReport(dimensions={}, overall_score=0.7, passed=False),
    ]
    result = gate.check_output("正常解读文本。结构完整。", eval_reports=reports)
    assert result.passed, "输出终检应 passed"
    assert any("Data_Warning" in s for s in result.suggestions), "应有 Data_Warning"
    print(f"  [PASS] Gate④ 连续失败 Data_Warning: {result.suggestions}")


def test_gate4_recover_after_failure():
    """失败1次后通过 → 复位计数器。"""
    gate = FeedbackGate()
    reports = [
        EvalReport(dimensions={}, overall_score=0.7, passed=False),
        EvalReport(dimensions={}, overall_score=0.9, passed=True),
    ]
    result = gate.check_output("正常解读文本。结构完整。", eval_reports=reports)
    assert result.passed
    assert not any("Data_Warning" in s for s in result.suggestions), "恢复后不应有警告"
    print("  [PASS] Gate④ 恢复后无警告")


def test_gate4_meltdown():
    """3 次失败 → 熔断保护。"""
    gate = FeedbackGate()
    reports = [
        EvalReport(dimensions={}, overall_score=0.3, passed=False),
        EvalReport(dimensions={}, overall_score=0.4, passed=False),
        EvalReport(dimensions={}, overall_score=0.5, passed=False),
    ]
    result = gate.check_output("正常解读文本。结构完整。", eval_reports=reports)
    assert result.passed, "熔断应强制通过"
    assert any("熔断" in s for s in result.suggestions) or any("Data_Warning" in s for s in result.suggestions)
    print(f"  [PASS] Gate④ 熔断: {result.suggestions}")


# ═══════════════════════════════════════
# 4. EvalReport 整体评估
# ═══════════════════════════════════════

async def test_eval_full_pipeline():
    """Evaluator 完整管线。"""
    ev = FiveDimEvaluator()
    parse_result = _make_parse_result()
    exec_result = _make_execution_result()
    report = await ev.evaluate(
        question="2023年GDP趋势",
        parse_result=parse_result,
        execution_result=exec_result,
    )
    assert isinstance(report, EvalReport)
    assert len(report.dimensions) == 5
    assert report.overall_score > 0
    print(f"  [PASS] 五维评估完整管线: overall={report.overall_score:.4f}, "
          f"passed={report.passed}")


# ═══════════════════════════════════════
# 执行
# ═══════════════════════════════════════

def run():
    import asyncio
    print("=" * 50)
    print("Day 8 — 反馈闸门 ③④ + 五维评估验证")
    print("=" * 50)

    print("\n=== 1. 五维评估矩阵 ===")
    test_eval_correctness_full()
    test_eval_correctness_empty()
    test_eval_correctness_partial()
    test_eval_correctness_dirty_data()
    test_eval_consistency_pct_sum()
    test_eval_consistency_pct_bad()
    test_eval_completeness_rule()
    test_eval_interpretability_rule()
    test_eval_interpretability_rule_long()
    test_eval_display_fitness()
    test_eval_display_fitness_empty()
    test_eval_parse_llm_score()

    print("\n=== 2. Gate③ — 约束+结果闸门 ===")
    test_gate3_sql_safe()
    test_gate3_sql_blocked()
    test_gate3_sql_all_blocked_functions()
    test_gate3_serialization_clean()
    test_gate3_serialization_nan()
    test_gate3_serialization_inf()
    test_gate3_serialization_dict()
    test_gate3_constraint_pass()
    test_gate3_constraint_blocked_sql()
    test_gate3_constraint_dirty_data()

    print("\n=== 3. Gate④ — 输出校验 ===")
    test_gate4_pass()
    test_gate4_empty()
    test_gate4_consecutive_failure()
    test_gate4_recover_after_failure()
    test_gate4_meltdown()

    print("\n=== 4. 完整管线 ===")
    asyncio.run(test_eval_full_pipeline())

    print("\n" + "=" * 50)
    print("全部通过")
    print("=" * 50)


if __name__ == "__main__":
    run()
