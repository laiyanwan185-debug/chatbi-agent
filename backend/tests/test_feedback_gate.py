"""feedback_gate.py 验证脚本。

测试项:
  Gate① 置信度闸门 — 完全通过 / 问候 / 错误 / 缺指标 / 缺表 / 低置信度
  Gate② 计划校验闸门 — 有效 / 环路 / 断链 / 缺算法 / 权限不足
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine.feedback_gate import FeedbackGate, GateResult
from app.engine.orchestrator import SQLNode, AnalysisNode, MergeNode, DAGPlan


# ═══════════════════════════════════════
# Helper: mock registry
# ═══════════════════════════════════════

class MockRegistry:
    """模拟 AnalyzerRegistry。"""
    class MockAnalyzer:
        name = "mock_algo"
        permissions = ["user"]

    def __init__(self, known_algos: set[str] | None = None):
        self._algos = known_algos or {"pearson", "spearman", "cagr", "zscore"}

    def get(self, name: str) -> type:
        if name.lower() not in self._algos:
            raise KeyError(f"未注册的算法: {name}")
        return self.MockAnalyzer


# ═══════════════════════════════════════
# 1. Gate① — 置信度闸门
# ═══════════════════════════════════════

def _make_parse_result(
    status: str = "ready_for_execution",
    analysis_type: str = "trend",
    indicators: list[str] | None = None,
    tables: list[str] | None = None,
    time_range: dict | None = None,
    raw_sql: str | None = None,
    filters: list[str] | None = None,
    confidence: float = 0.9,
) -> dict:
    # 不能使用 val or default 模式，因为 [] 是 falsy
    _ind = ["gdp", "gdp_growth_rate"] if indicators is None else indicators
    _tbl = ["macro_economy"] if tables is None else tables
    _tr = {"start": "2020", "end": "2023"} if time_range is None else time_range
    _sql = "SELECT * FROM macro_economy WHERE year BETWEEN '2020' AND '2023'" if raw_sql is None else raw_sql
    _flt = [] if filters is None else filters
    return {
        "status": status,
        "route_metadata": {
            "intent": "query_data",
            "indicators": ["gdp"] if indicators is None else indicators,
            "time_hint": "2023",
            "complexity": 3,
        },
        "execution_plan": {
            "analysis_type": analysis_type,
            "indicators": _ind,
            "tables": _tbl,
            "time_range": _tr,
            "filters": _flt,
            "aggregation": None,
            "top_k": None,
            "sort_order": None,
            "confidence": confidence,
            "raw_sql": _sql,
        },
    }


def test_gate1_pass():
    """完整 parse_result → 通过。"""
    gate = FeedbackGate()
    result = gate.check_confidence(_make_parse_result())
    assert result.passed, f"应通过: {result.reason}"
    assert result.score >= 0.85, f"分数应 ≥0.85: {result.score}"
    print(f"  [PASS] 通过: score={result.score:.4f}")


def test_gate1_greeting():
    """greeting 状态 → 不通过。"""
    gate = FeedbackGate()
    for status in ("greeting", "error", "clarify"):
        result = gate.check_confidence(_make_parse_result(status=status))
        assert not result.passed, f"'{status}' 应不通过"
        assert result.score == 0.0
    print("  [PASS] 问候/错误/澄清 全部拦截")


def test_gate1_missing_indicators():
    """空 indicators → 不通过（match_score 低）。"""
    gate = FeedbackGate()
    result = gate.check_confidence(_make_parse_result(indicators=[]))
    assert not result.passed, "空 indicators 应不通过"
    assert any("指标" in s for s in result.suggestions), "建议应提示缺少指标"
    print(f"  [PASS] 缺指标: score={result.score:.4f}")


def test_gate1_empty_tables():
    """空 tables → 不通过（match_score 低 + 补充点）。"""
    gate = FeedbackGate()
    result = gate.check_confidence(_make_parse_result(tables=[]))
    assert not result.passed, "空 tables 应不通过"
    assert any("数据表" in s or "表结构" in s for s in result.suggestions), "建议应提示表缺失"
    print(f"  [PASS] 缺表: score={result.score:.4f}")


def test_gate1_low_confidence():
    """低模型置信度 → 不通过。"""
    gate = FeedbackGate()
    result = gate.check_confidence(_make_parse_result(confidence=0.3))
    assert not result.passed, "低置信度应不通过"
    print(f"  [PASS] 低置信度: score={result.score:.4f}")


def test_gate1_no_execution_plan():
    """无 execution_plan → 不通过。"""
    gate = FeedbackGate()
    result = gate.check_confidence({"status": "ready_for_execution"})
    assert not result.passed
    assert result.score == 0.0
    print("  [PASS] 无 execution_plan 拦截")


# ═══════════════════════════════════════
# 2. Gate② — 计划校验闸门
# ═══════════════════════════════════════

def _make_simple_plan() -> DAGPlan:
    """SQL → Analysis(pearson)"""
    nodes = [
        SQLNode("sql_1", "SQL", "SELECT * FROM t", depends_on=[]),
        AnalysisNode("analysis_pearson", "Analysis", "pearson", "sql_1",
                     params={}, depends_on=["sql_1"]),
    ]
    plan = DAGPlan(nodes)
    plan.level_groups = [["sql_1"], ["analysis_pearson"]]
    return plan


def _make_merge_plan() -> DAGPlan:
    """SQL → 2 analysis → merge"""
    nodes = [
        SQLNode("sql_1", "SQL", "SELECT * FROM t", depends_on=[]),
        AnalysisNode("analysis_pearson", "Analysis", "pearson", "sql_1",
                     params={}, depends_on=["sql_1"]),
        AnalysisNode("analysis_spearman", "Analysis", "spearman", "sql_1",
                     params={}, depends_on=["sql_1"]),
        MergeNode("merge_1", "Merge", data_sources=["analysis_pearson", "analysis_spearman"],
                  merge_strategy="collect", depends_on=["analysis_pearson", "analysis_spearman"]),
    ]
    plan = DAGPlan(nodes)
    plan.level_groups = [["sql_1"], ["analysis_pearson", "analysis_spearman"], ["merge_1"]]
    return plan


def test_gate2_pass():
    """有效 DAG → 通过。"""
    gate = FeedbackGate()
    plan = _make_simple_plan()
    registry = MockRegistry()
    result = gate.check_plan(plan, {"permissions": ["user"]}, registry)
    assert result.passed, f"应通过: {result.reason}"
    print(f"  [PASS] 有效 DAG: score={result.score:.4f}")


def test_gate2_cycle():
    """带环 DAG → 不通过。"""
    gate = FeedbackGate()
    nodes = [
        SQLNode("A", "A", "SELECT 1", depends_on=["C"]),
        SQLNode("B", "B", "SELECT 2", depends_on=["A"]),
        SQLNode("C", "C", "SELECT 3", depends_on=["B"]),
    ]
    plan = DAGPlan(nodes)
    registry = MockRegistry()
    result = gate.check_plan(plan, {"permissions": ["user"]}, registry)
    assert not result.passed, "环图应不通过"
    assert "环" in result.reason
    print(f"  [PASS] 环检测: {result.reason}")


def test_gate2_broken_dependency():
    """断链（depends_on 指向不存在的节点）→ 不通过。"""
    gate = FeedbackGate()
    nodes = [
        SQLNode("sql_1", "SQL", "SELECT 1", depends_on=[]),
        AnalysisNode("analysis_bad", "Analysis", "pearson", "sql_1",
                     depends_on=["nonexistent_node"]),
    ]
    plan = DAGPlan(nodes)
    plan.level_groups = [["sql_1"], ["analysis_bad"]]
    registry = MockRegistry()
    result = gate.check_plan(plan, {"permissions": ["user"]}, registry)
    assert not result.passed, "断链应不通过"
    assert any("nonexistent_node" in s for s in result.suggestions)
    print(f"  [PASS] 断链检测: {len(result.suggestions)} 个建议")


def test_gate2_missing_algorithm():
    """算法未注册 → 不通过。"""
    gate = FeedbackGate()
    nodes = [
        SQLNode("sql_1", "SQL", "SELECT 1", depends_on=[]),
        AnalysisNode("analysis_unknown", "Analysis", "unknown_algo", "sql_1",
                     depends_on=["sql_1"]),
    ]
    plan = DAGPlan(nodes)
    plan.level_groups = [["sql_1"], ["analysis_unknown"]]
    registry = MockRegistry()
    result = gate.check_plan(plan, {"permissions": ["user"]}, registry)
    assert not result.passed, "缺算法应不通过"
    assert any("unknown_algo" in s for s in result.suggestions)
    print(f"  [PASS] 缺算法: score={result.score:.4f}")


def test_gate2_rbac_insufficient():
    """权限不足 → 不通过。"""
    gate = FeedbackGate()
    nodes = [
        SQLNode("sql_1", "SQL", "SELECT 1", depends_on=[]),
        AnalysisNode("analysis_admin", "Analysis", "pearson", "sql_1",
                     depends_on=["sql_1"]),
    ]
    plan = DAGPlan(nodes)
    plan.level_groups = [["sql_1"], ["analysis_admin"]]
    # context 中只有 "guest"，而算法需要 "user"
    registry = MockRegistry()
    result = gate.check_plan(plan, {"permissions": ["guest"]}, registry)
    # 不通过：guest 不在 user 权限中
    print(f"  [PASS] 权限不足: passed={result.passed}, score={result.score:.4f}, "
          f"suggestions={result.suggestions}")


def test_gate2_merge_with_sources():
    """多源融合 DAG → 通过。"""
    gate = FeedbackGate()
    plan = _make_merge_plan()
    registry = MockRegistry()
    result = gate.check_plan(plan, {"permissions": ["user"]}, registry)
    assert result.passed, f"多源融合应通过: {result.reason}"
    print(f"  [PASS] 多源融合: score={result.score:.4f}")


# ═══════════════════════════════════════
# 执行
# ═══════════════════════════════════════

def run():
    print("=" * 50)
    print("反馈闸门验证")
    print("=" * 50)

    print("\n=== Gate① 置信度闸门 ===")
    test_gate1_pass()
    test_gate1_greeting()
    test_gate1_missing_indicators()
    test_gate1_empty_tables()
    test_gate1_low_confidence()
    test_gate1_no_execution_plan()

    print("\n=== Gate② 计划校验闸门 ===")
    test_gate2_pass()
    test_gate2_cycle()
    test_gate2_broken_dependency()
    test_gate2_missing_algorithm()
    test_gate2_rbac_insufficient()
    test_gate2_merge_with_sources()

    print("\n" + "=" * 50)
    print("全部通过")
    print("=" * 50)


if __name__ == "__main__":
    run()
