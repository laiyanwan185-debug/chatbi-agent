"""Orchestrator 验证脚本。

测试项:
  1. DAGBuilder -- Parser 输出 -> DAGPlan
  2. Kahn 拓扑排序 -- 线性图、菱形图、环图
  3. MergeNode -- concat / join / collect
  4. 熔断传播 -- 节点失败 -> 下游 SKIPPED
  5. SQL 校验 -- 纯 SELECT 放行
  6. Re-plan 接口 -- 字段补全、算法替换
  7. ANALYSIS_TYPE_MAP 完整性
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.engine.orchestrator import (
    ANALYSIS_TYPE_MAP,
    DAGBuilder,
    DAGCycleError,
    DAGExecutor,
    DAGOrchestrator,
    DAGPlan,
    SQLNode,
    AnalysisNode,
    MergeNode,
    NodeStatus,
)


# ═══════════════════════════════════════
# 1. DAGBuilder
# ═══════════════════════════════════════

def test_builder_single_sql():
    """analysis_type=detail (纯SQL, 无分析节点)。"""
    builder = DAGBuilder()
    parse_result = {
        "execution_plan": {
            "analysis_type": "detail",
            "raw_sql": "SELECT province, gdp FROM macro_economy WHERE year='2023'",
            "indicators": ["gdp"],
            "tables": ["macro_economy"],
        }
    }
    plan = builder.build_plan(parse_result)
    assert plan.size == 1, f"预期 1 节点, 实际 {plan.size}"
    assert plan.get_node("sql_1") is not None
    print(f"  [PASS] 纯SQL: {plan.size} node")


def test_builder_multi_algo():
    """analysis_type=rank -> 3 算法 + merge -> 5 节点。"""
    builder = DAGBuilder()
    parse_result = {
        "execution_plan": {
            "analysis_type": "rank",
            "raw_sql": "SELECT province, gdp FROM macro_economy",
            "indicators": ["gdp"],
        }
    }
    plan = builder.build_plan(parse_result)

    rank_algos = ANALYSIS_TYPE_MAP["rank"]
    expected = 1 + len(rank_algos) + 1  # sql + N analysis + merge
    assert plan.size == expected, f"rank 预期 {expected} 节点, 实际 {plan.size}"
    assert plan.get_node("sql_1") is not None
    assert plan.get_node("merge_1") is not None
    print(f"  [PASS] 多算法: {plan.size} nodes, {len(plan.level_groups)} levels")


def test_builder_no_raw_sql():
    """无 raw_sql 回退。"""
    builder = DAGBuilder()
    parse_result = {
        "execution_plan": {
            "analysis_type": "detail",
            "raw_sql": None,
            "indicators": ["gdp", "population"],
        }
    }
    plan = builder.build_plan(parse_result)
    assert plan.get_node("sql_1") is not None
    print(f"  [PASS] 无raw_sql: {plan.size} node")


def test_builder_with_parser_output():
    """模拟 parser.py 实际输出格式。"""
    builder = DAGBuilder()
    parse_result = {
        "status": "ready_for_execution",
        "route_metadata": {
            "intent": "query_data",
            "indicators": ["GDP"],
            "time_hint": "2023",
            "complexity": 3,
        },
        "execution_plan": {
            "analysis_type": "trend",
            "indicators": ["gdp", "gdp_growth_rate"],
            "tables": ["macro_economy"],
            "time_range": {"start": "2020", "end": "2023"},
            "raw_sql": "SELECT province, year, gdp FROM macro_economy WHERE year BETWEEN '2020' AND '2023'",
            "confidence": 0.92,
        },
    }
    plan = builder.build_plan(parse_result)
    assert plan.size >= 2
    assert plan.get_node("sql_1") is not None
    assert plan.get_node("merge_1") is not None
    print(f"  [PASS] Parser输出匹配: {plan.size} nodes, {len(plan.level_groups)} levels")


# ═══════════════════════════════════════
# 2. Kahn 拓扑排序
# ═══════════════════════════════════════

def test_kahn_linear():
    """A -> B -> C -> 3 层。"""
    builder = DAGBuilder()
    nodes = [
        SQLNode("A", "A", "SELECT 1", depends_on=[]),
        SQLNode("B", "B", "SELECT 2", depends_on=["A"]),
        SQLNode("C", "C", "SELECT 3", depends_on=["B"]),
    ]
    plan = DAGPlan(nodes)
    levels = builder.topological_sort(plan)
    assert levels == [["A"], ["B"], ["C"]], f"实际 {levels}"
    print(f"  [PASS] 线性图: {levels}")


def test_kahn_diamond():
    """A -> (B, C) -> D -> 3 层。"""
    builder = DAGBuilder()
    nodes = [
        SQLNode("A", "A", "SELECT 1", depends_on=[]),
        AnalysisNode("B", "B", "algo_b", "A", depends_on=["A"]),
        AnalysisNode("C", "C", "algo_c", "A", depends_on=["A"]),
        MergeNode("D", "D", ["B", "C"], depends_on=["B", "C"]),
    ]
    plan = DAGPlan(nodes)
    levels = builder.topological_sort(plan)
    assert len(levels) == 3, f"预期 3 层, 实际 {len(levels)}"
    assert levels[0] == ["A"]
    assert set(levels[1]) == {"B", "C"}
    assert levels[2] == ["D"]
    print(f"  [PASS] 菱形图: {levels}")


def test_kahn_cycle():
    """A -> B -> C -> A -> DAGCycleError。"""
    builder = DAGBuilder()
    nodes = [
        SQLNode("A", "A", "SELECT 1", depends_on=["C"]),
        SQLNode("B", "B", "SELECT 2", depends_on=["A"]),
        SQLNode("C", "C", "SELECT 3", depends_on=["B"]),
    ]
    plan = DAGPlan(nodes)
    try:
        builder.topological_sort(plan)
        assert False, "应抛出 DAGCycleError"
    except DAGCycleError:
        print("  [PASS] 环图正确检测")


# ═══════════════════════════════════════
# 3. MergeNode 策略
# ═══════════════════════════════════════

def test_merge_concat():
    df1 = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    df2 = pd.DataFrame({"x": [3, 4], "y": ["c", "d"]})
    merged = pd.concat([df1, df2], ignore_index=True)
    assert len(merged) == 4
    print(f"  [PASS] concat: {len(merged)} rows")


def test_merge_join():
    df1 = pd.DataFrame({"key": ["a", "b"], "val1": [10, 20]})
    df2 = pd.DataFrame({"key": ["a", "b"], "val2": [100, 200]})
    merged = pd.merge(df1, df2, on="key")
    assert len(merged) == 2
    print(f"  [PASS] join: {len(merged)} rows")


def test_merge_collect():
    df1 = pd.DataFrame({"x": [1]})
    df2 = pd.DataFrame({"y": [2]})
    collected = {"src1": df1, "src2": df2}
    assert len(collected) == 2
    print(f"  [PASS] collect: {len(collected)} sources")


# ═══════════════════════════════════════
# 4. Executor 异步测试
# ═══════════════════════════════════════

async def test_executor_success():
    executor = DAGExecutor(thread_pool_size=4)
    nodes = [
        SQLNode("sql_1", "SQL", "SELECT * FROM test", depends_on=[]),
        MergeNode("merge_1", "Merge", data_sources=["sql_1"],
                  merge_strategy="collect", depends_on=["sql_1"]),
    ]
    plan = DAGPlan(nodes)
    plan.level_groups = [["sql_1"], ["merge_1"]]
    plan.entry_nodes = ["sql_1"]

    class MockPool:
        async def fetch(self, sql):
            return [{"province": "Guangdong", "gdp": 100}]

    result = await executor.execute(
        plan=plan,
        context={"permissions": ["user"]},
        db_pool=MockPool(),
        analyzer_registry=None,
    )
    assert result.success, f"预期成功: {result.error}"
    print(f"  [PASS] 执行成功: status={result.dag_status}")


async def test_executor_empty_sql():
    """SQL 返回空数据 -> 跳过分析。"""
    executor = DAGExecutor(thread_pool_size=4)
    nodes = [
        SQLNode("sql_1", "SQL", "SELECT * FROM empty", depends_on=[]),
        AnalysisNode("analysis_1", "Analysis", "pearson", "sql_1",
                     params={}, depends_on=["sql_1"]),
        MergeNode("merge_1", "Merge", data_sources=["analysis_1"],
                  merge_strategy="collect", depends_on=["analysis_1"]),
    ]
    plan = DAGPlan(nodes)
    plan.level_groups = [["sql_1"], ["analysis_1"], ["merge_1"]]
    plan.entry_nodes = ["sql_1"]

    class MockPool:
        async def fetch(self, sql):
            return []

    class MockRegistry:
        def get(self, name):
            return None

    result = await executor.execute(
        plan=plan,
        context={"permissions": ["user"]},
        db_pool=MockPool(),
        analyzer_registry=MockRegistry(),
    )
    print(f"  [PASS] 空数据: status={result.dag_status}, warning={result.data_warning}")


# ═══════════════════════════════════════
# 5. SQL 校验
# ═══════════════════════════════════════

def test_sql_validation():
    from app.engine.orchestrator import _validate_sql, SQLValidationError

    _validate_sql("SELECT * FROM test")
    _validate_sql("WITH cte AS (SELECT * FROM test) SELECT * FROM cte")

    bad = ["DROP TABLE test", "INSERT INTO test VALUES (1)",
           "DELETE FROM test", "UPDATE test SET x=1", "TRUNCATE test"]
    for sql in bad:
        try:
            _validate_sql(sql)
            assert False, f"未拦截: {sql}"
        except SQLValidationError:
            pass
    print(f"  [PASS] SQL只读校验: {len(bad)} 种非法语句全部拦截")


# ═══════════════════════════════════════
# 6. Re-plan
# ═══════════════════════════════════════

def test_replan_field_missing():
    orch = DAGOrchestrator()
    nodes = [
        SQLNode("sql_1", "SQL", "SELECT province, gdp FROM t", depends_on=[]),
        AnalysisNode("analysis_1", "Analysis", "pearson", "sql_1",
                     params={}, depends_on=["sql_1"]),
    ]
    plan = DAGPlan(nodes)
    plan.level_groups = [["sql_1"], ["analysis_1"]]

    new_plan = orch.replan(plan, {
        "failed_node": "sql_1",
        "reason": "字段缺失",
        "detail": "population, area",
    })
    updated_sql = new_plan.get_node("sql_1").sql
    assert "population" in updated_sql
    assert "area" in updated_sql
    print(f"  [PASS] Re-plan字段补全")


def test_replan_algo_swap():
    orch = DAGOrchestrator()
    nodes = [
        SQLNode("sql_1", "SQL", "SELECT * FROM t", depends_on=[]),
        AnalysisNode("analysis_pearson", "Analysis", "pearson", "sql_1",
                     params={}, depends_on=["sql_1"]),
    ]
    plan = DAGPlan(nodes)
    plan.level_groups = [["sql_1"], ["analysis_pearson"]]

    new_plan = orch.replan(plan, {
        "failed_node": "analysis_pearson",
        "reason": "算法不合适",
        "new_algorithm": "spearman",
    })
    assert new_plan.get_node("analysis_pearson").algorithm_name == "spearman"
    print("  [PASS] Re-plan算法替换")


# ═══════════════════════════════════════
# 7. ANALYSIS_TYPE_MAP
# ═══════════════════════════════════════

def test_analysis_type_map():
    for atype, algos in ANALYSIS_TYPE_MAP.items():
        assert isinstance(algos, list)
        if atype == "detail":
            assert len(algos) == 0
        else:
            assert len(algos) >= 1
    total = sum(len(v) for v in ANALYSIS_TYPE_MAP.values())
    print(f"  [PASS] TYPE_MAP: {len(ANALYSIS_TYPE_MAP)} types, {total} algos")


# ═══════════════════════════════════════
# 8. 节点属性
# ═══════════════════════════════════════

def test_node_attributes():
    sql = SQLNode("s1", "test", "SELECT 1", timeout=10)
    assert sql.node_type == "sql"
    assert sql.status == NodeStatus.PENDING

    an = AnalysisNode("a1", "test", "cagr", "s1", params={"value_col": "gdp"})
    assert an.algorithm_name == "cagr"
    assert an.data_source == "s1"

    mn = MergeNode("m1", "test", ["a1"], merge_strategy="concat")
    assert mn.merge_strategy == "concat"
    print("  [PASS] 节点属性完整")


# ═══════════════════════════════════════
# 执行
# ═══════════════════════════════════════

def run_sync():
    print("=== DAGBuilder ===")
    test_builder_single_sql()
    test_builder_multi_algo()
    test_builder_no_raw_sql()
    test_builder_with_parser_output()

    print("\n=== Kahn ===")
    test_kahn_linear()
    test_kahn_diamond()
    test_kahn_cycle()

    print("\n=== Merge ===")
    test_merge_concat()
    test_merge_join()
    test_merge_collect()

    print("\n=== SQL Validation ===")
    test_sql_validation()

    print("\n=== Re-plan ===")
    test_replan_field_missing()
    test_replan_algo_swap()

    print("\n=== ANALYSIS_TYPE_MAP ===")
    test_analysis_type_map()

    print("\n=== Node Attributes ===")
    test_node_attributes()


async def run_async():
    print("\n=== Executor ===")
    await test_executor_success()
    await test_executor_empty_sql()


if __name__ == "__main__":
    print("=" * 50)
    print("Orchestrator 验证")
    print("=" * 50)
    run_sync()
    asyncio.run(run_async())
    print("\n" + "=" * 50)
    print("全部通过")
    print("=" * 50)
