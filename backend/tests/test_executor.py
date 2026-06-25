"""executor.py 验证脚本。

测试项:
  1. DataCleaner — NaN/Infinity/NaT 清洗
  2. SQLExecutor — SQL 校验（非法语句拦截）
  3. MergeExecutor — concat / join / collect / 空数据
  4. AnalysisExecutor — 空数据跳过
  5. NodeExecutor — 完整执行验证
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.engine.executor import (
    AnalysisExecutor,
    DataCleaner,
    MergeExecutor,
    NodeExecutor,
    SQLExecutor,
    SQLValidationError,
    _align_types,
    _validate_sql,
)


# ═══════════════════════════════════════
# 1. DataCleaner
# ═══════════════════════════════════════

def test_cleaner_nan():
    """NaN → 0。"""
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    cleaned = DataCleaner.sanitize(df)
    assert cleaned["x"].iloc[1] == 0.0
    print("  [PASS] NaN → 0")


def test_cleaner_inf():
    """inf → None。"""
    df = pd.DataFrame({"x": [1.0, np.inf, -np.inf]})
    cleaned = DataCleaner.sanitize(df)
    assert cleaned["x"].iloc[1] is None or pd.isna(cleaned["x"].iloc[1])
    assert cleaned["x"].iloc[2] is None or pd.isna(cleaned["x"].iloc[2])
    print("  [PASS] inf → None")


def test_cleaner_datetime():
    """datetime → YYYY-MM-DD 字符串。"""
    df = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", "2024-06-01"])})
    cleaned = DataCleaner.sanitize(df)
    assert cleaned["date"].iloc[0] == "2023-01-15"
    assert cleaned["date"].iloc[1] == "2024-06-01"
    print("  [PASS] datetime → YYYY-MM-DD")


def test_cleaner_dict():
    """递归清洗 dict 中的 DataFrame。"""
    df = pd.DataFrame({"x": [np.nan, 2.0]})
    result = DataCleaner.sanitize_output({"a": df, "b": "hello"})
    assert result["a"]["x"].iloc[0] == 0.0
    assert result["b"] == "hello"
    print("  [PASS] dict 递归清洗")


# ═══════════════════════════════════════
# 2. SQLExecutor
# ═══════════════════════════════════════

def test_sql_validation():
    """只读校验：SELECT/WITH 放行，DML 拦截。"""
    _validate_sql("SELECT * FROM test")
    _validate_sql("WITH cte AS (SELECT * FROM test) SELECT * FROM cte")

    bad = [
        "DROP TABLE test", "INSERT INTO test VALUES (1)",
        "DELETE FROM test", "UPDATE test SET x=1", "TRUNCATE test",
    ]
    for sql in bad:
        try:
            _validate_sql(sql)
            assert False, f"未拦截: {sql}"
        except SQLValidationError:
            pass
    print(f"  [PASS] SQL 校验: {len(bad)} 种非法语句全部拦截")


def test_sql_executor_validation():
    """SQLExecutor 的 execute 方法也应拦截非法语句。"""
    executor = SQLExecutor()

    async def run():
        class MockPool:
            async def fetch(self, sql):
                return []

        try:
            await executor.execute("DROP TABLE test", MockPool())
            assert False, "应抛出 SQLValidationError"
        except SQLValidationError:
            pass

    asyncio.run(run())
    print("  [PASS] SQLExecutor 校验")


# ═══════════════════════════════════════
# 3. MergeExecutor
# ═══════════════════════════════════════

def test_merge_concat():
    executor = MergeExecutor()
    data_map = {
        "a": pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}),
        "b": pd.DataFrame({"x": [3, 4], "y": ["c", "d"]}),
    }
    result = executor.execute(["a", "b"], "concat", None, data_map)
    assert len(result) == 4
    print(f"  [PASS] concat: {len(result)} rows")


def test_merge_join():
    executor = MergeExecutor()
    data_map = {
        "a": pd.DataFrame({"key": ["a", "b"], "val1": [10, 20]}),
        "b": pd.DataFrame({"key": ["a", "b"], "val2": [100, 200]}),
    }
    result = executor.execute(["a", "b"], "join", "key", data_map)
    assert len(result) == 2
    assert "val1" in result.columns and "val2" in result.columns
    print(f"  [PASS] join: {len(result)} rows")


def test_merge_collect():
    executor = MergeExecutor()
    data_map = {
        "a": pd.DataFrame({"x": [1]}),
        "b": pd.DataFrame({"y": [2]}),
    }
    result = executor.execute(["a", "b"], "collect", None, data_map)
    assert isinstance(result, dict)
    assert len(result) == 2
    print(f"  [PASS] collect: {len(result)} sources")


def test_merge_empty():
    """所有上游数据不可用 → ValueError。"""
    executor = MergeExecutor()
    try:
        executor.execute(["a", "b"], "concat", None, {})
        assert False, "应抛出 ValueError"
    except ValueError:
        pass
    print("  [PASS] merge 空数据 ValueError")


# ═══════════════════════════════════════
# 4. AnalysisExecutor
# ═══════════════════════════════════════

def test_analysis_empty_data():
    """空输入 → ValueError。"""
    executor = AnalysisExecutor()
    empty_df = pd.DataFrame()

    async def run():
        try:
            await executor.execute(
                algorithm_name="test_algo",
                context={},
                registry=None,
                tool_cache={},
                data=empty_df,
                params={},
                loop=asyncio.get_running_loop(),
            )
            assert False, "应抛出 ValueError"
        except ValueError:
            pass

    asyncio.run(run())
    print("  [PASS] 分析空数据 ValueError")


# ═══════════════════════════════════════
# 5. NodeExecutor 整合
# ═══════════════════════════════════════

async def test_node_executor_sql():
    """NodeExecutor 执行 SQL 并自动清洗。"""
    ne = NodeExecutor()

    class MockPool:
        async def fetch(self, sql):
            return [{"province": "Guangdong", "gdp": 100}]

    df = await ne.execute_sql("SELECT * FROM test", MockPool())
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    print(f"  [PASS] NodeExecutor SQL: {len(df)} row")


async def test_node_executor_merge():
    """NodeExecutor 融合并自动清洗。"""
    ne = NodeExecutor()
    data_map = {
        "a": pd.DataFrame({"x": [np.nan, 2.0], "y": [3.0, 4.0]}),
        "b": pd.DataFrame({"x": [5.0, np.nan], "y": [7.0, 8.0]}),
    }
    result = ne.execute_merge(["a", "b"], "concat", None, data_map)
    if isinstance(result, pd.DataFrame):
        assert result["x"].isna().sum() == 0  # DataCleaner 已填补
    print(f"  [PASS] NodeExecutor Merge + Clean: cleaned={isinstance(result, pd.DataFrame)}")


async def test_node_executor_sanitize_output():
    """最终输出经过 DataCleaner 清洗（datetime 格式化、inf 处理）。"""
    ne = NodeExecutor()
    data_map = {
        "a": pd.DataFrame({"date": pd.to_datetime(["2023-06-15"])}),
        "b": pd.DataFrame({"val": [np.inf]}),
    }
    result = ne.execute_merge(["a", "b"], "collect", None, data_map)
    assert isinstance(result, dict)
    # datetime → string
    assert result["a"]["date"].iloc[0] == "2023-06-15"
    # inf 已被替换（float64 列中 None 会被存储为 NaN，但不影响 JSON 序列化）
    assert pd.isna(result["b"]["val"].iloc[0])
    print("  [PASS] NodeExecutor 最终输出清洗")


# ═══════════════════════════════════════
# 6. _align_types
# ═══════════════════════════════════════

def test_align_types():
    """int64 → float64。"""
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert df["x"].dtype == "int64"
    aligned = _align_types(df)
    assert aligned["x"].dtype == "float64"
    print("  [PASS] 类型对齐 int64→float64")


# ═══════════════════════════════════════
# 执行
# ═══════════════════════════════════════

def run_sync():
    print("=== DataCleaner ===")
    test_cleaner_nan()
    test_cleaner_inf()
    test_cleaner_datetime()
    test_cleaner_dict()

    print("\n=== SQL Validation ===")
    test_sql_validation()
    test_sql_executor_validation()

    print("\n=== MergeExecutor ===")
    test_merge_concat()
    test_merge_join()
    test_merge_collect()
    test_merge_empty()

    print("\n=== AnalysisExecutor ===")
    test_analysis_empty_data()

    print("\n=== _align_types ===")
    test_align_types()


async def run_async():
    print("\n=== NodeExecutor ===")
    await test_node_executor_sql()
    await test_node_executor_merge()
    await test_node_executor_sanitize_output()


if __name__ == "__main__":
    print("=" * 50)
    print("Executor 验证")
    print("=" * 50)
    run_sync()
    asyncio.run(run_async())
    print("\n" + "=" * 50)
    print("全部通过")
    print("=" * 50)
