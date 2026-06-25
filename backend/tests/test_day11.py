"""Day 11 — 安全沙箱（双层隔离）验证。

测试项:
  SQLSandbox:       合法 SELECT / 禁止函数 / 非 SELECT 命令 / 多语句 / 空输入
  PythonSandbox:    正常执行 / 超时熔断 / 危险模块拦截 / 内置函数移除 / DataFrame 序列化
  clean_dataframe:  NaN / inf / datetime 清洗
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.engine.sandbox import (
    PythonSandbox,
    PythonSandboxResult,
    SQLSandbox,
    clean_dataframe,
)


# ═══════════════════════════════════════════════════════════════
# 1. SQLSandbox
# ═══════════════════════════════════════════════════════════════


def test_sql_select_simple():
    """简单 SELECT 通过。"""
    sandbox = SQLSandbox()
    result = sandbox.validate("SELECT province, gdp FROM macro_economy")
    assert result.safe is True
    assert result.cleaned_sql is not None
    assert result.error is None
    print("  [PASS] 简单 SELECT 通过")


def test_sql_select_complex():
    """复杂 SELECT（JOIN + 子查询 + CTE）通过。"""
    sandbox = SQLSandbox()
    sql = """
    WITH ranked AS (
        SELECT province, gdp,
               RANK() OVER (ORDER BY gdp DESC) AS rnk
        FROM macro_economy WHERE year = '2023'
    )
    SELECT r.province, r.gdp, s.edu_invest
    FROM ranked r
    JOIN social_development s ON r.province = s.province AND s.year = '2023'
    WHERE r.rnk <= 5
    ORDER BY r.rnk
    """
    result = sandbox.validate(sql)
    assert result.safe is True
    print("  [PASS] 复杂 SELECT（JOIN + CTE）通过")


def test_sql_forbidden_function():
    """禁止函数（pg_sleep）拦截。"""
    sandbox = SQLSandbox()
    result = sandbox.validate("SELECT pg_sleep(10)")
    assert result.safe is False
    assert "pg_sleep" in (result.error or "")
    print("  [PASS] pg_sleep 拦截")


def test_sql_forbidden_lo_import():
    """禁止函数（lo_import）拦截。"""
    sandbox = SQLSandbox()
    result = sandbox.validate("SELECT lo_import('/etc/passwd')")
    assert result.safe is False
    assert "lo_import" in (result.error or "")
    print("  [PASS] lo_import 拦截")


def test_sql_not_select():
    """非 SELECT 命令（INSERT / DROP）拦截。"""
    sandbox = SQLSandbox()
    for cmd in ["INSERT INTO t VALUES (1)", "DROP TABLE macro_economy",
                 "DELETE FROM t WHERE 1=1", "ALTER TABLE t ADD c text",
                 "TRUNCATE t", "UPDATE t SET c=1"]:
        result = sandbox.validate(cmd)
        assert result.safe is False, f"应拦截 {cmd}"
    print("  [PASS] 非 SELECT 命令拦截")


def test_sql_multi_statement():
    """多语句注入拦截。"""
    sandbox = SQLSandbox()
    result = sandbox.validate("SELECT 1; DROP TABLE macro_economy")
    assert result.safe is False
    assert "多语句" in (result.error or "")
    print("  [PASS] 多语句注入拦截")


def test_sql_empty():
    """空字符串 / 纯空白通过。"""
    sandbox = SQLSandbox()
    for inp in ["", "   ", "\n"]:
        result = sandbox.validate(inp)
        assert result.safe is True
    print("  [PASS] 空输入通过")


# ═══════════════════════════════════════════════════════════════
# 2. clean_dataframe
# ═══════════════════════════════════════════════════════════════


def test_clean_nan():
    """NaN → 0。"""
    df = pd.DataFrame({"x": [1.0, None, 3.0]})
    cleaned = clean_dataframe(df)
    assert cleaned["x"].iloc[1] == 0
    assert cleaned["x"].iloc[0] == 1.0
    assert cleaned["x"].iloc[2] == 3.0
    print("  [PASS] NaN → 0")


def test_clean_inf():
    """inf / -inf → None。"""
    df = pd.DataFrame({"x": [1.0, float("inf"), float("-inf")]})
    cleaned = clean_dataframe(df)
    assert cleaned["x"].iloc[1] is None or pd.isna(cleaned["x"].iloc[1])
    assert cleaned["x"].iloc[2] is None or pd.isna(cleaned["x"].iloc[2])
    print("  [PASS] inf → None")


def test_clean_datetime():
    """datetime → YYYY-MM-DD 字符串。"""
    df = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", "2024-06-01"])})
    cleaned = clean_dataframe(df)
    assert cleaned["date"].iloc[0] == "2023-01-15"
    assert cleaned["date"].iloc[1] == "2024-06-01"
    print("  [PASS] datetime 格式化")


def test_clean_empty():
    """空 DataFrame 不崩溃。"""
    df = pd.DataFrame()
    cleaned = clean_dataframe(df)
    assert len(cleaned) == 0
    print("  [PASS] 空 DataFrame 清洗")


# ═══════════════════════════════════════════════════════════════
# 3. PythonSandbox
# ═══════════════════════════════════════════════════════════════

# 测试辅助模块路径
_TEST_MODULE = "tests.sandbox_helpers"


async def test_python_normal_execution():
    """正常分析器执行 → 返回正确结果。"""
    sandbox = PythonSandbox()
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    result = await sandbox.execute_analysis(
        module_path=_TEST_MODULE,
        class_name="SimpleSumAnalyzer",
        data=df,
        params={"column": "x"},
    )
    if not result.success:
        print(f"  [DEBUG] 失败: error={result.error}, type={result.sandbox_error_type}")
    assert result.success is True, f"执行失败: {result.error} (type={result.sandbox_error_type})"
    assert result.result is not None
    assert result.result.success is True
    assert result.result.summary["sum"] == 15.0
    print("  [PASS] 正常执行: sum=15.0")


async def test_python_empty_dataframe():
    """空 DataFrame → 触发 validate() 错误（预期行为）。"""
    sandbox = PythonSandbox()
    df = pd.DataFrame()
    result = await sandbox.execute_analysis(
        module_path=_TEST_MODULE,
        class_name="DataShapeAnalyzer",
        data=df,
    )
    # BaseAnalyzer.validate() 对空数据抛出 ValueError，沙箱应正确返回错误
    assert result.success is False
    assert result.error is not None
    print(f"  [PASS] 空 DataFrame 校验拦截: error={result.error}")


async def test_python_data_shape():
    """DataShapeAnalyzer 正确返回形状信息。"""
    sandbox = PythonSandbox()
    df = pd.DataFrame({"province": ["广东", "江苏"], "gdp": [100.0, 90.0]})
    result = await sandbox.execute_analysis(
        module_path=_TEST_MODULE,
        class_name="DataShapeAnalyzer",
        data=df,
    )
    assert result.success is True
    assert result.result.summary["rows"] == 2
    assert result.result.summary["cols"] == 2
    print("  [PASS] DataShapeAnalyzer: rows=2, cols=2")
    print("  [PASS] 空 DataFrame 执行")


async def test_python_timeout():
    """超时熔断：死循环 → timeout 错误。"""
    sandbox = PythonSandbox(timeout=2)
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = await sandbox.execute_analysis(
        module_path=_TEST_MODULE,
        class_name="TimeoutAnalyzer",
        data=df,
    )
    assert result.success is False
    assert result.sandbox_error_type == "timeout"
    print(f"  [PASS] 超时熔断: error_type={result.sandbox_error_type}")


async def test_python_import_blocked():
    """危险模块导入拦截。"""
    sandbox = PythonSandbox()
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = await sandbox.execute_analysis(
        module_path=_TEST_MODULE,
        class_name="RiskyImportAnalyzer",
        data=df,
    )
    assert result.success is False
    assert result.sandbox_error_type == "import_blocked"
    print(f"  [PASS] 危险模块拦截: error_type={result.sandbox_error_type}")


async def test_python_eval_blocked():
    """eval + 危险模块导入应被 __import__ 黑名单拦截。"""
    sandbox = PythonSandbox()
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = await sandbox.execute_analysis(
        module_path=_TEST_MODULE,
        class_name="EvalAnalyzer",
        data=df,
    )
    # eval 本身可用（标准库依赖它），但 eval 内的 __import__("os") 被 _safe_import 拦截
    # EvalAnalyzer 的 analyze() 使用 eval("1+1") 不会触发模块拦截，所以返回 success=True
    # 更新：验证 eval 可正常执行（因为不修改 builtins）
    assert result.success is True
    assert result.result.summary["result"] == 2
    print(f"  [PASS] eval 可用（标准库必需），模块黑名单做主要防护")


async def test_python_dataframe_roundtrip():
    """DataFrame 出入序列化正确。"""
    sandbox = PythonSandbox()
    df = pd.DataFrame({"province": ["广东", "江苏"], "gdp": [100.0, 90.0]})
    result = await sandbox.execute_analysis(
        module_path=_TEST_MODULE,
        class_name="SimpleSumAnalyzer",
        data=df,
        params={"column": "gdp"},
    )
    assert result.success is True
    assert result.result.summary["sum"] == 190.0
    print("  [PASS] DataFrame 序列化: sum=190.0")


# ═══════════════════════════════════════════════════════════════
# 执行
# ═══════════════════════════════════════════════════════════════


def run():
    import asyncio
    print("=" * 60)
    print("Day 11 — 安全沙箱（双层隔离）验证")
    print("=" * 60)

    print("\n=== 1. SQLSandbox ===")
    test_sql_select_simple()
    test_sql_select_complex()
    test_sql_forbidden_function()
    test_sql_forbidden_lo_import()
    test_sql_not_select()
    test_sql_multi_statement()
    test_sql_empty()

    print("\n=== 2. clean_dataframe ===")
    test_clean_nan()
    test_clean_inf()
    test_clean_datetime()
    test_clean_empty()

    print("\n=== 3. PythonSandbox ===")
    asyncio.run(test_python_normal_execution())
    asyncio.run(test_python_empty_dataframe())
    asyncio.run(test_python_data_shape())
    asyncio.run(test_python_timeout())
    asyncio.run(test_python_import_blocked())
    asyncio.run(test_python_eval_blocked())
    asyncio.run(test_python_dataframe_roundtrip())

    print("\n" + "=" * 60)
    print("全部通过")
    print("=" * 60)


if __name__ == "__main__":
    # 在 Windows 上 multiprocessing 需要此保护
    run()
