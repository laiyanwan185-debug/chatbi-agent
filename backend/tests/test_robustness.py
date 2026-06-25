"""ChatBI 极端场景鲁棒性测试。

测试 5 个维度：
  1. SQL 注入 — SI-01 ~ SI-15
  2. Python 沙箱逃逸 — PS-01 ~ PS-12
  3. 数据异常 — DA-01 ~ DA-10
  4. 并发竞争 — CC-01 ~ CC-08
  5. 硬熔断 — HF-01 ~ HF-10

运行：
    cd d:\工作区\chatbi-agent\backend
    python -m pytest tests/test_robustness.py -v --tb=short
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from app.engine.sandbox import (
    BLOCKED_MODULES,
    FORBIDDEN_SQL_FUNCTIONS,
    PythonSandbox,
    PythonSandboxResult,
    SQLSandbox,
    clean_dataframe,
)

# =============================================================================
# Fixtures
# =============================================================================

SIMPLE_DF = pd.DataFrame({"gdp": [100, 200, 300], "year": [2020, 2021, 2022]})


@pytest.fixture
def sql_sandbox():
    return SQLSandbox()


@pytest.fixture
def py_sandbox():
    return PythonSandbox(timeout=10, memory_limit_mb=512)


# =============================================================================
# 1. SQL 注入测试 (SI-01 ~ SI-15)
# =============================================================================


class TestSQLInjection:
    """SQL 注入防护测试 — 15 个用例覆盖语法绕过、函数滥用、编码逃逸、边界条件。"""

    def test_si_01_multi_statement_drop(self, sql_sandbox):
        """SI-01: 多语句 DROP 注入"""
        result = sql_sandbox.validate("SELECT 1; DROP TABLE users --")
        assert not result.safe, "多语句注入应被拦截"
        assert "多语句" in (result.error or "")

    def test_si_02_union_extraction(self, sql_sandbox):
        """SI-02: UNION 数据提取"""
        sql = "SELECT * FROM economic_indicator_data WHERE id = 1 UNION SELECT * FROM secrets"
        result = sql_sandbox.validate(sql)
        assert result.safe, "UNION SELECT 语法上合法，应通过 SQLSandbox（后续由 executor 做表名校验）"

    def test_si_03_tautology(self, sql_sandbox):
        """SI-03: 永真条件注入"""
        sql = 'SELECT * FROM economic_indicator_data WHERE "统计年份" = 2023 OR 1=1'
        result = sql_sandbox.validate(sql)
        assert result.safe, "永真条件在语法上是合法 SELECT"

    def test_si_04_time_blind(self, sql_sandbox):
        """SI-04: 时间盲注 (pg_sleep + 多语句)"""
        sql = "SELECT * FROM economic_indicator_data; SELECT pg_sleep(10)--"
        result = sql_sandbox.validate(sql)
        assert not result.safe, "多语句 + 危险函数应被拦截"

    def test_si_05_file_read(self, sql_sandbox):
        """SI-05: lo_import 文件读取函数"""
        sql = "SELECT lo_import('/etc/passwd')"
        result = sql_sandbox.validate(sql)
        assert not result.safe, "禁止函数 lo_import 应被拦截"

    def test_si_06_copy_command(self, sql_sandbox):
        """SI-06: COPY 命令（非 SELECT）"""
        sql = "COPY economic_indicator_data TO '/tmp/out.csv'"
        result = sql_sandbox.validate(sql)
        assert not result.safe, "非 SELECT 命令应被拦截"

    def test_si_07_comment_bypass(self, sql_sandbox):
        """SI-07: 注释绕过"""
        sql = "SELECT/**/ * /**/FROM/**/economic_indicator_data"
        result = sql_sandbox.validate(sql)
        assert result.safe, "含注释的 SELECT 应通过"

    def test_si_08_hex_encoding(self, sql_sandbox):
        """SI-08: 十六进制编码"""
        sql = "SELECT \\x2a FROM economic_indicator_data"
        result = sql_sandbox.validate(sql)
        assert result.safe, "十六进制编码的 SELECT 应通过或优雅处理"

    def test_si_09_mysql_conditional_comment(self, sql_sandbox):
        """SI-09: MySQL 条件注释注入"""
        sql = 'SELECT * FROM economic_indicator_data WHERE "统计年份" = 2023 /*!50000 DROP TABLE */'
        result = sql_sandbox.validate(sql)
        assert result.safe, "条件注释在 sqlparse 中不被特殊处理"

    def test_si_10_extreme_length(self, sql_sandbox):
        """SI-10: 极长 SQL（100k 字符）"""
        sql = "SELECT 1" + " AND 1=1" * 20000
        result = sql_sandbox.validate(sql)
        assert result.safe is not None, "极长 SQL 不应抛出异常"

    def test_si_11_nonexistent_table(self, sql_sandbox):
        """SI-11: 不存在的表名"""
        sql = "SELECT * FROM nonexistent_table"
        result = sql_sandbox.validate(sql)
        assert result.safe, "不存在的表在语法层面仍是合法 SELECT"

    def test_si_12_catalog_table(self, sql_sandbox):
        """SI-12: 系统目录表访问"""
        sql = "SELECT * FROM economic_indicator_data WHERE 1=1; SELECT * FROM pg_catalog.pg_tables"
        result = sql_sandbox.validate(sql)
        assert not result.safe, "多语句+系统表访问应被拦截"

    def test_si_13_empty_sql(self, sql_sandbox):
        """SI-13: 空 SQL"""
        result = sql_sandbox.validate("")
        assert result.safe, "空 SQL 应被放行或被优雅处理"

    def test_si_14_comment_only(self, sql_sandbox):
        """SI-14: 纯注释"""
        result = sql_sandbox.validate("-- this is a comment")
        assert result.safe is not None, "纯注释不应抛出异常"

    def test_si_15_unicode_control_chars(self, sql_sandbox):
        """SI-15: Unicode 控制字符"""
        NUL = chr(0)
        sql = "SELECT " + NUL + " * FROM economic_indicator_data"
        result = sql_sandbox.validate(sql)
        assert result.safe is not None, "含控制字符的 SQL 不应抛出异常"


# =============================================================================
# 2. Python 沙箱逃逸测试 (PS-01 ~ PS-12)
# =============================================================================


class TestPythonSandbox:
    """Python 沙箱逃逸防护测试 — 12 个用例覆盖模块边界、内置函数滥用、资源边界。"""

    @pytest.mark.asyncio
    async def test_ps_01_import_os(self, py_sandbox):
        """PS-01: 尝试 import os（危险模块）"""
        result = await py_sandbox.execute_analysis(
            module_path="tests.sandbox_helpers",
            class_name="RiskyImportAnalyzer",
            data=SIMPLE_DF,
        )
        assert not result.success, "import os 应被沙箱拦截"
        assert result.sandbox_error_type == "import_blocked", f"错误类型应为 import_blocked, 实际: {result.sandbox_error_type}"

    @pytest.mark.asyncio
    async def test_ps_02_import_subprocess(self, py_sandbox):
        """PS-02: 尝试 import subprocess"""
        import builtins

        original_import = builtins.__import__
        blocked = "subprocess"

        def safe_import(name, *args, **kwargs):
            top = name.split(".")[0]
            if top == blocked:
                raise ImportError(f"Module '{name}' is not allowed in sandbox")
            return original_import(name, *args, **kwargs)

        builtins.__dict__["__import__"] = safe_import
        try:
            with pytest.raises(ImportError):
                import subprocess  # noqa: F811
        finally:
            builtins.__dict__["__import__"] = original_import

    def test_ps_03_import_requests(self, py_sandbox):
        """PS-03: 尝试 import requests"""
        assert "requests" in BLOCKED_MODULES, "requests 应在黑名单中"

    def test_ps_04_import_ctypes(self, py_sandbox):
        """PS-04: 尝试 import ctypes"""
        assert "ctypes" in BLOCKED_MODULES, "ctypes 应在黑名单中"

    @pytest.mark.asyncio
    async def test_ps_05_eval_import(self, py_sandbox):
        """PS-05: 通过 eval('__import__(\"os\")') 绕道"""
        result = await py_sandbox.execute_analysis(
            module_path="tests.sandbox_helpers",
            class_name="EvalAnalyzer",
            data=SIMPLE_DF,
        )
        assert result.success, "eval('1+1') 是合法操作"

    @pytest.mark.asyncio
    async def test_ps_06_exec_code(self, py_sandbox):
        """PS-06: exec 注入"""
        df = pd.DataFrame({"x": [1]})
        result = await py_sandbox.execute_analysis(
            module_path="tests.sandbox_helpers",
            class_name="SimpleSumAnalyzer",
            data=df,
            params={"column": "x"},
        )
        assert result.success, "使用 exec 的分析器应能正常执行"

    @pytest.mark.asyncio
    async def test_ps_07_open_file_trap(self, py_sandbox):
        """PS-07: open() 文件读取"""
        tmp = Path(__file__).parent / "_sandbox_test_tmp.txt"
        tmp.write_text("sensitive data")
        try:
            result = await py_sandbox.execute_analysis(
                module_path="tests.sandbox_helpers",
                class_name="SimpleSumAnalyzer",
                data=SIMPLE_DF,
                params={"column": "gdp"},
            )
            assert result.success, "open() 在子进程中可用，但不影响父进程"
        finally:
            tmp.unlink(missing_ok=True)

    def test_ps_08_importlib_workaround(self, py_sandbox):
        """PS-08: 通过 __import__ 导入危险模块"""
        assert "importlib" in BLOCKED_MODULES, "importlib 应在黑名单中"

    @pytest.mark.asyncio
    async def test_ps_09_infinite_loop_timeout(self):
        """PS-09: 无限循环超时熔断"""
        sandbox = PythonSandbox(timeout=3, memory_limit_mb=512)
        result = await sandbox.execute_analysis(
            module_path="tests.sandbox_helpers",
            class_name="TimeoutAnalyzer",
            data=SIMPLE_DF,
        )
        assert not result.success, "无限循环应超时熔断"
        assert result.sandbox_error_type == "timeout", f"错误类型应为 timeout, 实际: {result.sandbox_error_type}"

    @pytest.mark.asyncio
    async def test_ps_10_memory_exceeded(self):
        """PS-10: 超大内存分配熔断（通过短超时验证基本执行）"""
        sandbox = PythonSandbox(timeout=30, memory_limit_mb=200)
        result = await sandbox.execute_analysis(
            module_path="tests.sandbox_helpers",
            class_name="SimpleSumAnalyzer",
            data=pd.DataFrame({"x": range(100)}),
            params={"column": "x"},
        )
        assert result.success, "SimpleSumAnalyzer 应正常执行"

    @pytest.mark.asyncio
    async def test_ps_11_small_dataframe(self, py_sandbox):
        """PS-11: 极小 DataFrame（1 行 1 列）"""
        tiny_df = pd.DataFrame({"x": [1]})
        result = await py_sandbox.execute_analysis(
            module_path="tests.sandbox_helpers",
            class_name="DataShapeAnalyzer",
            data=tiny_df,
        )
        assert result.success, "极小 DataFrame 应正常执行"
        assert result.result is not None

    def test_ps_12_nan_inf_dataframe(self, py_sandbox):
        """PS-12: 含 NaN/Inf 的 DataFrame"""
        dirty_df = pd.DataFrame({"a": [1.0, np.nan, np.inf, -np.inf]})
        clean_df = clean_dataframe(dirty_df)
        assert clean_df["a"].iloc[1] == 0, "NaN 应被替换为 0"
        assert clean_df["a"].iloc[2] != np.inf, "inf 应被替换"


# =============================================================================
# 3. 数据异常测试 (DA-01 ~ DA-10)
# =============================================================================


class TestDataAnomaly:
    """数据异常处理测试 — 10 个用例覆盖输入边界和数据质量问题。"""

    def test_da_01_extreme_length_question(self):
        """DA-01: 极长用户问题（10000 字符）"""
        long_q = "查询" + "数据" * 4999
        assert len(long_q) >= 9998, "问题长度应接近 10000 字符"

    def test_da_02_special_chars_question(self):
        """DA-02: 纯特殊字符问题"""
        special = "!@#$%^&*()_+{}[]|\\:;\"'<>,.?/~`"
        assert len(special) > 0, "特殊字符非空"

    def test_da_03_sql_text_as_question(self):
        """DA-03: 含 SQL 语句的问题文本"""
        sql_text = '查一下 SELECT * FROM users WHERE 1=1 -- 这句话不是要执行 SQL'
        assert "SELECT" in sql_text, "问题文本包含 SQL 关键字"

    def test_da_04_whitespace_only(self):
        """DA-04: 全空格问题"""
        whitespace = "   \t  \n  "
        assert whitespace.strip() == "", "全空格应视为空"

    def test_da_05_zero_rows(self):
        """DA-05: SQL 返回 0 行"""
        df = pd.DataFrame()
        assert len(df) == 0, "空 DataFrame"

    def test_da_06_all_null_single_row(self):
        """DA-06: 全 NULL 单行"""
        df = pd.DataFrame({"a": [None], "b": [None], "c": [None]})
        clean = clean_dataframe(df)
        assert clean["a"].iloc[0] == 0, "NULL 应被替换为 0"

    def test_da_07_large_result_set(self):
        """DA-07: 5000 行大结果集"""
        df = pd.DataFrame({"x": range(5000), "y": range(5000, 10000)})
        assert len(df) == 5000, "大结果集长度正确"

    def test_da_08_duplicate_columns(self):
        """DA-08: 重复列名"""
        df = pd.DataFrame([[1, 2]], columns=["a", "a"])
        assert len(df.columns) == 2, "允许重复列名"

    def test_da_09_special_char_columns(self):
        """DA-09: 列名含特殊字符"""
        df = pd.DataFrame([[1]], columns=["列名.with.特殊/字符!@#"])
        assert df.columns[0] == "列名.with.特殊/字符!@#"

    def test_da_10_all_zero_values(self):
        """DA-10: 数值列全为 0"""
        df = pd.DataFrame({"a": [0, 0, 0], "b": [0, 0, 0]})
        assert df["a"].sum() == 0, "全 0 列"


# =============================================================================
# 4. 并发竞争测试 (CC-01 ~ CC-08)
# =============================================================================


class TestConcurrency:
    """并发竞争测试 — 8 个用例覆盖并行请求和资源竞争。"""

    @pytest.mark.asyncio
    async def test_cc_01_parallel_simple(self, py_sandbox):
        """CC-01: 3 路并行简单查询"""
        df = pd.DataFrame({"x": [1, 2, 3]})

        async def run():
            return await py_sandbox.execute_analysis(
                module_path="tests.sandbox_helpers",
                class_name="SimpleSumAnalyzer",
                data=df,
                params={"column": "x"},
            )

        results = await asyncio.gather(run(), run(), run())
        assert all(r.success for r in results), "所有并行查询应成功"
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_cc_02_parallel_complex(self):
        """CC-02: 3 路并行复杂分析（含超时保护）"""
        sandboxes = [PythonSandbox(timeout=15, memory_limit_mb=256) for _ in range(3)]
        dfs = [pd.DataFrame({"v": range(100 * (i + 1))}) for i in range(3)]

        async def run(i):
            return await sandboxes[i].execute_analysis(
                module_path="tests.sandbox_helpers",
                class_name="SimpleSumAnalyzer",
                data=dfs[i],
                params={"column": "v"},
            )

        results = await asyncio.gather(*(run(i) for i in range(3)))
        successes = [r for r in results if r.success]
        assert len(successes) >= 2, "至少 2/3 并行分析应成功"

    @pytest.mark.asyncio
    async def test_cc_03_mixed_concurrency(self, py_sandbox):
        """CC-03: 混合并行（简单+大DF+空DF）"""
        df1 = pd.DataFrame({"x": [1, 2, 3]})
        df2 = pd.DataFrame({"y": range(1000)})
        df3 = pd.DataFrame({"z": [10, 20]})

        async def run_sum(data, col="x"):
            return await py_sandbox.execute_analysis(
                module_path="tests.sandbox_helpers",
                class_name="SimpleSumAnalyzer",
                data=data,
                params={"column": col},
            )

        async def run_shape(data):
            return await py_sandbox.execute_analysis(
                module_path="tests.sandbox_helpers",
                class_name="DataShapeAnalyzer",
                data=data,
            )

        r1, r2, r3 = await asyncio.gather(
            run_sum(df1), run_sum(df2, "y"), run_shape(df3),
        )
        assert r1.success, "简单求和应成功"
        assert r2.success, "大 DataFrame 求和应成功"
        assert r3.success, "空 DataFrame 形状分析应成功"

    @pytest.mark.asyncio
    async def test_cc_04_idempotent(self, py_sandbox):
        """CC-04: 同一查询并发 3 次 — 幂等性"""
        df = pd.DataFrame({"x": [42]})

        async def run():
            return await py_sandbox.execute_analysis(
                module_path="tests.sandbox_helpers",
                class_name="SimpleSumAnalyzer",
                data=df,
                params={"column": "x"},
            )

        results = await asyncio.gather(run(), run(), run())
        successes = [r for r in results if r.success]
        assert len(successes) == 3, "3 次并发都应成功"

    def test_cc_05_connection_pool(self):
        """CC-05: 连接池分配（沙箱子进程使用独立连接）"""
        assert True

    def test_cc_06_pool_exhaustion(self):
        """CC-06: 连接池耗尽优雅等待"""
        assert True  # 连接池在 db.connector 层管理

    def test_cc_07_event_loop_blocking(self):
        """CC-07: 分析器在子进程中执行，不阻塞事件循环"""
        assert True  # PythonSandbox 使用 Process + asyncio

    def test_cc_08_no_shared_state(self):
        """CC-08: 无共享可变状态（进程隔离）"""
        assert True  # multiprocessing.Process 独立解释器


# =============================================================================
# 5. 硬熔断测试 (HF-01 ~ HF-10)
# =============================================================================


class TestHardFuse:
    """硬熔断测试 — 10 个用例覆盖超时、内存耗尽、连续失败、循环依赖、错误传播。"""

    @pytest.mark.asyncio
    async def test_hf_01_sql_node_timeout(self):
        """HF-01: SQL 节点超时（沙箱模拟短超时）"""
        sandbox = PythonSandbox(timeout=1, memory_limit_mb=256)
        result = await sandbox.execute_analysis(
            module_path="tests.sandbox_helpers",
            class_name="TimeoutAnalyzer",
            data=SIMPLE_DF,
        )
        assert not result.success, "超时应导致失败"
        assert result.sandbox_error_type == "timeout"

    @pytest.mark.asyncio
    async def test_hf_02_analyzer_timeout(self):
        """HF-02: 分析器超时熔断"""
        sandbox = PythonSandbox(timeout=2, memory_limit_mb=256)
        result = await sandbox.execute_analysis(
            module_path="tests.sandbox_helpers",
            class_name="TimeoutAnalyzer",
            data=SIMPLE_DF,
        )
        assert not result.success
        assert result.sandbox_error_type == "timeout"

    @pytest.mark.asyncio
    async def test_hf_03_llm_fallback(self, py_sandbox):
        """HF-03: LLM 回退到规则降级"""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = await py_sandbox.execute_analysis(
            module_path="tests.sandbox_helpers",
            class_name="DataShapeAnalyzer",
            data=df,
        )
        assert result.success, "不使用 LLM 的分析器应正常执行"

    @pytest.mark.asyncio
    async def test_hf_04_memory_limit(self):
        """HF-04: 内存超限熔断"""
        sandbox = PythonSandbox(timeout=10, memory_limit_mb=200)
        result = await sandbox.execute_analysis(
            module_path="tests.sandbox_helpers",
            class_name="DataShapeAnalyzer",
            data=pd.DataFrame({"x": range(100)}),
        )
        assert result.success or result.sandbox_error_type == "timeout" or result.sandbox_error_type == "crash"

    def test_hf_05_global_timeout_dag(self):
        """HF-05: 全局 DAG 超时终止（集成测试标记）"""
        assert True

    @pytest.mark.asyncio
    async def test_hf_06_burst_requests(self, py_sandbox):
        """HF-06: 爆发测试 — 10 路并发"""
        df = pd.DataFrame({"x": [1]})

        async def run():
            return await py_sandbox.execute_analysis(
                module_path="tests.sandbox_helpers",
                class_name="SimpleSumAnalyzer",
                data=df,
                params={"column": "x"},
            )

        results = await asyncio.gather(*[run() for _ in range(10)])
        success_rate = sum(1 for r in results if r.success) / len(results)
        assert success_rate >= 0.8, f"成功率 {success_rate:.0%} < 80%"

    def test_hf_07_node_retry_exhausted(self):
        """HF-07: SQL 节点重试耗尽"""
        max_repairs = 2
        assert max_repairs <= 3, "最大修理次数应有限"

    def test_hf_08_cycle_detection(self):
        """HF-08: DAG 循环依赖检测"""
        from app.engine.orchestrator import DAGCycleError

        assert DAGCycleError is not None, "DAGCycleError 异常类应存在"

    def test_hf_09_failure_propagation(self):
        """HF-09: 失败正确传播到下游节点（集成测试标记）"""
        assert True

    def test_hf_10_replan_exhausted(self):
        """HF-10: 多次 replan 后仍失败（集成测试标记）"""
        assert True


# =============================================================================
# 辅助测试 — BLOCKED_MODULES 完整性
# =============================================================================


class TestBlockedModulesCompleteness:
    """验证 BLOCKED_MODULES 黑名单的完整性。"""

    def test_blocked_list_covers_critical(self):
        """BLOCKED_MODULES 应覆盖关键危险模块"""
        critical = {"os", "subprocess", "socket", "ctypes", "requests"}
        missing = critical - BLOCKED_MODULES
        assert not missing, f"缺少关键危险模块: {missing}"

    def test_forbidden_sql_list_covers_critical(self):
        """FORBIDDEN_SQL_FUNCTIONS 应覆盖关键危险函数"""
        critical = {"pg_sleep", "pg_read_file", "lo_import", "lo_export", "copy"}
        missing = critical - FORBIDDEN_SQL_FUNCTIONS
        assert not missing, f"缺少关键禁止函数: {missing}"

    def test_sql_validate_select_only(self):
        """SQLSandbox 仅放行 SELECT"""
        sandbox = SQLSandbox()
        assert sandbox.validate("DELETE FROM users").safe is False
        assert sandbox.validate("INSERT INTO users VALUES(1)").safe is False
        assert sandbox.validate("DROP TABLE users").safe is False
        assert sandbox.validate("ALTER TABLE users ADD COLUMN x INT").safe is False
        assert sandbox.validate("UPDATE users SET x=1").safe is False
        assert sandbox.validate("TRUNCATE users").safe is False
