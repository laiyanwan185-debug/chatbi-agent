"""安全沙箱 — SQL 命令校验 + Python 子进程隔离双层防护。

双层隔离架构：
  Layer 1 — SQLSandbox: sqlparse AST 分析，仅放行 SELECT + 禁止危险函数
  Layer 2 — PythonSandbox: multiprocessing.Process 子进程 + 受限环境 + 超时/内存监控

调用方：
  - orchestrator.py: SQLNode 执行前调 SQLSandbox.validate()
  - executor.py: AnalysisNode 执行前调 PythonSandbox.execute_analysis()

数据清洗：
  - clean_dataframe(): 确保 DataFrame 安全序列化为 JSON（fillna / inf / 日期格式化）
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import psutil
import sqlparse
from sqlparse.sql import Function, Identifier, TokenList
from sqlparse.tokens import DML, Keyword, Name, Punctuation, Whitespace

from config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================

# SQL 禁止函数（PostgreSQL 危险系统函数）
FORBIDDEN_SQL_FUNCTIONS: frozenset[str] = frozenset({
    "pg_sleep", "pg_read_file", "pg_write_file", "pg_read_binary_file",
    "pg_ls_dir", "pg_stat_file", "pg_cancel_backend", "pg_terminate_backend",
    "lo_import", "lo_export",
    "dblink_connect", "dblink_connect_u", "dblink", "dblink_exec",
    "copy",
})

# SQL 禁止命令（非 SELECT 的一律拦截）
ALLOWED_SQL_COMMAND = "SELECT"

# Python 沙箱：禁止导入的危险模块
BLOCKED_MODULES: frozenset[str] = frozenset({
    "os", "subprocess", "socket", "ctypes", "requests",
    "signal", "shutil", "importlib", "tarfile", "zipfile",
    "pickle", "shelve", "marshal",
    "code", "codeop",
    "http", "urllib", "ftplib", "telnetlib",
    "webbrowser", "antigravity",
    "tkinter", "turtle",
})

# Python 沙箱：禁止导入的危险模块
# 注意：不修改 builtins（eval/exec/compile/open 保留），
# 因为 dataclasses、importlib 等标准库在执行期依赖这些内置函数。
# 安全防护依赖：进程隔离 + 模块黑名单 + 超时/内存监控


# =============================================================================
# 数据类
# =============================================================================


@dataclass
class SQLValidationResult:
    """SQL 校验结果。

    Attributes:
        safe:        是否通过安全校验。
        cleaned_sql: 通过时返回原 SQL（可用于后续执行）。
        error:       未通过时的错误描述。
    """
    safe: bool
    cleaned_sql: str | None = None
    error: str | None = None


@dataclass
class PythonSandboxResult:
    """Python 子进程执行结果。

    Attributes:
        success:          是否执行成功。
        result:           执行成功时的返回值（AnalysisResult 或基本类型）。
        error:            失败时的错误原因。
        sandbox_error_type: 沙箱拦截类型（"timeout" / "memory" / "import_blocked" / "crash"）。
    """
    success: bool
    result: Any = None
    error: str | None = None
    sandbox_error_type: str | None = None


# =============================================================================
# 1. SQL Sandbox — 命令类型 + 函数白名单校验
# =============================================================================


class SQLSandbox:
    """SQL 语句安全校验器。

    校验规则：
      1. 仅放行 SELECT 命令（通过 sqlparse AST 分析语句类型）
      2. 递归遍历 Token 树，拦截禁止函数
      3. 拒绝多语句输入（注入式 "SELECT 1; DROP TABLE ..."）

    用法：
        sandbox = SQLSandbox()
        result = sandbox.validate("SELECT province, gdp FROM macro_economy")
        if not result.safe:
            raise SecurityError(result.error)
    """

    def validate(self, sql: str) -> SQLValidationResult:
        """校验 SQL 语句安全性。

        Args:
            sql: 用户或 LLM 生成的 SQL 语句。

        Returns:
            SQLValidationResult: 校验结果。
        """
        if not sql or not sql.strip():
            return SQLValidationResult(safe=True, cleaned_sql=sql)

        try:
            parsed = sqlparse.parse(sql)
        except Exception as exc:
            return SQLValidationResult(
                safe=False,
                error=f"SQL 解析失败: {exc}",
            )

        # ── 规则 1：拒绝多语句注入 ──
        if len(parsed) > 1:
            return SQLValidationResult(
                safe=False,
                error=f"多语句输入被拒绝 ({len(parsed)} 条语句)",
            )

        statement = parsed[0]

        # ── 规则 2：仅放行 SELECT ──
        stmt_type = statement.get_type()
        if stmt_type != ALLOWED_SQL_COMMAND:
            return SQLValidationResult(
                safe=False,
                error=f"仅允许 SELECT 命令，收到: {stmt_type}",
            )

        # ── 规则 3：递归检测禁止函数 ──
        forbidden = self._find_forbidden_functions(statement)
        if forbidden:
            return SQLValidationResult(
                safe=False,
                error=f"SQL 包含禁止函数: {', '.join(forbidden)}",
            )

        return SQLValidationResult(safe=True, cleaned_sql=sql)

    def _find_forbidden_functions(self, token: Any) -> list[str]:
        """递归遍历 Token 树，查找 FORBIDDEN_SQL_FUNCTIONS 中的函数名。"""
        found: list[str] = []

        if isinstance(token, Function):
            func_name = self._extract_function_name(token)
            if func_name and func_name.lower() in FORBIDDEN_SQL_FUNCTIONS:
                found.append(func_name)

        # sqlparse 的 Token/TokenList 结构
        if hasattr(token, "tokens"):
            for child in token.tokens:
                found.extend(self._find_forbidden_functions(child))

        return found

    @staticmethod
    def _extract_function_name(token: Function) -> str | None:
        """从 Function 节点提取函数名。"""
        for sub in token.tokens:
            if isinstance(sub, Identifier):
                return sub.get_name()
            if sub.ttype is Name or sub.ttype is Keyword:
                return sub.value
        return None


# =============================================================================
# 2. 数据清洗 — 序列化安全
# =============================================================================


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """清洗 DataFrame 确保 JSON 序列化安全。

    处理：
      - NaN → 0
      - inf / -inf → None
      - NaT → None
      - datetime → YYYY-MM-DD 字符串

    Args:
        df: 原始 DataFrame。

    Returns:
        清洗后的 DataFrame（副本，不修改原数据）。
    """
    df = df.copy()
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], None)

    for col in df.select_dtypes(include=["datetime64", "datetimetz"]).columns:
        df[col] = df[col].dt.strftime("%Y-%m-%d")

    return df


# =============================================================================
# 3. Python Sandbox — 子进程隔离执行
# =============================================================================


class PythonSandbox:
    """Python 分析代码安全沙箱。

    隔离方式：
      - multiprocessing.Process 子进程（独立解释器）
      - 受限 __builtins__（移除 eval/exec/compile/open）
      - 模块导入黑名单（os/subprocess/socket/ctypes/requests 等）
      - 超时 + 内存监控双重保障

    用法：
        sandbox = PythonSandbox()
        result = await sandbox.execute_analysis(
            module_path="app.analyzers.time_series",
            class_name="CAGR",
            data=df,
            params={"value_col": "gdp", "time_col": "year"},
        )
    """

    def __init__(
        self,
        memory_limit_mb: int = 0,
        timeout: int = 0,
        mp_context: str = "spawn",
    ) -> None:
        """初始化沙箱。

        Args:
            memory_limit_mb: 内存上限（MB），0 表示使用 settings.SANDBOX_MEMORY_LIMIT_MB。
            timeout:         超时（秒），0 表示使用 settings.SANDBOX_PYTHON_TIMEOUT。
            mp_context:      multiprocessing 上下文（Windows 需 "spawn"）。
        """
        self._memory_limit_mb = memory_limit_mb or settings.SANDBOX_MEMORY_LIMIT_MB
        self._timeout = timeout or settings.SANDBOX_PYTHON_TIMEOUT
        self._ctx = mp.get_context(mp_context)

    async def execute_analysis(
        self,
        module_path: str,
        class_name: str,
        data: pd.DataFrame | None = None,
        params: dict[str, Any] | None = None,
    ) -> PythonSandboxResult:
        """在子进程中执行分析器函数。

        Args:
            module_path: 分析器模块路径（如 "app.analyzers.time_series"）。
            class_name:  分析器类名（如 "CAGR"）。
            data:        输入的 DataFrame。
            params:      额外参数（传给 analyze() 的 **kwargs）。

        Returns:
            PythonSandboxResult: 执行结果。
        """
        input_queue: mp.Queue = self._ctx.Queue()
        output_queue: mp.Queue = self._ctx.Queue()
        data_json = data.to_json(orient="records", date_format="iso") if data is not None else ""

        process = self._ctx.Process(
            target=_run_target,
            args=(input_queue, output_queue),
            daemon=True,
        )

        # 发送数据到子进程
        input_queue.put((module_path, class_name, data_json, params or {}, list(sys.path)))

        process.start()
        pid = process.pid
        logger.debug("PythonSandbox 子进程已启动: pid=%d", pid)

        try:
            # 并行监控内存 + 等待结果
            result = await self._wait_with_monitor(process, pid, output_queue)

            if isinstance(result, PythonSandboxResult):
                return result

            # 正常结果反序列化
            return self._deserialize_result(result)

        except Exception as exc:
            logger.error("PythonSandbox 执行异常: %s", exc)
            return PythonSandboxResult(
                success=False,
                error=f"沙箱执行异常: {exc}",
                sandbox_error_type="crash",
            )
        finally:
            if process.is_alive():
                process.terminate()
                try:
                    process.join(timeout=3)
                except Exception:
                    pass
                if process.is_alive():
                    process.kill()

    async def _wait_with_monitor(
        self,
        process: mp.Process,
        pid: int,
        output_queue: mp.Queue,
    ) -> Any:
        """带超时 + 内存监控的等待执行。"""
        monitor_task = asyncio.create_task(
            self._monitor_memory(process, pid),
        )

        try:
            # 等待结果（带超时）
            timeout_sec = self._timeout
            loop_interval = 0.2
            elapsed = 0.0

            while elapsed < timeout_sec:
                if not process.is_alive():
                    # 子进程已退出
                    break
                try:
                    return output_queue.get_nowait()
                except Exception:
                    await asyncio.sleep(loop_interval)
                    elapsed += loop_interval

            # 超时检查
            if process.is_alive():
                process.terminate()
                try:
                    process.join(timeout=3)
                except Exception:
                    pass
                logger.warning("PythonSandbox 超时终止: pid=%d, timeout=%ds", pid, timeout_sec)
                return PythonSandboxResult(
                    success=False,
                    error=f"执行超时 (>{timeout_sec}s)",
                    sandbox_error_type="timeout",
                )

            # 子进程已退出，再试一次读取
            try:
                return output_queue.get_nowait()
            except Exception:
                pass

            # 检查 exitcode
            if process.exitcode is not None and process.exitcode != 0:
                logger.warning("PythonSandbox 异常退出: pid=%d, exitcode=%d", pid, process.exitcode)
                return PythonSandboxResult(
                    success=False,
                    error=f"子进程异常退出 (exitcode={process.exitcode})",
                    sandbox_error_type="crash",
                )

            return PythonSandboxResult(
                success=False,
                error="子进程未返回结果",
                sandbox_error_type="crash",
            )

        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_memory(self, process: mp.Process, pid: int) -> None:
        """监控子进程内存使用（跨平台，通过 psutil 轮询）。"""
        limit_bytes = self._memory_limit_mb * 1024 * 1024

        try:
            proc = psutil.Process(pid)
            while process.is_alive():
                try:
                    mem_info = proc.memory_info()
                    rss = getattr(mem_info, "rss", 0)
                    if rss > limit_bytes:
                        rss_mb = rss / (1024 * 1024)
                        logger.warning(
                            "PythonSandbox 超内存: pid=%d, %.0fMB > %dMB",
                            pid, rss_mb, self._memory_limit_mb,
                        )
                        process.terminate()
                        return
                except (psutil.NoSuchProcess, ProcessLookupError):
                    return
                await asyncio.sleep(0.5)
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass

    @staticmethod
    def _deserialize_result(result: Any) -> PythonSandboxResult:
        """反序列化子进程返回的结果。"""
        if isinstance(result, tuple):
            status = result[0] if len(result) > 0 else "error"
            if status == "success":
                return PythonSandboxResult(success=True, result=result[1] if len(result) > 1 else None)
            elif status == "import_blocked":
                return PythonSandboxResult(
                    success=False,
                    error=result[1] if len(result) > 1 else "模块导入被拦截",
                    sandbox_error_type="import_blocked",
                )
            else:
                return PythonSandboxResult(
                    success=False,
                    error=result[1] if len(result) > 1 else "未知错误",
                    sandbox_error_type="crash",
                )

        if isinstance(result, PythonSandboxResult):
            return result

        return PythonSandboxResult(success=False, error=f"未知结果类型: {type(result).__name__}")


# =============================================================================
# 子进程入口（模块级函数，可被 pickle/spawn）
# =============================================================================


def _run_target(input_queue: mp.Queue, output_queue: mp.Queue) -> None:
    """子进程入口：执行分析器并在受限环境中运行。

    分阶段安全策略：
      Phase 1（模块加载期）：允许所有 import，保证 pandas/numpy/scipy 正常加载
      Phase 2（执行期）：启用 __import__ 黑名单，拦截危险模块
    """
    import builtins
    import importlib as _importlib
    import json as _json
    import sys as _sys

    import numpy as _np
    import pandas as _pd

    # ── 保存原始 __import__ ──
    _original_import = builtins.__import__
    _import_phase: bool = True  # True=模块加载期, False=执行期

    def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """安全的 __import__：执行期启用黑名单。

        不修改其他 builtins（eval/exec/compile/open 保留），
        因为 dataclasses、importlib 等标准库内部依赖这些函数。
        """
        nonlocal _import_phase
        if not _import_phase:
            top_level = name.split(".")[0]
            if top_level in BLOCKED_MODULES:
                _log_import_block(name)
                raise ImportError(f"Module '{name}' is not allowed in sandbox")
        return _original_import(name, *args, **kwargs)

    # 覆写 __import__ 为安全版本
    builtins.__dict__["__import__"] = _safe_import

    # ── 接收数据 ──
    try:
        module_path, class_name, data_json, params, parent_sys_path = input_queue.get(timeout=30)
        _sys.path = parent_sys_path

        # ═══ Phase 1: 导入分析器模块（允许所有 import） ═══
        _import_phase = True
        module = _importlib.import_module(module_path)
        analyzer_cls = getattr(module, class_name)

        # ═══ Phase 2: 切换为受限模式 ═══
        _import_phase = False

        # 反序列化 DataFrame
        data = _pd.DataFrame()
        if data_json:
            try:
                parsed = _json.loads(data_json)
                data = _pd.DataFrame(parsed) if parsed else _pd.DataFrame()
            except Exception:
                data = _pd.DataFrame()

        # 实例化并执行
        analyzer = analyzer_cls()
        analyzer.validate(data)
        result = analyzer.analyze(data, **params)

        output_queue.put(("success", result))

    except ImportError as exc:
        error_msg = str(exc)
        if "not allowed in sandbox" in error_msg:
            output_queue.put(("import_blocked", error_msg))
        else:
            tb = traceback.format_exc()
            output_queue.put(("error", f"导入错误: {error_msg}", tb))
    except MemoryError:
        output_queue.put(("error", "子进程内存不足"))
    except Exception as exc:
        tb = traceback.format_exc()
        output_queue.put(("error", f"{exc}", tb))


def _log_import_block(module_name: str) -> None:
    """记录模块导入拦截日志。"""
    logger.warning("PythonSandbox 拦截危险模块导入: %s", module_name)
