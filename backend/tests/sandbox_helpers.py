"""Sandbox 测试辅助 — 模拟分析器类（供子进程导入使用）。"""

from __future__ import annotations

import time

import pandas as pd

from app.analyzers.base import AnalysisResult, BaseAnalyzer


class SimpleSumAnalyzer(BaseAnalyzer):
    """简单的求和分析器（验证正常执行）。"""

    name = "test_sum"
    category = "test"
    description = "Simple sum for sandbox testing"
    inputs = ["*"]
    executor = "python"

    def analyze(self, data: pd.DataFrame, **kwargs) -> AnalysisResult:
        col = kwargs.get("column", data.columns[0] if len(data.columns) > 0 else "x")
        return AnalysisResult(
            success=True,
            data=pd.DataFrame({"sum": [float(data[col].sum())]}),
            summary={"column": col, "sum": float(data[col].sum())},
        )


class DataShapeAnalyzer(BaseAnalyzer):
    """检查数据形状的分析器。"""

    name = "test_shape"
    category = "test"
    description = "Check data shape"
    inputs = ["*"]
    executor = "python"

    def analyze(self, data: pd.DataFrame, **kwargs) -> AnalysisResult:
        return AnalysisResult(
            success=True,
            data=pd.DataFrame({"rows": [len(data)], "cols": [len(data.columns)]}),
            summary={"rows": len(data), "cols": len(data.columns)},
        )


class TimeoutAnalyzer(BaseAnalyzer):
    """死循环分析器（测试超时熔断）。"""

    name = "test_timeout"
    category = "test"
    description = "Infinite loop for timeout testing"
    executor = "python"

    def analyze(self, data: pd.DataFrame, **kwargs) -> AnalysisResult:
        while True:
            time.sleep(0.1)


class RiskyImportAnalyzer(BaseAnalyzer):
    """尝试导入危险模块的分析器（测试导入拦截）。"""

    name = "test_risky_import"
    category = "test"
    description = "Try to import dangerous module"
    executor = "python"

    def analyze(self, data: pd.DataFrame, **kwargs) -> AnalysisResult:
        import os  # noqa: F841 — 这行应该被沙箱拦截
        return AnalysisResult(success=True, summary={"module": "os"})


class EvalAnalyzer(BaseAnalyzer):
    """尝试使用 eval 的分析器（测试内置函数移除）。"""

    name = "test_eval"
    category = "test"
    description = "Try to use eval"
    executor = "python"

    def analyze(self, data: pd.DataFrame, **kwargs) -> AnalysisResult:
        result = eval("1+1")  # noqa: PGH001 — 这行应该被沙箱拦截
        return AnalysisResult(success=True, summary={"result": result})
