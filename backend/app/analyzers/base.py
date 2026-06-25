"""算法分析器基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class AnalysisResult:
    """算法执行后的标准化输出。"""

    success: bool
    data: pd.DataFrame | dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAnalyzer(ABC):
    """所有分析算法的抽象基类。

    子类只需实现 analyze() 方法，注册中心通过 name 自动发现。

    Attributes:
        name: 算法唯一标识名（如 "cagr", "pearson"），用于 DAG 节点引用。
        category: 所属类别（如 "time_series", "correlation"）。
        description: 功能描述。
        inputs: 所需输入字段名列表（"*" 表示任意多列）。
        output_type: 输出类型描述（如 "scalar", "series", "dataframe"）。
        executor: 执行引擎偏好（"python"）。
        timeout: 节点超时阈值（秒）。
    """

    name: str = ""
    category: str = ""
    description: str = ""
    inputs: list[str] | None = None
    output_type: str = "dataframe"
    executor: str = "sql"  # "sql" | "python"
    timeout: int = 30

    def validate(self, data: pd.DataFrame) -> bool:
        """校验输入数据是否满足算法要求。返回 True 表示通过。"""
        if self.inputs and self.inputs != ["*"]:
            missing = [c for c in self.inputs if c not in data.columns]
            if missing:
                raise ValueError(f"缺少必要的输入字段: {missing}")
        if data.empty:
            raise ValueError("输入数据为空")
        return True

    @abstractmethod
    def analyze(self, data: pd.DataFrame, **kwargs: Any) -> AnalysisResult:
        """核心分析方法，子类必须实现。"""
        ...
