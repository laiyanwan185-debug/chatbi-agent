"""工具治理框架。

将 BaseAnalyzer 子类封装为受控工具，提供：
  - Pydantic 参数校验（自动从 analyze() 签名生成）
  - RBAC 权限标签隔离
  - 数据隔离（只接收 DataFrame，不持有 DB 连接）
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, get_type_hints

import pandas as pd
from pydantic import BaseModel, create_model
from pydantic import ValidationError as PydanticValidationError

from app.analyzers.base import BaseAnalyzer, AnalysisResult

logger = logging.getLogger(__name__)


class ToolSchema(BaseModel):
    """工具元数据 Schema，供编排层 / LLM 工具定义使用。"""

    name: str
    description: str
    parameters: type[BaseModel]
    category: str
    output_type: str
    timeout: int

    model_config = {"arbitrary_types_allowed": True}


class BaseActionTool:
    """分析算法工具封装。

    将 BaseAnalyzer 子类包装为受控工具，自动从 analyze() 签名生成
    Pydantic 参数校验模型，执行前校验参数、检查权限。

    Args:
        analyzer_cls: BaseAnalyzer 子类（非实例）。
        permissions: 所需权限标签列表，默认 ["user"]。
    """

    def __init__(
        self,
        analyzer_cls: type[BaseAnalyzer],
        permissions: list[str] | None = None,
    ) -> None:
        self._analyzer_cls = analyzer_cls
        self.permissions = permissions or ["user"]
        self._param_model = _build_param_model(analyzer_cls)

    # ------------------------------------------------------------------
    # 属性委托
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._analyzer_cls.name

    @property
    def description(self) -> str:
        return self._analyzer_cls.description

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    def validate_params(self, **kwargs: Any) -> BaseModel:
        """校验参数，返回 Pydantic 模型实例。"""
        return self._param_model.model_validate(kwargs)

    def execute(
        self,
        context: dict[str, Any],
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> AnalysisResult:
        """完整的受控执行管线：权限 → 参数 → validate → analyze。"""
        # 1. 权限检查
        if not self._check_permissions(context):
            return AnalysisResult(
                success=False,
                error=(
                    f"权限不足: 需要 {self.permissions}，"
                    f"当前上下文权限为 {context.get('permissions', [])}"
                ),
            )

        # 2. 参数校验
        try:
            validated = self.validate_params(**kwargs)
        except PydanticValidationError as e:
            return AnalysisResult(success=False, error=f"参数校验失败: {e}")

        # 3. 实例化 + 数据校验
        analyzer = self._analyzer_cls()
        try:
            analyzer.validate(data)
        except ValueError as e:
            return AnalysisResult(success=False, error=str(e))

        # 4. 执行
        try:
            return analyzer.analyze(data, **validated.model_dump())
        except Exception as e:
            logger.exception("算法 '%s' 执行失败", self.name)
            return AnalysisResult(success=False, error=f"执行异常: {e}")

    def get_tool_schema(self) -> ToolSchema:
        """返回工具元数据。"""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self._param_model,
            category=self._analyzer_cls.category,
            output_type=self._analyzer_cls.output_type,
            timeout=self._analyzer_cls.timeout,
        )

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _check_permissions(self, context: dict[str, Any]) -> bool:
        user_perms: list[str] = context.get("permissions", ["user"])
        return any(p in user_perms for p in self.permissions)


def _build_param_model(analyzer_cls: type[BaseAnalyzer]) -> type[BaseModel]:
    """从 analyze() 方法的签名自动生成 Pydantic 参数模型。

    跳过 self / data 参数和 **kwargs。
    """
    sig = inspect.signature(analyzer_cls.analyze)
    hints = get_type_hints(analyzer_cls.analyze)

    fields: dict[str, tuple[type, Any]] = {}
    for name, param in sig.parameters.items():
        if name in ("self", "data", "return"):
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue  # **kwargs

        field_type = hints.get(name, Any)
        if param.default is inspect.Parameter.empty:
            default: Any = ...
        else:
            default = param.default

        fields[name] = (field_type, default)

    model_name = f"{analyzer_cls.__name__}Params"
    return create_model(model_name, **fields)
