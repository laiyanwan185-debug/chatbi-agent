"""执行层 — 节点级执行器 + DataFrame 数据清洗防线。

职责边界：
  - SQLExecutor: asyncpg 执行 SQL → Pandas DataFrame
  - AnalysisExecutor: BaseActionTool 算法调用（Pydantic 校验 + RBAC）
  - MergeExecutor: 多 DataFrame 融合（concat / join / collect）+ 维度对齐
  - DataCleaner: 序列化前清洗（fillna → inf 替换 → 日期格式化）
  - NodeExecutor: 统一分派入口（orchestrator 中 DAGExecutor 的调用目标）

调用方：orchestrator.py 的 DAGExecutor。
"""

from __future__ import annotations

import asyncio
import logging
from functools import reduce
from typing import Any

from decimal import Decimal

import numpy as np
import pandas as pd

from app.engine.registry import AnalyzerRegistry
from app.engine.tool_base import BaseActionTool

logger = logging.getLogger(__name__)


# =============================================================================
# 1. 自定义异常
# =============================================================================


class SQLValidationError(Exception):
    """SQL 校验不通过（仅允许只读查询）。"""


# =============================================================================
# 2. DataCleaner — 数据清洗防线
# =============================================================================


class DataCleaner:
    """DataFrame 序列化前清洗管线。

    防止 NaN / Infinity / NaT 导致 Pydantic / JSON 序列化崩溃。
    """

    @staticmethod
    def sanitize(df: pd.DataFrame) -> pd.DataFrame:
        """对单 DataFrame 执行清洗管线。"""
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], None)
        # PostgreSQL 的 NUMERIC 类型返回 Decimal，转换为 float 便于计算
        for col in df.select_dtypes(include=["object"]):
            # 取首个非空元素判断是否为 Decimal
            first = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(first, Decimal):
                df[col] = df[col].astype(float)
        for col in df.select_dtypes(include=["datetime64", "datetimetz"]):
            df[col] = df[col].dt.strftime("%Y-%m-%d")
        return df

    @staticmethod
    def sanitize_output(data: Any) -> Any:
        """递归清洗输出数据（支持 DF / dict）。"""
        if isinstance(data, pd.DataFrame):
            return DataCleaner.sanitize(data)
        if isinstance(data, dict):
            return {k: DataCleaner.sanitize_output(v) for k, v in data.items()}
        return data


# =============================================================================
# 3. SQLExecutor
# =============================================================================


class SQLExecutor:
    """SQL 查询执行器。"""

    async def execute(self, sql: str, db_pool: Any) -> pd.DataFrame:
        """执行 SQL 查询并返回 DataFrame。

        Raises:
            SQLValidationError: SQL 不是只读查询。
            Exception: 数据库执行异常。
        """
        _validate_sql(sql)
        records = await db_pool.fetch(sql)
        return pd.DataFrame([dict(r) for r in records]) if records else pd.DataFrame()


# =============================================================================
# 4. AnalysisExecutor
# =============================================================================


class AnalysisExecutor:
    """分析算法执行器。"""

    async def execute(
        self,
        algorithm_name: str,
        context: dict[str, Any],
        registry: AnalyzerRegistry,
        tool_cache: dict[str, BaseActionTool],
        data: pd.DataFrame,
        params: dict[str, Any],
        loop: asyncio.AbstractEventLoop,
    ) -> pd.DataFrame:
        """在线程池中执行分析算法（CPU-bound）。

        执行前先对 params 中的列名参数做模糊解析，
        确保算法使用的列名与 DataFrame 实际列名匹配。

        Returns:
            分析结果的 DataFrame。

        Raises:
            ValueError: 输入数据为空。
            RuntimeError: 算法执行返回失败。
            Exception: 其他执行异常。
        """
        if data.empty:
            raise ValueError(f"算法 '{algorithm_name}' 的输入数据为空")

        # 列名模糊解析：确保算法 params 中的列名与 DataFrame 实际列名一致
        params = self._resolve_params_columns(params, data)

        if algorithm_name not in tool_cache:
            analyzer_cls = registry.get(algorithm_name)
            tool_cache[algorithm_name] = BaseActionTool(
                analyzer_cls, permissions=["user"],
            )
        tool = tool_cache[algorithm_name]

        result = await loop.run_in_executor(
            None,
            _run_analysis_sync,
            tool, context, data, params,
        )

        if not result.success:
            raise RuntimeError(result.error or f"算法 '{algorithm_name}' 执行返回失败")

        return result.data  # AnalysisResult.data → pd.DataFrame

    @staticmethod
    def _resolve_params_columns(
        params: dict[str, Any],
        data: pd.DataFrame,
    ) -> dict[str, Any]:
        """模糊解析算法参数中的列名，确保与实际 DataFrame 列名匹配。

        算法参数（如 value_col="固定资产投资"）可能因 LLM 生成的
        SQL 输出列名不一致而无法在 DataFrame 中找到。此方法按以下
        优先级尝试匹配：
          1. 精确匹配
          2. 忽略大小写
          3. 去引号/特殊字符
          4. 子串包含
          5. difflib 模糊匹配 (cutoff=0.6)
          6. indicator_registry 反向查找（参数名 → 物理列名）

        对 list 类型参数（如 value_cols=["GDP", "固定资产投资"]），
        逐一解析每个元素。

        Args:
            params: 算法参数字典。
            data:   SQL 执行后输出的 DataFrame。

        Returns:
            列名解析后的参数字典。
        """
        df_cols = list(data.columns)
        if not df_cols:
            return params

        df_cols_lower = {c.lower(): c for c in df_cols}
        resolved: dict[str, Any] = {}

        for key, val in params.items():
            if isinstance(val, list):
                # list 类型：逐一解析每个元素
                resolved_list: list[str] = []
                for item in val:
                    if isinstance(item, str) and len(item) >= 2:
                        matched = AnalysisExecutor._match_column(item, df_cols, df_cols_lower)
                        resolved_list.append(matched or item)
                    else:
                        resolved_list.append(item)
                resolved[key] = resolved_list
                continue

            if not isinstance(val, str) or len(val) < 2:
                resolved[key] = val
                continue

            matched = AnalysisExecutor._match_column(val, df_cols, df_cols_lower)
            resolved[key] = matched or val
            if matched and matched != val:
                logger.debug("Col '%s' → '%s' via match", val, matched)
            elif not matched:
                logger.warning("Col '%s' 在 DataFrame 列 %s 中无匹配", val, df_cols)

        return resolved

    @staticmethod
    def _match_column(val: str, df_cols: list[str], df_cols_lower: dict[str, str]) -> str | None:
        """单列名多级匹配。返回匹配到的 DataFrame 列名，None 表示无匹配。"""
        # Level 0: 精确匹配
        if val in df_cols:
            return val

        # Level 1: 忽略大小写
        if val.lower() in df_cols_lower:
            return df_cols_lower[val.lower()]

        # Level 2: 去引号和特殊字符
        clean = val.strip('"\'').strip()
        if clean and clean != val:
            for col in df_cols:
                if col == clean or col.lower() == clean.lower():
                    return col

        # Level 3: 子串包含
        for col in df_cols:
            if val.lower() in col.lower() or col.lower() in val.lower():
                return col

        # Level 3.5: indicator_registry 反向查找
        try:
            from app.engine.indicator_registry import indicator_registry
            physical_field = indicator_registry.search_field_by_name(val)
            if physical_field:
                if physical_field in df_cols:
                    return physical_field
                if physical_field.lower() in df_cols_lower:
                    return df_cols_lower[physical_field.lower()]
        except Exception:
            pass

        # Level 4: difflib 模糊匹配
        import difflib
        matches = difflib.get_close_matches(val, df_cols, n=1, cutoff=0.6)
        if matches:
            return matches[0]

        # Level 5: indicator_registry 反向查找
        try:
            from app.engine.indicator_registry import indicator_registry
            physical_field = indicator_registry.search_field_by_name(val)
            if physical_field:
                # 物理列名可能也不在 df_cols 中，尝试用物理列名再匹配
                if physical_field in df_cols:
                    return physical_field
                if physical_field.lower() in df_cols_lower:
                    return df_cols_lower[physical_field.lower()]
                # 物理列名模糊匹配
                matches2 = difflib.get_close_matches(physical_field, df_cols, n=1, cutoff=0.6)
                if matches2:
                    return matches2[0]
        except Exception:
            pass

        return None


# =============================================================================
# 5. MergeExecutor
# =============================================================================


class MergeExecutor:
    """结果融合执行器。"""

    def execute(
        self,
        data_sources: list[str],
        merge_strategy: str,
        merge_key: str | None,
        data_map: dict[str, Any],
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """融合多个上游 DataFrame。

        策略:
          "concat"  — 行拼接（同结构 DataFrame），自动对齐列类型
          "join"    — 按 merge_key 列合并，无 key 时按索引
          "collect" — 不合并，原样返回 dict[source_id, df]

        Returns:
            融合后的 DataFrame 或 dict。

        Raises:
            ValueError: 所有上游数据均不可用。
        """
        datas: list[pd.DataFrame] = []
        ds_map: dict[str, pd.DataFrame] = {}
        for ds in data_sources:
            df = data_map.get(ds)
            if isinstance(df, pd.DataFrame) and not df.empty:
                datas.append(df)
                ds_map[ds] = df

        if not datas:
            raise ValueError("所有上游数据均不可用")

        if merge_strategy == "concat":
            return self._concat(datas)
        if merge_strategy == "join":
            return self._join(datas, merge_key)
        return ds_map  # collect

    # ------------------------------------------------------------------
    # 内部合并方法
    # ------------------------------------------------------------------

    @staticmethod
    def _concat(datas: list[pd.DataFrame]) -> pd.DataFrame:
        """行拼接，自动对齐列类型。"""
        aligned = [_align_types(df) for df in datas]
        return pd.concat(aligned, ignore_index=True)

    @staticmethod
    def _join(datas: list[pd.DataFrame], merge_key: str | None) -> pd.DataFrame:
        """按 key 列合并，对齐类型。"""
        aligned = [_align_types(df) for df in datas]
        if merge_key:
            return reduce(
                lambda l, r: pd.merge(l, r, on=merge_key, how="outer"),
                aligned,
            )
        return reduce(
            lambda l, r: pd.merge(l, r, how="outer"),
            aligned,
        )


# =============================================================================
# 6. NodeExecutor — 统一分派入口
# =============================================================================


class NodeExecutor:
    """节点执行器 — 按 node_type 分派到具体执行器。

    内部持有 SQLExecutor / AnalysisExecutor / MergeExecutor / DataCleaner，
    orchestrator 层只需调用对应方法即可。
    """

    def __init__(self) -> None:
        self.sql = SQLExecutor()
        self.analysis = AnalysisExecutor()
        self.merge = MergeExecutor()
        self.cleaner = DataCleaner()

    async def execute_sql(
        self,
        sql: str,
        db_pool: Any,
    ) -> pd.DataFrame:
        """执行 SQL 并清洗。"""
        df = await self.sql.execute(sql, db_pool)
        return self.cleaner.sanitize(df)

    async def execute_analysis(
        self,
        algorithm_name: str,
        context: dict[str, Any],
        registry: AnalyzerRegistry,
        tool_cache: dict[str, BaseActionTool],
        data: pd.DataFrame,
        params: dict[str, Any],
        loop: asyncio.AbstractEventLoop,
    ) -> pd.DataFrame:
        """执行分析算法并清洗。"""
        df = await self.analysis.execute(
            algorithm_name, context, registry,
            tool_cache, data, params, loop,
        )
        if isinstance(df, pd.DataFrame):
            return self.cleaner.sanitize(df)
        return df

    def execute_merge(
        self,
        data_sources: list[str],
        merge_strategy: str,
        merge_key: str | None,
        data_map: dict[str, Any],
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """执行融合并清洗。"""
        result = self.merge.execute(
            data_sources, merge_strategy, merge_key, data_map,
        )
        return self.cleaner.sanitize_output(result)


# =============================================================================
# 7. 内部工具函数
# =============================================================================


def _validate_sql(sql: str) -> None:
    """校验 SQL 是否为只读查询（纯 SELECT / WITH ... SELECT）。"""
    stripped = sql.strip().rstrip(";")
    if not stripped:
        raise SQLValidationError("SQL 为空")

    # 支持 (SELECT ...) UNION (SELECT ...) 等括号包裹的查询
    upper = stripped.upper().lstrip("(")
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise SQLValidationError(f"仅允许 SELECT 查询，收到: {stripped[:80]}")


def _run_analysis_sync(
    tool: BaseActionTool,
    context: dict[str, Any],
    data: pd.DataFrame,
    params: dict[str, Any],
) -> Any:
    """同步执行分析算法（在线程池中运行）。

    执行前自动将 object 类型数值列转换为 float64，
    防止 Decimal/object 类型导致 scipy/numpy 报错。
    """
    # 自动转换数值列：尝试将 object/Decimal 列转为 float64
    for col in data.select_dtypes(include=["object"]).columns:
        try:
            # 取首个非空值判断是否为 Decimal 或可转为数字的类型
            first_valid = data[col].dropna().iloc[0] if not data[col].dropna().empty else None
            if first_valid is not None:
                import decimal
                if isinstance(first_valid, (decimal.Decimal, float, int)):
                    data[col] = data[col].astype(float)
        except (TypeError, ValueError, IndexError):
            pass

    return tool.execute(context, data, **params)


def _align_types(df: pd.DataFrame) -> pd.DataFrame:
    """对齐 DataFrame 列类型，避免 concat/merge 时类型不一致报错。"""
    result = df.copy()
    for col in result.columns:
        if result[col].dtype == "int64":
            result[col] = result[col].astype("float64")
    return result
