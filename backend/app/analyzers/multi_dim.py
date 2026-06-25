"""多维聚合分析算法库。

提供以下算法：
  - Cube             多维交叉汇总（CUBE / ROLLUP 语义）
  - Proportion       占比计算（贡献度/构成比）
  - CV               变异系数（离散度度量）
  - HierarchicalAgg  多级层次聚合（两级指标加权汇总）
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import AnalysisResult, BaseAnalyzer


# =============================================================================
# Cube —— 多维交叉汇总
# =============================================================================

class Cube(BaseAnalyzer):
    """多维交叉汇总（CUBE / ROLLUP 语义）。

    模拟 SQL CUBE/ROLLUP 的多维聚合，按维度组合分组求和/均值/计数等。
    - CUBE: 所有维度子集的全组合
    - ROLLUP: 维度层级递减（如 区域→省份 沿层级上卷）

    Args:
        agg_func: 聚合函数，默认 "sum"。
        agg_name: 聚合结果列名，默认 "value"。
        method: "cube"（全组合）或 "rollup"（层级递减），默认 "cube"。

    Examples:
        >>> c = Cube()
        >>> r = c.analyze(df, dim_cols=["region", "province"],
        ...               value_col="gdp", method="rollup")
        >>> r.data  # 小计行（region 汇总）会出现在结果中
    """

    name = "cube"
    category = "multi_dim"
    description = "CUBE/ROLLUP: 多维交叉汇总，含所有维度组合的小计"
    output_type = "dataframe"
    executor = "python"
    timeout = 15

    def analyze(
        self,
        data: pd.DataFrame,
        dim_cols: list[str] | None = None,
        value_col: str = "value",
        group_by: list[str] | str | None = None,
        agg_func: str = "sum",
        agg_name: str = "value",
        method: str = "cube",
    ) -> AnalysisResult:
        self.validate(data)

        try:
            # 兼容 group_by 参数
            cols = dim_cols if dim_cols else (group_by if isinstance(group_by, list) else [])
            if isinstance(group_by, str):
                cols = [group_by]

            if not cols:
                raise ValueError("请指定 dim_cols 或 group_by 维度列")

            _check_cols(data, cols + [value_col])

            if method == "rollup":
                # ROLLUP: 沿维度层级递减聚合
                frames = []
                for i in range(len(cols), 0, -1):
                    gb = cols[:i]
                    agg = data.groupby(gb, observed=True)[value_col].agg(agg_func).reset_index()
                    # 为未聚合的维度列填充占位符
                    for missing in cols[i:]:
                        agg[missing] = "__total__"
                    frames.append(agg)
                # 全部总计行
                total = pd.DataFrame({value_col: [data[value_col].agg(agg_func)]})
                for c in cols:
                    total[c] = "__total__"
                frames.append(total)
                result = pd.concat(frames, ignore_index=True)

            else:
                # CUBE: 所有维度子集的全组合
                frames = []
                for r in range(len(cols) + 1):
                    from itertools import combinations
                    for subset in combinations(cols, r):
                        if r == 0:
                            total = pd.DataFrame(
                                {value_col: [data[value_col].agg(agg_func)]}
                            )
                            for c in cols:
                                total[c] = "__total__"
                            frames.append(total)
                        else:
                            gb = list(subset)
                            agg = data.groupby(gb, observed=True)[value_col].agg(agg_func).reset_index()
                            for missing in set(cols) - set(subset):
                                agg[missing] = "__total__"
                            frames.append(agg)
                result = pd.concat(frames, ignore_index=True)

            result = result[cols + [value_col]].fillna("__total__")
            result.rename(columns={value_col: agg_name}, inplace=True)

            return AnalysisResult(
                success=True,
                data=result,
                summary={
                    "algorithm": f"cube_{method}",
                    "dimensions": cols,
                    "agg_func": agg_func,
                },
            )

        except (ValueError, TypeError, KeyError) as e:
            # 当维度列与聚合列冲突或数据类型不匹配时，优雅降级
            return AnalysisResult(
                success=False,
                error=f"Cube 分析失败: {e}",
            )


# =============================================================================
# Proportion —— 占比计算
# =============================================================================

class Proportion(BaseAnalyzer):
    """占比计算，计算每个类别在分组内的占比。

    常用于产业结构分析、贡献度计算等。

    Args:
        group_by: 分组列（计算组内占比）。
        as_percent: True 输出百分比（0-100），False 输出小数（0-1），默认 True。
        rank: True 同时输出排序，默认 False。

    Examples:
        >>> p = Proportion()
        >>> r = p.analyze(df, value_col="gdp",
        ...               group_by=["province", "industry"])
        >>> r.data["proportion"]  # 各产业在省份内的占比
    """

    name = "proportion"
    category = "multi_dim"
    description = "占比计算: 每个值在其分组内的占比，支持百分比输出"
    output_type = "dataframe"
    executor = "python"
    timeout = 10

    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str = "value",
        group_by: list[str] | None = None,
        partition_by: list[str] | None = None,
        as_percent: bool = True,
        rank: bool = False,
    ) -> AnalysisResult:
        self.validate(data)
        try:
            _check_cols(data, [value_col])
        except ValueError:
            return AnalysisResult(
                success=False,
                error=f"Proportion 分析缺值列: {value_col}",
            )

        try:
            # 尝试将 value_col 转为数值类型
            df = data[[value_col] + (group_by or []) + (partition_by or [])].copy()
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

            if group_by:
                total = df.groupby(group_by)[value_col].transform("sum")
            else:
                total = df[value_col].sum()

            df["proportion"] = df[value_col] / total
            if as_percent:
                df["proportion"] = df["proportion"] * 100

            if rank:
                if group_by:
                    df["rank"] = df.groupby(group_by)["proportion"].rank(
                        method="dense", ascending=False
                    ).astype(int)
                else:
                    df["rank"] = df["proportion"].rank(
                        method="dense", ascending=False
                    ).astype(int)

            df.sort_values("proportion", ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)

            return AnalysisResult(
                success=True,
                data=df,
                summary={
                    "algorithm": "proportion",
                    "as_percent": as_percent,
                    "group_by": group_by,
                },
            )
        except (ValueError, TypeError, ZeroDivisionError) as e:
            return AnalysisResult(
                success=False,
                error=f"Proportion 分析失败: {e}",
            )


# =============================================================================
# CV —— 变异系数
# =============================================================================

class CV(BaseAnalyzer):
    """变异系数 (Coefficient of Variation)。

    CV = std / mean × 100%，衡量相对离散程度。
    值越大表示组内差异越显著，适合比较不同量纲数据的分散程度。

    Examples:
        >>> cv = CV()
        >>> r = cv.analyze(df, value_col="gdp", group_by=["region"])
        >>> r.data
           region         cv
        0    东部  15.23
        1    西部  28.56
    """

    name = "cv"
    category = "multi_dim"
    description = "变异系数 CV = std/mean × 100%, 衡量组内离散度"
    output_type = "dataframe"
    executor = "python"
    timeout = 10

    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str = "value",
        group_by: list[str] | None = None,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [value_col])

        df = data[[value_col] + (group_by or [])].copy()

        def _cv(group: pd.DataFrame) -> dict[str, float]:
            vals = group[value_col]
            mean_val = vals.mean()
            std_val = vals.std(ddof=1)
            if mean_val == 0 or pd.isna(mean_val) or pd.isna(std_val):
                return {"cv": np.nan}
            return {"cv": std_val / mean_val * 100}

        if group_by:
            result = df.groupby(group_by).apply(_cv, include_groups=False)
            out = pd.json_normalize(result)
            out.index = result.index
            out = out.reset_index()
        else:
            out = pd.DataFrame([_cv(df)])

        return AnalysisResult(success=True, data=out)


# =============================================================================
# HierarchicalAgg —— 多级层次聚合
# =============================================================================

class HierarchicalAgg(BaseAnalyzer):
    """多级层次聚合（两级指标加权汇总）。

    适用于"一级维度等权、二级指标加权"的复合指标体系。
    如：经济分 = 0.5×GDP + 0.5×财政收入；综合分 = 0.3×经济 + 0.3×民生 + 0.4×环境。

    Args:
        level_1_group: 一级分组列（如 "province"）。
        level_2_group: 二级分组列（如 "dimension"）。
        value_col: 指标值列。
        weight_col: 权重列（若为 None 则等权）。
        level_2_agg: 二级聚合函数，默认 "sum"。

    Examples:
        >>> ha = HierarchicalAgg()
        >>> r = ha.analyze(df, level_1_group="province",
        ...                level_2_group="dimension",
        ...                value_col="score", weight_col="weight")
    """

    name = "hierarchical_agg"
    category = "multi_dim"
    description = "多级层次聚合: 一级等权 × 二级加权，输出各级汇总得分"
    output_type = "dataframe"
    executor = "python"
    timeout = 15

    def analyze(
        self,
        data: pd.DataFrame,
        level_1_group: str | list[str] | None = None,
        level_2_group: str | list[str] | None = None,
        value_col: str = "value",
        weight_col: str | None = None,
        level_2_agg: str = "sum",
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [value_col])
        if weight_col:
            _check_cols(data, [weight_col])

        df = data.copy()

        # L2 聚合：在 level_2_group 内加权求和
        l2_gb = []
        if level_1_group:
            l2_gb += [level_1_group] if isinstance(level_1_group, str) else level_1_group
        if level_2_group:
            l2_gb += [level_2_group] if isinstance(level_2_group, str) else level_2_group

        if weight_col:
            df["_weighted"] = df[value_col] * df[weight_col]
            l2 = df.groupby(l2_gb, observed=True)["_weighted"].agg(level_2_agg).reset_index()
        else:
            l2 = df.groupby(l2_gb, observed=True)[value_col].agg(level_2_agg).reset_index()

        l2.rename(columns={value_col if not weight_col else "_weighted": "l2_score"}, inplace=True)

        # L1 聚合：level_1_group 内各 L2 等权平均
        l1_result = l2.copy()
        if level_2_group:
            l1_gb = [level_1_group] if isinstance(level_1_group, str) else (level_1_group or [])
            l1 = l2.groupby(l1_gb, observed=True)["l2_score"].mean().reset_index()
            l1.rename(columns={"l2_score": "l1_score"}, inplace=True)
            l1_result = l1
        else:
            l1_result.rename(columns={"l2_score": "l1_score"}, inplace=True)

        l1_result.sort_values("l1_score", ascending=False, inplace=True)
        l1_result.reset_index(drop=True, inplace=True)

        return AnalysisResult(
            success=True,
            data=l1_result,
            summary={
                "algorithm": "hierarchical_agg",
                "level_1": level_1_group,
                "level_2": level_2_group,
                "level_2_agg": level_2_agg,
            },
        )


# =============================================================================
# 内部工具函数
# =============================================================================

def _check_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"数据中缺少必需的列: {missing}")


# =============================================================================
# __all__
# =============================================================================

__all__ = [
    "Cube",
    "Proportion",
    "CV",
    "HierarchicalAgg",
]
