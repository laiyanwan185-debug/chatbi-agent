"""排名与比较分析算法库。

提供以下算法：
  - Rank               排名（RANK / DENSE_RANK / 降序版）
  - PercentRank        百分位排名（PERCENT_RANK）
  - NTile              等频分桶
  - BenchmarkCompare   基准对比（与全国均值/标杆值的偏离分析）
  - RankDisparity      排名差分析（两组排名的差异比较）
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from .base import AnalysisResult, BaseAnalyzer


# =============================================================================
# Rank —— 排名
# =============================================================================

class Rank(BaseAnalyzer):
    """排名的两种模式：RANK（跳跃）与 DENSE_RANK（紧凑）。

    - RANK: 同值同序，后续跳过（如 1, 2, 2, 4）
    - DENSE_RANK: 同值同序，后续不跳过（如 1, 2, 2, 3）

    Args:
        method: "rank"(跳跃式) 或 "dense"(紧凑式)，默认 "dense"。
        ascending: True=从小到大（值越小排越前），False=从大到小，默认 False。

    Examples:
        >>> r = Rank()
        >>> result = r.analyze(df, value_col="gdp", group_by=["year"])
        >>> result.data.columns  # 含 "rank" 列
    """

    name = "rank"
    category = "ranking"
    description = "RANK / DENSE_RANK: 排名计算，支持分组和升降序"
    output_type = "dataframe"
    executor = "python"
    timeout = 10

    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str = "value",
        group_by: list[str] | None = None,
        method: Literal["rank", "dense"] = "dense",
        ascending: bool = False,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [value_col])

        df = data[[value_col] + (group_by or [])].copy()

        rank_method = "min" if method == "rank" else "dense"

        if group_by:
            df["rank"] = df.groupby(group_by)[value_col].rank(
                method=rank_method, ascending=ascending
            )
        else:
            df["rank"] = df[value_col].rank(
                method=rank_method, ascending=ascending
            )

        df["rank"] = df["rank"].astype(int)
        df.sort_values("rank", inplace=True)
        df.reset_index(drop=True, inplace=True)

        return AnalysisResult(
            success=True,
            data=df,
            summary={"algorithm": f"{method}_rank", "ascending": ascending},
        )


# =============================================================================
# PercentRank —— 百分位排名
# =============================================================================

class PercentRank(BaseAnalyzer):
    """百分位排名 (PERCENT_RANK)，将值映射到 [0, 1] 区间。

    PERCENT_RANK = (RANK - 1) / (N - 1)
    值越大表示排名越靠前（值越大），适合作为多维度融合的归一化步骤。

    Examples:
        >>> pr = PercentRank()
        >>> result = pr.analyze(df, value_col="gdp", group_by=["year"])
        >>> result.data["pct_rank"].between(0, 1).all()
        True
    """

    name = "percent_rank"
    category = "ranking"
    description = "PERCENT_RANK: 百分位排名 [0,1]，值越大排名越靠前"
    output_type = "dataframe"
    executor = "python"
    timeout = 10

    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str = "value",
        group_by: list[str] | None = None,
        ascending: bool = False,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [value_col])

        df = data[[value_col] + (group_by or [])].copy()

        if group_by:
            df["pct_rank"] = df.groupby(group_by)[value_col].rank(
                method="min", ascending=ascending, pct=True
            )
        else:
            df["pct_rank"] = df[value_col].rank(
                method="min", ascending=ascending, pct=True
            )

        df.sort_values("pct_rank", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return AnalysisResult(
            success=True,
            data=df,
            summary={"algorithm": "percent_rank", "ascending": ascending},
        )


# =============================================================================
# NTile —— 等频分桶
# =============================================================================

class NTile(BaseAnalyzer):
    """等频分桶 (NTILE)，将数据按值大小平均分成 N 桶。

    n=2 → 高/低两组；n=3 → 高/中/低三组；n=4 → 四分位。

    Args:
        n: 分桶数，默认 4。

    Examples:
        >>> nt = NTile(n=4)
        >>> result = nt.analyze(df, value_col="gdp", group_by=["year"])
        >>> result.data["ntile"].value_counts()  # 每桶约 25%
    """

    name = "ntile"
    category = "ranking"
    description = "NTILE: 等频分桶，将数据平均分成 N 组"
    output_type = "dataframe"
    executor = "python"
    timeout = 10

    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str = "value",
        group_by: list[str] | None = None,
        n: int = 4,
        ascending: bool = False,
        labels: list[str] | None = None,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [value_col])

        if n < 2:
            raise ValueError("分桶数 n 必须 >= 2")

        df = data[[value_col] + (group_by or [])].copy()

        # qcut 按分位数等频切分，支持复制值降级
        def _ntile(group: pd.DataFrame) -> pd.Series:
            vals = group[value_col]
            unique_count = vals.nunique()
            actual_n = min(n, unique_count)

            # 先尝试 qcut（等频分箱）
            if actual_n >= 2:
                try:
                    # 对排名做 qcut 避免重复值问题
                    ranked = vals.rank(method="first")
                    bins = pd.qcut(
                        ranked,
                        q=actual_n,
                        labels=labels[:actual_n] if labels else list(range(1, actual_n + 1)),
                        duplicates="drop",
                    )
                    # 如果 qcut 返回的 bin 数不等于 actual_n，用 cut 降级
                    if bins.nunique() < actual_n:
                        raise ValueError("qcut bin count mismatch")
                    return bins
                except Exception:
                    pass

            # Fallback: cut 等宽分箱
            try:
                bins = pd.cut(
                    vals,
                    bins=actual_n,
                    labels=labels[:actual_n] if labels else list(range(1, actual_n + 1)),
                )
                return bins
            except Exception:
                pass

            # 最终兜底: 全部分到第 1 桶
            return pd.Series([1] * len(vals), index=vals.index)

        if group_by:
            df["ntile"] = df.groupby(group_by, group_keys=False).apply(
                _ntile, include_groups=False
            )
        else:
            df["ntile"] = _ntile(df)

        if labels is None:
            df["ntile"] = df["ntile"].astype(int)

        df.sort_values("ntile", ascending=ascending, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return AnalysisResult(
            success=True,
            data=df,
            summary={"algorithm": "ntile", "n": n, "labels": labels},
        )


# =============================================================================
# BenchmarkCompare —— 基准对比
# =============================================================================

class BenchmarkCompare(BaseAnalyzer):
    """基准对比 (Benchmark Comparison)。

    计算每个值与基准值的偏离程度：偏离度 = (value - benchmark) / |benchmark| × 100

    支持多指标基准对比，自动处理正向/负向指标（负向指标逆转偏离方向）。

    Args:
        benchmark: 基准值。若为 None 则计算 data 的均值作为基准。
        reverse_cols: 负向指标列名列表（如失业率、AQI），值越低越好。
        as_score: True 时将偏离度缩放到 [0, 100] 正向分数。

    Examples:
        >>> bc = BenchmarkCompare()
        >>> r = bc.analyze(df, value_col=["gdp", "unemployment_rate"],
        ...                group_by=["province"],
        ...                reverse_cols=["unemployment_rate"])
    """

    name = "benchmark_compare"
    category = "ranking"
    description = "基准对比: 计算每个值与全国均值/标杆值的偏离百分比"
    output_type = "dataframe"
    executor = "python"
    timeout = 10

    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str | list[str] = "value",
        group_by: list[str] | None = None,
        benchmark: float | dict[str, float] | None = None,
        reverse_cols: list[str] | None = None,
        as_score: bool = False,
    ) -> AnalysisResult:
        cols = [value_col] if isinstance(value_col, str) else value_col
        self.validate(data)
        _check_cols(data, cols)

        df = data[cols + (group_by or [])].copy()
        reverse_cols = set(reverse_cols or [])
        benchmark_dict: dict[str, float] = {}

        for col in cols:
            if isinstance(benchmark, dict) and col in benchmark:
                bm = benchmark[col]
            elif isinstance(benchmark, (int, float)):
                bm = benchmark
            else:
                bm = data[col].mean()
            benchmark_dict[col] = bm

        for col in cols:
            bm = benchmark_dict[col]
            deviation = (df[col] - bm) / (abs(bm) if bm != 0 else 1) * 100
            # 负向指标：偏离方向逆转
            if col in reverse_cols:
                deviation = -deviation

            if as_score:
                # 缩放到 [0, 100]
                d_min, d_max = deviation.min(), deviation.max()
                if d_max > d_min:
                    df[f"{col}_score"] = (deviation - d_min) / (d_max - d_min) * 100
                else:
                    df[f"{col}_score"] = 50.0
            else:
                df[f"{col}_deviation"] = deviation

            df[f"{col}_benchmark"] = bm

        return AnalysisResult(
            success=True,
            data=df,
            summary={
                "algorithm": "benchmark_compare",
                "benchmarks": benchmark_dict,
                "reverse_cols": list(reverse_cols),
            },
        )


# =============================================================================
# RankDisparity —— 排名差分析
# =============================================================================

class RankDisparity(BaseAnalyzer):
    """排名差分析 (Rank Disparity)。

    计算两列排名的差异，找出"排名偏差"显著的个体。
    常用于交叉域对比（如"经济排名 vs 公共服务排名"）。

    Args:
        rank_a_col: 第一组排名（基准排名）。
        rank_b_col: 第二组排名（待比较排名）。
        label: 排名差异的标签前缀，默认 "rank_gap"。

    Examples:
        >>> rd = RankDisparity()
        >>> r = rd.analyze(df, rank_a_col="gdp_rank", rank_b_col="service_rank")
        >>> r.data.columns  # 含 "rank_gap" 列
    """

    name = "rank_disparity"
    category = "ranking"
    description = "排名差分析: 计算两组排名差异，识别排名偏差显著的个体"
    output_type = "dataframe"
    executor = "python"
    timeout = 10

    def analyze(
        self,
        data: pd.DataFrame,
        rank_a_col: str = "rank_a",
        rank_b_col: str = "rank_b",
        group_by: list[str] | None = None,
        label: str = "rank_gap",
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [rank_a_col, rank_b_col])

        df = data[[rank_a_col, rank_b_col] + (group_by or [])].copy()

        df[label] = df[rank_a_col] - df[rank_b_col]
        df[f"{label}_abs"] = df[label].abs()
        df[f"{label}_direction"] = df[label].apply(
            lambda v: "领先" if v > 0 else ("落后" if v < 0 else "持平")
        )

        df.sort_values(f"{label}_abs", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return AnalysisResult(
            success=True,
            data=df,
            summary={
                "algorithm": "rank_disparity",
                "rank_a": rank_a_col,
                "rank_b": rank_b_col,
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
    "Rank",
    "PercentRank",
    "NTile",
    "BenchmarkCompare",
    "RankDisparity",
]
