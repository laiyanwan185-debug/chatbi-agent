"""异常检测算法库。

提供以下算法：
  - ZScore       Z-score 标准差检测（基于正态假设）
  - IQR          IQR 箱线图检测（非参数、对异常值鲁棒）
  - ThreeSigma   3σ 原则检测（固定阈值版 Z-score）
  - YoYChange    同比突变检测（与同期对比的跳变探测）
  - DualFusion   双模并行融合（两种方法的交集/并集确认）
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from .base import AnalysisResult, BaseAnalyzer


# =============================================================================
# ZScore —— Z-score 标准差检测
# =============================================================================

class ZScore(BaseAnalyzer):
    """Z-score 标准差检测法。

    z = (x - mean) / std，|z| > threshold 判定为异常。

    Args:
        threshold: 阈值，默认 2.0（对应约 95% 置信区间）。
        direction: "both" 双侧异常，"upper" 仅上侧，"lower" 仅下侧。

    Examples:
        >>> zs = ZScore()
        >>> r = zs.analyze(df, value_col="gdp", group_by=["year"])
        >>> r.data[r.data["is_anomaly"]]
    """

    name = "zscore"
    category = "anomaly"
    description = "Z-score 检测: |z| > threshold 判定为异常"
    output_type = "dataframe"
    executor = "python"
    timeout = 10

    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str = "value",
        group_by: list[str] | None = None,
        threshold: float = 2.0,
        direction: Literal["both", "upper", "lower"] = "both",
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [value_col])

        df = data[[value_col] + (group_by or [])].copy()

        def _zscore(group: pd.DataFrame) -> pd.DataFrame:
            vals = group[value_col]
            mu, sigma = vals.mean(), vals.std(ddof=0)
            if sigma == 0 or pd.isna(sigma):
                group["z_score"] = 0.0
                group["is_anomaly"] = False
                return group

            z = (vals - mu) / sigma
            group["z_score"] = z
            if direction == "both":
                group["is_anomaly"] = z.abs() > threshold
            elif direction == "upper":
                group["is_anomaly"] = z > threshold
            else:
                group["is_anomaly"] = z < -threshold
            return group

        if group_by:
            df = df.groupby(group_by, group_keys=False).apply(_zscore, include_groups=False)
        else:
            df = _zscore(df)

        df.reset_index(drop=True, inplace=True)
        n_anomaly = df["is_anomaly"].sum()

        return AnalysisResult(
            success=True,
            data=df,
            summary={
                "algorithm": "zscore",
                "threshold": threshold,
                "direction": direction,
                "n_anomalies": int(n_anomaly),
            },
        )


# =============================================================================
# IQR —— IQR 箱线检测
# =============================================================================

class IQR(BaseAnalyzer):
    """IQR 箱线检测法（非参数异常检测）。

    异常判定标准：值 < Q1 - 1.5×IQR 或 值 > Q3 + 1.5×IQR。
    对数据分布无假设，不受极端值影响。

    Args:
        multiplier: IQR 倍数，默认 1.5（标准箱线图）。
        direction: "both" 双侧异常，"upper" 仅上侧，"lower" 仅下侧。

    Examples:
        >>> iqr = IQR()
        >>> r = iqr.analyze(df, value_col="pm25")
        >>> r.data[r.data["is_anomaly"]]
    """

    name = "iqr"
    category = "anomaly"
    description = "IQR 箱线检测: Q1-1.5*IQR 或 Q3+1.5*IQR 判定为异常"
    output_type = "dataframe"
    executor = "python"
    timeout = 10

    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str = "value",
        group_by: list[str] | None = None,
        multiplier: float = 1.5,
        direction: Literal["both", "upper", "lower"] = "both",
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [value_col])

        df = data[[value_col] + (group_by or [])].copy()

        def _iqr(group: pd.DataFrame) -> pd.DataFrame:
            vals = group[value_col]
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr_val = q3 - q1
            lower = q1 - multiplier * iqr_val
            upper = q3 + multiplier * iqr_val

            group["q1"] = q1
            group["q3"] = q3
            group["iqr"] = iqr_val
            group["lower_bound"] = lower
            group["upper_bound"] = upper

            if direction == "both":
                group["is_anomaly"] = (vals < lower) | (vals > upper)
            elif direction == "upper":
                group["is_anomaly"] = vals > upper
            else:
                group["is_anomaly"] = vals < lower
            return group

        if group_by:
            df = df.groupby(group_by, group_keys=False).apply(_iqr, include_groups=False)
        else:
            df = _iqr(df)

        df.reset_index(drop=True, inplace=True)
        n_anomaly = df["is_anomaly"].sum()

        return AnalysisResult(
            success=True,
            data=df,
            summary={
                "algorithm": "iqr",
                "multiplier": multiplier,
                "direction": direction,
                "n_anomalies": int(n_anomaly),
            },
        )


# =============================================================================
# ThreeSigma —— 3σ 原则
# =============================================================================

class ThreeSigma(BaseAnalyzer):
    """3σ 原则检测法。

    |x - mean| > 3*std 判定为异常。
    比 Z-score(threshold=2) 更严格，理论误报率约 0.3%。

    Args:
        sigma: σ 倍数，默认 3.0。
        direction: "both" 双侧异常，"upper" 仅上侧，"lower" 仅下侧。

    Examples:
        >>> ts = ThreeSigma()
        >>> r = ts.analyze(df, value_col="gdp_growth")
    """

    name = "three_sigma"
    category = "anomaly"
    description = "3σ 原则: |x - mean| > 3*std 判定为异常（严格模式）"
    output_type = "dataframe"
    executor = "python"
    timeout = 10

    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str = "value",
        group_by: list[str] | None = None,
        sigma: float = 3.0,
        direction: Literal["both", "upper", "lower"] = "both",
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [value_col])

        df = data[[value_col] + (group_by or [])].copy()

        def _three_sigma(group: pd.DataFrame) -> pd.DataFrame:
            vals = group[value_col]
            mu, std = vals.mean(), vals.std(ddof=0)
            if std == 0 or pd.isna(std):
                group["z_score"] = 0.0
                group["is_anomaly"] = False
                return group

            z = (vals - mu) / std
            group["z_score"] = z
            if direction == "both":
                group["is_anomaly"] = z.abs() > sigma
            elif direction == "upper":
                group["is_anomaly"] = z > sigma
            else:
                group["is_anomaly"] = z < -sigma
            return group

        if group_by:
            df = df.groupby(group_by, group_keys=False).apply(_three_sigma, include_groups=False)
        else:
            df = _three_sigma(df)

        df.reset_index(drop=True, inplace=True)
        n_anomaly = df["is_anomaly"].sum()

        return AnalysisResult(
            success=True,
            data=df,
            summary={
                "algorithm": "three_sigma",
                "sigma": sigma,
                "direction": direction,
                "n_anomalies": int(n_anomaly),
            },
        )


# =============================================================================
# YoYChange —— 同比突变检测
# =============================================================================

class YoYChange(BaseAnalyzer):
    """同比突变检测。

    计算与上一周期（同比/环比）的变化率，变化率超过阈值判定为突变。
    可检测连续趋势（连续 k 期变化同向则标记为持续恶化/改善）。

    Args:
        threshold: 变化率阈值（百分比），默认 20%。
        method: "yoy" 同比（同周期前一年）或 "qoq" 环比（前一期）。
        consecutive: 连续多少期同向变化视为持续趋势（默认 1，即单期突变）。
        direction: "both" 双向突变，"increase" 仅增长突变，"decrease" 仅下降突变。

    Examples:
        >>> yc = YoYChange()
        >>> r = yc.analyze(df, value_col="unemployment_rate",
        ...                time_col="year", group_by=["province"])
    """

    name = "yoy_change"
    category = "anomaly"
    description = "同比突变检测: 变化率超阈值 + 连续趋势判断"
    output_type = "dataframe"
    executor = "python"
    timeout = 15

    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str = "value",
        time_col: str = "year",
        group_by: list[str] | None = None,
        threshold: float = 20.0,
        method: Literal["yoy", "qoq"] = "yoy",
        consecutive: int = 1,
        direction: Literal["both", "increase", "decrease"] = "both",
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [value_col, time_col])

        df = data[[time_col, value_col] + (group_by or [])].copy()
        _sort_time(df, time_col, group_by)

        periods = 4 if method == "yoy" else 1

        if group_by:
            df["change_rate"] = df.groupby(group_by)[value_col].transform(
                lambda s: s.pct_change(periods=periods) * 100
            )
        else:
            df["change_rate"] = df[value_col].pct_change(periods=periods) * 100

        # 突变标记
        if direction == "both":
            df["is_sudden"] = df["change_rate"].abs() > threshold
        elif direction == "increase":
            df["is_sudden"] = df["change_rate"] > threshold
        else:
            df["is_sudden"] = df["change_rate"] < -threshold

        # 连续趋势判断
        if consecutive > 1 and group_by:
            def _consecutive_trend(group: pd.DataFrame) -> pd.Series:
                changes = group["change_rate"]
                same_sign = (changes > 0).astype(int)
                streak = same_sign.groupby(
                    (same_sign != same_sign.shift()).cumsum()
                ).cumsum() + 1
                streak[changes.isna() | (~changes.isna() & (changes == 0))] = 0
                return streak.where(same_sign == 1, -streak)

            df["trend_streak"] = df.groupby(group_by, group_keys=False).apply(
                _consecutive_trend, include_groups=False
            )
            abs_streak = df["trend_streak"].abs()
            df["is_trend"] = abs_streak >= consecutive
        else:
            df["is_trend"] = df["is_sudden"]

        n_sudden = df["is_sudden"].sum()
        n_trend = df["is_trend"].sum()

        return AnalysisResult(
            success=True,
            data=df,
            summary={
                "algorithm": "yoy_change",
                "threshold": threshold,
                "method": method,
                "consecutive": consecutive,
                "n_sudden": int(n_sudden),
                "n_trend": int(n_trend),
            },
        )


# =============================================================================
# DualFusion —— 双模并行异常融合
# =============================================================================

class DualFusion(BaseAnalyzer):
    """双模并行异常融合。

    同时使用两种异常检测方法，通过交集（AND）或并集（OR）确认异常。
    缺省使用 Z-score + IQR 双模并行，降低误报率。

    Args:
        mode: "intersection" 交集（默认，降低误报）或 "union" 并集（提高召回）。
        z_threshold: Z-score 阈值，默认 2.0。
        iqr_multiplier: IQR 倍数，默认 1.5。

    Examples:
        >>> df = DualFusion()
        >>> r = df.analyze(df, value_col="gdp_growth")
        >>> r.data[r.data["is_anomaly"]]  # 两种方法都标记的才是异常
    """

    name = "dual_fusion"
    category = "anomaly"
    description = "双模并行融合: Z-score + IQR 交集/并集确认，降低误报率"
    output_type = "dataframe"
    executor = "python"
    timeout = 15

    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str = "value",
        group_by: list[str] | None = None,
        mode: Literal["intersection", "union"] = "intersection",
        z_threshold: float = 2.0,
        iqr_multiplier: float = 1.5,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [value_col])

        # Z-score 检测
        zs = ZScore()
        z_result = zs.analyze(
            data, value_col=value_col, group_by=group_by,
            threshold=z_threshold,
        )

        # IQR 检测
        iqr = IQR()
        iqr_result = iqr.analyze(
            data, value_col=value_col, group_by=group_by,
            multiplier=iqr_multiplier,
        )

        # 融合
        df = z_result.data.copy()
        df["is_anomaly_z"] = df["is_anomaly"]
        df["is_anomaly_iqr"] = iqr_result.data["is_anomaly"]

        if mode == "intersection":
            df["is_anomaly"] = df["is_anomaly_z"] & df["is_anomaly_iqr"]
        else:
            df["is_anomaly"] = df["is_anomaly_z"] | df["is_anomaly_iqr"]

        df["fusion_method"] = mode

        anomalies = df[df["is_anomaly"]]
        n_anomaly = len(anomalies)

        return AnalysisResult(
            success=True,
            data=df,
            summary={
                "algorithm": "dual_fusion",
                "mode": mode,
                "z_threshold": z_threshold,
                "iqr_multiplier": iqr_multiplier,
                "n_anomalies_z": int(z_result.summary["n_anomalies"]),
                "n_anomalies_iqr": int(iqr_result.summary["n_anomalies"]),
                "n_anomalies_fused": n_anomaly,
            },
        )


# =============================================================================
# 内部工具函数
# =============================================================================

def _check_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"数据中缺少必需的列: {missing}")


def _sort_time(df: pd.DataFrame, time_col: str, group_by: list[str] | None) -> None:
    if group_by:
        df.sort_values([*group_by, time_col], inplace=True)
    else:
        df.sort_values(time_col, inplace=True)
    df.reset_index(drop=True, inplace=True)


# =============================================================================
# __all__
# =============================================================================

__all__ = [
    "ZScore",
    "IQR",
    "ThreeSigma",
    "YoYChange",
    "DualFusion",
]
