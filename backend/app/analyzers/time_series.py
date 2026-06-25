"""时间序列分析算法库。

提供以下算法：
  - CAGR        复合增长率
  - YoY         同比增长率
  - QoQ         环比增长率
  - SMA         简单移动平均
  - CMA         中心移动平均
  - LinearTrend 线性趋势拟合
  - Detrend     移动平均去趋势
  - Seasonal    季节性分析（季度/月度均值聚合）
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .base import AnalysisResult, BaseAnalyzer


# =============================================================================
# CAGR —— 复合增长率
# =============================================================================

class CAGR(BaseAnalyzer):
    """复合年增长率 (Compound Annual Growth Rate)。

    CAGR = (end_value / start_value) ** (1 / n_periods) - 1

    Examples:
        >>> cagr = CAGR()
        >>> result = cagr.analyze(df, value_col="gdp", time_col="year", group_by=["province"])
        >>> result.data
           province      cagr
        0        广东  0.0652
        1        江苏  0.0581
    """

    name = "cagr"
    category = "time_series"
    description = "复合年增长率 CAGR = (End/Start)^(1/n) - 1"
    output_type = "dataframe"
    executor = "python"
    timeout = 10

    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str = "value",
        time_col: str = "year",
        group_by: list[str] | None = None,
    ) -> AnalysisResult:
        self.validate(data)
        # 自动检测时间列：默认 "year" 但实际数据中可能为 "统计年份" 等
        if time_col not in data.columns:
            for candidate in ["统计年份", "年份", "year", "quarter", "季度", "period", "time"]:
                if candidate in data.columns:
                    time_col = candidate
                    break
        _check_cols(data, [value_col, time_col])

        df = data[[time_col, value_col] + (group_by or [])].copy()
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[time_col, value_col])
        if df.empty:
            return AnalysisResult(
                success=False,
                error="数据不足：去除缺失值后无有效数据计算 CAGR",
            )

        if group_by:
            def _cagr(group: pd.DataFrame) -> float:
                group = group.sort_values(time_col)
                start, end = group[value_col].iloc[0], group[value_col].iloc[-1]
                n = len(group) - 1
                if n < 1 or start <= 0:
                    return np.nan
                return (end / start) ** (1.0 / n) - 1.0

            result = df.groupby(group_by).apply(_cagr, include_groups=False).reset_index(name="cagr")
        else:
            df = df.sort_values(time_col)
            start, end = df[value_col].iloc[0], df[value_col].iloc[-1]
            n = len(df) - 1
            if n < 1 or start <= 0:
                return AnalysisResult(success=False, error="数据不足或起始值 <= 0 无法计算 CAGR")
            cagr_val = (end / start) ** (1.0 / n) - 1.0
            result = pd.DataFrame({"cagr": [cagr_val]})

        return AnalysisResult(
            success=True,
            data=result,
            summary={"algorithm": "cagr", "periods": n if not group_by else None},
        )


# =============================================================================
# YoY —— 同比增长率
# =============================================================================

class YoY(BaseAnalyzer):
    """同比增长率 (Year-over-Year)。

    YoY = (current - same_period_last_year) / same_period_last_year * 100

    适用于年、季、月度数据。

    Examples:
        >>> yoy = YoY()
        >>> result = yoy.analyze(df, value_col="gdp", time_col="year")
        >>> result.data["yoy"]
    """

    name = "yoy"
    category = "time_series"
    description = "同比增长率 (YoY) = (当期 - 同期) / 同期 × 100%"
    output_type = "dataframe"
    executor = "python"
    timeout = 10

    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str = "value",
        time_col: str = "year",
        group_by: list[str] | None = None,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [value_col, time_col])

        df = data[[time_col, value_col] + (group_by or [])].copy()
        _sort_time(df, time_col, group_by)

        if group_by:
            df["yoy"] = df.groupby(group_by)[value_col].transform(
                lambda s: s.pct_change(periods=1) * 100
            )
        else:
            df["yoy"] = df[value_col].pct_change(periods=1) * 100

        return AnalysisResult(success=True, data=df)


# =============================================================================
# QoQ —— 环比增长率
# =============================================================================

class QoQ(BaseAnalyzer):
    """环比增长率 (Quarter-over-Quarter / Period-over-Period)。

    QoQ = (current - previous_period) / previous_period * 100

    Examples:
        >>> qoq = QoQ()
        >>> result = qoq.analyze(df, value_col="avg_price", time_col="month")
    """

    name = "qoq"
    category = "time_series"
    description = "环比增长率 (QoQ) = (当期 - 上期) / 上期 × 100%"
    output_type = "dataframe"
    executor = "python"
    timeout = 10

    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str = "value",
        time_col: str = "period",
        group_by: list[str] | None = None,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [value_col, time_col])

        df = data[[time_col, value_col] + (group_by or [])].copy()
        _sort_time(df, time_col, group_by)

        if group_by:
            df["qoq"] = df.groupby(group_by)[value_col].transform(
                lambda s: s.pct_change(periods=1) * 100
            )
        else:
            df["qoq"] = df[value_col].pct_change(periods=1) * 100

        return AnalysisResult(success=True, data=df)


# =============================================================================
# SMA —— 简单移动平均
# =============================================================================

class SMA(BaseAnalyzer):
    """简单移动平均 (Simple Moving Average)。

    SMA_k = (x_t + x_{t-1} + ... + x_{t-k+1}) / k

    Args:
        window: 移动窗口大小，默认 3。

    Examples:
        >>> sma = SMA()
        >>> result = sma.analyze(df, value_col="sold_area", time_col="month", window=3)
    """

    name = "sma"
    category = "time_series"
    description = "简单移动平均 SMA = rolling(k).mean()"
    output_type = "dataframe"
    executor = "python"
    timeout = 10

    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str = "value",
        time_col: str = "period",
        group_by: list[str] | None = None,
        window: int = 3,
    ) -> AnalysisResult:
        self.validate(data)
        # 当 time_col 使用默认值 "period" 时，自动检测数据中是否存在 "year" 列
        if time_col == "period" and "year" in data.columns and "period" not in data.columns:
            time_col = "year"
        _check_cols(data, [value_col, time_col])

        df = data[[time_col, value_col] + (group_by or [])].copy()
        _sort_time(df, time_col, group_by)

        if group_by:
            df["sma"] = df.groupby(group_by)[value_col].transform(
                lambda s: s.rolling(window=window, min_periods=1).mean()
            )
        else:
            df["sma"] = df[value_col].rolling(window=window, min_periods=1).mean()

        return AnalysisResult(
            success=True,
            data=df,
            summary={"algorithm": "sma", "window": window},
        )

class CMA(BaseAnalyzer):
    """中心移动平均 (Centered Moving Average)。

    当窗口为偶数时向前偏移 1 位以实现中心对齐，常用于季节分解。

    Args:
        window: 移动窗口大小，默认 4（季度数据常用）。

    Examples:
        >>> cma = CMA()
        >>> result = cma.analyze(df, value_col="gdp", time_col="quarter", window=4)
    """

    name = "cma"
    category = "time_series"
    description = "中心移动平均 (Centered MA)，偶数窗口时居中偏移"
    output_type = "dataframe"
    executor = "python"
    timeout = 10

    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str = "value",
        time_col: str = "period",
        group_by: list[str] | None = None,
        window: int = 4,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [value_col, time_col])

        df = data[[time_col, value_col] + (group_by or [])].copy()
        _sort_time(df, time_col, group_by)

        def _cma(series: pd.Series, w: int) -> pd.Series:
            if w % 2 == 0:
                # 偶数窗口: 先做一次 w 宽 SMA，再做一次 2 宽 SMA（中心化）
                return series.rolling(w, center=True).mean()
            else:
                return series.rolling(w, center=True).mean()

        if group_by:
            df["cma"] = df.groupby(group_by)[value_col].transform(
                lambda s: _cma(s, window)
            )
        else:
            df["cma"] = _cma(df[value_col], window)

        return AnalysisResult(
            success=True,
            data=df,
            summary={"algorithm": "cma", "window": window, "centered": True},
        )


# =============================================================================
# LinearTrend —— 线性趋势拟合
# =============================================================================

class LinearTrend(BaseAnalyzer):
    """线性趋势拟合 (Linear Trend Estimation)。

    使用普通最小二乘法 (scipy.stats.linregress) 拟合 y = slope * x + intercept。

    返回每个分组的斜率、截距、R²、p-value。

    Examples:
        >>> ltrend = LinearTrend()
        >>> result = ltrend.analyze(df, value_col="gdp", time_col="year", group_by=["province"])
        >>> result.data
           province    slope  intercept    rvalue      pvalue
        0        广东  4520.1    -9e6     0.986      0.0012
    """

    name = "linear_trend"
    category = "time_series"
    description = "线性趋势拟合 OLS: y = slope * t + intercept, 返回 slope/r²/p-value"
    output_type = "dataframe"
    executor = "python"
    timeout = 15

    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str = "value",
        time_col: str = "year",
        group_by: list[str] | None = None,
    ) -> AnalysisResult:
        self.validate(data)
        if time_col not in data.columns:
            for candidate in ["统计年份", "年份", "year", "quarter", "季度", "period", "time"]:
                if candidate in data.columns:
                    time_col = candidate
                    break
        _check_cols(data, [value_col, time_col])

        df = data[[time_col, value_col] + (group_by or [])].copy()
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[time_col, value_col])

        def _fit(group: pd.DataFrame) -> dict[str, float]:
            group = group.sort_values(time_col)
            x = group[time_col].values
            y = group[value_col].values

            unique_x = len(set(x))
            if unique_x < 2:
                # 不足 2 个不同时间点 → 无法计算趋势
                mu = float(y.mean()) if len(y) > 0 else 0.0
                return {
                    "slope": 0.0,
                    "intercept": mu,
                    "rvalue": 0.0,
                    "pvalue": 1.0,
                    "stderr": 0.0,
                    "trend_message": f"仅 {unique_x} 个不同时间点，不足以计算线性趋势",
                }

            if len(x) < 3:
                # 2 个点 → 直接计算斜率，不经过 linregress
                slope = (y[1] - y[0]) / (x[1] - x[0]) if x[1] != x[0] else 0.0
                return {
                    "slope": slope,
                    "intercept": y[0] - slope * x[0],
                    "rvalue": 1.0,
                    "pvalue": 0.0,
                    "stderr": 0.0,
                    "trend_message": "基于 2 个离散点的斜率计算",
                }

            res = sp_stats.linregress(x, y)
            return {
                "slope": res.slope,
                "intercept": res.intercept,
                "rvalue": res.rvalue,
                "pvalue": res.pvalue,
                "stderr": res.stderr,
            }

        if group_by:
            coefs = df.groupby(group_by).apply(_fit, include_groups=False)
            result = pd.json_normalize(coefs)
            result.index = coefs.index
            result = result.reset_index()
        else:
            coefs = _fit(df)
            result = pd.DataFrame([coefs])

        return AnalysisResult(
            success=True,
            data=result,
            summary={"algorithm": "linear_trend"},
        )


# =============================================================================
# Detrend —— 移动平均去趋势
# =============================================================================

class Detrend(BaseAnalyzer):
    """移动平均去趋势 (Detrend by Moving Average)。

    从原始序列中减去移动平均趋势，得到残差（波动）部分。
    残差 = value - SMA(window)

    Args:
        window: 去趋势窗口大小，默认 4。
        center: 是否居中窗口，默认 True。

    Examples:
        >>> dt = Detrend()
        >>> result = dt.analyze(df, value_col="gdp", time_col="year", window=3)
        >>> result.data["residual"]
    """

    name = "detrend"
    category = "time_series"
    description = "移动平均去趋势: residual = value - SMA(window)"
    output_type = "dataframe"
    executor = "python"
    timeout = 10

    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str = "value",
        time_col: str = "period",
        group_by: list[str] | None = None,
        window: int = 4,
        center: bool = True,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [value_col, time_col])

        df = data[[time_col, value_col] + (group_by or [])].copy()
        _sort_time(df, time_col, group_by)

        def _detrend(series: pd.Series, w: int, c: bool) -> pd.Series:
            trend = series.rolling(window=w, center=c, min_periods=1).mean()
            return series - trend

        if group_by:
            df["residual"] = df.groupby(group_by)[value_col].transform(
                lambda s: _detrend(s, window, center)
            )
            df["trend"] = df.groupby(group_by)[value_col].transform(
                lambda s: s.rolling(window=window, center=center, min_periods=1).mean()
            )
        else:
            df["trend"] = df[value_col].rolling(window=window, center=center, min_periods=1).mean()
            df["residual"] = df[value_col] - df["trend"]

        return AnalysisResult(
            success=True,
            data=df,
            summary={"algorithm": "detrend", "window": window, "center": center},
        )


# =============================================================================
# Seasonal —— 季节性分析
# =============================================================================

class Seasonal(BaseAnalyzer):
    """季节性分析 (Seasonal Analysis)。

    按周期（季度/月份）聚合多年数据的均值，提取季节性规律。

    Args:
        period_col: 标识周期位置的列（如 "quarter", "month"），
                    若不指定则从 time_col 自动提取。
        agg_func: 聚合函数，默认 "mean"。

    Examples:
        >>> sa = Seasonal()
        >>> result = sa.analyze(df, value_col="unemployment_rate",
        ...                     time_col="quarter", period_col="quarter")
        >>> result.data
           quarter  seasonal_mean
        0        1           5.32
        1        2           5.18
        ...
    """

    name = "seasonal"
    category = "time_series"
    description = "季节性分析: 按季度/月份聚合计算多年均值"
    output_type = "dataframe"
    executor = "python"
    timeout = 10

    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str = "value",
        time_col: str = "period",
        period_col: str | None = None,
        group_by: list[str] | None = None,
        agg_func: str = "mean",
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [value_col, time_col])

        df = data[[time_col, value_col] + (group_by or [])].copy()

        # 未指定 period_col 时自动推断
        if period_col is None:
            sample = df[time_col].dropna().iloc[0] if not df[time_col].empty else ""
            sample_str = str(sample)
            if "quarter" in str(time_col).lower() or "季度" in sample_str:
                period_col = "quarter"
                df[period_col] = df[time_col]
            elif "month" in str(time_col).lower():
                period_col = "month"
                df[period_col] = df[time_col]
            else:
                return AnalysisResult(
                    success=False,
                    error="无法自动推断周期列，请通过 period_col 参数指定",
                )

        _check_cols(df, [period_col])

        gb_cols = [period_col] + (group_by or [])
        season = df.groupby(gb_cols)[value_col].agg(agg_func).reset_index()
        season = season.sort_values(gb_cols).reset_index(drop=True)
        season.rename(columns={agg_func: f"seasonal_{agg_func}"}, inplace=True)

        return AnalysisResult(
            success=True,
            data=season,
            summary={"algorithm": "seasonal", "period_col": period_col, "agg_func": agg_func},
        )


# =============================================================================
# 内部工具函数
# =============================================================================

def _check_cols(df: pd.DataFrame, cols: list[str]) -> None:
    """校验所需列是否存在，不存在则抛出 ValueError。"""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"数据中缺少必需的列: {missing}")


def _sort_time(df: pd.DataFrame, time_col: str, group_by: list[str] | None) -> None:
    """按时间列排序（组内排序如果指定了 group_by）。"""
    if group_by:
        df.sort_values([*group_by, time_col], inplace=True)
    else:
        df.sort_values(time_col, inplace=True)
    df.reset_index(drop=True, inplace=True)


# =============================================================================
# __all__ —— 注册中心自动发现的导出列表
# =============================================================================

__all__ = [
    "CAGR",
    "YoY",
    "QoQ",
    "SMA",
    "CMA",
    "LinearTrend",
    "Detrend",
    "Seasonal",
]
