"""相关性分析算法库。

提供以下算法：
  - Pearson          Pearson 相关系数（线性相关）
  - Spearman         Spearman 秩相关系数（单调相关）
  - TLCC             时滞交叉相关（滞后效应探测）
  - MutualInfo       互信息（非线性关联探测）
  - DetrendResidual  去趋势残差相关（剔除混杂因素后相关性）
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.feature_selection import mutual_info_regression

from .base import AnalysisResult, BaseAnalyzer


# =============================================================================
# Pearson —— 皮尔逊相关系数
# =============================================================================

class Pearson(BaseAnalyzer):
    """Pearson 相关系数，衡量两个连续变量之间的线性相关程度。

    值域 [-1, 1]：+1 完全正相关，-1 完全负相关，0 无线性相关。

    Examples:
        >>> p = Pearson()
        >>> r = p.analyze(df, x_col="gdp", y_col="edu_spending", group_by=["province"])
        >>> r.data
           province  pearson_r  p_value
        0        广东     0.985   0.0012
    """

    name = "pearson"
    category = "correlation"
    description = "Pearson 相关系数: 衡量两个变量的线性相关程度"
    output_type = "dataframe"
    executor = "python"
    timeout = 10

    def analyze(
        self,
        data: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        group_by: list[str] | None = None,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [x_col, y_col])

        df = data[[x_col, y_col] + (group_by or [])].copy()
        _drop_nan(df, [x_col, y_col])

        def _corr(group: pd.DataFrame) -> dict[str, float]:
            x, y = group[x_col].values, group[y_col].values
            if len(x) < 3:
                return {"pearson_r": np.nan, "p_value": np.nan}
            r, p = sp_stats.pearsonr(x, y)
            return {"pearson_r": r, "p_value": p}

        if group_by:
            result = df.groupby(group_by).apply(_corr, include_groups=False)
            out = pd.json_normalize(result)
            out.index = result.index
            out = out.reset_index()
        else:
            out = pd.DataFrame([_corr(df)])

        return AnalysisResult(success=True, data=out)


# =============================================================================
# Spearman —— 斯皮尔曼秩相关系数
# =============================================================================

class Spearman(BaseAnalyzer):
    """Spearman 秩相关系数，衡量两个变量的单调相关程度（不要求线性假设）。

    对异常值不敏感，适合非线性单调关系检测。

    Examples:
        >>> s = Spearman()
        >>> r = s.analyze(df, x_col="urbanization", y_col="service_ratio")
    """

    name = "spearman"
    category = "correlation"
    description = "Spearman 秩相关系数: 衡量两个变量的单调相关程度"
    output_type = "dataframe"
    executor = "python"
    timeout = 10

    def analyze(
        self,
        data: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        group_by: list[str] | None = None,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [x_col, y_col])

        df = data[[x_col, y_col] + (group_by or [])].copy()
        _drop_nan(df, [x_col, y_col])

        def _corr(group: pd.DataFrame) -> dict[str, float]:
            x, y = group[x_col].values, group[y_col].values
            if len(x) < 3:
                return {"spearman_r": np.nan, "p_value": np.nan}
            r, p = sp_stats.spearmanr(x, y)
            return {"spearman_r": r, "p_value": p}

        if group_by:
            result = df.groupby(group_by).apply(_corr, include_groups=False)
            out = pd.json_normalize(result)
            out.index = result.index
            out = out.reset_index()
        else:
            out = pd.DataFrame([_corr(df)])

        return AnalysisResult(success=True, data=out)


# =============================================================================
# TLCC —— 时滞交叉相关
# =============================================================================

class TLCC(BaseAnalyzer):
    """时滞交叉相关 (Time-Lagged Cross Correlation)。

    计算 x 在滞后 -max_lag 到 +max_lag 范围内与 y 的 Pearson 相关，
    找出最大相关量级以及对应的滞后阶数。

    正滞后表示 x 领先于 y（x 的过去预测 y 的现在），
    负滞后表示 x 落后于 y。

    Args:
        max_lag: 最大滞后阶数，默认 4。

    Examples:
        >>> tlcc = TLCC()
        >>> r = tlcc.analyze(df, x_col="edu_spending", y_col="gdp",
        ...                  time_col="year", group_by=["province"])
        >>> r.data["best_lag"]  # 最优滞后阶数
    """

    name = "tlcc"
    category = "correlation"
    description = "时滞交叉相关 (TLCC): 探测 x 领先/滞后 y 的最优滞后阶数"
    output_type = "dataframe"
    executor = "python"
    timeout = 15

    def analyze(
        self,
        data: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        time_col: str = "time",
        group_by: list[str] | None = None,
        max_lag: int = 4,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [x_col, y_col, time_col])

        df = data[[time_col, x_col, y_col] + (group_by or [])].copy()
        _drop_nan(df, [x_col, y_col])

        def _tlcc(group: pd.DataFrame) -> dict[str, Any]:
            g = group.sort_values(time_col)
            x, y = g[x_col].values, g[y_col].values
            n = len(x)
            if n < max_lag + 3:
                return {"best_lag": np.nan, "best_r": np.nan, "best_p": np.nan}

            lags = list(range(-max_lag, max_lag + 1))
            r_vals = []
            for lag in lags:
                if lag < 0:
                    xx, yy = x[:lag], y[-lag:]
                elif lag > 0:
                    xx, yy = x[lag:], y[:-lag]
                else:
                    xx, yy = x, y
                if len(xx) < 3:
                    r_vals.append(np.nan)
                else:
                    r, _ = sp_stats.pearsonr(xx, yy)
                    r_vals.append(r)

            r_vals = np.array(r_vals, dtype=float)
            valid = ~np.isnan(r_vals)
            if not valid.any():
                return {"best_lag": np.nan, "best_r": np.nan, "best_p": np.nan}

            best_idx = np.argmax(np.abs(r_vals[valid]))
            best_lag = int(np.array(lags)[valid][best_idx])
            best_r = float(r_vals[valid][best_idx])

            return {
                "best_lag": best_lag,
                "best_r": best_r,
                "best_p": float(np.nan),
            }

        if group_by:
            result = df.groupby(group_by).apply(_tlcc, include_groups=False)
            out = pd.json_normalize(result)
            out.index = result.index
            out = out.reset_index()
        else:
            out = pd.DataFrame([_tlcc(df)])

        return AnalysisResult(success=True, data=out)


# =============================================================================
# MutualInfo —— 互信息
# =============================================================================

class MutualInfo(BaseAnalyzer):
    """互信息 (Mutual Information)，衡量两个变量间的非线性关联。

    MI 为 0 表示独立，值越大表示关联越强。可探测 Pearson 无法捕捉的
    非线性模式（如 U 形、周期关系）。

    Args:
        n_neighbors: 近邻数量（控制估计平滑度），默认 3。

    Examples:
        >>> mi = MutualInfo()
        >>> r = mi.analyze(df, x_col="gdp_growth", y_col="pm25",
        ...                group_by=["province"])
    """

    name = "mutual_info"
    category = "correlation"
    description = "互信息 (MI): 探测非线性关联，可捕捉 Pearson 无法发现的模式"
    output_type = "dataframe"
    executor = "python"
    timeout = 15

    def analyze(
        self,
        data: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        group_by: list[str] | None = None,
        n_neighbors: int = 3,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [x_col, y_col])

        df = data[[x_col, y_col] + (group_by or [])].copy()
        _drop_nan(df, [x_col, y_col])

        def _mi(group: pd.DataFrame) -> dict[str, float]:
            x = group[x_col].values.reshape(-1, 1)
            y = group[y_col].values
            if len(x) < 5:
                return {"mutual_info": np.nan}
            mi_val = mutual_info_regression(x, y, n_neighbors=n_neighbors,
                                            random_state=42)[0]
            return {"mutual_info": float(mi_val)}

        if group_by:
            result = df.groupby(group_by).apply(_mi, include_groups=False)
            out = pd.json_normalize(result)
            out.index = result.index
            out = out.reset_index()
        else:
            out = pd.DataFrame([_mi(df)])

        return AnalysisResult(success=True, data=out)


# =============================================================================
# DetrendResidual —— 去趋势残差相关
# =============================================================================

class DetrendResidual(BaseAnalyzer):
    """去趋势残差相关分析。

    先分别对 x 和 y 做线性去趋势（剔除时间趋势的混杂影响），
    再计算残差之间的 Pearson / Spearman 相关。

    常用于判断两个变量在排除共同时间趋势后是否仍然存在关联
    （如 L4-L4-20：排除 GDP 影响后看老龄化与医疗支出的关系）。

    Args:
        method: 去趋势方法，可选 "linear" 或 "ma"（移动平均）。
        ma_window: 移动平均去趋势窗口（method="ma" 时生效），默认 4。

    Examples:
        >>> dr = DetrendResidual()
        >>> r = dr.analyze(df, x_col="aging_rate", y_col="med_spending",
        ...                time_col="year")
    """

    name = "detrend_residual"
    category = "correlation"
    description = "去趋势残差相关: 消除共同时间趋势后分析变量间真实关联"
    output_type = "dataframe"
    executor = "python"
    timeout = 15

    def analyze(
        self,
        data: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        time_col: str = "time",
        group_by: list[str] | None = None,
        control_cols: list[str] | None = None,
        method: str = "linear",
        ma_window: int = 4,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [x_col, y_col, time_col])

        df = data[[time_col, x_col, y_col] + (group_by or []) + (control_cols or [])].copy()
        _drop_nan(df, [x_col, y_col])

        def _residual_corr(group: pd.DataFrame) -> dict[str, float]:
            g = group.sort_values(time_col)
            time_vals = np.arange(len(g))

            if method == "linear":
                x_res = _linear_residual(g[x_col].values, time_vals)
                y_res = _linear_residual(g[y_col].values, time_vals)
            elif method == "ma":
                x_series = g[x_col]
                y_series = g[y_col]
                x_trend = x_series.rolling(ma_window, center=True, min_periods=1).mean()
                y_trend = y_series.rolling(ma_window, center=True, min_periods=1).mean()
                x_res = (x_series - x_trend).dropna().values
                y_res = (y_series - y_trend).dropna().values
                min_len = min(len(x_res), len(y_res))
                x_res, y_res = x_res[:min_len], y_res[:min_len]
            else:
                raise ValueError(f"未知去趋势方法: {method}")

            if control_cols:
                for ctrl in control_cols:
                    if ctrl in g.columns:
                        ctrl_vals = g[ctrl].values
                        x_res = _linear_residual(x_res, ctrl_vals)
                        y_res = _linear_residual(y_res, ctrl_vals)

            if len(x_res) < 3:
                return {"pearson_r": np.nan, "p_value_pearson": np.nan,
                        "spearman_r": np.nan, "p_value_spearman": np.nan}

            pr, pp = sp_stats.pearsonr(x_res, y_res)
            sr, sp_val = sp_stats.spearmanr(x_res, y_res)
            return {
                "pearson_r": pr,
                "p_value_pearson": pp,
                "spearman_r": sr,
                "p_value_spearman": sp_val,
            }

        if group_by:
            result = df.groupby(group_by).apply(_residual_corr, include_groups=False)
            out = pd.json_normalize(result)
            out.index = result.index
            out = out.reset_index()
        else:
            out = pd.DataFrame([_residual_corr(df)])

        return AnalysisResult(
            success=True,
            data=out,
            summary={"algorithm": "detrend_residual", "method": method},
        )


# =============================================================================
# 内部工具函数
# =============================================================================

def _check_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"数据中缺少必需的列: {missing}")


def _drop_nan(df: pd.DataFrame, cols: list[str]) -> None:
    df.dropna(subset=cols, inplace=True)
    df.reset_index(drop=True, inplace=True)


def _linear_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """对 y 做关于 x 的线性回归，返回残差。"""
    mask = ~(np.isnan(y) | np.isnan(x))
    y_clean, x_clean = y[mask], x[mask]
    if len(y_clean) < 3:
        return y - np.nanmean(y)
    slope, intercept, *_ = sp_stats.linregress(x_clean, y_clean)
    return y_clean - (slope * x_clean + intercept)


# =============================================================================
# __all__
# =============================================================================

__all__ = [
    "Pearson",
    "Spearman",
    "TLCC",
    "MutualInfo",
    "DetrendResidual",
]
