"""综合评价分析算法库。

提供以下算法：
  - MinMax          Min-Max 归一化/标准化（含正反向处理）
  - EntropyWeight   熵权法（数据自动定权重）
  - TOPSIS          优劣解距离法（基于熵权或自定义权重）
  - PCA             主成分分析（降维 + 载荷解释）
  - TwoLevelRank    两级综合排名（领域百分位 → 加权聚合）
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA as SklearnPCA

from .base import AnalysisResult, BaseAnalyzer


# =============================================================================
# MinMax —— Min-Max 归一化
# =============================================================================

class MinMax(BaseAnalyzer):
    """Min-Max 归一化，将指标缩放到指定区间。

    x_norm = (x - min) / (max - min) * (new_max - new_max) + new_min

    负向指标（越低越好）在归一化后做反向处理。

    Args:
        feature_range: 目标区间，默认 (0, 1)。
        reverse_cols: 负向指标列名（值越低越好），归一化后取 1 - x_norm。
        clip: 是否裁剪到 feature_range 内，默认 True。

    Examples:
        >>> mm = MinMax()
        >>> r = mm.analyze(df, value_cols=["gdp", "unemployment"],
        ...                reverse_cols=["unemployment"])
    """

    name = "minmax"
    category = "composite"
    description = "Min-Max 归一化: 缩放到 [0,1] 区间，支持正反向指标"
    output_type = "dataframe"
    executor = "python"
    timeout = 10

    def analyze(
        self,
        data: pd.DataFrame,
        value_cols: list[str] | None = None,
        group_by: list[str] | None = None,
        feature_range: tuple[float, float] = (0, 1),
        reverse_cols: list[str] | None = None,
        clip: bool = True,
    ) -> AnalysisResult:
        self.validate(data)
        cols = value_cols or list(data.select_dtypes(include=[np.number]).columns)
        _check_cols(data, cols)

        df = data[cols + (group_by or [])].copy()
        reverse_cols = set(reverse_cols or [])
        lo, hi = feature_range

        def _normalize(group: pd.DataFrame) -> pd.DataFrame:
            for col in cols:
                vals = group[col]
                vmin, vmax = vals.min(), vals.max()
                if vmax > vmin:
                    norm = (vals - vmin) / (vmax - vmin) * (hi - lo) + lo
                else:
                    norm = pd.Series(lo, index=vals.index)

                if col in reverse_cols:
                    norm = hi + lo - norm  # 反向

                if clip:
                    norm = norm.clip(lo, hi)

                group[f"{col}_norm"] = norm
            return group

        if group_by:
            df = df.groupby(group_by, group_keys=False).apply(_normalize, include_groups=False)
        else:
            df = _normalize(df)

        df.reset_index(drop=True, inplace=True)

        return AnalysisResult(
            success=True,
            data=df,
            summary={
                "algorithm": "minmax",
                "feature_range": feature_range,
                "reverse_cols": list(reverse_cols),
            },
        )


# =============================================================================
# EntropyWeight —— 熵权法
# =============================================================================

class EntropyWeight(BaseAnalyzer):
    """熵权法：基于数据本身的信息熵自动确定指标权重。

    步骤：
      1. 归一化决策矩阵到 [0, 1]
      2. 计算各指标的信息熵 e_j = -k * sum(p_ij * ln(p_ij))
      3. 差异系数 d_j = 1 - e_j
      4. 权重 w_j = d_j / sum(d_j)

    Args:
        reverse_cols: 负向指标列名，归一化前做反向处理。

    Examples:
        >>> ew = EntropyWeight()
        >>> r = ew.analyze(df, value_cols=["gdp", "edu_spending", "unemployment"],
        ...                reverse_cols=["unemployment"])
        >>> r.data  # 各指标权重
           indicator    weight  entropy
        0        gdp  0.352    0.981
        1 edu_spending  0.315    0.983
        2 unemployment  0.333    0.982
    """

    name = "entropy_weight"
    category = "composite"
    description = "熵权法: 基于数据信息熵自动计算指标权重"
    output_type = "dataframe"
    executor = "python"
    timeout = 15

    def analyze(
        self,
        data: pd.DataFrame,
        value_cols: list[str] | None = None,
        group_by: list[str] | None = None,
        reverse_cols: list[str] | None = None,
    ) -> AnalysisResult:
        self.validate(data)
        cols = value_cols or list(data.select_dtypes(include=[np.number]).columns)
        if len(cols) < 2:
            raise ValueError("熵权法至少需要 2 个指标列")
        _check_cols(data, cols)

        df = data[cols + (group_by or [])].copy()
        reverse_cols = set(reverse_cols or [])

        def _entropy(group: pd.DataFrame) -> dict[str, Any]:
            matrix = group[cols].values.astype(float)

            # 1. 归一化到 [0, 1]（含反向）
            for i, col in enumerate(cols):
                col_vals = matrix[:, i]
                vmin, vmax = col_vals.min(), col_vals.max()
                if vmax > vmin:
                    col_norm = (col_vals - vmin) / (vmax - vmin)
                else:
                    col_norm = np.ones_like(col_vals) * 0.5
                if col in reverse_cols:
                    col_norm = 1 - col_norm
                # 避免 ln(0)
                col_norm = np.clip(col_norm, 1e-10, 1 - 1e-10)
                matrix[:, i] = col_norm

            # 2. 概率化：p_ij = x_ij / sum(x_ij)
            col_sums = matrix.sum(axis=0, keepdims=True)
            col_sums = np.where(col_sums == 0, 1, col_sums)
            p = matrix / col_sums

            # 3. 熵值 e_j = -k * sum(p_ij * ln(p_ij))
            n = p.shape[0]
            k = 1.0 / np.log(n) if n > 1 else 1.0
            entropy = -k * np.sum(p * np.log(p), axis=0)
            entropy = np.clip(entropy, 0, 1)

            # 4. 差异系数 & 权重
            d = 1 - entropy
            w = d / d.sum() if d.sum() > 0 else np.ones_like(d) / len(d)

            return {
                "entropy": entropy.tolist(),
                "diversity": d.tolist(),
                "weight": w.tolist(),
            }

        if group_by:
            result = df.groupby(group_by).apply(_entropy, include_groups=False)
            all_rows = []
            for idx in result.index:
                row = {g: idx[i] for i, g in enumerate(group_by)} if isinstance(idx, tuple) else {group_by[0]: idx}
                for i, col in enumerate(cols):
                    row.update({
                        "indicator": col,
                        "entropy": result[idx]["entropy"][i],
                        "diversity": result[idx]["diversity"][i],
                        "weight": result[idx]["weight"][i],
                    })
                    all_rows.append(row.copy())
            out = pd.DataFrame(all_rows)
        else:
            weights = _entropy(df)
            out = pd.DataFrame({
                "indicator": cols,
                "entropy": weights["entropy"],
                "diversity": weights["diversity"],
                "weight": weights["weight"],
            })

        return AnalysisResult(
            success=True,
            data=out,
            summary={
                "algorithm": "entropy_weight",
                "n_indicators": len(cols),
            },
        )


# =============================================================================
# TOPSIS —— 优劣解距离法
# =============================================================================

class TOPSIS(BaseAnalyzer):
    """TOPSIS 综合评价法。

    步骤：
      1. 归一化决策矩阵（向量归一化）
      2. 加权（使用熵权法或自定义权重）
      3. 确定正负理想解
      4. 计算到正负理想解的距离（欧氏距离）
      5. 计算相对贴近度 C = D- / (D+ + D-)

    Args:
        weights: 各指标权重数组。若为 None 则使用熵权法自动计算。
        reverse_cols: 负向指标列名。

    Examples:
        >>> topsis = TOPSIS()
        >>> r = topsis.analyze(df, value_cols=["gdp", "edu", "unemployment"],
        ...                    reverse_cols=["unemployment"])
        >>> r.data  # 含相对贴近度 C
    """

    name = "topsis"
    category = "composite"
    description = "TOPSIS: 优劣解距离法，计算各方案与理想解的贴近度"
    output_type = "dataframe"
    executor = "python"
    timeout = 15

    def analyze(
        self,
        data: pd.DataFrame,
        value_cols: list[str] | None = None,
        group_by: list[str] | None = None,
        weights: list[float] | None = None,
        reverse_cols: list[str] | None = None,
    ) -> AnalysisResult:
        self.validate(data)
        cols = value_cols or list(data.select_dtypes(include=[np.number]).columns)
        if len(cols) < 2:
            raise ValueError("TOPSIS 至少需要 2 个指标列")
        _check_cols(data, cols)

        df = data[cols + (group_by or [])].copy()
        id_vars = [c for c in df.columns if c not in cols]
        reverse_cols = set(reverse_cols or [])

        def _topsis(group: pd.DataFrame) -> list[dict[str, Any]]:
            matrix = group[cols].values.astype(float)

            # 1. 向量归一化
            norm = np.sqrt((matrix ** 2).sum(axis=0, keepdims=True))
            norm = np.where(norm == 0, 1, norm)
            matrix_norm = matrix / norm

            # 2. 确定权重
            w = np.array(weights) if weights else _auto_weights(matrix, reverse_cols, cols)
            w = w / w.sum()

            # 3. 加权
            weighted = matrix_norm * w

            # 4. 正负理想解
            z_plus = np.where(
                [c not in reverse_cols for c in cols],
                weighted.max(axis=0),
                weighted.min(axis=0),
            )
            z_minus = np.where(
                [c not in reverse_cols for c in cols],
                weighted.min(axis=0),
                weighted.max(axis=0),
            )

            # 5. 距离
            d_plus = np.sqrt(((weighted - z_plus) ** 2).sum(axis=1))
            d_minus = np.sqrt(((weighted - z_minus) ** 2).sum(axis=1))

            # 6. 贴近度
            denominator = d_plus + d_minus
            c = np.where(denominator > 0, d_minus / denominator, 0.0)

            results = []
            for i in range(len(group)):
                results.append({
                    "d_plus": float(d_plus[i]),
                    "d_minus": float(d_minus[i]),
                    "c_score": float(c[i]),
                })
            return results

        if group_by:
            all_rows = []
            for g_name, g_group in df.groupby(group_by):
                gb_val = [g_name] if len(group_by) == 1 else list(g_name)
                for idx, res in zip(g_group.index, _topsis(g_group)):
                    row = df.loc[idx].to_dict()
                    row.update(res)
                    all_rows.append(row)
            out = pd.DataFrame(all_rows)
        else:
            rows = df[id_vars + cols].copy()
            results = _topsis(df)
            for i, res in enumerate(results):
                for k, v in res.items():
                    rows.loc[rows.index[i], k] = v
            out = rows

        return AnalysisResult(
            success=True,
            data=out,
            summary={
                "algorithm": "topsis",
                "n_indicators": len(cols),
                "weights": weights is not None,
            },
        )


# =============================================================================
# PCA —— 主成分分析
# =============================================================================

class PCA(BaseAnalyzer):
    """PCA 主成分分析，用于降维和综合评价。

    步骤：
      1. 标准化数据（Z-score）
      2. 奇异值分解
      3. 自动按累积方差 ≥ 85% 确定主成分数量
      4. 输出载荷矩阵、方差贡献率、综合得分

    Args:
        n_components: 主成分数量。若为 None 则按累积方差 ≥ 85% 自动确定。

    Examples:
        >>> pca = PCA()
        >>> r = pca.analyze(df, value_cols=["gdp", "edu", "med", "env", "pop"])
        >>> r.data["components"]   # 载荷矩阵
        >>> r.data["score"]        # 综合得分（方差加权）
    """

    name = "pca"
    category = "composite"
    description = "PCA 主成分分析: 降维 + 自动确定成分数 + 综合得分"
    output_type = "dataframe"
    executor = "python"
    timeout = 15

    def analyze(
        self,
        data: pd.DataFrame,
        value_cols: list[str] | None = None,
        n_components: int | None = None,
    ) -> AnalysisResult:
        self.validate(data)
        cols = value_cols or list(data.select_dtypes(include=[np.number]).columns)
        if len(cols) < 2:
            raise ValueError("PCA 至少需要 2 个指标列")
        _check_cols(data, cols)

        df = data[cols].copy().dropna()
        id_cols = [c for c in data.columns if c not in cols]

        X = df[cols].values
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0, ddof=0)
        X_std = np.where(X_std == 0, 1, X_std)
        X_scaled = (X - X_mean) / X_std

        # PCA
        pca_model = SklearnPCA(n_components=n_components)
        X_pca = pca_model.fit_transform(X_scaled)

        # 方差贡献
        explained_var = pca_model.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        # 确定有效主成分数
        if n_components is None:
            effective_n = int(np.searchsorted(cumulative_var, 0.85) + 1)
        else:
            effective_n = n_components

        # 综合得分（方差加权）
        weights = explained_var[:effective_n] / explained_var[:effective_n].sum()
        if X_pca.shape[1] >= effective_n:
            composite_score = (X_pca[:, :effective_n] * weights).sum(axis=1)
        else:
            composite_score = X_pca[:, 0] if X_pca.shape[1] > 0 else np.zeros(X_pca.shape[0])

        # 载荷矩阵
        loadings = pca_model.components_.T

        # 构建结果
        result_df = data.loc[df.index, id_cols + cols].copy()
        result_df["pca_score"] = composite_score

        component_cols = {}
        for i in range(min(effective_n, X_pca.shape[1])):
            col_name = f"pc{i + 1}"
            result_df[col_name] = X_pca[:, i]
            component_cols[col_name] = {
                "variance_ratio": float(explained_var[i]),
                "cumulative": float(cumulative_var[i]),
            }

        # 载荷表
        loadings_df = pd.DataFrame(
            loadings[:, :effective_n],
            index=cols,
            columns=[f"pc{i + 1}" for i in range(effective_n)],
        )
        loadings_df["communality"] = (loadings[:, :effective_n] ** 2).sum(axis=1)

        return AnalysisResult(
            success=True,
            data=result_df,
            metadata={
                "components": component_cols,
                "loadings": loadings_df.to_dict(),
                "effective_n": effective_n,
                "total_variance": float(cumulative_var[:effective_n].max()),
            },
            summary={
                "algorithm": "pca",
                "n_indicators": len(cols),
                "effective_components": effective_n,
                "total_variance_explained": float(cumulative_var[:effective_n].max()),
            },
        )


# =============================================================================
# TwoLevelRank —— 两级综合排名
# =============================================================================

class TwoLevelRank(BaseAnalyzer):
    """两级综合排名。

    第一级：在每个领域（dimension）内，对指标做百分位排名并平均得到领域分
    第二级：用熵权法确定领域权重，加权聚合得到综合分

    Args:
        dim_col: 领域列名（如 "dimension" 列标识经济/民生/环境）。
        value_col: 指标值列名。
        score_col: 领域内打分用指标列（若不指定则用 value_col）。

    Examples:
        >>> tlr = TwoLevelRank()
        >>> r = tlr.analyze(df, dim_col="dimension", value_col="score",
        ...                group_by=["province"])
        >>> r.data  # 含 l1_score、l2_score、final_rank
    """

    name = "two_level_rank"
    category = "composite"
    description = "两级综合排名: L1百分位归一化 + L2熵权聚合"
    output_type = "dataframe"
    executor = "python"
    timeout = 15

    def analyze(
        self,
        data: pd.DataFrame,
        dim_col: str = "dimension",
        value_col: str = "value",
        group_by: list[str] | None = None,
        reverse_cols: list[str] | None = None,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [dim_col, value_col])

        df = data[[dim_col, value_col] + (group_by or [])].copy()
        reverse_cols = set(reverse_cols or [])

        # L1：领域内百分位排名
        l1_gb = (group_by or []) + [dim_col]
        if value_col in reverse_cols:
            ascending = True  # 值越小越"好"，排名时升序
        else:
            ascending = False

        df["l1_score"] = df.groupby(l1_gb)[value_col].rank(
            method="min", ascending=ascending, pct=True
        )

        # L2：熵权法确定领域权重
        l2_input = df.groupby((group_by or []) + [dim_col])["l1_score"].mean().reset_index()
        l2_input.rename(columns={"l1_score": "domain_score"}, inplace=True)

        # 以 dim_col 为列做透视，每个领域一列
        pivot = l2_input.pivot_table(
            index=group_by or ["_dummy"],
            columns=dim_col,
            values="domain_score",
        ).reset_index()
        if "_dummy" in pivot.columns:
            pivot.drop(columns=["_dummy"], inplace=True)

        domain_cols = [c for c in pivot.columns if c not in (group_by or [])]

        # 执行熵权法
        ew = EntropyWeight()
        ew_result = ew.analyze(pivot, value_cols=domain_cols)

        weights = ew_result.data.set_index("indicator")["weight"].to_dict()

        # 计算综合得分
        for col, w in weights.items():
            pivot[f"{col}_weighted"] = pivot[col] * w

        pivot["final_score"] = pivot[[f"{c}_weighted" for c in weights]].sum(axis=1)
        pivot["final_rank"] = pivot["final_score"].rank(
            method="dense", ascending=False
        ).astype(int)
        pivot.sort_values("final_rank", inplace=True)

        return AnalysisResult(
            success=True,
            data=pivot,
            metadata={"domain_weights": weights},
            summary={
                "algorithm": "two_level_rank",
                "domains": domain_cols,
                "weights": weights,
            },
        )


# =============================================================================
# 内部工具函数
# =============================================================================

def _check_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"数据中缺少必需的列: {missing}")


def _auto_weights(
    matrix: np.ndarray,
    reverse_cols: set[str],
    cols: list[str],
) -> np.ndarray:
    """自动熵权法计算权重（TOPSIS 内部使用）。"""
    n, m = matrix.shape
    norm_matrix = matrix.copy().astype(float)

    for i, col in enumerate(cols):
        col_vals = norm_matrix[:, i]
        vmin, vmax = col_vals.min(), col_vals.max()
        if vmax > vmin:
            col_norm = (col_vals - vmin) / (vmax - vmin)
        else:
            col_norm = np.ones(n) * 0.5
        if col in reverse_cols:
            col_norm = 1 - col_norm
        col_norm = np.clip(col_norm, 1e-10, 1 - 1e-10)
        norm_matrix[:, i] = col_norm

    p = norm_matrix / norm_matrix.sum(axis=0, keepdims=True)
    p = np.where(p > 0, p, 1e-10)

    k = 1.0 / np.log(n) if n > 1 else 1.0
    e = -k * np.sum(p * np.log(p), axis=0)
    e = np.clip(e, 0, 1)
    d = 1 - e
    return d / d.sum() if d.sum() > 0 else np.ones(m) / m


# =============================================================================
# __all__
# =============================================================================

__all__ = [
    "MinMax",
    "EntropyWeight",
    "TOPSIS",
    "PCA",
    "TwoLevelRank",
]
