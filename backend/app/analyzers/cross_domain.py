"""跨领域综合分析算法库。

提供以下算法：
  - CCD                    耦合协调度（2系统 + n系统扩展）
  - DEA                    数据包络分析（CCR 模型）
  - SupplyDemandResidual   供需残差分析（回归拟合 + 差额诊断）
  - DecouplingIndex        OECD 脱钩指数
  - Elasticity             弹性系数模型
  - PanelOLS               面板数据多元回归（固定效应）
  - GrangerCausality       Granger 因果检验
  - KMeansClustering       K-means 聚类 + 轮廓系数定 k
  - HierarchicalClustering 层次聚类（Ward  linkage）
  - TheilIndex             泰尔指数 + 群组分解
  - TripleValidation       三级验证链（弹性 → 回归 → Granger）
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.optimize import linprog
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests

from .base import AnalysisResult, BaseAnalyzer


# =============================================================================
# CCD —— 耦合协调度
# =============================================================================

class CCD(BaseAnalyzer):
    """耦合协调度模型 (Coupling Coordination Degree)。

    2 系统: C = 2 * sqrt(u1 * u2) / (u1 + u2)
    n 系统: C = n * (u1*u2*...*un)^(1/n) / (u1 + u2 + ... + un)
    协调度: D = sqrt(C * T)
    T = 各子系统得分的加权和（等权或自定义权重）

    Args:
        n_systems: 系统个数。若为 None 则从 sub_col 自动推断。
        sub_col: 标识子系统名称的列名（如 "system" 列含 "经济"/"民生"）。
        value_col: 子系统得分列。
        weights: 子系统权重（若为 None 则等权）。

    Examples:
        >>> ccd = CCD()
        >>> r = ccd.analyze(df, sub_col="system", value_col="score",
        ...                 group_by=["province"])
        >>> r.data  # 含 coupling, coordination 列
    """

    name = "ccd"
    category = "cross_domain"
    description = "耦合协调度: 2系统/n系统的 C 耦合度 + D 协调度"
    output_type = "dataframe"
    executor = "python"
    timeout = 15

    def analyze(
        self,
        data: pd.DataFrame,
        sub_col: str | None = None,
        value_col: str = "value",
        group_by: list[str] | None = None,
        n_systems: int | None = None,
        weights: list[float] | None = None,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [value_col])

        # 子系统模式：data 含 sub_col + value_col
        if sub_col:
            _check_cols(data, [sub_col])
            pivot = data.pivot_table(
                index=group_by or [],
                columns=sub_col,
                values=value_col,
            ).reset_index()
            score_cols = [c for c in pivot.columns if c not in (group_by or [])]
            scores = pivot[score_cols].values
            n = scores.shape[1]
        else:
            df = data[[value_col] + (group_by or [])].copy()
            score_cols = [value_col]
            scores = df[value_col].values.reshape(-1, 1)
            n = 1

        n = n_systems or n

        # 归一化子系统得分到 [0, 1]
        scores_norm = scores.copy().astype(float)
        for i in range(scores_norm.shape[1]):
            col_vals = scores_norm[:, i]
            vmin, vmax = col_vals.min(), col_vals.max()
            if vmax > vmin:
                scores_norm[:, i] = (col_vals - vmin) / (vmax - vmin)
            else:
                scores_norm[:, i] = 0.5

        # 综合得分 T（等权或自定义权重）
        w = np.array(weights) if weights else np.ones(n) / n
        T = (scores_norm * w).sum(axis=1)

        if n == 1:
            C = np.ones_like(T)
        elif n == 2:
            u1, u2 = scores_norm[:, 0], scores_norm[:, 1]
            denom = u1 + u2
            C = np.where(denom > 0, 2 * np.sqrt(u1 * u2) / denom, 0)
        else:
            product = np.prod(scores_norm, axis=1)
            sum_scores = scores_norm.sum(axis=1)
            C = np.where(
                sum_scores > 0,
                n * (product ** (1.0 / n)) / sum_scores,
                0,
            )

        D = np.sqrt(C * T)

        # 回填 CCD 结果到原 DataFrame
        if sub_col:
            pivot["cc_coupling"] = C
            pivot["cc_coordination"] = D
            pivot["cc_T"] = T
            merge_cols = (group_by or []) + [sub_col]
            # 先把 pivot 还原为长格式后再合并
            id_vars = (group_by or []) + ["cc_coupling", "cc_coordination", "cc_T"]
            pivot_long = pivot.melt(
                id_vars=id_vars,
                value_vars=list(pivot.columns.difference(id_vars)),
                value_name="_score_placeholder",
            ).drop(columns=["_score_placeholder"])
            result_df = data.merge(
                pivot_long[merge_cols + ["cc_coupling", "cc_coordination", "cc_T"]],
                on=merge_cols, how="left",
            )
        else:
            result_df = data.copy()
            result_df["cc_coupling"] = C
            result_df["cc_coordination"] = D
            result_df["cc_T"] = T

        return AnalysisResult(
            success=True,
            data=result_df,
            summary={
                "algorithm": "ccd",
                "n_systems": n,
                "mean_coupling": float(C.mean()),
                "mean_coordination": float(D.mean()),
            },
        )


# =============================================================================
# DEA —— 数据包络分析
# =============================================================================

class DEA(BaseAnalyzer):
    """数据包络分析 (Data Envelopment Analysis, CCR 模型)。

    对每个 DMU（决策单元）求解线性规划，计算相对效率 score ∈ [0, 1]。
    score = 1 表示处于效率前沿面。

    假设：投入越小越好，产出越大越好。

    Args:
        input_cols: 投入指标列名列表。
        output_cols: 产出指标列名列表。

    Examples:
        >>> dea = DEA()
        >>> r = dea.analyze(df, input_cols=["investment", "edu_spending"],
        ...                 output_cols=["gdp", "literacy_rate"])
        >>> r.data["efficiency"]
    """

    name = "dea"
    category = "cross_domain"
    description = "DEA(CCR): 数据包络分析，计算每个 DMU 的相对效率 [0,1]"
    output_type = "dataframe"
    executor = "python"
    timeout = 30

    def analyze(
        self,
        data: pd.DataFrame,
        input_cols: list[str] | None = None,
        output_cols: list[str] | None = None,
    ) -> AnalysisResult:
        self.validate(data)
        if not input_cols or not output_cols:
            raise ValueError("必须指定 input_cols 和 output_cols")
        _check_cols(data, input_cols + output_cols)

        X = data[input_cols].values.astype(float)
        Y = data[output_cols].values.astype(float)
        n_dmus = len(data)

        efficiencies = []
        for k in range(n_dmus):
            # 变量: [θ, λ_1, ..., λ_n, s⁻_1, ..., s⁻_m, s⁺_1, ..., s⁺_r]
            m = len(input_cols)
            r = len(output_cols)
            n_vars = 1 + n_dmus + m + r

            # 目标: min θ (只保留 θ 的系数)
            c = np.zeros(n_vars)
            c[0] = 1.0

            # 不等式约束: Xλ + s⁻ <= θ*x_k  →  Xλ + s⁻ - θ*x_k <= 0
            A_ub = np.zeros((m, n_vars))
            for i in range(m):
                A_ub[i, 0] = -X[k, i]  # -θ * x_k,i
                A_ub[i, 1:1 + n_dmus] = X[:, i]  # + X_i * λ
                A_ub[i, 1 + n_dmus + i] = 1.0  # + s⁻_i
            b_ub = np.zeros(m)

            # 等式约束: Yλ - s⁺ = y_k
            A_eq = np.zeros((r, n_vars))
            for j in range(r):
                A_eq[j, 1:1 + n_dmus] = -Y[:, j]
                A_eq[j, 1 + n_dmus + m + j] = 1.0
            b_eq = -Y[k, :]

            # 变量边界: θ free, λ >= 0, s⁻ >= 0, s⁺ >= 0
            bounds = [(None, None)] + [(0, None)] * (n_vars - 1)
            bounds[0] = (0, None)

            try:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                              b_eq=b_eq, bounds=bounds, method="highs")
                eff = res.fun if res.success else np.nan
            except Exception:
                eff = np.nan

            efficiencies.append(eff)

        result = data.copy()
        result["efficiency"] = efficiencies
        result["is_frontier"] = (result["efficiency"] >= 0.999) & (
            result["efficiency"] <= 1.001
        )

        return AnalysisResult(
            success=True,
            data=result,
            summary={
                "algorithm": "dea_ccr",
                "n_dmus": n_dmus,
                "n_frontier": int(result["is_frontier"].sum()),
                "mean_efficiency": float(result["efficiency"].mean()),
            },
        )


# =============================================================================
# SupplyDemandResidual —— 供需残差分析
# =============================================================================

class SupplyDemandResidual(BaseAnalyzer):
    """供需残差分析。

    通过回归模型拟合"理论需求"量，残差 = 实际值 - 理论值。
    残差为负 → 供不应求；残差为正 → 供过于求。

    Args:
        supply_col: 实际供给量列名。
        demand_factors: 需求驱动因素列名（如人口、老龄化率等）。
        residual_scale: 残差缩放方式，"raw" 原始值，"zscore" 标准化，"ratio" 百分比。

    Examples:
        >>> sdr = SupplyDemandResidual()
        >>> r = sdr.analyze(df, supply_col="bed_count",
        ...                 demand_factors=["population", "aging_rate"])
        >>> r.data["residual"], r.data["status"]
    """

    name = "supply_demand_residual"
    category = "cross_domain"
    description = "供需残差分析: 回归拟合理论需求，残差 = 实际 - 理论"
    output_type = "dataframe"
    executor = "python"
    timeout = 15

    def analyze(
        self,
        data: pd.DataFrame,
        supply_col: str = "supply",
        demand_factors: list[str] | None = None,
        residual_scale: str = "ratio",
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [supply_col])
        if demand_factors:
            _check_cols(data, demand_factors)

        df = data[[supply_col] + (demand_factors or [])].copy().dropna()
        if len(df) < 5:
            return AnalysisResult(
                success=False,
                error=f"数据不足（{len(df)} 行），至少需要 5 行",
            )

        X = df[demand_factors].values if demand_factors else np.arange(len(df)).reshape(-1, 1)
        y = df[supply_col].values

        # OLS 拟合
        X = np.column_stack([np.ones(X.shape[0]), X])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred = X @ beta

        residuals = y - y_pred

        if residual_scale == "zscore":
            residuals_scaled = (residuals - residuals.mean()) / (residuals.std() + 1e-10)
        elif residual_scale == "ratio":
            residuals_scaled = residuals / (y_pred + 1e-10) * 100
        else:
            residuals_scaled = residuals

        result = df.copy()
        result["predicted"] = y_pred
        result["residual"] = residuals
        result["residual_scaled"] = residuals_scaled
        result["status"] = np.where(
            residuals_scaled > 0, "供过于求",
            np.where(residuals_scaled < 0, "供不应求", "均衡"),
        )

        return AnalysisResult(
            success=True,
            data=result,
            summary={
                "algorithm": "supply_demand_residual",
                "n_shortage": int((residuals_scaled < 0).sum()),
                "n_surplus": int((residuals_scaled > 0).sum()),
                "mean_residual": float(residuals.mean()),
                "r_squared": float(1 - (residuals ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            },
        )


# =============================================================================
# DecouplingIndex —— 脱钩指数
# =============================================================================

class DecouplingIndex(BaseAnalyzer):
    """OECD 脱钩指数 (Decoupling Index)。

    衡量经济增长与环境压力的关系：
      DF = 1 - (%ΔEnvPressure / %ΔEconomicDriver)

    脱钩状态：
      - 强脱钩 (DF > 1, env decrease, econ grow): 最理想
      - 弱脱钩 (0 < DF < 1): 环境压力增速低于经济增速
      - 衰退脱钩 (DF > 1, both decrease): 经济衰退伴随环境改善
      - 扩张负脱钩 (DF < 0, both grow): 环境压力增速高于经济增速
      - 强负脱钩 (DF < 0, env grow, econ shrink): 最差

    Args:
        env_col: 环境压力指标列（如 PM2.5、固体废物量）。
        econ_col: 经济驱动指标列（如 GDP）。
        time_col: 时间列，用于计算变化率。

    Examples:
        >>> di = DecouplingIndex()
        >>> r = di.analyze(df, env_col="pm25", econ_col="gdp",
        ...                group_by=["province"])
        >>> r.data["decoupling_status"]
    """

    name = "decoupling_index"
    category = "cross_domain"
    description = "OECD 脱钩指数: 衡量经济增长与环境压力的脱钩程度"
    output_type = "dataframe"
    executor = "python"
    timeout = 15

    _STATUS_MAP = {
        (True, True, True): "强脱钩",
        (True, True, False): "弱脱钩",
        (True, False, True): "衰退脱钩",
        (True, False, False): "衰退耦合",
        (False, True, True): "扩张负脱钩",
        (False, True, False): "扩张耦合",
        (False, False, True): "强负脱钩",
        (False, False, False): "弱负脱钩",
    }

    def analyze(
        self,
        data: pd.DataFrame,
        env_col: str = "env",
        econ_col: str = "econ",
        time_col: str | None = None,
        group_by: list[str] | None = None,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [env_col, econ_col])

        extra_cols = [time_col] if time_col else []
        df = data[[env_col, econ_col] + (group_by or []) + extra_cols].copy()

        if time_col:
            df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
            df = df.dropna(subset=[time_col])
            df = df.sort_values((group_by or []) + [time_col])
            df = df.dropna(subset=[time_col])
            df = df.sort_values((group_by or []) + [time_col])

            def _decouple(group: pd.DataFrame) -> dict[str, Any]:
                env_pct = group[env_col].pct_change() * 100
                econ_pct = group[econ_col].pct_change() * 100
                valid = ~(env_pct.isna() | econ_pct.isna() |
                          (env_pct.abs() < 0.001) | (econ_pct.abs() < 0.001))
                if not valid.any():
                    return {}

                env_change = env_pct[valid].mean()
                econ_change = econ_pct[valid].mean()

                if abs(econ_change) < 0.001:
                    df_val = np.nan
                else:
                    df_val = 1 - env_change / econ_change

                env_up = env_change > 0
                econ_up = econ_change > 0
                df_pos = df_val > 0

                status = _classify_decoupling(df_val, env_change, econ_change)

                return {
                    "decoupling_index": df_val,
                    "env_change_pct": env_change,
                    "econ_change_pct": econ_change,
                    "decoupling_status": status,
                }

            if group_by:
                result = df.groupby(group_by).apply(_decouple, include_groups=False)
                out = pd.json_normalize(result)
                out.index = result.index
                out = out.reset_index()
            else:
                out = pd.DataFrame([_decouple(df)])
        else:
            # 无时间列：直接算两期变化率
            def _simple_decouple(group: pd.DataFrame) -> dict[str, Any]:
                env_vals = group[env_col].values
                econ_vals = group[econ_col].values
                if len(env_vals) < 2:
                    return {}
                env_change = (env_vals[-1] - env_vals[0]) / env_vals[0] * 100
                econ_change = (econ_vals[-1] - econ_vals[0]) / econ_vals[0] * 100
                if abs(econ_change) < 0.001:
                    df_val = np.nan
                else:
                    df_val = 1 - env_change / econ_change
                status = _classify_decoupling(df_val, env_change, econ_change)
                return {
                    "decoupling_index": df_val,
                    "env_change_pct": env_change,
                    "econ_change_pct": econ_change,
                    "decoupling_status": status,
                }

            if group_by:
                result = df.groupby(group_by).apply(_simple_decouple, include_groups=False)
                out = pd.json_normalize(result)
                out.index = result.index
                out = out.reset_index()
            else:
                out = pd.DataFrame([_simple_decouple(df)])

        return AnalysisResult(success=True, data=out)


# =============================================================================
# Elasticity —— 弹性系数模型
# =============================================================================

class Elasticity(BaseAnalyzer):
    """弹性系数模型。

    ε = %ΔY / %ΔX
    表示 X 每变化 1%，Y 变化的百分比。

    支持滚动窗口弹性计算（观察弹性随时间的变化）。

    Args:
        y_col: 因变量列名。
        x_col: 自变量列名。
        time_col: 时间列（滚动窗口需要）。
        window: 滚动窗口大小，默认 None（全局弹性）。

    Examples:
        >>> el = Elasticity()
        >>> r = el.analyze(df, y_col="gdp", x_col="investment",
        ...                time_col="year", group_by=["province"])
        >>> r.data["elasticity"]
    """

    name = "elasticity"
    category = "cross_domain"
    description = "弹性系数: ε = %ΔY / %ΔX，支持滚动窗口"
    output_type = "dataframe"
    executor = "python"
    timeout = 15

    def analyze(
        self,
        data: pd.DataFrame,
        y_col: str = "y",
        x_col: str = "x",
        time_col: str | None = None,
        group_by: list[str] | None = None,
        window: int | None = None,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [y_col, x_col])

        extra_cols = [time_col] if time_col else []
        df = data[[y_col, x_col] + (group_by or []) + extra_cols].copy()

        if time_col:
            df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
            df = df.sort_values((group_by or []) + [time_col])
            df["y_pct"] = df.groupby(group_by or [])[y_col].transform(
                lambda s: s.pct_change() * 100
            )
            df["x_pct"] = df.groupby(group_by or [])[x_col].transform(
                lambda s: s.pct_change() * 100
            )
        else:
            df["y_pct"] = df[y_col].pct_change() * 100
            df["x_pct"] = df[x_col].pct_change() * 100

        if window:
            if group_by:
                df["elasticity"] = df.groupby(group_by).apply(
                    lambda g: g["y_pct"] / g["x_pct"].replace(0, np.nan)
                ).values
                df["elasticity"] = df.groupby(group_by)["elasticity"].transform(
                    lambda s: s.rolling(window, min_periods=1).mean()
                )
            else:
                df["elasticity"] = df["y_pct"] / df["x_pct"].replace(0, np.nan)
                df["elasticity"] = df["elasticity"].rolling(window, min_periods=1).mean()
        else:
            x_pct = df["x_pct"].replace([0, np.inf, -np.inf], np.nan)
            df["elasticity"] = df["y_pct"] / x_pct

        df["elasticity"] = df["elasticity"].replace([np.inf, -np.inf], np.nan)

        return AnalysisResult(
            success=True,
            data=df,
            summary={
                "algorithm": "elasticity",
                "window": window,
                "mean_elasticity": float(df["elasticity"].mean()) if not df["elasticity"].isna().all() else np.nan,
            },
        )


# =============================================================================
# PanelOLS —— 面板数据多元回归
# =============================================================================

class PanelOLS(BaseAnalyzer):
    """面板数据多元回归（实体固定效应模型）。

    使用最小二乘虚拟变量 (LSDV) 方法：为每个实体（省份）添加虚拟变量。
    输出标准化回归系数（beta 系数），用于比较各因素的相对重要性。

    Args:
        dep_col: 因变量列名。
        indep_cols: 自变量列名列表。
        entity_col: 实体标识列（如 "province"）。
        time_col: 时间列（可选）。
        standardized: True 输出标准化系数，默认 True。

    Examples:
        >>> pols = PanelOLS()
        >>> r = pols.analyze(df, dep_col="gdp",
        ...                  indep_cols=["investment", "edu", "med"],
        ...                  entity_col="province")
        >>> r.data  # 含 coef, std_err, pvalue, beta
    """

    name = "panel_ols"
    category = "cross_domain"
    description = "面板 OLS: 实体固定效应多元回归，输出标准化系数"
    output_type = "dataframe"
    executor = "python"
    timeout = 20

    def analyze(
        self,
        data: pd.DataFrame,
        dep_col: str = "y",
        indep_cols: list[str] | None = None,
        entity_col: str | None = None,
        time_col: str | None = None,
        standardized: bool = True,
    ) -> AnalysisResult:
        self.validate(data)
        if not indep_cols:
            raise ValueError("必须指定 indep_cols")
        _check_cols(data, [dep_col] + indep_cols)

        df = data[[dep_col] + indep_cols].copy()
        if entity_col:
            _check_cols(data, [entity_col])
            df[entity_col] = data[entity_col]

        df = df.dropna()

        # 构建特征矩阵
        X_cols = list(indep_cols)

        # 实体固定效应虚拟变量
        if entity_col:
            dummies = pd.get_dummies(df[entity_col], prefix="ent", drop_first=True)
            X_cols += list(dummies.columns)
            X = np.column_stack([df[indep_cols].values, dummies.values.astype(float)])
        else:
            X = df[indep_cols].values

        y = df[dep_col].values

        # OLS
        X = np.column_stack([np.ones(X.shape[0]), X])
        n, k = X.shape
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        mse = (residuals ** 2).sum() / (n - k)
        var_beta = mse * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(var_beta))
        t_stats = beta / se

        from scipy.stats import t as t_dist
        p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=n - k))

        # 标准化系数
        if standardized:
            y_std = y.std(ddof=1)
            x_std = np.std(X[:, 1:], axis=0, ddof=1)
            beta_std = beta[1:] * x_std / y_std if y_std > 0 else beta[1:]
        else:
            beta_std = beta[1:]

        # 构建输出 DataFrame
        result_cols = ["const"] + indep_cols
        if entity_col:
            result_cols += list(dummies.columns)

        result = pd.DataFrame({
            "variable": result_cols,
            "coefficient": beta,
            "std_error": se,
            "t_statistic": t_stats,
            "p_value": p_values,
        })
        result["beta"] = np.concatenate([[np.nan], beta_std])

        within_r2 = 1 - (residuals ** 2).sum() / ((y - y.mean()) ** 2).sum()

        if entity_col:
            result["type"] = ["constant"] + ["independent"] * len(indep_cols) + ["entity_fe"] * len(dummies.columns)
        else:
            result["type"] = ["constant"] + ["independent"] * len(indep_cols)

        return AnalysisResult(
            success=True,
            data=result,
            summary={
                "algorithm": "panel_ols",
                "n_obs": n,
                "n_params": k,
                "r_squared": float(within_r2),
                "entity_fe": entity_col is not None,
            },
        )


# =============================================================================
# GrangerCausality —— Granger 因果检验
# =============================================================================

class GrangerCausality(BaseAnalyzer):
    """Granger 因果检验。

    检验 x 的滞后值是否有助于预测 y（x → y 的 Granger 因果）。
    原假设 H0: x 不是 y 的 Granger 原因。

    Args:
        x_col: 原因变量。
        y_col: 结果变量。
        time_col: 时间列。
        max_lag: 最大滞后阶数，默认 4。
        significance: 显著性水平，默认 0.05。

    Examples:
        >>> gc = GrangerCausality()
        >>> r = gc.analyze(df, x_col="real_estate_inv", y_col="gdp",
        ...                time_col="quarter", group_by=["province"])
        >>> r.data["best_lag"], r.data["causal_direction"]
    """

    name = "granger_causality"
    category = "cross_domain"
    description = "Granger 因果检验: x 的滞后值是否有助于预测 y"
    output_type = "dataframe"
    executor = "python"
    timeout = 30

    def analyze(
        self,
        data: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        time_col: str = "time",
        group_by: list[str] | None = None,
        max_lag: int = 4,
        significance: float = 0.05,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [x_col, y_col, time_col])

        df = data[[time_col, x_col, y_col] + (group_by or [])].copy()
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col, x_col, y_col])

        def _granger(group: pd.DataFrame) -> dict[str, Any]:
            g = group.sort_values(time_col)
            test_data = g[[x_col, y_col]].values
            if len(test_data) < max_lag + 5:
                return {"best_lag": np.nan, "test_stat": np.nan,
                        "p_value": np.nan, "causal": False}

            try:
                result = grangercausalitytests(test_data, max_lag, verbose=False)
            except Exception:
                return {"best_lag": np.nan, "test_stat": np.nan,
                        "p_value": np.nan, "causal": False}

            best_lag = None
            best_p = np.inf
            best_stat = None
            for lag, r in result.items():
                # 使用 F-test 的 p-value
                p_val = r[0]["ssr_ftest"][1]
                stat = r[0]["ssr_ftest"][0]
                if p_val < best_p:
                    best_p = p_val
                    best_lag = lag
                    best_stat = stat

            return {
                "best_lag": int(best_lag) if best_lag is not None else np.nan,
                "test_stat": float(best_stat) if best_stat is not None else np.nan,
                "p_value": float(best_p) if best_p != np.inf else np.nan,
                "causal": bool(best_p < significance) if best_p != np.inf else False,
            }

        if group_by:
            result = df.groupby(group_by).apply(_granger, include_groups=False)
            out = pd.json_normalize(result)
            out.index = result.index
            out = out.reset_index()
        else:
            out = pd.DataFrame([_granger(df)])

        return AnalysisResult(
            success=True,
            data=out,
            summary={
                "algorithm": "granger_causality",
                "max_lag": max_lag,
                "significance": significance,
            },
        )


# =============================================================================
# KMeansClustering —— K-means 聚类
# =============================================================================

class KMeansClustering(BaseAnalyzer):
    """K-means 聚类 + 轮廓系数自动选 k。

    步骤：
      1. 标准化数据
      2. 在 k=2~K_max 范围内计算轮廓系数
      3. 选择轮廓系数最大的 k 值
      4. 执行 K-means 聚类

    Args:
        feature_cols: 聚类特征列。
        k: 聚类数。若为 None 则自动确定。
        k_max: 最大尝试聚类数，默认 10。

    Examples:
        >>> km = KMeansClustering()
        >>> r = km.analyze(df, feature_cols=["gdp", "edu", "med"])
        >>> r.data  # 含 cluster 列
    """

    name = "kmeans"
    category = "cross_domain"
    description = "K-means 聚类 + 轮廓系数自动选 k"
    output_type = "dataframe"
    executor = "python"
    timeout = 30

    def analyze(
        self,
        data: pd.DataFrame,
        feature_cols: list[str] | None = None,
        k: int | None = None,
        k_max: int = 10,
        random_state: int = 42,
    ) -> AnalysisResult:
        self.validate(data)
        cols = feature_cols or list(data.select_dtypes(include=[np.number]).columns)
        if len(cols) < 2:
            raise ValueError("聚类至少需要 2 个特征列")
        _check_cols(data, cols)

        df = data[cols].copy().dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df.values)

        n_samples = len(X_scaled)

        if k is not None:
            best_k = k
        else:
            max_k = min(k_max, n_samples - 1)
            if max_k < 2:
                return AnalysisResult(success=False, error="样本数不足，无法聚类")
            best_k = 2
            best_score = -1
            for k_candidate in range(2, max_k + 1):
                km = SklearnKMeans(n_clusters=k_candidate, random_state=random_state, n_init="auto")
                labels = km.fit_predict(X_scaled)
                unique_labels = len(set(labels))
                if unique_labels < 2:
                    continue
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_k = k_candidate

        model = SklearnKMeans(n_clusters=best_k, random_state=random_state, n_init="auto")
        df["cluster"] = model.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled, df["cluster"].values)

        # 聚类中心（原始尺度）
        centers = scaler.inverse_transform(model.cluster_centers_)
        centers_df = pd.DataFrame(centers, columns=cols)
        centers_df["cluster"] = range(best_k)

        # 各聚类统计
        cluster_stats = df.groupby("cluster")[cols].agg(["mean", "std", "count"])

        return AnalysisResult(
            success=True,
            data=df,
            metadata={
                "centers": centers_df.to_dict(),
                "cluster_stats": cluster_stats.to_dict(),
                "silhouette_score": sil_score,
            },
            summary={
                "algorithm": "kmeans",
                "k": best_k,
                "n_features": len(cols),
                "silhouette_score": float(sil_score),
                "auto_selected": k is None,
            },
        )


# =============================================================================
# HierarchicalClustering —— 层次聚类
# =============================================================================

class HierarchicalClustering(BaseAnalyzer):
    """层次聚类 (Hierarchical Clustering, Ward linkage)。

    步骤：
      1. 标准化数据
      2. 计算距离矩阵并执行 Ward 层次聚类
      3. 指定聚类数或自动用 inconsistency 确定
      4. 输出聚类标签

    Args:
        feature_cols: 聚类特征列。
        n_clusters: 聚类数。若为 None 则自动确定。
        t: inconsistency 阈值（自动定聚类数时用），默认 1.0。

    Examples:
        >>> hc = HierarchicalClustering()
        >>> r = hc.analyze(df, feature_cols=["gdp", "edu", "med"])
        >>> r.data["cluster"]
    """

    name = "hierarchical_clustering"
    category = "cross_domain"
    description = "层次聚类 (Ward linkage): 凝聚式层次聚类"
    output_type = "dataframe"
    executor = "python"
    timeout = 20

    def analyze(
        self,
        data: pd.DataFrame,
        feature_cols: list[str] | None = None,
        n_clusters: int | None = None,
        t: float = 1.0,
    ) -> AnalysisResult:
        self.validate(data)
        cols = feature_cols or list(data.select_dtypes(include=[np.number]).columns)
        if len(cols) < 2:
            raise ValueError("聚类至少需要 2 个特征列")
        _check_cols(data, cols)

        df = data[cols].copy().dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df.values)

        Z = linkage(X_scaled, method="ward")

        if n_clusters is not None:
            labels = fcluster(Z, n_clusters, criterion="maxclust")
        else:
            from scipy.cluster.hierarchy import inconsistent
            inc = inconsistent(Z)
            depth = min(t, len(inc))
            if depth > 0:
                n_clusters_auto = np.sum(inc[:, 2] > inc[:, 3] * t) + 1
                n_clusters_auto = min(n_clusters_auto, len(df) - 1)
            else:
                n_clusters_auto = 2
            labels = fcluster(Z, n_clusters_auto, criterion="maxclust")
            n_clusters = n_clusters_auto

        df["cluster"] = labels

        sil_score = (
            silhouette_score(X_scaled, labels)
            if len(set(labels)) > 1 and len(set(labels)) < len(df)
            else np.nan
        )

        return AnalysisResult(
            success=True,
            data=df,
            metadata={
                "linkage_matrix": Z.tolist(),
            },
            summary={
                "algorithm": "hierarchical_clustering",
                "n_clusters": n_clusters,
                "silhouette_score": float(sil_score) if not np.isnan(sil_score) else None,
            },
        )


# =============================================================================
# TheilIndex —— 泰尔指数
# =============================================================================

class TheilIndex(BaseAnalyzer):
    """泰尔指数 (Theil T index) + 群组分解。

    衡量不平等/差距程度：
      T = (1/N) * sum((yi/m) * log(yi/m))

    群组分解将总差距拆分为组间差距 + 组内差距：
      T_total = T_between + T_within

    Args:
        value_col: 指标列（如人均GDP、千人床位数）。
        group_col: 群组列（如 "region"，将省份分为东/中/西/东北）。

    Examples:
        >>> ti = TheilIndex()
        >>> r = ti.analyze(df, value_col="gdp_per_capita",
        ...                group_col="region")
        >>> r.data  # 含 total/between/within 三行
    """

    name = "theil_index"
    category = "cross_domain"
    description = "泰尔指数: 衡量不平等/差距 + 群组分解(组间+组内)"
    output_type = "dataframe"
    executor = "python"
    timeout = 15

    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str = "value",
        group_col: str | None = None,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [value_col])

        df = data[[value_col]].copy()
        if group_col:
            _check_cols(data, [group_col])
            df[group_col] = data[group_col]

        y = df[value_col].values.astype(float)
        y = y[y > 0]
        if len(y) < 2:
            return AnalysisResult(success=False, error="有效数据不足")

        n = len(y)
        mean_y = y.mean()

        # 总泰尔指数
        T_total = np.sum((y / mean_y) * np.log(y / mean_y)) / n

        results = [{"level": "total", "theil": T_total, "share": 1.0}]

        if group_col:
            df_valid = df[df[value_col] > 0].copy()
            groups = df_valid.groupby(group_col)

            # 组间差距
            group_means = groups[value_col].mean()
            group_sizes = groups.size()
            n_total = group_sizes.sum()
            overall_mean = df_valid[value_col].mean()

            T_between = np.sum(
                (group_sizes / n_total) * (group_means / overall_mean) *
                np.log(group_means / overall_mean)
            )

            # 组内差距（加权和）
            T_within = 0.0
            for g_name, g_df in groups:
                g_y = g_df[value_col].values
                g_n = len(g_y)
                g_mean = g_y.mean()
                if g_mean > 0 and g_n > 1:
                    g_T = np.sum((g_y / g_mean) * np.log(g_y / g_mean)) / g_n
                    T_within += (g_n / n_total) * g_T

            results.append({"level": "between_groups", "theil": T_between, "share": T_between / T_total if T_total > 0 else 0})
            results.append({"level": "within_groups", "theil": T_within, "share": T_within / T_total if T_total > 0 else 0})

            # 每组内部泰尔指数
            for g_name, g_df in groups:
                g_y = g_df[value_col].values
                g_n = len(g_y)
                g_mean = g_y.mean()
                if g_mean > 0 and g_n > 1:
                    g_T = np.sum((g_y / g_mean) * np.log(g_y / g_mean)) / g_n
                    results.append({
                        "level": f"within_{g_name}",
                        "theil": g_T,
                        "share": g_T / T_total if T_total > 0 else 0,
                    })

        result = pd.DataFrame(results)

        return AnalysisResult(
            success=True,
            data=result,
            summary={
                "algorithm": "theil_index",
                "n_observations": n,
                "total_theil": float(T_total),
                "has_decomposition": group_col is not None,
            },
        )


# =============================================================================
# TripleValidation —— 三级验证链
# =============================================================================

class TripleValidation(BaseAnalyzer):
    """三级验证链（弹性 → 回归 → Granger）。

    综合三种方法交叉验证 X 对 Y 的影响：
    一级（弹性）：ε = %ΔY / %ΔX，简单直观
    二级（回归）：面板 OLS 控制混杂因素
    三级（因果）：Granger 因果检验时间先后顺序

    Args:
        x_col: 自变量/原因变量。
        y_col: 因变量/结果变量。
        entity_col: 实体标识列（PanelOLS 需要）。
        time_col: 时间列。
        control_cols: 控制变量（PanelOLS 需要）。

    Examples:
        >>> tv = TripleValidation()
        >>> r = tv.analyze(df, x_col="edu_spending", y_col="gdp",
        ...                entity_col="province", time_col="year")
        >>> r.summary["consistency"]
    """

    name = "triple_validation"
    category = "cross_domain"
    description = "三级验证链: 弹性系数 + 面板 OLS + Granger 因果交叉验证"
    output_type = "dataframe"
    executor = "python"
    timeout = 60

    def analyze(
        self,
        data: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        entity_col: str | None = None,
        time_col: str | None = None,
        control_cols: list[str] | None = None,
    ) -> AnalysisResult:
        self.validate(data)
        _check_cols(data, [x_col, y_col])

        # 一级：弹性系数
        el = Elasticity()
        el_result = el.analyze(
            data, y_col=y_col, x_col=x_col,
            time_col=time_col, group_by=[entity_col] if entity_col else None,
        )
        mean_elasticity = float(el_result.data["elasticity"].mean()) if el_result.success else np.nan

        # 二级：面板 OLS
        indep = [x_col] + (control_cols or [])
        pols = PanelOLS()
        pols_result = pols.analyze(
            data, dep_col=y_col, indep_cols=indep,
            entity_col=entity_col, time_col=time_col,
        )
        x_row = pols_result.data[pols_result.data["variable"] == x_col]
        if not x_row.empty:
            x_coef = float(x_row["coefficient"].values[0])
            x_pval = float(x_row["p_value"].values[0])
            x_beta = float(x_row["beta"].values[0]) if "beta" in x_row.columns else np.nan
            regression_sig = bool(x_pval < 0.05)
        else:
            x_coef = np.nan
            x_pval = np.nan
            x_beta = np.nan
            regression_sig = False

        r_squared = float(pols_result.summary.get("r_squared", np.nan))

        # 三级：Granger 因果
        gc = GrangerCausality()
        gc_result = gc.analyze(
            data, x_col=x_col, y_col=y_col,
            time_col=time_col, max_lag=4,
        )
        granger_causal = bool(gc_result.data["causal"].values[0]) if not gc_result.data.empty else False
        granger_p = float(gc_result.data["p_value"].values[0]) if not gc_result.data.empty else np.nan

        # 一致性判断
        signs_agree = (mean_elasticity > 0) == (x_coef > 0) if not (np.isnan(mean_elasticity) or np.isnan(x_coef)) else None
        consistency = sum([
            bool(signs_agree) if signs_agree is not None else False,
            regression_sig,
            granger_causal,
        ])

        if consistency >= 2:
            consistency_label = "高度一致"
        elif consistency >= 1:
            consistency_label = "部分一致"
        else:
            consistency_label = "不一致"

        verdict = f"X→Y 关系: {'支持' if consistency >= 2 else '弱支持' if consistency >= 1 else '不支持'}"

        result = pd.DataFrame([{
            "level": "elasticity",
            "method": "弹性系数",
            "value": mean_elasticity,
            "interpretation": f"X 每变化 1%, Y 变化 {mean_elasticity:.3f}%" if not np.isnan(mean_elasticity) else "无法计算",
        }, {
            "level": "regression",
            "method": f"面板 OLS (R²={r_squared:.3f})",
            "value": x_coef,
            "interpretation": f"系数={x_coef:.4f}, p={x_pval:.4f}, {'显著' if regression_sig else '不显著'}",
        }, {
            "level": "granger",
            "method": "Granger 因果",
            "value": granger_p,
            "interpretation": f"p={granger_p:.4f}, {'存在因果' if granger_causal else '未发现因果'}",
        }, {
            "level": "conclusion",
            "method": "综合结论",
            "value": consistency,
            "interpretation": f"{consistency_label}: {verdict}",
        }])

        return AnalysisResult(
            success=True,
            data=result,
            summary={
                "algorithm": "triple_validation",
                "elasticity": mean_elasticity,
                "regression_coef": x_coef,
                "regression_p": x_pval,
                "granger_p": granger_p,
                "granger_causal": granger_causal,
                "consistency_score": consistency,
                "consistency_label": consistency_label,
                "verdict": verdict,
            },
        )


# =============================================================================
# 内部工具函数
# =============================================================================

def _check_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"数据中缺少必需的列: {missing}")


def _classify_decoupling(
    df_val: float, env_change: float, econ_change: float
) -> str:
    """根据 OECD 框架分类脱钩状态。"""
    if np.isnan(df_val):
        return "无法判断"
    env_up = env_change > 0
    econ_up = econ_change > 0
    df_pos = df_val > 0
    if econ_up and not env_up:
        return "强脱钩"
    if econ_up and env_up and df_pos and df_val < 1:
        return "弱脱钩"
    if not econ_up and not env_up and df_pos and df_val < 1:
        return "衰退脱钩"
    if not econ_up and not env_up:
        return "衰退耦合"
    if econ_up and env_up and not df_pos:
        return "扩张负脱钩"
    if econ_up and env_up and df_pos and df_val >= 1:
        return "强负脱钩"
    if not econ_up and env_up:
        return "强负脱钩"
    return "扩张耦合"


# =============================================================================
# __all__
# =============================================================================

__all__ = [
    "CCD",
    "DEA",
    "SupplyDemandResidual",
    "DecouplingIndex",
    "Elasticity",
    "PanelOLS",
    "GrangerCausality",
    "KMeansClustering",
    "HierarchicalClustering",
    "TheilIndex",
    "TripleValidation",
]
