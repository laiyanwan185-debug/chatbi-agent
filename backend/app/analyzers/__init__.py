"""分析算法库。

所有算法类通过 __all__ 导出，注册中心自动发现。
"""

from .base import AnalysisResult, BaseAnalyzer
from .anomaly import DualFusion, IQR, ThreeSigma, YoYChange, ZScore
from .composite import EntropyWeight, MinMax, PCA, TOPSIS, TwoLevelRank
from .correlation import DetrendResidual, MutualInfo, Pearson, Spearman, TLCC
from .cross_domain import (CCD, DEA, DecouplingIndex, Elasticity, GrangerCausality,
                            HierarchicalClustering, KMeansClustering, PanelOLS,
                            SupplyDemandResidual, TheilIndex, TripleValidation)
from .multi_dim import CV, Cube, HierarchicalAgg, Proportion
from .ranking import BenchmarkCompare, NTile, PercentRank, Rank, RankDisparity
from .time_series import CAGR, CMA, Detrend, LinearTrend, QoQ, SMA, Seasonal, YoY

__all__ = [
    # base
    "AnalysisResult",
    "BaseAnalyzer",
    # time_series
    "CAGR", "YoY", "QoQ", "SMA", "CMA", "LinearTrend", "Detrend", "Seasonal",
    # correlation
    "Pearson", "Spearman", "TLCC", "MutualInfo", "DetrendResidual",
    # ranking
    "Rank", "PercentRank", "NTile", "BenchmarkCompare", "RankDisparity",
    # anomaly
    "ZScore", "IQR", "ThreeSigma", "YoYChange", "DualFusion",
    # multi_dim
    "Cube", "Proportion", "CV", "HierarchicalAgg",
    # composite
    "MinMax", "EntropyWeight", "TOPSIS", "PCA", "TwoLevelRank",
    # cross_domain
    "CCD", "DEA", "SupplyDemandResidual", "DecouplingIndex", "Elasticity",
    "PanelOLS", "GrangerCausality", "KMeansClustering", "HierarchicalClustering",
    "TheilIndex", "TripleValidation",
]
