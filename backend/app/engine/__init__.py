"""Engine 模块 — 四层一反馈核心引擎。"""

import logging

logger = logging.getLogger(__name__)

_analyzers_registered = False


def ensure_analyzers_registered() -> None:
    """确保所有分析算法已注册到注册中心。

    应用启动时调用一次。
    """
    global _analyzers_registered
    if _analyzers_registered:
        return

    import app.analyzers  # noqa: F401 — 触发子类加载

    from app.engine.registry import registry

    count = registry.auto_discover()
    _analyzers_registered = True
    logger.info("分析算法注册完成: %d 个", count)


from app.engine.orchestrator import DAGOrchestrator, DAGPlan, orchestrator

__all__ = [
    "DAGOrchestrator",
    "DAGPlan",
    "orchestrator",
    "ensure_analyzers_registered",
]
