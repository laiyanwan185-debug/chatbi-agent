"""算法注册中心。

自动发现所有 BaseAnalyzer 子类，提供名称/类别查询。
应用启动时执行 registry.auto_discover()（需先 import app.analyzers）。
"""

from __future__ import annotations

import logging
from typing import Any

from app.analyzers.base import BaseAnalyzer

logger = logging.getLogger(__name__)


class RegistryError(Exception):
    """注册中心相关错误。"""


class AnalyzerRegistry:
    """算法注册中心。

    通过 BaseAnalyzer.__subclasses__() 自动发现所有继承子类，
    按 name（唯一标识）和 category（类别）索引。
    """

    def __init__(self) -> None:
        self._name_map: dict[str, type[BaseAnalyzer]] = {}
        self._category_map: dict[str, list[type[BaseAnalyzer]]] = {}

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------

    def register(self, analyzer_cls: type[BaseAnalyzer]) -> bool:
        """注册一个算法类。校验通过后加入索引。"""
        if not isinstance(analyzer_cls, type) or not issubclass(analyzer_cls, BaseAnalyzer):
            raise RegistryError(f"{analyzer_cls} 不是 BaseAnalyzer 的子类")

        name: str = getattr(analyzer_cls, "name", "")
        if not name:
            raise RegistryError(f"{analyzer_cls.__name__}.name 为空，跳过注册")

        if analyzer_cls.analyze is BaseAnalyzer.analyze:
            raise RegistryError(f"{analyzer_cls.__name__} 未重写 analyze() 方法")

        key = name.lower()
        if key in self._name_map:
            logger.warning("重复的算法名称 '%s' (类 %s)，跳过", name, analyzer_cls.__name__)
            return False

        self._name_map[key] = analyzer_cls
        cat = analyzer_cls.category or "_uncategorized"
        self._category_map.setdefault(cat, []).append(analyzer_cls)
        return True

    def auto_discover(self) -> int:
        """自动发现所有 BaseAnalyzer 子类并注册。

        需要在调用前确保 ``import app.analyzers`` 已执行。
        """
        count = 0
        for cls in BaseAnalyzer.__subclasses__():
            try:
                if self.register(cls):
                    count += 1
            except RegistryError as e:
                logger.warning("跳过 %s: %s", cls.__name__, e)
        return count

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get(self, name: str) -> type[BaseAnalyzer]:
        """按名称获取算法类。不存在时抛出 RegistryError。"""
        cls = self._name_map.get(name.lower())
        if cls is None:
            raise RegistryError(f"未注册的算法名称: '{name}'")
        return cls

    def get_by_category(self, category: str) -> list[type[BaseAnalyzer]]:
        """按类别获取算法类列表。未知类别返回空列表。"""
        return list(self._category_map.get(category, []))

    def list(self) -> list[dict[str, Any]]:
        """返回所有已注册算法的元数据摘要。"""
        return [
            {
                "name": cls.name,
                "category": cls.category,
                "description": cls.description,
                "output_type": cls.output_type,
                "executor": cls.executor,
                "timeout": cls.timeout,
            }
            for cls in self._name_map.values()
        ]

    @property
    def categories(self) -> list[str]:
        """返回所有已注册的类别名（排序）。"""
        return sorted(self._category_map.keys())

    @property
    def size(self) -> int:
        """已注册算法总数。"""
        return len(self._name_map)


# 模块级单例
registry = AnalyzerRegistry()
