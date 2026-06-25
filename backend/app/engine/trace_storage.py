"""Trace 持久化 — 基于 DiskCache 的追踪记录存储。

职责：
  - save(trace_id, trace_data): 管线执行完毕后写入磁盘
  - load(trace_id): GET /api/trace/{id} 时按 ID 读取
  - close(): 应用关闭时回收资源

DiskCache 自动管理数据过期和 LRU 淘汰。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from diskcache import Cache

logger = logging.getLogger(__name__)

# 默认缓存目录
DEFAULT_CACHE_DIR = str(Path("cache") / "traces")
# 每条 trace 保留 24 小时（秒）
DEFAULT_TTL = 86400
# 最大条目数
DEFAULT_MAX_ENTRIES = 5000


class TraceStorage:
    """DiskCache 封装的 Trace 持久化存储。"""

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        ttl: int = DEFAULT_TTL,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        self._cache = Cache(directory=cache_dir, size_limit=max_entries * 1024 * 10)  # ~10KB per trace
        self._ttl = ttl
        logger.info("TraceStorage initialized: dir=%s, ttl=%ds, max_entries=%d",
                     cache_dir, ttl, max_entries)

    def save(self, trace_id: str, trace_data: dict) -> None:
        """保存一条 trace 到缓存。

        Args:
            trace_id: 追踪 ID。
            trace_data: 序列化后的 trace dict（含 trace_id, steps, total_latency_ms）。
        """
        if not trace_id:
            logger.warning("TraceStorage.save: empty trace_id, skipping")
            return
        self._cache.set(trace_id, trace_data, expire=self._ttl)
        logger.debug("Trace saved: %s (%d steps, %.0fms)",
                     trace_id,
                     len(trace_data.get("steps", [])),
                     trace_data.get("total_latency_ms", 0))

    def load(self, trace_id: str) -> dict | None:
        """按 trace_id 读取 trace。

        Args:
            trace_id: 追踪 ID。

        Returns:
            trace dict，未找到或已过期返回 None。
        """
        data = self._cache.get(trace_id)
        if data is None:
            logger.debug("Trace not found (or expired): %s", trace_id)
            return None
        logger.debug("Trace loaded: %s", trace_id)
        return data

    def close(self) -> None:
        """关闭缓存，释放文件锁。"""
        try:
            self._cache.close()
            logger.info("TraceStorage closed")
        except Exception:
            logger.exception("TraceStorage close error")
