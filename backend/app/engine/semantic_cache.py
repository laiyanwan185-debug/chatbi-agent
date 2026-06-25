"""语义缓存 — DiskCache/Redis 双后端 + 余弦相似度检索 + LRU 淘汰。

职责边界：
  - SemanticCache: 问句 Embedding → 余弦相似度 → 历史结果命中
  - 键：问句 Embedding 向量，值：序列化 Parser + Executor + Interpreter 结果
  - Cosine > 0.98 命中，LRU 淘汰（max 1000 条）
  - 注入 Parser 流程：Cache Hit → 直接返回历史结果（0.1s + 零 Token）

调用方：parser.py 的 parse() 入口。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================

DEFAULT_CACHE_DIR = "_cache/semantic"

# 缓存条目最大数（LRU 淘汰上限）
MAX_ENTRIES: int = 1000


# =============================================================================
# CacheEntry — 缓存条目数据结构
# =============================================================================


@dataclass
class CacheEntry:
    """单条缓存记录。

    Attributes:
        query:                原始问句。
        query_vector:         问句 Embedding 向量（归一化后，用于余弦相似度）。
        parse_result:         Parser 输出 dict。
        execution_result:     Executor 输出 dict（含 final_data 的 JSON 序列化）。
        interpretation:       解读文本。
        cached_at:            缓存时间戳。
        hit_count:            命中次数（用于缓存热度和 LRU 辅助决策）。
    """
    query: str
    query_vector: list[float]
    parse_result: dict[str, Any] = field(default_factory=dict)
    execution_result: dict[str, Any] | None = None
    interpretation: str = ""
    cached_at: float = 0.0
    hit_count: int = 0


# =============================================================================
# SemanticCache
# =============================================================================


class SemanticCache:
    """语义缓存引擎。

    使用方式：
        cache = SemanticCache(embedding_model)
        result = await cache.lookup("2023年GDP排名")
        if result is None:
            result = await full_pipeline(query)
            await cache.store(query, parse_result, exec_result, interpretation)
    """

    def __init__(self, embedding_model: Any, backend: str = "diskcache") -> None:
        """初始化语义缓存。

        Args:
            embedding_model: Embedding 模型实例，需有 .encode(text) → np.ndarray 方法。
            backend:         后端类型（"diskcache" / "redis"）。
        """
        self._embedder = embedding_model
        self._threshold = settings.CACHE_SIMILARITY_THRESHOLD
        self._max_entries = MAX_ENTRIES
        self._backend = backend.lower()
        self._cache_store: dict[str, CacheEntry] = {}  # fallback in-memory
        self._backend_ready = False

        self._init_backend()

    # ── 初始化 ──

    def _init_backend(self) -> None:
        """初始化后端存储。"""
        if self._backend == "diskcache":
            self._init_diskcache()
        elif self._backend == "redis":
            logger.warning("Redis 后端未完全实现，降级到内存存储")
            self._backend = "memory"
        else:
            logger.info("语义缓存使用内存存储（backend=%s）", self._backend)

    def _init_diskcache(self) -> None:
        """初始化 DiskCache 后端。"""
        try:
            from diskcache import Cache

            self._cache = Cache(DEFAULT_CACHE_DIR, size_limit=500 * 1024 * 1024)
            self._key_list: list[str] = list(self._cache.iterkeys())
            self._backend_ready = True
            logger.info("语义缓存 DiskCache 就绪: %s (%d 条)", DEFAULT_CACHE_DIR, len(self._key_list))
        except ImportError:
            logger.warning("diskcache 未安装，降级到内存存储")
            self._backend = "memory"
        except Exception as exc:
            logger.warning("DiskCache 初始化异常: %s，降级到内存存储", exc)
            self._backend = "memory"

    # ── 核心操作 ──

    async def compute_embedding(self, query: str) -> list[float]:
        """计算问句的归一化 Embedding 向量。"""
        vec = self._embedder.encode(query, normalize_embeddings=True)
        if isinstance(vec, np.ndarray):
            return vec.tolist()
        return list(vec)

    async def lookup(self, query: str) -> dict[str, Any] | None:
        """查询语义缓存。

        计算问句 Embedding，在缓存中搜索余弦相似度 > threshold 的最佳匹配。

        Args:
            query: 用户问句。

        Returns:
            命中时返回反序列化的完整结果字典（含 parse_result / execution_result / interpretation），
            未命中返回 None。
        """
        query_vec = await self.compute_embedding(query)
        best_key, best_sim = None, 0.0

        if self._backend == "memory":
            for key, entry in self._cache_store.items():
                sim = _cosine_similarity(query_vec, entry.query_vector)
                if sim > best_sim:
                    best_sim = sim
                    best_key = key

        elif self._backend == "diskcache" and self._backend_ready:
            for key in self._key_list:
                try:
                    entry_data = self._cache.get(key)
                except Exception:
                    continue
                if entry_data is None:
                    continue
                entry = _deserialize_entry(entry_data)
                if entry is None:
                    continue
                sim = _cosine_similarity(query_vec, entry.query_vector)
                if sim > best_sim:
                    best_sim = sim
                    best_key = key

        if best_sim >= self._threshold and best_key is not None:
            entry = self._get_entry(best_key)
            if entry is not None:
                entry.hit_count += 1
                self._put_entry(best_key, entry)
                logger.info(
                    "语义缓存 HIT: sim=%.4f, query='%s' → cached='%s'",
                    best_sim, query[:50], entry.query[:50],
                )
                return _entry_to_result(entry)

        logger.debug("语义缓存 MISS: best_sim=%.4f (threshold=%.2f)", best_sim, self._threshold)
        return None

    async def store(
        self,
        query: str,
        parse_result: dict[str, Any],
        execution_result: dict[str, Any] | None = None,
        interpretation: str = "",
    ) -> None:
        """将问句及完整结果存入缓存。

        Args:
            query:            用户问句。
            parse_result:     Parser 输出。
            execution_result: Executor 输出（含 final_data）。
            interpretation:   解读文本。
        """
        query_vec = await self.compute_embedding(query)
        entry = CacheEntry(
            query=query,
            query_vector=query_vec,
            parse_result=parse_result,
            execution_result=execution_result,
            interpretation=interpretation,
            cached_at=time.time(),
        )
        self._put_entry(query, entry)
        logger.info("语义缓存 STORE: query='%s'", query[:50])

    # ── 内部工具 ──

    def _get_entry(self, key: str) -> CacheEntry | None:
        """从后端获取缓存条目。"""
        if self._backend == "memory":
            return self._cache_store.get(key)
        if self._backend == "diskcache" and self._backend_ready:
            try:
                data = self._cache.get(key)
                return _deserialize_entry(data) if data else None
            except Exception:
                return None
        return self._cache_store.get(key)

    def _put_entry(self, key: str, entry: CacheEntry) -> None:
        """写入缓存条目，超过 max 时 LRU 淘汰。"""
        if self._backend == "memory":
            self._cache_store[key] = entry
            self._evict_if_needed()
        elif self._backend == "diskcache" and self._backend_ready:
            try:
                self._cache[key] = _serialize_entry(entry)
            except Exception as exc:
                logger.warning("DiskCache 写入失败: %s", exc)
            # 重建 key list
            self._key_list = list(self._cache.iterkeys())
            self._evict_diskcache()

    def _evict_if_needed(self) -> None:
        """内存存储 LRU 淘汰：按 hit_count 排序，淘汰最少命中的。"""
        if len(self._cache_store) <= self._max_entries:
            return
        sorted_keys = sorted(
            self._cache_store.keys(),
            key=lambda k: self._cache_store[k].hit_count,
        )
        evict_count = len(self._cache_store) - self._max_entries
        for key in sorted_keys[:evict_count]:
            del self._cache_store[key]
        logger.debug("语义缓存 LRU 淘汰: %d 条", evict_count)

    def _evict_diskcache(self) -> None:
        """DiskCache LRU 淘汰：按 hit_count 升序淘汰最少命中的。"""
        if not self._backend_ready:
            return
        try:
            count = len(self._key_list)
            if count <= self._max_entries:
                return
            to_remove = count - self._max_entries

            # 读取每条记录的 hit_count + cached_at，按 (hit_count, cached_at) 排序
            scored: list[tuple[int, float, str]] = []
            fallback_keys: list[str] = []
            for key in self._key_list:
                try:
                    data = self._cache.get(key)
                    if data is None:
                        fallback_keys.append(key)
                        continue
                    entry = _deserialize_entry(data)
                    if entry is None:
                        fallback_keys.append(key)
                        continue
                    scored.append((entry.hit_count, entry.cached_at, key))
                except Exception:
                    fallback_keys.append(key)

            # 先删读取失败的（最不可靠的条目优先淘汰）
            for key in fallback_keys[:to_remove]:
                try:
                    del self._cache[key]
                except Exception:
                    pass
            remaining = to_remove - len(fallback_keys)
            if remaining > 0 and scored:
                scored.sort(key=lambda x: (x[0], x[1]))  # hit_count 升序，cached_at 升序决胜
                for _, _, key in scored[:remaining]:
                    try:
                        del self._cache[key]
                    except Exception:
                        pass

            self._key_list = list(self._cache.iterkeys())
            logger.debug("语义缓存 LRU 淘汰(DiskCache): %d 条 (fallback=%d)",
                         to_remove, len(fallback_keys))
        except Exception:
            pass

    def clear(self) -> None:
        """清空缓存。"""
        if self._backend == "memory":
            self._cache_store.clear()
        elif self._backend == "diskcache" and self._backend_ready:
            try:
                self._cache.clear()
                self._key_list = []
                logger.info("语义缓存已清空(DiskCache)")
            except Exception as exc:
                logger.warning("DiskCache 清空失败: %s", exc)
        logger.info("语义缓存已清空")

    @property
    def size(self) -> int:
        """当前缓存条目数。"""
        if self._backend == "memory":
            return len(self._cache_store)
        if self._backend == "diskcache" and self._backend_ready:
            return len(self._key_list)
        return len(self._cache_store)

    @property
    def threshold(self) -> float:
        return self._threshold


# =============================================================================
# 序列化 / 反序列化
# =============================================================================


def _serialize_entry(entry: CacheEntry) -> dict[str, Any]:
    """将 CacheEntry 转为可 JSON 序列化的 dict。

    DataFrane 通过 to_json() 序列化存为字符串。
    """
    d = asdict(entry)
    if d["execution_result"] and isinstance(d["execution_result"], dict):
        final_data = d["execution_result"].get("final_data")
        if final_data is not None:
            if isinstance(final_data, pd.DataFrame):
                if not final_data.empty:
                    d["execution_result"]["_final_data_json"] = final_data.to_json(
                        orient="records", date_format="iso",
                    )
                else:
                    d["execution_result"]["_final_data_json"] = "[]"
            del d["execution_result"]["final_data"]
    return d


def _deserialize_entry(data: dict[str, Any]) -> CacheEntry | None:
    """从序列化 dict 恢复 CacheEntry。

    通过 pd.read_json() 还原 DataFrame。
    """
    try:
        entry = CacheEntry(**data)
    except TypeError:
        return None

    if entry.execution_result and "_final_data_json" in entry.execution_result:
        json_str = entry.execution_result.pop("_final_data_json", None)
        if json_str and json_str != "[]":
            try:
                parsed = json.loads(json_str)
                entry.execution_result["final_data"] = pd.DataFrame(parsed)
            except Exception:
                entry.execution_result["final_data"] = pd.DataFrame()
        else:
            entry.execution_result["final_data"] = pd.DataFrame()
    else:
        entry.execution_result = entry.execution_result or {}

    return entry


def _entry_to_result(entry: CacheEntry) -> dict[str, Any]:
    """将 CacheEntry 转为 Parser.parse() 兼容的返回格式。"""
    result: dict[str, Any] = {
        "status": "cached",
        "cache_hit": True,
        "parse_result": entry.parse_result,
        "interpretation": entry.interpretation,
    }
    if entry.execution_result:
        result["execution_result"] = entry.execution_result
    return result


# =============================================================================
# 余弦相似度
# =============================================================================


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个归一化向量的余弦相似度（即点积）。"""
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    dot = float(np.dot(arr_a, arr_b))
    # 归一化后的向量 ||a|| = ||b|| = 1，dot 即为余弦值
    return max(-1.0, min(1.0, dot))
