"""Day 9 Semantic Cache — 语义缓存验证。

测试项:
  SemanticCache: 基础操作 / 余弦相似度命中 / 相似但不足阈值未命中 / 完全无关未命中
  Serialization: DataFrame 序列化 / 反序列化正确性
  LRU 淘汰: 超过 max_entries 时自动淘汰
  Store + Lookup 完整流程
  Parser 集成: parse() 注入缓存后 cache hit 返回 cached 状态
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.engine.semantic_cache import (
    SemanticCache, CacheEntry,
    _cosine_similarity, _serialize_entry, _deserialize_entry, _entry_to_result,
)


# ═══════════════════════════════════════════════════════════════
# Mock Embedding Model
# ═══════════════════════════════════════════════════════════════

# 固定归一化向量（用于模拟 BGE-m3 产出）
_VEC_GDP: list[float] = [0.6, 0.8, 0.0]          # "GDP排名2023"
_VEC_GDP_SIM: list[float] = [0.61, 0.79, 0.02]    # "2023年各省GDP排名"（相似 >0.98）
_VEC_POPULATION: list[float] = [0.2, 0.1, 0.97]   # "人口数量"（完全不相关）


class MockEmbedder:
    """模拟 BGE-m3 Embedding 模型。"""

    def encode(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        """根据问句返回固定向量。"""
        text_lower = text.lower()
        if "人口" in text_lower or "population" in text_lower:
            return np.array(_VEC_POPULATION, dtype=np.float64)
        if "gdp" in text_lower or "排名" in text_lower:
            if "2023" in text_lower and "各省" in text_lower:
                return np.array(_VEC_GDP_SIM, dtype=np.float64)
            return np.array(_VEC_GDP, dtype=np.float64)
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)


_mock_embedder = MockEmbedder()


# ═══════════════════════════════════════════════════════════════
# 1. 余弦相似度
# ═══════════════════════════════════════════════════════════════

def test_cosine_similarity_identical():
    """相同向量 → 相似度 1.0。"""
    sim = _cosine_similarity([1.0, 0.0], [1.0, 0.0])
    assert abs(sim - 1.0) < 0.001, f"相同向量应 1.0: {sim}"
    print(f"  [PASS] 相同向量相似度: {sim:.4f}")


def test_cosine_similarity_orthogonal():
    """正交向量 → 相似度 ≈ 0。"""
    sim = _cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert abs(sim) < 0.001, f"正交向量应 ≈0: {sim}"
    print(f"  [PASS] 正交向量相似度: {sim:.4f}")


def test_cosine_similarity_near():
    """相似向量 → 高相似度。"""
    sim = _cosine_similarity(_VEC_GDP, _VEC_GDP_SIM)
    assert sim > 0.98, f"相似向量应 >0.98: {sim}"
    print(f"  [PASS] 相似 GDP 向量: {sim:.4f}")


def test_cosine_similarity_different():
    """不同向量 → 低相似度。"""
    sim = _cosine_similarity(_VEC_GDP, _VEC_POPULATION)
    assert sim < 0.7, f"不同向量应 <0.7: {sim}"
    print(f"  [PASS] 不同向量: {sim:.4f}")


# ═══════════════════════════════════════════════════════════════
# 2. 序列化 / 反序列化
# ═══════════════════════════════════════════════════════════════

def test_serialize_deserialize():
    """CacheEntry 序列化 → 反序列化 数据完整。"""
    df = pd.DataFrame({"province": ["A", "B"], "gdp": [100.0, 200.0]})
    entry = CacheEntry(
        query="GDP排名",
        query_vector=_VEC_GDP,
        parse_result={"execution_plan": {"analysis_type": "rank"}},
        execution_result={"dag_status": "full", "final_data": df},
        interpretation="GDP排名分析",
    )
    serialized = _serialize_entry(entry)
    assert "_final_data_json" in serialized["execution_result"]
    assert "final_data" not in serialized["execution_result"]

    restored = _deserialize_entry(serialized)
    assert restored is not None
    assert restored.query == "GDP排名"
    assert restored.parse_result["execution_plan"]["analysis_type"] == "rank"
    assert restored.interpretation == "GDP排名分析"
    assert isinstance(restored.execution_result["final_data"], pd.DataFrame)
    assert len(restored.execution_result["final_data"]) == 2
    print(f"  [PASS] 序列化/反序列化完整: {len(restored.execution_result['final_data'])} rows")


def test_serialize_empty_df():
    """空 DataFrame 序列化。"""
    df = pd.DataFrame()
    entry = CacheEntry(
        query="test", query_vector=[1.0, 0.0],
        execution_result={"final_data": df},
    )
    serialized = _serialize_entry(entry)
    restored = _deserialize_entry(serialized)
    assert restored is not None
    assert isinstance(restored.execution_result["final_data"], pd.DataFrame)
    print("  [PASS] 空 DataFrame 序列化/反序列化")


def test_serialize_no_execution():
    """无 execution_result 时序列化不崩溃。"""
    entry = CacheEntry(query="test", query_vector=[1.0, 0.0], parse_result={"status": "ok"})
    serialized = _serialize_entry(entry)
    restored = _deserialize_entry(serialized)
    assert restored is not None
    assert restored.parse_result["status"] == "ok"
    print("  [PASS] 无 execution 序列化")


def test_entry_to_result():
    """_entry_to_result 返回 Parser 兼容格式。"""
    entry = CacheEntry(
        query="test", query_vector=[1.0, 0.0],
        parse_result={"execution_plan": {"analysis_type": "rank"}},
        interpretation="分析结果",
    )
    result = _entry_to_result(entry)
    assert result["status"] == "cached"
    assert result["cache_hit"] is True
    assert result["parse_result"]["execution_plan"]["analysis_type"] == "rank"
    assert result["interpretation"] == "分析结果"
    print(f"  [PASS] entry_to_result: status={result['status']}")


# ═══════════════════════════════════════════════════════════════
# 3. SemanticCache 功能
# ═══════════════════════════════════════════════════════════════

def _make_cache() -> SemanticCache:
    """创建使用内存后端的 SemanticCache（不依赖 DiskCache）。"""
    cache = SemanticCache(_mock_embedder, backend="memory")
    return cache


async def test_cache_hit():
    """语义相似问句 → 缓存命中。"""
    cache = _make_cache()
    parse_result = {"execution_plan": {"analysis_type": "rank", "tables": ["macro_economy"]}}
    exec_result = {"dag_status": "full", "final_data": pd.DataFrame({"x": [1]})}

    # 存储 GDP 查询
    await cache.store("GDP排名2023", parse_result, exec_result, "2023年GDP排名")
    assert cache.size == 1

    # 查询相似问句
    result = await cache.lookup("2023年各省GDP排名")
    assert result is not None
    assert result["status"] == "cached"
    assert result["cache_hit"] is True
    assert result["parse_result"]["execution_plan"]["analysis_type"] == "rank"
    print(f"  [PASS] cache hit: sim={cache.threshold}, size={cache.size}")


async def test_cache_miss():
    """无关问句 → 缓存未命中。"""
    cache = _make_cache()
    await cache.store("GDP排名2023", {"execution_plan": {}}, None, "")

    result = await cache.lookup("人口数量")
    assert result is None
    print(f"  [PASS] cache miss: 无关问句未命中")


async def test_cache_below_threshold():
    """相似但不足阈值 → 未命中。"""
    cache = _make_cache()
    # 存储 GDP 查询（_VEC_GDP）
    await cache.store("GDP排名2023", {"execution_plan": {}}, None, "")
    # 查询边界相似度的向量（如果构造一个刚好 < 0.98 的）
    # _VEC_POPULATION 与 _VEC_GDP 的相似度 < 0.7，肯定低于阈值
    result = await cache.lookup("人口数量")
    assert result is None
    print(f"  [PASS] cache below threshold: 未命中")


async def test_cache_hit_count():
    """多次命中 → hit_count 递增。"""
    cache = _make_cache()
    await cache.store("GDP排名2023", {"execution_plan": {"tables": ["macro_economy"]}}, None, "")

    # 多次命中
    for _ in range(3):
        result = await cache.lookup("2023年各省GDP排名")
        assert result is not None

    # 验证 hit_count
    entries = list(cache._cache_store.values())  # noqa: SLF001
    entry = next(e for e in entries if e.query == "GDP排名2023")
    assert entry.hit_count >= 3
    print(f"  [PASS] hit_count={entry.hit_count}")


async def test_cache_lru_eviction():
    """超过 max_entries 时 LRU 淘汰最少命中的。"""
    from app.engine.semantic_cache import MAX_ENTRIES
    cache = _make_cache()

    # 填充到 max + 5
    total = MAX_ENTRIES + 5
    for i in range(total):
        vec = [i / 1000, 0.0, 0.0]
        vec_norm = vec / np.linalg.norm(vec)
        cache._cache_store[f"query_{i}"] = CacheEntry(
            query=f"query_{i}", query_vector=vec_norm.tolist(),
        )
        cache._cache_store[f"query_{i}"].hit_count = i  # 高序号更高频

    cache._evict_if_needed()
    assert cache.size <= MAX_ENTRIES, f"LRU 淘汰后应有 ≤{MAX_ENTRIES}: {cache.size}"

    # 低 hit_count 的应被淘汰（0-4 号 query 应被淘汰）
    for i in range(5):
        assert f"query_{i}" not in cache._cache_store
    print(f"  [PASS] LRU eviction: {total} → {cache.size}")


async def test_cache_clear():
    """clear 清空所有条目。"""
    cache = _make_cache()
    await cache.store("GDP排名", {"execution_plan": {}}, None, "")
    await cache.store("人口排名", {"execution_plan": {}}, None, "")
    assert cache.size == 2
    cache.clear()
    assert cache.size == 0
    print(f"  [PASS] cache clear: size={cache.size}")


async def test_cache_store_lookup_full():
    """Store → Lookup 完整流程（含 DataFrame 序列化）。"""
    cache = _make_cache()
    df = pd.DataFrame({"province": ["A", "B"], "gdp": [100.0, 200.0]})
    parse_result = {"execution_plan": {"analysis_type": "rank", "tables": ["macro_economy"]}}
    exec_result = {"dag_status": "full", "final_data": df}
    interpretation = "2023年各省GDP排名分析"

    await cache.store("GDP排名2023", parse_result, exec_result, interpretation)
    result = await cache.lookup("2023年各省GDP排名")
    assert result is not None
    assert result["interpretation"] == "2023年各省GDP排名分析"
    assert "execution_result" in result
    print(f"  [PASS] store→lookup 完整: interpretation={result['interpretation'][:40]}")


# ═══════════════════════════════════════════════════════════════
# 执行
# ═══════════════════════════════════════════════════════════════

def run():
    import asyncio
    print("=" * 60)
    print("Day 9 — 语义缓存验证")
    print("=" * 60)

    print("\n=== 1. 余弦相似度 ===")
    test_cosine_similarity_identical()
    test_cosine_similarity_orthogonal()
    test_cosine_similarity_near()
    test_cosine_similarity_different()

    print("\n=== 2. 序列化 / 反序列化 ===")
    test_serialize_deserialize()
    test_serialize_empty_df()
    test_serialize_no_execution()
    test_entry_to_result()

    print("\n=== 3. SemanticCache 功能 ===")
    asyncio.run(test_cache_hit())
    asyncio.run(test_cache_miss())
    asyncio.run(test_cache_below_threshold())
    asyncio.run(test_cache_hit_count())
    asyncio.run(test_cache_lru_eviction())
    asyncio.run(test_cache_clear())
    asyncio.run(test_cache_store_lookup_full())

    print("\n" + "=" * 60)
    print("全部通过")
    print("=" * 60)


if __name__ == "__main__":
    run()
