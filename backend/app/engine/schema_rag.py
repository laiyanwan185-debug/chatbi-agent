"""
Schema-RAG 引擎 — BGE-m3 语义向量检索 + 确定性指标注册表双路融合检索器
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings
from app.engine.indicator_registry import indicator_registry
from app.db.schema_discovery import SchemaDiscovery, ColumnInfo

logger = logging.getLogger(__name__)


class SchemaRAGEngine:
    """双路融合检索 RAG 引擎：指标注册表确定性匹配 + 向量语义检索。"""

    def __init__(self, schema_discovery: SchemaDiscovery) -> None:
        self._schema_discovery = schema_discovery
        self._model: SentenceTransformer | None = None
        self._table_names: list[str] = []
        self._vectors: np.ndarray | None = None

    # ── 1. 延迟加载向量模型 ──

    @property
    def model(self) -> SentenceTransformer:
        """延迟加载 BGE-m3 Embedding 模型。"""
        if self._model is None:
            logger.info(
                "Loading SentenceTransformer [%s] on %s...",
                settings.EMBEDDING_MODEL, settings.EMBEDDING_DEVICE,
            )
            self._model = SentenceTransformer(
                model_name_or_path=settings.EMBEDDING_MODEL,
                device=settings.EMBEDDING_DEVICE,
            )
        return self._model

    # ── 2. 向量索引构建 ──

    def _textify_table_schema(self, table_name: str, cols: list[ColumnInfo]) -> str:
        """将物理表结构转为富语义文本，供 Embedding 编码。"""
        registered_aliases: set[str] = set()
        for ind in indicator_registry.get_all():
            if ind.table == table_name:
                registered_aliases.add(ind.name)
                registered_aliases.update(ind.aliases)

        alias_str = (
            f" 关联业务词: {', '.join(registered_aliases)}"
            if registered_aliases else ""
        )
        col_strs = [
            f"{c.name} ({c.dtype} - {c.comment or '无注释'})"
            for c in cols
        ]
        return (
            f"物理数据表名: {table_name}.{alias_str} "
            f"物理字段结构: {'; '.join(col_strs)}."
        )

    def build_vector_index(self) -> None:
        """全量扫描 SchemaDiscovery → 构建内存向量索引。"""
        tables = self._schema_discovery.get_all_tables()
        if not tables:
            logger.warning("Schema discovery contains 0 tables. Build index aborted.")
            return

        self._table_names = []
        texts: list[str] = []

        for tname in tables:
            tbl = self._schema_discovery.get_table(tname)
            if tbl is None:
                continue
            self._table_names.append(tname)
            texts.append(self._textify_table_schema(tname, tbl.columns))

        logger.info("Indexing %d tables with BGE-m3...", len(texts))
        embeddings = self.model.encode(
            texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._vectors = np.array(embeddings)
        logger.info("Vector index built. Shape: %s", self._vectors.shape)

    # ── 3. 向量检索 ──

    def _vector_search_tables(
        self, query: str, top_k: int,
    ) -> list[tuple[str, float]]:
        """余弦相似度 Top-K 表检索。"""
        if self._vectors is None or not self._table_names:
            return []

        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self._vectors, query_vec)

        top_k = min(top_k, len(self._table_names))
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [
            (self._table_names[idx], float(scores[idx]))
            for idx in top_indices
        ]

    # ── 4. 双路融合检索 ──

    def retrieve_schema_context(
        self, query: str, top_k: int | None = None,
    ) -> tuple[str, list[str]]:
        """双路融合检索：指标注册表 + 向量语义 → 结构化 Prompt。

        Returns:
            (格式化 Prompt 字符串, 最终选中的表名列表)
        """
        top_k = top_k or settings.SCHEMA_TOP_K

        # ── 第一路：指标注册表 ──
        indicator_tables: set[str] = set()
        matched_indicators: list[Any] = []

        for ind in indicator_registry.get_all():
            if ind.name in query or any(a in query.lower() for a in ind.aliases):
                matched_indicators.append(ind)
                if ind.table:
                    indicator_tables.add(ind.table)
                for mapping in ind.field_mappings.values():
                    indicator_tables.add(mapping["table"])

        # ── 第二路：向量检索 ──
        vector_results = self._vector_search_tables(query, top_k=top_k)
        vector_tables = [t for t, _ in vector_results]

        # ── 融合 ──
        final_tables = list(indicator_tables | set(vector_tables))

        logger.info(
            "Dual-RAG -> Rule: %s | Vector: %s | Final: %s",
            list(indicator_tables), vector_tables, final_tables,
        )

        # ── 拼装 Prompt Context ──
        lines: list[str] = [
            "## 【当前查询涉及的数据库物理结构（必看约束）】\n"
        ]

        # 1. 指标映射约束
        if matched_indicators:
            lines.append("### 1. 业务指标 → 物理字段强约束:")
            for ind in matched_indicators:
                if ind.type == "direct":
                    lines.append(
                        f"  - [{ind.name}] → 必须用 {ind.table}.{ind.field}"
                    )
                else:
                    mapping_desc = "; ".join(
                        f"变量 '{v}' → {m['table']}.{m['column']}"
                        for v, m in ind.field_mappings.items()
                    )
                    lines.append(
                        f"  - [{ind.name}] → 公式 `{ind.formula}`，{mapping_desc}"
                    )
            lines.append("")

        # 2. 物理表结构
        lines.append("### 2. 物理表结构定义:")
        for tname in final_tables:
            tbl = self._schema_discovery.get_table(tname)
            if tbl is None:
                continue

            lines.append(f"#### {tname} ({tbl.comment or '无注释'})")
            for col in tbl.columns:
                pk_tag = " [PK]" if col.is_pk else ""
                null_tag = " [NN]" if not col.nullable else ""
                cmt = f" | {col.comment}" if col.comment else ""
                lines.append(f"  - {col.name} ({col.dtype}){pk_tag}{null_tag}{cmt}")
            for fk in tbl.foreign_keys:
                lines.append(
                    f"  - FK: {fk.column} → {fk.ref_table}.{fk.ref_column}"
                )
            lines.append("")

        # 3. 溯源说明
        lines.append("### 3. 检索溯源:")
        lines.append(f"  - 指标命中表: {list(indicator_tables) or '无'}")
        lines.append(f"  - 向量召回表: {vector_tables}")
        lines.append("")

        return "\n".join(lines), final_tables
