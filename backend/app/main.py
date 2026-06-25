"""FastAPI 应用入口 — ChatBI 智能问数系统。

Lifespan 生命周期：
  1. DatabasePool 创建 + SchemaDiscovery 全量扫描
  2. SchemaRAGEngine 初始化（模型延迟加载）
  3. Indicator Registry 加载
  4. Analyzer Registry 自动发现
  5. QueryParserEngine + JoinPathFinder 注入
  6. SemanticCache 接入
  7. 组件存入 app.state
"""

from __future__ import annotations

import logging
import os

# ── 0. 环境适配（国内 HuggingFace 镜像，必须在导入 huggingface_hub 前设置）──
from config import settings

if settings.HF_ENDPOINT:
    os.environ["HF_ENDPOINT"] = settings.HF_ENDPOINT

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    # ── 1. DB 连接池 ──
    from app.db.connector import DatabasePool

    pool = DatabasePool(settings.DB_DSN)
    await pool.create()
    app.state.pool = pool
    logger.info("DB pool created")

    # ── 2. Schema 发现 ──
    from app.db.schema_discovery import SchemaDiscovery

    schema = SchemaDiscovery(pool)
    await schema.refresh()
    app.state.schema = schema
    table_count = len(schema._schema.tables) if schema._schema else 0
    logger.info("Schema discovered: %d tables", table_count)

    # ── 3. Schema-RAG（模型延迟加载）──
    from app.engine.schema_rag import SchemaRAGEngine

    schema_rag = SchemaRAGEngine(schema)
    app.state.schema_rag = schema_rag
    logger.info("SchemaRAGEngine initialized")

    # ── 4. Indicator Registry ──
    from app.engine.indicator_registry import indicator_registry

    indicator_registry.load()
    logger.info("Indicator registry loaded: %d indicators", indicator_registry.size)

    # ── 5. Analyzer Registry ──
    from app.engine import ensure_analyzers_registered
    from app.engine.registry import registry

    ensure_analyzers_registered()
    app.state.registry = registry
    logger.info("Analyzer registry: %d algorithms", registry.size)

    # ── 6. Parser 引擎 ──
    from app.engine.join_path_finder import JoinPathFinder
    from app.engine.parser import parser_engine

    join_finder = JoinPathFinder()
    parser_engine.initialize(schema_rag, join_finder)

    # ── 7. 语义缓存（复用 SchemaRAG 的 Embedding 模型）──
    from app.engine.semantic_cache import SemanticCache

    class _LazyEmbedder:
        """包装 SchemaRAGEngine.model，避免 lifespan 中触发模型加载。"""
        def __init__(self, rag: SchemaRAGEngine) -> None:
            self._rag = rag
        def encode(self, text: str, **kwargs):
            return self._rag.model.encode(text, **kwargs)

    cache = SemanticCache(_LazyEmbedder(schema_rag), backend=settings.CACHE_BACKEND)
    parser_engine.set_cache(cache)
    app.state.cache = cache
    logger.info("SemanticCache ready (backend=%s)", settings.CACHE_BACKEND)

    # ── 7.5. Table Importer（动态数据表导入服务）──
    from app.engine.table_importer import TableImporter

    table_importer = TableImporter(
        pool=pool,
        schema=schema,
        schema_rag=schema_rag,
        join_finder=join_finder,
    )
    app.state.table_importer = table_importer
    logger.info("TableImporter initialized")

    # 恢复已导入表的运行时状态（服务重启后）
    # 遍历 schema 中的表，对不在 PHYSICAL_TABLE_COLUMNS 中的表进行注册
    from app.engine.sql_builder import (
        PHYSICAL_TABLE_COLUMNS,
        register_table_columns,
        register_time_column,
        TIME_COLUMN_CANDIDATES as _TIME_CANDIDATES,
    )

    try:
        imported_tables: list[str] = []
        for tname in schema.get_all_tables():
            if tname not in PHYSICAL_TABLE_COLUMNS:
                tinfo = schema.get_table(tname)
                if tinfo:
                    col_set = {c.name for c in tinfo.columns}
                    register_table_columns(tname, col_set)
                    # 推断时间列
                    for tc in _TIME_CANDIDATES:
                        if tc in col_set:
                            register_time_column(tname, tc)
                            break
                    imported_tables.append(tname)

        if imported_tables:
            table_importer.restore_imported_tables(imported_tables)
            logger.info("Restored %d imported tables: %s", len(imported_tables), imported_tables)
    except Exception:
        logger.exception("恢复已导入表失败，跳过（不影响核心服务启动）")

    # ── 8. Trace 持久化存储 ──
    from app.engine.trace_storage import TraceStorage

    trace_storage = TraceStorage()
    app.state.trace_storage = trace_storage

    yield

    # ── 关闭 ──
    trace_storage.close()
    logger.info("TraceStorage closed")
    await pool.close()
    logger.info("DB pool closed")


app = FastAPI(
    title="ChatBI Agent",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# 全局异常处理器
from app.api.error_handler import register_error_handlers
register_error_handlers(app)


if __name__ == "__main__":
    # 启动前设置 HuggingFace 镜像（国内加速）
    if settings.HF_ENDPOINT:
        os.environ["HF_ENDPOINT"] = settings.HF_ENDPOINT
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
