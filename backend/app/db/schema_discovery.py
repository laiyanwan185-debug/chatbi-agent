"""
Schema 自动发现 — information_schema 全量扫描 + FK 提取 → 缓存
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")
CACHE_FILE = CACHE_DIR / "schema_cache.json"


@dataclass
class ColumnInfo:
    name: str
    dtype: str
    nullable: bool
    comment: str | None
    is_pk: bool = False


@dataclass
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass
class TableInfo:
    name: str
    comment: str | None
    columns: list[ColumnInfo] = field(default_factory=list)
    primary_keys: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)


@dataclass
class SchemaInfo:
    tables: dict[str, TableInfo] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class SchemaDiscovery:
    """从 PostgreSQL information_schema 扫描表结构并缓存。"""

    def __init__(self, pool) -> None:  # noqa: ANN001
        self._pool = pool
        self._schema: SchemaInfo | None = None

    # ── 全量扫描 ──

    async def refresh(self) -> SchemaInfo:
        """全量扫描 pg_catalog → 更新缓存。"""
        tables = await self._fetch_tables()
        all_fks = await self._fetch_foreign_keys()  # 一次批量查询

        # 按源表分组
        fk_by_table: dict[str, list[ForeignKey]] = {}
        for fk in all_fks:
            tname = fk["source_table"]
            if tname not in fk_by_table:
                fk_by_table[tname] = []
            fk_by_table[tname].append(ForeignKey(
                column=fk["source_column"],
                ref_table=fk["foreign_table"],
                ref_column=fk["foreign_column"],
            ))

        self._schema = SchemaInfo()
        for table_name, comment in tables:
            columns = await self._fetch_columns(table_name)
            pks = await self._fetch_primary_keys(table_name)

            self._schema.tables[table_name] = TableInfo(
                name=table_name,
                comment=comment,
                columns=[
                    ColumnInfo(
                        name=c["column_name"],
                        dtype=c["data_type"],
                        nullable=c["is_nullable"] == "YES",
                        comment=c.get("comment"),
                        is_pk=c["column_name"] in {pk["column_name"] for pk in pks},
                    )
                    for c in columns
                ],
                primary_keys=[pk["column_name"] for pk in pks],
                foreign_keys=fk_by_table.get(table_name, []),
            )

        self._save_cache()
        logger.info("Schema refreshed: %d tables", len(self._schema.tables))
        return self._schema

    async def get(self) -> SchemaInfo:
        """获取 Schema（懒加载：优先读缓存，失败则全量扫描）。"""
        if self._schema is not None:
            return self._schema

        if CACHE_FILE.exists():
            self._schema = self._load_cache()
            if self._schema is not None:
                return self._schema

        return await self.refresh()

    # ── 单表刷新（供 TableImporter 调用） ──

    async def refresh_single(self, table_name: str) -> TableInfo | None:
        """刷新单张表的 schema 信息（不触发全量扫描）。

        Returns:
            TableInfo 如果表存在；None 如果表不存在。
        """
        if self._schema is None:
            await self.refresh()

        columns = await self._fetch_columns(table_name)
        if not columns:
            logger.warning("refresh_single: table '%s' not found", table_name)
            return None

        pks = await self._fetch_primary_keys(table_name)
        all_fks = await self._fetch_foreign_keys()
        table_fks = [
            ForeignKey(
                column=fk["source_column"],
                ref_table=fk["foreign_table"],
                ref_column=fk["foreign_column"],
            )
            for fk in all_fks
            if fk["source_table"] == table_name
        ]

        pk_set = {pk["column_name"] for pk in pks}
        table_info = TableInfo(
            name=table_name,
            comment=None,  # 导入时由 TableImporter 设置
            columns=[
                ColumnInfo(
                    name=c["column_name"],
                    dtype=c["data_type"],
                    nullable=c["is_nullable"] == "YES",
                    comment=c.get("comment"),
                    is_pk=c["column_name"] in pk_set,
                )
                for c in columns
            ],
            primary_keys=list(pk_set),
            foreign_keys=table_fks,
        )

        self._schema.tables[table_name] = table_info
        self._save_cache()
        logger.info(
            "refresh_single: '%s' (%d columns, %d FKs)",
            table_name, len(columns), len(table_fks),
        )
        return table_info

    # ── 查询接口 ──

    def get_table(self, name: str) -> TableInfo | None:
        if self._schema is None:
            return None
        return self._schema.tables.get(name)

    def get_all_tables(self) -> list[str]:
        if self._schema is None:
            return []
        return list(self._schema.tables.keys())

    def to_llm_context(self) -> str:
        """序列化为自然语言描述，供 LLM Prompt 注入。"""
        if self._schema is None:
            return ""
        lines: list[str] = []
        for tname, tbl in self._schema.tables.items():
            header = f"[表名: {tname}] {tbl.comment or ''}"
            lines.append(header)
            for col in tbl.columns:
                pk_tag = " [PK]" if col.is_pk else ""
                lines.append(f"  - {col.name} ({col.dtype}){pk_tag}: {col.comment or ''}")
            for fk in tbl.foreign_keys:
                lines.append(f"  - FK: {fk.column} → {fk.ref_table}.{fk.ref_column}")
        return "\n".join(lines)

    # ── 缓存 ──

    def _save_cache(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if self._schema is None:
            return
        data = {
            "tables": {
                name: asdict(tbl) for name, tbl in self._schema.tables.items()
            },
        }
        CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_cache(self) -> SchemaInfo | None:
        try:
            data = json.loads(CACHE_FILE.read_text("utf-8"))
            tables: dict[str, TableInfo] = {}
            for name, tbl_data in data.get("tables", {}).items():
                tables[name] = TableInfo(
                    name=tbl_data["name"],
                    comment=tbl_data.get("comment"),
                    columns=[ColumnInfo(**c) for c in tbl_data.get("columns", [])],
                    primary_keys=list(tbl_data.get("primary_keys", [])),
                    foreign_keys=[ForeignKey(**fk) for fk in tbl_data.get("foreign_keys", [])],
                )
            logger.info("Schema cache loaded: %d tables", len(tables))
            return SchemaInfo(tables=tables)
        except Exception as exc:
            logger.warning("Failed to load schema cache: %s", exc)
            return None

    # ── 底层查询 ──

    async def _fetch_tables(self) -> list[tuple[str, str | None]]:
        rows = await self._pool.fetch("""
            SELECT
                t.table_name,
                pgd.description AS table_comment
            FROM information_schema.tables t
            LEFT JOIN pg_catalog.pg_description pgd
                ON pgd.objoid = (quote_ident(t.table_schema) || '.' || quote_ident(t.table_name))::regclass::oid
                AND pgd.objsubid = 0
            WHERE t.table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY t.table_name
        """)
        return [(r["table_name"], r.get("table_comment")) for r in rows]

    async def _fetch_columns(self, table: str) -> list[dict[str, Any]]:
        """使用 pg_attribute 替代 information_schema，规避 DROP COLUMN 导致的列与注释错位。"""
        rows = await self._pool.fetch("""
            SELECT
                a.attname AS column_name,
                format_type(a.atttypid, a.atttypmod) AS data_type,
                CASE WHEN a.attnotnull THEN 'NO' ELSE 'YES' END AS is_nullable,
                pg_catalog.col_description(a.attrelid, a.attnum) AS comment
            FROM pg_catalog.pg_attribute a
            JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
            JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = $1
              AND n.nspname = 'public'
              AND a.attnum > 0
              AND NOT a.attisdropped
            ORDER BY a.attnum
        """, table)
        return [dict(r) for r in rows]

    async def _fetch_primary_keys(self, table: str) -> list[dict[str, str]]:
        rows = await self._pool.fetch("""
            SELECT a.attname AS column_name
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = ($1::regclass)::oid
              AND i.indisprimary
        """, table)
        return [{"column_name": r["column_name"]} for r in rows]

    async def _fetch_foreign_keys(self) -> list[dict[str, str]]:
        """批量提取所有 FK 关系 — 一次查询替代逐表循环。"""
        rows = await self._pool.fetch("""
            SELECT
                c.relname AS source_table,
                a.attname AS source_column,
                conf.relname AS foreign_table,
                af.attname AS foreign_column
            FROM pg_constraint con
            JOIN pg_class c ON c.oid = con.conrelid
            JOIN pg_class conf ON conf.oid = con.confrelid
            JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = con.conkey[1]
            JOIN pg_attribute af ON af.attrelid = con.confrelid AND af.attnum = con.confkey[1]
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE con.contype = 'f'
              AND n.nspname = 'public'
        """)
        return [dict(r) for r in rows]
