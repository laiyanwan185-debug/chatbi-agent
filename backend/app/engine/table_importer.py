"""动态数据表导入服务。

编排流程：文件解析 → 建表入库 → schema 刷新 → 指标生成 → JOIN 检测 → 组件热更新
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from app.db.connector import DatabasePool
from app.db.schema_discovery import (
    ColumnInfo,
    SchemaDiscovery,
)
from app.engine import sql_builder as _sql_builder_mod
from app.engine.indicator_registry import IndicatorDef, indicator_registry
from app.engine.join_path_finder import JoinPathFinder
from app.engine.schema_rag import SchemaRAGEngine
from app.engine.sql_builder import (
    TIME_COLUMN_CANDIDATES,
    register_table_columns,
    register_time_column,
    unregister_table_columns,
    unregister_time_column,
)

logger = logging.getLogger(__name__)

# 导入时跳过的列名（不生成指标）
SKIP_INDICATOR_COLS: set[str] = {"创建时间", "更新时间", "_imported_id"}

# JOIN 检测时跳过这些列（它们不是真正的关联维度）
SKIP_JOIN_COLS: set[str] = {"创建时间", "更新时间", "_imported_id", "统计ID"}

# 已知的关联维度列名 → 目标表名（None 表示通用维度，不直接关联到某张表）
KNOWN_JOIN_KEYS: dict[str, str | None] = {
    "区划ID": "admin_region_data",
    "企业ID": "enterprise_data",
}

# 时间列名候选（复用 sql_builder 的定义）
_TIME_COL_NAMES = set(TIME_COLUMN_CANDIDATES)


# ── 导入结果模型 ──


class JoinRelationship(BaseModel):
    """新表与现有表的 JOIN 关系描述。"""
    target: str
    type: str = "left"
    on: list[str] = Field(default_factory=list)


class ImportResult(BaseModel):
    """导入结果。"""
    table_name: str
    row_count: int = 0
    column_count: int = 0
    columns: list[str] = Field(default_factory=list)
    join_relationships: list[JoinRelationship] = Field(default_factory=list)
    indicator_count: int = 0
    time_column: str | None = None


class ImportError_(Exception):
    """导入校验失败。"""


# ── 核心服务类 ──


class TableImporter:
    """动态数据表导入服务。

    编排流程：
    1. 文件解析（CSV/Excel）
    2. 类型推断 + 建表 + 数据入库
    3. Schema 刷新
    4. 指标自动生成
    5. JOIN 关系自动检测
    6. 组件热更新（SchemaRAG 向量索引、JoinPathFinder 边）
    """

    def __init__(
        self,
        pool: DatabasePool,
        schema: SchemaDiscovery,
        schema_rag: SchemaRAGEngine,
        join_finder: JoinPathFinder,
    ) -> None:
        self._pool = pool
        self._schema = schema
        self._schema_rag = schema_rag
        self._join_finder = join_finder
        self._imported_tables: set[str] = set()

    # ── 主入口 ──

    async def import_file(
        self,
        file_content: bytes,
        filename: str,
        table_name: str | None = None,
        table_comment: str | None = None,
    ) -> ImportResult:
        """导入 CSV/Excel 文件为新的数据表。

        完整流程：解析 → 校验 → 建表 → 入库 → 注册 → 热更新
        """
        # 1. 解析文件
        df = self._parse_file(file_content, filename)
        self._validate_dataframe(df, filename)

        # 2. 推断表名
        if not table_name:
            table_name = self._infer_table_name(filename)
        self._validate_table_name(table_name)

        # 3. 检查表名冲突
        existing = self._schema.get_all_tables()
        if table_name in existing:
            raise ImportError_(f"表名 '{table_name}' 已存在")

        # 4. 清洗列名（去重、去空）
        df = self._sanitize_columns(df)

        # 5. 推断列类型
        col_types = self._infer_column_types(df)

        # 6. 推断时间列
        time_col = self._detect_time_column(df)

        # 7. 建表 + 数据入库
        await self._create_table_and_load(df, table_name, col_types, table_comment, time_col)

        # 8-12. 元数据注册 + 组件热更新（失败时回滚已创建的表）
        try:
            # 8. Schema 刷新
            table_info = await self._schema.refresh_single(table_name)
            if table_info and table_comment:
                table_info.comment = table_comment

            # 9. 动态元数据注册
            col_set = {c.name for c in (table_info.columns if table_info else df.columns.tolist())}
            register_table_columns(table_name, col_set)
            if time_col:
                register_time_column(table_name, time_col)

            # 10. 自动生成指标
            indicators = self._auto_generate_indicators(
                table_name,
                table_info.columns if table_info else [ColumnInfo(name=c, dtype="text", nullable=True, comment=None) for c in df.columns],
                df,
            )
            indicator_registry.add_indicators(indicators)

            # 11. 自动检测 JOIN 关系
            joins = self._detect_join_relationships(
                table_name,
                table_info.columns if table_info else [ColumnInfo(name=c, dtype="text", nullable=True, comment=None) for c in df.columns],
                existing,
            )
            if joins:
                self._join_finder.add_table_node(table_name, [j.model_dump() for j in joins])

            # 12. SchemaRAG 向量索引重建
            try:
                self._schema_rag.build_vector_index()
            except Exception:
                logger.exception("SchemaRAG 向量索引重建失败，跳过")

        except Exception:
            # 回滚：删除已创建的表和所有元数据
            logger.exception("导入中间步骤失败，回滚表 '%s'", table_name)
            try:
                await self._pool.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')
            except Exception:
                logger.exception("回滚 DROP TABLE 也失败")
            # 清理可能已注册的元数据
            unregister_table_columns(table_name)
            unregister_time_column(table_name)
            indicator_registry.remove_indicators_for_table(table_name)
            self._join_finder.remove_table_node(table_name)
            if self._schema._schema and table_name in self._schema._schema.tables:
                del self._schema._schema.tables[table_name]
                self._schema._save_cache()
            raise

        # 13. 记录
        self._imported_tables.add(table_name)

        result = ImportResult(
            table_name=table_name,
            row_count=len(df),
            column_count=len(df.columns),
            columns=list(df.columns),
            join_relationships=joins,
            indicator_count=len(indicators),
            time_column=time_col,
        )
        logger.info(
            "Table imported: '%s' (%d rows, %d cols, %d indicators, %d joins)",
            table_name, result.row_count, result.column_count,
            result.indicator_count, len(joins),
        )
        return result

    async def delete_table(self, table_name: str) -> None:
        """删除已导入的表并清理所有关联元数据。"""
        if table_name not in self._imported_tables:
            raise ImportError_(f"表 '{table_name}' 不是通过 TableImporter 导入的")
        # 二次校验表名合法性（防 restore_imported_tables 引入恶意表名）
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
            raise ImportError_(f"表名不合法: {table_name}")

        # 1. DROP TABLE
        try:
            await self._pool.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')
            logger.info("DROP TABLE '%s'", table_name)
        except Exception:
            logger.exception("DROP TABLE '%s' 失败", table_name)

        # 2. 清理元数据
        unregister_table_columns(table_name)
        unregister_time_column(table_name)
        indicator_registry.remove_indicators_for_table(table_name)
        self._join_finder.remove_table_node(table_name)

        # 3. 从 SchemaDiscovery 中移除
        if self._schema._schema and table_name in self._schema._schema.tables:
            del self._schema._schema.tables[table_name]
            self._schema._save_cache()

        # 4. 重建向量索引
        try:
            self._schema_rag.build_vector_index()
        except Exception:
            logger.exception("SchemaRAG 向量索引重建失败")

        self._imported_tables.discard(table_name)
        logger.info("Table deleted: '%s'", table_name)

    def get_imported_tables(self) -> list[str]:
        """获取所有已导入的表名。"""
        return list(self._imported_tables)

    def restore_imported_tables(self, table_names: list[str]) -> None:
        """服务重启后恢复已导入表的记录（仅接受合法表名）。"""
        validated = [n for n in table_names if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", n)]
        self._imported_tables.update(validated)

    # ── 内部方法：文件解析 ──

    def _parse_file(self, content: bytes, filename: str) -> pd.DataFrame:
        """解析 CSV 或 Excel 文件。"""
        suffix = Path(filename).suffix.lower()

        if suffix == ".csv":
            # 验证 CSV 是纯文本（不含空字节）
            if b"\x00" in content[:1024]:
                raise ImportError_("文件不是有效的 CSV 格式（包含空字节）")
            # 尝试 UTF-8，失败则尝试 GBK
            for enc in ("utf-8", "utf-8-sig", "gbk", "gb2312", "gb18030"):
                try:
                    return pd.read_csv(io.BytesIO(content), encoding=enc)
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            raise ImportError_("CSV 文件编码无法识别，请使用 UTF-8 或 GBK 编码")

        elif suffix in (".xlsx", ".xls"):
            # 验证 Excel 文件魔数（以 PK 开头 = ZIP 格式）
            if suffix == ".xlsx" and len(content) > 2 and not content[:2].startswith(b"PK"):
                raise ImportError_("文件不是有效的 Excel 格式（.xlsx 应为 ZIP 格式）")
            try:
                return pd.read_excel(io.BytesIO(content))
            except Exception as e:
                raise ImportError_(f"Excel 文件解析失败: {e}")

        else:
            raise ImportError_(f"不支持的文件格式: {suffix}（仅支持 .csv, .xlsx, .xls）")

    def _validate_dataframe(self, df: pd.DataFrame, filename: str) -> None:
        """校验 DataFrame 质量。"""
        if df.empty:
            raise ImportError_("文件为空，没有数据行")
        if len(df.columns) == 0:
            raise ImportError_("文件没有列")
        if len(df.columns) > 500:
            raise ImportError_(f"列数过多（{len(df.columns)}），最大支持 500 列")
        if len(df) > 500_000:
            raise ImportError_(f"行数过多（{len(df)}），最大支持 500,000 行")

    def _validate_table_name(self, name: str) -> None:
        """校验表名合法性。"""
        if not name:
            raise ImportError_("表名不能为空")
        if len(name) > 63:
            raise ImportError_("表名不能超过 63 个字符")
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ImportError_(
                f"表名 '{name}' 不合法：仅允许字母、数字和下划线，且以字母或下划线开头"
            )

    def _infer_table_name(self, filename: str) -> str:
        """从文件名推断表名。"""
        stem = Path(filename).stem
        # 替换中文/特殊字符为下划线
        name = re.sub(r"[^\w]", "_", stem)
        name = re.sub(r"_+", "_", name).strip("_").lower()
        if not name or name[0].isdigit():
            name = "imported_" + name
        return name or "imported_table"

    def _sanitize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗列名：去空、去重、去除危险字符（防 SQL 注入）。"""
        new_cols: list[str] = []
        seen: dict[str, int] = {}
        for i, col in enumerate(df.columns):
            col_str = str(col).strip()
            # 移除双引号、单引号、反斜杠、分号等危险字符（防 DDL 注入）
            col_str = re.sub(r'''["';\\]''', '', col_str)
            if not col_str or col_str == f"Unnamed: {i}":
                col_str = f"col_{i + 1}"
            # 去重
            if col_str in seen:
                seen[col_str] += 1
                col_str = f"{col_str}_{seen[col_str]}"
            else:
                seen[col_str] = 0
            new_cols.append(col_str)
        df.columns = new_cols
        return df

    # ── 内部方法：类型推断 + 建表 ──

    def _infer_column_types(self, df: pd.DataFrame) -> list[tuple[str, str]]:
        """推断每列的 PostgreSQL 类型。返回 [(col_name, pg_type), ...]。"""
        result: list[tuple[str, str]] = []
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                result.append((col, "BIGINT"))
            elif pd.api.types.is_float_dtype(dtype):
                result.append((col, "NUMERIC"))
            elif pd.api.types.is_bool_dtype(dtype):
                result.append((col, "BOOLEAN"))
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                result.append((col, "TIMESTAMP"))
            else:
                # 文本类型：根据最大长度选择 VARCHAR 或 TEXT
                max_len = df[col].astype(str).str.len().max()
                if max_len and max_len <= 200:
                    result.append((col, f"VARCHAR({min(255, max(10, int(max_len * 1.2)))})"))
                else:
                    result.append((col, "TEXT"))
        return result

    def _detect_time_column(self, df: pd.DataFrame) -> str | None:
        """检测 DataFrame 中的时间列名。"""
        col_names_lower = {c.lower().strip('"\' '): c for c in df.columns}
        for candidate in TIME_COLUMN_CANDIDATES:
            if candidate.lower() in col_names_lower:
                return col_names_lower[candidate.lower()]
        return None

    async def _create_table_and_load(
        self,
        df: pd.DataFrame,
        table_name: str,
        col_types: list[tuple[str, str]],
        comment: str | None,
        time_col: str | None,
    ) -> None:
        """建表 + 插入数据到 PostgreSQL。"""
        # 构建 CREATE TABLE DDL（转义双引号防 SQL 注入）
        col_defs: list[str] = []
        col_defs.append('"_imported_id" BIGSERIAL PRIMARY KEY')
        for col_name, pg_type in col_types:
            safe_col = col_name.replace('"', '""')
            col_defs.append(f'"{safe_col}" {pg_type}')

        safe_table = table_name.replace('"', '""')
        ddl = f'CREATE TABLE "{safe_table}" ({", ".join(col_defs)})'
        await self._pool.execute(ddl)

        # 表注释
        if comment:
            escaped = comment.replace("'", "''")
            await self._pool.execute(f"COMMENT ON TABLE \"{safe_table}\" IS '{escaped}'")

        # 数据插入：批量 COPY
        # 将 datetime 列转为字符串以避免 asyncpg 类型不匹配
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col].dtype):
                df[col] = df[col].astype(str).replace("NaT", None)
            # NaN → None
            df[col] = df[col].where(pd.notnull(df[col]), None)

        records = df.to_records(index=False)
        columns_list = list(df.columns)

        # 使用 asyncpg 的 copy_records_to_table（需要直接访问连接池）
        pool = self._pool._require_pool()
        async with pool.acquire() as conn:
            await conn.copy_records_to_table(
                table_name,
                records=records,
                columns=columns_list,
            )

        logger.info("Table '%s' created with %d rows", table_name, len(df))

    # ── 内部方法：指标自动生成 ──

    def _auto_generate_indicators(
        self,
        table_name: str,
        columns: list[ColumnInfo],
        df: pd.DataFrame,
    ) -> list[IndicatorDef]:
        """根据列的数据类型和命名模式自动生成指标定义。"""
        indicators: list[IndicatorDef] = []

        for col in columns:
            col_name = col.name
            if col_name in SKIP_INDICATOR_COLS:
                continue
            if col_name.startswith("_"):
                continue

            dtype = col.dtype.lower()

            # 1. 数值列 → direct 指标
            if any(t in dtype for t in ("int", "numeric", "float", "double", "real", "decimal")):
                aliases = _infer_aliases(col_name)
                indicators.append(IndicatorDef(
                    name=col_name,
                    aliases=aliases,
                    type="direct",
                    field=col_name,
                    table=table_name,
                ))

            # 2. 文本列 → category 指标（但排除明显不是分类维度的列）
            elif any(t in dtype for t in ("varchar", "text", "character", "char")):
                if col_name not in _SKIP_CATEGORY_COLS and not col_name.endswith("时间") and not col_name.endswith("日期"):
                    aliases = _infer_category_aliases(col_name)
                    indicators.append(IndicatorDef(
                        name=col_name,
                        aliases=aliases,
                        type="category",
                        field=col_name,
                        table=table_name,
                    ))

            # 3. 布尔列 → category 指标
            elif "bool" in dtype:
                indicators.append(IndicatorDef(
                    name=col_name,
                    aliases=[],
                    type="category",
                    field=col_name,
                    table=table_name,
                ))

        logger.info(
            "Auto-generated %d indicators for '%s' (%d direct, %d category)",
            len(indicators), table_name,
            sum(1 for i in indicators if i.type == "direct"),
            sum(1 for i in indicators if i.type == "category"),
        )
        return indicators

    # ── 内部方法：JOIN 关系自动检测 ──

    def _detect_join_relationships(
        self,
        table_name: str,
        columns: list[ColumnInfo],
        existing_tables: list[str],
    ) -> list[JoinRelationship]:
        """检测新表与现有表的 JOIN 关系。"""
        col_names = {c.name for c in columns}
        joins: list[JoinRelationship] = []
        detected_targets: set[str] = set()

        # 1. 检查已知关联键
        for key_col, ref_table in KNOWN_JOIN_KEYS.items():
            if key_col in col_names and ref_table:
                if ref_table in existing_tables:
                    joins.append(JoinRelationship(
                        target=ref_table,
                        type="left",
                        on=[key_col],
                    ))
                    detected_targets.add(ref_table)

        # 2. 检查与其他数据表的共同维度列
        for existing in existing_tables:
            if existing == table_name or existing in detected_targets:
                continue
            existing_info = self._schema.get_table(existing)
            if not existing_info:
                continue
            existing_col_names = {c.name for c in existing_info.columns}
            common = col_names & existing_col_names - SKIP_JOIN_COLS
            # 排除自增主键和通用维度
            common = {c for c in common if c not in SKIP_JOIN_COLS and (not c.endswith("_id") or c == "区划ID")}
            if common:
                joins.append(JoinRelationship(
                    target=existing,
                    type="left",
                    on=list(common),
                ))
                detected_targets.add(existing)

        logger.info(
            "Detected %d JOIN relationships for '%s'", len(joins), table_name,
        )
        return joins


# ── 别名推断工具函数 ──

# 不应作为 category 指标的列名
_SKIP_CATEGORY_COLS: set[str] = {"创建时间", "更新时间", "备注", "说明", "描述", "_imported_id"}

# 数值列关键词 → 别名
_NUMERIC_ALIAS_RULES: dict[str, list[str]] = {
    "金额": ["金额", "资金"],
    "收入": ["收入", "营业收入"],
    "营收": ["营收", "营业收入"],
    "利润": ["利润", "盈利"],
    "数量": ["数量", "数目"],
    "总数": ["总数", "总量"],
    "个数": ["个数", "数目"],
    "比例": ["比例", "占比"],
    "占比": ["占比", "百分比"],
    "比率": ["比率", "百分比"],
    "价格": ["价格", "单价"],
    "成本": ["成本", "费用"],
    "支出": ["支出", "费用"],
    "投资": ["投资额", "投入"],
    "增长": ["增长率", "增速"],
    "下降": ["降幅", "减少量"],
}

# 分类列关键词 → 别名
_CATEGORY_ALIAS_RULES: dict[str, list[str]] = {
    "名称": ["名字", "名称"],
    "类型": ["类型", "类别"],
    "级别": ["级别", "等级"],
    "区域": ["区域", "地区"],
    "行业": ["行业", "产业"],
    "省份": ["省份", "地区"],
    "城市": ["城市", "地区"],
}


def _infer_aliases(col_name: str) -> list[str]:
    """根据列名关键词推断数值列的别名。"""
    aliases: list[str] = []
    for keyword, alias_list in _NUMERIC_ALIAS_RULES.items():
        if keyword in col_name:
            for alias in alias_list:
                if alias != col_name and alias not in aliases:
                    aliases.append(alias)
    return aliases[:3]  # 最多 3 个别名


def _infer_category_aliases(col_name: str) -> list[str]:
    """根据列名关键词推断分类列的别名。"""
    aliases: list[str] = []
    for keyword, alias_list in _CATEGORY_ALIAS_RULES.items():
        if keyword in col_name:
            for alias in alias_list:
                if alias != col_name and alias not in aliases:
                    aliases.append(alias)
    return aliases[:3]
