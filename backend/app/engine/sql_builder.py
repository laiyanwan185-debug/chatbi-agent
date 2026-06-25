"""确定性 SQL 构建器 — 从结构化 Parser 输出构建 SQL。

核心理念：Parser 已经输出结构化字段（tables, indicators, filters, time_range, dim_cols,
aggregation, top_k, sort_order），SQLBuilder 将这些字段直接组装为 SQL，不经过 LLM。

SQLBuilder 无法覆盖的场景（复杂计算指标、子查询、CTE 等），回退到 LLM raw_sql。
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# 时间列名候选（按优先级）
TIME_COLUMN_CANDIDATES = ["统计年份", "year", "年份", "quarter", "季度", "month", "月", "period", "time"]

# 各表的时间列名映射（DB 实际列名）
TABLE_TIME_COL_MAP: dict[str, str] = {
    "economic_indicator_data": "统计年份",
    "population_data": "统计年份",
    "employment_data": "统计年份",
    "real_estate_data": "统计年份",
    "medical_health_data": "统计年份",
    "edu_data": "统计年份",
    "transport_data": "统计年份",
    "env_monitor_data": "监测日期",
}

# 物理表 → 数字列名列表（用于校验 column 存在性）
# 直接从 DB 的 information_schema 提取
PHYSICAL_TABLE_COLUMNS: dict[str, set[str]] = {
    "economic_indicator_data": {"指标ID", "区划ID", "统计年份", "统计季度", "GDP", "第一产业", "第二产业", "第三产业", "GDP增长率", "人均GDP", "财政收入", "财政支出", "社会消费品零售总额", "固定资产投资", "居民消费价格指数", "创建时间"},
    "admin_region_data": {"区划ID", "区划代码", "区划名称", "区划级别", "上级ID", "人口", "面积", "GDP", "创建时间"},
    "population_data": {"统计ID", "区划ID", "统计年份", "统计季度", "总人口", "男性人口", "女性人口", "城镇人口", "农村人口", "出生率", "死亡率", "自然增长率", "老龄化率", "创建时间"},
    "employment_data": {"统计ID", "区划ID", "统计年份", "统计季度", "就业人口", "失业人口", "失业率", "城镇就业人口", "农村就业人口", "新增就业岗位", "平均工资", "最低工资", "创建时间"},
    "env_monitor_data": {"监测ID", "区划ID", "监测日期", "空气质量指数", "PM2.5", "PM10", "空气质量等级", "水质达标率", "森林覆盖率", "绿化覆盖率", "固体废物处理量", "污水处理率", "创建时间"},
    "real_estate_data": {"统计ID", "区划ID", "统计年份", "统计月份", "平均房价", "新建住宅面积", "已售住宅面积", "住宅销售额", "施工面积", "竣工面积", "房地产投资", "创建时间"},
    "medical_health_data": {"统计ID", "区划ID", "统计年份", "医院数量", "诊所数量", "医院床位数", "医生数量", "护士数量", "千人床位数", "千人医生数", "门诊就诊人次", "住院人次", "医疗卫生支出", "创建时间"},
    "edu_data": {"统计ID", "区划ID", "统计年份", "小学数量", "小学生数", "小学教师数", "中学数量", "中学生数", "中学教师数", "大学数量", "大学生数", "大学教师数", "识字率", "高等教育入学率", "教育支出", "创建时间"},
    "transport_data": {"统计ID", "区划ID", "统计年份", "公路里程", "铁路里程", "地铁里程", "民用车辆数", "客运量", "货运量", "机场数量", "港口数量", "创建时间"},
    "enterprise_data": {"企业ID", "区划ID", "企业名称", "统一社会信用代码", "行业类型", "企业类型", "注册资本", "员工数量", "年营收", "年纳税额", "注册日期", "状态", "创建时间"},
}

# 列名 → 所属表名反向索引（用于跨表 indicator 定位表名）
COLUMN_TO_TABLE: dict[str, str] = {}
for _tbl, _cols in PHYSICAL_TABLE_COLUMNS.items():
    for _col in _cols:
        if _col not in COLUMN_TO_TABLE:
            COLUMN_TO_TABLE[_col] = _tbl
        # 如果同一列名出现在多张表中，用 "/" 分隔存储
        elif COLUMN_TO_TABLE[_col] != _tbl:
            COLUMN_TO_TABLE[_col] = f"{COLUMN_TO_TABLE[_col]}/{_tbl}"


# ── 动态注册/注销函数（供 TableImporter 调用） ──


def register_table_columns(table_name: str, columns: set[str]) -> None:
    """运行时注册新表的列信息到全局字典。

    用于动态导入的新数据表加入 SQLBuilder 的校验范围。
    """
    PHYSICAL_TABLE_COLUMNS[table_name] = columns
    for col in columns:
        if col not in COLUMN_TO_TABLE:
            COLUMN_TO_TABLE[col] = table_name
        elif COLUMN_TO_TABLE[col] != table_name:
            # 同一列名出现在多张表中，追加表名
            existing = COLUMN_TO_TABLE[col]
            if table_name not in existing.split("/"):
                COLUMN_TO_TABLE[col] = f"{existing}/{table_name}"
    logger.info(
        "register_table_columns: '%s' (%d columns)", table_name, len(columns),
    )


def unregister_table_columns(table_name: str) -> None:
    """删除已注册的表列信息。"""
    removed_cols = PHYSICAL_TABLE_COLUMNS.pop(table_name, set())
    for col in removed_cols:
        if col not in COLUMN_TO_TABLE:
            continue
        val = COLUMN_TO_TABLE[col]
        parts = [p for p in val.split("/") if p != table_name]
        if len(parts) == 1:
            COLUMN_TO_TABLE[col] = parts[0]
        elif not parts:
            del COLUMN_TO_TABLE[col]
        else:
            COLUMN_TO_TABLE[col] = "/".join(parts)
    logger.info(
        "unregister_table_columns: '%s' (removed %d columns)",
        table_name, len(removed_cols),
    )


def register_time_column(table_name: str, time_col: str) -> None:
    """注册新表的时间列映射。"""
    TABLE_TIME_COL_MAP[table_name] = time_col
    logger.info("register_time_column: '%s' → '%s'", table_name, time_col)


def unregister_time_column(table_name: str) -> None:
    """删除新表的时间列映射。"""
    TABLE_TIME_COL_MAP.pop(table_name, None)


class SQLBuilder:
    """确定性 SQL 构建器。"""

    def __init__(self, join_instructions: str = "") -> None:
        self._join_instructions = join_instructions

    def can_build(self, execution_plan: dict[str, Any]) -> bool:
        """判断 SQLBuilder 能否处理该 plan。

        SQLBuilder 可以处理：
        - 简单查询：SELECT cols FROM table WHERE conditions
        - 多表 JOIN 查询：通过 JoinPathFinder 的路径
        - 聚合查询：有 aggregation 字段
        - 排行查询：有 top_k / sort_order
        - 指标带 / 后缀（自动剥离后匹配）

        SQLBuilder 不能处理：
        - 计算型指标需要跨表公式运算（如 A/B 跨表）
        - 指标列不在任何参与 JOIN 的表或 indicator_registry 中
        - 无 tables 信息
        """
        tables: list[str] = execution_plan.get("tables", [])
        if not tables:
            return False

        indicators: list[str] = execution_plan.get("indicators", [])
        if not indicators:
            return True

        # 收集所有参与 JOIN 表的列集合
        all_table_cols: set[str] = set()
        for t in tables:
            all_table_cols.update(PHYSICAL_TABLE_COLUMNS.get(t, set()))

        # 检查每个指标列是否至少存在于一个 JOIN 表中
        for ind in indicators:
            cleaned = ind.strip().strip('"\' ')
            # 直接匹配
            if cleaned in all_table_cols:
                continue
            # 带 / 后缀的指标：尝试剥离后缀（如 "医院数量/千人" → "医院数量"）
            if "/" in cleaned:
                import re
                base = re.sub(r'/.+$', '', cleaned).strip()
                if base and base in all_table_cols:
                    continue
            # indicator_registry 反向查找
            try:
                from app.engine.indicator_registry import indicator_registry
                physical = indicator_registry.search_field_by_name(cleaned)
                if physical and physical in all_table_cols:
                    continue
            except Exception:
                pass
            logger.info(
                "SQLBuilder 无法处理: 指标列 '%s' 不在任何 JOIN 表 %s 的列中",
                cleaned, tables,
            )
            return False

        return True

    @staticmethod
    def _has_complex_calculations(sql: str) -> bool:
        """判断 raw_sql 是否包含 SQLBuilder 无法处理的复杂计算。"""
        if not sql:
            return False
        upper = sql.upper()
        # CASE WHEN
        if "CASE" in upper and "WHEN" in upper:
            return True
        # CTE
        if upper.strip().startswith("WITH"):
            return True
        # 子查询
        if "SELECT" in upper[upper.find("FROM"):] if "FROM" in upper else False:
            return True
        # 复杂窗口函数
        if "OVER" in upper and ("PARTITION" in upper or "RANK" in upper or "ROW_NUMBER" in upper):
            return True
        return False

    def build(
        self,
        execution_plan: dict[str, Any],
    ) -> str | None:
        """从结构化 execution_plan 构建 SQL。

        Returns:
            SQL 字符串，若无法构建返回 None。
        """
        tables: list[str] = execution_plan.get("tables", [])
        indicators: list[str] = execution_plan.get("indicators", [])
        time_range: dict = execution_plan.get("time_range", {})
        filters: list[str] = execution_plan.get("filters", [])
        aggregation: str | None = execution_plan.get("aggregation")
        top_k: int | None = execution_plan.get("top_k")
        sort_order: str | None = execution_plan.get("sort_order")
        analysis_type: str = execution_plan.get("analysis_type", "detail")
        dim_cols: list[str] = execution_plan.get("dim_cols", [])

        if not tables:
            return None

        primary_table = tables[0]

        # SELECT 列（传入完整 tables 以支持跨表列前缀）
        select_cols = self._build_select_columns(indicators, time_range, analysis_type, dim_cols, tables)
        if not select_cols:
            return None

        # 查重列名
        select_cols = list(dict.fromkeys(select_cols))

        # 构建 SQL
        parts = [f"SELECT {', '.join(select_cols)}"]

        # FROM
        parts.append(f"FROM {self._quote_col(primary_table)}")

        # JOIN（多表时）
        if len(tables) > 1:
            joins = self._build_joins(tables, primary_table)
            if joins:
                parts.extend(joins)

        # WHERE
        where_clauses = self._build_where(time_range, filters, primary_table, tables)
        if where_clauses:
            parts.append(f"WHERE {' AND '.join(where_clauses)}")

        # GROUP BY（multi_dim 或 aggregation 时）
        group_cols = self._build_group_by(analysis_type, dim_cols, select_cols, indicators)
        if group_cols:
            parts.append(f"GROUP BY {', '.join(group_cols)}")

        # ORDER BY（校验排序列是否存在）
        if sort_order:
            order_col_exists = False
            for t in tables:
                if _extract_sort_col(sort_order) in PHYSICAL_TABLE_COLUMNS.get(t, set()):
                    order_col_exists = True
                    break
            if order_col_exists:
                order_sql = self._build_order_by(sort_order, indicators)
                if order_sql:
                    parts.append(order_sql)
            else:
                logger.info("SQLBuilder 跳过 ORDER BY: 排序列 '%s' 不存在于任何 JOIN 表",
                            _extract_sort_col(sort_order))

        # LIMIT（多表 JOIN 时不自动添加 LIMIT，避免截断分析数据）
        if top_k is not None:
            parts.append(f"LIMIT {top_k}")
        elif analysis_type == "rank" and top_k is None and len(tables) <= 1:
            parts.append("LIMIT 10")

        sql = " ".join(parts)
        logger.info("SQLBuilder 构建 SQL: %.150s", sql)
        return sql

    def _build_select_columns(
        self,
        indicators: list[str],
        time_range: dict,
        analysis_type: str,
        dim_cols: list[str],
        tables: list[str],
    ) -> list[str]:
        """构建 SELECT 列。

        包括：indicators + 时间列 + 维度列。
        当 indicator 来自非主表的 JOIN 表时，自动加表名前缀。
        """
        cols: list[str] = []
        time_col: str | None = None
        primary_table = tables[0] if tables else ""
        has_joins = len(tables) > 1

        # 列名 → 所属表名 映射（用于跨表列前缀）
        table_prefix_cache: dict[str, str] = {}
        for t in tables:
            for c in PHYSICAL_TABLE_COLUMNS.get(t, set()):
                if c not in table_prefix_cache:
                    table_prefix_cache[c] = t

        def _maybe_prefix(col_name: str) -> str:
            """如果列不在主表中，加表名前缀。"""
            cleaned = col_name.strip('"\' ')
            if not cleaned:
                return col_name
            if primary_table and cleaned in PHYSICAL_TABLE_COLUMNS.get(primary_table, set()):
                # 多表 JOIN 时即使列在主表中也加前缀避免歧义
                if has_joins:
                    return f"{self._quote_col(primary_table)}.{self._quote_col(col_name)}"
                return self._quote_col(col_name)
            # 列在 JOIN 表中才加前缀
            col_table = table_prefix_cache.get(cleaned)
            if col_table and col_table != primary_table:
                return f"{self._quote_col(col_table)}.{self._quote_col(col_name)}"
            return self._quote_col(col_name)

        # indicators → 列名（通过 indicator_registry 解析为物理列名）
        for ind in indicators:
            resolved_ind = ind
            # 检查该 indicator 是否在任意参与 JOIN 的表的列集中
            ind_clean = ind.strip().strip('"\' ')
            in_any_table = any(
                ind_clean in PHYSICAL_TABLE_COLUMNS.get(t, set())
                for t in tables
            )
            if not in_any_table:
                # 通过 indicator_registry 反向查找物理列名
                try:
                    from app.engine.indicator_registry import indicator_registry
                    physical = indicator_registry.search_field_by_name(ind)
                    if physical:
                        # 确认物理列名存在于表中
                        for t in tables:
                            if physical in PHYSICAL_TABLE_COLUMNS.get(t, set()):
                                resolved_ind = physical
                                logger.info("indicator '%s' → 物理列 '%s'", ind, physical)
                                break
                except Exception:
                    pass
            cols.append(_maybe_prefix(resolved_ind))

        # 时间列（trend 类型必须含时间列）
        if analysis_type in ("trend", "correlation", "composite", "cross_domain", "anomaly"):
            time_col = self._guess_time_col(time_range, primary_table)
            if time_col and time_col not in indicators:
                cols.append(_maybe_prefix(time_col))

        # 维度列（multi_dim 专用）
        if analysis_type == "multi_dim" and dim_cols:
            for col in dim_cols:
                cleaned = col.strip().strip('"\' ')
                if cleaned and cleaned != "*" and cleaned not in indicators and cleaned != time_col:
                    cols.append(self._quote_col(col))

        # 实体标识列自动补全：当查询涉及区域对比时，确保包含 区划名称
        entity_compare_types = {"rank", "correlation", "composite", "spatial", "cross_domain", "anomaly", "trend"}
        if analysis_type in entity_compare_types and "admin_region_data" in tables:
            name_col = "区划名称"
            # 检查是否已在 select 列表中（含表名前缀的情况）
            already_has_name = any(
                name_col in c for c in cols
            )
            if not already_has_name:
                cols.append(_maybe_prefix(name_col))

        return cols

    def _build_joins(self, tables: list[str], primary_table: str) -> list[str]:
        """构建 JOIN 子句。"""
        join_lines: list[str] = []

        # 从 join_instructions 解析 JOIN 条件（用 _generic_join 兜底）
        if self._join_instructions:
            for other in tables[1:]:
                join_lines.append(self._generic_join(primary_table, other))
            return join_lines

        # 无 join_instructions 时，多表 JOIN 使用通用 FK 关联
        for other in tables[1:]:
            join_lines.append(self._generic_join(primary_table, other))
        return join_lines

    @staticmethod
    def _generic_join(t1: str, t2: str) -> str:
        """通用 FK JOIN — 区划ID 关联，两表都有 统计年份 时追加年份对齐。"""
        t1_cols = PHYSICAL_TABLE_COLUMNS.get(t1, set())
        t2_cols = PHYSICAL_TABLE_COLUMNS.get(t2, set())
        has_yr = "统计年份" in t1_cols and "统计年份" in t2_cols
        on_clause = f"{_quote_col(t1)}.\"区划ID\" = {_quote_col(t2)}.\"区划ID\""
        if has_yr:
            on_clause += f" AND {_quote_col(t1)}.\"统计年份\" = {_quote_col(t2)}.\"统计年份\""
        return f"LEFT JOIN {_quote_col(t2)} ON {on_clause}"

    def _build_where(
        self,
        time_range: dict,
        filters: list[str],
        primary_table: str = "",
        tables: list[str] | None = None,
    ) -> list[str]:
        """构建 WHERE 条件。"""
        clauses: list[str] = []
        has_joins = tables and len(tables) > 1

        # 时间范围
        start = time_range.get("start", "")
        end = time_range.get("end", "")
        time_col = self._guess_time_col(time_range, primary_table) or "统计年份"

        # 检查时间列是否存在于主表中（admin_region_data 等表无时间列）
        has_time_col = False
        if primary_table and primary_table in PHYSICAL_TABLE_COLUMNS:
            has_time_col = time_col in PHYSICAL_TABLE_COLUMNS[primary_table]

        if has_time_col and start and end:
            # 清洗时间值：对于 统计年份（integer），从日期字符串中提取年份
            if time_col == "统计年份":
                if start and "-" in start:
                    start = start[:4]
                if end and "-" in end:
                    end = end[:4]
            # JOIN 时加表名前缀避免歧义
            time_col_ref = f"{self._quote_col(primary_table)}.{self._quote_col(time_col)}" if has_joins else self._quote_col(time_col)
            clauses.append(f"{time_col_ref} >= '{start}' AND {time_col_ref} <= '{end}'")
        elif has_time_col and start:
            time_col_ref = f"{self._quote_col(primary_table)}.{self._quote_col(time_col)}" if has_joins else self._quote_col(time_col)
            clauses.append(f"{time_col_ref} >= '{start}'")
        elif has_time_col and end:
            time_col_ref = f"{self._quote_col(primary_table)}.{self._quote_col(time_col)}" if has_joins else self._quote_col(time_col)
            clauses.append(f"{time_col_ref} <= '{end}'")

        # filters（自由文本，尝试结构化解析）
        for f in filters:
            structured = self._parse_filter(f)
            if structured:
                clauses.append(structured)

        return clauses

    @staticmethod
    def _parse_filter(filter_text: str) -> str | None:
        """尝试将自由文本 filter 解析为 SQL WHERE 条件。

        支持模式：
        - "广东省" → "省区名称" = '广东省'
        - "GDP > 1000" → gdp > 1000
        - "广东省且GDP>1000" → 复杂条件（暂不处理）
        """
        text = filter_text.strip()

        # 简单值过滤（如 "广东省" → 包含省名的条件）
        province_match = re.match(r'^(.{2,10}?(?:省|市|区|自治区))$', text)
        if province_match:
            return f"\"区划名称\" = '{province_match.group(1)}'"

        # 简单条件（如 "GDP>1000"）
        cond_match = re.match(r'^(\w+)\s*([><=!]+)\s*([\d.]+)$', text)
        if cond_match:
            col, op, val = cond_match.groups()
            return f"{_quote_col(col)} {op} {val}"

        return None

    @staticmethod
    def _guess_time_col(time_range: dict, primary_table: str = "") -> str | None:
        """猜测时间列名。

        根据主表名从 TABLE_TIME_COL_MAP 中查找实际时间列名。
        env_monitor_data → 监测日期，其它表 → 统计年份。
        """
        if not time_range:
            return None
        if primary_table in TABLE_TIME_COL_MAP:
            return TABLE_TIME_COL_MAP[primary_table]
        return "统计年份"

    @staticmethod
    def _build_group_by(
        analysis_type: str,
        dim_cols: list[str],
        select_cols: list[str],
        indicators: list[str],
    ) -> list[str]:
        """构建 GROUP BY 列。"""
        if analysis_type == "multi_dim" and dim_cols:
            return [c for c in dim_cols if c in select_cols]

        # 有 aggregation 时，GROUP BY 非指标列
        if analysis_type in ("rank", "detail"):
            return []

        return []

    @staticmethod
    def _build_order_by(sort_order: str, indicators: list[str]) -> str | None:
        """从 sort_order 构建 ORDER BY 子句。"""
        sort_lower = sort_order.lower()
        if "降序" in sort_lower or "desc" in sort_lower:
            direction = "DESC"
        elif "升序" in sort_lower or "asc" in sort_lower:
            direction = "ASC"
        else:
            direction = "DESC"

        # 提取排序列名
        for ind in indicators:
            if ind.lower() in sort_lower:
                return f"ORDER BY {_quote_col(ind)} {direction}"

        # 从文本中提取
        match = re.search(r'按(.+?)(?:降序|升序|$)', sort_order)
        if match:
            col_text = match.group(1).strip()
            return f"ORDER BY {_quote_col(col_text)} {direction}"

        return None

    @staticmethod
    def _quote_col(name: str) -> str:
        """为列名加双引号（如果需要）。"""
        import re
        if not name:
            return ""

        # 去掉已有引号
        cleaned = name.strip('"`\' ')
        if not cleaned:
            return ""

        # 纯数字 → 加引号
        if cleaned.replace('.', '', 1).lstrip('-').isdigit():
            return f'"{cleaned}"'

        # 包含中文或大写字母 → 加引号
        if re.search(r'[一-鿿]', cleaned) or cleaned != cleaned.lower():
            return f'"{cleaned}"'

        return cleaned


def _quote_col(name: str) -> str:
    """外部可用的列名引用函数。"""
    return SQLBuilder._quote_col(name)


# =============================================================================
# 内部工具函数
# =============================================================================


def _extract_sort_col(sort_order: str) -> str:
    """从 sort_order 文本中提取排序列名。"""
    if not sort_order:
        return ""
    # 尝试匹配 "按XX降序" 模式
    m = re.search(r'按(.+?)(?:降序|升序|$)', sort_order)
    if m:
        return m.group(1).strip(' "\'')
    # 直接返回文本（可能是列名本身）
    return sort_order.strip(' "\'')
