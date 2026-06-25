"""Parser 输出校验层 — 表名列名校验 + 自动修复。

在 Parser 输出进入 DAGBuilder 之前执行。
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# 语义表名 → 物理表名映射
TABLE_SYNONYM_MAP: dict[str, str] = {
    "macro_economy": "economic_indicator_data",
    "macroeconomic": "economic_indicator_data",
    "economy": "economic_indicator_data",
    "economic": "economic_indicator_data",
    "gdp": "economic_indicator_data",
    "real_estate": "real_estate_data",
    "realestate": "real_estate_data",
    "housing": "real_estate_data",
    "population": "population_data",
    "employment": "employment_data",
    "environment": "env_monitor_data",
    "env": "env_monitor_data",
    "medical": "medical_health_data",
    "healthcare": "medical_health_data",
    "education": "edu_data",
    "transport": "transport_data",
    "enterprise": "enterprise_data",
    "company": "enterprise_data",
    "admin_region": "admin_region_data",
    "region": "admin_region_data",
    "province": "admin_region_data",
    # 常见 LLM 幻觉表名映射
    "economic_environment": "env_monitor_data",
    "economic_environment_data": "env_monitor_data",
    "economic_env": "env_monitor_data",
    "social_economy": "economic_indicator_data",
    "social_economic": "economic_indicator_data",
    "public_service": "medical_health_data",
    "public_health": "medical_health_data",
    "composite_index": "economic_indicator_data",
    # === 扩展映射 ===
    "provinces": "admin_region_data",
    "provinces_data": "admin_region_data",
    "regions": "admin_region_data",
    "region_data": "admin_region_data",
    "district": "admin_region_data",
    "districts": "admin_region_data",
    "admin_regions": "admin_region_data",
    "population_data_table": "population_data",
    "populations": "population_data",
    "demographic": "population_data",
    "demographics": "population_data",
    "employment_data_table": "employment_data",
    "job": "employment_data",
    "jobs": "employment_data",
    "unemployment": "employment_data",
    "economic_indicators": "economic_indicator_data",
    "economic_indicator": "economic_indicator_data",
    "economy_data": "economic_indicator_data",
    "gdp_data": "economic_indicator_data",
    "finance": "economic_indicator_data",
    "fiscal": "economic_indicator_data",
    "invest": "economic_indicator_data",
    "investment": "economic_indicator_data",
    "env_data": "env_monitor_data",
    "environment_data": "env_monitor_data",
    "environmental": "env_monitor_data",
    "env_monitor": "env_monitor_data",
    "air_quality": "env_monitor_data",
    "pollution": "env_monitor_data",
    "medical_data": "medical_health_data",
    "health": "medical_health_data",
    "hospitals": "medical_health_data",
    "health_data": "medical_health_data",
    "medical_health": "medical_health_data",
    "edu_data_table": "edu_data",
    "school": "edu_data",
    "schools": "edu_data",
    "education_data": "edu_data",
    "transport_data_table": "transport_data",
    "transportation": "transport_data",
    "traffic": "transport_data",
    "infrastructure": "transport_data",
    "real_estate_data_table": "real_estate_data",
    "property": "real_estate_data",
    "realestate_data": "real_estate_data",
    "enterprise_data_table": "enterprise_data",
    "companies": "enterprise_data",
    "business": "enterprise_data",
    "firm": "enterprise_data",
    "firms": "enterprise_data",
    "industry": "enterprise_data",
    "industries": "enterprise_data",
    "social_welfare": "medical_health_data",
    "welfare": "medical_health_data",
    "human_capital": "edu_data",
    "labor": "employment_data",
    "labour": "employment_data",
    "manpower": "employment_data",
    "human_resource": "employment_data",
    "urbanization": "population_data",
    "urban": "population_data",
    "urban_rural": "population_data",
    "city": "admin_region_data",
    "cities": "admin_region_data",
}

# 已知的所有物理表名列表（sql_builder 中也会用到）
ALL_PHYSICAL_TABLES: set[str] = {
    "economic_indicator_data",
    "admin_region_data",
    "population_data",
    "employment_data",
    "env_monitor_data",
    "real_estate_data",
    "medical_health_data",
    "edu_data",
    "transport_data",
    "enterprise_data",
}


# =============================================================================
# 工具函数: 从 SQL 文本提取结构化信息
# =============================================================================


def _extract_select_columns_from_sql(sql: str) -> list[str]:
    """从 SQL 的 SELECT 子句中提取列名列表。"""
    import re
    # 去掉 CTE
    cleaned = re.sub(
        r'WITH\s+.*?\)\s*', '', sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    m = re.search(
        r'SELECT\s+(.*?)\s+FROM',
        cleaned, flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return []
    cols: list[str] = []
    for part in m.group(1).split(','):
        part = part.strip().strip('"`\' ')
        if not part:
            continue
        # 处理 AS 别名
        as_m = re.search(r'\bAS\s+["`]?(\w+)["`]?', part, flags=re.IGNORECASE)
        if as_m:
            cols.append(as_m.group(1))
        else:
            # 去掉表前缀 "table.col" → "col"
            col = part.split('.')[-1] if '.' in part else part
            cols.append(col.strip('"`\' '))
    return cols


def _extract_tables_from_sql(sql: str) -> list[str]:
    """从 SQL 的 FROM/JOIN 子句中提取表名列表。"""
    import re
    tables: set[str] = set()
    for m in re.finditer(
        r'(?:FROM|JOIN)\s+["`]?(\w+)["`]?\s*(?:AS\s+["`]?\w+["`]?)?(?:\s+|$)',
        sql, re.IGNORECASE,
    ):
        tables.add(m.group(1).lower())
    return list(tables)


class PlanValidator:
    """Parser 输出校验器。

    职责：
      1. 表名校验：tables 是否真实存在
      2. 语义→物理表名映射
      3. raw_sql 表名一致性：FROM/JOIN 中的表名必须在 tables 中
      4. dim_cols 清洗：去掉 '*'、过滤不存在的列
    """

    @staticmethod
    def validate_tables(tables: list[str]) -> list[str]:
        """校验表名，返回修复后的表名列表。

        将语义表名映射为物理表名，移除不存在的表。
        """
        fixed: list[str] = []
        for t in tables:
            lower = t.lower().strip()
            if lower in ALL_PHYSICAL_TABLES:
                fixed.append(t)
            elif lower in TABLE_SYNONYM_MAP:
                real = TABLE_SYNONYM_MAP[lower]
                logger.info("表名校验: '%s' → '%s'", t, real)
                fixed.append(real)
            else:
                # 尝试模糊匹配
                matched = PlanValidator._fuzzy_table_match(lower)
                if matched:
                    logger.info("表名校验模糊匹配: '%s' → '%s'", t, matched)
                    fixed.append(matched)
                else:
                    logger.warning("表名校验: 未知表名 '%s'，已移除", t)
        return fixed

    @staticmethod
    def _fuzzy_table_match(lower_name: str) -> str | None:
        """模糊匹配表名。"""
        for physical in ALL_PHYSICAL_TABLES:
            # 前缀匹配: "macro" in "economic_indicator_data"
            if lower_name in physical.lower().replace("_", ""):
                return physical
            # 反向前缀匹配
            if physical.replace("_", "") in lower_name:
                return physical
        return None

    @staticmethod
    def validate_raw_sql_tables(sql: str, tables: list[str]) -> tuple[str | None, list[str]]:
        """校验 raw_sql 中的 FROM/JOIN 表名是否在 tables 中。

        Returns:
            (修复后的 SQL, 补充的表名列表)。若无法修复返回 (None, [])。
        """
        if not sql:
            return None, []

        # 提取 SQL 中的表名（FROM/JOIN 后的标识符）
        sql_upper = sql.upper()
        used_tables: set[str] = set()

        # 匹配 FROM/JOIN 后的表名
        for m in re.finditer(
            r'(?:FROM|JOIN)\s+["`]?(\w+)["`]?\s*(?:AS\s+["`]?\w+["`]?)?(?:\s+|$)',
            sql_upper,
            re.IGNORECASE,
        ):
            used_tables.add(m.group(1).lower())

        table_lower = {t.lower() for t in tables}

        # 检查是否有不存在的表
        invalid = used_tables - table_lower
        if not invalid:
            return sql, []  # 全部合法

        # 尝试修复：将语义表名替换为物理表名，或补充缺失的物理表
        fixed_sql = sql
        added_tables: list[str] = []
        for bad in sorted(invalid):
            if bad in TABLE_SYNONYM_MAP:
                good = TABLE_SYNONYM_MAP[bad]
                logger.info("SQL表名修复: '%s' → '%s'", bad, good)
                fixed_sql = re.sub(
                    rf'\b{re.escape(bad)}\b',
                    good,
                    fixed_sql,
                    flags=re.IGNORECASE,
                )
                if good.lower() not in table_lower:
                    added_tables.append(good)
            elif bad in ALL_PHYSICAL_TABLES:
                # SQL 使用了有效的物理表名但不在 tables 中 → 补充到 tables
                logger.info("SQL使用了有效物理表 '%s'，不在tables中，已记录补充", bad)
                added_tables.append(bad)
            else:
                # 尝试模糊匹配
                fuzzy = PlanValidator._fuzzy_table_match(bad)
                if fuzzy:
                    logger.info("SQL表名模糊匹配修复: '%s' → '%s'", bad, fuzzy)
                    fixed_sql = re.sub(
                        rf'\b{re.escape(bad)}\b',
                        fuzzy,
                        fixed_sql,
                        flags=re.IGNORECASE,
                    )
                    if fuzzy.lower() not in table_lower:
                        added_tables.append(fuzzy)
                else:
                    logger.warning("SQL中使用了不在tables中的表名: '%s'，无法修复", bad)
                    return None, []  # 无法修复

        return fixed_sql, added_tables

    @staticmethod
    def clean_dim_cols(dim_cols: list[str]) -> list[str]:
        """清洗 dim_cols：去掉 '*'、空字符串、纯数字（作为列名使用时）。

        注意：纯数字字符串如 "100" 如果同时也是 DataFrame 的一列则不应当被过滤，
        此处仅清除明显无效的列名。
        """
        cleaned: list[str] = []
        for col in dim_cols:
            col_stripped = col.strip().strip('"\' ')
            if not col_stripped:
                continue
            if col_stripped == "*":
                logger.info("dim_cols 清洗: 移除 '*'")
                continue
            cleaned.append(col)
        return cleaned

    @staticmethod
    def _enrich_execution_plan(execution_plan: dict[str, Any]) -> dict[str, Any]:
        """补充 execution_plan 中可能缺失的关键结构化字段。

        Parser 输出的结构化字段经常不完整（indicators 缺项、dim_cols 脏数据等）。
        本方法基于 raw_sql 和 analysis_type 补充缺失信息，使 SQLBuilder
        和 DAGBuilder 能拿到完整的结构化字段。

        重要：先扩充 tables（从 raw_sql FROM/JOIN），再扩充 indicators
        （对照 PHYSICAL_TABLE_COLUMNS 确认列存在于某张表的列集中），
        避免引入不属于当前表的列。

        Returns:
            补充后的 execution_plan（原地修改）。
        """
        raw_sql: str | None = execution_plan.get("raw_sql")
        indicators: list[str] = execution_plan.get("indicators", [])
        tables: list[str] = execution_plan.get("tables", [])
        dim_cols: list[str] = execution_plan.get("dim_cols", [])

        # Step 1: 优先扩充 tables — 从 raw_sql FROM/JOIN 提取未登记的表名
        if raw_sql:
            sql_tables = _extract_tables_from_sql(raw_sql)
            existing_tables = {t.lower() for t in tables}
            added_tables: list[str] = []
            for st in sql_tables:
                if st.lower() not in existing_tables:
                    logger.info("enrich tables: 从 raw_sql 补充 '%s'", st)
                    added_tables.append(st)
                    existing_tables.add(st.lower())
            if added_tables:
                tables = tables + added_tables
                execution_plan["tables"] = tables

        # Step 1.5: dim_cols 补充 — 当 dim_cols 为 ["*"] 或被清空时，从 raw_sql SELECT 推断
        # 注意：必须在 indicators 补充之前执行，否则被加入 indicators 的列
        # 会被 _infer_dim_cols_DAGBuilder 排除（该函数排除 indicators 中的列）。
        if raw_sql:
            has_star = any(c and c.strip() == "*" for c in dim_cols)
            if has_star or not dim_cols:
                sql_cols = _extract_select_columns_from_sql(raw_sql)
                existing_indicator_set = {i.strip().strip('"\' ').lower() for i in indicators}
                # 从 SELECT 列中找出不在 indicators 中、且不是时间列、且非聚合表达式
                # 的列作为维度列
                DIM_TIME_LIKE = {n.lower() for n in {
                    "year", "年份", "统计年份", "quarter", "季度",
                    "month", "月", "period", "time",
                }}
                inferred_dims = []
                for sc in sql_cols:
                    sc_clean = sc.strip().strip('"\' ')
                    if not sc_clean:
                        continue
                    if sc_clean.lower() in existing_indicator_set:
                        continue
                    if sc_clean.lower() in DIM_TIME_LIKE:
                        continue
                    # 跳过明显的聚合表达式（包含括号的函数调用）
                    if "(" in sc_clean and ")" in sc_clean:
                        continue
                    inferred_dims.append(sc_clean)
                if inferred_dims and (has_star or not execution_plan.get("dim_cols")):
                    logger.info("enrich dim_cols: 从 raw_sql 推断维度列 → %s", inferred_dims)
                    execution_plan["dim_cols"] = inferred_dims
                    dim_cols = inferred_dims  # 更新局部变量，供后续步骤使用

        # Step 2: indicators 补充 — 从 raw_sql SELECT 提取，但只加物理存在的列
        # 注意：必须跳过时间列（如 "统计年份"、"year"），否则会破坏
        # _infer_time_column() 的时间列推断逻辑（它通过排除 indicators 来推断时间列）
        if raw_sql:
            from app.engine.sql_builder import PHYSICAL_TABLE_COLUMNS
            TIME_LIKE_NAMES = {n.lower() for n in {
                "year", "年份", "统计年份", "quarter", "季度",
                "month", "月", "period", "time",
            }}
            sql_indicators = _extract_select_columns_from_sql(raw_sql)
            if sql_indicators:
                existing_set = {i.strip().strip('"\' ').lower() for i in execution_plan.get("indicators", [])}
                # 收集已推断的 dim_cols（大小写不敏感）
                enriched_dim_cols_set = {c.lower() for c in execution_plan.get("dim_cols", [])}
                new_inds: list[str] = []
                # 收集所有当前表（含刚补充的表）的列集
                all_table_cols: set[str] = set()
                for t in tables:
                    all_table_cols.update(PHYSICAL_TABLE_COLUMNS.get(t, set()))
                for si in sql_indicators:
                    si_clean = si.strip().strip('"\' ')
                    if not si_clean or si_clean.lower() in existing_set:
                        continue
                    # 跳过时间列 — 它们会被 _infer_time_column 单独推断
                    if si_clean.lower() in TIME_LIKE_NAMES:
                        continue
                    # 跳过已经被推断为 dim_cols 的列
                    if si_clean.lower() in enriched_dim_cols_set:
                        continue
                    # 只补充在当前表列集中真实存在的列
                    # 避免引入 raw_sql 中写错了但数据库中不存在的列名
                    if si_clean in all_table_cols:
                        logger.info("enrich indicators: 从 raw_sql 补充 '%s' (物理列)", si_clean)
                        new_inds.append(si_clean)
                        existing_set.add(si_clean.lower())
                if new_inds:
                    execution_plan["indicators"] = indicators + new_inds

        # 3. dim_cols 清洗：移除 "*" 和空字符串
        if dim_cols:
            cleaned = [c for c in dim_cols
                       if c and c.strip() and c.strip() != "*"]
            if len(cleaned) != len(dim_cols):
                logger.info("dim_cols 清洗: 移除无效项 → %s", cleaned)
            execution_plan["dim_cols"] = cleaned

        return execution_plan

    @staticmethod
    def validate_plan(execution_plan: dict[str, Any]) -> dict[str, Any]:
        """对 execution_plan 执行完整的校验+修复。

        Returns:
            修复后的 execution_plan（原地修改）。
        """
        plan = execution_plan

        # 0. 结构化字段完整性补充（改造4.1）
        plan = PlanValidator._enrich_execution_plan(plan)

        # 1. 表名校验
        tables = plan.get("tables", [])
        if tables:
            plan["tables"] = PlanValidator.validate_tables(tables)

        # 2. raw_sql 校验 — 同时获取补充表名
        raw_sql = plan.get("raw_sql", "")
        if raw_sql:
            fixed_sql, added_tables = PlanValidator.validate_raw_sql_tables(raw_sql, plan["tables"])
            if fixed_sql is not None and fixed_sql != raw_sql:
                plan["raw_sql"] = fixed_sql
            if added_tables:
                existing = {t.lower() for t in plan["tables"]}
                for t in added_tables:
                    if t.lower() not in existing:
                        plan["tables"].append(t)
                        existing.add(t.lower())
                        logger.info("补充表名到 execution_plan: '%s'", t)

        # 3. dim_cols 清洗
        dim_cols = plan.get("dim_cols", [])
        if dim_cols:
            plan["dim_cols"] = PlanValidator.clean_dim_cols(dim_cols)

        return plan
