"""DAG 编排引擎。

四组件架构：
  DAGPlan         — 节点集合 + 邻接依赖 + Kahn 分层结果
  DAGBuilder      — Parser 输出 → DAGPlan（节点图 + 拓扑分层）
  DAGExecutor     — 层次化并行调度（层级内并发 + 层级间 Barrier）
  DAGOrchestrator — 总入口（build_plan / execute / replan）
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from enum import Enum
from typing import Any

import pandas as pd

from app.engine.executor import NodeExecutor, SQLValidationError, _validate_sql
from app.engine.registry import AnalyzerRegistry
from app.engine.tool_base import BaseActionTool
from app.engine.plan_validator import PlanValidator
from app.engine.sql_builder import SQLBuilder
from app.core.llm import chat_with_model
from config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# 1. 枚举 & 异常
# =============================================================================


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    REPLANNED = "replanned"


class DAGError(Exception):
    """DAG 相关异常基类。"""


class DAGCycleError(DAGError):
    """DAG 中存在环。"""


class DAGTimeoutError(DAGError):
    """DAG 整体执行超时。"""


# =============================================================================
# 2. 数据模型
# =============================================================================


class BaseNode:
    """DAG 节点基类。

    Attributes:
        node_id:      节点唯一标识，如 "sql_1" / "analysis_cagr" / "merge_1"。
        name:         节点显示名。
        node_type:    "sql" / "analysis" / "merge"。
        depends_on:   上游节点 ID 列表。
        timeout:      节点超时秒数。
        status:       执行状态。
        error:        错误信息。
        latency_ms:   执行耗时。
        output_shape: 输出概览（如 "100x5"）。
    """

    def __init__(
        self,
        node_id: str,
        name: str,
        node_type: str,
        depends_on: list[str] | None = None,
        timeout: int = 30,
    ) -> None:
        self.node_id = node_id
        self.name = name
        self.node_type = node_type
        self.depends_on = depends_on or []
        self.timeout = timeout
        self.status: NodeStatus = NodeStatus.PENDING
        self.error: str | None = None
        self.latency_ms: float = 0.0
        self.output_shape: str | None = None


class SQLNode(BaseNode):
    """SQL 查询节点。"""

    def __init__(
        self,
        node_id: str,
        name: str,
        sql: str,
        depends_on: list[str] | None = None,
        timeout: int = 30,
    ) -> None:
        super().__init__(node_id, name, "sql", depends_on, timeout)
        self.sql = sql


class AnalysisNode(BaseNode):
    """分析算法节点。"""

    def __init__(
        self,
        node_id: str,
        name: str,
        algorithm_name: str,
        data_source: str,
        params: dict[str, Any] | None = None,
        depends_on: list[str] | None = None,
        timeout: int = 15,
        required_columns: list[str] | None = None,
    ) -> None:
        super().__init__(node_id, name, "analysis", depends_on, timeout)
        self.algorithm_name = algorithm_name
        self.data_source = data_source
        self.params = params or {}
        self.required_columns = required_columns or []


class MergeNode(BaseNode):
    """结果融合节点。

    merge_strategy:
      "concat"  — 行拼接（同结构 DataFrame）
      "join"    — 按 merge_key 列合并
      "collect" — 不合并，以 dict[source_id, df] 原样返回
    """

    def __init__(
        self,
        node_id: str,
        name: str,
        data_sources: list[str],
        merge_strategy: str = "collect",
        merge_key: str | None = None,
        depends_on: list[str] | None = None,
        timeout: int = 5,
    ) -> None:
        super().__init__(node_id, name, "merge", depends_on, timeout)
        self.data_sources = data_sources
        self.merge_strategy = merge_strategy
        self.merge_key = merge_key


class DAGPlan:
    """DAG 执行计划。

    Attributes:
        nodes:        所有节点（按 node_id 索引）。
        entry_nodes:  入度为 0 的节点 ID 列表。
        level_groups: Kahn 分层结果，每层可并行执行。
    """

    def __init__(self, nodes: list[BaseNode]) -> None:
        self._node_map: dict[str, BaseNode] = {n.node_id: n for n in nodes}
        self.entry_nodes: list[str] = []
        self.level_groups: list[list[str]] = []
        self.metadata: dict[str, Any] = {}

    def get_node(self, node_id: str) -> BaseNode:
        node = self._node_map.get(node_id)
        if node is None:
            raise DAGError(f"未知节点: '{node_id}'")
        return node

    @property
    def nodes(self) -> list[BaseNode]:
        return list(self._node_map.values())

    @property
    def size(self) -> int:
        return len(self._node_map)

    def get_final_node_ids(self) -> list[str]:
        """返回拓扑序中最后层的节点 ID（即最终结果节点）。"""
        return list(self.level_groups[-1]) if self.level_groups else []

    def get_downstream(self, node_id: str) -> list[str]:
        """返回 node_id 的所有直接下游节点。"""
        results: list[str] = []
        for n in self.nodes:
            if node_id in n.depends_on:
                results.append(n.node_id)
        return results


class NodeExecution:
    """单节点执行记录，供外部查询使用。"""

    def __init__(
        self,
        node_id: str,
        status: NodeStatus = NodeStatus.PENDING,
        latency_ms: float = 0.0,
        error: str | None = None,
        output_rows: int | None = None,
    ) -> None:
        self.node_id = node_id
        self.status = status
        self.latency_ms = latency_ms
        self.error = error
        self.output_rows = output_rows


class ExecutionResult:
    """DAG 整体执行结果。"""

    def __init__(
        self,
        success: bool = False,
        dag_status: str = "failed",
        data_warning: bool = False,
        total_latency_ms: float = 0.0,
        final_data: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
        raw_data: pd.DataFrame | None = None,
        error: str | None = None,
    ) -> None:
        self.success = success
        self.dag_status = dag_status  # "full" | "partial" | "failed"
        self.data_warning = data_warning
        self.total_latency_ms = total_latency_ms
        self.final_data = final_data
        self.raw_data = raw_data
        self.error = error
        self.nodes_execution: list[NodeExecution] = []
        # 分析节点中间结果 keyed by node_id → dict 摘要
        self.analysis_results: dict[str, dict[str, Any]] = {}
        # 已执行的算法名列表（用于 interpretation 引用）
        self._algorithm_names: list[str] = []

    def add_node_execution(self, ne: NodeExecution) -> None:
        self.nodes_execution.append(ne)


# =============================================================================
# 3. analysis_type → 算法列表映射
# =============================================================================

ANALYSIS_TYPE_MAP: dict[str, list[str]] = {
    "trend": ["cagr", "linear_trend", "sma"],
    "rank": ["percent_rank", "ntile", "rank"],
    "detail": [],
    "advanced": ["pearson", "spearman", "zscore"],
    "correlation": ["pearson", "spearman", "mutual_info"],
    "anomaly": ["zscore", "iqr", "three_sigma"],
    "composite": ["minmax", "entropy_weight", "topsis"],
    "multi_dim": ["cube", "proportion"],            # 多维聚合: 交叉汇总 + 占比
    "spatial": ["percent_rank", "rank"],             # 区域对比: 基于排名的组间比较
    "cross_domain": ["minmax", "entropy_weight"],    # 跨域综合: 归一化 + 加权融合
}

# 映射中所有算法所属类别 → 节点参数默认值
ALGORITHM_CATEGORY_PARAMS: dict[str, dict[str, Any]] = {
    "cagr": {},
    "linear_trend": {},
    "sma": {"window": 3},
    "percent_rank": {},
    "ntile": {"n": 5},
    "rank": {},
    "pearson": {},
    "spearman": {},
    "mutual_info": {},
    "zscore": {"threshold": 3.0},
    "iqr": {},
    "three_sigma": {},
    "minmax": {"feature_range": (0, 1)},
    "entropy_weight": {},
    "topsis": {},
    "cube": {"method": "rollup"},
    "proportion": {},
}

# 算法参数解析映射表：声明每个算法需要从解析结果中提取哪些参数
# source 类型:
#   "indicator"      — 取 indicators[index]
#   "all_indicators" — 取整个 indicators 列表
#   "infer_time"     — 从 raw_sql 推断时间列名
ALGORITHM_PARAM_MAPPING: dict[str, list[dict[str, Any]]] = {
    # ── 单值算法: indicators[0] → value_col ──
    "rank":              [{"param": "value_col", "source": "indicator", "index": 0}],
    "percent_rank":      [{"param": "value_col", "source": "indicator", "index": 0}],
    "ntile":             [{"param": "value_col", "source": "indicator", "index": 0}],
    "zscore":            [{"param": "value_col", "source": "indicator", "index": 0}],
    "iqr":               [{"param": "value_col", "source": "indicator", "index": 0}],
    "three_sigma":       [{"param": "value_col", "source": "indicator", "index": 0}],
    # ── 双值算法: indicators[0] → x_col, indicators[1] → y_col ──
    "pearson":           [{"param": "x_col", "source": "indicator", "index": 0},
                          {"param": "y_col", "source": "indicator", "index": 1}],
    "spearman":          [{"param": "x_col", "source": "indicator", "index": 0},
                          {"param": "y_col", "source": "indicator", "index": 1}],
    "mutual_info":       [{"param": "x_col", "source": "indicator", "index": 0},
                          {"param": "y_col", "source": "indicator", "index": 1}],
    # ── 时序算法: indicators[0] → value_col + 时间列推断 ──
    "cagr":              [{"param": "value_col", "source": "indicator", "index": 0},
                          {"param": "time_col",  "source": "infer_time"}],
    "linear_trend":      [{"param": "value_col", "source": "indicator", "index": 0},
                          {"param": "time_col",  "source": "infer_time"}],
    "sma":               [{"param": "value_col", "source": "indicator", "index": 0},
                          {"param": "time_col",  "source": "infer_time"}],
    # ── 多列算法: 全量 indicators → value_cols 列表 ──
    "minmax":            [{"param": "value_cols", "source": "all_indicators"}],
    "entropy_weight":    [{"param": "value_cols", "source": "all_indicators"}],
    "topsis":            [{"param": "value_cols", "source": "all_indicators"}],
    # ── 多维聚合: indicators[0] → value_col, dim_cols → execution_plan ──
    "cube":              [{"param": "value_col", "source": "indicator", "index": 0},
                           {"param": "dim_cols", "source": "execution_plan", "key": "dim_cols"}],
    "proportion":        [{"param": "value_col", "source": "indicator", "index": 0},
                           {"param": "dim_cols", "source": "execution_plan", "key": "dim_cols"}],
}


# =============================================================================
# 4. DAGBuilder
# =============================================================================


class DAGBuilder:
    """Parser 输出 → DAGPlan。"""

    def build_plan(
        self,
        parse_result: dict[str, Any],
        analyzer_registry: AnalyzerRegistry | None = None,
    ) -> DAGPlan:
        """从 Parser 输出构建 DAGPlan。

        Args:
            parse_result: parser.py 的 parse() 返回值。
                          关键字段: ["execution_plan"]["analysis_type"],
                                    ["execution_plan"]["raw_sql"],
                                    ["execution_plan"]["indicators"],
                                    ["execution_plan"]["tables"]。

        Returns:
            DAGPlan 实例。
        """
        execution_plan: dict[str, Any] = parse_result.get("execution_plan", parse_result)
        analysis_type: str = execution_plan.get("analysis_type", "detail")
        raw_sql: str | None = execution_plan.get("raw_sql")
        indicators: list[str] = execution_plan.get("indicators", [])

        # ---- PlanValidator: 校验+修复 execution_plan ----
        execution_plan = PlanValidator.validate_plan(execution_plan)

        # ---- SQLBuilder: 尝试从结构化字段确定性构建 SQL ----
        sql_builder = SQLBuilder()
        builder_sql: str | None = None
        if sql_builder.can_build(execution_plan):
            builder_sql = sql_builder.build(execution_plan)
            if builder_sql:
                logger.info("SQLBuilder 成功构建 SQL，替代 LLM raw_sql")
                raw_sql = builder_sql
                execution_plan["raw_sql"] = raw_sql
                execution_plan["_sql_source"] = "builder"
                # 存储 builder_sql 用于运行时的 SQLBuilder 兜底（改造4.2）
                execution_plan["_builder_sql"] = builder_sql
            else:
                execution_plan["_sql_source"] = "raw_sql"
                logger.info("SQLBuilder 未能构建 SQL，回退到 LLM raw_sql")
        else:
            execution_plan["_sql_source"] = "raw_sql"
            logger.info("SQLBuilder 无法处理该 plan，使用 LLM raw_sql")
            # 尝试无备用时也存储，供运行时 fallback 使用（改造4.2）
            if not builder_sql and raw_sql and len(execution_plan.get("tables", [])) >= 2:
                try:
                    builder_sql = SQLBuilder().build(execution_plan)
                    if builder_sql:
                        execution_plan["_builder_sql"] = builder_sql
                        logger.info("存储备用 SQLBuilder SQL (can_build=False, 尝试性构建)")
                except Exception:
                    pass

        nodes: list[BaseNode] = []

        # 4a. SQL 节点 — trend 类型自动确保时间列和多期数据
        if analysis_type == "trend" and raw_sql:
            raw_sql = self._ensure_trend_sql(raw_sql, execution_plan)
            execution_plan["raw_sql"] = raw_sql
        sql_node = self._build_sql_node(raw_sql, indicators)
        nodes.append(sql_node)

        # 4b. 分析算法节点
        algo_names = ANALYSIS_TYPE_MAP.get(analysis_type, [])
        analysis_nodes = self._build_analysis_nodes(
            algo_names, sql_node.node_id, indicators, execution_plan,
        )
        nodes.extend(analysis_nodes)

        # 4c. 融合节点（仅当有多个分析节点时）
        if len(analysis_nodes) > 1:
            merge_node = self._build_merge_node(analysis_nodes)
            nodes.append(merge_node)
        elif len(analysis_nodes) == 1:
            merge_node = analysis_nodes[0]
        else:
            merge_node = sql_node

        # 4d. 拓扑排序
        plan = DAGPlan(nodes)
        plan.level_groups = self.topological_sort(plan)
        plan.entry_nodes = list(plan.level_groups[0]) if plan.level_groups else []
        # 携带 execution_plan 用于下游空数据降级等场景
        plan.metadata["execution_plan"] = execution_plan

        logger.info(
            "DAGPlan built: %d nodes, %d levels, analysis_type=%s, sql_source=%s",
            plan.size,
            len(plan.level_groups),
            analysis_type,
            execution_plan.get("_sql_source", "unknown"),
        )
        return plan

    @staticmethod
    def _try_fix_sql_from_hint(sql: str, error_msg: str) -> str | None:
        """利用 PostgreSQL HINT / 错误信息自动修复 SQL。

        支持多种修复模式：
        1. 列名错误：PostgreSQL HINT → 替换为正确列名
        2. 表名不存在 → 用已知表名模糊匹配替换
        3. 英文错误信息
        4. 别名冲突：列在错误表别名下不存在 → 交换表别名
        5. 丢失 FROM 子句项 → 自动补全缺失的表定义
        6. 日期类型转换错误 → 调整日期比较表达式
        """
        import re
        from app.engine.sql_builder import PHYSICAL_TABLE_COLUMNS

        fixed = None

        # 模式1: PostgreSQL HINT 列名修复
        # HINT: 也许您想要引用"economic_indicator_data.区划ID"
        hint_match = re.search(r'"([^"]+\.([^"]+))"', error_msg)
        # 英文: HINT: Perhaps you meant to reference the column "table.col"
        if not hint_match:
            hint_match = re.search(r'Perhaps you meant to reference the column "([^"]+\.([^"]+))"', error_msg, re.IGNORECASE)
        if hint_match:
            correct_ref = hint_match.group(1)
            correct_col = hint_match.group(2)

            # 提取错误的列名
            err_match = re.search(r'字段\s*"([^"]+)"\s*不存在', error_msg)
            if not err_match:
                err_match = re.search(r'column\s+"([^"]+)"\s+does not exist', error_msg, re.IGNORECASE)
            if err_match:
                wrong_col = err_match.group(1)
                # 如果错误列名包含表别名前缀（如 e1.失业率），尝试仅替换引号包裹版本
                fixed_sql = sql.replace(f'"{wrong_col}"', f'"{correct_col}"')
                if fixed_sql != sql:
                    fixed = fixed_sql

        # 模式2: 表名不存在 → 使用已知表名模糊匹配替换
        if not fixed:
            table_err = re.search(r'关系\s*"([^"]+)"\s*(?:不存在)|relation\s+"([^"]+)"\s+does not exist', error_msg, re.IGNORECASE)
            if table_err:
                wrong_table = table_err.group(1) or table_err.group(2)
                # 用 difflib 匹配已知表名
                known_tables = list(PHYSICAL_TABLE_COLUMNS.keys())
                import difflib
                matches = difflib.get_close_matches(wrong_table, known_tables, n=1, cutoff=0.5)
                if matches:
                    correct_table = matches[0]
                    fixed_sql = sql.replace(f'"{wrong_table}"', f'"{correct_table}"')
                    fixed_sql = fixed_sql.replace(wrong_table, correct_table)
                    if fixed_sql != sql:
                        fixed = fixed_sql
                        logger.info("表名修复: '%s' → '%s'", wrong_table, correct_table)

        # 模式3: 列在错误表名下不存在（如 e1.失业率 → 应改为 e2.失业率）
        if not fixed:
            alias_col_err = re.search(
                r'(?:字段|column)\s+"(\w+)\.(\w+)"\s+(?:不存在|does not exist)',
                error_msg,
                re.IGNORECASE,
            )
            if alias_col_err:
                bad_alias = alias_col_err.group(1)
                col_name = alias_col_err.group(2)
                aliases = set(re.findall(r'(\w+)\.', sql))
                for alt_alias in sorted(aliases, reverse=True):
                    if alt_alias != bad_alias:
                        fixed_sql = re.sub(
                            rf'\b{re.escape(bad_alias)}\.{re.escape(col_name)}\b',
                            f'{alt_alias}.{col_name}',
                            sql,
                        )
                        if fixed_sql != sql:
                            fixed = fixed_sql
                            logger.info("HINT修复: 列 '%s.%s' → '%s.%s'", bad_alias, col_name, alt_alias, col_name)
                            break

        # 模式4: 丢失 FROM 子句项（如 alias "a" 在 FROM 中未定义）
        if not fixed:
            from_err = re.search(r'(?:丢失FROM子句项|missing FROM-clause entry).*?"?(\w+)"?', error_msg, re.IGNORECASE)
            if from_err:
                missing_alias = from_err.group(1)
                # 从 SQL 中找出使用了该别名的列引用
                alias_cols = re.findall(rf'\b{re.escape(missing_alias)}\.(\w+)', sql)
                if alias_cols:
                    # 尝试为这个别名找合适的物理表
                    # 从已知表名生成别名首字母映射
                    table_alias_map = {}
                    for tbl_name in PHYSICAL_TABLE_COLUMNS:
                        alias = tbl_name.split("_")[0][0] if "_" in tbl_name else tbl_name[:1]
                        table_alias_map[alias] = tbl_name
                    if missing_alias in table_alias_map:
                        actual_table = table_alias_map[missing_alias]
                        # 在 JOIN 子句之前或 FROM 之后插入缺失的表
                        import re as _re_module
                        # 在 FROM 子句之后添加缺失的表定义
                        fixed_sql = _re_module.sub(
                            r'(FROM\s+"?\w+"?(?:\s+\w+)?)',
                            rf'\1, "{actual_table}" {missing_alias}',
                            sql,
                            count=1,
                        )
                        if fixed_sql != sql:
                            fixed = fixed_sql
                            logger.info("FROM 子句修复: 添加缺失表 '%s' AS %s", actual_table, missing_alias)

        # 模式5: 日期类型转换错误（date vs integer）
        if not fixed:
            date_err = re.search(r'(?:无法把类型 date 转换为 integer|cannot cast type date to integer)', error_msg, re.IGNORECASE)
            if date_err:
                # 尝试用 EXTRACT 或日期字面量替换整数比较
                # 如 "监测日期" >= '2022' → "监测日期" >= '2022-01-01'
                fixed_sql = re.sub(
                    r'"监测日期"\s*([><=!]+)\s*\'(\d{4})\'',
                    r'"监测日期" \1 \'\2-01-01\'',
                    sql,
                )
                if fixed_sql != sql:
                    fixed = fixed_sql
                    logger.info("日期比较修复: 将年份字面量转为完整日期")

        # 模式6: 列名带表名前缀但无此列 → 去掉表名前缀
        # 如 economic_indicator_data."GDP" 但表在 SQL 中别名是 e → 改为 e."GDP"
        if not fixed:
            prefixed_col_err = re.search(
                r'(?:字段|column)\s+"?(\w+)\.(\w+)"?\s+(?:不存在|does not exist)',
                error_msg,
                re.IGNORECASE,
            )
            if prefixed_col_err:
                table_name = prefixed_col_err.group(1)
                col_name = prefixed_col_err.group(2)
                # 收集 SQL 中所有表别名（FROM/JOIN 后面的别名）
                from re import findall as _re_findall
                aliases_in_sql = _re_findall(r'(?:FROM|JOIN)\s+"?(\w+)"?(?:\s+(\w+))?', sql)
                # 尝试用表名本身（去掉引号）或任何别名替换
                for tbl, alias in aliases_in_sql:
                    candidate = alias or tbl
                    # 尝试替换 table.col 为 alias.col
                    if candidate != table_name:
                        fixed_sql = sql.replace(f'"{table_name}.{col_name}"', f'{candidate}."{col_name}"')
                        fixed_sql = fixed_sql.replace(f'{table_name}.{col_name}', f'{candidate}.{col_name}')
                        if fixed_sql != sql:
                            fixed = fixed_sql
                            logger.info("表名前缀修复: '%s.%s' → '%s.%s'", table_name, col_name, candidate, col_name)
                            break
                # 如果所有别名都不行，直接去掉表名前缀
                if not fixed:
                    fixed_sql = re.sub(
                        rf'(?:{re.escape(table_name)}\.)?"?{re.escape(col_name)}"?',
                        f'"{col_name}"',
                        sql,
                    )
                    if fixed_sql != sql:
                        fixed = fixed_sql
                        logger.info("表名前缀剥离: 去掉 '%s.' 前缀引用列 '%s'", table_name, col_name)

        # 模式7: 列名在错误别名下 → 尝试使用 JOIN 表中实际存在的列名
        # 如 e.失业率 不存在于 economic_indicator_data → 尝试 e.失业人口
        if not fixed:
            alias_col_err2 = re.search(
                r'(?:字段|column)\s+"?(\w+)\.(\w+)"?\s+(?:不存在|does not exist)',
                error_msg,
                re.IGNORECASE,
            )
            if alias_col_err2:
                alias_name = alias_col_err2.group(1)
                col_name = alias_col_err2.group(2)
                # 查找该别名对应的物理表
                alias_to_table = {}
                from re import finditer as _re_finditer
                for m in _re_finditer(r'(?:FROM|JOIN)\s+"?(\w+)"?(?:\s+(\w+))?', sql):
                    tbl = m.group(1).strip('"')
                    als = (m.group(2) or tbl).strip('"')
                    alias_to_table[als] = tbl
                physical_table = alias_to_table.get(alias_name, "")
                if physical_table and physical_table in PHYSICAL_TABLE_COLUMNS:
                    known_cols = PHYSICAL_TABLE_COLUMNS[physical_table]
                    # 如果错误列名不在该表的列中，尝试模糊匹配
                    if col_name not in known_cols:
                        import difflib
                        fuzzy_matches = difflib.get_close_matches(col_name, list(known_cols), n=1, cutoff=0.5)
                        if fuzzy_matches:
                            correct_col = fuzzy_matches[0]
                            fixed_sql = re.sub(
                                rf'{re.escape(alias_name)}\.{re.escape(col_name)}\b',
                                f'{alias_name}.{correct_col}',
                                sql,
                            )
                            if fixed_sql != sql:
                                fixed = fixed_sql
                                logger.info("别名下列名模糊修复: '%s.%s' → '%s.%s'", alias_name, col_name, alias_name, correct_col)

        return fixed

    @staticmethod
    async def _try_fix_sql_with_llm(sql: str, error_msg: str) -> str | None:
        """利用 LLM 修复 SQL 中的错误。

        当 PostgreSQL HINT 无法自动修复时，将错误信息发给 LLM，
        让 LLM 重新生成正确的 SQL。

        Returns:
            修复后的 SQL 或 None（修复失败）。
        """
        import re
        from app.engine.sql_builder import PHYSICAL_TABLE_COLUMNS

        is_table_error = bool(re.search(r'关系.*不存在|does\s+not\s+exist', error_msg, re.IGNORECASE))
        is_column_error = bool(re.search(r'(?:字段|column).*(?:不存在|does not exist)', error_msg, re.IGNORECASE))
        is_union_error = bool(re.search(r'UNION|UNION', error_msg, re.IGNORECASE))
        table_hint = ""
        column_hint = ""
        union_hint = ""

        if is_table_error:
            table_hint = (
                "\n\n注意：你使用了不存在的表名或视图名。请确保只使用数据库实际存在的物理表名。"
                "\n实际可用的表名如下（不要猜测表名，不要使用CTE名称，只使用以下列表中的表）："
                "\n- economic_indicator_data (经济指标数据: GDP, 人均GDP, 固定资产投资, 财政收入等)"
                "\n- admin_region_data (行政区划数据: 省区名称, 区域ID, 板块)"
                "\n- population_data (人口数据: 常住人口, 城镇化率, 老龄化率等)"
                "\n- employment_data (就业数据: 失业率, 就业人口, 平均工资等)"
                "\n- env_monitor_data (环境监测数据: 森林覆盖率, PM2.5, AQI等)"
                "\n- real_estate_data (房地产数据: 商品房销售面积, 房地产投资, 房价等)"
                "\n- medical_health_data (医疗卫生数据: 千人床位数, 千人医生数, 医疗卫生支出等)"
                "\n- edu_data (教育数据: 识字率, 教育支出, 在校生数等)"
                "\n- transport_data (交通数据: 公路里程, 货运量, 客运量等)"
                "\n- enterprise_data (企业数据: 企业数量, 营收, 纳税额, 企业类型等)"
                "\n\n注意：绝对不要使用\"WITH xxx AS (SELECT...)\"定义的CTE名称作为表名引用。"
                "\n如果错误信息中提到\"关系不存在\"，检查是否是CTE名称被错误地当作了表名。"
            )

        if is_column_error:
            # 提供每张表的列名详情，帮助 LLM 确定列属于哪张表
            schema_lines = ["\n\n各表的可用列名（用于修复列不存在错误）："]
            for tbl, cols in sorted(PHYSICAL_TABLE_COLUMNS.items()):
                col_list = ", ".join(sorted(cols))
                schema_lines.append(f"  {tbl}: {col_list}")
            column_hint = "\n".join(schema_lines)

        prompt = (
            f"你是一个 PostgreSQL 专家。以下 SQL 语句执行出错，请修复。\n\n"
            f"原始 SQL:\n{sql}\n\n"
            f"错误信息:\n{error_msg}\n\n"
            f"{table_hint}{column_hint}{union_hint}"
            f"要求:\n"
            f"1. 只输出纯 SQL，不要 markdown 代码块，不要任何解释\n"
            f"2. 确保 SQL 语法正确，符合 PostgreSQL 规范\n"
            f"3. 如果错误信息提到\"关系不存在\"，检查表名是否正确，使用表名列表中的物理表名\n"
            f"4. 不要使用 CTE（WITH...SELECT...），替代使用子查询\n"
            f"5. 如果原始 SQL 使用了表别名，确保 FROM/JOIN 子句中有对应的别名定义\n"
            f"6. 如果错误是列不存在，检查该列属于哪个物理表，修正表别名前缀\n"
        )
        try:
            from app.core.llm import chat_with_model
            response = await chat_with_model("openai", [
                {"role": "system", "content": "你是一个 PostgreSQL 专家，只输出 SQL 代码。"},
                {"role": "user", "content": prompt},
            ], temperature=0.1)
            fixed = response.strip()
            if fixed.startswith("```"):
                fixed = re.sub(r'^```\w*\n?', '', fixed)
                fixed = re.sub(r'\n?```$', '', fixed)
            fixed = fixed.strip()
            if fixed and fixed.upper().startswith("SELECT"):
                return fixed
        except Exception as e:
            logger.warning("LLM SQL 修复失败: %s", e)
        return None

    def _build_sql_node(
        self,
        raw_sql: str | None,
        indicators: list[str],
    ) -> SQLNode:
        """构建 SQL 节点。

        自动修复：
        1. 聚合函数 AS 别名追加
        2. PostgreSQL UNION 语法（括号包裹）
        3. PERCENTILE_CONT WITHIN GROUP 语法
        """
        sql_text = raw_sql or ""
        if sql_text.strip():
            # 修复 1: 聚合函数 AS 别名
            if indicators:
                sql_text = self._fix_aggregate_aliases(sql_text, indicators)
            # 修复 2: UNION 语法
            sql_text = self._fix_union_syntax(sql_text)
            # 修复 3: PERCENTILE_CONT 语法
            sql_text = self._fix_percentile_cont(sql_text)
        if not sql_text.strip():
            # fallback: 按 indicator 生成 placeholder
            cols = ", ".join(indicators) if indicators else "*"
            sql_text = f"SELECT {cols} FROM /* 需 Parser 提供 raw_sql */"

        return SQLNode(
            node_id="sql_1",
            name=f"SQL 查询: {sql_text[:60]}",
            sql=sql_text,
            timeout=settings.DAG_SQL_TIMEOUT,
        )

    @staticmethod
    def _fix_aggregate_aliases(sql: str, indicators: list[str]) -> str:
        """给聚合函数包裹的指标列追加 AS 别名。

        例如：SUM("固定资产投资") → SUM("固定资产投资") AS "固定资产投资"
              但 SUM("固定资产投资") AS total 不变（已有别名）
        """
        import re
        for ind in indicators:
            # 匹配 aggregate(ind) 后面没有 AS alias 的用例
            escaped_ind = re.escape(ind)
            # 有引号版本：SUM("固定资产投资")
            pattern_quoted = re.compile(
                r'(SUM|AVG|COUNT|MAX|MIN|STRING_AGG|ARRAY_AGG)\s*\(\s*"'
                + escaped_ind + r'"\s*\)(?!\s+AS\b)',
                re.IGNORECASE,
            )
            sql = pattern_quoted.sub(r'\1("' + ind + r'") AS "' + ind + r'"', sql)

            # 无引号版本：SUM(gdp)
            pattern_unquoted = re.compile(
                r'(SUM|AVG|COUNT|MAX|MIN|STRING_AGG|ARRAY_AGG)\s*\(\s*'
                + escaped_ind + r'\s*\)(?!\s+AS\b)',
                re.IGNORECASE,
            )
            sql = pattern_unquoted.sub(r'\1(' + ind + r') AS "' + ind + r'"', sql)
        return sql

    @staticmethod
    def _fix_union_syntax(sql: str) -> str:
        """修复 PostgreSQL UNION 语法。

        PostgreSQL 规定 UNION 分支中不能使用 ORDER BY（除非该分支也有 LIMIT）。
        此方法对含 ORDER BY 但无 LIMIT 的分支移除 ORDER BY 子句（排序由最终 ORDER BY 处理）；
        对同时含 ORDER BY 和 LIMIT 的分支用括号包裹（PostgreSQL 允许此种语法）。
        """
        import re
        if not re.search(r'\bUNION\b', sql, re.IGNORECASE):
            return sql

        # 按 UNION [ALL] 分割，保留分隔符
        parts = re.split(r'(\bUNION\b(?:\s+ALL\b)?)', sql, flags=re.IGNORECASE)

        fixed_parts: list[str] = []
        for i, part in enumerate(parts):
            if re.match(r'^\s*UNION', part, re.IGNORECASE):
                fixed_parts.append(part)
                continue

            part_stripped = part.strip()
            if not part_stripped:
                fixed_parts.append(part)
                continue

            has_order = re.search(r'\bORDER\s+BY\s+', part_stripped, re.IGNORECASE)
            has_limit = re.search(r'\bLIMIT\s+\d+', part_stripped, re.IGNORECASE)

            if has_order and not has_limit:
                # 无 LIMIT → 移除最后一个 ORDER BY 子句（子查询中的 ORDER BY 不受影响）
                matches = list(re.finditer(r'\bORDER\s+BY\b', part_stripped, re.IGNORECASE))
                if matches:
                    last_ob = matches[-1]
                    before_ob = part_stripped[:last_ob.start()].rstrip()
                    fixed_parts.append(before_ob)
                else:
                    fixed_parts.append(part_stripped)
            elif has_order:
                # 有 ORDER BY + LIMIT → 必须用括号包裹（PostgreSQL 要求 UNION 分支中有 ORDER BY 时必须用括号）
                fixed_parts.append(f'({part_stripped})')
            else:
                fixed_parts.append(part_stripped)

        result = ' '.join(fixed_parts)

        # 清理多余空白
        result = re.sub(r'\s{2,}', ' ', result).strip()
        if result != sql:
            logger.info("UNION 语法已修正（移除了 UNION 分支中的 ORDER BY）")
        return result

    @staticmethod
    def _fix_percentile_cont(sql: str) -> str:
        """修复 PERCENTILE_CONT 语法为 PostgreSQL 兼容格式。

        错误: PERCENTILE_CONT(numeric, column) 或 "PERCENTILE_CONT"(numeric, column)
        正确: PERCENTILE_CONT(numeric) WITHIN GROUP (ORDER BY column)
        """
        import re
        pattern = re.compile(
            r'"?"?PERCENTILE_CONT"?\s*\(\s*([\d.]+)\s*,\s*([^)]+?)\s*\)',
            re.IGNORECASE,
        )
        fixed = pattern.sub(r'PERCENTILE_CONT(\1) WITHIN GROUP (ORDER BY \2)', sql)
        if fixed != sql:
            logger.info("PERCENTILE_CONT 语法已修复")
        return fixed

    @staticmethod
    def _fix_division_by_zero(sql: str) -> str:
        """修复 SQL 中的除以零错误：将除法 A/B 包装为 A / NULLIF(B, 0)。

        例如：GDP/"人口" → GDP / NULLIF("人口", 0)
        """
        import re
        # 匹配除法表达式：列/列 或 列/数值 或 数值/列
        pattern = re.compile(
            r'(\w+(?:\s*\|\|\s*\'[^\']*\')?)\s*/\s*(\w+)',
        )
        fixed = pattern.sub(r'\1 / NULLIF(\2, 0)', sql)
        if fixed != sql:
            logger.info("SQL 除以零已修复: 添加 NULLIF 保护")
        return fixed

    def _build_analysis_nodes(
        self,
        algo_names: list[str],
        data_source: str,
        indicators: list[str],
        execution_plan: dict[str, Any],
    ) -> list[AnalysisNode]:
        """为每个算法创建一个分析节点，按 ALGORITHM_PARAM_MAPPING 自动绑定参数。

        参数映射表会从 indicators / raw_sql 中推导出每个算法所需的
        value_col / x_col / y_col / time_col / value_cols 等参数。

        额外处理：
        - SQL 中 AS 别名 → 原始列名映射，避免算法找不到指标列
        """
        nodes: list[AnalysisNode] = []
        raw_sql: str | None = execution_plan.get("raw_sql", "")

        # 构建 AS 别名 → 原始列名映射（如 "SUM(指标) AS total" → {"total": "指标"}）
        alias_map: dict[str, str] = {}
        if raw_sql:
            import re
            # 匹配 func(...) AS alias、col AS alias、table.col AS alias
            as_pattern = re.compile(
                r'(?:(\w+)\(([^)]+)\)|(?:[\w"_]+\.)?([\w"_]+))\s+AS\s+"?(\w+)"?',
                re.IGNORECASE,
            )
            for m in as_pattern.finditer(raw_sql):
                func_name, inner_col, direct_col, alias = m.groups()
                original = (inner_col or direct_col or "").strip('"')
                if original and alias and original.lower() != alias.lower():
                    # 跳过映射到 '*' 的情况，避免将 SELECT * 的列名反向覆盖
                    if original.strip() != '*':
                        alias_map[alias] = original

        # 构建反向映射：原始列名 → 所有别名列表（用于 SQL 将原始列重命名的场景）
        reverse_alias_map: dict[str, list[str]] = {}
        for alias, original in alias_map.items():
            reverse_alias_map.setdefault(original, []).append(alias)
        if reverse_alias_map:
            logger.debug("反向别名映射: %s", reverse_alias_map)

        for algo in algo_names:
            params = dict(ALGORITHM_CATEGORY_PARAMS.get(algo, {}))
            # 按参数映射表绑定动态来源的参数
            bindings = ALGORITHM_PARAM_MAPPING.get(algo, [])
            has_unresolved = False
            for b in bindings:
                value = self._resolve_param_binding(b, indicators, execution_plan)
                if value is not None:
                    # 若 value 是 AS 别名，映射回原始列名
                    # 注意：如果 alias 本身就是 indicator 名称，说明 SQL 已将列别名对齐到 indicator，
                    # 不应再反向映射（如 COUNT("企业ID") AS "企业数量" 不应将 "企业数量" 映射回 "企业ID"）
                    _indicator_names_lower = {i.lower() for i in indicators}
                    if isinstance(value, str) and value in alias_map and value.lower() not in _indicator_names_lower:
                        params[b["param"]] = alias_map[value]
                    # 若 value 是原始列名但被 SQL 重命名（如 "失业率" AS start_unemployment），
                    # 使用别名作为参数值，让 executor 能匹配到实际 DataFrame 列名
                    elif isinstance(value, str) and value in reverse_alias_map:
                        params[b["param"]] = reverse_alias_map[value][0]
                        logger.info(
                            "列 '%s' 被 SQL 别名为 %s，使用 '%s' 作为参数值",
                            value, reverse_alias_map[value], params[b["param"]],
                        )
                    else:
                        params[b["param"]] = value
                elif b["source"] == "indicator":
                    # 从 indicators 解析的参数为 None → 该算法缺少必要指标
                    has_unresolved = True

            # dim_cols 为空或含无效占位符时兜底：从 SQL SELECT 列自动推断（排除 indicator 和 time 列）
            _dim_cols = params.get("dim_cols", [])
            _dim_cols_valid = _dim_cols and all(d.strip() != "*" for d in _dim_cols)
            if algo in ("cube", "proportion") and not _dim_cols_valid and raw_sql:
                inferred = self._infer_dim_cols(raw_sql, indicators, execution_plan)
                if inferred:
                    params["dim_cols"] = inferred
                    logger.info("自动推断多维维度列: %s", inferred)

            if has_unresolved:
                logger.info(
                    "跳过算法 '%s': 缺少必要指标 (indicators=%s)",
                    algo, indicators,
                )
                continue

            # 提取 required_columns：从 params 中收集所有列名类型的参数值
            required_columns: list[str] = []
            for pval in params.values():
                if isinstance(pval, str) and len(pval) >= 2:
                    required_columns.append(pval)
                elif isinstance(pval, list):
                    for item in pval:
                        if isinstance(item, str) and len(item) >= 2:
                            required_columns.append(item)

            nodes.append(AnalysisNode(
                node_id=f"analysis_{algo}",
                name=f"分析: {algo}",
                algorithm_name=algo,
                data_source=data_source,
                params=params,
                depends_on=[data_source],
                timeout=settings.DAG_PYTHON_TIMEOUT,
                required_columns=required_columns,
            ))
        return nodes

    @staticmethod
    def _resolve_param_binding(
        binding: dict[str, Any],
        indicators: list[str],
        execution_plan: dict[str, Any],
    ) -> Any:
        """按绑定的 source 类型从解析结果中提取参数值。"""
        source = binding.get("source", "")
        if source == "indicator":
            idx = binding.get("index", 0)
            return indicators[idx] if idx < len(indicators) else None
        if source == "all_indicators":
            return indicators if indicators else None
        if source == "infer_time":
            return DAGBuilder._infer_time_column(execution_plan)
        if source == "execution_plan":
            key = binding.get("key", "")
            return execution_plan.get(key)
        return None

    @staticmethod
    def _ensure_trend_sql(raw_sql: str, execution_plan: dict[str, Any]) -> str:
        """趋势分析：移除 SQL 中的单一年份过滤条件，确保返回多年数据。

        覆盖模式：
        1. time_col = '2023'          — 等号单年
        2. time_col BETWEEN '2023' AND '2023'  — BETWEEN 单年
        3. time_col >= '2023' AND time_col <= '2023'  — 区间单年
        """
        import re
        time_col = execution_plan.get("time_col")
        if not time_col:
            time_col = DAGBuilder._infer_time_column(execution_plan)
        if not time_col:
            return raw_sql

        quoted_col = re.escape(time_col)
        orig_sql = raw_sql
        cleaned = False

        # 模式1: time_col = '2023' — 等号单年
        year_cond = re.compile(
            rf'(?:WHERE|AND)\s+"?{quoted_col}"?\s*=\s*\'?(\d{{4}})\'?',
            re.IGNORECASE,
        )
        if year_cond.search(raw_sql):
            raw_sql = year_cond.sub('', raw_sql)
            cleaned = True

        # 模式2: time_col BETWEEN '2023' AND '2024' → 同一年时移除
        between_cond = re.compile(
            rf'(?:WHERE|AND)\s+"?{quoted_col}"?\s+BETWEEN\s+\'?(\d{{4}})\'?\s+AND\s+\'?(\d{{4}})\'?',
            re.IGNORECASE,
        )
        between_match = between_cond.search(raw_sql)
        if between_match and between_match.group(1) == between_match.group(2):
            raw_sql = between_cond.sub('', raw_sql)
            cleaned = True

        # 模式3: time_col >= '2023' AND time_col <= '2023' → 单年区间
        range_cond = re.compile(
            rf'(?:WHERE|AND)\s+"?{quoted_col}"?\s*>=\s*\'?(\d{{4}})\'?'
            rf'\s+AND\s+"?{quoted_col}"?\s*<=\s*\'?(\d{{4}})\'?',
            re.IGNORECASE,
        )
        range_match = range_cond.search(raw_sql)
        if range_match and range_match.group(1) == range_match.group(2):
            raw_sql = range_cond.sub('', raw_sql)
            cleaned = True

        if cleaned:
            raw_sql = re.sub(r'WHERE\s+(?:AND|OR)\s+', 'WHERE ', raw_sql, flags=re.IGNORECASE)
            raw_sql = re.sub(r'\b(?:AND|OR)\s+(ORDER\s+BY|LIMIT)', r'\1', raw_sql, flags=re.IGNORECASE)
            raw_sql = re.sub(r'WHERE\s+(ORDER\s+BY|LIMIT|;|$)', r'\1', raw_sql, flags=re.IGNORECASE)
            raw_sql = re.sub(r'\s{2,}', ' ', raw_sql).strip()
            if raw_sql != orig_sql:
                logger.info("Trend SQL: removed single-year filter on '%s'", time_col)
        return raw_sql

    @staticmethod
    def _infer_time_column(execution_plan: dict[str, Any]) -> str | None:
        """从 raw_sql 的 SELECT 子句中推断时间列名。

        在 SELECT 列中找出不属于 indicators 且匹配常见时间维度名（year/年份/季度等）的列。
        """
        raw_sql: str | None = execution_plan.get("raw_sql")
        if not raw_sql:
            return None
        indicators: list[str] = execution_plan.get("indicators", [])
        indicator_lower = {i.lower() for i in indicators}
        cols = DAGBuilder._extract_select_columns(raw_sql)
        if not cols:
            return None
        TIME_LIKE_NAMES = {n.lower() for n in {
            "year", "年份", "统计年份", "quarter", "季度",
            "month", "月", "period", "time",
        }}
        for col in cols:
            if col.lower() in TIME_LIKE_NAMES and col.lower() not in indicator_lower:
                return col
        return None

    @staticmethod
    def _infer_dim_cols(raw_sql: str, indicators: list[str], execution_plan: dict[str, Any]) -> list[str]:
        """从 SQL SELECT 列中推断多维分析的维度列。

        维度列 = SELECT 中的所有列 - indicators - time_col。
        例如：SELECT 省区名称, GDP, 统计年份 FROM ...
        indicators=["GDP"] → 维度列 = ["省区名称"]
        """
        cols = DAGBuilder._extract_select_columns(raw_sql)
        if not cols:
            return []
        # 排除 indicator 列
        indicator_lower = {i.lower() for i in (indicators or [])}
        # 排除 time 列
        time_col = DAGBuilder._infer_time_column(execution_plan)
        time_lower = time_col.lower() if time_col else None

        dims = [c for c in cols
                if c.lower() not in indicator_lower
                and (time_lower is None or c.lower() != time_lower)]
        return dims

    @staticmethod
    def _extract_select_columns(sql: str) -> list[str]:
        """简易提取 SELECT 子句中的列名（去掉 CTE / 函数调用 / AS 别名）。"""
        import re
        # 去掉 CTE（WITH ... SELECT ... ）→ 只保留最后一个 SELECT
        cleaned = re.sub(
            r'WITH\s+.*?\)\s*',
            '',
            sql,
            flags=re.IGNORECASE | re.DOTALL,
        )
        m = re.search(
            r'SELECT\s+(.*?)\s+FROM',
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
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
                # 去掉表前缀 "e.gdp" → "gdp"
                col = part.split('.')[-1] if '.' in part else part
                cols.append(col.strip('"`\' '))
        return cols

    def _build_merge_node(self, analysis_nodes: list[AnalysisNode]) -> MergeNode:
        """构建融合节点，依赖所有分析节点。"""
        data_sources = [n.node_id for n in analysis_nodes]
        return MergeNode(
            node_id="merge_1",
            name=f"融合: {', '.join(n.algorithm_name for n in analysis_nodes)}",
            data_sources=data_sources,
            merge_strategy="collect",
            depends_on=list(data_sources),
            timeout=settings.DAG_MERGE_TIMEOUT,
        )

    # ------------------------------------------------------------------
    # Kahn 拓扑排序
    # ------------------------------------------------------------------

    def topological_sort(self, plan: DAGPlan) -> list[list[str]]:
        """Kahn 算法拓扑排序 → 层级分组。

        Returns:
            level_groups: 每个元素是一层可并行执行的 node_id 列表。

        Raises:
            DAGCycleError: 检测到环。
        """
        # 入度表
        in_degree: dict[str, int] = {}
        for node in plan.nodes:
            in_degree[node.node_id] = len(node.depends_on)

        # 邻接表（反向：node → 被哪些节点依赖）
        dependents: dict[str, list[str]] = {n.node_id: [] for n in plan.nodes}
        for node in plan.nodes:
            for dep in node.depends_on:
                if dep in dependents:
                    dependents[dep].append(node.node_id)

        level_groups: list[list[str]] = []
        queue: deque[str] = deque()

        # level 0：入度为 0 的节点
        for nid, degree in in_degree.items():
            if degree == 0:
                queue.append(nid)

        processed = 0
        while queue:
            current_level: list[str] = []
            for _ in range(len(queue)):
                nid = queue.popleft()
                current_level.append(nid)
                processed += 1
                for child in dependents.get(nid, []):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
            level_groups.append(current_level)

        if processed != plan.size:
            remaining = [nid for nid, d in in_degree.items() if d > 0]
            raise DAGCycleError(
                f"DAG 中存在环，以下节点入度未归零: {remaining}"
            )

        return level_groups


# =============================================================================
# 5. DAGExecutor
# =============================================================================


class DAGExecutor:
    """DAG 层次化并行执行器。

    调度策略：
    - 按 level_groups 逐层执行（层级间 Barrier）
    - 层级内节点通过 asyncio.gather 并发
    - SQL 节点直接 await db_pool.fetch
    - 分析节点通过 loop.run_in_executor 在线程池执行
    - 全局超时保护，节点失败 → DFS 熔断传播
    """

    def __init__(
        self,
        thread_pool_size: int | None = None,
    ) -> None:
        self._thread_pool_size = thread_pool_size or settings.DAG_THREAD_POOL_SIZE
        self._node_executor = NodeExecutor()

    async def execute(
        self,
        plan: DAGPlan,
        context: dict[str, Any],
        db_pool: Any,
        analyzer_registry: AnalyzerRegistry,
    ) -> ExecutionResult:
        """执行 DAGPlan。

        Args:
            plan:     DAGPlan 实例。
            context:  执行上下文（含 permissions 等）。
            db_pool:  DatabasePool 实例（asyncpg 连接池）。
            analyzer_registry: 算法注册中心单例。

        Returns:
            ExecutionResult。
        """
        start_time = time.monotonic()
        result = ExecutionResult()
        data_context: dict[str, pd.DataFrame | dict[str, pd.DataFrame]] = {}
        loop = asyncio.get_running_loop()

        # 全局超时
        global_timeout = settings.DAG_GLOBAL_TIMEOUT

        try:
            await asyncio.wait_for(
                self._execute_levels(
                    plan, context, db_pool, analyzer_registry,
                    data_context, loop, result,
                ),
                timeout=global_timeout,
            )
        except asyncio.TimeoutError:
            elapsed = (time.monotonic() - start_time) * 1000
            result.total_latency_ms = round(elapsed, 1)
            result.success = False
            result.dag_status = "partial"
            result.data_warning = True
            result.error = f"DAG 全局超时 ({global_timeout}s)"
            logger.error("DAG 全局超时 (%ds)", global_timeout)
            return result

        elapsed = (time.monotonic() - start_time) * 1000
        result.total_latency_ms = round(elapsed, 1)

        # 提取最终结果
        final_node_ids = plan.get_final_node_ids()
        if final_node_ids:
            last_id = final_node_ids[0]
            final_data = data_context.get(last_id)
            result.final_data = final_data
            result.success = final_data is not None

        # 保留原始 SQL 数据（用于前端展示原始字段）
        raw_data = data_context.get("sql_1")
        if isinstance(raw_data, pd.DataFrame) and not raw_data.empty:
            result.raw_data = raw_data

        # 判定整体状态
        has_failed = any(
            ne.status in (NodeStatus.FAILED, NodeStatus.SKIPPED)
            for ne in result.nodes_execution
        )
        if has_failed:
            result.dag_status = "partial"
            result.data_warning = True
        elif result.success:
            result.dag_status = "full"

        return result

    # ------------------------------------------------------------------
    # 层级执行
    # ------------------------------------------------------------------

    async def _execute_levels(
        self,
        plan: DAGPlan,
        context: dict[str, Any],
        db_pool: Any,
        analyzer_registry: AnalyzerRegistry,
        data_context: dict[str, Any],
        loop: asyncio.AbstractEventLoop,
        result: ExecutionResult,
    ) -> None:
        """逐层执行 DAG，每层内节点并发。"""
        tool_cache: dict[str, BaseActionTool] = {}
        skipped_nodes: set[str] = set()

        for level_idx, level_node_ids in enumerate(plan.level_groups):
            level_tasks: list[asyncio.Task] = []

            for node_id in level_node_ids:
                if node_id in skipped_nodes:
                    continue

                node = plan.get_node(node_id)

                if node.node_type == "sql":
                    task = asyncio.create_task(
                        self._execute_sql_node(node, db_pool, plan, data_context)
                    )
                elif node.node_type == "analysis":
                    task = asyncio.create_task(
                        self._execute_analysis_node(
                            node, context, analyzer_registry,
                            tool_cache, data_context, plan, loop, db_pool,
                            result,
                        )
                    )
                elif node.node_type == "merge":
                    task = asyncio.create_task(
                        self._execute_merge_node(node, data_context)
                    )
                else:
                    continue

                level_tasks.append(task)

            if not level_tasks:
                continue

            # 层级内并发执行（带节点级超时）
            # 使用 return_exceptions=True 自行处理异常
            gathered = asyncio.gather(*level_tasks, return_exceptions=True)
            done, _ = await asyncio.wait(
                {asyncio.ensure_future(gathered)},
                timeout=_level_timeout(level_idx, plan),
            )

            if done:
                level_results = done.pop().result()
                if isinstance(level_results, list):
                    for node_id, node_result in zip(level_node_ids, level_results):
                        if isinstance(node_result, BaseNode):
                            continue  # 已通过异常对象传递的节点状态在 execute 函数内处理

                        # 收集 node_execution 记录
                        node = plan.get_node(node_id)
                        ne = NodeExecution(
                            node_id=node_id,
                            status=node.status,
                            latency_ms=node.latency_ms,
                            error=node.error,
                        )
                        result.add_node_execution(ne)

            # 熔断传播：如果当前层有失败节点，标记所有下游为 SKIPPED
            newly_skipped = self._propagate_failure(
                plan, level_node_ids, data_context, result,
            )
            skipped_nodes.update(newly_skipped)

        # 预埋点：trace_logger.log_dag_end(result)
        # 预埋点：feedback_gate.verify_plan(plan, result)
        _maybe_call_gate("verify_plan", plan, result)

    # ------------------------------------------------------------------
    # 单节点执行
    # ------------------------------------------------------------------

    async def _execute_sql_node(
        self,
        node: BaseNode,
        db_pool: Any,
        plan: DAGPlan,
        data_context: dict[str, Any],
    ) -> None:
        """委托 NodeExecutor 执行 SQL 节点。

        包含三级 SQL 错误修复（带熔断）：
        1. PostgreSQL HINT 自动修复列名
        2. LLM 重新生成修复后的 SQL（最多 2 次 LLM 调用）
        3. 除以零等运行时错误修复

        SQLBuilder 构建的 SQL 通常无需修复（确定性构建）。
        """
        start = time.monotonic()
        node.status = NodeStatus.RUNNING

        execution_plan = plan.metadata.get("execution_plan", {})
        sql_source = execution_plan.get("_sql_source", "raw_sql")
        repair_count = 0
        MAX_REPAIR = 2  # 最多 2 次 LLM 修复
        REPAIR_TIMEOUT = 25  # 修复阶段总预算 25s
        repair_start = time.monotonic()

        current_sql = node.sql

        try:
            logger.info("执行 SQL (source=%s): %.200s", sql_source, current_sql)
            df = await self._node_executor.execute_sql(current_sql, db_pool)
            # SQL 执行成功 → 检查是否为空
            if df.empty:
                logger.info("SQL 返回空结果，尝试空数据降级")
                fallback_df = await self._empty_data_fallback(
                    current_sql, db_pool, execution_plan, start,
                )
                if fallback_df is not None and not fallback_df.empty:
                    data_context[node.node_id] = fallback_df
                    node.status = NodeStatus.SUCCESS
                    node.output_shape = f"{len(fallback_df)}x{len(fallback_df.columns)}"
                else:
                    # 降级失败，保留空 DataFrame 传给下游
                    data_context[node.node_id] = df
                    node.status = NodeStatus.SUCCESS
                    node.output_shape = "0x0"
                    if execution_plan:
                        execution_plan["_empty_data"] = True
            else:
                data_context[node.node_id] = df
                node.status = NodeStatus.SUCCESS
                node.output_shape = f"{len(df)}x{len(df.columns)}" if not df.empty else "0x0"
        except Exception as exc:
            error_msg = str(exc)
            # 时间预算检查
            elapsed = time.monotonic() - repair_start
            if elapsed > REPAIR_TIMEOUT:
                logger.warning("SQL 修复预算超时 (%.1fs > %ds)，放弃修复", elapsed, REPAIR_TIMEOUT)
                node.status = NodeStatus.FAILED
                node.error = f"修复超时 ({elapsed:.1f}s): {error_msg[:200]}"
                node.latency_ms = round((time.monotonic() - start) * 1000, 1)
                return

            # 第1级修复：PostgreSQL HINT 自动修复
            if repair_count < MAX_REPAIR:
                fixed_sql = DAGBuilder._try_fix_sql_from_hint(current_sql, error_msg)
                if fixed_sql:
                    fixed_sql = DAGBuilder._fix_union_syntax(fixed_sql)
                    fixed_sql = DAGBuilder._fix_percentile_cont(fixed_sql)
                    logger.info("SQL HINT 修复，重试: %.200s", fixed_sql)
                    try:
                        df = await self._node_executor.execute_sql(fixed_sql, db_pool)
                        data_context[node.node_id] = df
                        node.status = NodeStatus.SUCCESS
                        node.output_shape = f"{len(df)}x{len(df.columns)}" if not df.empty else "0x0"
                        node.latency_ms = round((time.monotonic() - start) * 1000, 1)
                        return
                    except Exception:
                        repair_count += 1
                        error_msg = f"{error_msg[:300]} | HINT fix failed"

            # 第2级修复：LLM 重新生成修复后的 SQL（带完整 schema 上下文）
            if repair_count < MAX_REPAIR:
                llm_fixed = await DAGBuilder._try_fix_sql_with_llm(current_sql, error_msg)
                if llm_fixed:
                    llm_fixed = DAGBuilder._fix_union_syntax(llm_fixed)
                    llm_fixed = DAGBuilder._fix_percentile_cont(llm_fixed)
                    logger.info("LLM SQL 修复，重试: %.200s", llm_fixed)
                    try:
                        df = await self._node_executor.execute_sql(llm_fixed, db_pool)
                        data_context[node.node_id] = df
                        node.status = NodeStatus.SUCCESS
                        node.output_shape = f"{len(df)}x{len(df.columns)}" if not df.empty else "0x0"
                        node.latency_ms = round((time.monotonic() - start) * 1000, 1)
                        return
                    except Exception as llm_exc:
                        repair_count += 1
                        if repair_count >= MAX_REPAIR:
                            error_msg = f"{error_msg[:300]} | LLM fix #{repair_count} also failed: {str(llm_exc)[:200]}"
                        else:
                            error_msg = f"{error_msg[:300]} | LLM fix #{repair_count} failed: {str(llm_exc)[:200]}"

            # 第3级修复：除以零
            if "除以零" in error_msg or "division by zero" in error_msg.lower():
                simplified = DAGBuilder._fix_division_by_zero(current_sql)
                if simplified and simplified != current_sql:
                    logger.info("SQL 除以零修复，重试: %.200s", simplified)
                    try:
                        df = await self._node_executor.execute_sql(simplified, db_pool)
                        data_context[node.node_id] = df
                        node.status = NodeStatus.SUCCESS
                        node.output_shape = f"{len(df)}x{len(df.columns)}" if not df.empty else "0x0"
                        node.latency_ms = round((time.monotonic() - start) * 1000, 1)
                        return
                    except Exception:
                        pass

            # 第4级修复：SQLBuilder 兜底重建（改造4.2）
            # 当所有修复模式都失败时，从结构化 execution_plan 重建 SQL
            if node.status != NodeStatus.SUCCESS:
                _try_sqlbuilder_fallback = False
                _fallback_sql = execution_plan.get("_builder_sql", "")
                if not _fallback_sql:
                    # 尝试用执行计划的结构化字段即时构建
                    try:
                        from app.engine.sql_builder import SQLBuilder
                        _sb = SQLBuilder()
                        if _sb.can_build(execution_plan):
                            _fallback_sql = _sb.build(execution_plan)
                    except Exception:
                        _fallback_sql = ""

                if _fallback_sql and _fallback_sql != current_sql:
                    _try_sqlbuilder_fallback = True
                    logger.info(
                        "SQLBuilder 兜底: 从结构化字段重建 SQL，重试: %.200s",
                        _fallback_sql,
                    )
                    try:
                        df = await self._node_executor.execute_sql(_fallback_sql, db_pool)
                        data_context[node.node_id] = df
                        node.status = NodeStatus.SUCCESS
                        node.output_shape = f"{len(df)}x{len(df.columns)}" if not df.empty else "0x0"
                        # 更新 sql_source 标记
                        execution_plan["_sql_source"] = "builder_fallback"
                        node.latency_ms = round((time.monotonic() - start) * 1000, 1)
                        logger.info("SQLBuilder 兜底成功！")
                        return
                    except Exception as fb_exc:
                        logger.warning("SQLBuilder 兜底也失败: %s", str(fb_exc)[:150])
                        error_msg = f"{error_msg[:300]} | SQLBuilder fallback failed: {str(fb_exc)[:100]}"

            node.status = NodeStatus.FAILED
            node.error = error_msg[:500]
            logger.warning("SQL 节点 '%s' 失败 (repairs=%d): %s", node.node_id, repair_count, error_msg[:200])
        finally:
            node.latency_ms = round((time.monotonic() - start) * 1000, 1)

    async def _empty_data_fallback(
        self,
        sql: str,
        db_pool: Any,
        execution_plan: dict[str, Any],
        start_time: float,
    ) -> pd.DataFrame | None:
        """SQL 返回空数据时的降级查询。

        策略：
        1. 生成简化版 SQL（仅 SELECT 原始列，去除复杂计算）
        2. 去除 WHERE 中的非核心过滤条件
        3. SELECT DISTINCT * FROM 主表 LIMIT 100 兜底
        """
        tables = execution_plan.get("tables", [])
        indicators = execution_plan.get("indicators", [])
        if not tables:
            return None

        primary_table = tables[0]

        # 第1级：简化 SQL 重试
        try:
            simple_cols = ", ".join(f'"{c}"' if c != c.lower() or any('一' <= ch <= '鿿' for ch in c) else c for c in indicators) if indicators else "*"
            simple_sql = f"SELECT {simple_cols} FROM \"{primary_table}\""
            logger.info("空数据降级1: %s", simple_sql[:150])
            df = await self._node_executor.execute_sql(simple_sql, db_pool)
            if not df.empty:
                return df
        except Exception:
            pass

        # 第2级：SELECT * 不加条件
        try:
            fallback_sql = f"SELECT * FROM \"{primary_table}\" LIMIT 200"
            logger.info("空数据降级2: %s", fallback_sql[:150])
            df = await self._node_executor.execute_sql(fallback_sql, db_pool)
            if not df.empty:
                return df
        except Exception:
            pass

        # 第3级：跨表 JOIN 降级
        if len(tables) > 1:
            try:
                join_cols = ", ".join(f'{_quote_col(t)}."区划ID"' for t in tables[:3]) + (", " + ", ".join(f'{_quote_col(t)}."{c}"' if c != c.lower() else f'{_quote_col(t)}.{c}' for c in indicators[:3]) if indicators else "")
                join_sql = f"SELECT {join_cols} FROM \"{primary_table}\""
                for t in tables[1:3]:
                    join_sql += f" LEFT JOIN \"{t}\" ON \"{primary_table}\".\"区划ID\" = \"{t}\".\"区划ID\""
                join_sql += " LIMIT 100"
                logger.info("空数据降级3: %s", join_sql[:200])
                df = await self._node_executor.execute_sql(join_sql, db_pool)
                if not df.empty:
                    return df
            except Exception:
                pass

        return None

    async def _execute_analysis_node(
        self,
        node: BaseNode,
        context: dict[str, Any],
        analyzer_registry: AnalyzerRegistry,
        tool_cache: dict[str, BaseActionTool],
        data_context: dict[str, Any],
        plan: DAGPlan,
        loop: asyncio.AbstractEventLoop,
        db_pool: Any = None,
        exec_result: ExecutionResult | None = None,
    ) -> None:
        """委托 NodeExecutor 执行分析节点。

        当算法因缺少必需列而失败时，自动触发 SQL 补全重执行：
        1. 提取缺失列名
        2. 用 LLM 修补 SQL 以补全缺失列
        3. 重新执行 SQL 并更新 data_context
        4. 使用新数据重试分析（1次）

        执行成功后，将分析结果摘要收集到 exec_result.analysis_results 中。
        """
        start = time.monotonic()
        node.status = NodeStatus.RUNNING

        analysis_node = node  # type: AnalysisNode
        upstream_data = data_context.get(analysis_node.data_source)
        if upstream_data is None or (isinstance(upstream_data, pd.DataFrame) and upstream_data.empty):
            node.status = NodeStatus.SKIPPED
            node.error = f"上游数据 '{analysis_node.data_source}' 不可用或为空"
            node.latency_ms = 0.0
            logger.warning("分析节点 '%s' 跳过: %s", node.node_id, node.error)
            return

        # 最多尝试 2 次：原始数据 + SQL 补全后重试
        success = False

        # 预检：清洗 dim_cols — 过滤掉不存在于字段数据中的列名
        dim_cols = analysis_node.params.get("dim_cols", [])
        if isinstance(dim_cols, list) and dim_cols:
            valid_dims = [c for c in dim_cols if c.strip('"\' ') in upstream_data.columns]
            invalid_dims = [c for c in dim_cols if c.strip('"\' ') not in upstream_data.columns]
            if invalid_dims:
                logger.info("dim_cols 清洗: 过滤不存在的列 %s，保留 %s", invalid_dims, valid_dims)
            if not valid_dims:
                # 试图从数据列自动推断维度列
                auto_dims = [c for c in upstream_data.columns
                             if c not in (analysis_node.params.get("value_col", ""),)
                             and c not in (analysis_node.params.get("time_col", ""),)
                             and c != '"*"']
                if auto_dims:
                    analysis_node.params["dim_cols"] = auto_dims[:3]
                    logger.info("dim_cols 自动推断: %s", analysis_node.params["dim_cols"])
                else:
                    analysis_node.params["dim_cols"] = []
            else:
                analysis_node.params["dim_cols"] = valid_dims

            # dim_cols 变更后同步更新 required_columns，清除残留的占位符列
            new_dims = analysis_node.params.get("dim_cols", [])
            if new_dims != dim_cols:
                analysis_node.required_columns = [
                    c for c in analysis_node.required_columns
                    if c.strip('"\' ') in upstream_data.columns or c.strip('"\' ') in new_dims
                ]

        # ── 主动列名预校验：SQL 输出列 ≠ 分析器期望列时提前修复 ──
        required = analysis_node.required_columns
        if required and isinstance(upstream_data, pd.DataFrame) and not upstream_data.empty:
            df_cols_list = list(upstream_data.columns)
            missing_required = [c for c in required if c not in df_cols_list]
            if missing_required:
                logger.info(
                    "分析节点 '%s' 的必需列 %s 不在 DataFrame 列 %s 中，尝试预匹配",
                    node.node_id, missing_required, df_cols_list,
                )
                col_map = _try_column_mapping(missing_required, df_cols_list)
                if col_map:
                    logger.info("列预匹配成功: %s (共 %d/%d 列)", col_map, len(col_map), len(missing_required))
                    # 更新 params 中的列名引用
                    for param_key in ("value_col", "x_col", "y_col", "time_col"):
                        old_val = analysis_node.params.get(param_key, "")
                        if isinstance(old_val, str) and old_val in col_map:
                            analysis_node.params[param_key] = col_map[old_val]
                            logger.info("参数 '%s' (列): '%s' → '%s'", param_key, old_val, col_map[old_val])
                    # list 类型参数：value_cols 逐个映射
                    vc_list = analysis_node.params.get("value_cols", [])
                    if isinstance(vc_list, list):
                        remapped = [col_map.get(v, v) for v in vc_list]
                        if remapped != vc_list:
                            analysis_node.params["value_cols"] = remapped
                            logger.info("参数 'value_cols' 重映射: %s", remapped)
                    # 记录到 required_columns 供下游参考
                    analysis_node.required_columns = [
                        col_map.get(c, c) for c in required
                    ]

        for attempt in range(2):
            try:
                df = await self._node_executor.execute_analysis(
                    algorithm_name=analysis_node.algorithm_name,
                    context=context,
                    registry=analyzer_registry,
                    tool_cache=tool_cache,
                    data=upstream_data,
                    params=analysis_node.params,
                    loop=loop,
                )
                data_context[node.node_id] = df
                node.status = NodeStatus.SUCCESS
                if isinstance(df, pd.DataFrame):
                    node.output_shape = f"{len(df)}x{len(df.columns)}"
                elif isinstance(df, dict):
                    node.output_shape = f"dict({len(df)} keys)"
                else:
                    node.output_shape = type(df).__name__
                # 收集分析结果摘要
                if exec_result is not None:
                    summary = _extract_analysis_summary(df, analysis_node.algorithm_name)
                    exec_result.analysis_results[node.node_id] = summary
                    if analysis_node.algorithm_name not in exec_result._algorithm_names:
                        exec_result._algorithm_names.append(analysis_node.algorithm_name)
                success = True
                break  # 成功，退出重试循环

            except Exception as exc:
                error_msg = str(exc)
                if attempt == 0 and "缺少必需的列" in error_msg and db_pool is not None:
                    import re
                    missing_cols = re.findall(r"'([^']+)'", error_msg)
                    if missing_cols:
                        source_node = plan.get_node(analysis_node.data_source)

                        # ---- 尝试列名归一化：去除 / 后缀匹配已有列 ----
                        col_remap = _normalize_analysis_cols(missing_cols, upstream_data.columns)
                        if col_remap:
                            logger.info(
                                "分析节点 '%s' 列名归一化映射: %s",
                                node.node_id, col_remap,
                            )
                            # 重映射分析参数中的 value_cols
                            if analysis_node.params.get("value_cols"):
                                remapped = []
                                for vc in analysis_node.params["value_cols"]:
                                    remapped.append(col_remap.get(vc, vc))
                                analysis_node.params["value_cols"] = remapped
                            # 重映射 value_col
                            val = analysis_node.params.get("value_col", "")
                            if val in col_remap:
                                analysis_node.params["value_col"] = col_remap[val]

                            try:
                                # 直接用现有数据重试，不经过 SQL 补全
                                df = await self._node_executor.execute_analysis(
                                    algorithm_name=analysis_node.algorithm_name,
                                    context=context,
                                    registry=analyzer_registry,
                                    tool_cache=tool_cache,
                                    data=upstream_data,
                                    params=analysis_node.params,
                                    loop=loop,
                                )
                                data_context[node.node_id] = df
                                node.status = NodeStatus.SUCCESS
                                if isinstance(df, pd.DataFrame):
                                    node.output_shape = f"{len(df)}x{len(df.columns)}"
                                elif isinstance(df, dict):
                                    node.output_shape = f"dict({len(df)} keys)"
                                else:
                                    node.output_shape = type(df).__name__
                                # 收集分析结果摘要
                                if exec_result is not None:
                                    summary = _extract_analysis_summary(df, analysis_node.algorithm_name)
                                    exec_result.analysis_results[node.node_id] = summary
                                    if analysis_node.algorithm_name not in exec_result._algorithm_names:
                                        exec_result._algorithm_names.append(analysis_node.algorithm_name)
                                success = True
                                break
                            except Exception as norm_exc:
                                logger.warning(
                                    "列名归一化重试仍失败: %s，尝试 SQL 补全",
                                    norm_exc,
                                )

                        # ---- SQL 补全（LLM） ----
                        if isinstance(source_node, SQLNode) and source_node.sql.strip():
                            logger.info(
                                "分析节点 '%s' 缺少列 %s，尝试 SQL 补全",
                                node.node_id, missing_cols,
                            )
                            new_sql = await self._augment_sql_for_analysis(
                                source_node.sql, missing_cols, error_msg,
                            )
                            if new_sql:
                                try:
                                    new_df = await self._node_executor.execute_sql(new_sql, db_pool)
                                    if not new_df.empty:
                                        data_context[analysis_node.data_source] = new_df
                                        upstream_data = new_df
                                        logger.info(
                                            "SQL 补全成功，重试分析 '%s' (新 DF: %sx%s)",
                                            node.node_id, len(new_df), len(new_df.columns),
                                        )
                                        continue
                                except Exception as sql_exc:
                                    logger.warning("SQL 补全执行失败: %s", sql_exc)

                # 所有尝试失败
                node.status = NodeStatus.FAILED
                node.error = error_msg
                logger.warning("分析节点 '%s' 异常: %s", node.node_id, exc)
                break  # 退出重试循环

        node.latency_ms = round((time.monotonic() - start) * 1000, 1)

    @staticmethod
    async def _augment_sql_for_analysis(
        sql: str,
        missing_cols: list[str],
        error_msg: str,
    ) -> str | None:
        """用 LLM 修补 SQL，追加缺失的列到 SELECT 子句。

        当分析算法需要 DataFrame 中不存在的列时，
        调用此方法让 LLM 在保持原 SQL 语义不变的前提下，
        在 SELECT 子句中追加缺失列。

        Args:
            sql:          原始 SQL。
            missing_cols: 缺失的列名列表。
            error_msg:    原始错误信息（用于上下文）。

        Returns:
            修补后的 SQL，若失败返回 None。
        """
        logger.info("修补 SQL 以补全缺失列 %s", missing_cols)
        prompt = (
            f"你是一个 PostgreSQL 专家。以下 SQL 语句缺少算法分析所需的列，"
            f"请在 SELECT 子句中追加这些列（保持原 SQL 所有输出不变）。\n\n"
            f"原始 SQL:\n{sql}\n\n"
            f"需要追加的列: {missing_cols}\n\n"
            f"原始错误:\n{error_msg}\n\n"
            f"要求:\n"
            f"1. 只输出纯 SQL，不要 markdown 代码块\n"
            f"2. 保持原 SQL 的所有 SELECT 列不变，仅追加缺失的列\n"
            f"3. 确保缺失的列名与物理表中实际列名一致\n"
            f"4. 如果列可能在 JOIN 的另一张表中，使用表别名前缀\n"
            f"5. 不要添加 AS 别名掩盖原始列名"
        )
        try:
            from app.core.llm import chat_with_model
            response = await chat_with_model("openai", [
                {"role": "system", "content": "你是一个 PostgreSQL 专家，只输出 SQL 代码。"},
                {"role": "user", "content": prompt},
            ], temperature=0.1)
            import re
            fixed = response.strip()
            if fixed.startswith("```"):
                fixed = re.sub(r'^```\w*\n?', '', fixed)
                fixed = re.sub(r'\n?```$', '', fixed)
            fixed = fixed.strip()
            if fixed and (fixed.upper().startswith("SELECT") or fixed.upper().startswith("WITH")):
                if fixed != sql:
                    logger.info("SQL 已修补，追加了列 %s", missing_cols)
                    return fixed
            logger.warning("LLM 返回的 SQL 无变化或格式错误")
            return None
        except Exception as exc:
            logger.warning("SQL 补全失败: %s", exc)
            return None

    async def _execute_merge_node(
        self,
        node: BaseNode,
        data_context: dict[str, Any],
    ) -> None:
        """委托 NodeExecutor 执行融合节点。"""
        start = time.monotonic()
        node.status = NodeStatus.RUNNING

        merge_node = node  # type: MergeNode

        try:
            merged = self._node_executor.execute_merge(
                data_sources=merge_node.data_sources,
                merge_strategy=merge_node.merge_strategy,
                merge_key=merge_node.merge_key,
                data_map=data_context,
            )
            data_context[node.node_id] = merged
            node.status = NodeStatus.SUCCESS
            if isinstance(merged, pd.DataFrame):
                node.output_shape = f"{len(merged)}x{len(merged.columns)}"
            elif isinstance(merged, dict):
                node.output_shape = f"dict({len(merged)} keys)"
            else:
                node.output_shape = type(merged).__name__
        except ValueError as exc:
            # 所有上游不可用 → 跳过，不视为错误
            node.status = NodeStatus.SKIPPED
            node.error = str(exc)
            node.latency_ms = 0.0
        except Exception as exc:
            node.status = NodeStatus.FAILED
            node.error = str(exc)
            logger.warning("合并节点 '%s' 失败: %s", node.node_id, exc)
        finally:
            node.latency_ms = round((time.monotonic() - start) * 1000, 1)

    # ------------------------------------------------------------------
    # 熔断传播
    # ------------------------------------------------------------------

    def _propagate_failure(
        self,
        plan: DAGPlan,
        level_node_ids: list[str],
        data_context: dict[str, Any],
        result: ExecutionResult,
    ) -> set[str]:
        """DFS 标记所有下游节点为 SKIPPED。"""
        newly_skipped: set[str] = set()

        for node_id in level_node_ids:
            node = plan.get_node(node_id)
            if node.status != NodeStatus.FAILED:
                continue

            # DFS 遍历依赖图
            stack = [node_id]
            visited: set[str] = set()
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)

                for downstream_id in plan.get_downstream(current):
                    downstream = plan.get_downstream(current)
                    for d_id in downstream:
                        if d_id in visited:
                            continue
                        dn = plan.get_node(d_id)
                        if dn.status != NodeStatus.FAILED:
                            dn.status = NodeStatus.SKIPPED
                            dn.error = f"上游节点 '{node_id}' 失败，熔断跳过"
                            dn.latency_ms = 0.0
                            newly_skipped.add(d_id)
                            data_context.pop(d_id, None)
                            stack.append(d_id)
                            result.add_node_execution(NodeExecution(
                                node_id=d_id,
                                status=NodeStatus.SKIPPED,
                                error=dn.error,
                            ))

        return newly_skipped


# =============================================================================
# 6. DAGOrchestrator（总入口）
# =============================================================================


class DAGOrchestrator:
    """DAG 编排总入口。

    提供三个核心方法：
      build_plan(parse_result) → DAGPlan
      execute(plan, context, db_pool) → ExecutionResult
      replan(plan, feedback) → DAGPlan
    """

    def __init__(self) -> None:
        self._builder = DAGBuilder()
        self._executor = DAGExecutor()

    def build_plan(
        self,
        parse_result: dict[str, Any],
        analyzer_registry: AnalyzerRegistry | None = None,
    ) -> DAGPlan:
        """从 Parser 输出构建 DAGPlan。"""
        return self._builder.build_plan(parse_result, analyzer_registry)

    async def execute(
        self,
        plan: DAGPlan,
        context: dict[str, Any],
        db_pool: Any,
        analyzer_registry: AnalyzerRegistry,
    ) -> ExecutionResult:
        """执行 DAGPlan。"""
        return await self._executor.execute(plan, context, db_pool, analyzer_registry)

    def replan(self, plan: DAGPlan, feedback: dict[str, Any]) -> DAGPlan:
        """重新规划 DAG（Gate② 校验失败时触发）。

        Args:
            plan:     原 DAGPlan。
            feedback: 反馈信息，含 failed_node / reason / detail 等。

        Returns:
            修改后的新 DAGPlan。
        """
        failed_node_id = feedback.get("failed_node", "")
        reason = feedback.get("reason", "")

        if not failed_node_id or failed_node_id not in plan._node_map:  # noqa: SLF001
            logger.warning("Re-plan 跳过: 未知节点 '%s'", failed_node_id)
            return plan

        failed_node = plan._node_map[failed_node_id]  # noqa: SLF001
        failed_node.status = NodeStatus.REPLANNED

        if reason == "字段缺失":
            # 修改 SQLNode 的 SQL 以增加列（使用 feedback 中 detail 的补充字段）
            detail = feedback.get("detail", "")
            if detail and failed_node_id == "sql_1" and isinstance(failed_node, SQLNode):
                failed_node.sql = self._patch_sql_columns(failed_node.sql, detail)
                logger.info("Re-plan: SQLNode '%s' 补全列: %s", failed_node_id, detail)
        elif reason == "算法不合适":
            new_algo = feedback.get("new_algorithm", "")
            if new_algo and isinstance(failed_node, AnalysisNode):
                failed_node.algorithm_name = new_algo
                logger.info("Re-plan: AnalysisNode '%s' 替换算法 -> %s", failed_node_id, new_algo)
        elif reason == "超时":
            failed_node.timeout = min(failed_node.timeout * 2, 120)
            logger.info("Re-plan: Node '%s' 超时调整为 %ds", failed_node_id, failed_node.timeout)

        # 重新拓扑排序
        plan.level_groups = self._builder.topological_sort(plan)
        plan.entry_nodes = list(plan.level_groups[0]) if plan.level_groups else []

        return plan

    @staticmethod
    def _patch_sql_columns(sql: str, new_columns: str) -> str:
        """在 SELECT 子句中补充缺失的列。"""
        sql_stripped = sql.strip().rstrip(";")
        if sql_stripped.upper().startswith("SELECT"):
            # 在 SELECT 后、FROM 前插入新列
            select_end = sql_stripped.upper().find("FROM")
            if select_end > 0:
                select_part = sql_stripped[6:select_end].strip()
                if select_part.strip() != "*":
                    patched = f"SELECT {select_part}, {new_columns} {sql_stripped[select_end:]}"
                    return patched
        return sql


# =============================================================================
# 7. 模块级单例
# =============================================================================

orchestrator = DAGOrchestrator()


# =============================================================================
# 8. 内部工具函数
# =============================================================================


def _level_timeout(level_idx: int, plan: DAGPlan) -> float:
    """计算当前层级的超时时间（全局剩余时间）。"""
    max_level_time = max(
        (plan.get_node(nid).timeout for nid in plan.level_groups[level_idx]),
        default=30,
    )
    return float(max_level_time)


def _maybe_call_gate(_method: str, *args: Any, **kwargs: Any) -> None:
    """预埋点：调用 feedback_gate / trace_logger（空实现，Day 6 完善）。"""
    pass


def _quote_col(name: str) -> str:
    """列名自动引号包裹（复用 SQLBuilder 的引用规则）。

    包含中文或大写字母的列名 → 加双引号。
    """
    import re
    if not name:
        return ""
    cleaned = name.strip('"`\' ')
    if not cleaned:
        return ""
    if cleaned.replace('.', '', 1).lstrip('-').isdigit():
        return f'"{cleaned}"'
    if re.search(r'[一-鿿]', cleaned) or cleaned != cleaned.lower():
        return f'"{cleaned}"'
    return cleaned


def _try_column_mapping(
    required_cols: list[str],
    df_cols: list[str],
) -> dict[str, str]:
    """将必需列名通过多级匹配映射到 DataFrame 实际列名。

    与 executor._match_column 类似但增强：
    - 对每个必需列尝试 6 级匹配
    - 返回全部可映射的配对（不要求全部匹配）

    Returns:
        映射字典 {required_col → matched_col}，仅包含可匹配的项。
    """
    import difflib
    from app.engine.indicator_registry import indicator_registry

    df_set = set(df_cols)
    df_lower = {c.lower(): c for c in df_cols}
    mapping: dict[str, str] = {}

    for col in required_cols:
        if col in df_set:
            continue  # 已经匹配

        matched: str | None = None

        # Level 1: 忽略大小写
        if col.lower() in df_lower:
            matched = df_lower[col.lower()]

        # Level 2: 去引号和特殊字符
        if not matched:
            clean = col.strip('"\'').strip()
            if clean and clean != col:
                for c2 in df_cols:
                    if c2 == clean or c2.lower() == clean.lower():
                        matched = c2
                        break

        # Level 3: 子串包含
        if not matched:
            for c2 in df_cols:
                if col.lower() in c2.lower() or c2.lower() in col.lower():
                    matched = c2
                    break

        # Level 4: difflib 模糊匹配
        if not matched:
            fuzzy = difflib.get_close_matches(col, df_cols, n=1, cutoff=0.6)
            if fuzzy:
                matched = fuzzy[0]

        # Level 5: / 后缀去除（'医院数量/总数' → '医院数量'）
        if not matched and "/" in col:
            import re
            base = re.sub(r'/.+$', '', col).strip()
            if base and base in df_set:
                matched = base

        # Level 6: indicator_registry 反向查找（显示名 → 物理列名）
        if not matched:
            try:
                physical = indicator_registry.search_field_by_name(col)
                if physical:
                    if physical in df_set:
                        matched = physical
                    elif physical.lower() in df_lower:
                        matched = df_lower[physical.lower()]
                    else:
                        fuzzy2 = difflib.get_close_matches(physical, df_cols, n=1, cutoff=0.6)
                        if fuzzy2:
                            matched = fuzzy2[0]
            except Exception:
                pass

        if matched and matched != col:
            mapping[col] = matched

    return mapping


def _normalize_analysis_cols(
    missing_cols: list[str],
    existing_cols: pd.Index,
) -> dict[str, str]:
    """尝试将缺失列名通过归一化匹配到已有数据列。

    处理模式：
    1. 去除 / 后缀： '医院数量/总数' → '医院数量'
    2. 去除 / 中的分母单位： '医疗支出/亿' → '医疗支出'
    3. 精确匹配已有的列名

    Args:
        missing_cols: 分析器报告的缺失列名列表。
        existing_cols: 上游 DataFrame 的列索引。

    Returns:
        列名映射 dict[缺失列名 → 已有列名]，
        若无法全部映射则返回空 dict。
    """
    import re as _re
    mapping: dict[str, str] = {}
    existing_set = set(existing_cols)

    for col in missing_cols:
        cleaned = col.strip().strip('"\' ')
        if cleaned in existing_set:
            mapping[col] = cleaned
            continue

        # 尝试去除 / 后缀：'医院数量/总数' → 检查 '医院数量'
        base = _re.sub(r'/.+$', '', cleaned).strip()
        if base and base != cleaned and base in existing_set:
            mapping[col] = base
            continue

        # 尝试匹配：missing_col 可能是已有列名的子串
        for existing in existing_set:
            if isinstance(existing, str):
                # 已有列包含缺失列的基准名
                if base and (base in existing or existing in base):
                    mapping[col] = existing
                    break
                # 模糊匹配：忽略差异字符
                if cleaned and _fuzzy_col_match(cleaned, existing):
                    mapping[col] = existing
                    break

    # 返回部分匹配结果（之前要求全部匹配才能返回）
    # 注意：返回 partial mapping 比全部失败的 SQL augmentation 更高效
    if mapping:
        return mapping
    return {}


def _fuzzy_col_match(name1: str, name2: str) -> bool:
    """模糊列名匹配，忽略非中文字符差异。

    如 '城镇就业人口/总就业人口' 与 '城镇就业人口' 匹配。
    """
    import re as _re
    # 提取所有中文字符
    c1 = ''.join(_re.findall(r'[一-鿿]+', name1))
    c2 = ''.join(_re.findall(r'[一-鿿]+', name2))
    if not c1 or not c2:
        return False
    # 一个包含另一个
    return c1 in c2 or c2 in c1


def _extract_analysis_summary(
    df: pd.DataFrame | dict[str, pd.DataFrame],
    algorithm_name: str,
) -> dict[str, Any]:
    """从分析算法输出中提取结构化摘要，供 interpretation 使用。"""
    if isinstance(df, dict):
        return {"output_type": "dict", "keys": list(df.keys()), "algorithm": algorithm_name}

    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"output_type": "empty", "algorithm": algorithm_name}

    # 单行或少行 DataFrame → 转为键值对（如 correlation 输出单行系数）
    if len(df) <= 3:
        kv_pairs: dict[str, Any] = {}
        for col in df.columns:
            vals = df[col].tolist()
            significant = [v for v in vals if v is not None and v != 0 and not (isinstance(v, float) and (v != v))]
            if significant:
                kv_pairs[col] = significant[0] if len(significant) == 1 else significant[:3]
            else:
                kv_pairs[col] = vals[0] if len(vals) == 1 else vals[:3]
        kv_pairs["output_type"] = "single_row"
        kv_pairs["algorithm"] = algorithm_name
        kv_pairs["rows"] = len(df)
        return kv_pairs

    # 多行 DataFrame → 输出统计摘要
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return {
        "output_type": "multi_row",
        "algorithm": algorithm_name,
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": numeric_cols[:10],
    }
