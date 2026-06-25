"""
大模型语义解析层 — 意图路由 + 结构化多任务规划
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.core.llm import chat_with_model
from app.engine.schema_rag import SchemaRAGEngine
from app.engine.join_path_finder import JoinPathFinder
from app.engine.semantic_cache import SemanticCache
import json_repair

logger = logging.getLogger(__name__)

# ── 1. 结构化 Pydantic 模型 (大模型必须遵循的输出契约) ──

class RouteResult(BaseModel):
    intent: str = Field(..., description="意图分类：query_data(查数), greetings(问候), ambiguous(指代不明,需澄清)")
    indicators: List[str] = Field(default=[], description="识别出的核心业务指标, 如 ['GDP', '空气质量']")
    time_hint: Optional[str] = Field(None, description="识别出的时间约束，如 '2023年', '过去三年'")
    complexity: int = Field(..., ge=1, le=5, description="分析复杂度评分 1-5")

class SQLAgentPlan(BaseModel):
    analysis_type: str = Field(..., description="趋势分析(trend) / 排行对比(rank) / 维度明细(detail) / 高级预测(advanced) / 综合评价(composite) / 相关分析(correlation) / 异常检测(anomaly) / 多维交叉(multi_dim) / 区域空间(spatial) / 跨域综合(cross_domain)")
    indicators: List[str] = Field(..., description="标准化的核心指标数组")
    tables: List[str] = Field(..., description="本次查询必须访问的真实物理表名")
    time_range: Dict[str, str] = Field(default={}, description="标准化时间范围，包含 start 和 end 字段")
    filters: List[str] = Field(default=[], description="自然语言描述的过滤条件，例如 '广东省且GDP>1000'")
    aggregation: Optional[str] = Field(None, description="需要采用的数据聚合方式 (如 sum, avg, count)")
    top_k: Optional[int] = Field(None, description="若涉及排行，返回 Limit 截断数值")
    sort_order: Optional[str] = Field(None, description="结果排序意图 (如 按GDP降序)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="大模型对自己此次规划逻辑的自信心评估 (0-1)")
    raw_sql: Optional[str] = Field(None, description="【预览生成】大模型草拟的可执行 PostgreSQL 语句 (供后续编排器参考或直接执行)")
    dim_cols: List[str] = Field(default=[], description="【多维分析(multi_dim)专用】交叉分析的维度列名，如 ['省区名称', '年份']。非 multi_dim 类型可以为空。")

    @model_validator(mode="before")
    @classmethod
    def _normalize_time_range(cls, data: Any) -> Any:
        """将 null time_range 归一化为空字典，避免 Pydantic 校验失败。"""
        if isinstance(data, dict) and not isinstance(data.get("time_range"), dict):
            data["time_range"] = {}
        return data

# ── 2. Parser 解析指挥调度中心 ──

class QueryParserEngine:

    def __init__(self) -> None:
        self._schema_rag: SchemaRAGEngine | None = None
        self._join_finder: JoinPathFinder | None = None
        self._cache: SemanticCache | None = None

    def initialize(
        self, schema_rag: SchemaRAGEngine, join_finder: JoinPathFinder
    ) -> None:
        """应用启动时注入依赖。"""
        self._schema_rag = schema_rag
        self._join_finder = join_finder
        logger.info("QueryParserEngine initialized with SchemaRAGEngine + JoinPathFinder")

    def set_cache(self, cache: SemanticCache) -> None:
        """注入语义缓存引擎。"""
        self._cache = cache
        logger.info("QueryParserEngine 已接入语义缓存")

    @staticmethod
    def _clean_llm_json(text: str) -> str:
        """清理 LLM 输出中的常见 JSON 格式问题。"""
        # 移除 markdown 代码块标记
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        # 替换单引号为双引号（仅 JSON key 和字符串值）
        # 但保留 raw_sql 中的单引号（PostgreSQL 字符串字面量）
        import re
        # 找出 raw_sql 部分并保护
        sql_match = re.search(r'"raw_sql"\s*:\s*"([^"]*)"', text, re.DOTALL)
        protected_sql = ""
        if sql_match:
            protected_sql = sql_match.group(1)
            text = text.replace(protected_sql, "__RAW_SQL_PLACEHOLDER__")

        # 替换键名周围的单引号为双引号
        text = re.sub(r"(?<!\\)'(?:\w+)'(?=\s*:)", lambda m: m.group(0).replace("'", '"'), text)

        # 还原 raw_sql
        if protected_sql:
            text = text.replace("__RAW_SQL_PLACEHOLDER__", protected_sql)

        return text

    @staticmethod
    def _quote_chinese_cols(sql: str) -> str:
        """自动给 SQL 中需要引号的列名加上双引号。

        PostgreSQL 对以下标识符要求双引号：
        - 包含中文的列名（如 区划ID → "区划ID"）
        - 包含大写字母的列名（如 GDP → "GDP"，否则被折叠为 gdp）
        """
        import re
        SQL_KEYWORDS = {
            'SELECT', 'FROM', 'WHERE', 'AS', 'AND', 'OR', 'NOT', 'IN',
            'ON', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'FULL',
            'CROSS', 'NATURAL', 'ORDER', 'BY', 'GROUP', 'HAVING',
            'LIMIT', 'OFFSET', 'ASC', 'DESC', 'NULL', 'IS', 'LIKE',
            'BETWEEN', 'EXISTS', 'DISTINCT', 'ALL', 'UNION', 'CASE',
            'WHEN', 'THEN', 'ELSE', 'END', 'CAST', 'COALESCE',
            # CTE / 子查询
            'WITH', 'RECURSIVE', 'ARRAY',
            # 窗口函数关键字
            'OVER', 'PARTITION', 'ROWS', 'RANGE', 'UNBOUNDED',
            'PRECEDING', 'FOLLOWING', 'CURRENT',
            # 窗口函数名
            'ROW_NUMBER', 'RANK', 'DENSE_RANK', 'NTILE',
            'LAG', 'LEAD', 'FIRST_VALUE', 'LAST_VALUE', 'NTH_VALUE',
            # 排序与过滤
            'FILTER', 'WITHIN', 'HAVING',
            # 表操作
            'TABLE', 'FOR', 'LIKE', 'ILIKE', 'SOME', 'ANY',
            'FETCH', 'ONLY', 'TIES', 'EXCEPT', 'INTERSECT',
            # 值与类型
            'TRUE', 'FALSE', 'ROW', 'SET',
            'EXTRACT', 'DATE_TRUNC', 'INTERVAL',
            'OVERLAPS', 'ISNULL', 'NOTNULL',
        }
        FUNC_NAMES = {
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'ROUND',
            'COALESCE', 'CAST', 'ABS', 'UPPER', 'LOWER',
            'STRING_AGG', 'ARRAY_AGG', 'JSON_AGG',
            'GREATEST', 'LEAST', 'NULLIF',
            'PERCENTILE_CONT', 'PERCENTILE_DISC',
            'NTILE', 'ROW_NUMBER', 'RANK', 'DENSE_RANK',
            'LAG', 'LEAD', 'FIRST_VALUE', 'LAST_VALUE', 'NTH_VALUE',
        }
        ALL_KEYWORDS = SQL_KEYWORDS | FUNC_NAMES

        def _quote_col(m):
            word = m.group(0)
            # 跳过已有引号 — 但若引号内是 SQL 关键字，则去掉引号
            if word.startswith('"'):
                inner = word.strip('"')
                if inner.upper() in ALL_KEYWORDS:
                    return inner
                return word
            # 跳过 SQL 关键字/函数
            if word.upper() in ALL_KEYWORDS:
                return word
            # 跳过纯数字
            if word.isdigit():
                return word
            # 包含中文字符 → 必须引
            if re.search(r'[一-鿿]', word):
                return f'"{word}"'
            # 包含大写字母且不是全小写 → 必须引（如 GDP, GDP增长率）
            if word != word.lower():
                return f'"{word}"'
            return word

        # 匹配标识符：字母/数字/中文/下划线序列（不能以数字开头）
        pattern = r'[一-鿿\w][一-鿿\w]*'
        # 后处理：修复列名中包含点号的情况（如 PM2.5 -> "PM2.5"）
        result = re.sub(pattern, _quote_col, sql)

        # 通用 quoting 后 PM2.5 会被拆成 "PM2".5，需要合并
        result = re.sub(r'"([A-Za-z_]\w+)"\.(\w+)', r'"\1.\2"', result)

        # 清理嵌套引号
        result = result.replace('""', '"')

        # 全局卸载 SQL 关键字的引号（兜底：处理像 "WITH" 被 LLM 提前加引号的情况）
        for kw in sorted(SQL_KEYWORDS, key=len, reverse=True):
            result = result.replace(f'"{kw}"', kw)
            result = result.replace(f'"{kw.lower()}"', kw.lower())

        return result

    async def _route_intent(self, user_query: str) -> RouteResult:
        """
        【导诊台 / 路由专家】：使用低成本模型快速打标
        """
        prompt = f"""
        你是一个数据分析入口路由专家。请分析用户的输入，并严格以 JSON 格式输出。
        不要附带任何 markdown 标记或额外文本，仅输出纯 JSON！

        【用户输入】: "{user_query}"

        【输出示例（严格遵循此结构，字段类型必须一致）】
        {{
            "intent": "query_data",
            "indicators": ["GDP", "空气质量指数"],
            "time_hint": "2023年",
            "complexity": 4
        }}

        字段说明：
        - intent: 必须为 "query_data" 或 "greetings" 或 "ambiguous"
        - indicators: 字符串数组，提取的业务指标
        - time_hint: 字符串或 null（没有时间约束则为 null）
        - complexity: 整数 1-5
        """
        response_text = await chat_with_model("openai", [{"role": "user", "content": prompt}], temperature=0.0)
        
        try:
            #  生产级容错：使用 json_repair 修复大模型偶尔漏掉引号/括号的脑残 JSON
            parsed_dict = json_repair.loads(response_text)
            return RouteResult.model_validate(parsed_dict)
        except Exception as e:
            logger.error(f"路由解析 JSON 失败: {e} \nRaw: {response_text}")
            # Fail-safe：路由失败时，强制认定为复杂查数，走全链路兜底
            return RouteResult(intent="query_data", indicators=[], complexity=5)

    async def _complex_disambiguation(self, user_query: str) -> Dict[str, Any]:
        """
        【消歧专家】：进行复杂语意澄清
        """
        prompt = f"""
        用户输入了一句非常模糊或具有歧义的查数请求："{user_query}"。
        你需要分析其中的歧义点，并向用户发起友好的追问。
        请直接输出追问的文本，不要附带 JSON。
        """
        clarification_text = await chat_with_model("openai", [{"role": "user", "content": prompt}])
        return {"status": "clarify", "detail": clarification_text}

    async def _generate_sql_plan(self, user_query: str, route_meta: RouteResult) -> SQLAgentPlan:
        """
        【SQL架构主刀医生】：融合 RAG 上下文，生成标准执行规划
        """
        logger.info("Executing Schema-RAG dual-fusion retrieval...")
        
        # 1. 将用户问句与路由提取的 indicators 作为核心特征进行双路召回
        search_target = f"{user_query} {' '.join(route_meta.indicators)} {route_meta.time_hint or ''}"
        schema_context, final_tables = self._schema_rag.retrieve_schema_context(search_target)

        # 2. 调用 BFS 寻路算法，生成确定性的多表 JOIN 物理指导建议
        logger.info(f"Target Tables for Planning: {final_tables}")
        join_instructions = self._join_finder.get_join_instructions(final_tables)

        # 3. 组装极度严苛的 System Prompt
        system_prompt = f"""
        你是一个处于世界顶尖水平的 PostgreSQL 商业智能数据架构师。
        你的唯一任务是将用户的自然语言问题转化为高度结构化的 JSON 业务执行规划。

        {schema_context}

        ## 【跨表安全关联强制指导原则 (BFS Join Map)】
        如果你生成的 SQL 必须使用两个及以上的表，请严格遵循以下物理寻路系统生成的 JOIN 关联条件，绝对禁止自由发挥引发笛卡尔积或断桥：
        {join_instructions}

        ## 【JSON 输出契约——必须严格遵守】

        仅输出纯 JSON，禁止附带任何 markdown 标记、代码块、注释或额外文本。

        【完整输出示例（字段值仅供格式参考）】
        {{
            "analysis_type": "trend",
            "indicators": ["GDP", "GDP增长率"],
            "tables": ["economic_indicator_data"],
            "time_range": {{"start": "2020", "end": "2023"}},
            "filters": [],
            "aggregation": null,
            "top_k": null,
            "sort_order": null,
            "confidence": 0.92,
            "raw_sql": "SELECT \"区划名称\", \"统计年份\", \"GDP\", \"GDP增长率\" FROM economic_indicator_data WHERE \"统计年份\" >= '2020' AND \"统计年份\" <= '2023' ORDER BY \"统计年份\""
        }}

        【字段类型要求——类型错误会导致系统崩溃】
        - analysis_type: 字符串，必须为以下 10 种类型之一（仔细阅读每种的使用场景）：
          1. "trend"       — 时间序列/趋势分析：CAGR、线性趋势、移动平均。适用场景：增长率、趋势线、滚动平均。
          2. "rank"        — 排行与对比：百分位排名、等距分档(5档)、直接排序。适用场景：综合排名、分级分类。
          3. "detail"      — 纯明细查询：不需要分析算法。适用场景：仅查询原始数据、简单汇总。
          4. "correlation" — 相关与因果分析：Pearson/Spearman相关系数、互信息、Granger因果。适用场景：相关关系、因果关系、时间滞后分析。
          5. "anomaly"     — 异常检测：Z-score、IQR四分位距、3σ原则、双重融合检测。适用场景：异常值发现、突变检测。
          6. "composite"   — 综合评价：熵权法自动定权重、TOPSIS优劣解距离法、PCA主成分分析。适用场景：综合发展指数、多指标加权评分。
          7. "multi_dim"   — 多维交叉分析：多维聚合(Cube)、占比计算(Proportion)、变异系数(CV)。适用场景：多维度交叉对比。
          8. "spatial"     — 区域空间分析：百分位排名、聚类(K-means/层次聚类)、泰尔指数分解。适用场景：区域对比、省际差距分析。
          9. "cross_domain" — 跨域综合分析：熵权+MinMax、耦合协调度(CCD)、DEA效率、面板OLS、Granger因果、弹性系数。适用场景：多系统协调度、投入产出效率。
          10. "advanced"    — 高级统计：Pearson/Spearman相关系数、Z-score标准化。适用场景：简单相关性分析、通用统计。

          选择依据：看用户问题核心分析方法。"综合评价用权重"→composite，"因果/弹性/相关"→correlation，
          "多系统协调"→cross_domain，"聚类/区域差距"→spatial，"多维交叉"→multi_dim，
          "检测异常"→anomaly，"趋势/增长率"→trend，"排名/对比"→rank。
        - indicators: 字符串数组，不能是 null 或缺失
        - tables: 字符串数组，必须存在的数据表名
        - time_range: 对象（字典），不能是字符串，格式为 {{"start": "年份", "end": "年份"}}
        - filters: 字符串数组，无条件则为空数组 []
        - aggregation: 字符串或 null
        - top_k: 整数或 null，不能加引号
        - sort_order: 字符串或 null
        - confidence: 浮点数 0-1 之间，不能加引号
        - raw_sql: 字符串或 null，必须是合法 PostgreSQL 查询语句
        - dim_cols: 字符串数组，仅当 analysis_type 为 "multi_dim" 时必须提供；其他类型可以为空数组 []。
          含义是多维交叉分析中的维度列名，如 ["省区名称", "年份"]。这些列会在 Cube(多维汇总)和 Proportion(占比计算)中被用作分组/透视维度。

        注意：raw_sql 中字符串值必须使用单引号（PostgreSQL 标准），而非双引号。

        【列名准确性——这是最重要的要求】
        raw_sql 中使用的所有列名必须与上方提供的物理表结构中的列名完全一致，不可修改。
        例如：如果表结构显示列名为"区划ID"，你必须在 SQL 中写 "区划ID"，不能写成"区域id"或"地区ID"。
        列名是中文就写中文，是英文就写英文，一字不差。
        包含中文的列名必须用双引号包裹，例如 "区划ID"。英文列名如 GDP 不需要引号。

        【原始指标列保留规则——算法分析的前置条件】
        当 analysis_type 不是 "detail" 时，这是必须遵守的硬性要求：
        1. raw_sql 的 SELECT 子句必须包含 indicators 中列出的所有指标对应的原始列名。
           例如 indicators=["GDP","固定资产投资"] 则 SELECT 中必须出现 "GDP","固定资产投资" 两列。
        2. 不允许用 AS 别名掩盖原始列名。如果需要在 SELECT 中做计算（如 GDP/人口），
           必须在 SELECT 中同时保留原始列和计算结果，不能只用计算结果替代原始列。
        3. 此规则的目的是保证后续分析算法能获取到原始指标数据。

        【维度标识列保留规则——数据可读性的前置条件】
        当查询涉及多个实体（省/市/区/行业/产品）的对比、排行、相关性计算或明细展示时，
        raw_sql 的 SELECT 子句中必须至少包含一个能区分不同实体的维度标识列
        （如"省区名称""区划名称""行业名称"等），
        否则用户无法识别每条数据对应哪个实体，数据展示将失去意义。

        例如：查询各省 GDP 对比 → SELECT 必须有 "省区名称"
              查询各行业营收排名 → SELECT 必须有 "行业名称"
              查询各区县人口 → SELECT 必须有 "区划名称"
        注意：如果查询仅涉及单实体或汇总数据（如全国总量），则无需添加此列。

        【SQL 关键字——绝对不能使用双引号包裹】
        SQL 关键字（WITH, OVER, PARTITION BY, AS, AND, OR, ON, JOIN,
        LEFT, RIGHT, INNER, OUTER, FULL, CROSS, WHERE, ORDER BY,
        GROUP BY, HAVING, SELECT, FROM, DISTINCT, CASE WHEN, THEN,
        ELSE, END, NOT, IN, BETWEEN, EXISTS, UNION, ALL, LIMIT,
        DESC, ASC, LAG, LEAD, ROW_NUMBER, RANK, DENSE_RANK, NTILE,
        SUM, AVG, COUNT, MAX, MIN, COALESCE, NULLIF, CAST 等）
        绝对不能使用双引号包裹。
        只有包含中文的列名（如 "省区名称"、"固定资产投资"）才需要使用双引号。
        反例：SELECT "省区名称", "GDP" FROM ... WHERE "年份" = '2023' ORDER BY "GDP" DESC
              上面的 "WHERE"、"FROM"、"ORDER"、"BY"、"DESC" 等关键字都不应该有引号！
        """

        # 4. 提交给 LLM 生成
        logger.info("Submitting planning task to LLM...")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"用户业务问题: {user_query}"}
        ]

        response_text = await chat_with_model("openai", messages, temperature=0.1)

        # 5. 预处理：清理常见格式问题
        response_text = self._clean_llm_json(response_text)

        # 6. 反序列化（失败时自动重试 1 次）
        for attempt in range(2):
            try:
                parsed_dict = json_repair.loads(response_text)

                # 清理 Pydantic 校验前的不合法值
                if not isinstance(parsed_dict.get("time_range"), dict):
                    parsed_dict["time_range"] = {}

                plan = SQLAgentPlan.model_validate(parsed_dict)

                # 7. SQL 后处理：自动给中文列名加双引号（PostgreSQL 要求）
                if plan.raw_sql:
                    plan.raw_sql = self._quote_chinese_cols(plan.raw_sql)

                return plan
            except Exception as e:
                if attempt == 0:
                    logger.warning("JSON 校验失败，重试中: %s", e)
                    # 追加包含具体校验错误的格式说明到 system prompt 并重试
                    strict_hint = (
                        f"\n\n【重要】上次输出 JSON 格式校验失败。错误详情:\n{e}\n\n"
                        f"请严格遵守以下字段类型：\n"
                        f"- time_range 必须是一个对象（字典），如 {{\"start\": \"2020\", \"end\": \"2023\"}}，不能是字符串或 null\n"
                        f"- indicators 必须是字符串数组，不能是 null\n"
                        f"- confidence 必须是浮点数（0-1），不能加引号\n"
                        f"- top_k 必须是整数或 null，不能加引号\n"
                        f"- 只输出纯 JSON，不要 markdown 代码块，不要添加任何注释\n"
                        f"- 使用双引号包裹所有 JSON key 和字符串值"
                    )
                    messages[0]["content"] += strict_hint
                    response_text = await chat_with_model("openai", messages, temperature=0.3)
                    response_text = self._clean_llm_json(response_text)
                else:
                    logger.error(f"JSON 校验重试仍失败: {e} \nRaw: {response_text}")
                    raise RuntimeError(f"大模型未能输出合法的结构化 JSON: {str(e)}")

    # ── 对外暴露主入口 ──

    async def parse(self, user_query: str) -> Dict[str, Any]:
        """全流程四步总干事"""
        logger.info(f" Parsing query intent: {user_query}")

        # Step 0: 语义缓存快速命中（0.1s + 零 Token）
        if self._cache is not None:
            try:
                cached = await self._cache.lookup(user_query)
                if cached is not None:
                    logger.info("语义缓存 HIT: query='%s'", user_query[:80])
                    return cached
            except Exception as exc:
                logger.warning("语义缓存查询异常（忽略，走正常流程）: %s", exc)

        # Step 1: 低成本意图拦截
        route_meta = await self._route_intent(user_query)
        logger.info(f"Router Result: Intent={route_meta.intent}, Cplx={route_meta.complexity}, Ind={route_meta.indicators}")

        # Step 2: 意图过滤
        if route_meta.intent == "greetings":
            return {"status": "reply", "message": "您好！我是城市综合治理与经济分析的专属数据助手，请问您需要分析哪些指标？"}
            
        if route_meta.intent == "ambiguous":
            logger.warning("Triggering disambiguation fallback.")
            return await self._complex_disambiguation(user_query)

        # Step 3: 全核心运转，深度业务规划
        try:
            plan = await self._generate_sql_plan(user_query, route_meta)
            
            return {
                "status": "ready_for_execution",
                "route_metadata": route_meta.model_dump(),
                "execution_plan": plan.model_dump()
            }
            
        except Exception as e:
            logger.exception("Parse phase failed.")
            return {"status": "error", "message": f"系统解析用户意图时发生未知异常: {str(e)}"}

# 实例化全局单例
parser_engine = QueryParserEngine()