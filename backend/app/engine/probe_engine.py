"""
因果探针引擎 — 受控自动归因分析
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── 探针模板 ──

PROBE_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "fluctuation",
        "name": "波动探针",
        "description": "当指标 X 波动 >20% 时，查询同一时间窗口的相关指标 Y",
        "condition": "波动 > 20%",
    },
    {
        "id": "ranking_change",
        "name": "排名探针",
        "description": "当排名变化时，查询对比维度的细分数据",
        "condition": "排名变化",
    },
    {
        "id": "anomaly",
        "name": "异常探针",
        "description": "当检测到异常值时，查询同期历史基准",
        "condition": "异常值检测",
    },
]


@dataclass
class ProbeResult:
    """单次探针执行结果。"""
    template_id: str
    sql: str
    data: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0


class ProbeEngine:
    """因果探针引擎。

    集成位置：Gate③ 校验通过后 → ProbeEngine → Interpreter 解读生成。

    防护栏：
    - 最多 1 轮探针
    - 单探针 ≤5s 超时
    - 仅预置模板，不自由生成 SQL
    """

    def __init__(self, db_pool) -> None:  # noqa: ANN001
        self._pool = db_pool

    async def probe(self, main_result: dict[str, Any]) -> list[ProbeResult]:
        """执行因果探针查询。

        触发条件：
        - Interpreter 检测到主结果存在显著数据波动（待 Interpreter 调用时传入）
        - 调用方判定符合条件后调用此方法
        """
        results: list[ProbeResult] = []

        for template in PROBE_TEMPLATES:
            sql = self._build_sql(template, main_result)
            if not sql:
                continue

            start = asyncio.get_event_loop().time()
            try:
                probe_task = asyncio.wait_for(
                    self._pool.fetch(sql),
                    timeout=5.0,
                )
                rows = await probe_task
                latency = (asyncio.get_event_loop().time() - start) * 1000

                results.append(ProbeResult(
                    template_id=template["id"],
                    sql=sql,
                    data=[dict(r) for r in rows],
                    latency_ms=round(latency, 1),
                ))
                logger.info("Probe '%s' done in %.0fms", template["id"], latency)
            except asyncio.TimeoutError:
                latency = (asyncio.get_event_loop().time() - start) * 1000
                logger.warning("Probe '%s' timed out (%.0fms)", template["id"], latency)
                results.append(ProbeResult(
                    template_id=template["id"],
                    sql=sql,
                    error="timeout",
                    latency_ms=round(latency, 1),
                ))
            except Exception as exc:
                logger.warning("Probe '%s' failed: %s", template["id"], exc)
                results.append(ProbeResult(
                    template_id=template["id"],
                    sql=sql,
                    error=str(exc),
                ))

        return results

    def merge_to_insight(self, main_result: dict[str, Any], probe_results: list[ProbeResult]) -> dict[str, Any]:
        """将探针结果合并到主结果中，供 LLM 归因分析使用。"""
        main_result["probe_results"] = [
            {
                "template_id": r.template_id,
                "data": r.data[:10],         # 截断长结果
                "error": r.error,
                "latency_ms": r.latency_ms,
            }
            for r in probe_results
        ]
        main_result["probe_summary"] = (
            f"因果探针: {sum(1 for r in probe_results if r.error is None)}/3 成功"
        )
        return main_result

    # ── 内部 ──

    def _build_sql(self, template: dict[str, Any], main_result: dict[str, Any]) -> str | None:
        """根据模板 + 主结果生成探针 SQL。"""
        tables = main_result.get("tables", [])
        indicators = main_result.get("indicators", [])
        time_range = main_result.get("time_range", {})

        if not tables or not indicators:
            return None

        table = tables[0] if isinstance(tables, list) else tables
        indicator = indicators[0] if isinstance(indicators, list) else indicators
        time_col = "year"

        if template["id"] == "fluctuation":
            # 查询同一时间窗口的所有指标值
            return (
                f"SELECT {time_col}, {indicator} FROM {table} "
                f"WHERE {time_col} BETWEEN '{time_range.get('start', '2020')}' AND '{time_range.get('end', '2023')}' "
                f"ORDER BY {time_col}"
            )

        if template["id"] == "ranking_change":
            # 查询排名对比数据
            return (
                f"SELECT province, {time_col}, {indicator} FROM {table} "
                f"WHERE {time_col} IN ('{time_range.get('start', '2020')}', '{time_range.get('end', '2023')}') "
                f"ORDER BY {indicator} DESC"
            )

        if template["id"] == "anomaly":
            # 查询历史基准
            return (
                f"SELECT {time_col}, AVG({indicator}) as avg_val, "
                f"STDDEV({indicator}) as std_val "
                f"FROM {table} GROUP BY {time_col} ORDER BY {time_col}"
            )

        return None
