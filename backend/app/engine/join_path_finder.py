"""
Join 路径发现 — BFS 搜索表间最短关联路径
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class JoinEdge:
    target: str
    type: str = "inner"
    on: list[str] | None = None
    via: str | None = None


@dataclass
class JoinPath:
    """可序列化的关联路径。"""
    path: list[str]                       # 表名序列（含中介表）
    joins: list[dict[str, Any]] = field(default_factory=list)  # 每步 JOIN 详情

    def to_llm_context(self) -> str:
        """序列化为 LLM 可读的 SQL JOIN 提示。"""
        if not self.joins:
            return ""
        lines: list[str] = ["关联路径 (Join Path):"]
        for j in self.joins:
            if j.get("on"):
                lines.append(f"  {j['from']} {j['type']} JOIN {j['to']} ON {j['from']}.{j['on']} = {j['to']}.{j['on']}")
            elif j.get("via"):
                lines.append(f"  {j['from']} → {j['via']} → {j['to']} (经由中介表)")
            else:
                lines.append(f"  {j['from']} {j['type']} JOIN {j['to']}")
        return "\n".join(lines)


class JoinPathFinder:
    """全局表关联图 + BFS 最短路径搜索。"""

    def __init__(self, yaml_path: str | Path = "app/configs/join_graph.yaml") -> None:
        self._yaml_path = Path(yaml_path)
        self._graph: dict[str, list[JoinEdge]] = {}
        self._loaded: bool = False

    def load(self) -> None:
        """加载 YAML 关联图谱。"""
        if not self._yaml_path.exists():
            msg = f"Join graph not found: {self._yaml_path}"
            raise FileNotFoundError(msg)

        with self._yaml_path.open("r", encoding="utf-8") as f:
            data: dict[str, list[dict[str, Any]]] = yaml.safe_load(f)

        self._graph.clear()
        for item in data.get("tables", []):
            name = item["name"]
            edges: list[JoinEdge] = []
            for j in item.get("joins", []):
                edges.append(JoinEdge(
                    target=j["target"],
                    type=j.get("type", "inner"),
                    on=j.get("on"),
                    via=j.get("via"),
                ))
            self._graph[name] = edges
        self._loaded = True

    def find_path(self, from_table: str, to_table: str) -> JoinPath | None:
        """BFS 搜索 from_table → to_table 的最短关联路径。"""
        if from_table not in self._graph or to_table not in self._graph:
            return None
        if from_table == to_table:
            return JoinPath(path=[from_table])

        visited: set[str] = {from_table}
        queue: deque[tuple[str, list[tuple[str, JoinEdge]]]] = deque()
        queue.append((from_table, []))

        while queue:
            current, edges = queue.popleft()
            for edge in self._graph.get(current, []):
                if edge.target in visited:
                    continue
                new_edges = [*edges, (current, edge)]
                if edge.target == to_table:
                    return self._build_path(new_edges)
                visited.add(edge.target)
                queue.append((edge.target, new_edges))

        return None

    def find_all_paths(self, tables: list[str]) -> dict[tuple[str, str], JoinPath | None]:
        """批量查找：对 tables 中所有表对执行 BFS。"""
        results: dict[tuple[str, str], JoinPath | None] = {}
        for i, t1 in enumerate(tables):
            for t2 in tables[i + 1:]:
                path = self.find_path(t1, t2)
                results[(t1, t2)] = path
                results[(t2, t1)] = path
        return results

    # ── 内部 ──

    def _build_path(self, edges: list[tuple[str, JoinEdge]]) -> JoinPath:
        path: list[str] = []
        joins: list[dict[str, Any]] = []
        for src, edge in edges:
            if not path or path[-1] != src:
                path.append(src)
            if edge.via:
                path.append(edge.via)
            path.append(edge.target)
            joins.append({
                "from": src,
                "to": edge.target,
                "type": edge.type,
                "on": edge.on[0] if edge.on and len(edge.on) == 1 else edge.on,
                "via": edge.via,
            })
        return JoinPath(path=path, joins=joins)

    @property
    def table_count(self) -> int:
        return len(self._graph)

    def get_all_tables(self) -> list[str]:
        return list(self._graph.keys())

    def get_join_instructions(self, tables: list[str]) -> str:
        """生成 LLM 可读的 JOIN 指令（多表关联时使用）。"""
        if len(tables) <= 1:
            return "单表查询，无需 JOIN。"

        paths = self.find_all_paths(tables)
        parts: list[str] = []
        for (t1, t2), path in paths.items():
            if path is None:
                parts.append(f"  ⚠ {t1} ↔ {t2}: 未发现可用关联路径")
            else:
                ctx = path.to_llm_context()
                if ctx:
                    parts.append(ctx)
        return "\n".join(parts) if parts else "未发现任何表间关联路径。"

    # ── 运行时追加/删除（供 TableImporter 调用） ──

    def _ensure_loaded(self) -> None:
        """确保图谱已从 YAML 加载（懒加载保护）。"""
        if not self._loaded:
            if self._yaml_path.exists():
                self.load()
            else:
                self._loaded = True  # 文件不存在时标记为已加载（空图）

    def add_table_node(
        self, table_name: str, joins: list[dict[str, Any]],
    ) -> None:
        """追加表节点及其 JOIN 边到图谱，同时更新双向边。

        joins: [{"target": str, "type": str, "on": list[str], "via": str|None}, ...]
        """
        self._ensure_loaded()
        edges: list[JoinEdge] = []
        for j in joins:
            edge = JoinEdge(
                target=j["target"],
                type=j.get("type", "left"),
                on=j.get("on"),
                via=j.get("via"),
            )
            edges.append(edge)
            # 在目标表中添加反向边，实现双向可达
            target = j["target"]
            if target in self._graph:
                reverse_edge = JoinEdge(
                    target=table_name,
                    type=j.get("type", "left"),
                    on=j.get("on"),
                    via=j.get("via"),
                )
                # 避免重复添加
                if not any(e.target == table_name for e in self._graph[target]):
                    self._graph[target].append(reverse_edge)
        self._graph[table_name] = edges

        # 同步追加到 admin_region_data（如果新表通过区划ID关联）
        hub = "admin_region_data"
        if table_name != hub and hub in self._graph:
            hub_joins = [
                j for j in joins
                if j["target"] == hub
                or (j.get("on") and "区划ID" in (j["on"] or []))
            ]
            if hub_joins:
                hub_join = hub_joins[0]
                hub_edge = JoinEdge(
                    target=table_name,
                    type=hub_join.get("type", "left"),
                    on=hub_join.get("on"),
                    via=None,
                )
                if not any(e.target == table_name for e in self._graph[hub]):
                    self._graph[hub].append(hub_edge)

        self._rewrite_yaml()
        logger.info(
            "add_table_node: '%s' (%d join edges)", table_name, len(edges),
        )

    def remove_table_node(self, table_name: str) -> None:
        """删除表节点及其所有关联边。"""
        self._ensure_loaded()
        self._graph.pop(table_name, None)
        # 从其他表的 join 列表中移除指向此表的边
        for edges in self._graph.values():
            edges[:] = [e for e in edges if e.target != table_name]
        self._rewrite_yaml()
        logger.info("remove_table_node: '%s'", table_name)

    def _rewrite_yaml(self) -> None:
        """全量重写 join_graph.yaml。"""
        if not self._yaml_path.exists():
            logger.warning("join_graph.yaml 不存在，无法重写")
            return

        # 读取现有 YAML 保留注释头部
        existing = self._yaml_path.read_text(encoding="utf-8")
        header_lines: list[str] = []
        for line in existing.split("\n"):
            if line.startswith("#") or line.strip() == "":
                header_lines.append(line)
            else:
                break

        # 构建 tables 列表
        tables_data: list[dict[str, Any]] = []
        for table_name, edges in self._graph.items():
            table_entry: dict[str, Any] = {
                "name": table_name,
                "joins": [
                    {
                        "target": e.target,
                        "type": e.type,
                        "on": e.on,
                        "via": e.via,
                    }
                    for e in edges
                ],
            }
            tables_data.append(table_entry)

        content = "\n".join(header_lines)
        if content and not content.endswith("\n"):
            content += "\n"
        content += yaml.dump(
            {"tables": tables_data},
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
        self._yaml_path.write_text(content, encoding="utf-8")
        logger.info("join_graph.yaml rewritten: %d tables", len(tables_data))
