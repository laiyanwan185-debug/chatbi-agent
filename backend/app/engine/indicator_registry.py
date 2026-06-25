"""
指标注册表 — 业务名到物理字段/跨表公式的确定性映射引擎
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)


# ── 1. 结构化指标数据容器 ──
class IndicatorDef(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    type: str = "direct"          # direct | computed | category
    field: str | None = None      # 物理列名（仅 type=direct 有效）
    formula: str | None = None    # 计算公式（仅 type=computed 有效）
    table: str | None = None      # 物理主表名
    field_mappings: dict[str, dict[str, str]] = Field(default_factory=dict)
    # 格式: {"变量名": {"table": "物理表", "column": "物理列"}}
    # 用于 computed 指标跨表公式溯源


# ── 2. 指标注册中心管理引擎 ──
class IndicatorRegistry:
    """提供业务指标名（及同义词）到数据库物理表/物理列的确定性翻译。"""

    def __init__(self, yaml_path: str | Path | None = None) -> None:
        self._yaml_path = Path(yaml_path or settings.INDICATORS_CONFIG_PATH)
        self._indicators: dict[str, IndicatorDef] = {}   # 标准名 → Def
        self._alias_map: dict[str, str] = {}              # 别名 → 标准名

    def load(self) -> None:
        """从 YAML 文件装载指标关系。"""
        if not self._yaml_path.exists():
            msg = f"Indicator registry file not found: {self._yaml_path.absolute()}"
            raise FileNotFoundError(msg)

        with self._yaml_path.open("r", encoding="utf-8") as f:
            data: dict[str, list[dict[str, Any]]] = yaml.safe_load(f)

        self._indicators.clear()
        self._alias_map.clear()

        for item in data.get("indicators", []):
            name = item["name"]
            aliases = item.get("aliases", [])
            ind_type = item.get("type", "direct")

            # 校验：direct 类型必须指定 field
            if ind_type == "direct" and not item.get("field"):
                logger.warning(
                    "direct indicator '%s' missing 'field', skipped", name
                )
                continue

            ind = IndicatorDef(
                name=name,
                aliases=aliases,
                type=ind_type,
                field=item.get("field"),
                formula=item.get("formula"),
                table=item.get("table"),
                field_mappings=item.get("field_mappings", {}),
            )
            self._indicators[name] = ind
            for alias in aliases:
                self._alias_map[alias.lower()] = name

        logger.info("Indicator registry loaded: %d items", len(self._indicators))

    def resolve(self, indicator_name: str) -> IndicatorDef | None:
        """指标名（或别名）→ IndicatorDef。"""
        query = indicator_name.lower()
        canonical = self._alias_map.get(query, indicator_name)
        return self._indicators.get(canonical)

    def search_field_by_name(self, name: str) -> str | None:
        """通过中文名/别名反向查找物理列名（field）。

        用于分析器参数名（如 "识字率"）与 SQL 输出列名（如 "literacy_rate"）
        不匹配时的反向查找。返回第一个匹配的 direct 类型指标的 field 值。
        """
        if not name:
            return None
        name_lower = name.lower().strip('"\' ')
        for ind in self._indicators.values():
            if ind.type != "direct" or not ind.field:
                continue
            if ind.name.lower() == name_lower:
                return ind.field
            if any(name_lower == alias.lower() for alias in ind.aliases):
                return ind.field
        return None

    def search(self, keyword: str) -> list[IndicatorDef]:
        """关键词模糊匹配（name + aliases）。"""
        kw = keyword.lower()
        results: list[IndicatorDef] = []
        for ind in self._indicators.values():
            if kw in ind.name.lower():
                results.append(ind)
                continue
            if any(kw in alias.lower() for alias in ind.aliases):
                results.append(ind)
        return results

    def get_all(self) -> list[IndicatorDef]:
        return list(self._indicators.values())

    @property
    def size(self) -> int:
        return len(self._indicators)

    # ── 2b. 运行时追加/删除（供 TableImporter 调用） ──

    def add_indicator(self, ind: IndicatorDef) -> None:
        """运行时追加指标定义，同时回写 YAML 文件保持持久化。"""
        self._indicators[ind.name] = ind
        for alias in ind.aliases:
            self._alias_map[alias.lower()] = ind.name
        self._append_to_yaml(ind)
        logger.info("add_indicator: '%s' (table=%s)", ind.name, ind.table)

    def add_indicators(self, indicators: list[IndicatorDef]) -> None:
        """批量追加指标，统一一次 YAML 回写。"""
        for ind in indicators:
            self._indicators[ind.name] = ind
            for alias in ind.aliases:
                self._alias_map[alias.lower()] = ind.name
        # 统一追加到 YAML 文件（带注释分段）
        self._append_multi_to_yaml(indicators)
        logger.info(
            "add_indicators: %d items for %d tables",
            len(indicators),
            len({i.table for i in indicators}),
        )

    def remove_indicators_for_table(self, table_name: str) -> int:
        """删除指定表的所有指标（表删除时调用）。返回删除数量。"""
        to_remove = [
            name for name, ind in self._indicators.items()
            if ind.table == table_name
        ]
        for name in to_remove:
            ind = self._indicators.pop(name)
            for alias in ind.aliases:
                self._alias_map.pop(alias.lower(), None)
        if to_remove:
            self._rewrite_yaml_without_table(table_name)
        logger.info(
            "remove_indicators_for_table: '%s' → removed %d",
            table_name, len(to_remove),
        )
        return len(to_remove)

    def _append_to_yaml(self, ind: IndicatorDef) -> None:
        """追加单个指标到 YAML 文件末尾（带注释分段）。"""
        self._append_multi_to_yaml([ind])

    def _append_multi_to_yaml(self, indicators: list[IndicatorDef]) -> None:
        """批量追加指标到 YAML 文件，带 [AUTO-IMPORTED] 注释标记。"""
        if not self._yaml_path.exists():
            self._rewrite_yaml()
            return

        # 读取现有 YAML 内容
        try:
            with self._yaml_path.open("r", encoding="utf-8") as f:
                content = f.read()
            data = yaml.safe_load(content) or {"indicators": []}
        except Exception:
            logger.warning("_append_multi_to_yaml: 无法解析现有 YAML，全量重写")
            self._rewrite_yaml()
            return

        # 追加新指标
        existing = data.get("indicators", [])
        for ind in indicators:
            entry: dict[str, Any] = {
                "name": ind.name,
                "aliases": ind.aliases,
                "type": ind.type,
                "table": ind.table,
            }
            if ind.field:
                entry["field"] = ind.field
            if ind.formula:
                entry["formula"] = ind.formula
            if ind.field_mappings:
                entry["field_mappings"] = ind.field_mappings
            existing.append(entry)

        data["indicators"] = existing

        # 保留文件头注释，重新写入
        header_lines: list[str] = []
        for line in content.split("\n"):
            if line.startswith("#") or line.strip() == "":
                header_lines.append(line)
            else:
                break
        header = "\n".join(header_lines)
        if header and not header.endswith("\n"):
            header += "\n"
        header += "\n# [AUTO-IMPORTED] 以下指标由 TableImporter 自动生成\n\n"

        # 重新写入整个文件
        self._yaml_path.write_text(
            header + yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )

    def _rewrite_yaml(self) -> None:
        """全量重写 YAML 文件（删除表时使用）。"""
        data: dict[str, list[dict[str, Any]]] = {"indicators": []}
        for ind in self._indicators.values():
            entry: dict[str, Any] = {
                "name": ind.name,
                "aliases": ind.aliases,
                "type": ind.type,
                "table": ind.table,
            }
            if ind.field:
                entry["field"] = ind.field
            if ind.formula:
                entry["formula"] = ind.formula
            if ind.field_mappings:
                entry["field_mappings"] = ind.field_mappings
            data["indicators"].append(entry)
        self._yaml_path.write_text(
            yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )

    def _rewrite_yaml_without_table(self, table_name: str) -> None:
        """重写 YAML 文件时排除指定表的所有指标。"""
        self._rewrite_yaml()

    # ── 3. 序列化为 LLM Context ──

    def to_llm_context(self) -> str:
        """生成高可读性的注册表描述，约束 LLM 的字段/公式选择。"""
        if not self._indicators:
            return "No registered business indicators."

        lines: list[str] = ["【业务指标 → 物理字段映射字典】:"]
        for ind in self._indicators.values():
            if ind.type == "direct":
                lines.append(
                    f"  - [{ind.name}] (同义词: {ind.aliases}) "
                    f"→ {ind.table}.{ind.field}"
                )
            elif ind.type == "computed":
                mapping_desc = "\n".join(
                    f"      {var} → {m['table']}.{m['column']}"
                    for var, m in ind.field_mappings.items()
                )
                lines.append(
                    f"  - [{ind.name}] (同义词: {ind.aliases}) "
                    f"→ 公式: `{ind.formula}`\n"
                    f"    变量溯源:\n{mapping_desc}"
                )
            else:  # category
                lines.append(
                    f"  - [{ind.name}] (同义词: {ind.aliases}) "
                    f"→ {ind.table}.{ind.field} (分类标签)"
                )
        return "\n".join(lines)


# ── 全局单例（应用启动时调用 load 激活） ──
indicator_registry = IndicatorRegistry()
