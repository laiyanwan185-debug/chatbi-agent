"""将 L4/L5 markdown 题目转换为 golden_answers YAML 格式。"""
import re
import yaml
from pathlib import Path

def parse_l4l5_md(filepath: str, level: int) -> list[dict]:
    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    questions = []
    # 匹配 Qx-Lx-XX 块
    pattern = r"### Q\d-L\d-(\d+)\s*\n-\s*\*\*问题\*\*：(.+?)\n-\s*\*\*涉及表\*\*：(.+?)\n-\s*\*\*分析类别\*\*：(.+?)(?:\n-\s*\*\*所需算法\*\*：(.+?))?(?:\n-\s*\*\*考察点\*\*：(.+?))?(?=\n###|\Z)"

    for m in re.finditer(pattern, text, re.DOTALL):
        qid = m.group(1)
        question = m.group(2).strip()
        tables_str = m.group(3).strip()
        analysis_type = m.group(4).strip()
        algorithm = m.group(5).strip() if m.group(5) else ""
        keywords = m.group(6).strip() if m.group(6) else ""

        # 解析表名：去掉 ``` 后按 、 或 , 分割
        clean = tables_str.replace("`", "").replace(" ", "")
        tables = [t.strip() for t in re.split(r"[、,，]", clean) if t.strip()]

        # 映射分析类别 -> analysis_type（含括号说明）
        type_map = {
            "跨领域综合分析": "cross_domain",
            "时间序列与趋势分析": "trend",
            "排名与比较": "rank",
            "多维度深度分析": "multi_dim",
            "政策评估与关联分析": "correlation",
            "区域协同与空间分析": "spatial",
            "异常检测与专项分析": "anomaly",
        }
        # 取括号前的主类别
        main_type = analysis_type.split("（")[0].split("(")[0].strip()
        atype = type_map.get(main_type, "detail")

        q = {
            "id": f"L{level}-{qid}",
            "level": level,
            "question": question,
            "expected_analysis_type": atype,
            "expected_indicators": [],
            "expected_tables": tables,
            "expected_sql": "",
            "expected_ranges": {},
            "expected_keywords": [],
        }
        questions.append(q)

    return questions


def main():
    base = Path(__file__).resolve().parent  # tests/
    root = base.parent.resolve()
    l4 = parse_l4l5_md(str(root / "20_BI_questions_L4.md"), 4)
    l5 = parse_l4l5_md(str(root / "20_BI_questions_L5.md"), 5)

    all_q = l4 + l5
    data = {
        "questions": all_q,
    }

    out_path = base / "golden_answers_L4L5.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"Generated {len(all_q)} questions → {out_path}")
    for lvl in [4, 5]:
        n = sum(1 for q in all_q if q["level"] == lvl)
        print(f"  L{lvl}: {n} questions")


if __name__ == "__main__":
    main()
