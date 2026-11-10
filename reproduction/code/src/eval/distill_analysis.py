"""Analyze teacher->student reasoning compression effects."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from src.data.build_dag import build_dag_from_answer, extract_steps_from_answer


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_text(row: dict[str, Any]) -> str:
    for k in ["prediction", "output", "response", "text"]:
        if k in row and isinstance(row[k], str):
            return row[k]
    if "messages" in row and isinstance(row["messages"], list):
        for m in reversed(row["messages"]):
            if m.get("role") == "assistant":
                return m.get("content", "")
    return ""


def _extract_think(text: str) -> str:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else text


def _answer_correct_proxy(text: str) -> float:
    # Approximate correctness proxy for distillation analysis when GT is absent.
    if "<answer>" not in text:
        return 0.0
    if "结论批改" in text or "conclusion" in text:
        return 1.0
    return 0.5


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    lengths = []
    steps = []
    nodes = []
    edges = []
    depths = []
    acyclic = []
    correct_proxy = []

    for r in rows:
        txt = _extract_text(r)
        if not txt:
            continue
        think = _extract_think(txt)
        dag = build_dag_from_answer(think)
        val = dag.validate_dag()

        lengths.append(float(len(txt)))
        steps.append(float(len(extract_steps_from_answer(think))))
        nodes.append(float(dag.num_nodes))
        edges.append(float(dag.num_edges))
        depths.append(float(dag.get_dependency_depth()))
        acyclic.append(1.0 if val.get("is_acyclic", False) else 0.0)
        correct_proxy.append(_answer_correct_proxy(txt))

    def m(v: list[float]) -> float:
        return float(mean(v)) if v else 0.0

    return {
        "num_samples": float(len(lengths)),
        "avg_chars": m(lengths),
        "avg_steps": m(steps),
        "avg_nodes": m(nodes),
        "avg_edges": m(edges),
        "avg_dep_depth": m(depths),
        "acyclic_rate": m(acyclic),
        "answer_proxy": m(correct_proxy),
    }


def compare(teacher_rows: list[dict[str, Any]], student_rows: list[dict[str, Any]]) -> dict[str, Any]:
    t = _metrics(teacher_rows)
    s = _metrics(student_rows)

    def ratio(a: float, b: float) -> float:
        return (b / a) if a > 0 else 0.0

    return {
        "teacher": t,
        "student": s,
        "compression": {
            "char_ratio_student_over_teacher": ratio(t["avg_chars"], s["avg_chars"]),
            "step_ratio_student_over_teacher": ratio(t["avg_steps"], s["avg_steps"]),
            "node_ratio_student_over_teacher": ratio(t["avg_nodes"], s["avg_nodes"]),
            "depth_ratio_student_over_teacher": ratio(t["avg_dep_depth"], s["avg_dep_depth"]),
            "acyclic_delta": s["acyclic_rate"] - t["acyclic_rate"],
            "answer_proxy_delta": s["answer_proxy"] - t["answer_proxy"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher-student compression analysis")
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/eval/distill_compression.json"))
    args = parser.parse_args()

    teacher_rows = _read_jsonl(args.teacher)
    student_rows = _read_jsonl(args.student)
    report = compare(teacher_rows, student_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved distill analysis to {args.output}")


if __name__ == "__main__":
    main()
