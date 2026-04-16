"""Compute structural DAG metrics from critique <answer> JSON in eval jsonl files.

The private benchmark often leaves <redacted_thinking> empty; we approximate
reasoning structure from the structured grading JSON: sub-questions and step
blocks form a linear chain per sub-question, linked across sub-questions.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from src.dag.compress import compress_dag
from src.dag.graph import ReasoningDAG
from src.dag.node import Node, StepType

# JSON keys emitted by the critique prompt (Chinese field names).
_K_BLOCKS = "?????????"
_K_SUB_BLOCKS = "???????????"
_K_CAUSE = "??"


def _extract_answer_json(text: str) -> dict[str, Any] | None:
    m = re.search(r"<answer>\s*(.*)", text, re.DOTALL)
    if not m:
        return None
    payload = m.group(1).strip()
    if "</answer>" in payload:
        payload = payload.split("</answer>", 1)[0].strip()
    try:
        out = json.loads(payload)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", payload)
        try:
            out = json.loads(cleaned)
            return out if isinstance(out, dict) else None
        except json.JSONDecodeError:
            pass
    # Truncated generations: trim from the end until JSON parses.
    for end in range(len(payload), max(0, len(payload) - 8000), -1):
        chunk = payload[:end].rstrip()
        if not chunk.endswith("}"):
            continue
        try:
            out = json.loads(chunk)
            if isinstance(out, dict) and _K_BLOCKS in out:
                return out
        except json.JSONDecodeError:
            continue
    return None


def _dag_from_critique_answer(data: dict[str, Any], problem_id: str) -> ReasoningDAG:
    dag = ReasoningDAG(problem_id=problem_id)
    items = data.get(_K_BLOCKS) or []
    node_id = 0
    prev_global: int | None = None
    last_added: int | None = None

    for sq in items:
        blocks = (sq.get(_K_SUB_BLOCKS) or []) if isinstance(sq, dict) else []
        prev_in_sq: int | None = None
        for blk in blocks:
            if not isinstance(blk, dict):
                continue
            sid = node_id
            node_id += 1
            dag.add_node(
                Node(
                    step_id=sid,
                    step_type=StepType.DERIVATION,
                    raw_text=str(blk.get(_K_CAUSE, "")),
                    normalized_text="",
                )
            )
            if prev_in_sq is not None:
                dag.add_dependency_edge(prev_in_sq, sid, "virtual")
            elif prev_global is not None:
                dag.add_dependency_edge(prev_global, sid, "virtual")
            prev_in_sq = sid
            prev_global = sid
            last_added = sid

    if last_added is not None:
        n = dag.nodes[last_added]
        dag._nodes[last_added] = Node(
            step_id=n.step_id,
            raw_text=n.raw_text,
            normalized_text=n.normalized_text,
            exprs=n.exprs,
            claims=n.claims,
            step_type=StepType.CONCLUSION,
            local_verdict=n.local_verdict,
            sub_question_id=n.sub_question_id,
        )

    if not dag.nodes:
        dag.add_node(
            Node(
                step_id=0,
                step_type=StepType.CONCLUSION,
                raw_text="empty",
                normalized_text="",
            )
        )
    return dag


def _dependency_edges(dag: ReasoningDAG) -> set[tuple[int, int]]:
    return {
        (int(e.source), int(e.target))
        for e in dag.edges
        if dag.is_virtual_edge(e.edge_type)
    }


def _orphan_conclusion_ratio(dag: ReasoningDAG) -> float:
    conclusion_nodes = [
        sid
        for sid, n in dag.nodes.items()
        if getattr(n.step_type, "value", str(n.step_type)) == "conclusion"
    ]
    if not conclusion_nodes:
        return 0.0
    orphan = 0
    for sid in conclusion_nodes:
        in_dep = 0
        for _u, _v, data in dag.graph.in_edges(sid, data=True):
            if dag.is_virtual_edge(data.get("edge_type", "")):
                in_dep += 1
        if in_dep == 0:
            # Single-node traces are JSON-only answers without an explicit chain.
            if dag.num_nodes <= 1:
                continue
            orphan += 1
    return orphan / len(conclusion_nodes)


def evaluate_jsonl(path: Path, limit: int | None = None) -> dict[str, Any]:
    acyclic: list[float] = []
    orphan_ratio: list[float] = []
    direction: list[float] = []
    depth: list[float] = []
    dep_edge_keep: list[float] = []
    n_ok = 0

    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and n_ok >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(rec.get("response") or rec.get("prediction") or rec.get("output") or "")
            aj = _extract_answer_json(text)
            if not aj:
                continue
            dag = _dag_from_critique_answer(aj, problem_id=f"line_{i}")
            val = dag.validate_dag()
            acyclic.append(1.0 if val["is_acyclic"] else 0.0)
            orphan_ratio.append(_orphan_conclusion_ratio(dag))
            direction.append(dag.direction_consistency())
            d = float(dag.get_dependency_depth())
            if dag.num_nodes >= 1:
                d = max(d, 1.0)
            depth.append(d)
            compressed = compress_dag(dag)
            old_dep = _dependency_edges(dag)
            new_dep = _dependency_edges(compressed)
            keep_r = len(old_dep & new_dep) / len(old_dep) if old_dep else 1.0
            dep_edge_keep.append(keep_r)
            n_ok += 1

    def m(xs: list[float]) -> float:
        return float(mean(xs)) if xs else 0.0

    return {
        "source": str(path),
        "num_used": n_ok,
        "acyclic_pct": m(acyclic) * 100.0,
        "no_orphan_pct": (1.0 - m(orphan_ratio)) * 100.0,
        "dir_cons": m(direction),
        "dag_depth": m(depth),
        "edge_keep_pct": m(dep_edge_keep) * 100.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Structural metrics from critique jsonl")
    p.add_argument("jsonl", type=Path)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    out = evaluate_jsonl(args.jsonl, limit=args.limit)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
