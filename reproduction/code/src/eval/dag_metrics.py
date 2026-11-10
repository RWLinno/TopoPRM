"""DAG explainability metrics and compression analysis for TopoPRM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from src.dag.compress import compress_dag
from src.dag.graph import ReasoningDAG


def _safe_mean(vals: list[float]) -> float:
    return float(mean(vals)) if vals else 0.0


def _dependency_edges(dag: ReasoningDAG) -> set[tuple[int, int]]:
    return {
        (e.source, e.target)
        for e in dag.edges
        if dag.is_virtual_edge(e.edge_type)
    }


def _orphan_conclusion_ratio(dag: ReasoningDAG) -> float:
    conclusion_nodes = [
        sid for sid, n in dag.nodes.items() if getattr(n.step_type, "value", str(n.step_type)) == "conclusion"
    ]
    if not conclusion_nodes:
        return 0.0

    orphan = 0
    for sid in conclusion_nodes:
        in_dep = 0
        for u, v, data in dag.graph.in_edges(sid, data=True):
            if dag.is_virtual_edge(data.get("edge_type", "")):
                in_dep += 1
        if in_dep == 0:
            orphan += 1
    return orphan / len(conclusion_nodes)


def evaluate_dag_dir(dag_dir: Path) -> dict[str, Any]:
    files = sorted(dag_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No DAG json files found in {dag_dir}")

    acyclic = []
    connected = []
    orphan_ratio = []
    direction = []
    depth = []
    num_nodes = []
    num_edges = []

    comp_node_ratio = []
    comp_edge_ratio = []
    comp_depth_ratio = []
    dep_edge_keep_ratio = []

    per_problem: dict[str, Any] = {}

    for fp in files:
        dag = ReasoningDAG.from_json(fp.read_text(encoding="utf-8"))
        val = dag.validate_dag()

        acyclic.append(1.0 if val["is_acyclic"] else 0.0)
        connected.append(1.0 if val["is_connected"] else 0.0)
        orphan_ratio.append(_orphan_conclusion_ratio(dag))
        direction.append(dag.direction_consistency())
        depth.append(float(dag.get_dependency_depth()))
        num_nodes.append(float(dag.num_nodes))
        num_edges.append(float(dag.num_edges))

        compressed = compress_dag(dag)
        old_dep = _dependency_edges(dag)
        new_dep = _dependency_edges(compressed)

        node_r = compressed.num_nodes / dag.num_nodes if dag.num_nodes else 1.0
        edge_r = compressed.num_edges / dag.num_edges if dag.num_edges else 1.0
        old_d = dag.get_dependency_depth()
        new_d = compressed.get_dependency_depth()
        depth_r = new_d / old_d if old_d > 0 else 1.0
        keep_r = len(old_dep & new_dep) / len(old_dep) if old_dep else 1.0

        comp_node_ratio.append(node_r)
        comp_edge_ratio.append(edge_r)
        comp_depth_ratio.append(depth_r)
        dep_edge_keep_ratio.append(keep_r)

        pred_dep_edges = sorted([list(e) for e in _dependency_edges(dag)])

        per_problem[dag.problem_id] = {
            "num_nodes": dag.num_nodes,
            "num_edges": dag.num_edges,
            "acyclic": val["is_acyclic"],
            "connected": val["is_connected"],
            "orphan_conclusion_ratio": _orphan_conclusion_ratio(dag),
            "direction_consistency": dag.direction_consistency(),
            "dependency_depth": dag.get_dependency_depth(),
            "pred_virtual_edges": pred_dep_edges,
            "compressed": {
                "num_nodes": compressed.num_nodes,
                "num_edges": compressed.num_edges,
                "dependency_depth": compressed.get_dependency_depth(),
                "node_ratio": node_r,
                "edge_ratio": edge_r,
                "depth_ratio": depth_r,
                "dependency_edge_keep_ratio": keep_r,
            },
        }

    return {
        "num_samples": len(files),
        "dag_quality": {
            "acyclic_rate": _safe_mean(acyclic),
            "connected_rate": _safe_mean(connected),
            "orphan_conclusion_ratio": _safe_mean(orphan_ratio),
            "direction_consistency": _safe_mean(direction),
            "dependency_depth": _safe_mean(depth),
            "num_nodes": _safe_mean(num_nodes),
            "num_edges": _safe_mean(num_edges),
        },
        "compression": {
            "node_ratio": _safe_mean(comp_node_ratio),
            "edge_ratio": _safe_mean(comp_edge_ratio),
            "depth_ratio": _safe_mean(comp_depth_ratio),
            "dependency_edge_keep_ratio": _safe_mean(dep_edge_keep_ratio),
        },
        "per_problem": per_problem,
    }


def evaluate_annotation_alignment(metrics: dict[str, Any], annotation_path: Path) -> dict[str, Any]:
    """Evaluate extraction quality against manual edge annotations.

    Annotation format:
    {
      "problem_id": {
        "gold_virtual_edges": [[0,2], [2,3]]
      }
    }
    """
    ann = json.loads(annotation_path.read_text(encoding="utf-8"))
    per_problem = metrics["per_problem"]

    precision_vals: list[float] = []
    recall_vals: list[float] = []
    f1_vals: list[float] = []

    for pid, v in ann.items():
        if pid not in per_problem:
            continue
        pred = {
            tuple(edge)
            for edge in per_problem[pid].get("pred_virtual_edges", per_problem[pid].get("pred_dependency_edges", []))
        }
        if not pred:
            # Recover from original stats when explicit field is absent.
            # Users can optionally enrich per_problem with this field.
            continue

        gold = {tuple(x) for x in v.get("gold_virtual_edges", v.get("gold_dependency_edges", []))}
        inter = pred & gold
        p = len(inter) / len(pred) if pred else 0.0
        r = len(inter) / len(gold) if gold else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precision_vals.append(p)
        recall_vals.append(r)
        f1_vals.append(f1)

    return {
        "num_annotated_used": len(precision_vals),
        "edge_precision": _safe_mean(precision_vals),
        "edge_recall": _safe_mean(recall_vals),
        "edge_f1": _safe_mean(f1_vals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute DAG quality and compression metrics")
    parser.add_argument("--dag_dir", type=Path, default=Path("data/dag"))
    parser.add_argument("--output", type=Path, default=Path("output/eval/dag_metrics.json"))
    parser.add_argument("--annotation", type=Path, default=None, help="Optional manual annotation JSON path")
    args = parser.parse_args()

    metrics = evaluate_dag_dir(args.dag_dir)
    if args.annotation and args.annotation.exists():
        metrics["annotation_alignment"] = evaluate_annotation_alignment(metrics, args.annotation)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved DAG metrics to {args.output}")


if __name__ == "__main__":
    main()
