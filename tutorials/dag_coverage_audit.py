"""Coverage audit for the DAG extractor.

Why this script exists
----------------------
The GRPO training reward (``src/reward/topo_reward.py``) consumes DAGs
extracted from free-form reasoning text by ``src/data/build_dag.py``. If the
extractor under-recovers dependency edges, q_topo / q_cont will collapse and
the reward signal degrades to a constant, which kills the RL signal.

This script takes a stratified sample of (a) ground-truth reasoning chains
from the GRPO training data (``reference_dag``), and (b) the rollouts in
``data/srt_raw/rollouts_minimal.jsonl``, extracts a ``ReasoningDAG`` for each,
and reports percentile statistics of:

    nodes, edges, edge-type mix, orphan(node), orphan(conclusion),
    dependency_depth, acyclic-rate, q_topo_structural, q_cont_structural

Usage::

    python3 tutorials/dag_coverage_audit.py                # default 200 samples
    python3 tutorials/dag_coverage_audit.py --n 400        # more samples
    python3 tutorials/dag_coverage_audit.py --n 200 --out report.json

The script runs purely on the reasoning text already shipped in the repo;
no GPU required. It is safe to rerun after modifying the extractor.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import networkx as nx  # noqa: E402

from src.dag.graph import (  # noqa: E402
    DOUBLE_BARRIER_EDGE,
    SOLID_EDGE,
    VIRTUAL_EDGE,
    ReasoningDAG,
)
from src.dag.node import StepType  # noqa: E402
from src.data.build_dag import parse_answer_to_dag_debug  # noqa: E402


DEFAULT_TRAIN_JSONL = _REPO_ROOT / "data" / "grpo_ready" / "train_public.jsonl"
DEFAULT_ROLLOUT_JSONL = _REPO_ROOT / "data" / "srt_raw" / "rollouts_minimal.jsonl"


def _percentiles(xs: List[float], qs: Sequence[float]) -> Dict[str, float]:
    if not xs:
        return {f"p{int(q*100)}": 0.0 for q in qs}
    xs_sorted = sorted(xs)
    out: Dict[str, float] = {}
    for q in qs:
        k = max(0, min(len(xs_sorted) - 1, int(round(q * (len(xs_sorted) - 1)))))
        out[f"p{int(q * 100)}"] = float(xs_sorted[k])
    return out


def _summary(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"n": 0, "mean": 0.0, "median": 0.0, **_percentiles([], (0.1, 0.5, 0.9))}
    return {
        "n": len(xs),
        "mean": round(float(sum(xs) / len(xs)), 3),
        "median": round(float(statistics.median(xs)), 3),
        **{k: round(v, 3) for k, v in _percentiles(xs, (0.1, 0.5, 0.9)).items()},
    }


def _conclusion_orphan_rate(dag: ReasoningDAG) -> float:
    g = dag.graph
    conclusion_ids = [
        nid
        for nid, data in g.nodes(data=True)
        if data.get("step_type") == StepType.CONCLUSION
    ]
    if not conclusion_ids:
        return 0.0
    n_orphan = 0
    for cid in conclusion_ids:
        has_virtual = any(
            dag.is_virtual_edge(g.edges[u, cid].get("edge_type", ""))
            for u in g.predecessors(cid)
        )
        if not has_virtual:
            n_orphan += 1
    return n_orphan / len(conclusion_ids)


def _direction_consistency(dag: ReasoningDAG) -> float:
    g = dag.graph
    n_edges = g.number_of_edges()
    if n_edges == 0:
        return 1.0
    forward = sum(1 for u, v in g.edges() if u < v)
    return forward / n_edges


def _structural_q(dag: ReasoningDAG) -> Tuple[float, float]:
    """Structural approximations matching the production reward terms.

    q_topo here uses *node-level* orphan rate (which is what the extractor's
    virtual-edge recall actually controls). The production reward uses a
    conclusion-level orphan rate instead — both are reported separately in
    the coverage report so the extractor can be tuned without regressing the
    downstream reward formula.
    """
    g = dag.graph
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    if n_nodes == 0:
        return 0.0, 0.0
    acyclic = 1.0 if nx.is_directed_acyclic_graph(g) else 0.0
    node_orphan_rate = len(dag.orphan_nodes()) / float(n_nodes)
    no_orphan = 1.0 - node_orphan_rate
    direction = _direction_consistency(dag)
    try:
        layered = sum(len(layer) for layer in nx.topological_generations(g))
    except nx.NetworkXUnfeasible:
        layered = 0
    step_align = layered / float(n_nodes)
    q_topo = 0.3 * acyclic + 0.3 * no_orphan + 0.2 * direction + 0.2 * step_align
    adj = sum(
        1
        for u in range(n_nodes - 1)
        if g.has_edge(u, u + 1) or g.has_edge(u + 1, u)
    )
    q_cont = adj / float(max(n_nodes - 1, 1))
    return float(max(0.0, min(1.0, q_topo))), float(max(0.0, min(1.0, q_cont)))


def _dag_from_reference(raw: Any) -> Optional[ReasoningDAG]:
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raw = json.loads(raw)
    try:
        return ReasoningDAG.from_dict(raw)
    except Exception:
        return None


def _edge_mix(dag: ReasoningDAG) -> Dict[str, int]:
    mix = {SOLID_EDGE: 0, VIRTUAL_EDGE: 0, DOUBLE_BARRIER_EDGE: 0, "other": 0}
    for _, _, data in dag.graph.edges(data=True):
        et = data.get("edge_type", "")
        if dag.is_virtual_edge(et):
            mix[VIRTUAL_EDGE] += 1
        elif dag.is_barrier_edge(et):
            mix[DOUBLE_BARRIER_EDGE] += 1
        elif et in {SOLID_EDGE, "sequential"}:
            mix[SOLID_EDGE] += 1
        else:
            mix["other"] += 1
    return mix


def _metrics_for_dag(dag: ReasoningDAG) -> Dict[str, float]:
    q_topo, q_cont = _structural_q(dag)
    mix = _edge_mix(dag)
    return {
        "nodes": dag.graph.number_of_nodes(),
        "edges": dag.graph.number_of_edges(),
        "mix_solid": mix[SOLID_EDGE],
        "mix_virtual": mix[VIRTUAL_EDGE],
        "mix_barrier": mix[DOUBLE_BARRIER_EDGE],
        "acyclic": 1.0 if nx.is_directed_acyclic_graph(dag.graph) else 0.0,
        "orphan_node_rate": len(dag.orphan_nodes()) / max(dag.graph.number_of_nodes(), 1),
        "orphan_conclusion_rate": _conclusion_orphan_rate(dag),
        "q_topo": q_topo,
        "q_cont": q_cont,
    }


def iter_reference_samples(
    jsonl_path: Path, n: int, seed: int = 0
) -> List[Tuple[str, str, Dict[str, float]]]:
    """Two passes per row: (a) the cached reference_dag blob, and (b) a fresh
    re-extraction from ``standard_answer`` (or ``solution``/``answer``) so we
    can measure the *current* extractor output, not historical snapshots.
    """
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rng.shuffle(rows)
    out: List[Tuple[str, str, Dict[str, float]]] = []
    bucket_cached: Dict[str, int] = {}
    bucket_reparsed: Dict[str, int] = {}
    per_bucket = max(1, n // 2)
    for row in rows:
        if len(out) >= 2 * n:
            break
        src = row.get("source", "unknown")

        # (a) cached reference_dag, baseline for extractor regression check.
        if bucket_cached.get(src, 0) < per_bucket:
            dag = _dag_from_reference(row.get("reference_dag"))
            if dag is not None:
                out.append((src, "reference", _metrics_for_dag(dag)))
                bucket_cached[src] = bucket_cached.get(src, 0) + 1

        # (b) fresh parse through the current extractor, using the canonical
        # reasoning text bundled with this row.
        if bucket_reparsed.get(src, 0) < per_bucket:
            text = (
                row.get("standard_answer")
                or row.get("solution")
                or row.get("answer")
                or ""
            )
            if text:
                try:
                    dag_re, _ = parse_answer_to_dag_debug(
                        text, problem_id=row.get("record_id", src)
                    )
                except Exception:
                    dag_re = None
                if dag_re is not None and dag_re.graph.number_of_nodes() > 0:
                    out.append((src, "extractor", _metrics_for_dag(dag_re)))
                    bucket_reparsed[src] = bucket_reparsed.get(src, 0) + 1
    return out


def iter_rollout_samples(
    jsonl_path: Path, n: int
) -> List[Tuple[str, str, Dict[str, float]]]:
    if not jsonl_path.exists():
        return []
    out: List[Tuple[str, str, Dict[str, float]]] = []
    for line in jsonl_path.open("r", encoding="utf-8"):
        row = json.loads(line)
        for key in ("y_init", "y_revised", "response", "completion", "pred_raw"):
            text = row.get(key)
            if not text:
                continue
            try:
                dag, _ = parse_answer_to_dag_debug(text, problem_id=row.get("problem_id", "r"))
            except Exception:
                continue
            if dag.graph.number_of_nodes() == 0:
                continue
            m = _metrics_for_dag(dag)
            out.append((row.get("source", "rollout"), f"rollout:{key}", m))
            if len(out) >= n:
                return out
    return out


def aggregate(samples: List[Tuple[str, str, Dict[str, float]]]) -> Dict[str, Any]:
    by_group: Dict[str, List[Dict[str, float]]] = {}
    for source, scheme, m in samples:
        key = f"{scheme}/{source}"
        by_group.setdefault(key, []).append(m)
    report: Dict[str, Any] = {"groups": {}, "overall": {}}
    fields = [
        "nodes",
        "edges",
        "mix_solid",
        "mix_virtual",
        "mix_barrier",
        "acyclic",
        "orphan_node_rate",
        "orphan_conclusion_rate",
        "q_topo",
        "q_cont",
    ]
    for group, ms in by_group.items():
        report["groups"][group] = {
            field: _summary([float(m[field]) for m in ms]) for field in fields
        }
    all_ms = [m for _, _, m in samples]
    report["overall"] = {
        field: _summary([float(m[field]) for m in all_ms]) for field in fields
    }
    return report


def print_report(report: Dict[str, Any]) -> None:
    print("=" * 72)
    print("DAG extractor coverage report")
    print("=" * 72)

    def _row(label: str, stats: Dict[str, Any]) -> None:
        fmt = (
            "  {label:<26} n={n:>4}  mean={mean:>7}  "
            "p10={p10:>7}  p50={p50:>7}  p90={p90:>7}"
        )
        if stats["n"] == 0:
            print(f"  {label:<26} n=   0   (no data)")
            return
        print(fmt.format(label=label, **stats))

    for group, gstats in report["groups"].items():
        print(f"\n[{group}]")
        for field in gstats:
            _row(field, gstats[field])

    print("\n[overall]")
    for field, stats in report["overall"].items():
        _row(field, stats)

    # Quick health judgement
    overall = report["overall"]
    q_topo_mean = overall["q_topo"]["mean"]
    q_cont_mean = overall["q_cont"]["mean"]
    mix_virtual = overall["mix_virtual"]["mean"]
    print("\nHealth hints (lower is worse):")
    print(f"  mean q_topo  = {q_topo_mean:>6}  target > 0.55")
    print(f"  mean q_cont  = {q_cont_mean:>6}  target > 0.40")
    print(f"  mean virtual = {mix_virtual:>6}  target > 1.0 (edges/sample)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-jsonl", default=str(DEFAULT_TRAIN_JSONL))
    parser.add_argument("--rollout-jsonl", default=str(DEFAULT_ROLLOUT_JSONL))
    parser.add_argument(
        "--out",
        default=str(_REPO_ROOT / "topoprm_paper" / "figures" / "dag_coverage_report.json"),
    )
    args = parser.parse_args()

    samples: List[Tuple[str, str, Dict[str, float]]] = []
    samples.extend(iter_reference_samples(Path(args.train_jsonl), args.n, seed=args.seed))
    samples.extend(iter_rollout_samples(Path(args.rollout_jsonl), args.n))

    report = aggregate(samples)
    print_report(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nFull JSON written to {out_path}")


if __name__ == "__main__":
    main()
