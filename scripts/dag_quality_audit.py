#!/usr/bin/env python3
"""Audit DAG quality on rollout trace pools.

For each ``output/dag_audit/*_traces.jsonl``:
1) Materialize cached DAGs via ``scripts/preprocess_dag_cache.py``
2) Aggregate diagnostics and write ``<bench>_diagnostics.json``
3) Optionally print pass/fail report and write ``report.md``
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import os


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = os.path.expandvars("${PYTHON_ENV_BIN}/python")
DEFAULT_AUDIT_DIR = REPO_ROOT / "output" / "dag_audit"
NON_IMPLICIT_DEP_TYPES = {
    "expr_ref",
    "expr_overlap",
    "claim_ref",
    "var_ref",
    "llm_semantic",
    "llm_subgoal",
}


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _run_preprocess(
    *,
    python_bin: str,
    input_path: Path,
    output_path: Path,
    use_llm: bool,
    llm_model: str,
    llm_device: str,
    llm_max_steps: int,
) -> None:
    cmd = [
        python_bin,
        "scripts/preprocess_dag_cache.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--field",
        "cached_dag",
        "--use-llm",
        "1" if use_llm else "0",
        "--llm-model",
        llm_model,
        "--llm-device",
        llm_device,
        "--max-steps",
        str(llm_max_steps),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _safe_hist(values: list[int]) -> dict[str, int]:
    c = Counter(values)
    return {str(k): int(v) for k, v in sorted(c.items(), key=lambda x: x[0])}


@dataclass
class BenchMetrics:
    benchmark: str
    n: int
    multi_node_rate: float
    valid_dag_rate: float
    q_topo_mean: float
    q_topo_var: float
    non_implicit_block_ratio: float
    passed: bool
    reasons: list[str]


def _node_has_features(n: dict) -> bool:
    if n.get("exprs") or n.get("claims"):
        return True
    if str(n.get("step_type", "")) not in {"", "unknown"}:
        return True
    return False


def _compute_q_topo(nodes: list[dict], edges: list[dict]) -> tuple[float, dict[str, float]]:
    node_ids = {int(n.get("step_id", -1)) for n in nodes if isinstance(n.get("step_id"), int)}
    num_nodes = len(node_ids)
    if num_nodes == 0:
        return 0.0, {"valid": 0.0, "acyclic": 0.0, "rho_orphan": 1.0, "delta": 0.0}

    g = nx.DiGraph()
    g.add_nodes_from(node_ids)
    valid_edge_count = 0
    forward_count = 0
    non_impl_edges = 0
    nodes_with_struct_in: set[int] = set()
    for e in edges:
        try:
            u = int(e.get("source"))
            v = int(e.get("target"))
        except (TypeError, ValueError):
            continue
        if u not in node_ids or v not in node_ids or u == v:
            continue
        g.add_edge(u, v)
        valid_edge_count += 1
        if u < v:
            forward_count += 1
        dep_type = str(e.get("dep_type", ""))
        if dep_type in NON_IMPLICIT_DEP_TYPES:
            non_impl_edges += 1
            nodes_with_struct_in.add(v)

    is_acyclic = 1.0 if nx.is_directed_acyclic_graph(g) else 0.0
    is_valid = 1.0 if (is_acyclic > 0 and num_nodes > 0) else 0.0
    delta = (forward_count / valid_edge_count) if valid_edge_count else 1.0

    step_type = {int(n.get("step_id", -1)): str(n.get("step_type", "")) for n in nodes}
    conclusions = [sid for sid, st in step_type.items() if st == "conclusion"]
    if not conclusions:
        conclusions = sorted(node_ids)
    orphan = 0
    for sid in conclusions:
        if g.in_degree(sid) == 0:
            orphan += 1
    rho_orphan = orphan / max(len(conclusions), 1)
    no_orphan_continuous = 1.0 - rho_orphan

    # Continuous, multi-source diagnostic score so traces with different
    # extractor-detectable carry structure produce different q_topo values.
    non_source_nodes = max(num_nodes - 1, 1)
    structural_support = len(nodes_with_struct_in) / non_source_nodes
    non_impl_ratio_local = (
        non_impl_edges / valid_edge_count if valid_edge_count else 0.0
    )
    edge_density = min(1.0, valid_edge_count / max(num_nodes, 1))
    feature_density = (
        sum(1 for n in nodes if _node_has_features(n)) / max(num_nodes, 1)
    )
    typed_step_types = {
        str(n.get("step_type", ""))
        for n in nodes
        if str(n.get("step_type", "")) not in {"", "unknown"}
    }
    type_diversity = min(1.0, len(typed_step_types) / 4.0)

    lam_a, lam_o, lam_d, lam_s, lam_n, lam_f, lam_t = (
        0.10,
        0.15,
        0.10,
        0.20,
        0.15,
        0.20,
        0.10,
    )
    i_acyclic = 1.0 if is_acyclic > 0 else 0.0
    denom = lam_a + lam_o + lam_d + lam_s + lam_n + lam_f + lam_t
    score = (
        lam_a * i_acyclic
        + lam_o * no_orphan_continuous
        + lam_d * delta * edge_density
        + lam_s * structural_support
        + lam_n * non_impl_ratio_local
        + lam_f * feature_density
        + lam_t * type_diversity
    ) / max(denom, 1e-8)
    if is_valid <= 0:
        score = 0.0
    return float(max(0.0, min(1.0, score))), {
        "valid": is_valid,
        "acyclic": is_acyclic,
        "rho_orphan": rho_orphan,
        "delta": delta,
    }


def _failure_tags(nodes: list[dict], q_topo: float, metrics: dict[str, float], dep_counter: Counter) -> list[str]:
    tags: list[str] = []
    if len(nodes) <= 1:
        tags.append("0_or_1_step")
    empty_features = True
    for n in nodes:
        if n.get("exprs") or n.get("claims"):
            empty_features = False
            break
    if empty_features:
        tags.append("no_expr_no_claim_no_var")
    if q_topo >= 0.99 or q_topo <= 0.01:
        tags.append("q_topo_saturated")
    if metrics.get("acyclic", 0.0) < 1.0:
        tags.append("cycle_detected")
    if dep_counter and dep_counter.get("implicit_block", 0) == sum(dep_counter.values()):
        tags.append("orphan_only")
    return tags


def _audit_cached_file(
    cached_path: Path,
    benchmark: str,
    out_dir: Path,
    *,
    min_multi_node: float,
    min_valid_dag: float,
    min_q_var: float,
    min_non_implicit: float,
) -> BenchMetrics:
    num_steps_hist: list[int] = []
    num_edges_hist: list[int] = []
    q_topo_vals: list[float] = []
    valid_flags: list[float] = []
    multi_node_flags: list[float] = []
    dep_type_counter: Counter = Counter()
    failure_cases: list[dict[str, Any]] = []
    fail_dump: list[dict[str, Any]] = []

    for row in _iter_jsonl(cached_path):
        dag = row.get("cached_dag") or {}
        nodes = dag.get("nodes") or []
        edges = dag.get("edges") or []
        num_steps_hist.append(len(nodes))
        num_edges_hist.append(len(edges))
        dep_counter = Counter(str(e.get("dep_type", "")) for e in edges)
        dep_type_counter.update(dep_counter)

        q_topo, comp = _compute_q_topo(nodes, edges)
        q_topo_vals.append(q_topo)
        valid_flags.append(float(comp["valid"]))
        multi_node_flags.append(1.0 if len(nodes) >= 3 else 0.0)

        tags = _failure_tags(nodes, q_topo, comp, dep_counter)
        if tags:
            case = {
                "benchmark": benchmark,
                "question": row.get("question", ""),
                "gold": row.get("gold", ""),
                "response": row.get("response", ""),
                "tags": tags,
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "q_topo": q_topo,
                "components": comp,
            }
            fail_dump.append(case)
            failure_cases.append(case)

    n = len(q_topo_vals)
    q_mean = statistics.fmean(q_topo_vals) if q_topo_vals else 0.0
    q_var = statistics.pvariance(q_topo_vals) if len(q_topo_vals) > 1 else 0.0
    valid_rate = statistics.fmean(valid_flags) if valid_flags else 0.0
    multi_rate = statistics.fmean(multi_node_flags) if multi_node_flags else 0.0
    total_edges = sum(dep_type_counter.values())
    non_impl = sum(dep_type_counter.get(t, 0) for t in NON_IMPLICIT_DEP_TYPES)
    non_implicit_ratio = (non_impl / total_edges) if total_edges else 0.0

    reasons: list[str] = []
    if multi_rate < min_multi_node:
        reasons.append("multi_node_rate")
    if valid_rate < min_valid_dag:
        reasons.append("valid_dag_rate")
    if q_var < min_q_var:
        reasons.append("q_topo_var")
    if non_implicit_ratio < min_non_implicit:
        reasons.append("non_implicit_block_ratio")
    passed = len(reasons) == 0

    diagnostics = {
        "benchmark": benchmark,
        "n": n,
        "num_steps_hist": _safe_hist(num_steps_hist),
        "num_edges_hist": _safe_hist(num_edges_hist),
        "q_topo_mean": q_mean,
        "q_topo_var": q_var,
        "valid_dag_rate": valid_rate,
        "multi_node_rate": multi_rate,
        "non_implicit_block_ratio": non_implicit_ratio,
        "dep_type_hist": dict(sorted(dep_type_counter.items(), key=lambda kv: kv[0])),
        "top_failure_cases": failure_cases[:10],
        "pass": passed,
        "fail_reasons": reasons,
    }
    out_diag = out_dir / f"{benchmark}_diagnostics.json"
    out_diag.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_fail = out_dir / f"{benchmark}_failures.jsonl"
    with out_fail.open("w", encoding="utf-8") as f:
        for item in fail_dump[:10]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return BenchMetrics(
        benchmark=benchmark,
        n=n,
        multi_node_rate=multi_rate,
        valid_dag_rate=valid_rate,
        q_topo_mean=q_mean,
        q_topo_var=q_var,
        non_implicit_block_ratio=non_implicit_ratio,
        passed=passed,
        reasons=reasons,
    )


def _write_report(rows: list[BenchMetrics], out_path: Path) -> None:
    lines = [
        "# DAG Quality Audit Report",
        "",
        "| benchmark | n | multi_node_rate | valid_dag_rate | q_topo_var | non_implicit_block_ratio | pass | fail_reasons |",
        "|---|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.benchmark} | {r.n} | {r.multi_node_rate:.3f} | {r.valid_dag_rate:.3f} | "
            f"{r.q_topo_var:.3f} | {r.non_implicit_block_ratio:.3f} | "
            f"{'PASS' if r.passed else 'FAIL'} | {', '.join(r.reasons) if r.reasons else '-'} |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="DAG quality diagnostics + threshold report.")
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--benchmarks", nargs="+", default=[])
    parser.add_argument("--label", default="dag_audit_dr1_7b")
    parser.add_argument("--use-llm", action="store_true", default=False)
    parser.add_argument(
        "--llm-model",
        default=os.path.expandvars("${MODEL_ROOT}/Qwen/Qwen2.5-Math-1.5B-Instruct"),
    )
    parser.add_argument("--llm-device", default="auto")
    parser.add_argument("--llm-max-steps", type=int, default=16)
    parser.add_argument("--min-multi-node-rate", type=float, default=0.90)
    parser.add_argument("--min-valid-dag-rate", type=float, default=0.95)
    parser.add_argument("--min-q-topo-var", type=float, default=0.05)
    parser.add_argument("--min-non-implicit-ratio", type=float, default=0.50)
    parser.add_argument("--report", action="store_true", default=False)
    args = parser.parse_args()

    if args.benchmarks:
        benches = args.benchmarks
    else:
        benches = []
        for p in sorted(args.trace_dir.glob(f"{args.label}_*_traces.jsonl")):
            suffix = p.name.replace(f"{args.label}_", "")
            benches.append(suffix.replace("_traces.jsonl", ""))
    if not benches:
        raise SystemExit("No trace pools found; run scripts/build_trace_pool.py first.")

    rows: list[BenchMetrics] = []
    for bench in benches:
        trace_path = args.trace_dir / f"{args.label}_{bench}_traces.jsonl"
        if not trace_path.is_file():
            print(f"[skip] missing trace file: {trace_path}")
            continue
        cached_path = args.trace_dir / f"{args.label}_{bench}_cached.jsonl"
        _run_preprocess(
            python_bin=args.python_bin,
            input_path=trace_path,
            output_path=cached_path,
            use_llm=args.use_llm,
            llm_model=args.llm_model,
            llm_device=args.llm_device,
            llm_max_steps=args.llm_max_steps,
        )
        row = _audit_cached_file(
            cached_path,
            benchmark=bench,
            out_dir=args.trace_dir,
            min_multi_node=args.min_multi_node_rate,
            min_valid_dag=args.min_valid_dag_rate,
            min_q_var=args.min_q_topo_var,
            min_non_implicit=args.min_non_implicit_ratio,
        )
        rows.append(row)
        print(
            f"[{bench}] n={row.n} multi={row.multi_node_rate:.3f} valid={row.valid_dag_rate:.3f} "
            f"q_var={row.q_topo_var:.3f} non_impl={row.non_implicit_block_ratio:.3f} "
            f"{'PASS' if row.passed else 'FAIL'}"
        )

    if args.report:
        report_path = args.trace_dir / "report.md"
        _write_report(rows, report_path)
        print(f"[report] {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
