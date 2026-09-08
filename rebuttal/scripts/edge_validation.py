#!/usr/bin/env python3
"""LLM-based edge validation for the TopoPRM DAG extractor.

Independent-judge protocol (answers HxUk W1, B5w7 W1, TsKG W1):

  1. Sample N traces from data/grpo_ready/train_public.jsonl, stratified by
     source (gsm8k / math) and trace length (num steps).
  2. For each trace, re-run the extractor (src.data.build_dag) to obtain the
     candidate support edges and the segmented steps.
  3. Ask a strong LLM, *blind to the extractor edges*, to judge for every
     ordered step pair (i<j) whether step i provides necessary support for
     step j. The union of LLM "yes" pairs is the reference edge set A*.
  4. Compare extractor edges A_E against A* -> precision / recall / F1 /
     per-dep-type reliability, plus a false-positive taxonomy.

Two phases so the expensive LLM calls are decoupled from sampling:

    python edge_validation.py sample   --n 120 --out <pack.jsonl>
    python edge_validation.py annotate --pack <pack.jsonl> --out <ann.jsonl>
    python edge_validation.py score    --pack <pack.jsonl> --ann <ann.jsonl> --out <results.json>

The annotate phase is model-agnostic (any OpenAI-compatible endpoint via
--base-url / --model, key from OPENAI_API_KEY or --api-key).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.build_dag import (  # noqa: E402
    build_dag_from_answer,
    extract_steps_from_answer,
    segmentation_issue_reasons,
)
from src.dag.graph import analyze_topology_projection  # noqa: E402

DATA = REPO / "data" / "grpo_ready" / "train_public.jsonl"


def _steps_and_edges(answer: str) -> tuple[list[str], list[dict[str, Any]]]:
    steps = extract_steps_from_answer(answer)
    texts = [s.get("normalized_text") or s.get("raw_text", "") for s in steps]
    dag = build_dag_from_answer(answer)
    raw_edges = dag.graph.graph.get("raw_dependency_edges")
    if isinstance(raw_edges, list):
        edges = [
            {
                "source": int(edge["source"]),
                "target": int(edge["target"]),
                "edge_type": edge["edge_type"],
                "dep_type": edge.get("dep_type"),
            }
            for edge in raw_edges
        ]
    else:
        edges = [
            {
                "source": edge.source,
                "target": edge.target,
                "edge_type": edge.edge_type,
                "dep_type": getattr(edge, "dep_type", None),
            }
            for edge in dag.edges
        ]
    return texts, edges


def _step_texts(answer: str) -> list[str]:
    return [
        step.get("normalized_text") or step.get("raw_text", "")
        for step in extract_steps_from_answer(answer)
    ]


def _segmentation_issues(steps: list[str]) -> list[str]:
    return segmentation_issue_reasons(steps)


def cmd_sample(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in DATA.open() if l.strip()]
    # bucket by (source, length-band)
    buckets: dict[tuple, list] = defaultdict(list)
    prepared = []
    rejected = Counter()
    for r in rows:
        texts = _step_texts(r["standard_answer"])
        n = len(texts)
        if n < 2 or n > 12:  # need >=2 steps to have any edge; cap for annotation cost
            continue
        issues = _segmentation_issues(texts)
        if args.reject_malformed and issues:
            rejected.update(issues)
            continue
        band = "short" if n <= 3 else ("mid" if n <= 6 else "long")
        rec = {
            "record_id": r["record_id"],
            "source": r["source"],
            "question": r["question"],
            "final_answer": r["final_answer"],
            "steps": texts,
            "n_steps": n,
            "band": band,
            "standard_answer": r["standard_answer"],
        }
        buckets[(r["source"], band)].append(rec)
    # even allocation across buckets
    keys = sorted(buckets)
    per = max(1, args.n // len(keys))
    for k in keys:
        rng.shuffle(buckets[k])
        prepared.extend(buckets[k][:per])
    rng.shuffle(prepared)
    prepared = prepared[: args.n]
    for rec in prepared:
        _, rec["extractor_edges"] = _steps_and_edges(rec.pop("standard_answer"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rec in prepared:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    dist = Counter((r["source"], r["band"]) for r in prepared)
    n_edges = sum(len(r["extractor_edges"]) for r in prepared)
    print(f"[sample] wrote {len(prepared)} traces, {n_edges} extractor edges -> {out}")
    print(f"[sample] strata: {dict(dist)}")
    print(f"[sample] malformed candidates rejected: {dict(rejected)}")


def _order_control(num_steps: int, index: int) -> tuple[str, list[int]]:
    """Match the human annotation platform's deterministic order condition."""
    return _order_controls(num_steps)[index % 3]


def _order_controls(num_steps: int) -> list[tuple[str, list[int]]]:
    original = list(range(num_steps))
    return [
        ("original", original),
        ("reversed", list(reversed(original))),
        ("interleaved", original[::2] + original[1::2]),
    ]


def cmd_predict_encoder(args: argparse.Namespace) -> None:
    """Run the frozen pair encoder under the annotation order controls."""
    os.environ["TOPO_DAG_EDGE_CHECKPOINT"] = str(Path(args.checkpoint).resolve())
    os.environ["TOPO_DAG_EDGE_MODEL"] = str(Path(args.model).resolve())
    os.environ["TOPO_DAG_EDGE_DEVICE"] = args.device
    os.environ["TOPO_DAG_EDGE_BATCH_SIZE"] = str(args.batch_size)
    os.environ["TOPO_DAG_EDGE_MAX_STEPS"] = str(args.max_steps)
    os.environ["TOPO_DAG_EDGE_REQUIRED"] = "1"

    from src.data.build_dag import predict_semantic_dependency_edges_batch

    records = [json.loads(line) for line in Path(args.pack).open() if line.strip()]
    step_batches = []
    controls = []
    for index, record in enumerate(records):
        steps = list(record["steps"])
        record_controls = (
            _order_controls(len(steps))
            if args.all_order_controls
            else [_order_control(len(steps), index)]
        )
        for variant, order in record_controls:
            controls.append((record, variant, order))
            step_batches.append([
                {"raw_text": steps[source_index], "sub_question_id": None}
                for source_index in order
            ])

    predictions = predict_semantic_dependency_edges_batch(step_batches)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    retained = candidate_pairs = 0
    with output.open("w", encoding="utf-8") as handle:
        for predicted, (record, variant, order) in zip(predictions, controls):
            edges = []
            display_edges = []
            for edge in predicted:
                display_source = int(edge["source"])
                display_target = int(edge["target"])
                shared = {
                    "dep_type": edge.get("dep_type", "encoder_semantic"),
                    "confidence": float(edge.get("confidence", 0.0)),
                    "direction_margin": float(edge.get("direction_margin", 0.0)),
                }
                display_edges.append({
                    "source": display_source,
                    "target": display_target,
                    **shared,
                })
                edges.append({
                    "source": order[display_source],
                    "target": order[display_target],
                    **shared,
                })
            total_pairs = len(order) * (len(order) - 1) // 2
            retained += len(edges)
            candidate_pairs += total_pairs
            source_record_id = str(record["record_id"])
            row = {
                "record_id": (
                    f"{source_record_id}::{variant}"
                    if args.all_order_controls
                    else source_record_id
                ),
                "source_record_id": source_record_id,
                "order_variant": variant,
                "step_order": order,
                "edges": edges,
                "display_edges": display_edges,
                "evaluated_pairs": len(edges),
                "abstained_pairs": total_pairs - len(edges),
                "candidate_unordered_pairs": total_pairs,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    coverage = retained / candidate_pairs if candidate_pairs else 0.0
    print(
        f"[predict-encoder] records={len(records)} retained={retained}/"
        f"{candidate_pairs} coverage={coverage:.4f} -> {output}"
    )


def _topology_metrics(num_steps: int, edges: list[dict[str, Any]]) -> dict[str, float]:
    weighted = [
        (
            int(edge["source"]),
            int(edge["target"]),
            max(0.0, float(edge.get("confidence", edge.get("weight", 1.0)))),
        )
        for edge in edges
        if int(edge["source"]) != int(edge["target"])
    ]
    metrics, _ = analyze_topology_projection(range(num_steps), weighted)
    return {
        key: metrics[key]
        for key in (
            "dependency_coverage",
            "raw_direction",
            "backward_edge_mass",
            "cycle_edge_mass",
            "direction_score",
            "acyclicity_score",
            "edge_count",
        )
    }


def _mean_interval(values: list[float], reps: int, seed: int) -> dict[str, Any]:
    if not values:
        return {"mean": None, "bootstrap_95": None, "n": 0}
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        draws.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    draws.sort()
    return {
        "mean": sum(values) / len(values),
        "bootstrap_95": [
            draws[math.floor(0.025 * (len(draws) - 1))],
            draws[math.ceil(0.975 * (len(draws) - 1))],
        ],
        "n": len(values),
    }


def _source_aggregation_diagnostic(
    metrics_by_source: dict[str, dict[str, dict[str, float]]],
    reps: int,
    seed: int,
) -> dict[str, Any]:
    rows = [
        (
            variants["original"]["direction_score"],
            variants["original"]["acyclicity_score"],
            variants["reversed"]["direction_score"],
            variants["reversed"]["acyclicity_score"],
        )
        for variants in metrics_by_source.values()
    ]

    def summarize(sample: list[tuple[float, float, float, float]]) -> dict[str, float]:
        reversed_direction = [row[2] for row in sample]
        reversed_acyclicity = [row[3] for row in sample]
        averaged = [
            (direction + acyclicity) / 2
            for direction, acyclicity in zip(reversed_direction, reversed_acyclicity)
        ]
        direction_std = statistics.pstdev(reversed_direction)
        acyclicity_std = statistics.pstdev(reversed_acyclicity)
        average_std = statistics.pstdev(averaged)
        correlation = statistics.correlation(reversed_direction, reversed_acyclicity)
        opposite = sum(
            ((reversed_direction_score - original_direction) *
             (reversed_acyclicity_score - original_acyclicity)) < 0
            for (
                original_direction,
                original_acyclicity,
                reversed_direction_score,
                reversed_acyclicity_score,
            ) in sample
        ) / len(sample)
        return {
            "correlation": correlation,
            "direction_std": direction_std,
            "acyclicity_std": acyclicity_std,
            "average_std": average_std,
            "average_to_direction_std_ratio": (
                average_std / direction_std if direction_std else math.nan
            ),
            "opposite_delta_fraction": opposite,
        }

    point = summarize(rows)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {name: [] for name in point}
    for _ in range(reps):
        sampled = [rows[rng.randrange(len(rows))] for _ in rows]
        try:
            values = summarize(sampled)
        except statistics.StatisticsError:
            continue
        for name, value in values.items():
            if math.isfinite(value):
                draws[name].append(value)

    output = {"n": len(rows)}
    for name, value in point.items():
        ordered = sorted(draws[name])
        output[name] = {
            "value": value,
            "bootstrap_95": [
                ordered[math.floor(0.025 * (len(ordered) - 1))],
                ordered[math.ceil(0.975 * (len(ordered) - 1))],
            ],
        }
    return output


def _edge_jaccard(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    left_pairs = {(int(edge["source"]), int(edge["target"])) for edge in left}
    right_pairs = {(int(edge["source"]), int(edge["target"])) for edge in right}
    union = left_pairs | right_pairs
    return len(left_pairs & right_pairs) / len(union) if union else 1.0


def _matched_cycle_control(
    num_steps: int,
    edges: list[dict[str, Any]],
) -> dict[str, Any] | None:
    pairs = {(int(edge["source"]), int(edge["target"])) for edge in edges}
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_steps))
    graph.add_edges_from(pairs)
    if not nx.is_directed_acyclic_graph(graph):
        return None
    incident = {node for pair in pairs for node in pair}
    cycle_closing = []
    non_closing = []
    for source in range(num_steps):
        for target in range(source):
            if (source, target) in pairs or source not in incident or target not in incident:
                continue
            candidate = (source, target)
            if nx.has_path(graph, target, source):
                cycle_closing.append(candidate)
            else:
                non_closing.append(candidate)
    if not cycle_closing or not non_closing:
        return None
    cycle_edge, control_edge = min(
        (
            (cycle_edge, control_edge)
            for cycle_edge in cycle_closing
            for control_edge in non_closing
        ),
        key=lambda pair: (
            abs((pair[0][0] - pair[0][1]) - (pair[1][0] - pair[1][1])),
            pair,
        ),
    )
    positive_weights = [
        max(0.0, float(edge.get("confidence", edge.get("weight", 1.0))))
        for edge in edges
    ]
    weight = sum(positive_weights) / len(positive_weights) if positive_weights else 1.0
    shared = {"confidence": weight, "dep_type": "matched_backward_control"}
    cycle_metrics = _topology_metrics(
        num_steps,
        [*edges, {"source": cycle_edge[0], "target": cycle_edge[1], **shared}],
    )
    control_metrics = _topology_metrics(
        num_steps,
        [*edges, {"source": control_edge[0], "target": control_edge[1], **shared}],
    )
    return {
        "cycle_edge": list(cycle_edge),
        "control_edge": list(control_edge),
        "direction_score_delta": control_metrics["direction_score"] - cycle_metrics["direction_score"],
        "backward_mass_delta": control_metrics["backward_edge_mass"] - cycle_metrics["backward_edge_mass"],
        "coverage_delta": control_metrics["dependency_coverage"] - cycle_metrics["dependency_coverage"],
        "acyclicity_gap": control_metrics["acyclicity_score"] - cycle_metrics["acyclicity_score"],
    }


def cmd_order_stress(args: argparse.Namespace) -> None:
    pack_rows = [json.loads(line) for line in Path(args.pack).open() if line.strip()]
    pack = {str(row["record_id"]): row for row in pack_rows}
    predictions = [json.loads(line) for line in Path(args.predictions).open() if line.strip()]
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in predictions:
        source_id = str(row.get("source_record_id", row["record_id"]))
        grouped[source_id][str(row["order_variant"])] = row

    complete = {
        source_id: variants
        for source_id, variants in grouped.items()
        if source_id in pack and all(name in variants for name in ("original", "reversed", "interleaved"))
    }
    if not complete:
        raise RuntimeError("No records contain all three order controls")

    metric_names = (
        "dependency_coverage",
        "raw_direction",
        "backward_edge_mass",
        "cycle_edge_mass",
        "direction_score",
        "acyclicity_score",
        "edge_count",
    )
    metrics_by_source: dict[str, dict[str, dict[str, float]]] = {}
    for source_id, variants in complete.items():
        metrics_by_source[source_id] = {
            name: _topology_metrics(
                int(pack[source_id]["n_steps"]),
                variants[name].get("display_edges", variants[name]["edges"]),
            )
            for name in ("original", "reversed", "interleaved")
        }

    by_variant = {}
    for variant in ("original", "reversed", "interleaved"):
        by_variant[variant] = {
            metric: _mean_interval(
                [rows[variant][metric] for rows in metrics_by_source.values()],
                args.bootstrap_reps,
                args.seed,
            )
            for metric in metric_names
        }

    paired = {
        "original_minus_reversed_direction_score": _mean_interval(
            [rows["original"]["direction_score"] - rows["reversed"]["direction_score"] for rows in metrics_by_source.values()],
            args.bootstrap_reps,
            args.seed,
        ),
        "original_minus_interleaved_direction_score": _mean_interval(
            [rows["original"]["direction_score"] - rows["interleaved"]["direction_score"] for rows in metrics_by_source.values()],
            args.bootstrap_reps,
            args.seed,
        ),
        "original_minus_reversed_acyclicity_score": _mean_interval(
            [rows["original"]["acyclicity_score"] - rows["reversed"]["acyclicity_score"] for rows in metrics_by_source.values()],
            args.bootstrap_reps,
            args.seed,
        ),
    }
    invariance = {
        variant: _mean_interval(
            [
                _edge_jaccard(variants["original"]["edges"], variants[variant]["edges"])
                for variants in complete.values()
            ],
            args.bootstrap_reps,
            args.seed,
        )
        for variant in ("reversed", "interleaved")
    }

    cycle_controls = []
    for source_id, variants in complete.items():
        control = _matched_cycle_control(
            int(pack[source_id]["n_steps"]),
            variants["original"].get("display_edges", variants["original"]["edges"]),
        )
        if control is not None:
            cycle_controls.append(control)
    cycle_summary = {
        metric: _mean_interval(
            [float(row[metric]) for row in cycle_controls],
            args.bootstrap_reps,
            args.seed,
        )
        for metric in (
            "direction_score_delta",
            "backward_mass_delta",
            "coverage_delta",
            "acyclicity_gap",
        )
    }
    output = {
        "schema_version": "topoprm.order_stress.v1",
        "n_complete_source_traces": len(complete),
        "bootstrap": {
            "unit": "source_trace",
            "repetitions": args.bootstrap_reps,
            "seed": args.seed,
        },
        "by_variant": by_variant,
        "paired": paired,
        "source_aggregation_diagnostic": _source_aggregation_diagnostic(
            metrics_by_source,
            args.bootstrap_reps,
            args.seed,
        ),
        "canonical_edge_jaccard": invariance,
        "matched_cycle_control": {
            "n_usable_traces": len(cycle_controls),
            "construction": "equal-weight backward additions with fixed edge count, backward mass, and incident-node coverage; only one addition closes a path into a cycle",
            "summary": cycle_summary,
        },
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"[order-stress] -> {destination}")


def _noise_rng(seed: int, record_id: str, noise_type: str, rate: float) -> random.Random:
    material = f"{seed}:{record_id}:{noise_type}:{rate:.8f}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def _corrupt_edges(
    num_steps: int,
    edges: list[dict[str, Any]],
    noise_type: str,
    rate: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], int]:
    copied = [dict(edge) for edge in edges]
    if noise_type == "drop":
        retained = [edge for edge in copied if rng.random() >= rate]
        return retained, len(copied) - len(retained)

    if noise_type == "flip":
        selected = {index for index in range(len(copied)) if rng.random() < rate}
        by_pair: dict[tuple[int, int], dict[str, Any]] = {}
        changed = 0
        for index, edge in enumerate(copied):
            source = int(edge["source"])
            target = int(edge["target"])
            if index in selected:
                source, target = target, source
                changed += 1
            candidate = {**edge, "source": source, "target": target}
            pair = (source, target)
            incumbent = by_pair.get(pair)
            if incumbent is None or float(candidate.get("confidence", 1.0)) > float(
                incumbent.get("confidence", 1.0)
            ):
                by_pair[pair] = candidate
        return list(by_pair.values()), changed

    if noise_type == "add":
        existing = {(int(edge["source"]), int(edge["target"])) for edge in copied}
        candidates = [
            (source, target)
            for source in range(num_steps)
            for target in range(num_steps)
            if source != target and (source, target) not in existing
        ]
        rng.shuffle(candidates)
        count = min(len(candidates), int(math.floor(rate * len(copied) + 0.5)))
        weights = sorted(
            max(0.0, float(edge.get("confidence", edge.get("weight", 1.0))))
            for edge in copied
        )
        weight = weights[len(weights) // 2] if weights else 1.0
        additions = [
            {
                "source": source,
                "target": target,
                "confidence": weight,
                "dep_type": "controlled_false_edge",
            }
            for source, target in candidates[:count]
        ]
        return [*copied, *additions], len(additions)

    raise ValueError(f"Unknown noise type: {noise_type}")


def cmd_edge_noise(args: argparse.Namespace) -> None:
    pack_rows = [json.loads(line) for line in Path(args.pack).open() if line.strip()]
    pack = {str(row["record_id"]): row for row in pack_rows}
    predictions = [json.loads(line) for line in Path(args.predictions).open() if line.strip()]
    original: dict[str, dict[str, Any]] = {}
    for row in predictions:
        source_id = str(row.get("source_record_id", row["record_id"]))
        if str(row.get("order_variant", "original")) == "original" and source_id in pack:
            original[source_id] = row
    if not original:
        raise RuntimeError("No original-order predictions match the audit pack")

    summary: dict[str, Any] = {}
    for noise_type in args.noise_types:
        by_rate: dict[str, Any] = {}
        for rate in args.rates:
            per_trace = []
            for source_id, row in original.items():
                n_steps = int(pack[source_id]["n_steps"])
                edges = row.get("display_edges", row["edges"])
                baseline = _topology_metrics(n_steps, edges)
                corrupted, modifications = _corrupt_edges(
                    n_steps,
                    edges,
                    noise_type,
                    rate,
                    _noise_rng(args.seed, source_id, noise_type, rate),
                )
                perturbed = _topology_metrics(n_steps, corrupted)
                edge_count = max(1.0, baseline["edge_count"])
                per_trace.append(
                    {
                        "direction_score_delta": perturbed["direction_score"] - baseline["direction_score"],
                        "acyclicity_score_delta": perturbed["acyclicity_score"] - baseline["acyclicity_score"],
                        "direction_score_absolute_change": abs(
                            perturbed["direction_score"] - baseline["direction_score"]
                        ),
                        "acyclicity_score_absolute_change": abs(
                            perturbed["acyclicity_score"] - baseline["acyclicity_score"]
                        ),
                        "coverage_delta": perturbed["dependency_coverage"] - baseline["dependency_coverage"],
                        "cycle_mass_delta": perturbed["cycle_edge_mass"] - baseline["cycle_edge_mass"],
                        "realized_modification_rate": modifications / edge_count,
                    }
                )
            by_rate[f"{rate:.3f}"] = {
                metric: _mean_interval(
                    [float(row[metric]) for row in per_trace],
                    args.bootstrap_reps,
                    args.seed,
                )
                for metric in per_trace[0]
            }
        summary[noise_type] = by_rate

    output = {
        "schema_version": "topoprm.edge_noise.v1",
        "n_source_traces": len(original),
        "protocol": {
            "unit": "source_trace",
            "perturbation_seed": args.seed,
            "bootstrap_repetitions": args.bootstrap_reps,
            "bootstrap_seed": args.seed,
            "drop": "delete each retained edge independently with probability p",
            "flip": "reverse each retained edge independently with probability p; pair collisions retain the higher-confidence edge",
            "add": "add round(p times retained-edge-count) uniformly shuffled non-edges at the trace median edge confidence",
        },
        "summary": summary,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"[edge-noise] -> {destination}")


_JUDGE_SYS = (
    "You are a meticulous mathematics grader. You are given a math problem and a "
    "solution that has been split into numbered steps. Your job is to identify the "
    "SUPPORT DEPENDENCIES between steps: step i supports step j (i<j) if the result, "
    "quantity, or established fact in step i is NECESSARY to derive or justify step j. "
    "Ignore mere textual similarity or shared words that are not logically used. "
    "Only output dependencies you are confident a human grader would agree with."
)


def _judge_prompt(rec: dict[str, Any]) -> str:
    lines = [f"Problem: {rec['question']}", "", "Steps:"]
    for i, s in enumerate(rec["steps"]):
        lines.append(f"[{i}] {s}")
    lines += [
        "",
        "For every ordered pair (i, j) with i < j where step i provides NECESSARY "
        "support for step j, identify the pair. You may reason briefly, but you "
        "MUST end your reply with a line of the exact form:",
        "ANSWER: [[i,j], ...]",
        "where the value is a JSON array of [i,j] integer pairs (use [] if there "
        "are no dependencies). The ANSWER line must be the last line.",
    ]
    return "\n".join(lines)


def _parse_pairs(text: str, n: int) -> list[list[int]]:
    # Prefer content after an explicit answer marker (handles thinking models).
    for marker in ("ANSWER:", "Answer:", "</think>", "Final answer:", "FINAL:"):
        if marker in text:
            text = text.split(marker)[-1]
            break
    # Find ALL bracketed arrays of pairs and take the last parseable one that
    # looks like a list of [i, j] pairs (thinking models restate the array).
    candidates = re.findall(r"\[\s*(?:\[\s*\d+\s*,\s*\d+\s*\]\s*,?\s*)*\]", text, re.DOTALL)
    arr = None
    for cand in reversed(candidates):
        try:
            parsed = json.loads(cand)
        except Exception:
            continue
        if isinstance(parsed, list) and (not parsed or isinstance(parsed[0], list)):
            arr = parsed
            break
    if arr is None:
        return []
    out = []
    seen = set()
    for p in arr:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            try:
                i, j = int(p[0]), int(p[1])
            except (ValueError, TypeError):
                continue
            if 0 <= i < j < n and (i, j) not in seen:
                seen.add((i, j))
                out.append([i, j])
    return out


def cmd_annotate_local(args: argparse.Namespace) -> None:
    """Independent judge via a local HF transformers model (no external API)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    recs = [json.loads(l) for l in Path(args.pack).open() if l.strip()]
    out = Path(args.out)
    done: set[str] = set()
    if out.exists() and args.resume:
        for l in out.open():
            if l.strip():
                done.add(json.loads(l)["record_id"])
    todo = [r for r in recs if r["record_id"] not in done]
    if not todo:
        print("[annotate-local] nothing to do")
        return

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    print(f"[annotate-local] loaded {args.model}; {len(todo)} traces to judge", flush=True)

    def _gen(rec: dict[str, Any], sample: bool) -> str:
        msgs = [
            {"role": "system", "content": _JUDGE_SYS},
            {"role": "user", "content": _judge_prompt(rec)},
        ]
        ct_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if "enable_thinking" in _chat_kwargs(tok):
            ct_kwargs["enable_thinking"] = False  # Qwen3: disable thinking for JSON output
        text = tok.apply_chat_template(msgs, **ct_kwargs)
        inputs = tok(text, return_tensors="pt").to(model.device)
        gk = dict(max_new_tokens=args.max_tokens, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if sample:
            gk.update(do_sample=True, temperature=args.vote_temperature, top_p=0.95)
        else:
            gk.update(do_sample=False)
        with torch.no_grad():
            gen = model.generate(**inputs, **gk)
        return tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def _vote(rec: dict[str, Any]) -> tuple[list[list[int]], list[dict]]:
        """Self-consistency: K judges, keep edges agreed by >= ceil(K/2) votes.

        Majority voting denoises the judge: spurious edges appear in few
        samples and drop out, genuine edges recur, which lifts both precision
        (fewer flukes) and recall (union catches edges greedy decoding missed).
        """
        from collections import Counter
        votes: Counter = Counter()
        raws = []
        k = max(1, args.votes)
        for s in range(k):
            content = _gen(rec, sample=(s > 0 or k == 1 and args.vote_temperature > 0))
            raws.append(content[-200:])
            for pr in _parse_pairs(content, rec["n_steps"]):
                votes[tuple(pr)] += 1
        thresh = (k // 2) + 1 if k > 1 else 1
        edges = sorted([list(e) for e, c in votes.items() if c >= thresh])
        return edges, raws

    with out.open("a" if args.resume else "w") as f:
        for k, rec in enumerate(todo):
            try:
                if args.votes > 1:
                    pairs, raws = _vote(rec)
                    raw_str = " || ".join(raws)
                else:
                    content = _gen(rec, sample=False)
                    pairs = _parse_pairs(content, rec["n_steps"])
                    raw_str = content[-400:]
                err = None
            except Exception as e:  # noqa: BLE001
                pairs, raw_str, err = [], "", str(e)[:160]
            f.write(
                json.dumps(
                    {
                        "record_id": rec["record_id"],
                        "n_steps": rec["n_steps"],
                        "llm_edges": pairs,
                        "raw": raw_str,
                        "error": err,
                        "votes": args.votes,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
            print(f"[annotate-local {k+1}/{len(todo)}] {rec['record_id']} "
                  f"{'ERR' if err else 'ok'} edges={len(pairs)} votes={args.votes}", flush=True)
    print(f"[annotate-local] wrote {len(todo)} annotations -> {out}")


def _chat_kwargs(tok) -> set:
    import inspect
    try:
        return set(inspect.signature(tok.apply_chat_template).parameters)
    except Exception:
        return set()


def cmd_annotate(args: argparse.Namespace) -> None:
    if getattr(args, "local", False):
        return cmd_annotate_local(args)
    from openai import OpenAI

    key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    client = OpenAI(base_url=args.base_url, api_key=key, timeout=args.timeout)
    recs = [json.loads(l) for l in Path(args.pack).open() if l.strip()]
    out = Path(args.out)
    done: dict[str, Any] = {}
    if out.exists() and args.resume:
        for l in out.open():
            if l.strip():
                d = json.loads(l)
                done[d["record_id"]] = d
    with out.open("a" if args.resume else "w") as f:
        for k, rec in enumerate(recs):
            if rec["record_id"] in done:
                continue
            prompt = _judge_prompt(rec)
            content, err = "", None
            for attempt in range(args.retries):
                try:
                    r = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": _JUDGE_SYS},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=512,
                        temperature=0.0,
                    )
                    content = r.choices[0].message.content or ""
                    err = None
                    break
                except Exception as e:  # noqa: BLE001
                    err = str(e)[:160]
                    time.sleep(2 * (attempt + 1))
            pairs = _parse_pairs(content, rec["n_steps"]) if not err else []
            rowout = {
                "record_id": rec["record_id"],
                "n_steps": rec["n_steps"],
                "llm_edges": pairs,
                "raw": content,
                "error": err,
            }
            f.write(json.dumps(rowout, ensure_ascii=False) + "\n")
            f.flush()
            tag = "ERR" if err else "ok"
            print(f"[annotate {k+1}/{len(recs)}] {rec['record_id']} {tag} edges={len(pairs)}", flush=True)


def cmd_score(args: argparse.Namespace) -> None:
    packs = {r["record_id"]: r for r in (json.loads(l) for l in Path(args.pack).open() if l.strip())}
    anns = {r["record_id"]: r for r in (json.loads(l) for l in Path(args.ann).open() if l.strip())}

    tp = fp = fn = 0
    by_type_tp: Counter = Counter()
    by_type_fp: Counter = Counter()
    fp_examples: list[dict[str, Any]] = []
    per_trace = []
    n_used = 0
    for rid, pack in packs.items():
        ann = anns.get(rid)
        if ann is None or ann.get("error"):
            continue
        n_used += 1
        gold = {tuple(p) for p in ann["llm_edges"]}
        ext = {(e["source"], e["target"]): e for e in pack["extractor_edges"]}
        ext_set = set(ext)
        t_tp = len(ext_set & gold)
        t_fp = len(ext_set - gold)
        t_fn = len(gold - ext_set)
        tp += t_tp
        fp += t_fp
        fn += t_fn
        for pr in ext_set & gold:
            by_type_tp[ext[pr].get("dep_type") or ext[pr]["edge_type"]] += 1
        for pr in ext_set - gold:
            dt = ext[pr].get("dep_type") or ext[pr]["edge_type"]
            by_type_fp[dt] += 1
            if len(fp_examples) < 12:
                fp_examples.append(
                    {
                        "record_id": rid,
                        "edge": list(pr),
                        "dep_type": dt,
                        "src": pack["steps"][pr[0]],
                        "tgt": pack["steps"][pr[1]],
                    }
                )
        per_trace.append({"record_id": rid, "tp": t_tp, "fp": t_fp, "fn": t_fn})

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    type_reliability = {}
    for dt in set(list(by_type_tp) + list(by_type_fp)):
        d_tp, d_fp = by_type_tp[dt], by_type_fp[dt]
        type_reliability[dt] = {
            "precision": round(d_tp / (d_tp + d_fp), 4) if (d_tp + d_fp) else 0.0,
            "tp": d_tp,
            "fp": d_fp,
        }

    result = {
        "n_traces_scored": n_used,
        "n_traces_total": len(packs),
        "edges": {"tp": tp, "fp": fp, "fn": fn},
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "per_dep_type": type_reliability,
        "false_positive_examples": fp_examples,
        "judge_model": args.model,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "false_positive_examples"}, indent=2))
    print(f"[score] -> {args.out}")


def _cohen_kappa(left: list[int], right: list[int]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_yes = sum(left) / len(left)
    right_yes = sum(right) / len(right)
    expected = left_yes * right_yes + (1.0 - left_yes) * (1.0 - right_yes)
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)


def _binary_f1(left: list[int], right: list[int]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    tp = sum(a == b == 1 for a, b in zip(left, right))
    fp = sum(a == 1 and b == 0 for a, b in zip(left, right))
    fn = sum(a == 0 and b == 1 for a, b in zip(left, right))
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else None


def _fleiss_kappa(vote_counts: list[int], num_raters: int) -> float | None:
    if not vote_counts or num_raters < 2:
        return None
    agreement = [
        (yes * (yes - 1) + (num_raters - yes) * (num_raters - yes - 1))
        / (num_raters * (num_raters - 1))
        for yes in vote_counts
    ]
    yes_rate = sum(vote_counts) / (len(vote_counts) * num_raters)
    expected = yes_rate**2 + (1.0 - yes_rate) ** 2
    if math.isclose(expected, 1.0):
        return None
    return (sum(agreement) / len(agreement) - expected) / (1.0 - expected)


def _load_human_annotations(path: Path, packs: dict[str, dict]) -> tuple[str, dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    slot = str(payload.get("annotator_slot", path.stem))
    raw_annotations = payload.get("annotations")
    if not isinstance(raw_annotations, dict):
        raise ValueError(f"Missing annotations mapping in {path}")

    normalized: dict[str, dict] = {}
    for annotation in raw_annotations.values():
        if not isinstance(annotation, dict):
            raise ValueError(f"Malformed annotation in {path}")
        source_id = str(annotation.get("source_record_id", ""))
        if source_id not in packs:
            raise ValueError(f"Unknown source_record_id {source_id!r} in {path}")
        if source_id in normalized:
            raise ValueError(f"Duplicate source_record_id {source_id!r} in {path}")
        n_steps = len(packs[source_id]["steps"])
        order = [int(index) for index in annotation.get("step_order", [])]
        if sorted(order) != list(range(n_steps)):
            raise ValueError(f"Invalid step_order for {source_id!r} in {path}")
        edges: set[tuple[int, int]] = set()
        for edge in annotation.get("edges", []):
            if not isinstance(edge, list) or len(edge) != 2:
                raise ValueError(f"Malformed edge for {source_id!r} in {path}")
            display_source, display_target = int(edge[0]), int(edge[1])
            if not (
                0 <= display_source < n_steps
                and 0 <= display_target < n_steps
                and display_source != display_target
            ):
                raise ValueError(f"Edge outside step range for {source_id!r} in {path}")
            edges.add((order[display_source], order[display_target]))
        normalized[source_id] = {
            "edges": edges,
            "order_variant": str(annotation.get("order_variant", "")),
            "segmentation_unusable": bool(annotation.get("segmentation_unusable", False)),
            "confidence": str(annotation.get("confidence", "medium")),
        }

    missing = sorted(set(packs) - set(normalized))
    extra = sorted(set(normalized) - set(packs))
    if missing or extra:
        raise ValueError(
            f"Incomplete annotation file {path}: missing={len(missing)}, extra={len(extra)}"
        )
    return slot, normalized


def cmd_reconcile_human(args: argparse.Namespace) -> None:
    packs = {
        row["record_id"]: row
        for row in (json.loads(line) for line in Path(args.pack).open() if line.strip())
    }
    if not packs:
        raise ValueError("Annotation pack is empty")
    loaded = [_load_human_annotations(Path(path), packs) for path in args.annotations]
    slots = [slot for slot, _ in loaded]
    if len(set(slots)) != len(slots):
        raise ValueError(f"Annotator slots must be distinct: {slots}")
    annotations = {slot: rows for slot, rows in loaded}
    threshold = len(slots) // 2 + 1

    pair_labels: dict[tuple[str, str], tuple[list[int], list[int]]] = {
        pair: ([], []) for pair in combinations(slots, 2)
    }
    segmentation_labels: dict[tuple[str, str], tuple[list[int], list[int]]] = {
        pair: ([], []) for pair in combinations(slots, 2)
    }
    all_usable_votes: list[int] = []
    references: list[dict[str, Any]] = []
    majority_unusable = 0

    for record_id, pack in packs.items():
        n_steps = len(pack["steps"])
        order_variants = {annotations[slot][record_id]["order_variant"] for slot in slots}
        if len(order_variants) != 1 or not next(iter(order_variants)):
            raise ValueError(
                f"Annotators must share one non-empty order_variant for {record_id}: "
                f"{sorted(order_variants)}"
            )
        order_variant = next(iter(order_variants))
        universe = [(source, target) for source in range(n_steps) for target in range(n_steps) if source != target]
        unusable_votes = sum(
            annotations[slot][record_id]["segmentation_unusable"] for slot in slots
        )
        usable_slots = [
            slot for slot in slots if not annotations[slot][record_id]["segmentation_unusable"]
        ]
        is_majority_unusable = unusable_votes >= threshold
        if is_majority_unusable:
            majority_unusable += 1
            majority_edges: list[list[int]] = []
        else:
            usable_threshold = len(usable_slots) // 2 + 1
            counts = Counter(
                edge
                for slot in usable_slots
                for edge in annotations[slot][record_id]["edges"]
            )
            majority_edges = [list(edge) for edge, count in sorted(counts.items()) if count >= usable_threshold]

        references.append({
            "record_id": record_id,
            "n_steps": n_steps,
            "order_variant": order_variant,
            "llm_edges": majority_edges,
            "error": "majority_segmentation_unusable" if is_majority_unusable else None,
            "reference_type": "strict_human_majority",
            "annotator_slots": slots,
            "usable_annotators": len(usable_slots),
            "segmentation_unusable_votes": unusable_votes,
        })

        for left_slot, right_slot in combinations(slots, 2):
            left_ann = annotations[left_slot][record_id]
            right_ann = annotations[right_slot][record_id]
            seg_left, seg_right = segmentation_labels[(left_slot, right_slot)]
            seg_left.append(int(left_ann["segmentation_unusable"]))
            seg_right.append(int(right_ann["segmentation_unusable"]))
            if left_ann["segmentation_unusable"] or right_ann["segmentation_unusable"]:
                continue
            edge_left, edge_right = pair_labels[(left_slot, right_slot)]
            left_edges, right_edges = left_ann["edges"], right_ann["edges"]
            edge_left.extend(int(pair in left_edges) for pair in universe)
            edge_right.extend(int(pair in right_edges) for pair in universe)

        if len(usable_slots) == len(slots):
            for pair in universe:
                all_usable_votes.append(sum(pair in annotations[slot][record_id]["edges"] for slot in slots))

    pairwise = {}
    for pair, (left, right) in pair_labels.items():
        seg_left, seg_right = segmentation_labels[pair]
        pairwise[f"{pair[0]}__{pair[1]}"] = {
            "directed_pair_items": len(left),
            "edge_cohen_kappa": _cohen_kappa(left, right),
            "edge_positive_f1": _binary_f1(left, right),
            "segmentation_cohen_kappa": _cohen_kappa(seg_left, seg_right),
            "segmentation_exact_agreement": sum(a == b for a, b in zip(seg_left, seg_right)) / len(seg_left),
        }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in references:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)
    reference_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {
        "schema_version": "topoprm.human_edge_reconciliation.v1",
        "records": len(references),
        "annotator_slots": slots,
        "strict_majority_threshold": threshold,
        "majority_segmentation_unusable": majority_unusable,
        "all_usable_records": sum(row["usable_annotators"] == len(slots) for row in references),
        "pairwise_agreement": pairwise,
        "edge_fleiss_kappa_all_usable": _fleiss_kappa(all_usable_votes, len(slots)),
        "reference_path": str(output),
        "reference_sha256": reference_sha256,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[reconcile-human] -> {output} and {report_path}")


def _edge_counts(predicted: set[tuple[int, int]], gold: set[tuple[int, int]]) -> tuple[int, int, int]:
    return len(predicted & gold), len(predicted - gold), len(gold - predicted)


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _contains_cycle(num_steps: int, edges: set[tuple[int, int]]) -> bool:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_steps))
    graph.add_edges_from(edges)
    return not nx.is_directed_acyclic_graph(graph)


def _human_trace_summary(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    edge = _prf(tp, fp, fn)
    direction_exact = sum(int(row["direction_exact"]) for row in rows)
    direction_comparable = sum(int(row["direction_comparable"]) for row in rows)
    cycle_tp = sum(bool(row["gold_cycle"]) and bool(row["predicted_cycle"]) for row in rows)
    cycle_fp = sum(not bool(row["gold_cycle"]) and bool(row["predicted_cycle"]) for row in rows)
    cycle_fn = sum(bool(row["gold_cycle"]) and not bool(row["predicted_cycle"]) for row in rows)
    cycle_f1 = _prf(cycle_tp, cycle_fp, cycle_fn)["f1"] if cycle_tp + cycle_fp + cycle_fn else None
    retained = sum(int(row["retained_edges"]) for row in rows)
    candidates = sum(int(row["candidate_unordered_pairs"]) for row in rows)
    return {
        "edge_precision": edge["precision"],
        "edge_recall": edge["recall"],
        "edge_f1": edge["f1"],
        "direction_accuracy": direction_exact / direction_comparable if direction_comparable else None,
        "cycle_f1": cycle_f1,
        "prediction_coverage": retained / candidates if candidates else None,
    }


def _bootstrap_human_traces(
    rows: list[dict[str, Any]], reps: int, seed: int
) -> dict[str, list[float] | None]:
    if not rows or reps <= 0:
        return {}
    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(reps):
        sampled = [rows[rng.randrange(len(rows))] for _ in rows]
        for metric, value in _human_trace_summary(sampled).items():
            if value is not None:
                draws[metric].append(float(value))
    intervals: dict[str, list[float] | None] = {}
    for metric in (
        "edge_precision",
        "edge_recall",
        "edge_f1",
        "direction_accuracy",
        "cycle_f1",
        "prediction_coverage",
    ):
        values = sorted(draws.get(metric, []))
        intervals[metric] = (
            [
                values[math.floor(0.025 * (len(values) - 1))],
                values[math.ceil(0.975 * (len(values) - 1))],
            ]
            if values
            else None
        )
    return intervals


def cmd_score_human(args: argparse.Namespace) -> None:
    packs = {
        row["record_id"]: row
        for row in (json.loads(line) for line in Path(args.pack).open() if line.strip())
    }
    references = {
        row["record_id"]: row
        for row in (json.loads(line) for line in Path(args.reference).open() if line.strip())
    }
    predictions: dict[tuple[str, str], dict] = {}
    for row in (json.loads(line) for line in Path(args.predictions).open() if line.strip()):
        source_id = str(row.get("source_record_id", row.get("record_id", "")))
        variant = str(row.get("order_variant", "original"))
        key = (source_id, variant)
        if key in predictions:
            raise ValueError(f"Duplicate prediction for {source_id}::{variant}")
        predictions[key] = row

    if set(references) != set(packs):
        raise ValueError(
            f"Reference/pack mismatch: reference={len(references)}, pack={len(packs)}"
        )
    totals = Counter(tp=0, fp=0, fn=0)
    by_length: dict[str, Counter] = {
        "short": Counter(tp=0, fp=0, fn=0),
        "medium": Counter(tp=0, fp=0, fn=0),
        "long": Counter(tp=0, fp=0, fn=0),
    }
    cycle_gold: list[int] = []
    cycle_predicted: list[int] = []
    direction_exact = direction_comparable = 0
    retained_edges = candidate_pairs = 0
    per_trace: list[dict[str, Any]] = []
    skipped = 0

    for record_id, pack in packs.items():
        reference = references[record_id]
        if reference.get("error"):
            skipped += 1
            continue
        variant = str(reference.get("order_variant", ""))
        prediction = predictions.get((record_id, variant))
        if prediction is None:
            raise ValueError(f"Missing prediction for {record_id}::{variant}")
        n_steps = len(pack["steps"])
        gold = {tuple(map(int, edge)) for edge in reference.get("llm_edges", [])}
        predicted = {
            (int(edge["source"]), int(edge["target"]))
            for edge in prediction.get("edges", [])
        }
        for source, target in gold | predicted:
            if not (0 <= source < n_steps and 0 <= target < n_steps and source != target):
                raise ValueError(f"Edge outside step range for {record_id}: {(source, target)}")
        tp, fp, fn = _edge_counts(predicted, gold)
        totals.update(tp=tp, fp=fp, fn=fn)
        band = "short" if n_steps <= 3 else ("medium" if n_steps <= 6 else "long")
        by_length[band].update(tp=tp, fp=fp, fn=fn)

        predicted_unordered = {frozenset(edge) for edge in predicted}
        comparable_gold = [edge for edge in gold if frozenset(edge) in predicted_unordered]
        trace_direction_comparable = len(comparable_gold)
        trace_direction_exact = sum(edge in predicted for edge in comparable_gold)
        direction_comparable += trace_direction_comparable
        direction_exact += trace_direction_exact
        gold_cycle = int(_contains_cycle(n_steps, gold))
        predicted_cycle = int(_contains_cycle(n_steps, predicted))
        cycle_gold.append(gold_cycle)
        cycle_predicted.append(predicted_cycle)
        retained_edges += len(predicted)
        candidate_pairs += n_steps * (n_steps - 1) // 2
        per_trace.append({
            "record_id": record_id,
            "order_variant": variant,
            "length_band": band,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "direction_exact": trace_direction_exact,
            "direction_comparable": trace_direction_comparable,
            "gold_cycle": bool(gold_cycle),
            "predicted_cycle": bool(predicted_cycle),
            "retained_edges": len(predicted),
            "candidate_unordered_pairs": n_steps * (n_steps - 1) // 2,
        })

    cycle_tp, cycle_fp, cycle_fn = _edge_counts(
        {index for index, value in enumerate(cycle_predicted) if value},
        {index for index, value in enumerate(cycle_gold) if value},
    )
    overall = _prf(totals["tp"], totals["fp"], totals["fn"])
    cycle_metrics: dict[str, float | None] = _prf(cycle_tp, cycle_fp, cycle_fn)
    if cycle_tp + cycle_fp + cycle_fn == 0:
        cycle_metrics = {"precision": None, "recall": None, "f1": None}
    result = {
        "schema_version": "topoprm.human_edge_score.v1",
        "extractor": args.label,
        "records_total": len(packs),
        "records_scored": len(per_trace),
        "records_skipped_segmentation": skipped,
        "edges": dict(totals),
        **overall,
        "direction_accuracy_on_recovered_gold_pairs": (
            direction_exact / direction_comparable if direction_comparable else None
        ),
        "direction_exact": direction_exact,
        "direction_comparable_gold_edges": direction_comparable,
        "cycle": {
            "tp": cycle_tp,
            "fp": cycle_fp,
            "fn": cycle_fn,
            **cycle_metrics,
        },
        "prediction_coverage": retained_edges / candidate_pairs if candidate_pairs else 0.0,
        "retained_edges": retained_edges,
        "candidate_unordered_pairs": candidate_pairs,
        "edge_f1_by_trace_length": {
            band: _prf(counts["tp"], counts["fp"], counts["fn"])["f1"]
            for band, counts in by_length.items()
        },
        "bootstrap": {
            "unit": "human_audit_trace",
            "repetitions": args.bootstrap_reps,
            "seed": args.seed,
            "ci95": _bootstrap_human_traces(per_trace, args.bootstrap_reps, args.seed),
        },
        "per_trace": per_trace,
        "prediction_sha256": hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_trace"}, indent=2))
    print(f"[score-human] -> {output}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=120)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="rebuttal/outputs/edge_validation_pack.jsonl")
    s.add_argument("--reject-malformed", action="store_true")
    s.set_defaults(func=cmd_sample)

    p = sub.add_parser("predict-encoder")
    p.add_argument("--pack", default="rebuttal/outputs/edge_validation_pack.jsonl")
    p.add_argument("--out", default="rebuttal/outputs/edge_encoder_predictions.jsonl")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=32)
    p.add_argument(
        "--all-order-controls",
        action="store_true",
        help="run original, reversed, and interleaved presentation orders for every trace",
    )
    p.set_defaults(func=cmd_predict_encoder)

    o = sub.add_parser("order-stress")
    o.add_argument("--pack", default="rebuttal/outputs/edge_validation_pack.jsonl")
    o.add_argument("--predictions", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--bootstrap-reps", type=int, default=10_000)
    o.add_argument("--seed", type=int, default=0)
    o.set_defaults(func=cmd_order_stress)

    n = sub.add_parser("edge-noise")
    n.add_argument("--pack", default="rebuttal/outputs/edge_validation_pack.jsonl")
    n.add_argument("--predictions", required=True)
    n.add_argument("--out", required=True)
    n.add_argument("--noise-types", nargs="+", choices=("drop", "add", "flip"), default=("drop", "add", "flip"))
    n.add_argument("--rates", nargs="+", type=float, default=(0.1, 0.2, 0.3))
    n.add_argument("--bootstrap-reps", type=int, default=10_000)
    n.add_argument("--seed", type=int, default=0)
    n.set_defaults(func=cmd_edge_noise)

    a = sub.add_parser("annotate")
    a.add_argument("--pack", default="rebuttal/outputs/edge_validation_pack.jsonl")
    a.add_argument("--out", default="rebuttal/outputs/edge_validation_annotations.jsonl")
    a.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    a.add_argument("--model", default=os.environ.get("EDGE_JUDGE_MODEL", "gpt-4o"))
    a.add_argument("--api-key", default=None)
    a.add_argument("--timeout", type=float, default=60.0)
    a.add_argument("--retries", type=int, default=3)
    a.add_argument("--resume", action="store_true", default=True)
    a.add_argument("--local", action="store_true", help="use local vLLM judge instead of external API")
    a.add_argument("--tp", type=int, default=2, help="tensor parallel size for local vLLM")
    a.add_argument("--max-model-len", type=int, default=8192)
    a.add_argument("--max-tokens", type=int, default=512)
    a.add_argument("--votes", type=int, default=1,
                   help="self-consistency: number of judge samples; majority vote if >1")
    a.add_argument("--vote-temperature", type=float, default=0.7,
                   help="sampling temperature for the vote samples (>0)")
    a.set_defaults(func=cmd_annotate)

    c = sub.add_parser("score")
    c.add_argument("--pack", default="rebuttal/outputs/edge_validation_pack.jsonl")
    c.add_argument("--ann", default="rebuttal/outputs/edge_validation_annotations.jsonl")
    c.add_argument("--out", default="rebuttal/outputs/edge_validation_results.json")
    c.add_argument("--model", default=os.environ.get("EDGE_JUDGE_MODEL", "gpt-4o"))
    c.set_defaults(func=cmd_score)

    h = sub.add_parser("reconcile-human")
    h.add_argument("--pack", default="rebuttal/outputs/edge_validation_pack.jsonl")
    h.add_argument("--annotations", nargs=3, required=True)
    h.add_argument("--out", default="rebuttal/outputs/human_edge_majority.jsonl")
    h.add_argument("--report", default="rebuttal/outputs/human_edge_agreement.json")
    h.set_defaults(func=cmd_reconcile_human)

    hs = sub.add_parser("score-human")
    hs.add_argument("--pack", default="rebuttal/outputs/edge_validation_pack.jsonl")
    hs.add_argument("--reference", required=True)
    hs.add_argument("--predictions", required=True)
    hs.add_argument("--out", required=True)
    hs.add_argument("--label", required=True)
    hs.add_argument("--bootstrap-reps", type=int, default=10_000)
    hs.add_argument("--seed", type=int, default=0)
    hs.set_defaults(func=cmd_score_human)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
