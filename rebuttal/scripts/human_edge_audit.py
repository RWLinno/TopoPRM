#!/usr/bin/env python3
"""Aggregate blinded human edge annotations in source-step coordinates."""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _canonical_edges(annotation: dict[str, Any], n_steps: int) -> set[tuple[int, int]]:
    """Map displayed edge indices back to the source trace indices."""
    order = annotation.get("step_order")
    if not isinstance(order, list) or sorted(order) != list(range(n_steps)):
        raise ValueError(f"invalid step_order for {annotation.get('record_id')}")
    edges = set()
    for edge in annotation.get("edges", []):
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError(f"malformed edge for {annotation.get('record_id')}")
        source, target = map(int, edge)
        if not (0 <= source < n_steps and 0 <= target < n_steps) or source == target:
            raise ValueError(f"edge outside step range for {annotation.get('record_id')}")
        edges.add((order[source], order[target]))
    return edges


def _load_slots(
    annotation_dir: Path, slots: list[str], pack: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[str]]:
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    warnings: list[str] = []
    for slot in slots:
        path = annotation_dir / f"annotator_{slot}.json"
        if not path.exists():
            loaded[slot] = {}
            warnings.append(f"missing annotation file: {path}")
            continue
        payload = json.loads(path.read_text())
        raw = payload.get("annotations", {})
        normalized = {}
        for opaque_id, ann in raw.items():
            source_id = ann.get("source_record_id")
            if source_id not in pack:
                raise ValueError(f"unknown source_record_id {source_id!r} in slot {slot}")
            if ann.get("record_id") != opaque_id:
                raise ValueError(f"record key mismatch in slot {slot}: {opaque_id}")
            ann = dict(ann)
            ann["canonical_edges"] = _canonical_edges(ann, pack[source_id]["n_steps"])
            normalized[source_id] = ann
        loaded[slot] = normalized
    return loaded, warnings


def _binary_agreement(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    pack: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter()
    used = []
    for rid in pack:
        a, b = left.get(rid), right.get(rid)
        if not a or not b or a.get("segmentation_unusable") or b.get("segmentation_unusable"):
            continue
        used.append(rid)
        ae, be = a["canonical_edges"], b["canonical_edges"]
        n = pack[rid]["n_steps"]
        for edge in ((i, j) for i in range(n) for j in range(n) if i != j):
            counts[(edge in ae, edge in be)] += 1
    total = sum(counts.values())
    if not total:
        return {"n_traces": 0, "n_pair_items": 0, "raw_agreement": None, "cohen_kappa": None}
    both_1 = counts[(True, True)]
    both_0 = counts[(False, False)]
    a_only = counts[(True, False)]
    b_only = counts[(False, True)]
    observed = (both_1 + both_0) / total
    p_a = (both_1 + a_only) / total
    p_b = (both_1 + b_only) / total
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return {
        "n_traces": len(used),
        "n_pair_items": total,
        "raw_agreement": observed,
        "cohen_kappa": kappa,
        "counts": {
            "both_edge": both_1,
            "both_no_edge": both_0,
            "left_only": a_only,
            "right_only": b_only,
        },
    }


def _majority_reference(
    annotations: dict[str, dict[str, dict[str, Any]]],
    pack: dict[str, dict[str, Any]],
    slots: list[str],
) -> tuple[dict[str, set[tuple[int, int]]], dict[str, Any]]:
    references = {}
    excluded = Counter()
    for rid in pack:
        anns = [annotations[slot].get(rid) for slot in slots]
        if any(ann is None for ann in anns):
            excluded["incomplete"] += 1
            continue
        if any(ann.get("segmentation_unusable") for ann in anns if ann):
            excluded["segmentation_unusable"] += 1
            continue
        votes = Counter(edge for ann in anns for edge in ann["canonical_edges"])
        threshold = len(slots) // 2 + 1
        references[rid] = {edge for edge, count in votes.items() if count >= threshold}
    return references, {
        "rule": f"strict majority across {len(slots)} complete, usable annotations",
        "n_reference_traces": len(references),
        "excluded": dict(excluded),
    }


def _load_predictions(
    spec: str, pack: dict[str, dict[str, Any]]
) -> tuple[str, dict[str, dict[str, Any]]]:
    if "=" not in spec:
        raise ValueError("prediction spec must be LABEL=PATH or LABEL=PACK")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("prediction label cannot be empty")
    if raw_path == "PACK":
        rows = list(pack.values())
        edge_key = "extractor_edges"
    else:
        rows = _read_jsonl(Path(raw_path))
        edge_key = "edges"
    result = {}
    duplicate_variants: dict[str, set[str]] = {}
    for row in rows:
        # Order-control artifacts identify a row as ``source::variant`` while
        # keeping ``edges`` in source-step coordinates. Human references also
        # use source coordinates, so match on source_record_id and use the
        # original-order prediction for the primary edge audit.
        rid = row.get("source_record_id") or row.get("record_id")
        if rid not in pack:
            continue
        variant = str(row.get("order_variant") or "original")
        if rid in result:
            duplicate_variants.setdefault(rid, {result[rid]["order_variant"]}).add(variant)
            if result[rid]["order_variant"] == "original":
                continue
            if variant != "original":
                continue
        typed_edges = {}
        for edge in row.get(edge_key, []):
            if isinstance(edge, dict):
                pair = (int(edge["source"]), int(edge["target"]))
                edge_type = edge.get("dep_type") or edge.get("edge_type") or "unspecified"
            else:
                pair = tuple(map(int, edge))
                edge_type = "unspecified"
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError(f"invalid prediction edge for {rid}")
            typed_edges[pair] = edge_type
        result[rid] = {
            "edges": set(typed_edges),
            "types": typed_edges,
            "evaluated_pairs": row.get("evaluated_pairs"),
            "abstained_pairs": row.get("abstained_pairs"),
            "order_variant": variant,
        }
    unresolved = {
        rid: sorted(variants)
        for rid, variants in duplicate_variants.items()
        if result[rid]["order_variant"] != "original"
    }
    if unresolved:
        rid = next(iter(unresolved))
        raise ValueError(
            f"duplicate predictions for {rid} lack an original-order row: "
            f"{unresolved[rid]}"
        )
    return label, result


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _score_trace(
    rid: str,
    pred: dict[str, Any],
    gold: set[tuple[int, int]],
    pack: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    predicted = pred["edges"]
    tp, fp, fn = len(predicted & gold), len(predicted - gold), len(gold - predicted)
    shared_pairs = {
        frozenset(edge) for edge in predicted
    } & {frozenset(edge) for edge in gold}
    direction_correct = sum(
        1 for endpoints in shared_pairs if tuple(endpoints) in predicted & gold
        or tuple(reversed(tuple(endpoints))) in predicted & gold
    )
    # The three-way pair encoder evaluates one unordered pair and predicts
    # {no edge, left->right, right->left}; coverage therefore uses C(n, 2).
    candidate_count = pack[rid]["n_steps"] * (pack[rid]["n_steps"] - 1) // 2
    evaluated = pred["evaluated_pairs"]
    if evaluated is None:
        abstained = pred["abstained_pairs"]
        evaluated = candidate_count - int(abstained) if abstained is not None else candidate_count
    evaluated = max(0, min(int(evaluated), candidate_count))
    predicted_cycle, _ = _has_cycle(pack[rid]["n_steps"], predicted)
    gold_cycle, _ = _has_cycle(pack[rid]["n_steps"], gold)
    return {
        "record_id": rid,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "direction_correct": direction_correct,
        "direction_total": len(shared_pairs),
        "evaluated_pairs": evaluated,
        "candidate_pairs": candidate_count,
        "cycle_predicted": int(predicted_cycle),
        "cycle_gold": int(gold_cycle),
        **_prf(tp, fp, fn),
    }


def _pooled(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    rows = list(rows)
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    result = _prf(tp, fp, fn)
    cycle_tp = sum(row["cycle_predicted"] and row["cycle_gold"] for row in rows)
    cycle_fp = sum(row["cycle_predicted"] and not row["cycle_gold"] for row in rows)
    cycle_fn = sum(not row["cycle_predicted"] and row["cycle_gold"] for row in rows)
    cycle_metrics = _prf(cycle_tp, cycle_fp, cycle_fn)
    result.update({f"cycle_{key}": value for key, value in cycle_metrics.items()})
    result["direction_accuracy"] = (
        sum(row["direction_correct"] for row in rows)
        / sum(row["direction_total"] for row in rows)
        if sum(row["direction_total"] for row in rows)
        else 0.0
    )
    result["coverage"] = (
        sum(row["evaluated_pairs"] for row in rows)
        / sum(row["candidate_pairs"] for row in rows)
        if rows
        else 0.0
    )
    return result


def _bootstrap(
    rows: list[dict[str, Any]], reps: int, seed: int
) -> dict[str, list[float]]:
    if not rows or reps <= 0:
        return {}
    rng = random.Random(seed)
    samples = {
        key: []
        for key in (
            "precision",
            "recall",
            "f1",
            "direction_accuracy",
            "coverage",
            "cycle_precision",
            "cycle_recall",
            "cycle_f1",
        )
    }
    for _ in range(reps):
        draw = [rows[rng.randrange(len(rows))] for _ in rows]
        metrics = _pooled(draw)
        for key in samples:
            samples[key].append(metrics[key])
    result = {}
    for key, values in samples.items():
        values.sort()
        lo = values[math.floor(0.025 * (len(values) - 1))]
        hi = values[math.ceil(0.975 * (len(values) - 1))]
        result[key] = [lo, hi]
    return result


def _has_cycle(n_steps: int, edges: set[tuple[int, int]]) -> tuple[bool, set[tuple[int, int]]]:
    graph = {i: [] for i in range(n_steps)}
    for source, target in edges:
        graph[source].append(target)
    index = 0
    stack: list[int] = []
    indices: dict[int, int] = {}
    low: dict[int, int] = {}
    on_stack = set()
    cyclic_nodes = set()

    def visit(node: int) -> None:
        nonlocal index
        indices[node] = low[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in graph[node]:
            if target not in indices:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in on_stack:
                low[node] = min(low[node], indices[target])
        if low[node] == indices[node]:
            component = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > 1:
                cyclic_nodes.update(component)

    for node in range(n_steps):
        if node not in indices:
            visit(node)
    cycle_edges = {edge for edge in edges if edge[0] in cyclic_nodes and edge[1] in cyclic_nodes}
    return bool(cycle_edges), cycle_edges


def _reference_topology(
    references: dict[str, set[tuple[int, int]]], pack: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    total_edges = backward = 0
    cycle_edges: set[tuple[str, int, int]] = set()
    cyclic_traces = 0
    for rid, edges in references.items():
        total_edges += len(edges)
        backward += sum(source > target for source, target in edges)
        has_cycle, local_cycle_edges = _has_cycle(pack[rid]["n_steps"], edges)
        cyclic_traces += int(has_cycle)
        cycle_edges.update((rid, source, target) for source, target in local_cycle_edges)
    return {
        "n_edges": total_edges,
        "backward_edge_rate": backward / total_edges if total_edges else 0.0,
        "direction_consistency": 1 - backward / total_edges if total_edges else 0.0,
        "trace_cycle_rate": cyclic_traces / len(references) if references else 0.0,
        "cycle_edge_rate": len(cycle_edges) / total_edges if total_edges else 0.0,
        "text_order_dag_projection_cost": backward / total_edges if total_edges else 0.0,
    }


def _score_predictions(
    predictions: dict[str, dict[str, Any]],
    references: dict[str, set[tuple[int, int]]],
    pack: dict[str, dict[str, Any]],
    reps: int,
    seed: int,
) -> dict[str, Any]:
    common = [rid for rid in references if rid in predictions]
    rows = [_score_trace(rid, predictions[rid], references[rid], pack) for rid in common]
    pooled = _pooled(rows)
    macro = {
        key: sum(row[key] for row in rows) / len(rows) if rows else 0.0
        for key in ("precision", "recall", "f1")
    }
    by_type = Counter()
    for rid in common:
        pred = predictions[rid]
        gold = references[rid]
        for edge, edge_type in pred["types"].items():
            by_type[(edge_type, "tp" if edge in gold else "fp")] += 1
    type_metrics = {}
    for edge_type in sorted({key[0] for key in by_type}):
        tp = by_type[(edge_type, "tp")]
        fp = by_type[(edge_type, "fp")]
        type_metrics[edge_type] = {
            "tp": tp,
            "fp": fp,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
        }
    return {
        "n_traces": len(rows),
        "selected_order_variants": dict(sorted(Counter(
            predictions[rid]["order_variant"] for rid in common
        ).items())),
        "micro": pooled,
        "macro": macro,
        "trace_bootstrap_95ci": _bootstrap(rows, reps, seed),
        "per_edge_type": type_metrics,
        "counts": {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn")},
        "cycle_counts": {
            "tp": sum(row["cycle_predicted"] and row["cycle_gold"] for row in rows),
            "fp": sum(row["cycle_predicted"] and not row["cycle_gold"] for row in rows),
            "fn": sum(not row["cycle_predicted"] and row["cycle_gold"] for row in rows),
        },
    }


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def _self_check() -> None:
    reversed_ann = {"record_id": "x", "step_order": [2, 1, 0], "edges": [[2, 0]]}
    interleaved_ann = {"record_id": "x", "step_order": [0, 2, 1, 3], "edges": [[2, 1]]}
    assert _canonical_edges(reversed_ann, 3) == {(0, 2)}
    assert _canonical_edges(interleaved_ann, 4) == {(1, 2)}
    pack = {"x": {"record_id": "x", "n_steps": 3}}
    differently_ordered = {
        "a": {"x": {"canonical_edges": _canonical_edges(
            {"record_id": "x", "step_order": [0, 1, 2], "edges": [[0, 2]]}, 3
        )}},
        "b": {"x": {"canonical_edges": _canonical_edges(
            {"record_id": "x", "step_order": [2, 1, 0], "edges": [[2, 0]]}, 3
        )}},
        "c": {"x": {"canonical_edges": _canonical_edges(
            {"record_id": "x", "step_order": [0, 2, 1], "edges": [[0, 1], [2, 1]]}, 3
        )}},
    }
    reference, construction = _majority_reference(differently_ordered, pack, ["a", "b", "c"])
    assert reference == {"x": {(0, 2)}}
    assert construction["n_reference_traces"] == 1
    has_cycle, cycle_edges = _has_cycle(3, {(0, 1), (1, 2), (2, 0)})
    assert has_cycle and cycle_edges == {(0, 1), (1, 2), (2, 0)}
    row = _score_trace(
        "x",
        {"edges": {(0, 2)}, "types": {(0, 2): "t"}, "evaluated_pairs": None, "abstained_pairs": None},
        {(0, 2)},
        {"x": {"n_steps": 3}},
    )
    assert row["direction_correct"] == row["direction_total"] == 1
    cycle_row = _score_trace(
        "x",
        {
            "edges": {(0, 1), (1, 2), (2, 0)},
            "types": {(0, 1): "t", (1, 2): "t", (2, 0): "t"},
            "evaluated_pairs": None,
            "abstained_pairs": None,
        },
        {(0, 1), (1, 2), (2, 0)},
        {"x": {"n_steps": 3}},
    )
    cycle_metrics = _pooled([cycle_row])
    assert cycle_metrics["cycle_f1"] == 1.0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "all_orders.jsonl"
        rows = [
            {
                "record_id": f"x::{variant}",
                "source_record_id": "x",
                "order_variant": variant,
                "edges": [{"source": edge[0], "target": edge[1]}],
            }
            for variant, edge in (
                ("reversed", (2, 0)),
                ("original", (0, 2)),
                ("interleaved", (1, 2)),
            )
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        label, predictions = _load_predictions(f"encoder={path}", pack)
        assert label == "encoder"
        assert predictions["x"]["edges"] == {(0, 2)}
        assert predictions["x"]["order_variant"] == "original"
    print("human edge audit self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=Path("rebuttal/outputs/edge_validation_pack.jsonl"))
    parser.add_argument("--annotation-dir", type=Path, default=Path("rebuttal/outputs/human_edge_annotations"))
    parser.add_argument("--slots", default="a,b,c")
    parser.add_argument("--predictions", action="append", default=["legacy_rules=PACK"])
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("rebuttal/outputs/human_edge_audit.json"))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return

    slots = [slot.strip().lower() for slot in args.slots.split(",") if slot.strip()]
    if len(slots) < 2:
        parser.error("at least two annotator slots are required")
    rows = _read_jsonl(args.pack)
    pack = {row["record_id"]: row for row in rows}
    if len(pack) != len(rows):
        raise ValueError("pack contains duplicate record IDs")
    annotations, warnings = _load_slots(args.annotation_dir, slots, pack)
    completion = {
        slot: {
            "saved": len(annotations[slot]),
            "total": len(pack),
            "segmentation_unusable": sum(
                bool(ann.get("segmentation_unusable")) for ann in annotations[slot].values()
            ),
        }
        for slot in slots
    }
    agreements = {
        f"{left}_vs_{right}": _binary_agreement(annotations[left], annotations[right], pack)
        for left, right in combinations(slots, 2)
    }
    references, reference_construction = _majority_reference(annotations, pack, slots)
    if not references:
        parser.error(
            "no complete, usable annotation overlap across all requested slots; "
            "human audit output was not written"
        )
    result: dict[str, Any] = {
        "schema_version": 1,
        "pack": str(args.pack),
        "n_pack_records": len(pack),
        "slots": slots,
        "completion": completion,
        "warnings": warnings,
        "agreement": agreements,
        "reference_construction": reference_construction,
        "reference_topology": _reference_topology(references, pack),
        "extractors": {},
    }
    if references:
        for prediction_spec in args.predictions:
            label, predictions = _load_predictions(prediction_spec, pack)
            result["extractors"][label] = _score_predictions(
                predictions, references, pack, args.bootstrap_reps, args.seed
            )
    result = _round_floats(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"human edge audit -> {args.out}")


if __name__ == "__main__":
    main()
