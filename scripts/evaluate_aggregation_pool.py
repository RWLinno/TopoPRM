#!/usr/bin/env python3
"""Evaluate reward aggregators on one frozen multi-candidate rollout pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]
SOURCE_NAMES = ("outcome", "format", "direction", "acyclicity", "continuity")
RELEVANT_ENV = (
    "TOPO_DAG_RAW_DIRECTED",
    "TOPO_DAG_EDGE_CHECKPOINT",
    "TOPO_DAG_EDGE_MODEL",
    "TOPO_DAG_EDGE_REQUIRED",
    "TOPO_DAG_EDGE_MAX_STEPS",
    "TOPO_DAG_SENTENCE_FALLBACK",
    "TOPO_DAG_SENTENCE_MIN_LEN",
    "TOPO_DAG_EXTRA_STEP_MARKERS",
    "TOPO_DAG_LATEX_EXPR",
    "TOPO_DAG_BARRIER_STRICT",
    "TOPO_DAG_BARRIER_MIN_OVERLAP",
    "TOPO_DAG_FILTER_FORMATTING",
    "TOPO_DAG_SEQ_WHEN_NO_DEP_ONLY",
    "TOPO_CONT_REQUIRE_EVIDENCE",
    "TOPO_FORMAT_PROTOCOL",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield value


def _load_env(path: Path) -> None:
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ValueError(f"Unsupported env line {path}:{line_number}: {raw}")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if not key or any(char.isspace() for char in key):
            raise ValueError(f"Invalid env key at {path}:{line_number}")
        os.environ[key] = value


def _require_protocol(metrics: dict, args: argparse.Namespace) -> None:
    expected = {
        "num_samples_per_item": args.candidates,
        "max_new_tokens": args.max_new_tokens,
        "prompt_profile": args.prompt_profile,
        "response_envelope": args.response_envelope,
        "eval_seed": args.eval_seed,
    }
    mismatches = {
        key: {"expected": value, "observed": metrics.get(key)}
        for key, value in expected.items()
        if metrics.get(key) != value
    }
    if metrics.get("pass1_do_sample") is not True:
        mismatches["pass1_do_sample"] = {
            "expected": True,
            "observed": metrics.get("pass1_do_sample"),
        }
    if mismatches:
        raise ValueError(f"Candidate-pool protocol mismatch: {mismatches}")
    sampling = metrics.get("item_sampling", {})
    expected_sampling = {
        "method": "uniform_without_replacement",
        "sample_items": args.item_sample_size,
        "seed": args.item_sample_seed,
    }
    sampling_mismatches = {
        key: {"expected": value, "observed": sampling.get(key)}
        for key, value in expected_sampling.items()
        if sampling.get(key) != value
    }
    if sampling_mismatches or not sampling.get("item_id_set_sha256"):
        raise ValueError(
            f"Candidate-pool item sampling mismatch: {sampling_mismatches}"
        )
    if metrics.get("provenance_schema_version", 0) < 3:
        raise ValueError("Candidate pool requires provenance schema version >= 3")


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(
    values: list[float], *, reps: int, seed: int, scale: float = 1.0
) -> dict[str, object]:
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    rng = random.Random(seed)
    n = len(values)
    draws = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(reps)
    ]
    return {
        "n": n,
        "mean": round(scale * statistics.fmean(values), 6),
        "ci95": [
            round(scale * _percentile(draws, 0.025), 6),
            round(scale * _percentile(draws, 0.975), 6),
        ],
    }


def _pairwise_auc(scores: list[float], correct: list[bool]) -> float | None:
    positive = [index for index, flag in enumerate(correct) if flag]
    negative = [index for index, flag in enumerate(correct) if not flag]
    if not positive or not negative:
        return None
    wins = 0.0
    total = 0
    for pos in positive:
        for neg in negative:
            total += 1
            if scores[pos] > scores[neg]:
                wins += 1.0
            elif math.isclose(scores[pos], scores[neg], abs_tol=1e-12):
                wins += 0.5
    return wins / total


def _fingerprint_artifact(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    records = []
    for item in files:
        relative = item.name if path.is_file() else str(item.relative_to(path))
        size = item.stat().st_size
        record: dict[str, object] = {"path": relative, "bytes": size}
        if size <= 256 * 1024 * 1024:
            record["sha256"] = _sha256(item)
        records.append(record)
    manifest = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "path": str(path),
        "files": records,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }


def _validate_row(row: dict, candidates: int, seen: set[str]) -> None:
    item_id = str(row.get("item_id", ""))
    if not item_id or item_id in seen:
        raise ValueError(f"Missing or duplicate item_id: {item_id!r}")
    seen.add(item_id)
    arrays = {
        "responses_all": row.get("responses_all"),
        "correct_flags_all": row.get("correct_flags_all"),
        "gen_tokens_all": row.get("gen_tokens_all"),
    }
    bad = {key: len(value) if isinstance(value, list) else None for key, value in arrays.items() if not isinstance(value, list) or len(value) != candidates}
    if bad:
        raise ValueError(f"Candidate arrays misaligned for {item_id}: {bad}")
    if row.get("num_samples") != candidates:
        raise ValueError(f"num_samples mismatch for {item_id}")
    flags = arrays["correct_flags_all"]
    if any(not isinstance(flag, bool) for flag in flags):
        raise ValueError(f"Non-boolean correctness flag for {item_id}")
    if row.get("correct_count") != sum(flags):
        raise ValueError(f"correct_count mismatch for {item_id}")
    if not str(row.get("gold", "")).strip():
        raise ValueError(f"Missing gold answer for {item_id}")


def _method_summary(rows: list[dict], method: str, reps: int, seed: int) -> dict:
    def values(name: str) -> list[float]:
        return [float(row[name]) for row in rows if row.get(name) is not None]

    return {
        "selection_accuracy_pct": _summary(values("selected_correct"), reps=reps, seed=seed, scale=100.0),
        "tie_averaged_accuracy_pct": _summary(values("tie_accuracy"), reps=reps, seed=seed + 1, scale=100.0),
        "oracle_accuracy_pct": _summary(values("oracle_correct"), reps=reps, seed=seed + 2, scale=100.0),
        "pairwise_correctness_auc": _summary(values("pairwise_auc"), reps=reps, seed=seed + 3),
        "structured_wrong_selection_pct": _summary(values("structured_wrong"), reps=reps, seed=seed + 4, scale=100.0),
        "selected_direction": _summary(values("selected_direction"), reps=reps, seed=seed + 5),
        "selected_acyclicity": _summary(values("selected_acyclicity"), reps=reps, seed=seed + 6),
        "selected_tokens": _summary(values("selected_tokens"), reps=reps, seed=seed + 7),
        "tie_group_pct": _summary(values("tie_group"), reps=reps, seed=seed + 8, scale=100.0),
        "zero_spread_group_pct": _summary(values("zero_spread"), reps=reps, seed=seed + 9, scale=100.0),
        "method": method,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--capacity-profile", choices=("balanced", "structure_forward"), required=True)
    parser.add_argument("--candidates", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--prompt-profile", required=True)
    parser.add_argument("--response-envelope", required=True)
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument("--item-sample-size", type=int, required=True)
    parser.add_argument("--item-sample-seed", type=int, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--structured-threshold", type=float, default=0.8)
    parser.add_argument("--score-batch-items", type=int, default=16)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scored-output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.candidates < 2 or args.bootstrap_reps < 1000:
        raise ValueError("Formal pool evaluation requires >=2 candidates and >=1000 bootstrap reps")
    if args.item_sample_size < 1 or args.item_sample_seed < 0:
        raise ValueError("Formal pool evaluation requires an explicit sampled item set")
    if args.score_batch_items < 1:
        raise ValueError("score batch must contain at least one item")
    if not 0.0 <= args.structured_threshold <= 1.0:
        raise ValueError("structured threshold must lie in [0, 1]")
    for output in (args.output, args.scored_output):
        if output.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite {output}; pass --force")

    _load_env(args.env_file)
    if os.environ.get("TOPO_DAG_EDGE_REQUIRED") != "1":
        raise ValueError("TOPO_DAG_EDGE_REQUIRED=1 is mandatory")
    if os.environ.get("TOPO_DAG_RAW_DIRECTED") != "1":
        raise ValueError("TOPO_DAG_RAW_DIRECTED=1 is mandatory")
    if os.environ.get("TOPO_FORMAT_PROTOCOL") != args.response_envelope:
        raise ValueError("format protocol and response envelope must match")

    metrics = _read_json(args.metrics)
    _require_protocol(metrics, args)
    if metrics.get("n_items") is None:
        raise ValueError("Metrics file lacks n_items")

    from src.reward.composite_reward import (
        TopoIndependentChoquetReward,
        TopoIndependentEqualAdditiveReward,
        TopoIndependentMatchedAdditiveReward,
        TopoIndependentMatchedMultiplicativeReward,
    )

    full = TopoIndependentChoquetReward(capacity_profile=args.capacity_profile)
    aggregators = {
        "outcome_only": None,
        "equal_additive": TopoIndependentEqualAdditiveReward(capacity_profile=args.capacity_profile),
        "matched_additive": TopoIndependentMatchedAdditiveReward(capacity_profile=args.capacity_profile),
        "matched_multiplicative": TopoIndependentMatchedMultiplicativeReward(capacity_profile=args.capacity_profile),
        "topology_conditional_choquet": full,
    }
    previous = os.environ.get("TOPO_DISABLE_DIRECTION_ACYCLICITY_INTERACTION")
    os.environ["TOPO_DISABLE_DIRECTION_ACYCLICITY_INTERACTION"] = "1"
    aggregators["choquet_without_direction_acyclicity_interaction"] = (
        TopoIndependentChoquetReward(capacity_profile=args.capacity_profile)
    )
    if previous is None:
        os.environ.pop("TOPO_DISABLE_DIRECTION_ACYCLICITY_INTERACTION", None)
    else:
        os.environ["TOPO_DISABLE_DIRECTION_ACYCLICITY_INTERACTION"] = previous

    seen: set[str] = set()
    source_rows: list[dict] = []
    method_rows: dict[str, list[dict]] = {name: [] for name in aggregators}
    outcome_disagreements = 0
    total_candidates = 0
    input_rows = list(_iter_jsonl(args.details))
    if args.max_items:
        input_rows = input_rows[: args.max_items]
    for row in input_rows:
        _validate_row(row, args.candidates, seen)
    for start in range(0, len(input_rows), args.score_batch_items):
        batch_rows = input_rows[start : start + args.score_batch_items]
        completions = [
            [{"role": "assistant", "content": response}]
            for row in batch_rows
            for response in row["responses_all"]
        ]
        solutions = [
            row["gold"]
            for row in batch_rows
            for _ in range(args.candidates)
        ]
        flat_sources = full.score_sources(completions, solution=solutions)
        expected_candidates = len(batch_rows) * args.candidates
        if set(flat_sources) != set(SOURCE_NAMES) or any(
            len(values) != expected_candidates for values in flat_sources.values()
        ):
            raise RuntimeError(f"Misaligned source scores for item batch starting at {start}")
        for offset, row in enumerate(batch_rows):
            left, right = offset * args.candidates, (offset + 1) * args.candidates
            sources = {name: flat_sources[name][left:right] for name in SOURCE_NAMES}
            correct = row["correct_flags_all"]
            tokens = row["gen_tokens_all"]
            outcome_disagreements += sum(
                (score >= 0.5) != flag
                for score, flag in zip(sources["outcome"], correct)
            )
            total_candidates += args.candidates
            audit_row = {
                "item_id": row["item_id"],
                "correct_flags": correct,
                "gen_tokens": tokens,
                "sources": sources,
                "methods": {},
            }
            for method, aggregator in aggregators.items():
                scores = (
                    list(sources["outcome"])
                    if aggregator is None
                    else [
                        aggregator.aggregate_values(
                            {name: sources[name][index] for name in SOURCE_NAMES}
                        )
                        for index in range(args.candidates)
                    ]
                )
                best = max(scores)
                ties = [index for index, score in enumerate(scores) if math.isclose(score, best, abs_tol=1e-12)]
                selected = ties[0]
                auc = _pairwise_auc(scores, correct)
                result = {
                    "selected_correct": float(correct[selected]),
                    "tie_accuracy": statistics.fmean(float(correct[index]) for index in ties),
                    "oracle_correct": float(any(correct)),
                    "pairwise_auc": auc,
                    "structured_wrong": float(
                        not correct[selected]
                        and sources["direction"][selected] >= args.structured_threshold
                        and sources["acyclicity"][selected] >= args.structured_threshold
                    ),
                    "selected_direction": sources["direction"][selected],
                    "selected_acyclicity": sources["acyclicity"][selected],
                    "selected_tokens": tokens[selected],
                    "tie_group": float(len(ties) > 1),
                    "zero_spread": float(statistics.pstdev(scores) <= 1e-12),
                }
                method_rows[method].append(result)
                audit_row["methods"][method] = {
                    "scores": [round(value, 8) for value in scores],
                    "selected_index": selected,
                    "tie_indices": ties,
                }
            source_rows.append(audit_row)

    expected_rows = int(metrics["n_items"])
    if args.max_items == 0 and len(source_rows) != expected_rows:
        raise ValueError(f"Details/metrics row mismatch: {len(source_rows)} != {expected_rows}")
    if not source_rows:
        raise ValueError("Candidate pool is empty")
    item_id_set_sha256 = hashlib.sha256(
        "\n".join(sorted(seen)).encode()
    ).hexdigest()
    recorded_item_hash = metrics["item_sampling"]["item_id_set_sha256"]
    if args.max_items == 0 and item_id_set_sha256 != recorded_item_hash:
        raise ValueError(
            "Sampled item-id set does not match metrics provenance: "
            f"{item_id_set_sha256} != {recorded_item_hash}"
        )

    source_summary = {
        name: _summary(
            [value for row in source_rows for value in row["sources"][name]],
            reps=args.bootstrap_reps,
            seed=args.bootstrap_seed + index,
        )
        for index, name in enumerate(SOURCE_NAMES)
    }
    output = {
        "schema_version": 1,
        "formal_candidate_pool": args.max_items == 0,
        "benchmark": args.benchmark,
        "capacity_profile": args.capacity_profile,
        "n_items": len(source_rows),
        "candidates_per_item": args.candidates,
        "score_batch_items": args.score_batch_items,
        "structured_wrong_threshold": args.structured_threshold,
        "interval_scope": "item_bootstrap_not_training_variance",
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed,
        "outcome_eval_disagreement_pct": round(100.0 * outcome_disagreements / total_candidates, 6),
        "source_summary": source_summary,
        "methods": {
            method: _method_summary(rows, method, args.bootstrap_reps, args.bootstrap_seed + 100 * index)
            for index, (method, rows) in enumerate(method_rows.items())
        },
        "protocol": {
            key: metrics.get(key)
            for key in (
                "label", "model", "adapter", "num_samples_per_item", "max_new_tokens",
                "prompt_profile", "response_envelope", "eval_seed", "temperature",
                "top_p", "top_k", "min_p", "repetition_penalty", "pass1_do_sample",
            )
        },
        "provenance": {
            "details": {"path": str(args.details), "sha256": _sha256(args.details)},
            "metrics": {"path": str(args.metrics), "sha256": _sha256(args.metrics)},
            "env_file": {"path": str(args.env_file), "sha256": _sha256(args.env_file)},
            "item_id_set_sha256": item_id_set_sha256,
            "edge_checkpoint": _fingerprint_artifact(Path(os.environ["TOPO_DAG_EDGE_CHECKPOINT"])),
            "edge_model": _fingerprint_artifact(Path(os.environ["TOPO_DAG_EDGE_MODEL"])),
            "source_files": {
                str(path.relative_to(REPO)): _sha256(path)
                for path in (
                    Path(__file__).resolve(),
                    REPO / "src" / "reward" / "composite_reward.py",
                    REPO / "src" / "reward" / "topo_reward.py",
                    REPO / "src" / "capacity_profiles.py",
                    REPO / "src" / "data" / "build_dag.py",
                )
            },
            "environment": {key: os.environ.get(key) for key in RELEVANT_ENV},
        },
    }
    output["protocol"]["item_sampling"] = metrics["item_sampling"]
    for path in (args.output, args.scored_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.scored_output.open("w", encoding="utf-8") as handle:
        for row in source_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "scored_output": str(args.scored_output),
        "n_items": len(source_rows),
        "details_sha256": output["provenance"]["details"]["sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
