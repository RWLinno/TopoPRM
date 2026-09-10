#!/usr/bin/env python3
"""Recompute evaluation metrics from saved responses without regeneration."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

from tokenizers import Tokenizer

from src.eval.unified_benchmark import bootstrap_item_metrics
from src.eval.math_scoring import verify_math_response


BENCHMARKS = (
    "olympiadbench",
    "gpqa_diamond",
    "aime2024",
    "aime2025",
    "aime2026",
    "math500",
    "gsm8k",
    "omni_math",
    "cnmo2024",
    "mmlu",
)
TOKEN_PROTOCOL = "retokenized_decoded_response_no_special_tokens"
MATH_RESPONSE_BENCHMARKS = {
    "gsm8k",
    "math500",
    "olympiadbench",
    "omni_math",
    "aime2024",
    "aime2025",
    "aime2026",
    "cnmo2024",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_item_ids(rows: list[dict], details_path: Path) -> tuple[list[str], str]:
    item_ids = [str(row.get("item_id", "")).strip() for row in rows]
    missing = [index for index, item_id in enumerate(item_ids) if not item_id]
    if missing:
        preview = ", ".join(str(index) for index in missing[:5])
        raise ValueError(f"Missing item_id in {details_path} at row indices: {preview}")
    if len(set(item_ids)) != len(item_ids):
        seen: set[str] = set()
        duplicates: list[str] = []
        for item_id in item_ids:
            if item_id in seen and item_id not in duplicates:
                duplicates.append(item_id)
            seen.add(item_id)
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"Duplicate item_id values in {details_path}: {preview}")
    identity_payload = "\n".join(sorted(item_ids)).encode("utf-8")
    return item_ids, hashlib.sha256(identity_payload).hexdigest()


def _evaluator_source_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    sources = (
        root / "scripts" / "bench_transformers.py",
        root / "src" / "eval" / "unified_benchmark.py",
        root / "src" / "eval" / "math_scoring.py",
        root / "src" / "reward" / "outcome_reward.py",
    )
    return {str(path.relative_to(root)): _sha256_file(path) for path in sources}


def _identity(path: Path) -> tuple[str, str]:
    suffix = "_details.jsonl"
    if not path.name.endswith(suffix):
        raise ValueError(f"Not a details artifact: {path}")
    stem = path.name[: -len(suffix)]
    for benchmark in BENCHMARKS:
        marker = f"_{benchmark}"
        if stem.endswith(marker):
            return stem[: -len(marker)], benchmark
    raise ValueError(f"Cannot infer benchmark from {path.name}")


def _model_default_use_cache(model_path: str) -> bool:
    """Read the generation default used by legacy evaluator runs."""
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return True
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return bool(config.get("use_cache", True))


def _responses(row: dict) -> list[str]:
    values = row.get("responses_all")
    if isinstance(values, list) and values:
        return [str(value) for value in values]
    return [str(row.get("response", ""))]


def _prompt_profile(metrics: dict) -> str:
    if not metrics.get("use_chat_template", False):
        return "bare_text"
    if metrics.get("fold_system_into_user", False):
        return "task_user"
    if metrics.get("empty_system_prompt", False):
        return "empty_system_task_user"
    if metrics.get("system_control", ""):
        return "control_system_task_user"
    if metrics.get("user_suffix", ""):
        return "task_system_user_suffix"
    return "task_system"


def _legacy_response_envelope(rows: list[dict]) -> tuple[str, dict]:
    responses = [_responses(row)[0] for row in rows]
    nonempty = [response for response in responses if response.strip()]
    if not nonempty:
        raise ValueError("Cannot infer response envelope from empty legacy responses")
    starts_with_think = sum(
        response.lstrip().startswith("<think>") for response in nonempty
    )
    if starts_with_think == len(nonempty):
        envelope = "full_think"
    elif starts_with_think == 0:
        envelope = "prefilled_think"
    else:
        raise ValueError(
            "Legacy responses mix full and prefilled <think> envelopes; "
            "manual protocol audit required"
        )
    return envelope, {
        "source": "legacy_saved_response_opening",
        "nonempty_rows": len(nonempty),
        "starts_with_think_rows": starts_with_think,
    }


def _score_math(payload: tuple[str, str]) -> bool:
    response, gold = payload
    return verify_math_response(response, gold)


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def rescore(
    details_path: Path,
    tokenizer_cache: dict[str, Tokenizer],
    workers: int,
) -> dict:
    label, benchmark = _identity(details_path)
    metrics_path = details_path.with_name(f"{label}_{benchmark}_metrics.json")
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in details_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No rows in {details_path}")
    prompt_profile = _prompt_profile(metrics)
    recorded_profile = metrics.get("prompt_profile")
    if recorded_profile is not None and recorded_profile != prompt_profile:
        raise ValueError(
            f"Prompt profile mismatch in {metrics_path}: "
            f"recorded={recorded_profile}, derived={prompt_profile}"
        )
    metrics["prompt_profile"] = prompt_profile
    response_envelope = metrics.get("response_envelope")
    if response_envelope:
        envelope_evidence = metrics.get("provenance", {}).get(
            "response_envelope_evidence"
        )
        if not envelope_evidence:
            inferred_envelope, envelope_evidence = _legacy_response_envelope(rows)
            if inferred_envelope != response_envelope:
                raise ValueError(
                    f"Response envelope mismatch in {metrics_path}: "
                    f"recorded={response_envelope}, responses={inferred_envelope}"
                )
            envelope_evidence["source"] = (
                "recorded_metadata_cross_checked_with_saved_response_opening"
            )
    else:
        response_envelope, envelope_evidence = _legacy_response_envelope(rows)
        metrics["response_envelope"] = response_envelope
    item_ids, item_id_set_sha256 = _validate_item_ids(rows, details_path)
    recorded_n_items = metrics.get("n_items", metrics.get("num_samples"))
    if recorded_n_items is not None and int(recorded_n_items) != len(rows):
        raise ValueError(
            f"Metrics/details row mismatch for {details_path}: "
            f"metrics={recorded_n_items}, details={len(rows)}"
        )

    model_path = str(metrics.get("model", ""))
    if not model_path:
        raise ValueError(f"Missing model path in {metrics_path}")
    if model_path not in tokenizer_cache:
        tokenizer_file = Path(model_path) / "tokenizer.json"
        if not tokenizer_file.is_file():
            raise FileNotFoundError(tokenizer_file)
        tokenizer_cache[model_path] = Tokenizer.from_file(str(tokenizer_file))
    tokenizer = tokenizer_cache[model_path]

    k_values = [int(value) for value in metrics.get("k_values", [1])]
    if k_values != [1] or int(metrics.get("num_samples_per_item", 1)) != 1:
        raise ValueError("Rescoring is intentionally limited to canonical single-response k=1 runs")
    flip_count = 0
    math_flags: list[bool] | None = None
    if benchmark in MATH_RESPONSE_BENCHMARKS:
        payloads = [(_responses(row)[0], str(row["gold"])) for row in rows]
        with mp.Pool(processes=workers) as pool:
            math_flags = pool.map(_score_math, payloads, chunksize=8)

    for row_index, row in enumerate(rows):
        responses = _responses(row)
        if len(responses) != 1:
            raise ValueError(f"Expected one response for {row.get('item_id')}")
        flags = [
            math_flags[row_index]
            if benchmark in MATH_RESPONSE_BENCHMARKS
            else bool(row.get("correct_pass1", False))
        ]
        counts = [
            len(tokenizer.encode(response, add_special_tokens=False).ids)
            for response in responses
        ]
        previous = bool(
            row.get("previous_correct_pass1", row.get("correct_pass1", False))
        )
        current = bool(flags[0]) if flags else False
        if previous != current:
            row["previous_correct_pass1"] = previous
            flip_count += 1
        else:
            row.pop("previous_correct_pass1", None)
        row["correct_pass1"] = current
        row["correct_count"] = sum(flags)
        row["gen_tokens_pass1"] = counts[0] if counts else 0
        row["avg_gen_tokens"] = round(sum(counts) / max(len(counts), 1), 1)
    bootstrap = bootstrap_item_metrics(rows, n_resamples=10_000, seed=0)
    prior_audit = metrics.get("rescore_audit", {})
    previous_accuracy = float(
        prior_audit.get("previous_accuracy_pct", metrics.get("accuracy_pct", 0.0))
    )
    previous_tokens = float(
        prior_audit.get(
            "previous_mean_tokens_pass1",
            metrics.get("mean_tokens_pass1", metrics.get("avg_tokens", 0.0)),
        )
    )

    metrics["pass@1"] = round(sum(row["correct_pass1"] for row in rows) / len(rows), 4)
    metrics["maj@1"] = metrics["pass@1"]
    metrics["pass_at_k"] = {"1": metrics["pass@1"]}
    metrics["maj_at_k"] = {"1": metrics["maj@1"]}
    metrics["n_items"] = len(rows)
    metrics["avg_tokens"] = round(
        sum(row["gen_tokens_pass1"] for row in rows) / len(rows), 1
    )
    metrics["accuracy"] = metrics["pass@1"]
    metrics["accuracy_pct"] = round(100 * metrics["pass@1"], 2)
    metrics["correct"] = sum(row["correct_pass1"] for row in rows)
    metrics["error"] = len(rows) - metrics["correct"]
    # Keep the legacy aliases internally consistent with canonical pass@1.
    # These fields predate raw-response math scoring and otherwise retain the
    # stale extractor-based counts from the original evaluation.
    metrics["correct_count"] = metrics["correct"]
    metrics["error_count"] = metrics["error"]
    metrics["precision"] = metrics["pass@1"]
    metrics["recall"] = metrics["pass@1"]
    metrics["f1"] = metrics["pass@1"]
    metrics["num_samples"] = len(rows)
    metrics["bootstrap_n_resamples"] = bootstrap["n_resamples"]
    metrics["bootstrap_seed"] = bootstrap["seed"]
    metrics["accuracy_pct_ci95"] = [
        round(value, 4) for value in bootstrap["accuracy_pct_ci95"]
    ]
    metrics["accuracy_pct_wilson95"] = [
        round(value, 4) for value in bootstrap["accuracy_pct_wilson95"]
    ]
    metrics["mean_tokens_pass1"] = round(bootstrap["mean_tokens"], 4)
    metrics["mean_tokens_pass1_ci95"] = [
        round(value, 4) for value in bootstrap["mean_tokens_ci95"]
    ]
    metrics["interval_scope"] = "evaluation_items_not_training_variance"
    model_default_use_cache = _model_default_use_cache(model_path)
    metrics.setdefault("model_default_use_cache", model_default_use_cache)
    if "kv_cache_enabled" not in metrics:
        # Evaluator versions before the explicit cache switch omitted
        # ``use_cache`` from model.generate, so the checkpoint default was the
        # effective runtime value. Record that inference rather than silently
        # treating the artifact as if it used the current explicit default.
        metrics["kv_cache_enabled"] = model_default_use_cache
        metrics["kv_cache_provenance"] = (
            "legacy_generation_inferred_from_model_config_default"
        )
    else:
        metrics.setdefault("kv_cache_provenance", "explicit_generation_argument")
    metrics["provenance_schema_version"] = 3
    metrics["visible_token_counting"] = TOKEN_PROTOCOL
    metrics["math_scoring_protocol"] = (
        "last_nonempty_box_else_explicit_final_answer_math_verify_gold_first"
        if benchmark in MATH_RESPONSE_BENCHMARKS
        else "extracted_answer_matcher"
    )
    provenance = metrics.setdefault("provenance", {})
    provenance["schema_version"] = 3
    provenance.setdefault(
        "generation_evaluator_sources", provenance.get("evaluator_sources", {})
    )
    provenance["rescoring_sources"] = _evaluator_source_fingerprints()
    nested_profile = provenance.get("prompt_profile")
    nested_envelope = provenance.get("response_envelope")
    if nested_profile is not None and nested_profile != prompt_profile:
        raise ValueError(f"Nested prompt profile mismatch in {metrics_path}")
    if nested_envelope is not None and nested_envelope != response_envelope:
        raise ValueError(f"Nested response envelope mismatch in {metrics_path}")
    provenance["prompt_profile"] = prompt_profile
    provenance["response_envelope"] = response_envelope
    provenance["response_envelope_evidence"] = envelope_evidence
    metrics["rescore_audit"] = {
        "source": details_path.name,
        "rows": len(rows),
        "unique_item_ids": len(item_ids),
        "item_id_set_sha256": item_id_set_sha256,
        "previous_accuracy_pct": previous_accuracy,
        "current_accuracy_pct": metrics["accuracy_pct"],
        "previous_mean_tokens_pass1": previous_tokens,
        "current_mean_tokens_pass1": metrics["mean_tokens_pass1"],
        "correctness_flips": flip_count,
    }

    _write_jsonl_atomic(details_path, rows)
    metrics["rescore_audit"]["details_sha256"] = _sha256_file(details_path)
    _write_json_atomic(metrics_path, metrics)
    return metrics["rescore_audit"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("details", nargs="+", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    tokenizer_cache: dict[str, Tokenizer] = {}
    for details_path in args.details:
        audit = rescore(details_path.resolve(), tokenizer_cache, args.workers)
        print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
