#!/usr/bin/env python3
"""Unified benchmark evaluation framework with multi-sample metrics.

Supports:
  - Multiple benchmarks via registry (GSM8K, MATH-500, OlympiadBench, etc.)
  - Configurable pass@1 decoding + sampled pass@k, maj@k, prm@k
  - Token counting
  - Error/Correct/F1 computation
  - Standardized metric JSON output

This module is additive ? it does NOT replace bench_transformers.py or
run_public_benchmarks.sh. Those continue to work for quick single-benchmark runs.

Usage:
    python -m src.eval.unified_benchmark \\
        --model /path/to/model \\
        --adapter /path/to/adapter \\
        --label topoprm_hier_9b \\
        --benchmarks gsm8k math500 \\
        --k 1 5 \\
        --output_dir output/eval_unified
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Benchmark Registry
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
        "split": "test",
        "question_key": "question",
        "answer_key": "answer",
        "extract_gold": "_extract_gsm8k_gold",
        "prompt_template": "gsm8k",
        "answer_extractor": "numeric",
    },
    "math500": {
        "hf_path": "HuggingFaceH4/MATH-500",
        "hf_name": None,
        "split": "test",
        "question_key": "problem",
        "answer_key": "answer",
        "extract_gold": "_extract_math_gold",
        "prompt_template": "math",
        "answer_extractor": "boxed_or_numeric",
    },
    "olympiadbench": {
        "hf_path": "lmms-lab/OlympiadBench",
        "hf_name": None,
        "split": "test_en",
        "question_key": "question",
        "answer_key": "final_answer",
        "extract_gold": "_extract_plain_gold",
        "prompt_template": "math",
        "answer_extractor": "boxed_or_numeric",
    },
    "omni_math": {
        "hf_path": "KbsdJames/Omni-MATH",
        "hf_name": None,
        "split": "test",
        "question_key": "problem",
        "answer_key": "answer",
        "extract_gold": "_extract_plain_gold",
        "prompt_template": "math",
        "answer_extractor": "boxed_or_numeric",
    },
    "aime2024": {
        "hf_path": "AI-MO/aimo-validation-aime",
        "hf_name": None,
        "split": "train",
        "question_key": "problem",
        "answer_key": "answer",
        "extract_gold": "_extract_plain_gold",
        "prompt_template": "math",
        "answer_extractor": "numeric",
    },
    "cnmo2024": {
        "local_path": "data/benchmarks/cnmo2024.jsonl",
        "question_key": "problem",
        "answer_key": "answer",
        "extract_gold": "_extract_plain_gold",
        "prompt_template": "math_zh",
        "answer_extractor": "boxed_or_numeric",
    },
    # LiveCode dropped 2026-04-21: math-only models always score ~0%.
    "mmlu": {
        "hf_path": "cais/mmlu",
        "hf_name": "all",
        "split": "test",
        "question_key": "question",
        "answer_key": "answer",
        "extract_gold": "_extract_mcq_gold",
        "prompt_template": "mcq",
        "answer_extractor": "mcq",
    },
    "gpqa_diamond": {
        "hf_path": "Idavidrein/gpqa",
        "hf_name": "gpqa_diamond",
        "split": "train",
        "question_key": "Question",
        "answer_key": "Correct Answer",
        "extract_gold": "_extract_plain_gold",
        "prompt_template": "mcq",
        "answer_extractor": "mcq",
    },
}


# ---------------------------------------------------------------------------
# Gold answer extractors
# ---------------------------------------------------------------------------

def _extract_gsm8k_gold(row: dict) -> str:
    m = re.search(r"####\s*(.+)", str(row.get("answer", "")))
    return m.group(1).strip() if m else str(row.get("answer", "")).strip()

def _extract_math_gold(row: dict) -> str:
    ans = str(row.get("answer", row.get("solution", "")))
    m = re.search(r"\\boxed\{([^}]+)\}", ans)
    return m.group(1).strip() if m else ans.strip()

def _extract_plain_gold(row: dict) -> str:
    return str(row.get("answer", row.get("final_answer", ""))).strip()

def _extract_mcq_gold(row: dict) -> str:
    ans = row.get("answer", "")
    if isinstance(ans, int):
        return chr(ord("A") + ans)
    return str(ans).strip()

def _extract_code_gold(row: dict) -> str:
    return str(row.get("test", row.get("answer", ""))).strip()


GOLD_EXTRACTORS: dict[str, Callable] = {
    "_extract_gsm8k_gold": _extract_gsm8k_gold,
    "_extract_math_gold": _extract_math_gold,
    "_extract_plain_gold": _extract_plain_gold,
    "_extract_mcq_gold": _extract_mcq_gold,
    "_extract_code_gold": _extract_code_gold,
}


# ---------------------------------------------------------------------------
# Answer extraction from model output
# ---------------------------------------------------------------------------

def extract_numeric(text: str) -> Optional[str]:
    m = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"[+-]?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def extract_boxed_or_numeric(text: str) -> Optional[str]:
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    return extract_numeric(text)


def extract_mcq(text: str) -> Optional[str]:
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1) if m else None


def extract_code(text: str) -> Optional[str]:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


ANSWER_EXTRACTORS: dict[str, Callable] = {
    "numeric": extract_numeric,
    "boxed_or_numeric": extract_boxed_or_numeric,
    "mcq": extract_mcq,
    "code": extract_code,
}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES: dict[str, str] = {
    "gsm8k": (
        "Solve the following math problem step by step. "
        "Put your final answer after ####.\n\n"
        "Question: {question}\n\nAnswer:"
    ),
    "math": (
        "Solve the following math problem. "
        "Put your final answer in \\boxed{{}}.\n\n"
        "Problem: {question}\n\nSolution:"
    ),
    "math_zh": (
        "???????????????? \\boxed{{}} ??\n\n"
        "???{question}\n\n??"
    ),
    "mcq": (
        "Answer the following question by choosing A, B, C, or D.\n\n"
        "{question}\n\nAnswer:"
    ),
    "code": (
        "Solve the following programming problem. "
        "Output your solution in a Python code block.\n\n"
        "{question}\n\nSolution:"
    ),
}


# ---------------------------------------------------------------------------
# Normalization & comparison
# ---------------------------------------------------------------------------

def normalize_answer(ans: str) -> str:
    ans = ans.strip().replace(",", "").replace("$", "").replace("%", "")
    if ans.endswith("."):
        ans = ans[:-1]
    return ans.lower()


def answers_match(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    return normalize_answer(pred) == normalize_answer(gold)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator for pass@k (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_maj_at_k(
    predictions: list[Optional[str]],
    gold: str,
    k: int,
    answer_matcher: Callable[[Optional[str], str], bool] = answers_match,
) -> float:
    """Majority voting accuracy with equivalence-aware answer classes."""
    if not predictions or k <= 0:
        return 0.0
    subset = predictions[:k]
    classes: list[tuple[Optional[str], int]] = []
    for prediction in subset:
        for class_index, (representative, count) in enumerate(classes):
            if prediction is None or representative is None:
                equivalent = prediction is None and representative is None
            else:
                equivalent = answer_matcher(prediction, representative)
            if equivalent:
                classes[class_index] = (representative, count + 1)
                break
        else:
            classes.append((prediction, 1))

    majority = max(classes, key=lambda item: item[1])[0]
    return 1.0 if answer_matcher(majority, gold) else 0.0


def compute_error_correct_f1(
    per_item: list[dict],
) -> dict[str, float]:
    """Compute error/correct/F1 from a list of items with 'correct' and 'has_error_tag' keys."""
    tp = sum(1 for it in per_item if it.get("correct") and not it.get("has_error"))
    fp = sum(1 for it in per_item if it.get("correct") and it.get("has_error"))
    fn = sum(1 for it in per_item if not it.get("correct") and not it.get("has_error"))
    tn = sum(1 for it in per_item if not it.get("correct") and it.get("has_error"))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "error_count": fp + tn,
        "correct_count": tp + fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop (designed for integration, not standalone generation)
# ---------------------------------------------------------------------------

def evaluate_predictions(
    predictions_per_item: list[list[str]],
    gold_answers: list[str],
    answer_extractor: Callable,
    k_values: list[int],
    token_counts: Optional[list[list[int]]] = None,
    answer_matcher: Callable[[Optional[str], str], bool] = answers_match,
    raw_answer_matcher: Optional[Callable[[str, str], bool]] = None,
) -> dict[str, Any]:
    """Compute unified metrics from pre-generated predictions.

    Args:
        predictions_per_item: [n_items][n_samples] raw model outputs
        gold_answers: [n_items] gold answer strings
        answer_extractor: function to extract answer from model output
        k_values: list of k for pass@k, maj@k
        token_counts: optional [n_items][n_samples] token counts
        raw_answer_matcher: optional scorer that consumes the complete response
    """
    n_items = len(predictions_per_item)
    assert len(gold_answers) == n_items

    per_item_results = []
    pass_at_k_accum = {k: 0.0 for k in k_values}
    maj_at_k_accum = {k: 0.0 for k in k_values}
    total_tokens = 0
    total_samples = 0

    for i in range(n_items):
        preds_raw = predictions_per_item[i]
        gold = gold_answers[i]
        n_samples = len(preds_raw)

        extracted = [answer_extractor(p) for p in preds_raw]
        correct_flags = (
            [raw_answer_matcher(raw, gold) for raw in preds_raw]
            if raw_answer_matcher is not None
            else [answer_matcher(e, gold) for e in extracted]
        )
        n_correct = sum(correct_flags)

        item_result = {
            "gold": gold,
            "n_samples": n_samples,
            "n_correct": n_correct,
            "correct": correct_flags[0] if correct_flags else False,
            "has_error": not (correct_flags[0] if correct_flags else False),
        }

        for k in k_values:
            if k <= n_samples:
                pass_at_k_accum[k] += compute_pass_at_k(n_samples, n_correct, k)
                if raw_answer_matcher is not None and k == 1:
                    maj_at_k_accum[k] += float(bool(correct_flags[0]))
                else:
                    maj_at_k_accum[k] += compute_maj_at_k(
                        extracted, gold, k, answer_matcher=answer_matcher
                    )

        if token_counts and i < len(token_counts):
            item_tokens = sum(token_counts[i])
            total_tokens += item_tokens
            total_samples += len(token_counts[i])
            item_result["avg_tokens"] = item_tokens / max(len(token_counts[i]), 1)

        per_item_results.append(item_result)

    metrics: dict[str, Any] = {
        "n_items": n_items,
    }

    ecf = compute_error_correct_f1(per_item_results)
    metrics.update(ecf)

    for k in k_values:
        metrics[f"pass@{k}"] = round(pass_at_k_accum[k] / max(n_items, 1), 4)
        metrics[f"maj@{k}"] = round(maj_at_k_accum[k] / max(n_items, 1), 4)

    if total_samples > 0:
        metrics["avg_tokens"] = round(total_tokens / total_samples, 1)

    return metrics


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def wilson_accuracy_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> list[float]:
    """Return a non-degenerate Wilson 95% interval in percentage points."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("successes must lie in [0, total] with total positive")
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    return [100.0 * max(0.0, center - margin), 100.0 * min(1.0, center + margin)]


def bootstrap_item_metrics(
    items: list[dict[str, Any]],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap pass@1 accuracy and mean pass@1 completion length."""
    if not items or n_resamples <= 0:
        raise ValueError("items and n_resamples must be non-empty and positive")
    correct = [float(bool(item["correct_pass1"])) for item in items]
    tokens = [float(item["gen_tokens_pass1"]) for item in items]
    rng = random.Random(seed)
    n_items = len(items)
    accuracy_samples = []
    token_samples = []
    for _ in range(n_resamples):
        indices = [rng.randrange(n_items) for _ in range(n_items)]
        accuracy_samples.append(sum(correct[i] for i in indices) / n_items)
        token_samples.append(sum(tokens[i] for i in indices) / n_items)
    successes = int(sum(correct))
    return {
        "n_items": n_items,
        "n_resamples": n_resamples,
        "seed": seed,
        "accuracy_pct": 100 * sum(correct) / n_items,
        "accuracy_pct_wilson95": wilson_accuracy_interval(successes, n_items),
        "accuracy_pct_ci95": [
            100 * _percentile(accuracy_samples, 0.025),
            100 * _percentile(accuracy_samples, 0.975),
        ],
        "mean_tokens": sum(tokens) / n_items,
        "mean_tokens_ci95": [
            _percentile(token_samples, 0.025),
            _percentile(token_samples, 0.975),
        ],
    }


def paired_bootstrap_difference(
    candidate_items: list[dict[str, Any]],
    baseline_items: list[dict[str, Any]],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired bootstrap deltas on shared item IDs (candidate minus baseline)."""
    if not candidate_items or n_resamples <= 0:
        raise ValueError("items and n_resamples must be non-empty and positive")
    candidate = {str(item["item_id"]): item for item in candidate_items}
    baseline = {str(item["item_id"]): item for item in baseline_items}
    if len(candidate) != len(candidate_items) or len(baseline) != len(baseline_items):
        raise ValueError("item_id values must be unique within each evaluation")
    if set(candidate) != set(baseline):
        raise ValueError("Paired bootstrap requires identical item_id sets")

    item_ids = sorted(candidate)
    accuracy_delta = [
        float(bool(candidate[item_id]["correct_pass1"]))
        - float(bool(baseline[item_id]["correct_pass1"]))
        for item_id in item_ids
    ]
    token_delta = [
        float(candidate[item_id]["gen_tokens_pass1"])
        - float(baseline[item_id]["gen_tokens_pass1"])
        for item_id in item_ids
    ]
    rng = random.Random(seed)
    n_items = len(item_ids)
    accuracy_samples = []
    token_samples = []
    for _ in range(n_resamples):
        indices = [rng.randrange(n_items) for _ in range(n_items)]
        accuracy_samples.append(sum(accuracy_delta[i] for i in indices) / n_items)
        token_samples.append(sum(token_delta[i] for i in indices) / n_items)
    return {
        "n_items": n_items,
        "n_resamples": n_resamples,
        "seed": seed,
        "accuracy_delta_pp": 100 * sum(accuracy_delta) / n_items,
        "accuracy_delta_pp_ci95": [
            100 * _percentile(accuracy_samples, 0.025),
            100 * _percentile(accuracy_samples, 0.975),
        ],
        "mean_token_delta": sum(token_delta) / n_items,
        "mean_token_delta_ci95": [
            _percentile(token_samples, 0.025),
            _percentile(token_samples, 0.975),
        ],
    }


# ---------------------------------------------------------------------------
# CLI (placeholder ? full generation loop to be wired with bench_transformers)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified benchmark evaluation framework.")
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "math500"])
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--output_dir", default="output/eval_unified")
    parser.add_argument("--list_benchmarks", action="store_true")
    args = parser.parse_args()

    if args.list_benchmarks:
        print("Available benchmarks:")
        for name, cfg in BENCHMARK_REGISTRY.items():
            src = cfg.get("hf_path", cfg.get("local_path", "N/A"))
            print(f"  {name:20s} -> {src}")
        return

    print(f"Benchmarks: {args.benchmarks}")
    print(f"k values: {args.k}")
    print(f"Output: {args.output_dir}")
    print()
    print("NOTE: This module provides the metric computation framework.")
    print("For full end-to-end generation + evaluation, use todo_exp_ours.sh")
    print("which wires this with bench_transformers.py for model inference.")


if __name__ == "__main__":
    main()
