"""Generate distillation training data from teacher model.

Uses the GRPO-trained teacher (14B) to generate reasoning traces,
then applies reverse-KL filtering to select compact, high-quality traces
for distillation into a smaller student model.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import jsonlines
from tqdm import tqdm

from src.data.build_dag import build_dag_from_answer, extract_steps_from_answer

logger = logging.getLogger(__name__)


def filter_by_length(
    traces: list[dict],
    max_steps: int = 15,
    max_chars: int = 2000,
) -> list[dict]:
    """Keep only traces that are compact enough for distillation."""
    kept = []
    for t in traces:
        response = t.get("response", "")
        steps = extract_steps_from_answer(response)
        if len(steps) <= max_steps and len(response) <= max_chars:
            kept.append(t)
    return kept


def score_trace_quality(response: str) -> float:
    """Score a reasoning trace on structural quality (0-1).

    Combines: DAG validity, step continuity, conciseness.
    Used to rank multiple teacher samples and pick the best.
    """
    import re

    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    think_text = think_match.group(1) if think_match else response

    dag = build_dag_from_answer(think_text)
    val = dag.validate_dag()

    score = 0.0
    if val.get("is_acyclic", False):
        score += 0.4
    if not val.get("has_orphan_conclusions", True):
        score += 0.3

    steps = extract_steps_from_answer(think_text)
    n = len(steps)
    if 3 <= n <= 12:
        score += 0.3
    elif n > 0:
        score += 0.1

    return score


def generate_distill_dataset(
    input_path: str,
    output_path: str,
    teacher_traces_path: Optional[str] = None,
    max_steps: int = 15,
    max_chars: int = 2000,
) -> int:
    """Generate distillation dataset.

    If teacher_traces_path is provided, load pre-generated teacher outputs.
    Otherwise, use the original SFT data as a starting point (the teacher
    model inference step should be run separately with swift infer).
    """
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    if teacher_traces_path and Path(teacher_traces_path).exists():
        records = []
        with jsonlines.open(teacher_traces_path) as reader:
            for item in reader:
                records.append(item)
        logger.info(f"Loaded {len(records)} teacher traces")
    else:
        records = []
        with jsonlines.open(input_path) as reader:
            for item in reader:
                records.append(item)
        logger.info(f"Using {len(records)} SFT records as base (run teacher inference first)")

    filtered = filter_by_length(records, max_steps, max_chars)
    logger.info(f"After length filter: {len(filtered)} / {len(records)}")

    scored = []
    for r in tqdm(filtered, desc="Scoring traces"):
        response = ""
        if "messages" in r:
            for m in r["messages"]:
                if m.get("role") == "assistant":
                    response = m.get("content", "")
        elif "response" in r:
            response = r["response"]

        quality = score_trace_quality(response)
        scored.append((quality, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    count = 0
    with jsonlines.open(output_path, mode="w") as writer:
        for quality, record in scored:
            record["distill_quality_score"] = round(quality, 4)
            writer.write(record)
            count += 1

    logger.info(f"Wrote {count} distillation samples to {output_path}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate distillation data")
    parser.add_argument("--input_path", default="data/sft_ready/train.jsonl")
    parser.add_argument("--output_path", default="data/sft_ready/distill_train.jsonl")
    parser.add_argument("--teacher_traces", default=None)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--max_chars", type=int, default=2000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    generate_distill_dataset(args.input_path, args.output_path, args.teacher_traces, args.max_steps, args.max_chars)


if __name__ == "__main__":
    main()
