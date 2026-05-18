#!/usr/bin/env python3
"""Prepare English-math datasets for exp_May18.

Outputs:
1) GRPO swift messages dataset from public math data.
2) Long-CoT SFT merged dataset for Qwen2.5-7B.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_user_text(record: dict[str, Any]) -> str:
    question = str(record.get("question", "")).strip()
    if question:
        return question

    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            text = str(msg.get("content", "")).strip()
            if text:
                return text
    return ""


def _extract_solution_text(record: dict[str, Any]) -> str:
    for key in ("solution", "standard_answer", "final_answer"):
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def build_grpo_swift_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in records:
        user_text = _extract_user_text(item)
        solution = _extract_solution_text(item)
        if not user_text or not solution:
            continue

        reference_dag = item.get("reference_dag")
        if reference_dag is None:
            reference_dag = ""
        elif not isinstance(reference_dag, str):
            reference_dag = json.dumps(reference_dag, ensure_ascii=False)

        row = {
            "messages": [{"role": "user", "content": user_text}],
            "solution": solution,
            "reference_dag": reference_dag,
        }
        if item.get("record_id"):
            row["record_id"] = item["record_id"]
        if item.get("source"):
            row["source"] = item["source"]
        out.append(row)
    return out


def _extract_assistant_text(record: dict[str, Any]) -> str:
    # Distilled file has a dedicated "response" field with <think> traces.
    response = str(record.get("response", "")).strip()
    if response:
        return response
    for msg in reversed(record.get("messages", [])):
        if msg.get("role") == "assistant":
            text = str(msg.get("content", "")).strip()
            if text:
                return text
    return ""


def _normalize_sft_record(record: dict[str, Any]) -> dict[str, Any] | None:
    messages = list(record.get("messages", []))
    if messages:
        has_user = any(m.get("role") == "user" and str(m.get("content", "")).strip() for m in messages)
        if not has_user:
            return None

    assistant_text = _extract_assistant_text(record)
    if not assistant_text:
        return None

    normalized_messages: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        if role == "assistant":
            # Ensure long-CoT traces from "response" replace weak assistant text.
            content = assistant_text
        normalized_messages.append({"role": role, "content": content})

    if not normalized_messages:
        return None

    has_assistant = any(m["role"] == "assistant" for m in normalized_messages)
    if not has_assistant:
        normalized_messages.append({"role": "assistant", "content": assistant_text})

    out: dict[str, Any] = {"messages": normalized_messages}
    if record.get("source"):
        out["source"] = record["source"]
    if record.get("lang"):
        out["lang"] = record["lang"]
    if record.get("distill_quality_score") is not None:
        out["distill_quality_score"] = record["distill_quality_score"]
    return out


def build_longcot_sft_rows(
    augmented: list[dict[str, Any]],
    distilled: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in augmented + distilled:
        lang = str(row.get("lang", "en")).lower()
        if lang and lang != "en":
            continue
        norm = _normalize_sft_record(row)
        if norm is not None:
            out.append(norm)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare exp_May18 data.")
    parser.add_argument(
        "--grpo-input",
        default="data/grpo_ready/train_public.jsonl",
        help="Public English math input for GRPO conversion.",
    )
    parser.add_argument(
        "--grpo-output",
        default="data/grpo_ready/train_public_swift.jsonl",
        help="Swift-messages GRPO output path.",
    )
    parser.add_argument(
        "--sft-augmented",
        default="data/sft_ready/train_augmented.jsonl",
        help="Augmented English SFT source.",
    )
    parser.add_argument(
        "--sft-distill",
        default="data/sft_ready/distill_train_from_teacher.jsonl",
        help="Teacher-distilled long-CoT SFT source.",
    )
    parser.add_argument(
        "--sft-output",
        default="data/sft_ready/train_longcot_mix.jsonl",
        help="Merged long-CoT SFT output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grpo_input = Path(args.grpo_input)
    grpo_output = Path(args.grpo_output)
    sft_augmented = Path(args.sft_augmented)
    sft_distill = Path(args.sft_distill)
    sft_output = Path(args.sft_output)

    grpo_rows = build_grpo_swift_rows(_load_jsonl(grpo_input))
    _dump_jsonl(grpo_output, grpo_rows)

    sft_rows = build_longcot_sft_rows(
        _load_jsonl(sft_augmented),
        _load_jsonl(sft_distill),
    )
    _dump_jsonl(sft_output, sft_rows)

    print(f"[exp_May18] GRPO swift rows: {len(grpo_rows)} -> {grpo_output}")
    print(f"[exp_May18] SFT long-CoT rows: {len(sft_rows)} -> {sft_output}")


if __name__ == "__main__":
    main()
