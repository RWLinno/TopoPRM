"""Prepare a supervised fine-tuning (SFT) dataset from cleaned records.

Formats each record as a multi-turn chat message suitable for training
a critique / grading model.

Usage::

    python -m src.data.prepare_sft \\
        --input_path  data/processed/cleaned.jsonl \\
        --output_path data/sft_ready/train.jsonl \\
        --format      critique
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT: str = (
    "你是一名资深、严谨、和蔼的数学老师，"
    "擅长批改学生的数学作业并给出详细的分析和评分。"
)


# ---------------------------------------------------------------------------
# Prompt / response formatters
# ---------------------------------------------------------------------------

def format_critique_prompt(
    stem: str,
    standard_answer: str,
    student_answer: str,
) -> str:
    """Build the user-turn prompt for the *critique* format."""
    return (
        f"请你批改以下学生的数学作答。\n\n"
        f"【题目】\n{stem}\n\n"
        f"【标准答案】\n{standard_answer}\n\n"
        f"【学生作答】\n{student_answer}\n\n"
        f"请分析学生的作答过程，指出正确和错误之处，并给出评分。"
    )


def format_critique_response(llm_result: str) -> str:
    """Build the assistant-turn response for the *critique* format.

    We keep the original LLM result as-is because it already contains
    ``<think>`` and ``<answer>`` blocks.
    """
    return llm_result.strip()


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def prepare_sft_dataset(
    input_path: str,
    output_path: str,
    format_type: str = "critique",
    include_math_qa: bool = False,
    max_samples: Optional[int] = None,
) -> int:
    """Convert cleaned JSONL into SFT-ready messages JSONL.

    Parameters
    ----------
    input_path:
        Cleaned JSONL from :mod:`src.data.clean`.
    output_path:
        Destination JSONL.
    format_type:
        Currently only ``"critique"`` is supported.
    include_math_qa:
        If *True*, additionally generate a QA pair where the assistant
        directly solves the problem (using the standard answer).
    max_samples:
        Cap on the number of output samples.

    Returns
    -------
    int
        Number of samples written.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if max_samples is not None and count >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            stem = rec.get("stem", "")
            std_ans = rec.get("standard_answer", "")
            stu_ans = rec.get("student_answer", "")
            llm_res = rec.get("llm_result", "")

            if not stem or not llm_res:
                continue

            if format_type == "critique":
                user_msg = format_critique_prompt(stem, std_ans, stu_ans)
                asst_msg = format_critique_response(llm_res)
            else:
                logger.warning("Unknown format_type=%s, skipping", format_type)
                continue

            messages: List[dict] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": asst_msg},
            ]
            fout.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1

            # Optional math-QA pair
            if include_math_qa and std_ans:
                qa_messages: List[dict] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"请解答以下数学题：\n\n{stem}"},
                    {"role": "assistant", "content": std_ans},
                ]
                fout.write(
                    json.dumps({"messages": qa_messages}, ensure_ascii=False) + "\n"
                )
                count += 1

    logger.info("Wrote %d SFT samples → %s", count, output_path)
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SFT training data from cleaned math-grading records."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/cleaned.jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/sft_ready/train.jsonl",
    )
    parser.add_argument(
        "--format",
        dest="format_type",
        type=str,
        default="critique",
        choices=["critique"],
        help="Output conversation format.",
    )
    parser.add_argument(
        "--include_math_qa",
        action="store_true",
        help="Also emit a QA pair where the assistant solves the problem.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    n = prepare_sft_dataset(
        args.input_path,
        args.output_path,
        format_type=args.format_type,
        include_math_qa=args.include_math_qa,
        max_samples=args.max_samples,
    )
    logger.info("Total SFT samples: %d", n)


if __name__ == "__main__":
    main()
