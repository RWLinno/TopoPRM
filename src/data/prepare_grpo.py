"""Prepare a GRPO (Group Relative Policy Optimisation) dataset.

Each record is a *prompt-only* message list with two sidecar fields:

* ``solution``      – the reference LLM critique (JSON string).
* ``reference_dag`` – the DAG JSON built from the standard answer.

Usage::

    python -m src.data.prepare_grpo \\
        --input_path  data/processed/cleaned.jsonl \\
        --dag_dir     data/dag \\
        --output_path data/grpo_ready/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from src.data.prepare_sft import SYSTEM_PROMPT, format_critique_prompt

logger = logging.getLogger(__name__)


def _load_dag_json(dag_dir: Path, record_id: str) -> Optional[str]:
    """Try to load the DAG JSON for *record_id* from *dag_dir*."""
    dag_file = dag_dir / f"{record_id}.json"
    if dag_file.exists():
        return dag_file.read_text(encoding="utf-8").strip()
    return None


def prepare_grpo_dataset(
    input_path: str,
    dag_dir: str,
    output_path: str,
) -> int:
    """Build the GRPO-ready JSONL dataset.

    Parameters
    ----------
    input_path:
        Cleaned JSONL from :mod:`src.data.clean`.
    dag_dir:
        Directory of per-record DAG ``.json`` files (from :mod:`build_dag`).
    output_path:
        Destination JSONL.

    Returns
    -------
    int
        Number of records written.
    """
    dag_dir_path = Path(dag_dir)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped_no_dag = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
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
            record_id = rec.get("record_id", "")

            if not stem or not llm_res:
                continue

            user_msg = format_critique_prompt(stem, std_ans, stu_ans)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]

            dag_json = _load_dag_json(dag_dir_path, record_id)
            if dag_json is None:
                skipped_no_dag += 1
                dag_json = "{}"

            grpo_record = {
                "messages": messages,
                "solution": json.dumps(llm_res, ensure_ascii=False),
                "reference_dag": dag_json,
            }

            fout.write(json.dumps(grpo_record, ensure_ascii=False) + "\n")
            count += 1

    if skipped_no_dag:
        logger.warning(
            "%d records had no matching DAG file; used empty DAG", skipped_no_dag
        )
    logger.info("Wrote %d GRPO samples → %s", count, output_path)
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare GRPO training data (prompt-only + solution + DAG)."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/cleaned.jsonl",
    )
    parser.add_argument(
        "--dag_dir",
        type=str,
        default="data/dag",
        help="Directory containing per-record DAG JSON files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/grpo_ready/train.jsonl",
    )
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

    n = prepare_grpo_dataset(args.input_path, args.dag_dir, args.output_path)
    logger.info("Total GRPO samples: %d", n)


if __name__ == "__main__":
    main()
