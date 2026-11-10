"""Clean and filter the parsed math-grading dataset.

Applies heuristic quality filters to remove records that would hurt
downstream training:

* Empty or missing core fields.
* Student answers that are header-only (too short).
* ``llm_result`` that is not valid JSON inside the ``<answer>`` block.
* Excessive overlap (>80 %) between ``llm_result`` and ``standard_answer``
  indicating the model simply copied the reference.

Usage::

    python -m src.data.clean \\
        --input_path  data/processed/parsed.jsonl \\
        --output_path data/processed/cleaned.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _char_overlap(a: str, b: str) -> float:
    """Return the fraction of characters in *a* that also appear in *b*."""
    if not a:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    return len(set_a & set_b) / len(set_a)


_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _llm_result_has_valid_json(llm_result: str) -> bool:
    """Check that ``llm_result`` contains a parseable JSON block
    inside ``<answer>…</answer>`` tags.
    """
    m = _ANSWER_TAG_RE.search(llm_result)
    if m is None:
        return False
    try:
        json.loads(m.group(1).strip().replace("'", '"'))
        return True
    except (json.JSONDecodeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Main filter
# ---------------------------------------------------------------------------

def clean_dataset(
    input_path: str,
    output_path: str,
    min_student_len: int = 15,
    max_overlap: float = 0.8,
) -> Dict[str, int]:
    """Read *input_path* JSONL, apply quality filters, write to *output_path*.

    Parameters
    ----------
    input_path:
        Source JSONL produced by :func:`parse_raw.parse_dataset`.
    output_path:
        Destination JSONL.
    min_student_len:
        Minimum character length for a student answer to be kept.
    max_overlap:
        Maximum character overlap ratio between ``llm_result`` and
        ``standard_answer``.  Records above this threshold are dropped.

    Returns
    -------
    dict
        Statistics: ``total``, ``kept``, ``drop_empty``, ``drop_short``,
        ``drop_json``, ``drop_overlap``.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, int] = {
        "total": 0,
        "kept": 0,
        "drop_empty": 0,
        "drop_short": 0,
        "drop_json": 0,
        "drop_overlap": 0,
    }

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["drop_empty"] += 1
                continue

            stem = rec.get("stem", "")
            std_ans = rec.get("standard_answer", "")
            stu_ans = rec.get("student_answer", "")
            llm_res = rec.get("llm_result", "")

            # 1. core fields must be non-empty
            if not stem or not std_ans or not stu_ans or not llm_res:
                stats["drop_empty"] += 1
                continue

            # 2. student answer must be long enough
            if len(stu_ans.strip()) < min_student_len:
                stats["drop_short"] += 1
                continue

            # 3. llm_result must contain a valid JSON block
            if not _llm_result_has_valid_json(llm_res):
                stats["drop_json"] += 1
                continue

            # 4. excessive overlap check
            overlap = _char_overlap(llm_res, std_ans)
            if overlap > max_overlap:
                stats["drop_overlap"] += 1
                continue

            stats["kept"] += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Clean stats: %s", json.dumps(stats, ensure_ascii=False))
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter parsed math-grading records by quality heuristics."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/parsed.jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/cleaned.jsonl",
    )
    parser.add_argument("--min_student_len", type=int, default=15)
    parser.add_argument("--max_overlap", type=float, default=0.8)
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

    stats = clean_dataset(
        args.input_path,
        args.output_path,
        min_student_len=args.min_student_len,
        max_overlap=args.max_overlap,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
