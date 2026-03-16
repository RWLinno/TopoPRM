"""Parse raw multi-line JSON files produced by the math-grading API.

Each *.json file under the raw data directory contains two JSON lines:
  - status=0  →  header (ack)
  - status=2  →  payload whose ``output.content`` is a *stringified* JSON
    that itself must be parsed with a second ``json.loads`` call.

The key grading payload lives at::

    data.stepCorrectInfo.extendInfos[key=="StepCorrProcess"].value

Usage::

    python -m src.data.parse_raw \\
        --input_dir /mnt/users/postrain/ms-swift/data/数学0305 \\
        --output_path data/processed/parsed.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.build_dag import build_dag_from_answer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-record parsing
# ---------------------------------------------------------------------------

def _find_step_corr_value(extend_infos: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the ``value`` dict whose ``key`` equals ``StepCorrProcess``."""
    for info in extend_infos:
        if info.get("key") == "StepCorrProcess":
            return info.get("value")
    return None


def parse_single_record(raw_json: dict) -> Optional[dict]:
    """Extract fields from a *payload* line (status == 2).

    Parameters
    ----------
    raw_json:
        The full parsed JSON object of the payload line.

    Returns
    -------
    dict or None
        Flat record with stem, standard_answer, student_answer, llm_result,
        score, and sub_correct_infos.  ``None`` when data is missing.
    """
    try:
        content_str: str = raw_json["payload"]["output"]["content"]
        content: dict = json.loads(content_str)
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        logger.debug("Skipping record – content parse failed: %s", exc)
        return None

    data: dict = content.get("data", {})
    step_info: dict = data.get("stepCorrectInfo", {})
    extend_infos: list = step_info.get("extendInfos", [])

    scp = _find_step_corr_value(extend_infos)
    if scp is None:
        logger.debug("Skipping record – no StepCorrProcess entry")
        return None

    score_info = step_info.get("correctInfo", {}).get("scoreInfo", {})
    sub_correct_infos = step_info.get("subCorrectInfos", [])

    return {
        "stem": scp.get("llm_stem", ""),
        "standard_answer": scp.get("llm_stdanswer", ""),
        "student_answer": scp.get("llm_user", ""),
        "llm_result": scp.get("llm_result", ""),
        "score": score_info.get("score", -1.0),
        "procedure_score": score_info.get("procedureScore", -1.0),
        "sub_correct_infos": sub_correct_infos,
        "topic_id": data.get("topicId", ""),
        "topic_type": data.get("topicType", ""),
    }


# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------

def _parse_file(filepath: str) -> Optional[dict]:
    """Read a raw JSON file (two lines) and return the parsed record."""
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except Exception as exc:
        logger.warning("Cannot read %s: %s", filepath, exc)
        return None

    payload_line: Optional[dict] = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        status = obj.get("header", {}).get("status")
        if status == 2:
            payload_line = obj
            break

    if payload_line is None:
        logger.debug("No status=2 payload in %s", filepath)
        return None

    record = parse_single_record(payload_line)
    if record is not None:
        record["source_file"] = os.path.basename(filepath)
    return record


# ---------------------------------------------------------------------------
# Dataset-level parsing
# ---------------------------------------------------------------------------

def parse_dataset(input_dir: str, output_path: str) -> int:
    """Walk *input_dir* recursively, parse every ``.json`` file, and write
    parsed records as JSONL.  A companion ``.dag.jsonl`` file with the
    DAG representation of each standard answer is written alongside.

    Returns the number of successfully parsed records.
    """
    input_dir_path = Path(input_dir)
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    dag_path = str(output_path).replace(".jsonl", ".dag.jsonl")
    dag_path_obj = Path(dag_path)
    dag_path_obj.parent.mkdir(parents=True, exist_ok=True)

    json_files: List[Path] = sorted(input_dir_path.rglob("*.json"))
    logger.info("Found %d .json files under %s", len(json_files), input_dir)

    count = 0
    with open(output_path, "w", encoding="utf-8") as fout, \
         open(dag_path, "w", encoding="utf-8") as fdag:
        for jf in json_files:
            record = _parse_file(str(jf))
            if record is None:
                continue

            # Determine category from parent directory name
            category = jf.parent.name
            record["category"] = category
            record["record_id"] = f"{category}_{count}"

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Build DAG from standard answer
            try:
                dag = build_dag_from_answer(
                    record["standard_answer"],
                    problem_id=record["record_id"],
                )
                fdag.write(dag.to_json().replace("\n", " ") + "\n")
            except Exception as exc:
                logger.warning("DAG build failed for %s: %s", record["record_id"], exc)

            count += 1
            if count % 500 == 0:
                logger.info("Parsed %d records so far …", count)

    logger.info(
        "Finished – %d records written to %s (DAGs → %s)", count, output_path, dag_path
    )
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse raw math-grading JSON files into a JSONL dataset."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/mnt/users/postrain/ms-swift/data/数学0305",
        help="Root directory containing raw .json files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/parsed.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    n = parse_dataset(args.input_dir, args.output_path)
    logger.info("Total parsed: %d", n)


if __name__ == "__main__":
    main()
