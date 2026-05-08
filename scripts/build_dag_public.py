"""Build DAGs at scale from public math datasets (GSM8K, MATH, NuminaMath-CoT).

Usage:
    python3 scripts/build_dag_public.py \
        --datasets gsm8k math numina \
        --output_dir data/dag_public \
        --output_jsonl data/grpo_ready/train_public.jsonl \
        --max_samples 100000

Produces:
    1. data/dag_public/<dataset>_<idx>.json   -- per-problem DAG files
    2. data/grpo_ready/train_public.jsonl      -- GRPO-ready records with reference_dag
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.build_dag import build_dag_from_answer, parse_answer_to_dag_debug


def load_gsm8k(split: str = "train", max_samples: int = -1) -> List[Dict[str, Any]]:
    """Load GSM8K from local JSONL first, fallback to HuggingFace."""
    local = Path(f"data/benchmarks/GSM8K/{split}.jsonl")
    if local.exists():
        ds = []
        for l in local.read_text(encoding="utf-8").splitlines():
            l = l.strip()
            if not l:
                continue
            try:
                ds.append(json.loads(l))
            except json.JSONDecodeError:
                continue
        logger.info("Loading gsm8k from local: %s (%d records)", local, len(ds))
    else:
        try:
            from datasets import load_dataset
            ds = list(load_dataset("openai/gsm8k", "main", split=split))
        except Exception as e:
            logger.error("gsm8k not available: %s", e)
            return []

    records = []
    for i, item in enumerate(ds):
        if 0 < max_samples <= i:
            break
        answer_text = item.get("answer", "")
        question = item.get("question", "")
        final_answer = ""
        if "####" in answer_text:
            parts = answer_text.split("####")
            answer_text = parts[0].strip()
            final_answer = parts[1].strip()
        records.append({
            "id": f"gsm8k_{i}",
            "question": question,
            "solution": answer_text,
            "final_answer": final_answer,
            "source": "gsm8k",
        })
    logger.info("Loaded %d records from gsm8k/%s", len(records), split)
    return records


def load_math(split: str = "train", max_samples: int = -1) -> List[Dict[str, Any]]:
    """Load MATH dataset from local JSONL first, fallback to HuggingFace."""
    local = Path(f"data/benchmarks/MATH/{split}.jsonl")
    if local.exists():
        ds = []
        for l in local.read_text(encoding="utf-8").splitlines():
            l = l.strip()
            if not l:
                continue
            try:
                ds.append(json.loads(l))
            except json.JSONDecodeError:
                continue
        logger.info("Loading MATH from local: %s (%d records)", local, len(ds))
    else:
        try:
            from datasets import load_dataset
            ds = list(load_dataset("lighteval/MATH", "all", split=split, trust_remote_code=True))
        except Exception as e:
            logger.error("MATH not available: %s", e)
            return []

    records = []
    for i, item in enumerate(ds):
        if 0 < max_samples <= i:
            break
        solution = item.get("solution", "")
        question = item.get("problem", "")
        records.append({
            "id": f"math_{i}",
            "question": question,
            "solution": solution,
            "final_answer": item.get("answer", ""),
            "source": "math",
            "level": item.get("level", ""),
            "type": item.get("type", ""),
        })
    logger.info("Loaded %d records from MATH/%s", len(records), split)
    return records


def load_numina(max_samples: int = -1) -> List[Dict[str, Any]]:
    """Load NuminaMath-CoT from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    except Exception as e:
        logger.error("NuminaMath-CoT not available: %s", e)
        return []

    records = []
    for i, item in enumerate(ds):
        if 0 < max_samples <= i:
            break
        solution = item.get("solution", "")
        question = item.get("problem", "")
        if not solution or len(solution) < 20:
            continue
        records.append({
            "id": f"numina_{i}",
            "question": question,
            "solution": solution,
            "final_answer": "",
            "source": "numina",
        })
    logger.info("Loaded %d records from NuminaMath-CoT", len(records))
    return records


def build_dags_from_records(
    records: List[Dict[str, Any]],
    output_dir: Path,
    output_jsonl: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build DAGs for all records, write to output_dir and optional JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "valid": 0, "acyclic": 0, "with_virtual_edges": 0,
             "avg_nodes": 0.0, "avg_edges": 0.0, "avg_virtual_edges": 0.0}
    total_nodes = 0
    total_edges = 0
    total_virtual = 0

    jsonl_file = None
    if output_jsonl:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = open(output_jsonl, "w", encoding="utf-8")

    for record in records:
        rid = record["id"]
        solution = record["solution"]
        stats["total"] += 1

        dag, debug = parse_answer_to_dag_debug(
            answer=solution,
            problem_id=rid,
        )

        if dag.num_nodes == 0:
            continue

        stats["valid"] += 1
        total_nodes += dag.num_nodes
        total_edges += dag.num_edges

        is_acyclic = dag.is_acyclic() if hasattr(dag, "is_acyclic") else True
        if is_acyclic:
            stats["acyclic"] += 1

        n_virtual = sum(
            1 for _, _, d in dag.graph.edges(data=True)
            if dag.is_virtual_edge(d.get("edge_type", ""))
        )
        total_virtual += n_virtual
        if n_virtual > 0:
            stats["with_virtual_edges"] += 1

        out_file = output_dir / f"{rid}.json"
        out_file.write_text(dag.to_json(), encoding="utf-8")

        if jsonl_file:
            grpo_record = {
                "record_id": rid,
                "question": record["question"],
                "standard_answer": solution,
                "final_answer": record.get("final_answer", ""),
                "source": record.get("source", ""),
                "reference_dag": json.loads(dag.to_json()),
            }
            jsonl_file.write(json.dumps(grpo_record, ensure_ascii=False) + "\n")

        if stats["total"] % 1000 == 0:
            logger.info("Progress: %d / %d processed, %d valid DAGs",
                        stats["total"], len(records), stats["valid"])

    if jsonl_file:
        jsonl_file.close()

    if stats["valid"] > 0:
        stats["avg_nodes"] = round(total_nodes / stats["valid"], 1)
        stats["avg_edges"] = round(total_edges / stats["valid"], 1)
        stats["avg_virtual_edges"] = round(total_virtual / stats["valid"], 1)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Build DAGs from public math datasets")
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "math"],
                        choices=["gsm8k", "math", "numina"],
                        help="Which datasets to process")
    parser.add_argument("--output_dir", type=str, default="data/dag_public")
    parser.add_argument("--output_jsonl", type=str, default="data/grpo_ready/train_public.jsonl")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples per dataset (-1 = all)")
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    all_records: List[Dict[str, Any]] = []

    for ds_name in args.datasets:
        if ds_name == "gsm8k":
            all_records.extend(load_gsm8k(max_samples=args.max_samples))
        elif ds_name == "math":
            all_records.extend(load_math(max_samples=args.max_samples))
        elif ds_name == "numina":
            all_records.extend(load_numina(max_samples=args.max_samples))

    logger.info("Total records to process: %d", len(all_records))

    stats = build_dags_from_records(
        records=all_records,
        output_dir=Path(args.output_dir),
        output_jsonl=Path(args.output_jsonl) if args.output_jsonl else None,
    )

    logger.info("=" * 60)
    logger.info("DAG Construction Summary")
    logger.info("=" * 60)
    logger.info("  Total records:       %d", stats["total"])
    logger.info("  Valid DAGs:          %d (%.1f%%)",
                stats["valid"], 100 * stats["valid"] / max(1, stats["total"]))
    logger.info("  Acyclic:             %d (%.1f%%)",
                stats["acyclic"], 100 * stats["acyclic"] / max(1, stats["valid"]))
    logger.info("  With virtual edges:  %d (%.1f%%)",
                stats["with_virtual_edges"],
                100 * stats["with_virtual_edges"] / max(1, stats["valid"]))
    logger.info("  Avg nodes/DAG:       %.1f", stats["avg_nodes"])
    logger.info("  Avg edges/DAG:       %.1f", stats["avg_edges"])
    logger.info("  Avg virtual edges:   %.1f", stats["avg_virtual_edges"])


if __name__ == "__main__":
    main()
