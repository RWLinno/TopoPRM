"""Offline three-phase DAG preprocessing with optional LLM refinement.

Pipeline (offline only; no online cost during GRPO/eval):
  Phase 1 - Step segmentation & parsing
  Phase 2 - Implicit feature extraction (LLM-assisted, strict JSON schema)
  Phase 3 - Build dependency DAG and cache it as JSON

Usage
-----
    python scripts/preprocess_dag_cache.py \
        --input data/grpo_ready/train_public.jsonl \
        --output data/grpo_ready/train_public.dag_cached.jsonl \
        --field reference_dag \
        --use-llm 1 \
        --llm-model /Knowin/foundation/weilinruan/hf_models/Qwen/Qwen2.5-Math-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.build_dag import parse_answer_to_dag_debug  # noqa: E402

logger = logging.getLogger("preprocess_dag_cache")


def _extract_trace(record: Dict[str, Any]) -> str:
    for key in ("response", "solution", "trace", "answer", "completion"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val
    msgs = record.get("messages")
    if isinstance(msgs, list):
        for msg in reversed(msgs):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline 3-phase DAG preprocessing with LLM refinement.")
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL file.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL file.")
    parser.add_argument(
        "--field",
        default="cached_dag",
        help="Field name under which to store the cached DAG dict.",
    )
    parser.add_argument("--use-llm", type=int, default=1, help="Enable Phase 2 LLM refinement (default 1).")
    parser.add_argument(
        "--llm-model",
        default=os.environ.get(
            "TOPO_DAG_LLM_MODEL",
            "/Knowin/foundation/weilinruan/hf_models/Qwen/Qwen2.5-Math-1.5B-Instruct",
        ),
        help="Local HF model path for Phase 2.",
    )
    parser.add_argument("--llm-device", default=os.environ.get("TOPO_DAG_LLM_DEVICE", "auto"))
    parser.add_argument("--max-steps", type=int, default=16, help="Skip LLM if more steps than this.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max records.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    os.environ["TOPO_DAG_LLM_REFINE"] = "1" if args.use_llm else "0"
    if args.llm_model:
        os.environ["TOPO_DAG_LLM_MODEL"] = args.llm_model
    if args.llm_device:
        os.environ["TOPO_DAG_LLM_DEVICE"] = args.llm_device
    os.environ["TOPO_DAG_LLM_MAX_STEPS"] = str(args.max_steps)

    if not args.input.is_file():
        raise FileNotFoundError(f"Input not found: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_cached = 0
    n_failed = 0
    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                n_failed += 1
                continue
            trace = _extract_trace(record)
            if not trace:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            try:
                dag, debug = parse_answer_to_dag_debug(
                    trace,
                    problem_id=str(record.get("problem_id", record.get("id", n_total))),
                )
                evidence_index = {
                    (e["source"], e["target"], e.get("dep_type", "")): e
                    for e in debug.get("edges", [])
                }
                cached = {
                    "schema_version": "topoprm.dag.v1",
                    "nodes": [
                        {
                            "step_id": n.step_id,
                            "raw_text": n.raw_text,
                            "normalized_text": n.normalized_text,
                            "step_type": n.step_type.value,
                            "local_verdict": n.local_verdict.value,
                            "sub_question_id": n.sub_question_id,
                            "exprs": list(n.exprs),
                            "claims": list(n.claims),
                        }
                        for n in dag.nodes.values()
                    ],
                    "edges": [
                        {
                            "source": u,
                            "target": v,
                            "edge_type": d.get("edge_type", ""),
                            "dep_type": d.get("dep_type", ""),
                            "weight": float(d.get("weight", 0.0)),
                            "source_kind": (
                                "llm" if d.get("dep_type", "").startswith("llm_") else "rule"
                            ),
                            "evidence": evidence_index.get(
                                (u, v, d.get("dep_type", "")),
                                {},
                            ).get("evidence", ""),
                        }
                        for u, v, d in dag.graph.edges(data=True)
                    ],
                    "summary": debug.get("summary", {}),
                }
                record[args.field] = cached
                n_cached += 1
            except Exception as exc:
                logger.warning("DAG preprocessing failed on record %d: %s", n_total, exc)
                n_failed += 1
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.limit and n_total >= args.limit:
                break

    logger.info("done: total=%d cached=%d failed=%d -> %s", n_total, n_cached, n_failed, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
