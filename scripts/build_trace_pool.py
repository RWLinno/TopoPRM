#!/usr/bin/env python3
"""Build per-benchmark rollout trace pools for DAG quality auditing.

This script wraps ``scripts/bench_transformers.py --save_solutions`` and writes:

    output/dag_audit/<label>_<bench>_traces.jsonl

Each output row contains at least ``question``, ``gold``, and ``response``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/Knowin/foundation/weilinruan/env/topoprm/bin/python"
DEFAULT_MODEL = "/Knowin/foundation/weilinruan/hf_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_OUT_DIR = REPO_ROOT / "output" / "dag_audit"
DEFAULT_EVAL_DIR = REPO_ROOT / "output" / "eval"

BENCHMARKS = [
    "gsm8k",
    "math500",
    "olympiadbench",
    "omni_math",
    "aime2024",
    "aime2025",
    "cnmo2024",
    "mmlu",
    "gpqa_diamond",
]


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _ensure_rollout_details(
    *,
    python_bin: str,
    model: str,
    label: str,
    bench: str,
    max_items: int,
    max_new_tokens: int,
    batch_size: int,
    output_dir: Path,
    force_overwrite: bool,
    use_chat_template: bool,
    sft_style: bool,
) -> Path:
    details_path = output_dir / f"{label}_{bench}_details.jsonl"
    if details_path.is_file() and not force_overwrite:
        # If existing details already contain "response", reuse them.
        has_response = False
        for row in _iter_jsonl(details_path):
            if isinstance(row.get("response"), str) and row["response"].strip():
                has_response = True
                break
        if has_response:
            print(f"[reuse] {details_path}")
            return details_path

    cmd = [
        python_bin,
        "scripts/bench_transformers.py",
        "--model",
        model,
        "--label",
        label,
        "--benchmarks",
        bench,
        "--max_items",
        str(max_items),
        "--max_new_tokens",
        str(max_new_tokens),
        "--batch_size",
        str(batch_size),
        "--output_dir",
        str(output_dir),
        "--save_solutions",
        "--force_overwrite",
    ]
    if use_chat_template:
        cmd.append("--use_chat_template")
    if sft_style:
        cmd.append("--sft_style")

    print("[run]", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)
    if not details_path.is_file():
        raise FileNotFoundError(f"Expected details file missing: {details_path}")
    return details_path


def _run_benchmarks_once(
    *,
    python_bin: str,
    model: str,
    label: str,
    benches: list[str],
    max_items: int,
    max_new_tokens: int,
    batch_size: int,
    output_dir: Path,
    use_chat_template: bool,
    sft_style: bool,
) -> None:
    cmd = [
        python_bin,
        "scripts/bench_transformers.py",
        "--model",
        model,
        "--label",
        label,
        "--benchmarks",
        *benches,
        "--max_items",
        str(max_items),
        "--max_new_tokens",
        str(max_new_tokens),
        "--batch_size",
        str(batch_size),
        "--output_dir",
        str(output_dir),
        "--save_solutions",
        "--force_overwrite",
    ]
    if use_chat_template:
        cmd.append("--use_chat_template")
    if sft_style:
        cmd.append("--sft_style")
    print("[run-once]", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


def _convert_details_to_trace_pool(details_path: Path, out_path: Path, bench: str, sample_size: int) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrote = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for row in _iter_jsonl(details_path):
            response = row.get("response")
            if not isinstance(response, str) or not response.strip():
                continue
            trace = {
                "benchmark": bench,
                "question": row.get("question", ""),
                "gold": row.get("gold", ""),
                "response": response,
                "pred_pass1": row.get("pred_pass1"),
                "correct_pass1": row.get("correct_pass1"),
            }
            fout.write(json.dumps(trace, ensure_ascii=False) + "\n")
            wrote += 1
            if sample_size > 0 and wrote >= sample_size:
                break
    return wrote


def main() -> int:
    parser = argparse.ArgumentParser(description="Build per-benchmark trace pools for DAG audit.")
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--label", default="dag_audit_dr1_7b")
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--max-items", type=int, default=50, help="Passed to bench_transformers.")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--use-chat-template", action="store_true", default=True)
    parser.add_argument("--sft-style", action="store_true", default=False)
    parser.add_argument("--force-overwrite", action="store_true", default=False)
    parser.add_argument(
        "--single-run",
        action="store_true",
        default=True,
        help="Run all benchmarks in one bench_transformers call to avoid repeated model loads.",
    )
    args = parser.parse_args()

    if args.single_run:
        _run_benchmarks_once(
            python_bin=args.python_bin,
            model=args.model,
            label=args.label,
            benches=args.benchmarks,
            max_items=args.max_items,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            output_dir=args.eval_dir,
            use_chat_template=args.use_chat_template,
            sft_style=args.sft_style,
        )

    for bench in args.benchmarks:
        details_path = args.eval_dir / f"{args.label}_{bench}_details.jsonl"
        if not details_path.is_file() or (not args.single_run):
            details_path = _ensure_rollout_details(
                python_bin=args.python_bin,
                model=args.model,
                label=args.label,
                bench=bench,
                max_items=args.max_items,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                output_dir=args.eval_dir,
                force_overwrite=args.force_overwrite,
                use_chat_template=args.use_chat_template,
                sft_style=args.sft_style,
            )
        out_path = args.out_dir / f"{args.label}_{bench}_traces.jsonl"
        n = _convert_details_to_trace_pool(details_path, out_path, bench=bench, sample_size=args.sample_size)
        print(f"[ok] {bench}: wrote {n} traces -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
