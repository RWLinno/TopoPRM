#!/usr/bin/env python3
"""Download and prepare all benchmark datasets to data/benchmarks/.

Creates standardized JSONL files with fields: Problem, Answer, source.
Datasets that require login or are unavailable are marked as SKIP in status.

Usage:
    python3 scripts/download_benchmarks.py [--output_dir data/benchmarks]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from datasets import load_dataset


CACHE_DIR = os.environ.get(
    "HF_DATASETS_CACHE",
    os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "datasets",
    ),
)


def write_jsonl(path: Path, rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def safe_load(hf_path, *, name=None, split="test", trust_remote_code=False):
    kwargs = {"cache_dir": CACHE_DIR, "trust_remote_code": trust_remote_code}
    attempts = [name] if name else [None, "default", "all"]
    for n in attempts:
        try:
            if n:
                return load_dataset(hf_path, n, split=split, **kwargs)
            else:
                return load_dataset(hf_path, split=split, **kwargs)
        except Exception:
            continue
    return None


def download_gsm8k(out_dir: Path) -> dict:
    ds = safe_load("openai/gsm8k", name="main", split="test")
    if ds is None:
        return {"name": "GSM8K", "status": "FAILED", "count": 0}
    rows = []
    for r in ds:
        m = re.search(r"####\s*(.+)", r["answer"])
        gold = m.group(1).strip() if m else r["answer"].strip()
        rows.append({"Problem": r["question"], "Answer": gold, "source": "gsm8k"})
    n = write_jsonl(out_dir / "GSM8K" / "test.jsonl", rows)
    return {"name": "GSM8K", "status": "OK", "count": n}


def download_math500(out_dir: Path) -> dict:
    ds = safe_load("HuggingFaceH4/MATH-500", split="test")
    if ds is None:
        return {"name": "MATH-500", "status": "FAILED", "count": 0}
    rows = []
    for r in ds:
        q = r.get("problem", r.get("question", ""))
        gold = r.get("answer", r.get("solution", ""))
        rows.append({"Problem": q, "Answer": gold, "source": "math500"})
    n = write_jsonl(out_dir / "MATH-500" / "test.jsonl", rows)
    return {"name": "MATH-500", "status": "OK", "count": n}


def download_math_full(out_dir: Path) -> dict:
    ds = safe_load("lighteval/MATH", split="test")
    if ds is None:
        ds = safe_load("hendrycks/competition_math", split="test")
    if ds is None:
        return {"name": "MATH", "status": "FAILED", "count": 0}
    rows = []
    for r in ds:
        q = r.get("problem", r.get("question", ""))
        gold = r.get("answer", r.get("solution", ""))
        level = r.get("level", None)
        rows.append({"Problem": q, "Answer": gold, "level": level, "source": "math"})
    n = write_jsonl(out_dir / "MATH" / "test.jsonl", rows)
    return {"name": "MATH", "status": "OK", "count": n}


def download_mmlu(out_dir: Path) -> dict:
    ds = safe_load("cais/mmlu", name="all", split="test")
    if ds is None:
        return {"name": "MMLU", "status": "FAILED", "count": 0}
    rows = []
    for r in ds:
        choices = r.get("choices", r.get("options", []))
        q = r.get("question", "")
        gold = r.get("answer", "")
        if isinstance(gold, int) and 0 <= gold < 5:
            gold = "ABCDE"[gold]
        rows.append({
            "Problem": q, "choices": choices, "Answer": str(gold), "source": "mmlu",
            "subject": r.get("subject", ""),
        })
    n = write_jsonl(out_dir / "MMLU" / "test.jsonl", rows)
    return {"name": "MMLU", "status": "OK", "count": n}


def download_gpqa_diamond(out_dir: Path) -> dict:
    ds = safe_load("Idavidrein/gpqa", name="gpqa_diamond", split="train")
    if ds is None:
        return {"name": "GPQA_Diamond", "status": "FAILED", "count": 0}
    rows = []
    for r in ds:
        rows.append({
            "Problem": r.get("question", r.get("Question", "")),
            "Correct Answer": r.get("Correct Answer", r.get("answer", "")),
            "Incorrect Answer 1": r.get("Incorrect Answer 1", ""),
            "Incorrect Answer 2": r.get("Incorrect Answer 2", ""),
            "Incorrect Answer 3": r.get("Incorrect Answer 3", ""),
            "source": "gpqa_diamond",
        })
    n = write_jsonl(out_dir / "GPQA_Diamond" / "test.jsonl", rows)
    return {"name": "GPQA_Diamond", "status": "OK", "count": n}


def download_olympiadbench(out_dir: Path) -> dict:
    """Download the official English text-only math subset (675 items)."""
    ds = safe_load("lmms-lab/OlympiadBench", split="test_en")
    if ds is None:
        return {"name": "OlympiadBench", "status": "FAILED", "count": 0}
    rows = []
    for r in ds:
        if r.get("source") != "OE_TO_maths_en_COMP":
            continue
        gold = r.get("final_answer")
        if not isinstance(gold, list) or len(gold) != 1:
            continue
        question = str(r.get("question", ""))
        if not question or not str(gold[0]).strip():
            continue
        rows.append({
            "question_id": str(r.get("question_id", "")),
            "question": question,
            "final_answer": [str(gold[0])],
            "answer_type": str(r.get("answer_type", "")),
            "source": "OE_TO_maths_en_COMP",
        })
    if len(rows) != 675:
        return {
            "name": "OlympiadBench",
            "status": f"FAILED (expected 675, found {len(rows)})",
            "count": len(rows),
        }
    n = write_jsonl(out_dir / "OlympiadBench" / "test_en_oe_to_math.jsonl", rows)
    return {"name": "OlympiadBench", "status": "OK", "count": n}


def download_aime(out_dir: Path, year: str) -> dict:
    name = f"AIME{year}"
    out_path = out_dir / name / "train.jsonl"
    if out_path.exists():
        return {"name": name, "status": "EXISTS", "count": sum(1 for _ in out_path.open())}

    ds = safe_load("AI-MO/aimo-validation-aime", split="train")
    if ds is None:
        # Try MathArena
        ds = safe_load(f"MathArena/aime_{year}", split="train")
    if ds is None:
        return {"name": name, "status": "FAILED", "count": 0}

    rows = []
    for r in ds:
        row_id = str(r.get("id", r.get("url", "")))
        if year not in row_id and ds.num_rows > 30:
            continue
        q = r.get("problem", r.get("Problem", r.get("question", "")))
        gold = r.get("answer", r.get("Answer", ""))
        if q and str(gold).strip():
            rows.append({"Problem": str(q), "Answer": str(gold), "source": name.lower()})
    if not rows:
        return {"name": name, "status": "EMPTY", "count": 0}
    n = write_jsonl(out_path, rows)
    return {"name": name, "status": "OK", "count": n}


def download_cnmo(out_dir: Path) -> dict:
    name = "CNMO2024"
    out_path = out_dir / name / "train.jsonl"
    if out_path.exists():
        return {"name": name, "status": "EXISTS", "count": sum(1 for _ in out_path.open())}
    # CNMO has no standard HF dataset — try known repos
    for repo in ["anonymous/topoprm-data", "math-eval/cnmo2024"]:
        try:
            ds = safe_load(repo, split="train")
            if ds is not None:
                rows = []
                for r in ds:
                    q = r.get("problem", r.get("Problem", ""))
                    gold = r.get("answer", r.get("Answer", ""))
                    if q and str(gold).strip():
                        rows.append({"Problem": str(q), "Answer": str(gold), "source": "cnmo2024"})
                if rows:
                    n = write_jsonl(out_path, rows)
                    return {"name": name, "status": "OK", "count": n}
        except Exception:
            continue
    return {"name": name, "status": "SKIP (no public source, needs manual data)", "count": 0}


def download_amc23(out_dir: Path) -> dict:
    name = "AMC23"
    out_path = out_dir / name / "train.jsonl"
    if out_path.exists():
        return {"name": name, "status": "EXISTS", "count": sum(1 for _ in out_path.open())}
    ds = safe_load("AI-MO/aimo-validation-amc", split="train")
    if ds is None:
        return {"name": name, "status": "FAILED", "count": 0}
    rows = []
    for r in ds:
        q = r.get("problem", r.get("Problem", ""))
        gold = r.get("answer", r.get("Answer", ""))
        if q and str(gold).strip():
            rows.append({"Problem": str(q), "Answer": str(gold), "source": "amc23"})
    if not rows:
        return {"name": name, "status": "EMPTY", "count": 0}
    n = write_jsonl(out_path, rows)
    return {"name": name, "status": "OK", "count": n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/benchmarks")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print("=" * 60)
    print("TopoPRM Benchmark Downloader")
    print("=" * 60)

    tasks = [
        ("GSM8K", lambda: download_gsm8k(out_dir)),
        ("MATH-500", lambda: download_math500(out_dir)),
        ("MATH (full)", lambda: download_math_full(out_dir)),
        ("MMLU", lambda: download_mmlu(out_dir)),
        ("GPQA_Diamond", lambda: download_gpqa_diamond(out_dir)),
        ("OlympiadBench", lambda: download_olympiadbench(out_dir)),
        ("AIME2024", lambda: download_aime(out_dir, "2024")),
        ("AIME2025", lambda: download_aime(out_dir, "2025")),
        ("AIME2026", lambda: download_aime(out_dir, "2026")),
        ("CNMO2024", lambda: download_cnmo(out_dir)),
        ("AMC23", lambda: download_amc23(out_dir)),
    ]

    for label, fn in tasks:
        print(f"\n[{label}] Downloading...", flush=True)
        try:
            r = fn()
        except Exception as e:
            r = {"name": label, "status": f"ERROR: {e}", "count": 0}
        results.append(r)
        print(f"  -> {r['status']} ({r['count']} items)")

    # Also run prepare_frontiermath_aime2026 if available
    frontier_script = Path("scripts/prepare_frontiermath_aime2026.py")
    if frontier_script.exists():
        print(f"\n[FrontierMath+AIME2026] Running {frontier_script}...", flush=True)
        import subprocess
        ret = subprocess.run(
            [sys.executable, str(frontier_script), "--bench_root", str(out_dir)],
            capture_output=True, text=True,
        )
        if ret.returncode == 0:
            results.append({"name": "FrontierMath", "status": "OK (via script)", "count": -1})
        else:
            results.append({"name": "FrontierMath", "status": f"WARN: {ret.stderr[-200:]}", "count": 0})

    # Write manifest
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Write status summary
    status_path = out_dir / "status.md"
    with status_path.open("w") as f:
        f.write("# Benchmark Download Status\n\n")
        f.write("| Dataset | Status | Count |\n")
        f.write("|---------|--------|-------|\n")
        for r in results:
            f.write(f"| {r['name']} | {r['status']} | {r['count']} |\n")

    print("\n" + "=" * 60)
    print(f"Done. {sum(1 for r in results if 'OK' in r['status'])}/{len(results)} succeeded.")
    print(f"Manifest: {manifest_path}")
    print(f"Status:   {status_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
