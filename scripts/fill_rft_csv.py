#!/usr/bin/env python3
"""Fill the rft.csv template with our measured benchmark results.

Reads all output/eval/*_metrics.json files and generates a CSV in the
format of docs/rft (1).csv:
    Model, Params, Avg.F1,
    GSM8K: [error, correct, F1, pass@1, pass@k, maj@k, prm@k, #Tokens]
    MATH-500: [...]
    OlympiadBench: [...]
    Omni-MATH: [...]
    AIME 2024: [...]
    AIME 2025: [...]
    CNMO 2024: [...]
    LiveCode: [...]
    MMLU: [...]
    GPQA-D: [...]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BENCH_ORDER = [
    "gsm8k", "math500", "olympiadbench", "omni_math",
    "aime2024", "aime2025", "cnmo2024", "livecode",
    "mmlu", "gpqa_diamond",
]

BENCH_LABELS = {
    "gsm8k": "GSM8K",
    "math500": "MATH-500",
    "olympiadbench": "OlympiadBench",
    "omni_math": "Omni-MATH",
    "aime2024": "AIME 2024",
    "aime2025": "AIME 2025",
    "cnmo2024": "CNMO 2024",
    "livecode": "LiveCode",
    "mmlu": "MMLU",
    "gpqa_diamond": "GPQA-D",
}

METRIC_COLS = ["error", "correct", "F1", "pass@1", "pass@k", "maj@k", "prm@k", "#Tokens"]

# Model display names + params (in order)
MODEL_REGISTRY: list[tuple[str, str, str]] = [
    # label_prefix, display_name, params
    ("base_9b",            "Qwen3.5-9B (base)",               "9B"),
    ("base_9b_v2",         "Qwen3.5-9B (base, v2)",           "9B"),
    ("sft_9b",             "+ SFT",                           "9B"),
    ("sft_9b_v2",          "+ SFT (v2 chat)",                 "9B"),
    ("topoprm_hier_9b",    "+ GRPO (TopoPRM hierarchical)",   "9B"),
    ("topoprm_hier_9b_v2", "+ GRPO (TopoPRM hierarchical v2)", "9B"),
    ("topoprm_gated_9b",   "+ GRPO (TopoPRM gated)",          "9B"),
    ("topoprm_gated_9b_v2", "+ GRPO (TopoPRM gated v2)",      "9B"),
    ("outcome_only_9b",    "+ GRPO (outcome-only)",           "9B"),
    ("no_topo_9b",         "+ GRPO (w/o topology)",           "9B"),
    ("no_continuity_9b",   "+ GRPO (w/o continuity)",         "9B"),
    ("base_qwen25_7b",     "Qwen2.5-7B (base)",               "7B"),
    ("topoprm_hier_7b",    "+ TopoPRM (Qwen2.5-7B)",          "7B"),
    ("distill_rkl_8b",     "Student (8B, RKL legacy)",        "8B"),
    ("distill_sft_4b",     "Student (Qwen3.5-4B, SFT distill)", "4B"),
    ("distill_opsd_4b",    "Student (Qwen3.5-4B, TVSD)", "4B"),
    ("distill_sft_2b",     "Student (Qwen3.5-2B, SFT distill)", "2B"),
    ("distill_opsd_2b",    "Student (Qwen3.5-2B, TVSD)", "2B"),
    ("distill_sft_0p8b",   "Student (Qwen3.5-0.8B, SFT distill)", "0.8B"),
    ("distill_opsd_0p8b",  "Student (Qwen3.5-0.8B, TVSD)", "0.8B"),
]


def _read_metrics(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pct(v: Any, digits: int = 1) -> str:
    if v is None or v == "":
        return "-"
    try:
        x = float(v)
        if x <= 1.0001:
            x *= 100.0
        return f"{x:.{digits}f}"
    except Exception:
        return "-"


def _count(v: Any) -> str:
    if v is None or v == "":
        return "-"
    try:
        return str(int(float(v)))
    except Exception:
        return str(v)


def metric_row(label: str, eval_dir: Path) -> tuple[dict[str, dict[str, str]], int]:
    """Return {bench: {metric: str}} and a samples_count for Avg.F1 computation."""
    row: dict[str, dict[str, str]] = {}
    f1_vals = []
    for bench in BENCH_ORDER:
        mf = eval_dir / f"{label}_{bench}_metrics.json"
        if not mf.exists():
            row[bench] = {m: "-" for m in METRIC_COLS}
            continue
        d = _read_metrics(mf)
        if not d:
            row[bench] = {m: "-" for m in METRIC_COLS}
            continue
        # k: prefer the largest k in k_values (default 5)
        k_vals = d.get("k_values", [1, 5])
        k_main = max(k_vals) if isinstance(k_vals, list) else 5
        row[bench] = {
            "error":   _count(d.get("error", d.get("error_count"))),
            "correct": _count(d.get("correct", d.get("correct_count"))),
            "F1":      _pct(d.get("f1", d.get("pass@1"))),
            "pass@1":  _pct(d.get("pass@1", d.get("accuracy"))),
            "pass@k":  _pct(d.get(f"pass@{k_main}")),
            "maj@k":   _pct(d.get(f"maj@{k_main}")),
            "prm@k":   _pct(d.get(f"prm@{k_main}")),
            "#Tokens": _count(d.get("avg_tokens", d.get("avg_gen_tokens"))),
        }
        f1 = d.get("f1", d.get("pass@1", 0.0))
        if f1:
            try:
                f1_vals.append(float(f1) * (100 if float(f1) <= 1.0001 else 1))
            except Exception:
                pass
    avg_f1 = round(sum(f1_vals) / len(f1_vals), 1) if f1_vals else None
    return row, avg_f1


def build_header() -> list[list[str]]:
    # Header row 1: bench group names
    h1 = ["Model", "Params", "Avg. F1"]
    for bench in BENCH_ORDER:
        label = BENCH_LABELS[bench]
        # Each bench spans 8 columns but first col gets the label and rest empty
        h1.append(label)
        h1.extend([""] * (len(METRIC_COLS) - 1))
    # Header row 2: metric names repeated
    h2 = ["", "", ""]
    for _ in BENCH_ORDER:
        h2.extend(METRIC_COLS)
    return [h1, h2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=Path, default=Path("output/eval"))
    ap.add_argument("--output", type=Path, default=Path("docs/rft_ours.csv"))
    ap.add_argument("--only_with_data", action="store_true",
                    help="Only include model rows that have at least one measured benchmark")
    args = ap.parse_args()

    out_rows: list[list[str]] = build_header()

    for prefix, display, params in MODEL_REGISTRY:
        row, avg_f1 = metric_row(prefix, args.eval_dir)
        has_data = any(cell.get("pass@1", "-") != "-" for cell in row.values())
        if args.only_with_data and not has_data:
            continue

        r = [display, params, f"{avg_f1:.1f}" if avg_f1 is not None else "-"]
        for bench in BENCH_ORDER:
            cells = row[bench]
            for m in METRIC_COLS:
                r.append(cells.get(m, "-"))
        out_rows.append(r)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for r in out_rows:
            writer.writerow(r)

    print(f"Wrote {args.output}  ({len(out_rows) - 2} model rows, {len(BENCH_ORDER)} benchmarks)")


if __name__ == "__main__":
    main()
