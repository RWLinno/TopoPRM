#!/usr/bin/env python3
"""dump_results_for_paper.py

Scan output/eval/*_metrics.json and print a compact table of pass@1 values
keyed by (label, benchmark), used to refill the paper tables.

Designed for the DR1-7B family rows in
topoprm_paper/tables/public_results_unified.tex; see the experiment resync
notes.

Usage:
    python3 scripts/dist/dump_results_for_paper.py
    python3 scripts/dist/dump_results_for_paper.py --labels baseline_dr1_7b_chat sft_dr1_7b grpo_outcome_only grpo_no_topo grpo_no_continuity topoprm_full_dr1_7b
    python3 scripts/dist/dump_results_for_paper.py --format latex   # print LaTeX rows
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DR1_7B_FAMILY = [
    "baseline_dr1_7b_chat",
    "sft_dr1_7b",
    "grpo_outcome_only",
    "grpo_no_topo",
    "grpo_no_continuity",
    "topoprm_full_dr1_7b",
]

PAPER_BENCH_ORDER = [
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

PAPER_BENCH_LABEL = {
    "gsm8k": "GSM8K",
    "math500": "MATH-500",
    "olympiadbench": "Olympiad",
    "omni_math": "Omni-MATH",
    "aime2024": "AIME'24",
    "aime2025": "AIME'25",
    "cnmo2024": "CNMO'24",
    "mmlu": "MMLU",
    "gpqa_diamond": "GPQA-D",
}

ROW_DISPLAY = {
    "baseline_dr1_7b_chat": "DR1-7B (base, measured)",
    "sft_dr1_7b": "\\quad + SFT",
    "grpo_outcome_only": "\\quad + GRPO (outcome-only)",
    "grpo_no_topo": "\\quad + GRPO (w/o topology)",
    "grpo_no_continuity": "\\quad + GRPO (w/o continuity)",
    "topoprm_full_dr1_7b": "\\quad + TopoPRM (hierarchical)",
}


def pct(v):
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x * 100.0 if x <= 1.0001 else x


def load_metrics(eval_dir: Path):
    """Return {label: {bench: metrics_dict}}."""
    out = {}
    for f in sorted(eval_dir.glob("*_metrics.json")):
        stem = f.stem
        if not stem.endswith("_metrics"):
            continue
        base = stem[: -len("_metrics")]
        for b in PAPER_BENCH_ORDER:
            suffix = f"_{b}"
            if base.endswith(suffix):
                label = base[: -len(suffix)]
                out.setdefault(label, {})[b] = json.loads(f.read_text())
                break
    return out


def fmt_cell(metrics):
    """Format pass@1 as a percentage string with one decimal."""
    if not metrics:
        return "--"
    p1 = pct(metrics.get("pass@1", metrics.get("accuracy")))
    if p1 is None:
        return "--"
    return f"{p1:.1f}"


def print_grid(data, labels):
    headers = ["label"] + [PAPER_BENCH_LABEL[b] for b in PAPER_BENCH_ORDER]
    col_widths = [max(len(h), 18) for h in headers]
    for li, lbl in enumerate(labels):
        col_widths[0] = max(col_widths[0], len(lbl))
    for i, b in enumerate(PAPER_BENCH_ORDER, start=1):
        col_widths[i] = max(col_widths[i], len(PAPER_BENCH_LABEL[b]))
    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "-+-".join("-" * w for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for lbl in labels:
        row = data.get(lbl, {})
        cells = [lbl] + [fmt_cell(row.get(b)) for b in PAPER_BENCH_ORDER]
        print(fmt.format(*cells))


def print_latex(data, labels):
    print("% paste this DR1-7B block into "
          "topoprm_paper/tables/public_results_unified.tex (replacing the existing one).")
    for lbl in labels:
        row = data.get(lbl, {})
        cells = [ROW_DISPLAY.get(lbl, lbl), "7B"]
        for b in PAPER_BENCH_ORDER:
            cells.append(fmt_cell(row.get(b)))
        # Insert blue highlight for the TopoPRM row.
        if lbl == "topoprm_full_dr1_7b":
            print("\\rowcolor{blue!5}")
        print(" & ".join(cells) + " \\\\")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="output/eval")
    ap.add_argument("--labels", nargs="*", default=DR1_7B_FAMILY,
                    help="Labels (in row order) to include.")
    ap.add_argument("--format", choices=["grid", "latex"], default="grid")
    args = ap.parse_args()

    data = load_metrics(Path(args.eval_dir))
    missing = [l for l in args.labels if l not in data]
    if missing:
        print(f"# WARN: no metrics yet for: {missing}", file=sys.stderr)

    if args.format == "latex":
        print_latex(data, args.labels)
    else:
        print_grid(data, args.labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
