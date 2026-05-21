#!/usr/bin/env python3
"""Aggregate Server B baseline metrics into results/baseline/.

Walks output/eval/<label>_<bench>_metrics.json (only labels carrying the
server_B suffix by default) and emits:
  - leaderboard_baseline.csv      one row per (label, bench)
  - metrics_full_baseline.json    full metrics nested by label/bench
  - eval_trace_baseline.md        chronology of runs (mtime + duration)
  - missing_cells_report.md       (label, bench) cells still empty
  - merge_manifest.md             how to join with results/method_v2/*

Usage:
    python scripts/aggregate_baseline_results_B.py \
        --eval_dir output/eval \
        --output_dir results/baseline \
        --label_suffix _B \
        [--emit_manifest]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

ALL_BENCHMARKS = [
    "gsm8k",
    "math500",
    "aime2024",
    "aime2025",
    "cnmo2024",
    "olympiadbench",
    "omni_math",
    "gpqa_diamond",
    "mmlu",
]

REQUIRED_METRICS = [
    "pass@1",
    "pass@5",
    "maj@5",
    "prm@5",
    "F1",
    "correct",
    "error",
    "avg_tokens",
]

METRICS_RE = re.compile(
    r"^(?P<label>.+)_(?P<bench>"
    + "|".join(re.escape(b) for b in ALL_BENCHMARKS)
    + r")_metrics\.json$"
)


def parse_metrics_filename(name: str) -> Optional[tuple[str, str]]:
    m = METRICS_RE.match(name)
    if not m:
        return None
    return m.group("label"), m.group("bench")


def safe_load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect(eval_dir: Path, suffix: str) -> dict[str, dict[str, dict]]:
    by_label: dict[str, dict[str, dict]] = defaultdict(dict)
    for metrics_file in sorted(eval_dir.glob("*_metrics.json")):
        parsed = parse_metrics_filename(metrics_file.name)
        if not parsed:
            continue
        label, bench = parsed
        if suffix and not label.endswith(suffix):
            continue
        data = safe_load(metrics_file)
        if not data:
            continue
        data["_path"] = str(metrics_file)
        data["_mtime"] = metrics_file.stat().st_mtime
        by_label[label][bench] = data
    return by_label


def write_leaderboard(by_label: dict, out_path: Path) -> int:
    rows = []
    for label in sorted(by_label):
        for bench in ALL_BENCHMARKS:
            entry = by_label[label].get(bench)
            if not entry:
                continue
            row = {
                "label": label,
                "benchmark": bench,
            }
            for key in REQUIRED_METRICS:
                row[key] = entry.get(key, "")
            row["elapsed_sec"] = entry.get("elapsed_sec", "")
            row["mtime_iso"] = datetime.fromtimestamp(entry["_mtime"]).isoformat(
                timespec="seconds"
            )
            rows.append(row)
    fieldnames = [
        "label",
        "benchmark",
        *REQUIRED_METRICS,
        "elapsed_sec",
        "mtime_iso",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def write_metrics_full(by_label: dict, out_path: Path) -> None:
    serialisable = {}
    for label, benches in by_label.items():
        serialisable[label] = {}
        for bench, entry in benches.items():
            clean = {k: v for k, v in entry.items() if not k.startswith("_")}
            serialisable[label][bench] = clean
    out_path.write_text(
        json.dumps(serialisable, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_eval_trace(by_label: dict, out_path: Path) -> None:
    flat = []
    for label, benches in by_label.items():
        for bench, entry in benches.items():
            flat.append((entry["_mtime"], label, bench, entry))
    flat.sort()
    lines = ["# eval_trace_baseline (server_B)\n"]
    for mtime, label, bench, entry in flat:
        ts = datetime.fromtimestamp(mtime).isoformat(timespec="seconds")
        p1 = entry.get("pass@1", "?")
        elapsed = entry.get("elapsed_sec", "?")
        lines.append(f"- {ts}  `{label}` * `{bench}`  pass@1={p1}  elapsed={elapsed}s")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_missing_cells(by_label: dict, out_path: Path,
                        expected_labels: Optional[list[str]] = None) -> None:
    lines = ["# missing_cells_report (server_B)\n"]
    lines.append("Cells still missing per `(label, benchmark)`:\n")
    seen = set(by_label.keys())
    expected = set(expected_labels or [])
    all_labels = sorted(seen | expected)
    any_missing = False
    for label in all_labels:
        present = by_label.get(label, {})
        missing = [b for b in ALL_BENCHMARKS if b not in present]
        if missing:
            any_missing = True
            lines.append(f"- `{label}`: {', '.join(missing)}")
    if not any_missing:
        lines.append("(none -- every tracked label has all 9 benchmarks)")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(out_path: Path, eval_dir: Path) -> None:
    body = f"""# merge_manifest (server_B)

This manifest tells server_A's `results/method_v2/*` how to join the
server_B baseline outputs.

## Source files (server_B)
- `results/baseline/leaderboard_baseline.csv` (one row per `(label, benchmark)`)
- `results/baseline/metrics_full_baseline.json` (full metrics nested)
- `results/baseline/eval_trace_baseline.md` (chronological run trace)
- `results/baseline/missing_cells_report.md`

## Raw evidence
- All raw `*_metrics.json` and `*_details.jsonl` live under `{eval_dir}`.
- Server_B labels carry the `_B` suffix (e.g. `sft_qwen35_9b_B`).

## Suggested join
Join key: `(label, benchmark)`. The server_A leaderboard
(`results/method_v2/leaderboard_method_v2.csv`) follows the same schema, so
a `csv.DictReader` over both files can be concatenated as-is.

When normalising labels for the paper table:
- strip the `_B` suffix when reporting headline numbers; keep the suffix in
  the audit trail to track which server produced each cell.
"""
    out_path.write_text(body, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default="output/eval")
    ap.add_argument("--output_dir", default="results/baseline")
    ap.add_argument("--label_suffix", default="_B",
                    help="only collect labels ending with this suffix")
    ap.add_argument("--expected_labels", nargs="*", default=None,
                    help="optional list of expected labels (used to surface "
                         "labels that have been launched but produced no metrics yet)")
    ap.add_argument("--emit_manifest", action="store_true")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_label = collect(eval_dir, args.label_suffix)
    rows = write_leaderboard(by_label, out_dir / "leaderboard_baseline.csv")
    write_metrics_full(by_label, out_dir / "metrics_full_baseline.json")
    write_eval_trace(by_label, out_dir / "eval_trace_baseline.md")
    write_missing_cells(
        by_label,
        out_dir / "missing_cells_report.md",
        expected_labels=args.expected_labels,
    )

    if args.emit_manifest:
        write_manifest(out_dir / "merge_manifest.md", eval_dir)

    print(
        f"[aggregate] labels={len(by_label)} rows={rows} suffix='{args.label_suffix}' "
        f"output_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
