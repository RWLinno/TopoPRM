#!/usr/bin/env python3
"""Fill the DR1-7B family rows of public_results_unified.tex from
measured output/eval/*_metrics.json files.

Idempotent: replaces "--" placeholder cells with measured pass@1 values
while preserving existing \\textbf and \\underline markers.

Only rows inside the "DR1-7B family" block are modified (the block header
contains the phrase "DR1-7B family" -- rows before this marker are kept as-is).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import os

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "output" / "eval"
TABLE = REPO / "output" / "tables" / "public_results_unified.tex"

# Column index is 0-based within the cell-list, where cell 0 = Model, cell 1 = Params.
BENCH_ORDER = [
    ("gsm8k",         2),
    ("math500",       3),
    ("olympiadbench", 4),
    ("omni_math",     5),
    ("aime2024",      6),
    ("aime2025",      7),
    ("cnmo2024",      8),
    ("mmlu",          9),
    ("gpqa_diamond", 10),
]

# Row model-name prefix (first cell text, stripped) -> metrics label
ROWS = {
    "DR1-7B (base, measured)":                  "baseline_dr1_7b_chat",
    "\\quad + SFT":                              "sft_dr1_7b",
    "\\quad + GRPO (outcome-only)":              "grpo_outcome_only",
    "\\quad + GRPO (w/o topology)":              "grpo_no_topo",
    "\\quad + GRPO (w/o continuity)":            "grpo_no_continuity",
    "\\quad + TopoPRM (hierarchical)":           "topoprm_full_dr1_7b",
}


def load_metrics(label):
    out = {}
    for bench, _ in BENCH_ORDER:
        p = EVAL_DIR / f"{label}_{bench}_metrics.json"
        if not p.exists():
            continue
        try:
            m = json.loads(p.read_text())
        except Exception:
            continue
        v = m.get("pass@1") or m.get("pass_at_1") or m.get("accuracy")
        if v is None:
            continue
        out[bench] = round(float(v) * 100.0, 1)
    return out


_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")


def cell_has_number(cell: str) -> bool:
    """Return True if the cell contains a real decimal (not --)."""
    stripped = cell.strip()
    if stripped.startswith("--") or stripped == "---":
        return False
    return bool(_NUM_RE.search(stripped))


def cell_number(cell: str) -> float:
    m = _NUM_RE.search(cell)
    return float(m.group(1)) if m else None


def replace_cell_number(cell: str, new_value: float) -> str:
    """Replace the first decimal number in a cell while preserving \\textbf/\\underline markup."""
    new_str = f"{new_value:.1f}"
    return _NUM_RE.sub(new_str, cell, count=1)


def split_row(line: str):
    """Split a LaTeX row into cells; keep trailing \\\\."""
    # Strip the line ending & trailing "\\"
    m = re.match(r"^(.*?)(\s*\\\\\s*)$", line.rstrip("\n"))
    if not m:
        return None, None, line
    body, tail = m.group(1), m.group(2)
    cells = body.split("&")
    return cells, tail, None


def join_row(cells, tail):
    return "&".join(cells) + tail + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="Modify the .tex file in place")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite cells even if they already have numeric values")
    args = ap.parse_args()

    if not TABLE.exists():
        raise SystemExit(f"table not found: {TABLE}")

    raw = TABLE.read_text()
    lines = raw.splitlines(keepends=True)

    # Find DR1-7B family block start (header line that's actually typeset, not a comment)
    dr1_block_start = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("%"):
            continue
        if "DR1-7B family" in ln:
            dr1_block_start = i
            break
    if dr1_block_start is None:
        raise SystemExit("could not find 'DR1-7B family' block marker")

    changes = []
    for prefix, label in ROWS.items():
        metrics = load_metrics(label)
        if not metrics:
            print(f"[SKIP] {prefix}: no metrics found (label={label})")
            continue
        found_idx = None
        for i in range(dr1_block_start, len(lines)):
            stripped = lines[i].lstrip()
            if "\\bottomrule" in lines[i] or "Ours: Distilled" in lines[i]:
                break
            if stripped.startswith(prefix + " ") or stripped.startswith(prefix + "\t") or stripped.startswith(prefix + "&"):
                if "&" not in lines[i]:
                    continue
                found_idx = i
                break
        if found_idx is None:
            print(f"[WARN] row not found for prefix '{prefix}' in DR1-7B block")
            continue

        cells, tail, bad = split_row(lines[found_idx])
        if cells is None:
            print(f"[WARN] could not parse line {found_idx+1}: {bad!r}")
            continue

        touched = False
        for bench, col in BENCH_ORDER:
            if col >= len(cells):
                continue
            v = metrics.get(bench)
            if v is None:
                continue
            cell = cells[col]
            if cell_has_number(cell) and not args.force:
                # Already filled from a previous run; only replace if the number differs significantly.
                existing = cell_number(cell)
                if existing is not None and abs(existing - v) < 0.05:
                    continue
                new_cell = replace_cell_number(cell, v)
            elif cell_has_number(cell) and args.force:
                new_cell = replace_cell_number(cell, v)
            else:
                # Blank (--) cell; preserve surrounding whitespace.
                leading = re.match(r"^(\s*)", cell).group(1)
                trailing = re.search(r"(\s*)$", cell).group(1)
                new_cell = f"{leading}{v:.1f}{trailing}"
            if new_cell != cell:
                cells[col] = new_cell
                touched = True

        if touched:
            new_line = join_row(cells, tail)
            changes.append((found_idx, lines[found_idx], new_line))
            lines[found_idx] = new_line

    if not changes:
        print("[INFO] no changes needed")
        return

    for idx, old, new in changes:
        print(f"\n--- line {idx+1} ---")
        print(f"- {old.rstrip()}")
        print(f"+ {new.rstrip()}")

    if args.write:
        TABLE.write_text("".join(lines))
        print(f"\n[WROTE] {TABLE}")
    else:
        print("\n[DRY-RUN] use --write to apply")


if __name__ == "__main__":
    main()
