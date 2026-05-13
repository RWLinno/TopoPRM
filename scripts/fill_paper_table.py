#!/usr/bin/env python3
"""Aggregate per-bench metrics.json and fill the Qwen3.5-9B rows of
`topoprm_paper/tables/public_results_unified.tex` with real pass@1.

By default dry-runs and prints a diff; pass --write to modify the tex file.

Usage:
    python scripts/fill_paper_table.py --label qwen35_9b_base --row "Qwen3.5-9B (base)"
    python scripts/fill_paper_table.py --label qwen35_9b_sft --row "+ SFT" --write

Row matching is substring-based on the first column up to '&'. The nine columns
filled (in order) are:

    GSM8K | MATH-500 | Olympiad | Omni-MATH | AIME'24 | AIME'25 | CNMO'24 | MMLU | GPQA-D
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


BENCH_TO_COLUMN_IDX = {
    "gsm8k":         2,  # col index (1-based as LaTeX would write, here 0-based)
    "math500":       3,
    "olympiadbench": 4,
    "omni_math":     5,
    "aime2024":      6,
    "aime2025":      7,
    "cnmo2024":      8,
    "mmlu":          9,
    "gpqa_diamond":  10,
}
# public_results_unified.tex column layout (0-based):
#   0: Model name
#   1: Size
#   2: GSM8K
#   3: MATH-500
#   4: Olympiad
#   5: Omni-MATH
#   6: AIME'24
#   7: AIME'25
#   8: CNMO'24
#   9: MMLU (note: last header says MMLU-Pro, but our data column is "MMLU")
#  10: GPQA-D

TEX_PATH_DEFAULT = "topoprm_paper/tables/public_results_unified.tex"


def load_pass1(label: str, eval_dir: Path) -> dict[str, float]:
    out = {}
    for bench in BENCH_TO_COLUMN_IDX:
        p = eval_dir / f"{label}_{bench}_metrics.json"
        if p.exists():
            try:
                d = json.loads(p.read_text())
                p1 = d.get("pass@1") or d.get("accuracy") or 0.0
                out[bench] = float(p1) * 100.0
            except Exception as e:
                print(f"[warn] failed to parse {p}: {e}", file=sys.stderr)
    return out


def fmt_cell(val: float) -> str:
    return f"{val:.1f}"


def split_tex_cells(body: str) -> list[str]:
    r"""Split a tex row body on top-level '&' (ignoring '\&')."""
    cells = []
    cur = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body) and body[i + 1] == "&":
            cur.append(body[i:i + 2])
            i += 2
            continue
        if ch == "&":
            cells.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    cells.append("".join(cur))
    return cells


def join_cells(cells: list[str]) -> str:
    return "&".join(cells)


def update_row(tex: str, row_key: str, pass1: dict[str, float]) -> tuple[str, list[tuple[str, str, str]]]:
    """Return (new_tex, list_of (bench, old, new) for diff)."""
    lines = tex.splitlines(keepends=True)
    changes: list[tuple[str, str, str]] = []
    out_lines = []
    for line in lines:
        stripped = line.rstrip("\r\n")
        # Match any tex row whose first cell's text contains row_key.
        if "&" not in stripped or stripped.lstrip().startswith("%") or stripped.lstrip().startswith("\\"):
            # still might be a model row starting with \texttt{} etc.; accept
            # lines whose first column text includes row_key
            pass
        # heuristic: skip lines that are clearly not data rows
        if r"\multicolumn" in stripped or stripped.strip().startswith("\\midrule") or stripped.strip().startswith("\\bottomrule"):
            out_lines.append(line)
            continue
        if "\\\\" not in stripped:
            out_lines.append(line)
            continue
        # Separate the trailing \\ ... from the body
        m = re.match(r"^(.*?)(\s*\\\\.*?)$", stripped)
        if not m:
            out_lines.append(line)
            continue
        body, trailer = m.group(1), m.group(2)
        cells = split_tex_cells(body)
        # first column identifies the row
        first_col = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", cells[0]).strip()
        if row_key not in first_col:
            out_lines.append(line)
            continue
        if len(cells) < 11:
            out_lines.append(line)
            continue
        for bench, idx in BENCH_TO_COLUMN_IDX.items():
            if bench not in pass1:
                continue
            old_cell = cells[idx]
            new_val = fmt_cell(pass1[bench])
            # preserve any trailing spaces but replace the core content
            m2 = re.match(r"^(\s*)(.*?)(\s*)$", old_cell)
            if m2:
                leading, content, trailing = m2.group(1), m2.group(2), m2.group(3)
            else:
                leading, content, trailing = "", old_cell, ""
            changes.append((bench, content.strip(), new_val))
            cells[idx] = f"{leading}{new_val}{trailing}"
        new_body = join_cells(cells)
        # Preserve original line ending if any
        end = line[len(line.rstrip("\r\n")):]
        out_lines.append(new_body + trailer + end)
    return "".join(out_lines), changes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True, help="Run label whose metrics to load.")
    p.add_argument("--row", required=True,
                   help="First-column identifier to match, e.g. 'Qwen3.5-9B (base)' or '+ SFT'.")
    p.add_argument("--tex", default=TEX_PATH_DEFAULT)
    p.add_argument("--eval_dir", default="output/eval")
    p.add_argument("--write", action="store_true",
                   help="Write changes back to the tex file (otherwise dry-run).")
    args = p.parse_args(argv)

    tex_path = Path(args.tex)
    if not tex_path.exists():
        print(f"[error] tex not found: {tex_path}", file=sys.stderr)
        return 2
    pass1 = load_pass1(args.label, Path(args.eval_dir))
    if not pass1:
        print(f"[warn] no metrics.json found for label={args.label} under {args.eval_dir}")
        return 1
    tex = tex_path.read_text(encoding="utf-8")
    new_tex, changes = update_row(tex, args.row, pass1)
    if not changes:
        print(f"[warn] no row matching '{args.row}' updated; is the row label correct?")
        print("available benches with pass@1:")
        for b, v in pass1.items():
            print(f"  {b:<15} {v:5.1f}%")
        return 1
    print(f"label={args.label} row='{args.row}' tex={tex_path}")
    for bench, old, new in changes:
        print(f"  {bench:<15} {old:>10}  ->  {new}")
    if args.write:
        tex_path.write_text(new_tex, encoding="utf-8")
        print(f"[write] updated {tex_path}")
    else:
        print("(dry-run; pass --write to apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
