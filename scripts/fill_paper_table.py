#!/usr/bin/env python3
"""Safely fill one row of the ICLR accuracy--length table.

The command dry-runs by default. It accepts only complete canonical single-
response artifacts that passed strict rescoring and whose details SHA-256 still
matches the recorded audit.

Example:
    python scripts/fill_paper_table.py \
      --label public_qwen3_8b --row Qwen3-8B \
      --expected-prompt-role system_user \
      --expected-prompt-profile task_system \
      --expected-response-envelope full_think
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
import os


BENCHMARKS = ("gsm8k", "math500", "olympiadbench", "aime2024")
EXPECTED_ITEMS = {
    "gsm8k": 1319,
    "math500": 500,
    "olympiadbench": 675,
    "aime2024": 30,
}
STRICT_MATH_PROTOCOL = (
    "last_nonempty_box_else_explicit_final_answer_math_verify_gold_first"
)
DEFAULT_EVAL_DIR = Path(
    os.path.expandvars("${EXP_ROOT}/canonical/eval")
)
DEFAULT_TEX = Path(
    os.path.expandvars("${EXP_ROOT}/tables/main_accuracy.tex")
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_canonical_metrics(
    label: str,
    eval_dir: Path,
    expected_prompt_role: str,
    expected_prompt_profile: str,
    expected_response_envelope: str,
) -> tuple[dict[str, tuple[float, float]], list[str]]:
    values: dict[str, tuple[float, float]] = {}
    pending: list[str] = []
    for benchmark in BENCHMARKS:
        prefix = f"{label}_{benchmark}"
        metrics_path = eval_dir / f"{prefix}_metrics.json"
        details_path = eval_dir / f"{prefix}_details.jsonl"
        if not metrics_path.is_file() or not details_path.is_file():
            pending.append(benchmark)
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        audit = metrics.get("rescore_audit")
        expected_n = EXPECTED_ITEMS[benchmark]
        errors = []
        if int(metrics.get("provenance_schema_version", 0)) < 3:
            errors.append("provenance_schema_version<3")
        if metrics.get("math_scoring_protocol") != STRICT_MATH_PROTOCOL:
            errors.append("non-canonical math scoring")
        if int(metrics.get("num_samples_per_item", 0)) != 1:
            errors.append("num_samples_per_item!=1")
        if [int(value) for value in metrics.get("k_values", [])] != [1]:
            errors.append("k_values!=[1]")
        if int(metrics.get("n_items", -1)) != expected_n:
            errors.append(f"n_items!={expected_n}")
        if metrics.get("prompt_role_protocol") != expected_prompt_role:
            errors.append(f"prompt_role_protocol!={expected_prompt_role}")
        if metrics.get("prompt_profile") != expected_prompt_profile:
            errors.append(f"prompt_profile!={expected_prompt_profile}")
        if metrics.get("response_envelope") != expected_response_envelope:
            errors.append(f"response_envelope!={expected_response_envelope}")
        provenance = metrics.get("provenance", {})
        if provenance.get("prompt_profile") != metrics.get("prompt_profile"):
            errors.append("nested prompt_profile mismatch")
        if provenance.get("response_envelope") != metrics.get("response_envelope"):
            errors.append("nested response_envelope mismatch")
        if not isinstance(audit, dict):
            errors.append("missing rescore_audit")
        else:
            if int(audit.get("rows", -1)) != expected_n:
                errors.append(f"audit.rows!={expected_n}")
            if int(audit.get("unique_item_ids", -1)) != expected_n:
                errors.append(f"audit.unique_item_ids!={expected_n}")
            recorded_sha = str(audit.get("details_sha256", ""))
            if not recorded_sha or recorded_sha != sha256_file(details_path):
                errors.append("details SHA-256 mismatch")
        if errors:
            raise ValueError(f"{metrics_path}: " + "; ".join(errors))
        values[benchmark] = (
            float(metrics["accuracy_pct"]),
            float(metrics["mean_tokens_pass1"]),
        )
    return values, pending


def split_tex_cells(body: str) -> list[str]:
    r"""Split a TeX row body on ``&`` while retaining escaped ``\&``."""
    cells: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(body):
        if body[index : index + 2] == r"\&":
            current.append(r"\&")
            index += 2
        elif body[index] == "&":
            cells.append("".join(current))
            current = []
            index += 1
        else:
            current.append(body[index])
            index += 1
    cells.append("".join(current))
    return cells


def locate_row(lines: list[str], row_key: str) -> tuple[int, int]:
    starts = [
        index
        for index, line in enumerate(lines)
        if line.strip().startswith(row_key)
        and not line.lstrip().startswith("%")
    ]
    if len(starts) != 1:
        raise ValueError(
            f"Expected one row start containing {row_key!r}, found {len(starts)}"
        )
    start = starts[0]
    for end in range(start, min(start + 8, len(lines))):
        if r"\\" in lines[end]:
            return start, end
    raise ValueError(f"Could not find row terminator after {row_key!r}")


def fmt_accuracy(value: float) -> str:
    return f"{value:.1f}"


def fmt_tokens(value: float) -> str:
    return f"{round(value):,d}"


def update_row(
    tex: str,
    row_key: str,
    values: dict[str, tuple[float, float]],
) -> tuple[str, list[tuple[str, str, str]]]:
    lines = tex.splitlines(keepends=True)
    start, end = locate_row(lines, row_key)
    block = "".join(lines[start : end + 1])
    body, terminator = block.rsplit(r"\\", 1)
    cells = split_tex_cells(body)
    if len(cells) != 11:
        raise ValueError(
            f"Expected 11 cells in {row_key!r}, found {len(cells)}"
        )
    prefix = cells[0].strip()
    output_cells = [cell.strip() for cell in cells[1:]]
    changes: list[tuple[str, str, str]] = []
    for index, benchmark in enumerate(BENCHMARKS):
        if benchmark not in values:
            continue
        accuracy, tokens = values[benchmark]
        for cell_index, replacement, label in (
            (2 * index, fmt_accuracy(accuracy), f"{benchmark}.acc"),
            (2 * index + 1, fmt_tokens(tokens), f"{benchmark}.tok"),
        ):
            changes.append((label, output_cells[cell_index], replacement))
            output_cells[cell_index] = replacement

    if len(values) == len(BENCHMARKS):
        macro_accuracy = sum(value[0] for value in values.values()) / len(values)
        macro_tokens = sum(value[1] for value in values.values()) / len(values)
        for cell_index, replacement, label in (
            (8, fmt_accuracy(macro_accuracy), "macro.acc"),
            (9, fmt_tokens(macro_tokens), "macro.tok"),
        ):
            changes.append((label, output_cells[cell_index], replacement))
            output_cells[cell_index] = replacement

    new_block = (
        f"{prefix}\n"
        f"& {output_cells[0]} & {output_cells[1]} "
        f"& {output_cells[2]} & {output_cells[3]}\n"
        f"& {output_cells[4]} & {output_cells[5]} "
        f"& {output_cells[6]} & {output_cells[7]}\n"
        f"& {output_cells[8]} & {output_cells[9]} \\\\{terminator}"
    )
    lines[start : end + 1] = [new_block]
    return "".join(lines), changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--row", required=True)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument(
        "--expected-prompt-role",
        required=True,
        choices=("system_user", "user_only"),
    )
    parser.add_argument("--expected-prompt-profile", required=True)
    parser.add_argument(
        "--expected-response-envelope",
        required=True,
        choices=("full_think", "prefilled_think", "plain_text"),
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    values, pending = load_canonical_metrics(
        args.label,
        args.eval_dir,
        args.expected_prompt_role,
        args.expected_prompt_profile,
        args.expected_response_envelope,
    )
    if not values:
        print(f"[pending] no complete canonical artifacts for {args.label}")
        return 1
    tex = args.tex.read_text(encoding="utf-8")
    updated, changes = update_row(tex, args.row, values)
    print(f"label={args.label} row={args.row!r} tex={args.tex}")
    for label, old, new in changes:
        print(f"  {label:<22} {old:>24} -> {new}")
    if pending:
        print("  pending: " + ", ".join(pending))
    if args.write:
        temporary = args.tex.with_suffix(args.tex.suffix + ".partial")
        temporary.write_text(updated, encoding="utf-8")
        temporary.replace(args.tex)
        print(f"[write] updated {args.tex}")
    else:
        print("(dry-run; pass --write to apply)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(2)
