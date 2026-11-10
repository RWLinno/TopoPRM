"""Export paper-ready summaries from eval artifacts.

Outputs both CSV and LaTeX table fragments for direct inclusion in the paper.
Covers an internal held-out critique set (not part of the public benchmarks
reported here) and the public math benchmarks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _pick(d: dict[str, Any], *keys: str, default: Any = "TBD") -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return default


def _fmt(v: Any, decimals: int = 1) -> str:
    if v == "TBD" or v is None:
        return "TBD"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _metric_num(report: dict[str, Any]) -> Any:
    metrics = report.get("metrics")
    if isinstance(metrics, list) and metrics:
        first = metrics[0]
        if isinstance(first, dict):
            return first.get("num", "TBD")
    return "TBD"


def _collect_private(eval_dir: Path) -> list[dict[str, str]]:
    """Collect results on the internal held-out critique set."""
    rows = []
    for f in sorted(eval_dir.glob("*_middle_metrics.json")):
        name = f.name.replace("_middle_metrics.json", "")
        middle = _read_json(f)
        high = _read_json(eval_dir / f"{name}_high_metrics.json")
        rows.append({
            "model": name,
            "mid_acc": _fmt(_pick(middle, "score_accuracy", "accuracy")),
            "mid_f1": _fmt(_pick(middle, "error_identification_f1", "error_f1")),
            "mid_precision": _fmt(_pick(middle, "error_identification_precision")),
            "mid_recall": _fmt(_pick(middle, "error_identification_recall")),
            "mid_format": _fmt(_pick(middle, "format_compliance", "format_rate")),
            "mid_len": _fmt(_pick(middle, "avg_prediction_tokens", "avg_length", "mean_length"), 0),
            "high_acc": _fmt(_pick(high, "score_accuracy", "accuracy")),
            "high_f1": _fmt(_pick(high, "error_identification_f1", "error_f1")),
            "high_precision": _fmt(_pick(high, "error_identification_precision")),
            "high_recall": _fmt(_pick(high, "error_identification_recall")),
            "high_format": _fmt(_pick(high, "format_compliance", "format_rate")),
            "high_len": _fmt(_pick(high, "avg_prediction_tokens", "avg_length", "mean_length"), 0),
            "n_middle": _fmt(_pick(middle, "num_samples"), 0),
            "n_high": _fmt(_pick(high, "num_samples"), 0),
        })
    return rows


def _collect_public(eval_dir: Path) -> list[dict[str, str]]:
    """Collect public benchmark results (GSM8K, MATH-500)."""
    rows = []
    seen: set[str] = set()
    for f in sorted(eval_dir.glob("*_gsm8k_metrics.json")):
        name = f.name.replace("_gsm8k_metrics.json", "")
        gsm = _read_json(f)
        math = _read_json(eval_dir / f"{name}_math500_metrics.json")
        seen.add(name)
        rows.append({
            "model": name,
            "gsm8k_acc": _fmt(_pick(gsm, "accuracy", "acc")),
            "gsm8k_len": _fmt(_pick(gsm, "avg_length", "mean_length"), 0),
            "gsm8k_n": _fmt(_pick(gsm, "num_samples", "n"), 0),
            "math500_acc": _fmt(_pick(math, "accuracy", "acc")),
            "math500_len": _fmt(_pick(math, "avg_length", "mean_length"), 0),
            "math500_n": _fmt(_pick(math, "num_samples", "n"), 0),
        })

    # Fallback: parse swift benchmark_light reports when *_metrics.json is absent.
    benchmark_root = eval_dir / "benchmark_light"
    if benchmark_root.exists():
        grouped: dict[str, dict[str, Any]] = {}
        for run_dir in sorted(benchmark_root.glob("*")):
            if not run_dir.is_dir():
                continue
            run_name = run_dir.name
            base = run_name
            if "_gsm8k" in base:
                base = base.split("_gsm8k")[0]
            if "_math500" in base:
                base = base.split("_math500")[0]
            grouped.setdefault(base, {})
            gsm_report = next(iter(run_dir.glob("**/reports/**/gsm8k.json")), None)
            math_report = next(iter(run_dir.glob("**/reports/**/math_500.json")), None)
            if gsm_report is not None:
                grouped[base]["gsm"] = _read_json(gsm_report)
            if math_report is not None:
                grouped[base]["math"] = _read_json(math_report)

        for base, pair in grouped.items():
            if base in seen:
                continue
            gsm = pair.get("gsm", {})
            math = pair.get("math", {})
            rows.append({
                "model": base,
                "gsm8k_acc": _fmt(_pick(gsm, "score", "accuracy", "acc")) if gsm else "TBD",
                "gsm8k_len": "TBD",
                "gsm8k_n": _fmt(_pick(gsm, "num_samples", "n", default=_metric_num(gsm)), 0) if gsm else "TBD",
                "math500_acc": _fmt(_pick(math, "score", "accuracy", "acc")) if math else "TBD",
                "math500_len": "TBD",
                "math500_n": _fmt(_pick(math, "num_samples", "n", default=_metric_num(math)), 0) if math else "TBD",
            })
            seen.add(base)
    return rows


def _write_csv(rows: list[dict[str, str]], path: Path, label: str) -> None:
    if not rows:
        return
    header = list(rows[0].keys())
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(r.get(k, "TBD") for k in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [{label}] {len(rows)} models -> {path}")


def _write_latex_private(rows: list[dict[str, str]], path: Path) -> None:
    """Write a LaTeX table fragment for the internal held-out set
    (not part of the public benchmarks reported here)."""
    lines = [
        "% Auto-generated by export_paper_tables.py",
        "% Model & Mid-Acc & Mid-F1 & Mid-Fmt & Mid-Len & High-Acc & High-F1 & High-Fmt & High-Len \\\\",
    ]
    for r in rows:
        line = (
            f"{r['model']} & {r['mid_acc']} & {r['mid_f1']} & {r['mid_format']} & {r['mid_len']}"
            f" & {r['high_acc']} & {r['high_f1']} & {r['high_format']} & {r['high_len']} \\\\"
        )
        lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [LaTeX private] -> {path}")


def _write_latex_public(rows: list[dict[str, str]], path: Path) -> None:
    """Write a LaTeX table fragment for public benchmarks."""
    lines = [
        "% Auto-generated by export_paper_tables.py",
        "% Model & GSM8K-Acc & GSM8K-Len & MATH500-Acc & MATH500-Len \\\\",
    ]
    for r in rows:
        line = (
            f"{r['model']} & {r['gsm8k_acc']} & {r['gsm8k_len']}"
            f" & {r['math500_acc']} & {r['math500_len']} \\\\"
        )
        lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [LaTeX public] -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper table summaries from eval JSON files")
    parser.add_argument("--eval_dir", type=Path, default=Path("output/eval"))
    parser.add_argument("--output", type=Path, default=Path("output/eval/paper_table_summary.csv"))
    args = parser.parse_args()

    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting paper tables...")

    # Private benchmarks
    private_rows = _collect_private(args.eval_dir)
    _write_csv(private_rows, args.output, "private CSV")
    _write_latex_private(private_rows, out_dir / "private_results_fragment.tex")

    # Public benchmarks
    public_rows = _collect_public(args.eval_dir)
    _write_csv(public_rows, out_dir / "public_benchmark_summary.csv", "public CSV")
    _write_latex_public(public_rows, out_dir / "public_results_fragment.tex")

    print("Done.")


if __name__ == "__main__":
    main()
