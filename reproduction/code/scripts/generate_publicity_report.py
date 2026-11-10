#!/usr/bin/env python3
"""Generate publicity-ready JSON/Markdown summaries from eval artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_metrics(eval_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fp in sorted(glob.glob(str(eval_dir / "*_metrics.json"))):
        d = _load_json(Path(fp))
        if not d:
            continue
        rows.append(
            {
                "file": Path(fp).name,
                "score_accuracy": float(d.get("score_accuracy", 0.0) or 0.0),
                "format_compliance": float(d.get("format_compliance", 0.0) or 0.0),
                "step_coverage": float(d.get("step_coverage", 0.0) or 0.0),
                "error_identification_f1": float(d.get("error_identification_f1", 0.0) or 0.0),
                "num_samples": int(d.get("num_samples", 0) or 0),
            }
        )
    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "num_runs": 0,
            "averages": {},
            "top_by_accuracy": [],
            "runs": [],
        }

    acc = [r["score_accuracy"] for r in rows]
    fmt = [r["format_compliance"] for r in rows]
    cov = [r["step_coverage"] for r in rows]
    top = sorted(rows, key=lambda r: r["score_accuracy"], reverse=True)[:10]

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "num_runs": len(rows),
        "averages": {
            "score_accuracy": mean(acc),
            "format_compliance": mean(fmt),
            "step_coverage": mean(cov),
        },
        "top_by_accuracy": top,
        "runs": rows,
    }


def write_outputs(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# TopoPRM Publicity Summary",
        "",
        f"- generated_at: {summary.get('generated_at')}",
        f"- num_runs: {summary.get('num_runs', 0)}",
        "",
    ]
    avg = summary.get("averages", {})
    if avg:
        lines += [
            "## Averages",
            "",
            f"- score_accuracy: {avg.get('score_accuracy', 0.0):.4f}",
            f"- format_compliance: {avg.get('format_compliance', 0.0):.4f}",
            f"- step_coverage: {avg.get('step_coverage', 0.0):.4f}",
            "",
        ]

    lines += ["## Top Runs by Accuracy", ""]
    for r in summary.get("top_by_accuracy", []):
        lines.append(
            f"- `{r['file']}`: acc={r['score_accuracy']:.4f}, format={r['format_compliance']:.4f}, n={r['num_samples']}"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publicity report artifacts")
    parser.add_argument("--eval_dir", type=Path, default=Path("output/eval"))
    parser.add_argument("--output_dir", type=Path, default=Path("docs/publicity/data"))
    args = parser.parse_args()

    rows = collect_metrics(args.eval_dir)
    summary = build_summary(rows)
    json_path, md_path = write_outputs(summary, args.output_dir)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
