"""Export quick paper-ready summaries from eval artifacts."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper table csv from eval json")
    parser.add_argument("--eval_dir", type=Path, default=Path("output/eval"))
    parser.add_argument("--output", type=Path, default=Path("output/eval/paper_table_summary.csv"))
    args = parser.parse_args()

    rows = ["model,middle_acc,high_acc,format,continuity,avg_len"]
    for f in sorted(args.eval_dir.glob("*_middle_metrics.json")):
        name = f.name.replace("_middle_metrics.json", "")
        middle = _read_json(f)
        high = _read_json(args.eval_dir / f"{name}_high_metrics.json")
        row = [
            name,
            str(_pick(middle, "accuracy", "score_accuracy")),
            str(_pick(high, "accuracy", "score_accuracy")),
            str(_pick(middle, "format_compliance", "format_rate")),
            str(_pick(middle, "step_coverage", "continuity")),
            str(_pick(middle, "avg_length", "mean_length")),
        ]
        rows.append(",".join(row))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Saved summary -> {args.output}")


if __name__ == "__main__":
    main()
