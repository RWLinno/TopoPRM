import json
from pathlib import Path

from scripts.generate_publicity_report import build_summary, collect_metrics


def test_collect_metrics_and_build_summary(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    (eval_dir / "run_a_metrics.json").write_text(
        json.dumps({
            "score_accuracy": 0.3,
            "format_compliance": 0.9,
            "step_coverage": 0.1,
            "error_identification_f1": 0.0,
            "num_samples": 200,
        }),
        encoding="utf-8",
    )
    (eval_dir / "run_b_metrics.json").write_text(
        json.dumps({
            "score_accuracy": 0.5,
            "format_compliance": 0.8,
            "step_coverage": 0.2,
            "error_identification_f1": 0.0,
            "num_samples": 100,
        }),
        encoding="utf-8",
    )

    rows = collect_metrics(eval_dir)
    assert len(rows) == 2

    summary = build_summary(rows)
    assert summary["num_runs"] == 2
    assert summary["top_by_accuracy"][0]["score_accuracy"] >= summary["top_by_accuracy"][1]["score_accuracy"]
