from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

PUBLIC_BENCHMARKS = (
    "gpqa_diamond",
    "olympiadbench",
    "math500",
    "omni_math",
    "aime2024",
    "aime2025",
    "cnmo2024",
    "gsm8k",
    "mmlu",
)  # livecode dropped 2026-04-21 (math-only models score ~0%)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_registry_latest(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        task_id = obj.get("task_id")
        if task_id:
            latest[task_id] = obj
    return latest


def _collect_private(eval_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for middle_path in sorted(eval_dir.glob("*_middle_metrics.json")):
        prefix = middle_path.name.replace("_middle_metrics.json", "")
        high_path = eval_dir / f"{prefix}_high_metrics.json"
        middle = _read_json(middle_path)
        high = _read_json(high_path)
        if not high:
            continue

        m_acc = float(middle.get("score_accuracy", middle.get("accuracy", 0.0)))
        h_acc = float(high.get("score_accuracy", high.get("accuracy", 0.0)))
        m_recall = float(middle.get("error_identification_recall", 0.0))
        h_recall = float(high.get("error_identification_recall", 0.0))
        m_f1 = float(middle.get("error_identification_f1", 0.0))
        h_f1 = float(high.get("error_identification_f1", 0.0))
        m_tok = float(middle.get("avg_prediction_tokens", middle.get("avg_length", 0.0)))
        h_tok = float(high.get("avg_prediction_tokens", high.get("avg_length", 0.0)))
        if m_acc == 0.0 and h_acc == 0.0 and m_tok == 0.0 and h_tok == 0.0:
            continue

        rows.append(
            {
                "model": prefix,
                "mid_acc": m_acc,
                "mid_recall": m_recall,
                "mid_f1": m_f1,
                "mid_tokens": m_tok,
                "high_acc": h_acc,
                "high_recall": h_recall,
                "high_f1": h_f1,
                "high_tokens": h_tok,
                "overall_acc": (m_acc + h_acc) / 2.0,
                "overall_tokens": (m_tok + h_tok) / 2.0,
                "acc_per_1k_tok": ((m_acc + h_acc) / 2.0) / max(((m_tok + h_tok) / 2.0) / 1000.0, 1e-6),
            }
        )
    return rows


def _collect_unified(eval_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for metrics_path in sorted(eval_dir.glob("*_metrics.json")):
        stem = metrics_path.stem
        if not stem.endswith("_metrics"):
            continue
        base = stem[:-8]
        label = ""
        bench = ""
        for b in PUBLIC_BENCHMARKS:
            suffix = f"_{b}"
            if base.endswith(suffix):
                label = base[: -len(suffix)]
                bench = b
                break
        if not label or not bench:
            continue
        data = _read_json(metrics_path)
        if not data:
            continue
        entry = {
            "label": label,
            "benchmark": bench,
            "pass@1": float(data.get("pass@1", 0.0)),
            "pass@5": float(data.get("pass@5", 0.0)),
            "maj@5": float(data.get("maj@5", 0.0)),
            "prm@5": float(data.get("prm@5", 0.0)),
            "f1": float(data.get("f1", 0.0)),
            "tokens": float(data.get("avg_tokens", data.get("avg_prediction_tokens", 0.0))),
            "correct": int(data.get("correct", 0)),
            "error": int(data.get("error", data.get("incorrect", 0))),
            "num_samples": int(data.get("num_samples", 0)),
        }
        rows.append(entry)
        by_model.setdefault(label, {})[bench] = entry

    model_macro: list[dict[str, Any]] = []
    for label, bench_map in sorted(by_model.items()):
        vals = list(bench_map.values())
        if not vals:
            continue
        model_macro.append(
            {
                "label": label,
                "benchmarks": len(vals),
                "macro_pass@1": mean(v["pass@1"] for v in vals),
                "macro_pass@5": mean(v["pass@5"] for v in vals),
                "macro_maj@5": mean(v["maj@5"] for v in vals),
                "macro_prm@5": mean(v["prm@5"] for v in vals),
                "macro_f1": mean(v["f1"] for v in vals),
                "macro_tokens": mean(v["tokens"] for v in vals),
            }
        )
    return {"rows": rows, "model_macro": model_macro}


def _build_observations(private_rows: list[dict[str, Any]], registry_latest: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Experiment Observations")
    lines.append("")

    if private_rows:
        best = max(private_rows, key=lambda x: x["overall_acc"])
        lines.append(
            f"- Best private overall accuracy: `{best['model']}` = `{best['overall_acc']:.4f}` "
            f"(tokens={best['overall_tokens']:.1f}, acc/1kTok={best['acc_per_1k_tok']:.3f})."
        )

        # Prioritize TopoPRM ablation deltas when present.
        full = next((r for r in private_rows if "hierarchical" in r["model"] or "main_full" in r["model"]), None)
        no_topo = next((r for r in private_rows if "no_topo" in r["model"]), None)
        no_cont = next((r for r in private_rows if "no_cont" in r["model"] or "no_continuity" in r["model"]), None)
        outcome = next((r for r in private_rows if "outcome_only" in r["model"] or "outcome" in r["model"]), None)
        if full and no_topo:
            lines.append(
                f"- Ablation: removing topology drops overall acc by `{full['overall_acc'] - no_topo['overall_acc']:.4f}`."
            )
        if full and no_cont:
            lines.append(
                f"- Ablation: removing continuity drops overall acc by `{full['overall_acc'] - no_cont['overall_acc']:.4f}`."
            )
        if full and outcome:
            lines.append(
                f"- Ablation: outcome-only drops overall acc by `{full['overall_acc'] - outcome['overall_acc']:.4f}`."
            )

    final_failed = [k for k, v in registry_latest.items() if v.get("status") == "failed_final"]
    if final_failed:
        lines.append(f"- Final failed tasks ({len(final_failed)}): `{', '.join(sorted(final_failed))}`.")
    else:
        lines.append("- No final-failed tasks in current registry snapshot.")

    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect registry/eval outputs into unified summaries.")
    parser.add_argument("--registry", type=Path, default=Path("output/analysis/experiment_registry.jsonl"))
    parser.add_argument("--eval_dir", type=Path, default=Path("output/eval"))
    parser.add_argument("--output_dir", type=Path, default=Path("output/analysis"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    registry_latest = _read_registry_latest(args.registry)
    private_rows = _collect_private(args.eval_dir)
    unified = _collect_unified(args.eval_dir)

    status_count: dict[str, int] = {}
    for event in registry_latest.values():
        key = str(event.get("status", "unknown"))
        status_count[key] = status_count.get(key, 0) + 1

    summary = {
        "registry_path": str(args.registry),
        "num_tasks_seen": len(registry_latest),
        "status_count": status_count,
        "private_models": private_rows,
        "private_overall_mean_acc": mean([r["overall_acc"] for r in private_rows]) if private_rows else None,
        "unified_benchmarks": unified,
    }

    summary_json = args.output_dir / "experiment_summary.json"
    summary_csv = args.output_dir / "experiment_summary.csv"
    unified_csv = args.output_dir / "unified_benchmark_summary.csv"
    obs_md = args.output_dir / "experiment_observations.md"

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(summary_csv, private_rows)
    _write_csv(unified_csv, unified.get("rows", []))
    obs_md.write_text(_build_observations(private_rows, registry_latest), encoding="utf-8")

    print(f"[collect] wrote {summary_json}")
    print(f"[collect] wrote {summary_csv}")
    print(f"[collect] wrote {unified_csv}")
    print(f"[collect] wrote {obs_md}")


if __name__ == "__main__":
    main()

