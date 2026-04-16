from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

PUBLIC_BENCHMARKS = (
    "gpqa_diamond",
    "olympiadbench",
    "math500",
    "omni_math",
    "aime2024",
    "cnmo2024",
    "livecode",
    "gsm8k",
    "mmlu",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: Any, d: int = 1) -> str:
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return "TBD"


def _replace_block(text: str, begin: str, end: str, new_block: str) -> str:
    if begin in text and end in text:
        left = text.split(begin, 1)[0]
        right = text.split(end, 1)[1]
        return f"{left}{begin}\n{new_block}\n{end}{right}"
    return f"{text}\n{begin}\n{new_block}\n{end}\n"


def _build_private_block(summary: dict[str, Any]) -> str:
    rows = summary.get("private_models", [])
    if not rows:
        return "% No private models available yet."

    # Keep TopoPRM-ablation-relevant rows first.
    priority_keys = [
        "grpo_hierarchical_qwen35_9b_light200",
        "grpo_no_topo_qwen35_9b_light200",
        "grpo_no_continuity_qwen35_9b_light200",
        "grpo_outcome_only_qwen35_9b_light200",
        "grpo_hierarchical_qwen25_7b_light200",
    ]

    bucket = {r["model"]: r for r in rows if "model" in r}
    ordered = [bucket[k] for k in priority_keys if k in bucket]
    ordered += [r for r in rows if r["model"] not in {x["model"] for x in ordered}]

    lines = ["% Auto-synced private snapshot (do not cite as final table body directly)."]
    for r in ordered[:10]:
        lines.append(
            f"% {r['model']}: mid_acc={_fmt(r.get('mid_acc'), 4)}, high_acc={_fmt(r.get('high_acc'), 4)}, "
            f"mid_tok={_fmt(r.get('mid_tokens'), 0)}, high_tok={_fmt(r.get('high_tokens'), 0)}"
        )
    return "\n".join(lines)


def _build_public_block(eval_dir: Path) -> str:
    lines = ["% Auto-synced public benchmark snapshot from output/eval."]
    metric_files = sorted(eval_dir.glob("*_metrics.json"))
    if not metric_files:
        lines.append("% no *_metrics.json found.")
        return "\n".join(lines)

    by_label: dict[str, dict[str, dict[str, Any]]] = {}
    for f in metric_files:
        stem = f.stem
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
        by_label.setdefault(label, {})[bench] = _read_json(f)

    for label in sorted(by_label.keys())[:16]:
        bench_map = by_label[label]
        if not bench_map:
            continue
        p1 = []
        p5 = []
        prm5 = []
        toks = []
        for bench, v in bench_map.items():
            p1.append(float(v.get("pass@1", v.get("accuracy", 0.0))))
            p5.append(float(v.get("pass@5", v.get("pass@1", 0.0))))
            prm5.append(float(v.get("prm@5", 0.0)))
            toks.append(float(v.get("avg_tokens", v.get("avg_prediction_tokens", 0.0))))
            lines.append(
                f"% {label}/{bench}: pass@1={_fmt(v.get('pass@1', v.get('accuracy')), 4)}, "
                f"pass@5={_fmt(v.get('pass@5', v.get('pass@1')), 4)}, "
                f"maj@5={_fmt(v.get('maj@5'), 4)}, prm@5={_fmt(v.get('prm@5'), 4)}, "
                f"f1={_fmt(v.get('f1'), 4)}, tok={_fmt(v.get('avg_tokens', v.get('avg_prediction_tokens')), 1)}"
            )
        if p1:
            lines.append(
                f"% {label}/macro: pass@1={_fmt(sum(p1)/len(p1), 4)}, "
                f"pass@5={_fmt(sum(p5)/len(p5), 4)}, prm@5={_fmt(sum(prm5)/len(prm5), 4)}, "
                f"tok={_fmt(sum(toks)/len(toks), 1)}"
            )
    return "\n".join(lines)


def _append_progress(progress_file: Path, summary: dict[str, Any]) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "",
        f"## Auto-sync snapshot ({ts})",
        f"- Source summary: `output/analysis/experiment_summary.json`",
        f"- Task status counts: `{summary.get('status_count', {})}`",
    ]
    rows = summary.get("private_models", [])
    if rows:
        best = max(rows, key=lambda x: float(x.get("overall_acc", 0.0)))
        lines.append(
            f"- Best private overall acc: `{best['model']}` = `{_fmt(best.get('overall_acc'), 4)}` "
            f"(tokens={_fmt(best.get('overall_tokens'), 1)})."
        )
    else:
        lines.append("- No private eval rows available in this sync.")

    unified = summary.get("unified_benchmarks", {}) if isinstance(summary, dict) else {}
    macro_rows = unified.get("model_macro", []) if isinstance(unified, dict) else []
    if macro_rows:
        best_macro = max(macro_rows, key=lambda x: float(x.get("macro_pass@1", 0.0)))
        lines.append(
            f"- Best unified macro pass@1: `{best_macro['label']}` = `{_fmt(best_macro.get('macro_pass@1'), 4)}` "
            f"(macro pass@5={_fmt(best_macro.get('macro_pass@5'), 4)}, macro prm@5={_fmt(best_macro.get('macro_prm@5'), 4)})."
        )
    else:
        lines.append("- No unified benchmark macro rows available in this sync.")

    with progress_file.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync collected summaries into paper/progress files.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--paper_dir", type=Path, default=Path("topoprm_paper"))
    parser.add_argument("--progress_file", type=Path, default=Path("output/analysis/experiment_progress.md"))
    parser.add_argument("--report", type=Path, default=Path("output/analysis/paper_sync_report.md"))
    parser.add_argument("--eval_dir", type=Path, default=Path("output/eval"))
    args = parser.parse_args()

    summary = _read_json(args.summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    private_tex = args.paper_dir / "tables" / "private_results.tex"
    public_tex = args.paper_dir / "tables" / "public_results.tex"
    ablation_tex = args.paper_dir / "tables" / "ablation_reward.tex"

    private_text = private_tex.read_text(encoding="utf-8") if private_tex.exists() else ""
    public_text = public_tex.read_text(encoding="utf-8") if public_tex.exists() else ""
    ablation_text = ablation_tex.read_text(encoding="utf-8") if ablation_tex.exists() else ""

    private_block = _build_private_block(summary)
    public_block = _build_public_block(args.eval_dir)
    ablation_block = "% Auto-synced note: ablation deltas should be recomputed from experiment_summary before camera-ready."

    private_text = _replace_block(private_text, "% AUTO_SYNC_PRIVATE_BEGIN", "% AUTO_SYNC_PRIVATE_END", private_block)
    public_text = _replace_block(public_text, "% AUTO_SYNC_PUBLIC_BEGIN", "% AUTO_SYNC_PUBLIC_END", public_block)
    ablation_text = _replace_block(ablation_text, "% AUTO_SYNC_ABLATION_BEGIN", "% AUTO_SYNC_ABLATION_END", ablation_block)

    if private_tex.exists():
        private_tex.write_text(private_text, encoding="utf-8")
    if public_tex.exists():
        public_tex.write_text(public_text, encoding="utf-8")
    if ablation_tex.exists():
        ablation_tex.write_text(ablation_text, encoding="utf-8")

    _append_progress(args.progress_file, summary)

    report_lines = [
        "# Paper Sync Report",
        "",
        f"- summary: `{args.summary}`",
        f"- private table updated: `{private_tex}`",
        f"- public table updated: `{public_tex}`",
        f"- ablation table updated: `{ablation_tex}`",
        f"- progress appended: `{args.progress_file}`",
    ]
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[sync] wrote {args.report}")


if __name__ == "__main__":
    main()

