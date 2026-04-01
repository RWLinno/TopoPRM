"""Generate dataset/benchmark visualization and summary report."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_jsonl(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
            if max_rows is not None and (i + 1) >= max_rows:
                break
    return rows


def text_len_from_messages(row: dict[str, Any]) -> int:
    msgs = row.get("messages", [])
    if not isinstance(msgs, list):
        return 0
    total = 0
    for m in msgs:
        if isinstance(m, dict):
            total += len(str(m.get("content", "")))
    return total


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    c = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            c += 1
    return c


def benchmark_counts(bench_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not bench_root.exists():
        return counts
    for d in sorted(p for p in bench_root.iterdir() if p.is_dir()):
        total = 0
        for fp in d.glob("*.jsonl"):
            total += count_jsonl_lines(fp)
        counts[d.name] = total
    return counts


def plot_bar(data: dict[str, int], title: str, out_path: Path) -> None:
    if not data:
        return
    names = list(data.keys())
    vals = [data[k] for k in names]
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_hist(values: list[int], title: str, out_path: Path) -> None:
    if not values:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=40)
    plt.title(title)
    plt.xlabel("chars")
    plt.ylabel("count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    root = Path("/mnt/users/rwl/topoprm")
    out_dir = root / "output" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_root = root / "data" / "benchmarks"
    sft_aug = root / "data" / "sft_ready" / "train_augmented.jsonl"
    sft_mix = root / "data" / "sft_ready" / "train_mixed.jsonl"
    sft_senior = root / "data" / "sft_ready" / "infer_new_senior_50000_20260311.jsonl"
    sft_junior = root / "data" / "sft_ready" / "infer_new_junior_50000_20260311.jsonl"

    bcounts = benchmark_counts(bench_root)
    plot_bar(bcounts, "Benchmark sample counts", out_dir / "benchmark_counts.png")

    aug_rows = read_jsonl(sft_aug)
    mix_rows = read_jsonl(sft_mix)
    aug_lens = [text_len_from_messages(r) for r in aug_rows if text_len_from_messages(r) > 0]
    mix_lens = [text_len_from_messages(r) for r in mix_rows if text_len_from_messages(r) > 0]
    plot_hist(aug_lens, "SFT augmented message length", out_dir / "sft_augmented_length_hist.png")
    plot_hist(mix_lens, "SFT mixed message length", out_dir / "sft_mixed_length_hist.png")

    report = {
        "benchmark_counts": bcounts,
        "sft": {
            "train_augmented_rows": len(aug_rows),
            "train_augmented_avg_chars": round(mean(aug_lens), 2) if aug_lens else 0,
            "train_mixed_rows": len(mix_rows),
            "train_mixed_avg_chars": round(mean(mix_lens), 2) if mix_lens else 0,
            "new_senior_exists": sft_senior.exists(),
            "new_junior_exists": sft_junior.exists(),
            "new_senior_lines": count_jsonl_lines(sft_senior),
            "new_junior_lines": count_jsonl_lines(sft_junior),
        },
    }

    (out_dir / "data_profile.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Data Visualization Summary",
        "",
        "Generated files:",
        "- output/analysis/benchmark_counts.png",
        "- output/analysis/sft_augmented_length_hist.png",
        "- output/analysis/sft_mixed_length_hist.png",
        "- output/analysis/data_profile.json",
        "",
        "## Benchmark counts",
    ]
    for k, v in bcounts.items():
        md.append(f"- {k}: {v}")
    md += [
        "",
        "## SFT corpus",
        f"- train_augmented rows: {report['sft']['train_augmented_rows']}",
        f"- train_augmented avg chars: {report['sft']['train_augmented_avg_chars']}",
        f"- train_mixed rows: {report['sft']['train_mixed_rows']}",
        f"- train_mixed avg chars: {report['sft']['train_mixed_avg_chars']}",
        f"- new senior exists: {report['sft']['new_senior_exists']} (lines={report['sft']['new_senior_lines']})",
        f"- new junior exists: {report['sft']['new_junior_exists']} (lines={report['sft']['new_junior_lines']})",
    ]
    (out_dir / "data_profile.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("Saved analysis to", out_dir)


if __name__ == "__main__":
    main()
