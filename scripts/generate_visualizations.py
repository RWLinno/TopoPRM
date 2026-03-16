#!/usr/bin/env python3
"""Generate all visualisation figures for the TopoPRM project."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
SFT_LOG = (
    PROJECT_ROOT
    / "output"
    / "sft_qwen3_14b"
    / "v0-20260313-195147"
    / "logging.jsonl"
)
DAG_DIR = PROJECT_ROOT / "data" / "dag"
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned.jsonl"

CATEGORY_ORDER = ["初中代数", "初中几何", "高中代数", "高中几何"]
CATEGORY_COLORS = ["#6baed6", "#74c476", "#fd8d3c", "#e6550d"]

# ---------------------------------------------------------------------------
# Font configuration — prefer WenQuanYi (available on this system),
# fall back to SimHei, then to DejaVu Sans.
# ---------------------------------------------------------------------------

def _setup_chinese_font() -> None:
    from matplotlib.font_manager import fontManager

    candidates = [
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
    ]
    available = {f.name for f in fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in available:
            chosen = name
            break

    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    else:
        print("[WARN] No CJK font found; Chinese text may render as boxes.")
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 1. SFT training curves
# ---------------------------------------------------------------------------

def plot_sft_training_curves() -> None:
    """Plot loss, token accuracy, and learning rate vs training step."""
    records = []
    with open(SFT_LOG) as f:
        for line in f:
            rec = json.loads(line)
            step_field = rec.get("global_step/max_steps", "")
            if "/" in str(step_field):
                step = int(str(step_field).split("/")[0])
                rec["step"] = step
                records.append(rec)

    if not records:
        print("[WARN] No training step records found in logging.jsonl")
        return

    seen = set()
    unique = []
    for r in records:
        if r["step"] not in seen:
            seen.add(r["step"])
            unique.append(r)
    records = sorted(unique, key=lambda r: r["step"])

    steps = [r["step"] for r in records]
    losses = [r["loss"] for r in records]
    accs = [r["token_acc"] for r in records]
    lrs = [r["learning_rate"] for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Loss
    ax = axes[0]
    ax.plot(steps, losses, "o-", color="#d62728", markersize=4, linewidth=1.5)
    ax.set_xlabel("训练步数 (Step)")
    ax.set_ylabel("Loss")
    ax.set_title("SFT 训练损失曲线")
    ax.grid(True, alpha=0.3)
    ax.annotate(
        f"{losses[0]:.3f}",
        (steps[0], losses[0]),
        textcoords="offset points",
        xytext=(8, 6),
        fontsize=8,
    )
    ax.annotate(
        f"{losses[-1]:.3f}",
        (steps[-1], losses[-1]),
        textcoords="offset points",
        xytext=(-30, 8),
        fontsize=8,
    )

    # Token accuracy
    ax = axes[1]
    ax.plot(steps, accs, "s-", color="#2ca02c", markersize=4, linewidth=1.5)
    ax.set_xlabel("训练步数 (Step)")
    ax.set_ylabel("Token Accuracy")
    ax.set_title("SFT Token 准确率曲线")
    ax.grid(True, alpha=0.3)
    ax.annotate(
        f"{accs[0]:.3f}",
        (steps[0], accs[0]),
        textcoords="offset points",
        xytext=(8, -12),
        fontsize=8,
    )
    ax.annotate(
        f"{accs[-1]:.3f}",
        (steps[-1], accs[-1]),
        textcoords="offset points",
        xytext=(-30, -12),
        fontsize=8,
    )

    # Learning rate
    ax = axes[2]
    ax.plot(steps, lrs, "^-", color="#1f77b4", markersize=4, linewidth=1.5)
    ax.set_xlabel("训练步数 (Step)")
    ax.set_ylabel("Learning Rate")
    ax.set_title("学习率调度曲线")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))

    fig.tight_layout()
    out = FIGURES_DIR / "sft_training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out}")


# ---------------------------------------------------------------------------
# 2. DAG visualisations
# ---------------------------------------------------------------------------

def plot_dag_examples(n: int = 4) -> None:
    """Visualise a few DAG JSON files using ReasoningDAG.visualize()."""
    from src.dag import ReasoningDAG

    dag_files = sorted(DAG_DIR.glob("*.json"))
    if not dag_files:
        print("[WARN] No DAG files found in", DAG_DIR)
        return

    scored: list[tuple[Path, int]] = []
    for p in dag_files:
        with open(p) as f:
            d = json.load(f)
        scored.append((p, len(d.get("nodes", []))))
    scored.sort(key=lambda x: -x[1])
    selected = [p for p, _ in scored[:n]]

    for dag_path in selected:
        with open(dag_path) as f:
            data = json.load(f)
        dag = ReasoningDAG.from_dict(data)
        stem = dag_path.stem
        out = FIGURES_DIR / f"dag_{stem}.png"
        dag.visualize(output_path=str(out))
        print(f"[OK] Saved {out}  ({dag.num_nodes} nodes, {dag.num_edges} edges)")


# ---------------------------------------------------------------------------
# 3. Data statistics
# ---------------------------------------------------------------------------

def plot_data_statistics() -> None:
    """Category distribution bar chart + score histogram from cleaned.jsonl."""
    categories: Counter = Counter()
    scores: list[float] = []

    with open(CLEANED_PATH) as f:
        for line in f:
            rec = json.loads(line)
            categories[rec.get("category", "未知")] += 1
            s = rec.get("procedure_score", rec.get("score"))
            if s is not None:
                scores.append(float(s))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Category bar chart
    ax = axes[0]
    cats = CATEGORY_ORDER
    counts = [categories.get(c, 0) for c in cats]
    bars = ax.bar(cats, counts, color=CATEGORY_COLORS, edgecolor="white", linewidth=0.8)
    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            str(cnt),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylabel("样本数")
    ax.set_title(f"数据类别分布 (共 {sum(counts)} 条)")
    ax.grid(axis="y", alpha=0.3)

    # Score histogram
    ax = axes[1]
    bins = np.arange(0, 1.15, 0.1)
    ax.hist(
        scores,
        bins=bins,
        color="#9467bd",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
    )
    ax.set_xlabel("过程得分 (procedure_score)")
    ax.set_ylabel("频次")
    ax.set_title(f"得分分布 (均值={np.mean(scores):.3f}, 中位数={np.median(scores):.3f})")
    ax.grid(axis="y", alpha=0.3)
    ax.axvline(np.mean(scores), color="#d62728", linestyle="--", linewidth=1, label="均值")
    ax.legend()

    fig.tight_layout()
    out = FIGURES_DIR / "data_statistics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out}")


# ---------------------------------------------------------------------------
# 4. Reward component weights pie chart
# ---------------------------------------------------------------------------

def plot_reward_weights() -> None:
    """Pie chart of the 5 reward components."""
    labels = ["Outcome (O)", "Format (F)", "Topology (T)", "Continuity (C)", "Length (L)"]
    weights = [0.40, 0.15, 0.20, 0.15, 0.10]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
    explode = [0.05] * 5

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        weights,
        labels=labels,
        autopct="%1.0f%%",
        colors=colors,
        explode=explode,
        startangle=140,
        textprops={"fontsize": 10},
    )
    for t in autotexts:
        t.set_fontweight("bold")
    ax.set_title("TopoPRM 奖励函数权重分布")

    out = FIGURES_DIR / "reward_weights.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _setup_chinese_font()

    print("=" * 60)
    print("TopoPRM Visualisation Generator")
    print("=" * 60)

    print("\n[1/4] SFT training curves ...")
    plot_sft_training_curves()

    print("\n[2/4] DAG example visualisations ...")
    plot_dag_examples(n=4)

    print("\n[3/4] Data statistics ...")
    plot_data_statistics()

    print("\n[4/4] Reward weights ...")
    plot_reward_weights()

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
