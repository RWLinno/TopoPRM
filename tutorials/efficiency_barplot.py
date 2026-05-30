"""Render efficiency bar-plot for the TopoPRM paper (Fig 5).

Usage
-----
    python tutorials/efficiency_barplot.py

Produces ``papaer_20260522/figures/Fig5.Efficiency_Barplot.pdf``.
Data from tables/efficiency.tex (Qwen3.5-9B, 4 primary benchmarks).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_OUT = _REPO_ROOT / "papaer_20260522" / "figures" / "Fig5.Efficiency_Barplot.pdf"

PALETTE = {
    "blue": "#2E6FB5",
    "amber": "#D97706",
    "green": "#2E8B57",
    "red": "#C0392B",
    "gray": "#6B7280",
    "lightgray": "#D1D5DB",
    "text": "#333333",
    "grid": "#E5E7EB",
}

# Methods (ordered by Acc/kTok descending for visual impact)
METHODS = [
    "TopoPRM\n(Full)",
    "w/o SCAE\n(Gated)",
    "w/o\nTopology",
    "Outcome\nOnly",
    "w/o\nContinuity",
]

# Data from tables/efficiency.tex & main_detailed.tex
# Average tokens across 4 benchmarks (GSM8K, MATH-500, AIME'24, Omni-MATH)
AVG_TOKENS = np.array([
    np.mean([796, 1454, 2431, 1592]),    # TopoPRM Full
    np.mean([1006, 1515, 2560, 2463]),   # Gated (w/o SCAE)
    np.mean([946, 1506, 2560, 2430]),    # w/o Topology
    np.mean([1047, 1511, 2546, 1845]),   # Outcome Only
    np.mean([1528, 1536, 2560, 2560]),   # w/o Continuity
])

# Acc/kTok from efficiency.tex (average across 4 benchmarks)
ACC_KTOK = np.array([
    np.mean([118.2, 35.1, 12.3, 26.6]),  # TopoPRM Full
    np.mean([93.2, 33.5, 7.8, 17.0]),    # Gated
    np.mean([98.3, 33.9, 7.8, 17.4]),    # w/o Topology
    np.mean([89.1, 33.6, 6.5, 22.5]),    # Outcome Only
    np.mean([33.4, 14.1, 2.6, 5.8]),     # w/o Continuity
])


def render(out_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.edgecolor": PALETTE["text"],
        "axes.labelcolor": PALETTE["text"],
        "text.color": PALETTE["text"],
    })

    fig, ax1 = plt.subplots(figsize=(5.5, 3.0))

    x = np.arange(len(METHODS))
    width = 0.52

    colors = [PALETTE["blue"]] + [PALETTE["lightgray"]] * 4

    bars = ax1.bar(x, AVG_TOKENS, width, color=colors, alpha=0.8,
                   edgecolor="white", linewidth=0.6)

    ax1.set_ylabel("Mean Generation Tokens", fontsize=8.5, color=PALETTE["text"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(METHODS, fontsize=7.5, ha="center")
    ax1.set_ylim(0, max(AVG_TOKENS) * 1.18)
    ax1.tick_params(axis="both", which="both", length=3, colors=PALETTE["text"],
                    labelsize=7.5)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)
    ax1.spines["left"].set_linewidth(0.7)
    ax1.spines["bottom"].set_linewidth(0.7)
    ax1.grid(axis="y", color=PALETTE["grid"], linewidth=0.4, alpha=0.9)
    ax1.set_axisbelow(True)

    # Right axis: Acc/kTok with diamond markers
    ax2 = ax1.twinx()
    ax2.plot(
        x, ACC_KTOK, "D-",
        color=PALETTE["red"], lw=1.6, markersize=5.5,
        markerfacecolor="white", markeredgecolor=PALETTE["red"],
        markeredgewidth=1.4, zorder=5,
    )
    ax2.set_ylabel("Acc / kTok  (higher is better)", fontsize=8.5,
                   color=PALETTE["red"])
    ax2.tick_params(axis="y", colors=PALETTE["red"], labelsize=7.5)
    ax2.spines["right"].set_color(PALETTE["red"])
    ax2.spines["right"].set_linewidth(0.7)
    for spine in ("top", "left"):
        ax2.spines[spine].set_visible(False)
    ax2.set_ylim(0, max(ACC_KTOK) * 1.25)

    # Annotate TopoPRM Full
    ax2.annotate(
        f"{ACC_KTOK[0]:.1f}",
        xy=(0, ACC_KTOK[0]), xytext=(0.35, ACC_KTOK[0] + 4),
        fontsize=7, color=PALETTE["red"], fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=PALETTE["red"], lw=0.4),
    )
    # Annotate w/o Continuity (catastrophic drop)
    ax2.annotate(
        f"{ACC_KTOK[4]:.1f}",
        xy=(4, ACC_KTOK[4]), xytext=(3.65, ACC_KTOK[4] + 6),
        fontsize=7, color=PALETTE["red"],
        arrowprops=dict(arrowstyle="-", color=PALETTE["red"], lw=0.4),
    )

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=PALETTE["lightgray"], alpha=0.8, edgecolor="white",
              label="Mean Tokens"),
        Line2D([0], [0], color=PALETTE["red"], marker="D", lw=1.4,
               markerfacecolor="white", markeredgecolor=PALETTE["red"],
               markersize=4.5, label="Acc/kTok"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", frameon=False,
               fontsize=7.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight",
                pad_inches=0.04, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"Wrote {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    render(FIG_OUT)
