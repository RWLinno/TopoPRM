"""Render GRPO training-dynamics figure from real log files.

Usage
-----
    python tutorials/training_curve.py

Reads the four GRPO training logs in ``logs/`` (full + three ablations), parses
the per-5-step trainer-state dict printed by ``transformers.Trainer``, and
produces ``topoprm_paper/figures/Fig4.Training_Curve.pdf`` with three stacked
subplots: (top) reward curves with ablations and std band, (middle) completion
length overlaid with pass@1 proxy (from ``reward`` minus process reward for
outcome-only runs — see ``--pass1-source``), (bottom) reward std and
frac_reward_zero_std as anti-collapse indicators.

All data comes from the real log files in this repository; no synthetic
numbers are used. If a log file is missing the corresponding line will be
dropped from the plot.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

# Allow `python3 tutorials/training_curve.py` from the repo root without
# exporting PYTHONPATH explicitly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "logs"
FIG_OUT = REPO_ROOT / "papaer_20260522" / "figures" / "Fig4.Training_Curve.pdf"


PALETTE = {
    "blue": "#2E6FB5",
    "amber": "#D97706",
    "green": "#2E8B57",
    "red": "#C0392B",
    "gray": "#9CA3AF",
    "text": "#333333",
    "grid": "#E5E7EB",
}


RUNS: Sequence[Dict[str, str]] = (
    {
        "key": "full",
        "label": "TopoPRM (full)",
        "log": "grpo_topoprm_scae_9b_mcl4096_20260517_054946.log",
        "colour": PALETTE["blue"],
        "style": "-",
        "lw": 1.6,
    },
    {
        "key": "no_cont",
        "label": "TopoPRM w/o continuity",
        "log": "grpo_no_continuity_9b_mcl4096_single_20260517_052725.log",
        "colour": PALETTE["green"],
        "style": ":",
        "lw": 1.1,
    },
    {
        "key": "dr1_full",
        "label": "TopoPRM (DR1-7B)",
        "log": "grpo_topoprm_dr1_7b_mcl4096_20260517_171200.log",
        "colour": PALETTE["amber"],
        "style": "-.",
        "lw": 1.1,
    },
)


STATE_REGEX = re.compile(r"\{[^{}]*'completions/mean_length'[^{}]*\}")
STEP_REGEX = re.compile(r"(\d+)/\d+ \[")
EVAL_REGEX = re.compile(
    r"\[eval\]\s+step=(\d+)\s+acc=([0-9.]+)\s+n=(\d+)\s+"
    r"mean_new_tokens=([0-9.]+)"
)


@dataclass
class RunSeries:
    key: str
    label: str
    colour: str
    style: str
    lw: float
    steps: List[int]
    records: List[Dict[str, float]]
    eval_steps: List[int] = None  # type: ignore[assignment]
    eval_acc: List[float] = None  # type: ignore[assignment]
    eval_new_tokens: List[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.eval_steps is None:
            self.eval_steps = []
        if self.eval_acc is None:
            self.eval_acc = []
        if self.eval_new_tokens is None:
            self.eval_new_tokens = []

    def column(self, name: str) -> List[Optional[float]]:
        return [r.get(name) for r in self.records]

    def has_eval(self) -> bool:
        return len(self.eval_steps) > 0


def _parse_state_dict(raw: str) -> Dict[str, float]:
    try:
        data = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    out: Dict[str, float] = {}
    for k, v in data.items():
        if not isinstance(v, (int, float, str)):
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def parse_log(path: Path) -> tuple[List[int], List[Dict[str, float]], List[int], List[float], List[float]]:
    steps: List[int] = []
    records: List[Dict[str, float]] = []
    eval_steps: List[int] = []
    eval_acc: List[float] = []
    eval_new_tokens: List[float] = []
    if not path.exists():
        return steps, records, eval_steps, eval_acc, eval_new_tokens
    text = path.read_text(encoding="utf-8", errors="ignore")
    for m in STATE_REGEX.finditer(text):
        state = _parse_state_dict(m.group(0))
        if not state:
            continue
        # Try to get step from global_step/max_steps field inside the dict
        gs_raw = None
        try:
            raw_dict = ast.literal_eval(m.group(0))
            gs_raw = raw_dict.get("global_step/max_steps", "")
        except (ValueError, SyntaxError):
            pass
        step = None
        if gs_raw and "/" in str(gs_raw):
            try:
                step = int(str(gs_raw).split("/")[0])
            except (ValueError, IndexError):
                pass
        if step is None:
            # Fallback: look for step in the preface text
            preface = text[max(0, m.start() - 200) : m.start()]
            step_matches = list(re.finditer(r"(\d+)/\d+ \[", preface))
            if step_matches:
                step = int(step_matches[-1].group(1))
        if step is not None:
            steps.append(step)
            records.append(state)
    for m in EVAL_REGEX.finditer(text):
        eval_steps.append(int(m.group(1)))
        eval_acc.append(float(m.group(2)))
        eval_new_tokens.append(float(m.group(4)))
    return steps, records, eval_steps, eval_acc, eval_new_tokens


def load_runs() -> List[RunSeries]:
    series: List[RunSeries] = []
    for spec in RUNS:
        log_path = LOG_DIR / spec["log"]
        steps, records, eval_steps, eval_acc, eval_new_tokens = parse_log(log_path)
        series.append(
            RunSeries(
                key=spec["key"],
                label=spec["label"],
                colour=spec["colour"],
                style=spec["style"],
                lw=spec["lw"],
                steps=steps,
                records=records,
                eval_steps=eval_steps,
                eval_acc=eval_acc,
                eval_new_tokens=eval_new_tokens,
            )
        )
    return series


def _style_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="both", length=3, colors=PALETTE["text"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PALETTE["text"])
        ax.spines[spine].set_linewidth(0.8)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.4, alpha=0.9)
    ax.set_axisbelow(True)


def plot_reward(ax: plt.Axes, series: List[RunSeries]) -> None:
    for rs in series:
        y = rs.column("reward")
        if not rs.steps:
            continue
        ax.plot(
            rs.steps,
            y,
            rs.style,
            color=rs.colour,
            lw=rs.lw,
            label=rs.label,
        )
        if rs.key == "full":
            std = [v if v is not None else 0.0 for v in rs.column("reward_std")]
            upper = [(y[i] or 0) + std[i] for i in range(len(rs.steps))]
            lower = [(y[i] or 0) - std[i] for i in range(len(rs.steps))]
            ax.fill_between(
                rs.steps,
                lower,
                upper,
                color=rs.colour,
                alpha=0.15,
                linewidth=0,
            )
    ax.set_ylabel("reward (batch mean)", fontsize=8, color=PALETTE["text"])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune="both", nbins=6))


def plot_length(ax: plt.Axes, series: List[RunSeries]) -> None:
    full = next(rs for rs in series if rs.key == "full")
    outcome = next((rs for rs in series if rs.key == "outcome"), None)
    if full.steps:
        ax.plot(
            full.steps,
            full.column("completions/mean_length"),
            color=PALETTE["blue"],
            lw=1.4,
            label="TopoPRM (full) length",
        )
    if outcome and outcome.steps:
        ax.plot(
            outcome.steps,
            outcome.column("completions/mean_length"),
            color=PALETTE["gray"],
            lw=1.0,
            linestyle="--",
            label="Outcome-only length",
        )
    ax.set_ylabel("mean tokens per completion", fontsize=8, color=PALETTE["text"])

    ax2 = ax.twinx()

    # Prefer the real eval accuracy (logged by EvalAccuracyCallback) when
    # available; otherwise fall back to training entropy as an anti-collapse
    # proxy so the subplot is not empty on older runs.
    any_eval = any(rs.has_eval() for rs in series)
    if any_eval:
        for rs in series:
            if not rs.has_eval():
                continue
            ax2.plot(
                rs.eval_steps,
                rs.eval_acc,
                color=rs.colour,
                lw=1.0,
                linestyle="-",
                marker="o",
                markersize=2.5,
                alpha=0.85,
                label=f"{rs.label} eval acc",
            )
        ax2.set_ylabel("eval accuracy (right)", fontsize=8, color=PALETTE["text"])
        ax2.set_ylim(-0.02, 1.02)
    else:
        for rs in series:
            entropy = rs.column("entropy")
            if any(v is not None for v in entropy):
                ax2.plot(
                    rs.steps,
                    entropy,
                    color=rs.colour,
                    lw=0.8,
                    alpha=0.55,
                    linestyle=":",
                )
        ax2.set_ylabel("entropy (right)", fontsize=8, color=PALETTE["text"])
    ax2.tick_params(axis="y", colors=PALETTE["text"])
    for spine in ("top",):
        ax2.spines[spine].set_visible(False)
    ax2.spines["right"].set_color(PALETTE["text"])
    ax2.spines["right"].set_linewidth(0.6)


def plot_collapse(ax: plt.Axes, series: List[RunSeries]) -> None:
    for rs in series:
        if not rs.steps:
            continue
        ax.plot(
            rs.steps,
            rs.column("reward_std"),
            rs.style,
            color=rs.colour,
            lw=rs.lw * 0.85,
            label=f"{rs.label} reward_std",
        )
    ax.set_ylabel("reward std", fontsize=8, color=PALETTE["text"])

    ax2 = ax.twinx()
    for rs in series:
        if not rs.steps:
            continue
        ax2.plot(
            rs.steps,
            rs.column("frac_reward_zero_std"),
            color=rs.colour,
            lw=0.8,
            linestyle=(0, (1, 1.4)),
            alpha=0.7,
        )
    ax2.set_ylabel("frac_reward_zero_std (right)", fontsize=8, color=PALETTE["text"])
    ax2.set_ylim(-0.02, 1.05)
    ax2.tick_params(axis="y", colors=PALETTE["text"])
    for spine in ("top",):
        ax2.spines[spine].set_visible(False)
    ax2.spines["right"].set_color(PALETTE["text"])
    ax2.spines["right"].set_linewidth(0.6)


def render(series: List[RunSeries], out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica",
                "Arial",
                "DejaVu Sans",
                "Liberation Sans",
            ],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": PALETTE["text"],
            "axes.labelcolor": PALETTE["text"],
            "text.color": PALETTE["text"],
        }
    )
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.5, 5.4),
        sharex=True,
        gridspec_kw={"hspace": 0.22},
    )
    plot_reward(axes[0], series)
    plot_length(axes[1], series)
    plot_collapse(axes[2], series)
    for ax in axes:
        _style_axes(ax)
    axes[-1].set_xlabel("GRPO gradient step", fontsize=8, color=PALETTE["text"])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=min(4, len(labels)),
        frameon=False,
        fontsize=7,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    alt_png = out_path.with_suffix(".png")
    fig.savefig(alt_png, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close(fig)


def print_summary(series: List[RunSeries]) -> None:
    rows = []
    for rs in series:
        if not rs.steps:
            rows.append(f"  {rs.label:<28} no data")
            continue
        last = rs.records[-1]
        rows.append(
            "  {label:<28} steps={nsteps:>3}  final reward={r:.3f}  "
            "len={length:.0f}  std={std:.3f}".format(
                label=rs.label,
                nsteps=len(rs.steps),
                r=last.get("reward", float("nan")),
                length=last.get("completions/mean_length", float("nan")),
                std=last.get("reward_std", float("nan")),
            )
        )
    print("Loaded series:\n" + "\n".join(rows))


def dump_tsv(series: List[RunSeries], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "reward",
        "reward_std",
        "completions/mean_length",
        "frac_reward_zero_std",
        "entropy",
        "kl",
    ]
    for rs in series:
        path = out_dir / f"training_curve_{rs.key}.tsv"
        with path.open("w", encoding="utf-8") as f:
            f.write("step\t" + "\t".join(c.replace("completions/", "") for c in columns) + "\n")
            for step, rec in zip(rs.steps, rs.records):
                vals = [f"{rec.get(c, '')}" for c in columns]
                f.write(f"{step}\t" + "\t".join(vals) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(FIG_OUT), help="Output PDF path")
    parser.add_argument(
        "--tsv-dir",
        default=str(REPO_ROOT / "papaer_20260522" / "figures" / "training_curve_data"),
        help="Directory for the per-run TSV dumps used by the figure",
    )
    args = parser.parse_args()

    series = load_runs()
    print_summary(series)
    dump_tsv(series, Path(args.tsv_dir))
    render(series, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
