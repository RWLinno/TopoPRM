#!/usr/bin/env python3
"""Triage all training checkpoints by scanning trainer_state.json.

Classifies each checkpoint as healthy / stalled / collapsed / broken based on
reward / reward_std / frac_reward_zero_std / grad_norm trajectories.

Output: docs/checkpoints_triage_<YYYY-MM-DD>.md  (Markdown report, see B1).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CkptSummary:
    path: str
    steps: int
    size_mb: float
    reward_first: float | None
    reward_last: float | None
    reward_max: float | None
    reward_std_last: float | None
    frac_zero_std_last: float | None
    grad_norm_last: float | None
    kl_last: float | None
    verdict: str
    reason: str


def _mean_last_window(series: list[float], k: int = 3) -> float | None:
    if not series:
        return None
    window = series[-k:]
    return sum(window) / len(window) if window else None


def _finite(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _dir_size_mb(p: Path) -> float:
    total = 0
    for root, _, files in os.walk(p):
        for fn in files:
            try:
                total += (Path(root) / fn).stat().st_size
            except Exception:
                pass
    return total / (1024 * 1024)


def classify(
    reward_first: float | None,
    reward_last: float | None,
    reward_std_last: float | None,
    frac_zero: float | None,
    grad_norm_last: float | None,
    kl_last: float | None,
    steps: int,
    is_grpo: bool,
    loss_last: float | None,
) -> tuple[str, str]:
    # Handle SFT runs (no reward tracking expected)
    if not is_grpo:
        if grad_norm_last is not None and grad_norm_last > 100:
            return "broken", f"sft grad_norm explosion: {grad_norm_last:.1f}"
        if loss_last is None:
            return "broken", "sft no loss entries in trainer_state"
        if loss_last != loss_last:  # NaN
            return "broken", "sft loss NaN"
        if steps < 50:
            return "stalled", f"sft undertrained steps={steps}"
        return "healthy", (
            f"sft steps={steps}, loss_last={loss_last:.4f}, "
            f"grad_norm_last={grad_norm_last or 0:.2f}"
        )

    # GRPO-specific verdicts
    if reward_last is None and reward_first is None:
        return "broken", "grpo no reward entries in trainer_state"
    if grad_norm_last is not None and grad_norm_last > 100:
        return "broken", f"grad_norm explosion: {grad_norm_last:.1f}"
    if kl_last is not None and kl_last > 50:
        return "broken", f"kl exploded: {kl_last:.1f}"

    # collapsed
    if reward_std_last is not None and reward_std_last < 0.02:
        return "collapsed", f"reward_std={reward_std_last:.4f} near 0"
    if frac_zero is not None and frac_zero > 0.8:
        return "collapsed", f"frac_reward_zero_std={frac_zero:.2f} too high"

    # stalled
    trend = None
    if reward_first is not None and reward_last is not None:
        trend = reward_last - reward_first
    stall_conditions = []
    if trend is not None and abs(trend) < 0.02:
        stall_conditions.append(f"reward trend flat ({trend:+.3f})")
    if frac_zero is not None and frac_zero > 0.3:
        stall_conditions.append(f"frac_zero_std={frac_zero:.2f}")
    if steps < 150:
        stall_conditions.append(f"undertrained steps={steps}")
    if stall_conditions:
        return "stalled", "; ".join(stall_conditions)

    return "healthy", (
        f"reward {reward_first or 0:.3f}->{reward_last or 0:.3f}, "
        f"std_last={reward_std_last or 0:.3f}, "
        f"frac_zero={frac_zero or 0:.2f}"
    )


def summarize_ckpt(ckpt_dir: Path) -> CkptSummary | None:
    ts_path = ckpt_dir / "trainer_state.json"
    if not ts_path.exists():
        return None
    try:
        d = json.loads(ts_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    hist = d.get("log_history", []) or []
    steps = int(d.get("global_step", len(hist)) or 0)

    def series(key):
        return [
            _finite(e.get(key))
            for e in hist
            if _finite(e.get(key)) is not None
        ]

    rewards = series("reward")
    stds = series("reward_std")
    fracs = series("frac_reward_zero_std")
    grads = series("grad_norm")
    kls = series("kl")
    losses = series("loss")

    reward_first = rewards[0] if rewards else None
    reward_last = _mean_last_window(rewards, 3)
    reward_max = max(rewards) if rewards else None
    reward_std_last = _mean_last_window(stds, 3)
    frac_zero_last = _mean_last_window(fracs, 3)
    grad_norm_last = _mean_last_window(grads, 3)
    kl_last = _mean_last_window(kls, 3)
    loss_last = _mean_last_window(losses, 3)

    path_str = str(ckpt_dir).lower()
    is_grpo = "/grpo_" in path_str or path_str.startswith("grpo_")

    verdict, reason = classify(
        reward_first, reward_last, reward_std_last, frac_zero_last,
        grad_norm_last, kl_last, steps,
        is_grpo=is_grpo, loss_last=loss_last,
    )

    return CkptSummary(
        path=str(ckpt_dir),
        steps=steps,
        size_mb=_dir_size_mb(ckpt_dir),
        reward_first=reward_first,
        reward_last=reward_last,
        reward_max=reward_max,
        reward_std_last=reward_std_last,
        frac_zero_std_last=frac_zero_last,
        grad_norm_last=grad_norm_last,
        kl_last=kl_last,
        verdict=verdict,
        reason=reason,
    )


def _fmt(v: float | None, d: int = 3) -> str:
    if v is None:
        return "-"
    return f"{v:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("output"))
    ap.add_argument("--output", type=Path,
                    default=Path("docs/checkpoints_triage_2026-04-20.md"))
    args = ap.parse_args()

    rows: list[CkptSummary] = []
    for ts in sorted(args.root.glob("**/checkpoint-*/trainer_state.json")):
        ckpt = ts.parent
        s = summarize_ckpt(ckpt)
        if s is not None:
            rows.append(s)

    rows.sort(key=lambda r: (r.verdict, -r.size_mb, r.path))

    verdict_counts: dict[str, int] = {}
    size_by_verdict: dict[str, float] = {}
    for r in rows:
        verdict_counts[r.verdict] = verdict_counts.get(r.verdict, 0) + 1
        size_by_verdict[r.verdict] = size_by_verdict.get(r.verdict, 0.0) + r.size_mb

    lines: list[str] = []
    lines.append("# Checkpoint Triage Report (2026-04-20)")
    lines.append("")
    lines.append(f"Scanned `{args.root}` for `**/checkpoint-*/trainer_state.json` -> {len(rows)} checkpoints.")
    lines.append("")
    lines.append("## Summary by verdict")
    lines.append("")
    lines.append("| Verdict | Count | Total size (MB) |")
    lines.append("|---------|------:|----------------:|")
    for v in ("healthy", "stalled", "collapsed", "broken"):
        lines.append(f"| {v} | {verdict_counts.get(v, 0)} | {size_by_verdict.get(v, 0.0):.0f} |")
    lines.append("")
    lines.append("## Classification rules")
    lines.append("")
    lines.append("- healthy: reward has upward trend, reward_std > 0.05, frac_reward_zero_std <= 0.3")
    lines.append("- stalled: reward flat (|delta|<0.02), or frac_reward_zero_std > 0.3, or undertrained (<150 steps)")
    lines.append("- collapsed: reward_std < 0.02 or frac_reward_zero_std > 0.8")
    lines.append("- broken: grad_norm > 100, kl > 50, or no reward entries")
    lines.append("")
    lines.append("## Detailed rows")
    lines.append("")
    lines.append(
        "| Verdict | Steps | Size(MB) | reward_first | reward_last | reward_max | "
        "reward_std_last | frac_zero_std | grad_norm_last | kl_last | Path | Reason |"
    )
    lines.append(
        "|--------|------:|---------:|-------------:|------------:|----------:|"
        "----------------:|--------------:|---------------:|--------:|------|--------|"
    )
    for r in rows:
        lines.append(
            f"| {r.verdict} | {r.steps} | {r.size_mb:.0f} | "
            f"{_fmt(r.reward_first)} | {_fmt(r.reward_last)} | {_fmt(r.reward_max)} | "
            f"{_fmt(r.reward_std_last)} | {_fmt(r.frac_zero_std_last)} | "
            f"{_fmt(r.grad_norm_last, 2)} | {_fmt(r.kl_last, 2)} | "
            f"`{r.path}` | {r.reason} |"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}  ({len(rows)} checkpoints, verdicts={verdict_counts})")


if __name__ == "__main__":
    main()
