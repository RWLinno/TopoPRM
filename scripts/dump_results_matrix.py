#!/usr/bin/env python3
"""Honest diagnostic dump of current DR1-7B family results.

Prints a full matrix (accuracy), avg-token matrix, per-bench winner, and
computes head-to-head TopoPRM vs outcome-only deltas. No cherry-picking.
"""
from __future__ import annotations
import json
from pathlib import Path

EVAL = Path("${TOPOPRM_ROOT:-.}/output/eval")

LABELS = [
    ("baseline_dr1_7b_chat",    "DR1-7B base"),
    ("sft_dr1_7b",              "+ SFT"),
    ("grpo_outcome_only",       "+ GRPO (outcome-only)"),
    ("grpo_no_topo",            "+ GRPO (w/o topology)"),
    ("grpo_no_continuity",      "+ GRPO (w/o continuity)"),
    ("topoprm_full_dr1_7b",     "+ TopoPRM (full)"),
]

BENCHES = [
    "gsm8k", "math500", "olympiadbench", "omni_math",
    "aime2024", "aime2025", "cnmo2024", "mmlu", "gpqa_diamond",
]

BENCH_HDR = {
    "gsm8k": "GSM8K",  "math500": "MATH500",  "olympiadbench": "Olymp",
    "omni_math": "Omni",   "aime2024": "AIME24", "aime2025": "AIME25",
    "cnmo2024": "CNMO",   "mmlu": "MMLU",     "gpqa_diamond": "GPQA-D",
}


def load(label: str, bench: str):
    p = EVAL / f"{label}_{bench}_metrics.json"
    if not p.exists():
        return None, None
    try:
        m = json.loads(p.read_text())
    except Exception:
        return None, None
    v = m.get("pass@1") or m.get("pass_at_1") or m.get("accuracy")
    acc = round(float(v) * 100.0, 1) if v is not None else None
    tok = m.get("avg_tokens")
    tok = int(tok) if tok is not None else None
    return acc, tok


def fmt(v, w=6, prec=1):
    if v is None:
        return f"{'—':>{w}}"
    if isinstance(v, int):
        return f"{v:>{w}d}"
    return f"{v:>{w}.{prec}f}"


def main():
    # Collect
    data_acc = {name: {} for _, name in LABELS}
    data_tok = {name: {} for _, name in LABELS}
    for key, name in LABELS:
        for b in BENCHES:
            acc, tok = load(key, b)
            data_acc[name][b] = acc
            data_tok[name][b] = tok

    bar = "=" * 110
    # Accuracy matrix
    print(bar)
    print("ACCURACY (pass@1, %)")
    print(bar)
    header = f"{'Variant':28} | " + " | ".join(f"{BENCH_HDR[b]:>6}" for b in BENCHES)
    print(header)
    print("-" * len(header))
    for _, name in LABELS:
        cells = [fmt(data_acc[name][b]) for b in BENCHES]
        print(f"{name:28} | " + " | ".join(cells))

    # Average tokens
    print()
    print(bar)
    print("AVG RESPONSE TOKENS")
    print(bar)
    print(header)
    print("-" * len(header))
    for _, name in LABELS:
        cells = [fmt(data_tok[name][b]) for b in BENCHES]
        print(f"{name:28} | " + " | ".join(cells))

    # Per-bench winner among the 4 trained variants
    trained = ["+ GRPO (outcome-only)", "+ GRPO (w/o topology)",
               "+ GRPO (w/o continuity)", "+ TopoPRM (full)"]
    print()
    print(bar)
    print("PER-BENCH RANKING AMONG 4 TRAINED VARIANTS (best -> worst)")
    print(bar)
    for b in BENCHES:
        pairs = [(n, data_acc[n][b]) for n in trained if data_acc[n][b] is not None]
        if not pairs:
            print(f"  {BENCH_HDR[b]:8}: (no data yet)")
            continue
        pairs.sort(key=lambda x: -x[1])
        ranking = " > ".join(f"{n.split('(')[-1].rstrip(')')}:{v:.1f}" for n, v in pairs)
        print(f"  {BENCH_HDR[b]:8}: {ranking}")

    # Head-to-head: TopoPRM - outcome_only, and TopoPRM - SFT
    print()
    print(bar)
    print("HEAD-TO-HEAD vs outcome-only GRPO (positive = TopoPRM wins)")
    print(bar)
    oo = data_acc["+ GRPO (outcome-only)"]
    full = data_acc["+ TopoPRM (full)"]
    total_wins = 0; total_losses = 0
    for b in BENCHES:
        if oo[b] is None or full[b] is None:
            print(f"  {BENCH_HDR[b]:8}: (incomplete)")
            continue
        d = full[b] - oo[b]
        mark = "WIN " if d > 0.5 else ("tie " if abs(d) <= 0.5 else "LOSE")
        print(f"  {BENCH_HDR[b]:8}: {full[b]:5.1f} - {oo[b]:5.1f} = {d:+5.1f}  {mark}")
        if d > 0.5:   total_wins += 1
        elif d < -0.5: total_losses += 1
    print(f"\n  Score: TopoPRM wins {total_wins}, loses {total_losses} across {sum(1 for b in BENCHES if oo[b] and full[b])} benchmarks")

    # Compactness head-to-head
    print()
    print(bar)
    print("TOKEN BUDGET vs outcome-only (negative = TopoPRM shorter)")
    print(bar)
    oo_t = data_tok["+ GRPO (outcome-only)"]
    full_t = data_tok["+ TopoPRM (full)"]
    for b in BENCHES:
        if oo_t[b] is None or full_t[b] is None:
            print(f"  {BENCH_HDR[b]:8}: (incomplete)")
            continue
        d = full_t[b] - oo_t[b]
        ratio = (full_t[b] / oo_t[b]) if oo_t[b] else None
        print(f"  {BENCH_HDR[b]:8}: {full_t[b]:6d} vs {oo_t[b]:6d}   Δ={d:+6d}  ratio={ratio:.2f}x")

    # w/o variants ablation (TopoPRM - ablated)
    print()
    print(bar)
    print("ABLATION CONTRIBUTION  (TopoPRM - ablated, positive = component helps)")
    print(bar)
    for ablname in ["+ GRPO (w/o topology)", "+ GRPO (w/o continuity)"]:
        print(f"\n  {ablname}:")
        abl = data_acc[ablname]
        for b in BENCHES:
            if full[b] is None or abl[b] is None:
                continue
            d = full[b] - abl[b]
            print(f"    {BENCH_HDR[b]:8}: {full[b]:5.1f} - {abl[b]:5.1f} = {d:+5.1f}")


if __name__ == "__main__":
    main()
