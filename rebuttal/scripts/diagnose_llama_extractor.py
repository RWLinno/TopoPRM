#!/usr/bin/env python3
"""Diagnose why Full TopoPRM ~= outcome-only on Llama-3.1-8B (Req-2).

Hypothesis: the rule DAG extractor was tuned on Qwen/DeepSeek-R1 traces that
carry explicit "Step N:" / <think></answer> structure.  Llama-3.1-8B-Instruct
emits *marker-free natural-language prose* CoT.  With the default flags the
step segmenter collapses such prose to <=1 node, the DAG is trivial, q_topo
saturates near 1.0 for EVERY rollout, so the topology term contributes no
usable within-group variance and the "topology-aware" reward degenerates to
outcome+format (i.e. ~ outcome-only).

This script contrasts, on the SAME traces, the extractor under:
  (A) default flags (released behaviour)
  (B) the non-Qwen prose profile (sentence fallback + extra markers + filter +
      Req-1 precision guards)
and reports #nodes, #edges, and the spread of a proxy topology score.  A large
jump in nodes/edges + a drop in q_topo saturation under (B) confirms the fix.
"""
from __future__ import annotations

import importlib
import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Representative marker-free prose CoT (Llama-3.1-8B-Instruct style): flowing
# sentences joined by "First/Then/So/Therefore", no "Step N", no bullets.
LLAMA_STYLE = [
    (
        "Natalia sold clips to 48 friends in April. In May she sold half as many, "
        "so she sold 48 divided by 2 which is 24 clips in May. To find the total "
        "we add April and May together, so 48 plus 24 gives 72. Therefore Natalia "
        "sold 72 clips altogether. The final answer is 72."
    ),
    (
        "First I need the number of hours. Weng earns 12 dollars per hour and she "
        "worked 50 minutes. Since 50 minutes is 50 over 60 of an hour, that is "
        "five sixths of an hour. So her earnings are 12 times five sixths which "
        "equals 10 dollars. Therefore she earned 10 dollars."
    ),
    (
        "Let me work out how much water is used. The pool holds 120 gallons and it "
        "fills at 6 gallons per minute, so the time to fill is 120 divided by 6 "
        "which is 20 minutes. But the drain removes 2 gallons per minute, so the "
        "net fill rate is 6 minus 2 equals 4 gallons per minute. Then the real "
        "fill time is 120 divided by 4 which is 30 minutes. So it takes 30 minutes."
    ),
]

# Qwen/R1 style with explicit markers, for contrast.
QWEN_STYLE = [
    (
        "Step 1: Natalia sold 48 clips in April.\n"
        "Step 2: In May she sold 48/2 = 24 clips.\n"
        "Step 3: Total = 48 + 24 = 72.\n"
        "The final answer is \\boxed{72}."
    ),
]

PROSE_PROFILE = {
    "TOPO_DAG_SENTENCE_FALLBACK": "1",
    "TOPO_DAG_EXTRA_STEP_MARKERS": "1",
    "TOPO_DAG_FILTER_FORMATTING": "1",
    "TOPO_DAG_SENTENCE_MIN_LEN": "20",
    # Req-1 precision guards so the extra prose edges stay clean:
    "TOPO_VAR_REF_REQUIRE_MULTI": "1",
    "TOPO_SEQ_WEAK_EDGE_MODE": "full",
    "TOPO_SEQ_REQUIRE_OVERLAP": "1",
    "TOPO_SEQ_MIN_OVERLAP": "0.06",
}
_ALL_FLAGS = list(PROSE_PROFILE) + [
    "TOPO_VAR_REF_MIN_SHARED", "TOPO_ORDER_REQUIRE_NUMERIC",
]


def _reload(flags: dict):
    for k in _ALL_FLAGS:
        os.environ.pop(k, None)
    for k, v in flags.items():
        os.environ[k] = str(v)
    import src.data.build_dag as bd
    importlib.reload(bd)
    return bd


def _analyze(bd, traces, tag):
    from src.reward.topo_reward import TopoReward
    tr = TopoReward()
    rows = []
    qtopo = []
    for t in traces:
        dag = bd.build_dag_from_answer(t)
        comp = [{"role": "assistant", "content": t}]
        try:
            q = tr(comp)[0]
        except Exception:
            q = float("nan")
        qtopo.append(q)
        rows.append((dag.num_nodes, dag.num_edges, round(q, 4)))
    print(f"\n[{tag}] per-trace (nodes, edges, q_topo):")
    for r in rows:
        print(f"    nodes={r[0]:2d} edges={r[1]:2d} q_topo={r[2]}")
    mean_nodes = statistics.mean(r[0] for r in rows)
    mean_edges = statistics.mean(r[1] for r in rows)
    q_std = statistics.pstdev(qtopo) if len(qtopo) > 1 else 0.0
    q_mean = statistics.mean(qtopo)
    print(f"[{tag}] mean_nodes={mean_nodes:.2f} mean_edges={mean_edges:.2f} "
          f"q_topo mean={q_mean:.4f} std={q_std:.4f}")
    return {"mean_nodes": mean_nodes, "mean_edges": mean_edges,
            "q_topo_mean": round(q_mean, 4), "q_topo_std": round(q_std, 4)}


def main():
    print("=" * 64)
    print("LLAMA-STYLE (marker-free prose) traces")
    print("=" * 64)
    bd = _reload({})
    a_default = _analyze(bd, LLAMA_STYLE, "A: default flags")
    bd = _reload(PROSE_PROFILE)
    a_prose = _analyze(bd, LLAMA_STYLE, "B: non-Qwen prose profile")

    print("\n" + "=" * 64)
    print("QWEN-STYLE (explicit markers) trace, default flags")
    print("=" * 64)
    bd = _reload({})
    q_default = _analyze(bd, QWEN_STYLE, "Qwen default")

    out = {
        "llama_style": {"default": a_default, "prose_profile": a_prose},
        "qwen_style_default": q_default,
        "prose_profile_flags": PROSE_PROFILE,
    }
    Path(REPO / "rebuttal/outputs/llama_extractor_diagnosis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2)
    )
    print("\n-> rebuttal/outputs/llama_extractor_diagnosis.json")
    print(f"\nSUMMARY: Llama prose nodes {a_default['mean_nodes']:.1f} -> "
          f"{a_prose['mean_nodes']:.1f}, edges {a_default['mean_edges']:.1f} -> "
          f"{a_prose['mean_edges']:.1f}; q_topo std {a_default['q_topo_std']:.3f} -> "
          f"{a_prose['q_topo_std']:.3f} (higher std = usable training signal).")


if __name__ == "__main__":
    main()
