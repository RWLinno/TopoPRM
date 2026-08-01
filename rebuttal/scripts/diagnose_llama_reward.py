#!/usr/bin/env python3
"""Confirm the Req-2 mechanism at the reward level.

TopoReward scores ONLY the text inside <think>...</think>.  The DR1-7B runs use
an SFT adapter that emits <think>/<answer>; the Llama-3.1-8B runs used
--sft_adapter "" so the instruct model emits plain prose with NO <think> block.
Consequently extract_think_block() returns "" and TopoReward returns 0.0 for
*every* rollout -> zero within-group variance -> the topology term is inert and
"Full TopoPRM" collapses to outcome+format+length (~ outcome-only).

We verify by scoring the same reasoning as (a) bare prose (Llama), (b) prose
wrapped in <think></think><answer></answer> (what an SFT'd model emits), under
default flags and the non-Qwen prose profile.
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

PROSE = [
    "Natalia sold clips to 48 friends in April. In May she sold half as many, so "
    "she sold 48 divided by 2 which is 24 clips in May. Adding April and May gives "
    "48 plus 24 equals 72. Therefore Natalia sold 72 clips. The final answer is 72.",
    "Weng earns 12 dollars per hour and worked 50 minutes. Since 50 minutes is 50 "
    "over 60 of an hour, her pay is 12 times 50 over 60 which equals 10 dollars. So "
    "she earned 10 dollars.",
    "The pool holds 120 gallons and fills at 6 gallons per minute while draining 2 "
    "per minute, so the net rate is 6 minus 2 equals 4 gallons per minute. Then 120 "
    "divided by 4 equals 30 minutes. So it takes 30 minutes.",
    "There are 15 trees and after planting there are 21 trees, so the workers "
    "planted 21 minus 15 equals 6 trees today. Therefore they planted 6 trees.",
]

PROSE_PROFILE = {
    "TOPO_DAG_SENTENCE_FALLBACK": "1",
    "TOPO_DAG_EXTRA_STEP_MARKERS": "1",
    "TOPO_DAG_FILTER_FORMATTING": "1",
    "TOPO_VAR_REF_REQUIRE_MULTI": "1",
    "TOPO_SEQ_WEAK_EDGE_MODE": "full",
    "TOPO_SEQ_REQUIRE_OVERLAP": "1",
    "TOPO_SEQ_MIN_OVERLAP": "0.06",
}
# The Req-2 reward fix: also let TopoReward fall back to the whole completion
# when there is no <think> block (non-Qwen / non-SFT models).
_ALL = list(PROSE_PROFILE) + ["TOPO_TOPO_NO_THINK_FALLBACK"]


def _reload(flags):
    for k in _ALL:
        os.environ.pop(k, None)
    for k, v in flags.items():
        os.environ[k] = str(v)
    import src.data.build_dag as bd
    import src.reward.topo_reward as tr
    importlib.reload(bd)
    importlib.reload(tr)
    return tr


def _score(tr_mod, traces):
    reward = tr_mod.TopoReward()
    comps = [[{"role": "assistant", "content": t}] for t in traces]
    return reward(comps)


def main():
    bare = PROSE
    wrapped = [f"<think>{t}</think><answer>{t.split()[-1].rstrip('.')}</answer>" for t in PROSE]

    print("=" * 64)
    print("A. Llama bare prose, DEFAULT flags (current released behaviour)")
    tr = _reload({})
    s = _score(tr, bare)
    print(f"   topo scores = {[round(x,4) for x in s]}  std={statistics.pstdev(s):.4f}")

    print("\nB. SFT-style <think> wrapped, DEFAULT flags (DR1-7B case)")
    tr = _reload({})
    s = _score(tr, wrapped)
    print(f"   topo scores = {[round(x,4) for x in s]}  std={statistics.pstdev(s):.4f}")

    print("\nC. Llama bare prose, PROSE PROFILE + no-think fallback (the FIX)")
    prof = dict(PROSE_PROFILE, TOPO_TOPO_NO_THINK_FALLBACK="1")
    tr = _reload(prof)
    s = _score(tr, bare)
    print(f"   topo scores = {[round(x,4) for x in s]}  std={statistics.pstdev(s):.4f}")

    out = {
        "A_llama_bare_default": _score(_reload({}), bare),
        "B_sft_wrapped_default": _score(_reload({}), wrapped),
        "C_llama_bare_fixed": _score(_reload(dict(PROSE_PROFILE, TOPO_TOPO_NO_THINK_FALLBACK="1")), bare),
    }
    Path(REPO / "rebuttal/outputs/llama_reward_diagnosis.json").write_text(
        json.dumps({k: [round(x, 4) for x in v] for k, v in out.items()}, indent=2)
    )
    a_std = statistics.pstdev(out["A_llama_bare_default"])
    c_std = statistics.pstdev(out["C_llama_bare_fixed"])
    print(f"\nSUMMARY: bare-prose topo std {a_std:.4f} (default, inert) -> "
          f"{c_std:.4f} (fixed, usable signal). -> rebuttal/outputs/llama_reward_diagnosis.json")


if __name__ == "__main__":
    main()
