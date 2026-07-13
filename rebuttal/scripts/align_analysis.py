#!/usr/bin/env python3
"""A1: structure->correctness alignment after training.

For each trained policy's trace pool, compute how well a HIGH topology score
predicts a CORRECT answer. TopoPRM training should tighten this coupling
(higher Pr(correct|high q_topo) and higher point-biserial correlation between
q_topo and correctness) relative to outcome-only / outcome+length training.

Reuses semantic_gap scoring (topology score) + outcome correctness.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main() -> None:
    from src.data.build_dag import build_dag_from_answer  # noqa
    from src.reward.topo_reward import TopoReward
    from src.reward.outcome_reward import OutcomeReward

    topo = TopoReward()
    topo.REQUIRE_VALID_DAG = False
    outcome = OutcomeReward()

    def q_topo(resp: str) -> float:
        wrapped = resp if "<think>" in resp else f"<think>{resp}</think>"
        try:
            return topo([[{"role": "assistant", "content": wrapped}]])[0]
        except Exception:
            return 0.0

    def correct(resp: str, gold: str) -> int:
        try:
            return int(outcome([[{"role": "assistant", "content": resp}]], solution=[gold])[0])
        except Exception:
            return 0

    results = {}
    for fp in sorted(glob.glob("rebuttal/outputs/align_pool_*.jsonl")):
        label = os.path.basename(fp).replace("align_pool_", "").replace(".jsonl", "")
        rows = [json.loads(l) for l in open(fp) if l.strip()]
        qs, cs = [], []
        for r in rows:
            qs.append(q_topo(r["response"]))
            cs.append(correct(r["response"], r["gold"]))
        n = len(qs)
        if n == 0:
            continue
        thr = sorted(qs)[int(0.66 * n)]  # top third by topology
        hi = [(q, c) for q, c in zip(qs, cs) if q >= thr]
        lo = [(q, c) for q, c in zip(qs, cs) if q < thr]
        acc = sum(cs) / n
        pr_c_hi = sum(c for _, c in hi) / len(hi) if hi else None
        pr_c_lo = sum(c for _, c in lo) / len(lo) if lo else None
        # point-biserial correlation between q_topo and correctness
        import statistics
        try:
            mq = statistics.mean(qs); mc = statistics.mean(cs)
            sq = statistics.pstdev(qs); sc = statistics.pstdev(cs)
            cov = sum((q - mq) * (c - mc) for q, c in zip(qs, cs)) / n
            corr = cov / (sq * sc) if sq > 0 and sc > 0 else 0.0
        except Exception:
            corr = 0.0
        results[label] = {
            "n": n, "acc": round(acc, 4),
            "mean_qtopo": round(statistics.mean(qs), 4),
            "pr_correct_given_high_topo": round(pr_c_hi, 4) if pr_c_hi is not None else None,
            "pr_correct_given_low_topo": round(pr_c_lo, 4) if pr_c_lo is not None else None,
            "topo_correctness_corr": round(corr, 4),
        }
    Path("rebuttal/outputs/align_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
