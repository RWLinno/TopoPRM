#!/usr/bin/env python3
"""Inter-annotator agreement + extractor P/R/F1 against two independent annotators.

Reference set A: Qwen3-32B annotator (edge_validation_annotations.jsonl)
Reference set B: a second, architecturally-distinct annotator
                 (edge_validation_annotations_qwen25_32b.jsonl)

We treat each ordered step pair (i<j) within a trace as a binary item
("is i a necessary support for j?"). Agreement is computed over the union of
candidate pairs each annotator considered (all ordered pairs with i<j up to
n_steps). We report:
  * raw pairwise agreement between the two annotators
  * Cohen's kappa
  * extractor edge-level P/R/F1 against each reference and against their
    intersection (consensus) and union.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "rebuttal/outputs/edge_validation_pack.jsonl"
ANN_A = REPO / "rebuttal/outputs/edge_validation_annotations.jsonl"
ANN_B = REPO / "rebuttal/outputs/edge_validation_annotations_qwen25_32b.jsonl"


def load_map(path):
    out = {}
    for l in open(path):
        if not l.strip():
            continue
        d = json.loads(l)
        if d.get("error"):
            continue
        out[d["record_id"]] = {tuple(p) for p in d["llm_edges"]}
    return out


def main():
    packs = {json.loads(l)["record_id"]: json.loads(l)
             for l in open(PACK) if l.strip()}
    A = load_map(ANN_A)
    B = load_map(ANN_B)
    common = [r for r in packs if r in A and r in B]
    print(f"[agreement] traces with both annotators: {len(common)}")

    # Agreement over all candidate ordered pairs (i<j).
    both00 = both11 = a1b0 = a0b1 = 0
    for rid in common:
        n = packs[rid]["n_steps"]
        allpairs = set(combinations(range(n), 2))
        a, b = A[rid], B[rid]
        for pr in allpairs:
            ia, ib = pr in a, pr in b
            if ia and ib:
                both11 += 1
            elif not ia and not ib:
                both00 += 1
            elif ia and not ib:
                a1b0 += 1
            else:
                a0b1 += 1
    tot = both00 + both11 + a1b0 + a0b1
    po = (both00 + both11) / tot if tot else 0.0
    # Cohen's kappa
    pa1 = (both11 + a1b0) / tot
    pb1 = (both11 + a0b1) / tot
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 0.0

    def prf(ref):
        tp = fp = fn = 0
        for rid in common:
            ext = {(e["source"], e["target"]) for e in packs[rid]["extractor_edges"]}
            g = ref[rid]
            tp += len(ext & g)
            fp += len(ext - g)
            fn += len(g - ext)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return round(p, 4), round(r, 4), round(f1, 4)

    consensus = {rid: A[rid] & B[rid] for rid in common}
    union = {rid: A[rid] | B[rid] for rid in common}

    result = {
        "n_traces": len(common),
        "raw_agreement": round(po, 4),
        "cohen_kappa": round(kappa, 4),
        "counts": {"both_support": both11, "both_none": both00,
                   "A_only": a1b0, "B_only": a0b1},
        "extractor_vs_A_qwen3_32b": dict(zip("PRF", prf(A))),
        "extractor_vs_B_qwen25_32b": dict(zip("PRF", prf(B))),
        "extractor_vs_consensus": dict(zip("PRF", prf(consensus))),
        "extractor_vs_union": dict(zip("PRF", prf(union))),
    }
    out = REPO / "rebuttal/outputs/edge_agreement.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[agreement] -> {out}")


if __name__ == "__main__":
    main()
