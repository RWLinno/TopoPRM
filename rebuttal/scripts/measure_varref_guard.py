#!/usr/bin/env python3
"""Measure precision effect of TOPO_VAR_REF_REQUIRE_MULTI guard.

Re-extracts DAGs for the edge-validation traces with the guard OFF and ON,
and re-scores both against the same independent-judge labels.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# map record_id -> standard_answer from training data
ans = {}
for l in open("data/grpo_ready/train_public.jsonl"):
    d = json.loads(l)
    ans[d["record_id"]] = d["standard_answer"]

packs = {json.loads(l)["record_id"]: json.loads(l)
         for l in open("rebuttal/outputs/edge_validation_pack.jsonl")}
anns = {json.loads(l)["record_id"]: json.loads(l)
        for l in open("rebuttal/outputs/edge_validation_annotations.jsonl")}


def score(require_multi: bool):
    os.environ["TOPO_VAR_REF_REQUIRE_MULTI"] = "1" if require_multi else "0"
    import importlib
    import src.data.build_dag as bd
    importlib.reload(bd)
    tp = fp = fn = 0
    for rid, pack in packs.items():
        ann = anns.get(rid)
        if not ann or ann.get("error") or rid not in ans:
            continue
        gold = {tuple(p) for p in ann["llm_edges"]}
        dag = bd.build_dag_from_answer(ans[rid])
        ext = {(e.source, e.target) for e in dag.edges}
        tp += len(ext & gold)
        fp += len(ext - gold)
        fn += len(gold - ext)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4), tp=tp, fp=fp, fn=fn)


off = score(False)
on = score(True)
out = {"guard_off": off, "guard_on": on}
Path("rebuttal/outputs/varref_guard_effect.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
