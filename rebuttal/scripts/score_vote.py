#!/usr/bin/env python3
"""Score extractor (guard on/off) against self-consistency vote labels.

Compares single-judge vs vote-judge labels, and default vs var_ref-guarded
extractor, to quantify the combined improvement.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ANN_SINGLE = "rebuttal/outputs/edge_validation_annotations.jsonl"
ANN_VOTE = sys.argv[1] if len(sys.argv) > 1 else "rebuttal/outputs/edge_validation_annotations_vote5.jsonl"

ans = {}
for l in open("data/grpo_ready/train_public.jsonl"):
    d = json.loads(l)
    ans[d["record_id"]] = d["standard_answer"]


def load_ann(fp):
    out = {}
    for l in open(fp):
        if l.strip():
            d = json.loads(l)
            if not d.get("error"):
                out[d["record_id"]] = {tuple(p) for p in d["llm_edges"]}
    return out


def score(ann, require_multi):
    os.environ["TOPO_VAR_REF_REQUIRE_MULTI"] = "1" if require_multi else "0"
    import importlib
    import src.data.build_dag as bd
    importlib.reload(bd)
    tp = fp = fn = 0
    for rid, gold in ann.items():
        if rid not in ans:
            continue
        dag = bd.build_dag_from_answer(ans[rid])
        ext = {(e.source, e.target) for e in dag.edges}
        tp += len(ext & gold)
        fp += len(ext - gold)
        fn += len(gold - ext)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4), n=len(ann))


res = {}
if Path(ANN_SINGLE).exists():
    a1 = load_ann(ANN_SINGLE)
    res["single_judge_guard_off"] = score(a1, False)
    res["single_judge_guard_on"] = score(a1, True)
av = load_ann(ANN_VOTE)
res["vote_judge_guard_off"] = score(av, False)
res["vote_judge_guard_on"] = score(av, True)

Path("rebuttal/outputs/edge_vote_comparison.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
