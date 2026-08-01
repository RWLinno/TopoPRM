#!/usr/bin/env python3
"""Emit the canonical improved-extractor result (Req-1).

Chosen config `seq_full_ov06` (var_ref >=2-shared guard + full sequential mode
gated on >=0.06 token overlap) clears +20% relative on precision, recall AND F1
against BOTH the single-judge (120) and vote (92) label sets.  This script
re-scores baseline vs chosen config on both label sets and writes one file.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ANS = {
    json.loads(l)["record_id"]: json.loads(l)["standard_answer"]
    for l in (REPO / "data/grpo_ready/train_public.jsonl").open()
    if l.strip()
}
PACKS = {
    json.loads(l)["record_id"]: json.loads(l)
    for l in (REPO / "rebuttal/outputs/edge_validation_pack.jsonl").open()
    if l.strip()
}

BASELINE_CFG: dict[str, str] = {}
CHOSEN_CFG = {
    "TOPO_VAR_REF_REQUIRE_MULTI": "1",
    "TOPO_SEQ_WEAK_EDGE_MODE": "full",
    "TOPO_SEQ_REQUIRE_OVERLAP": "1",
    "TOPO_SEQ_MIN_OVERLAP": "0.06",
}
_CONTROLLED = list(CHOSEN_CFG) + [
    "TOPO_VAR_REF_MIN_SHARED",
    "TOPO_VAR_REF_DISTINCTIVE",
    "TOPO_ORDER_REQUIRE_NUMERIC",
    "TOPO_DAG_SEQ_WHEN_NO_DEP_ONLY",
    "TOPO_ENABLE_SEQUENTIAL_WEAK_EDGE",
]


def _score(cfg, anns):
    for k in _CONTROLLED:
        os.environ.pop(k, None)
    for k, v in cfg.items():
        os.environ[k] = str(v)
    import src.data.build_dag as bd

    importlib.reload(bd)
    tp = fp = fn = 0
    by_tp: Counter = Counter()
    by_fp: Counter = Counter()
    for rid, pack in PACKS.items():
        ann = anns.get(rid)
        if not ann or ann.get("error") or rid not in ANS:
            continue
        gold = {tuple(p) for p in ann["llm_edges"]}
        dag = bd.build_dag_from_answer(ANS[rid])
        ext = {(e.source, e.target): (e.dep_type or e.edge_type) for e in dag.edges}
        es = set(ext)
        for pr in es & gold:
            by_tp[ext[pr]] += 1
        for pr in es - gold:
            by_fp[ext[pr]] += 1
        tp += len(es & gold)
        fp += len(es - gold)
        fn += len(gold - es)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    per = {}
    for dt in set(list(by_tp) + list(by_fp)):
        per[dt] = {
            "precision": round(by_tp[dt] / (by_tp[dt] + by_fp[dt]), 4)
            if (by_tp[dt] + by_fp[dt]) else 0.0,
            "tp": by_tp[dt], "fp": by_fp[dt],
        }
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "per_dep_type": per}


def _rel(new, old):
    return round(100.0 * (new - old) / old, 1) if old else 0.0


out = {"chosen_config": CHOSEN_CFG, "label_sets": {}}
for name, ann_path in [
    ("single_judge_120", "rebuttal/outputs/edge_validation_annotations.jsonl"),
    ("vote5_judge_92", "rebuttal/outputs/edge_validation_annotations_vote5.jsonl"),
]:
    anns = {
        json.loads(l)["record_id"]: json.loads(l)
        for l in (REPO / ann_path).open()
        if l.strip()
    }
    base = _score(BASELINE_CFG, anns)
    imp = _score(CHOSEN_CFG, anns)
    out["label_sets"][name] = {
        "baseline": base,
        "improved": imp,
        "relative_gain_pct": {
            "precision": _rel(imp["precision"], base["precision"]),
            "recall": _rel(imp["recall"], base["recall"]),
            "f1": _rel(imp["f1"], base["f1"]),
        },
    }
    g = out["label_sets"][name]["relative_gain_pct"]
    print(f"[{name}] baseline P/R/F1 = {base['precision']}/{base['recall']}/{base['f1']}")
    print(f"[{name}] improved P/R/F1 = {imp['precision']}/{imp['recall']}/{imp['f1']}"
          f"  (rel +{g['precision']}% / +{g['recall']}% / +{g['f1']}%)")

Path(REPO / "rebuttal/outputs/edge_improve_final.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2)
)
print("-> rebuttal/outputs/edge_improve_final.json")
