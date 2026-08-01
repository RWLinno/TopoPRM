#!/usr/bin/env python3
"""Sweep extractor configurations against the fixed independent-judge labels.

Req-1 of the rebuttal follow-up: push edge P/R/F1 by >20% relative over the
0.53 baseline by improving the extractor (not by re-labelling).  We keep the
judge annotations fixed (single-judge 120 traces = larger, more stable n; vote
92 traces as a secondary check) and only vary the rule extractor via env flags.

Each config re-imports src.data.build_dag under a set of TOPO_* env flags,
re-extracts DAG edges for every annotated trace, and recomputes edge-level
P/R/F1 plus per-dep-type precision against the SAME gold edges.

Usage:
    python rebuttal/scripts/edge_improve_sweep.py \
        --pack rebuttal/outputs/edge_validation_pack.jsonl \
        --ann  rebuttal/outputs/edge_validation_annotations.jsonl \
        --out  rebuttal/outputs/edge_improve_sweep.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(REPO))

ANS_PATH = REPO / "data" / "grpo_ready" / "train_public.jsonl"

# Env flags we control per config.  Anything not listed is cleared to default.
_CONTROLLED = [
    "TOPO_VAR_REF_REQUIRE_MULTI",
    "TOPO_VAR_REF_MIN_SHARED",
    "TOPO_VAR_REF_DISTINCTIVE",
    "TOPO_SEQ_REQUIRE_OVERLAP",
    "TOPO_SEQ_MIN_OVERLAP",
    "TOPO_DAG_SEQ_WHEN_NO_DEP_ONLY",
    "TOPO_ENABLE_SEQUENTIAL_WEAK_EDGE",
    "TOPO_SEQ_WEAK_EDGE_MODE",
    "TOPO_DAG_BARRIER_STRICT",
    "TOPO_EXPR_OVERLAP_MIN",
    "TOPO_ORDER_REQUIRE_NUMERIC",
]


def _load_answers() -> dict[str, str]:
    return {
        json.loads(l)["record_id"]: json.loads(l)["standard_answer"]
        for l in ANS_PATH.open()
        if l.strip()
    }


def _apply_env(cfg: dict[str, str]) -> None:
    for k in _CONTROLLED:
        os.environ.pop(k, None)
    for k, v in cfg.items():
        os.environ[k] = str(v)


def score_config(cfg: dict[str, str], packs, anns, answers) -> dict[str, Any]:
    _apply_env(cfg)
    import src.data.build_dag as bd

    importlib.reload(bd)

    tp = fp = fn = 0
    by_tp: Counter = Counter()
    by_fp: Counter = Counter()
    for rid, pack in packs.items():
        ann = anns.get(rid)
        if not ann or ann.get("error") or rid not in answers:
            continue
        gold = {tuple(p) for p in ann["llm_edges"]}
        dag = bd.build_dag_from_answer(answers[rid])
        ext = {(e.source, e.target): {"dep_type": e.dep_type, "edge_type": e.edge_type}
               for e in dag.edges}
        # Deduplicate edges by (src,tgt); keep first dep_type seen.
        ext_set = set(ext)
        cur_tp = ext_set & gold
        cur_fp = ext_set - gold
        tp += len(cur_tp)
        fp += len(cur_fp)
        fn += len(gold - ext_set)
        for pr in cur_tp:
            by_tp[ext[pr].get("dep_type") or ext[pr]["edge_type"]] += 1
        for pr in cur_fp:
            by_fp[ext[pr].get("dep_type") or ext[pr]["edge_type"]] += 1

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    per_type = {}
    for dt in set(list(by_tp) + list(by_fp)):
        d_tp, d_fp = by_tp[dt], by_fp[dt]
        per_type[dt] = {
            "precision": round(d_tp / (d_tp + d_fp), 4) if (d_tp + d_fp) else 0.0,
            "tp": d_tp,
            "fp": d_fp,
        }
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "per_dep_type": per_type,
    }


CONFIGS: dict[str, dict[str, str]] = {
    "baseline": {},
    "guard_varref2": {"TOPO_VAR_REF_REQUIRE_MULTI": "1"},
    "seq_overlap": {
        "TOPO_VAR_REF_REQUIRE_MULTI": "1",
        "TOPO_SEQ_REQUIRE_OVERLAP": "1",
        "TOPO_SEQ_MIN_OVERLAP": "0.12",
    },
    "seq_overlap_15": {
        "TOPO_VAR_REF_REQUIRE_MULTI": "1",
        "TOPO_SEQ_REQUIRE_OVERLAP": "1",
        "TOPO_SEQ_MIN_OVERLAP": "0.15",
    },
    "seq_numeric": {
        "TOPO_VAR_REF_REQUIRE_MULTI": "1",
        "TOPO_ORDER_REQUIRE_NUMERIC": "1",
    },
    "full_combo": {
        "TOPO_VAR_REF_REQUIRE_MULTI": "1",
        "TOPO_SEQ_REQUIRE_OVERLAP": "1",
        "TOPO_SEQ_MIN_OVERLAP": "0.12",
        "TOPO_ORDER_REQUIRE_NUMERIC": "1",
    },
    "full_combo_strict": {
        "TOPO_VAR_REF_REQUIRE_MULTI": "1",
        "TOPO_VAR_REF_MIN_SHARED": "2",
        "TOPO_SEQ_REQUIRE_OVERLAP": "1",
        "TOPO_SEQ_MIN_OVERLAP": "0.15",
        "TOPO_ORDER_REQUIRE_NUMERIC": "1",
    },
    # Full sequential mode (every consecutive pair) but gated on overlap so
    # only content-carrying adjacencies survive -> recovers the consecutive
    # support chains the adaptive mode suppresses (recall) without the raw
    # positional FP flood (precision).
    "seq_full_ov10": {
        "TOPO_VAR_REF_REQUIRE_MULTI": "1",
        "TOPO_SEQ_WEAK_EDGE_MODE": "full",
        "TOPO_SEQ_REQUIRE_OVERLAP": "1",
        "TOPO_SEQ_MIN_OVERLAP": "0.10",
    },
    "seq_full_ov08": {
        "TOPO_VAR_REF_REQUIRE_MULTI": "1",
        "TOPO_SEQ_WEAK_EDGE_MODE": "full",
        "TOPO_SEQ_REQUIRE_OVERLAP": "1",
        "TOPO_SEQ_MIN_OVERLAP": "0.08",
    },
    "seq_full_ov12": {
        "TOPO_VAR_REF_REQUIRE_MULTI": "1",
        "TOPO_SEQ_WEAK_EDGE_MODE": "full",
        "TOPO_SEQ_REQUIRE_OVERLAP": "1",
        "TOPO_SEQ_MIN_OVERLAP": "0.12",
    },
    # Distinctive-variable recall recovery: keep single-shared var_ref only
    # when that variable is "distinctive" (multi-char or appears in an expr),
    # combined with the full-overlap seq gate.
    "distinctive_var_seqfull": {
        "TOPO_VAR_REF_REQUIRE_MULTI": "1",
        "TOPO_VAR_REF_DISTINCTIVE": "1",
        "TOPO_SEQ_WEAK_EDGE_MODE": "full",
        "TOPO_SEQ_REQUIRE_OVERLAP": "1",
        "TOPO_SEQ_MIN_OVERLAP": "0.10",
    },
    "seq_full_ov06": {
        "TOPO_VAR_REF_REQUIRE_MULTI": "1",
        "TOPO_SEQ_WEAK_EDGE_MODE": "full",
        "TOPO_SEQ_REQUIRE_OVERLAP": "1",
        "TOPO_SEQ_MIN_OVERLAP": "0.06",
    },
    "seq_full_ov08_dist": {
        "TOPO_VAR_REF_REQUIRE_MULTI": "1",
        "TOPO_VAR_REF_DISTINCTIVE": "1",
        "TOPO_SEQ_WEAK_EDGE_MODE": "full",
        "TOPO_SEQ_REQUIRE_OVERLAP": "1",
        "TOPO_SEQ_MIN_OVERLAP": "0.08",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="rebuttal/outputs/edge_validation_pack.jsonl")
    ap.add_argument("--ann", default="rebuttal/outputs/edge_validation_annotations.jsonl")
    ap.add_argument("--out", default="rebuttal/outputs/edge_improve_sweep.json")
    args = ap.parse_args()

    packs = {
        json.loads(l)["record_id"]: json.loads(l)
        for l in Path(args.pack).open()
        if l.strip()
    }
    anns = {
        json.loads(l)["record_id"]: json.loads(l)
        for l in Path(args.ann).open()
        if l.strip()
    }
    answers = _load_answers()

    results = {}
    for name, cfg in CONFIGS.items():
        res = score_config(cfg, packs, anns, answers)
        results[name] = {"config": cfg, **res}
        print(f"[{name:20s}] P={res['precision']:.4f} R={res['recall']:.4f} "
              f"F1={res['f1']:.4f}  tp={res['tp']} fp={res['fp']} fn={res['fn']}")

    base_f1 = results["baseline"]["f1"]
    best = max(results, key=lambda k: results[k]["f1"])
    rel = 100.0 * (results[best]["f1"] - base_f1) / base_f1 if base_f1 else 0.0
    summary = {
        "baseline_f1": base_f1,
        "best_config": best,
        "best_f1": results[best]["f1"],
        "relative_gain_pct": round(rel, 1),
    }
    print(f"[summary] best={best} F1={results[best]['f1']:.4f} "
          f"(+{rel:.1f}% rel over {base_f1:.4f})")
    Path(args.out).write_text(json.dumps({"summary": summary, "configs": results},
                                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
