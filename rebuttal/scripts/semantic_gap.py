#!/usr/bin/env python3
"""Structure-semantic gap analysis (answers B5w7 W2, TsKG comment).

Generates traces from a model on stratified math problems, computes the
topology score q_topo and continuity score q_cont for each trace, and the
final-answer correctness r_out. Then reports:

  * Pr(wrong | q_topo > hi)   -- structurally clean but semantically wrong
  * Pr(correct | q_topo < lo) -- structurally poor but semantically correct
  * high-topology-wrong rate, per benchmark

Two phases:
    python semantic_gap.py generate --model <path> --benchmarks gsm8k math500 --n 100
    python semantic_gap.py score    --pool <pool.jsonl> --out <table.csv>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.build_dag import build_dag_from_answer  # noqa: E402
from src.reward.topo_reward import TopoReward  # noqa: E402
from src.reward.continuity_reward import ContinuityReward  # noqa: E402
from src.reward.outcome_reward import OutcomeReward  # noqa: E402

BENCH_FILES = {
    "gsm8k": REPO / "data" / "benchmarks" / "GSM8K" / "test.jsonl",
    "math500": REPO / "data" / "benchmarks" / "MATH-500" / "test.jsonl",
    "olympiad": REPO / "data" / "benchmarks" / "MATH" / "test.jsonl",
}

_PROMPT = (
    "Solve the following math problem step by step. Put your final answer in "
    "\\boxed{{}}.\n\nProblem: {q}\n"
)


def _load_problems(bench: str, n: int, seed: int) -> list[dict[str, Any]]:
    import random

    fp = BENCH_FILES[bench]
    rows = [json.loads(l) for l in fp.open() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    out = []
    for r in rows[:n]:
        q = r.get("question") or r.get("problem") or r.get("Problem") or ""
        gold = (
            r.get("answer")
            or r.get("final_answer")
            or r.get("solution")
            or r.get("Answer")
            or ""
        )
        if q:
            out.append({"question": q, "gold": str(gold), "bench": bench})
    return out


def cmd_generate(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    probs: list[dict[str, Any]] = []
    for b in args.benchmarks:
        probs.extend(_load_problems(b, args.n, args.seed))
    print(f"[generate] {len(probs)} problems from {args.benchmarks}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    if out.exists() and args.resume:
        done = sum(1 for _ in out.open())
    with out.open("a" if args.resume else "w") as f:
        for k, p in enumerate(probs):
            if k < done:
                continue
            msgs = [{"role": "user", "content": _PROMPT.format(q=p["question"])}]
            ct = dict(tokenize=False, add_generation_prompt=True)
            try:
                import inspect

                if "enable_thinking" in inspect.signature(tok.apply_chat_template).parameters:
                    ct["enable_thinking"] = False
            except Exception:
                pass
            text = tok.apply_chat_template(msgs, **ct)
            inputs = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs, max_new_tokens=args.max_tokens, do_sample=(args.temperature > 0),
                    temperature=args.temperature or None, top_p=0.95 if args.temperature > 0 else None,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            resp = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            f.write(json.dumps({**p, "response": resp}, ensure_ascii=False) + "\n")
            f.flush()
            if (k + 1) % 10 == 0:
                print(f"[generate {k+1}/{len(probs)}]", flush=True)
    print(f"[generate] done -> {out}")


def cmd_score(args: argparse.Namespace) -> None:
    # Use the continuous structural topology score (do not hard-gate invalid
    # DAGs to 0), so q_topo varies across model traces and the gap analysis is
    # meaningful. The hard validity gate is a training-time device; here we
    # want the graded structural quality.
    topo = TopoReward()
    topo.REQUIRE_VALID_DAG = False
    cont = ContinuityReward()
    outcome = OutcomeReward()

    from src.data.build_dag import build_dag_from_answer
    from src.reward.utils import extract_think_block

    def _wrap_think(text: str) -> str:
        # These model traces have no <think> tags; the topology/continuity
        # rewards score the think block. Wrap the reasoning (everything before
        # the final boxed answer) so the extractor sees the full trace.
        if "<think>" in text:
            return text
        return f"<think>{text}</think>"

    rows = [json.loads(l) for l in Path(args.pool).open() if l.strip()]
    recs = []
    for r in rows:
        wrapped = _wrap_think(r["response"])
        comp = [{"role": "assistant", "content": wrapped}]
        try:
            qt = topo([comp])[0]
        except Exception:
            qt = 0.0
        try:
            qc = cont([comp])[0]
        except Exception:
            qc = 0.0
        try:
            ro = outcome([[{"role": "assistant", "content": r["response"]}]], solution=[r["gold"]])[0]
        except Exception:
            ro = 0.0
        recs.append({"bench": r["bench"], "q_topo": qt, "q_cont": qc, "r_out": ro})

    hi, lo = args.hi, args.lo
    def _stats(subset):
        n = len(subset)
        if n == 0:
            return {}
        hi_topo = [x for x in subset if x["q_topo"] > hi]
        lo_topo = [x for x in subset if x["q_topo"] < lo]
        wrong_hi = sum(1 for x in hi_topo if x["r_out"] == 0)
        corr_lo = sum(1 for x in lo_topo if x["r_out"] == 1)
        return {
            "n": n,
            "acc": round(sum(x["r_out"] for x in subset) / n, 4),
            "mean_qtopo": round(sum(x["q_topo"] for x in subset) / n, 4),
            "n_high_topo": len(hi_topo),
            "pr_wrong_given_high_topo": round(wrong_hi / len(hi_topo), 4) if hi_topo else None,
            "n_low_topo": len(lo_topo),
            "pr_correct_given_low_topo": round(corr_lo / len(lo_topo), 4) if lo_topo else None,
        }

    benches = sorted(set(r["bench"] for r in recs))
    table = {b: _stats([x for x in recs if x["bench"] == b]) for b in benches}
    table["ALL"] = _stats(recs)

    # CSV
    cols = ["bench", "n", "acc", "mean_qtopo", "n_high_topo",
            "pr_wrong_given_high_topo", "n_low_topo", "pr_correct_given_low_topo"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(",".join(cols) + "\n")
        for b, s in table.items():
            if not s:
                continue
            f.write(",".join(str(s.get(c, b if c == "bench" else "")) if c != "bench" else b for c in cols) + "\n")
    print(json.dumps(table, indent=2))
    print(f"[score] hi={hi} lo={lo} -> {out}")
    Path(str(out).replace(".csv", ".json")).write_text(json.dumps(table, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--benchmarks", nargs="+", default=["gsm8k", "math500"])
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--max-tokens", type=int, default=2048)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--out", default="rebuttal/outputs/semantic_gap_pool.jsonl")
    g.add_argument("--resume", action="store_true", default=True)
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("score")
    s.add_argument("--pool", default="rebuttal/outputs/semantic_gap_pool.jsonl")
    s.add_argument("--hi", type=float, default=0.8)
    s.add_argument("--lo", type=float, default=0.5)
    s.add_argument("--out", default="rebuttal/outputs/semantic_gap_table.csv")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
