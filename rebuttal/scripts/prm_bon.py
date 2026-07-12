#!/usr/bin/env python3
"""Best-of-N reranking baseline: Qwen2.5-Math-PRM-7B vs TopoPRM process score.

Answers B5w7 W3 / TsKG (a real PRM baseline instead of a text promise).

Protocol (matched, single shared candidate pool):
  1. generate: sample N candidates per problem from a policy model.
  2. score:    rerank the SAME pool with
                 - maj@N  (self-consistency majority vote)
                 - PRM-rm@N (Qwen2.5-Math-PRM-7B, product of step scores)
                 - Topo-rm@N (TopoPRM hierarchical/topo process score)
               and report accuracy of the selected answer vs pass@1 (greedy).

Two phases so generation (policy) and scoring (PRM+Topo) are decoupled.

  python prm_bon.py generate --model <policy> --benchmarks gsm8k math500 --n 100 --num_samples 8
  python prm_bon.py score    --pool <pool.jsonl> --prm /path/Qwen2.5-Math-PRM-7B
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BENCH_FILES = {
    "gsm8k": REPO / "data" / "benchmarks" / "GSM8K" / "test.jsonl",
    "math500": REPO / "data" / "benchmarks" / "MATH-500" / "test.jsonl",
}
SYS_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def _boxed(text: str) -> str | None:
    m = list(re.finditer(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text))
    if m:
        return m[-1].group(1).strip()
    m2 = list(re.finditer(r"(-?\d+\.?\d*)", text))
    return m2[-1].group(1) if m2 else None


def _correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    from src.reward.outcome_reward import OutcomeReward
    r = OutcomeReward()
    try:
        return r._verify_equivalence(pred, str(gold))  # type: ignore[attr-defined]
    except Exception:
        return pred.strip() == str(gold).strip()


def _load_problems(bench: str, n: int, seed: int) -> list[dict]:
    import random
    rows = [json.loads(l) for l in BENCH_FILES[bench].open() if l.strip()]
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        q = r.get("Problem") or r.get("question") or r.get("problem") or ""
        g = r.get("Answer") or r.get("answer") or r.get("final_answer") or ""
        if q:
            out.append({"question": q, "gold": str(g), "bench": bench})
    return out


def cmd_generate(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    probs = []
    for b in args.benchmarks:
        probs.extend(_load_problems(b, args.n, args.seed))
    print(f"[gen] {len(probs)} problems x {args.num_samples} samples", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = sum(1 for _ in out.open()) if (out.exists() and args.resume) else 0
    with out.open("a" if (args.resume and done) else "w") as f:
        for k, p in enumerate(probs):
            if k < done:
                continue
            msgs = [{"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": p["question"]}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt").to(model.device)
            cands = []
            for s in range(args.num_samples):
                with torch.no_grad():
                    g = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens,
                        do_sample=(s > 0), temperature=(0.0 if s == 0 else args.temperature),
                        top_p=0.95, pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id, repetition_penalty=1.05)
                cands.append(tok.decode(g[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
            f.write(json.dumps({**p, "candidates": cands}, ensure_ascii=False) + "\n")
            f.flush()
            if (k + 1) % 10 == 0:
                print(f"[gen {k+1}/{len(probs)}]", flush=True)
    print(f"[gen] done -> {out}")


def _prm_scores(model, tok, sep_id, question: str, response: str) -> list[float]:
    import torch
    steps = [s for s in response.split("\n\n") if s.strip()]
    if not steps:
        steps = [response]
    msgs = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
    ]
    conv = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    ids = tok.encode(conv, return_tensors="pt").to(model.device)
    with torch.no_grad():
        # use_cache=False avoids the bundled modeling code's DynamicCache
        # .from_legacy_cache path, which was removed in transformers>=5.
        out = model(input_ids=ids, use_cache=False)
    import torch.nn.functional as F
    probs = F.softmax(out[0], dim=-1)
    mask = (ids == sep_id)
    probs = probs * mask.unsqueeze(-1)
    sample = probs[0]
    pos = sample[sample != 0].view(-1, 2)[:, 1]
    return pos.cpu().tolist()


def _agg(scores: list[float], mode: str) -> float:
    if not scores:
        return 0.0
    if mode == "prod":
        p = 1.0
        for s in scores:
            p *= s
        return p
    if mode == "min":
        return min(scores)
    if mode == "last":
        return scores[-1]
    return sum(scores) / len(scores)


def cmd_score(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer
    from src.reward.topo_reward import TopoReward

    rows = [json.loads(l) for l in Path(args.pool).open() if l.strip()]

    # PRM. The bundled remote modeling code expects config.pad_token_id, which
    # transformers>=5 no longer sets by default on this custom config; inject it.
    from transformers import AutoConfig
    ptok = AutoTokenizer.from_pretrained(args.prm, trust_remote_code=True)
    pcfg = AutoConfig.from_pretrained(args.prm, trust_remote_code=True)
    if not hasattr(pcfg, "pad_token_id") or pcfg.pad_token_id is None:
        pcfg.pad_token_id = ptok.pad_token_id or ptok.eos_token_id
    pmodel = AutoModel.from_pretrained(args.prm, config=pcfg, torch_dtype=torch.bfloat16,
                                       device_map="auto", trust_remote_code=True).eval()
    sep_id = ptok.encode("<extra_0>")[0]

    topo = TopoReward()
    topo.REQUIRE_VALID_DAG = False

    def topo_score(resp: str) -> float:
        wrapped = resp if "<think>" in resp else f"<think>{resp}</think>"
        try:
            return topo([[{"role": "assistant", "content": wrapped}]])[0]
        except Exception:
            return 0.0

    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(int))  # bench -> metric -> correct count
    n_by_bench: Counter = Counter()

    for r in rows:
        bench, gold, cands = r["bench"], r["gold"], r["candidates"]
        n_by_bench[bench] += 1
        flags = [_correct(_boxed(c), gold) for c in cands]
        preds = [_boxed(c) for c in cands]

        # pass@1 (greedy = sample 0)
        agg[bench]["pass@1"] += int(flags[0])
        # oracle pass@N
        agg[bench]["pass@N"] += int(any(flags))
        # maj@N
        vote = Counter(p for p in preds if p is not None)
        if vote:
            maj = vote.most_common(1)[0][0]
            agg[bench]["maj@N"] += int(_correct(maj, gold))
        # PRM-rm@N
        prm_s = [_agg(_prm_scores(pmodel, ptok, sep_id, r["question"], c), args.prm_agg) for c in cands]
        agg[bench][f"prm-rm@N"] += int(flags[int(max(range(len(cands)), key=lambda i: prm_s[i]))])
        # Topo-rm@N
        topo_s = [topo_score(c) for c in cands]
        agg[bench]["topo-rm@N"] += int(flags[int(max(range(len(cands)), key=lambda i: topo_s[i]))])

    result = {}
    for bench, n in n_by_bench.items():
        result[bench] = {m: round(agg[bench][m] / n, 4) for m in
                         ["pass@1", "maj@N", "prm-rm@N", "topo-rm@N", "pass@N"]}
        result[bench]["n"] = n
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[score] prm_agg={args.prm_agg} -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--benchmarks", nargs="+", default=["gsm8k", "math500"])
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--num_samples", type=int, default=8)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--max_new_tokens", type=int, default=2048)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--out", default="rebuttal/outputs/prm_bon_pool.jsonl")
    g.add_argument("--resume", action="store_true", default=True)
    g.set_defaults(func=cmd_generate)
    s = sub.add_parser("score")
    s.add_argument("--pool", default="rebuttal/outputs/prm_bon_pool.jsonl")
    s.add_argument("--prm", default="/Knowin/foundation/models/Qwen/Qwen2.5-Math-PRM-7B")
    s.add_argument("--prm_agg", default="prod", choices=["prod", "min", "last", "mean"])
    s.add_argument("--out", default="rebuttal/outputs/prm_bon_results.json")
    s.set_defaults(func=cmd_score)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
