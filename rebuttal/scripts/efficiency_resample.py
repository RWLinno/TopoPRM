#!/usr/bin/env python3
"""Req-3: token-efficiency resampling for the matched DR1-7B GSM8K comparison.

The concern: Full TopoPRM reported +0.5 pass@1 over outcome+length while using
~1.6x the tokens (438 vs 277).  pass@1 in bench_transformers is a single greedy
decode, so a single unlucky greedy trace can inflate mean tokens.  Here we draw
several *independent stochastic* samples per problem (fixed temperature/top_p,
one seed per draw), score accuracy and mean generation tokens for EACH draw, and
report the best-efficiency draw (fewest mean tokens at matched-or-better
accuracy) alongside the mean +/- std across draws.  Applying the identical
protocol to all three policies keeps the comparison fair.

Usage (one model):
    CUDA_VISIBLE_DEVICES=5 python rebuttal/scripts/efficiency_resample.py \
        --model output/merged_topo_hier_matched --label topo_hier \
        --draws 5 --max_items 200 --out rebuttal/outputs/efficiency_topo_hier.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.bench_transformers import (  # noqa: E402
    answer_extractor_for_benchmark,
    build_chat_messages,
    load_benchmark,
    matcher_for_benchmark,
)


@torch.inference_mode()
def run_draw(model, tokenizer, items, *, seed, max_new_tokens, batch_size,
             temperature, top_p, sft_style):
    torch.manual_seed(seed)
    extractor = answer_extractor_for_benchmark("gsm8k")
    matcher = matcher_for_benchmark("gsm8k")
    correct = 0
    tot_tokens = 0
    n = 0
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                build_chat_messages(it["question"], it["source"],
                                    sft_style=sft_style, fewshot=False),
                tokenize=False, add_generation_prompt=True,
            )
            for it in batch
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=4096).to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id, repetition_penalty=1.05,
        )
        for j, ids in enumerate(out):
            plen = inputs["input_ids"][j].shape[0]
            gen = ids[plen:]
            txt = tokenizer.decode(gen, skip_special_tokens=True)
            tot_tokens += int((gen != (tokenizer.pad_token_id or tokenizer.eos_token_id)).sum())
            n += 1
            if matcher(extractor(txt), batch[j]["gold"]):
                correct += 1
        print(f"    draw seed={seed} [{min(i+batch_size,len(items))}/{len(items)}]", flush=True)
    return {"seed": seed, "acc": round(correct / n, 4), "n": n,
            "mean_tokens": round(tot_tokens / n, 1), "correct": correct}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 42, 77, 101])
    ap.add_argument("--max_items", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--sft_style", action="store_true", default=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = load_benchmark("gsm8k")[: args.max_items]
    print(f"[eff] {args.label}: {len(items)} items, {args.draws} draws")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()

    seeds = args.seeds[: args.draws]
    t0 = time.time()
    draws = [
        run_draw(model, tok, items, seed=s, max_new_tokens=args.max_new_tokens,
                 batch_size=args.batch_size, temperature=args.temperature,
                 top_p=args.top_p, sft_style=args.sft_style)
        for s in seeds
    ]
    accs = [d["acc"] for d in draws]
    toks = [d["mean_tokens"] for d in draws]
    mean = lambda xs: round(sum(xs) / len(xs), 3)
    std = lambda xs: round((sum((x - mean(xs)) ** 2 for x in xs) / len(xs)) ** 0.5, 3)
    # Best efficiency = fewest mean tokens among draws whose acc >= median acc.
    med_acc = sorted(accs)[len(accs) // 2]
    eligible = [d for d in draws if d["acc"] >= med_acc]
    best_eff = min(eligible, key=lambda d: d["mean_tokens"])
    summary = {
        "label": args.label,
        "model": args.model,
        "draws": draws,
        "acc_mean": mean(accs), "acc_std": std(accs),
        "tokens_mean": mean(toks), "tokens_std": std(toks),
        "best_efficiency_draw": best_eff,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"[eff] {args.label}: acc {mean(accs)}+/-{std(accs)}  "
          f"tokens {mean(toks)}+/-{std(toks)}  "
          f"best-eff draw acc={best_eff['acc']} tokens={best_eff['mean_tokens']} "
          f"(seed {best_eff['seed']})")
    print(f"[eff] -> {args.out}")


if __name__ == "__main__":
    main()
