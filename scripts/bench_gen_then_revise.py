#!/usr/bin/env python3
"""topology-verified Generate-then-Revise evaluation.

For each benchmark item:
    1) y_init = model.generate(x)           (first attempt)
    2) Score r_out, r_topo on y_init
    3) P_r = build_prompt_dispatch(...)
    4) y_revised = model.generate(x, y_init, P_r)
    5) Report both First-Attempt and Revised accuracy

Usage:
    CUDA_VISIBLE_DEVICES=4 python3 scripts/bench_gen_then_revise.py \
        --model ${MODEL_ROOT}/qwen/Qwen3.5-9B \
        --adapter output/srt_9b/final \
        --label srt_9b_gtr \
        --benchmarks aime2024 math500 \
        --max_items 200
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.bench_transformers import (
    answer_extractor_for_benchmark,
    matcher_for_benchmark,
    load_benchmark,
    patch_swift_adapter_namespace,
)
from src.distill.build_srt_data import build_prompt_dispatch
from src.reward.outcome_reward import OutcomeReward
from src.reward.topo_reward import TopoReward


SYSTEM_PROMPT = (
    "You are a math reasoning assistant. Think step by step inside "
    "<think>...</think>, then put the final answer inside <answer>...</answer>."
)


@torch.inference_mode()
def sample(model, tok, msgs, *, max_new_tokens=1536, temperature=0.7):
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=5120).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    plen = inputs["input_ids"][0].shape[0]
    text = tok.decode(out[0][plen:], skip_special_tokens=True)
    gen_tokens = int(out[0].shape[0] - plen)
    return text, gen_tokens


def score_trace(text: str, gold: str, matcher, extractor) -> dict:
    """Binary r_out via matcher against gold; r_topo via TopoReward."""
    r_out_bin = 1 if matcher(extractor(text), gold) else 0
    try:
        r_topo = float(TopoReward()([[{"role": "assistant", "content": text}]])[0])
    except Exception:
        r_topo = 0.0
    return {"r_out": r_out_bin, "r_topo": r_topo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--label", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=["aime2024", "math500"])
    ap.add_argument("--max_items", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--topo_threshold", type=float, default=0.5)
    ap.add_argument("--use_topo_aware_pr", action="store_true", default=True)
    ap.add_argument("--no_topo_aware_pr", dest="use_topo_aware_pr", action="store_false")
    ap.add_argument("--output_dir", default="output/eval")
    args = ap.parse_args()

    print(f"Loading model {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    if args.adapter and Path(args.adapter).is_dir():
        print(f"Loading adapter {args.adapter}")
        patched = patch_swift_adapter_namespace(Path(args.adapter))
        model = PeftModel.from_pretrained(model, str(patched))
        shutil.rmtree(patched.parent, ignore_errors=True)
        model = model.merge_and_unload()
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for bench in args.benchmarks:
        items = load_benchmark(bench)
        if args.max_items and args.max_items < len(items):
            items = items[: args.max_items]

        extractor = answer_extractor_for_benchmark(bench)
        matcher = matcher_for_benchmark(bench)

        t0 = time.time()
        per_item = []
        first_correct = 0
        rev_correct = 0
        first_len = 0
        rev_len = 0
        for i, it in enumerate(items):
            msgs_init = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": it["question"]},
            ]
            y_init, tk_init = sample(model, tok, msgs_init,
                                     max_new_tokens=args.max_new_tokens,
                                     temperature=args.temperature)
            sc_init = score_trace(y_init, it["gold"], matcher, extractor)
            if args.use_topo_aware_pr:
                _bk, P_r = build_prompt_dispatch(
                    sc_init["r_out"], sc_init["r_topo"], args.topo_threshold,
                )
            else:
                P_r = ("Let me rephrase the above solution." if sc_init["r_out"]
                       else "Wait, this response is not correct, let me start over.")

            msgs_rev = msgs_init + [
                {"role": "assistant", "content": y_init},
                {"role": "user", "content": P_r},
            ]
            y_rev, tk_rev = sample(model, tok, msgs_rev,
                                    max_new_tokens=args.max_new_tokens,
                                    temperature=args.temperature)
            sc_rev = score_trace(y_rev, it["gold"], matcher, extractor)

            first_correct += sc_init["r_out"]
            rev_correct += sc_rev["r_out"]
            first_len += tk_init
            rev_len += tk_rev

            per_item.append({
                "gold": it["gold"],
                "first": extractor(y_init),
                "revised": extractor(y_rev),
                "first_correct": sc_init["r_out"],
                "revised_correct": sc_rev["r_out"],
                "P_r": P_r,
                "first_tokens": tk_init,
                "revised_tokens": tk_rev,
                "r_topo_init": sc_init["r_topo"],
                "r_topo_revised": sc_rev["r_topo"],
            })

            if (i + 1) % 10 == 0:
                n = i + 1
                print(f"  [{n}/{len(items)}] first={first_correct/n*100:.1f}% "
                      f"revised={rev_correct/n*100:.1f}% "
                      f"delta=+{(rev_correct-first_correct)/n*100:.1f}%")

        elapsed = time.time() - t0
        n = len(items)
        metrics = {
            "label": args.label,
            "benchmark": bench,
            "n_items": n,
            "first_attempt_acc": round(first_correct / n, 4),
            "revised_attempt_acc": round(rev_correct / n, 4),
            "revision_gain": round((rev_correct - first_correct) / n, 4),
            "avg_first_tokens": round(first_len / n, 1),
            "avg_revised_tokens": round(rev_len / n, 1),
            "topo_aware_pr": args.use_topo_aware_pr,
            "elapsed_sec": round(elapsed, 1),
        }

        mp = out_dir / f"{args.label}_{bench}_gtr.json"
        mp.write_text(json.dumps(metrics, indent=2) + "\n")
        dp = out_dir / f"{args.label}_{bench}_gtr_details.jsonl"
        with dp.open("w") as f:
            for r in per_item:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"\n{bench}: first={metrics['first_attempt_acc']*100:.1f}% "
              f"-> revised={metrics['revised_attempt_acc']*100:.1f}% "
              f"(gain +{metrics['revision_gain']*100:.1f}%) in {elapsed:.0f}s")
        print(f"  Saved {mp}")


if __name__ == "__main__":
    main()
