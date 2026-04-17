#!/usr/bin/env python3
"""TopoSD-Zero Phase 1 data rollout.

For each problem in input, sample y_init on-policy, score it with
(R_out, R_topo, R_cont), build a topology-aware revision prompt, then sample
y_revised. Output raw rollouts for src/distill/build_srt_data.py to filter.

Usage:
    CUDA_VISIBLE_DEVICES=4 python3 scripts/rollout_srt.py \
        --model /mnt/data/huggingface_downloads/models/qwen/Qwen3.5-9B \
        --adapter output/sft_qwen35_9b/v0-20260407-011328/checkpoint-626 \
        --input data/grpo_ready/train.jsonl \
        --output data/srt_raw/rollouts.jsonl \
        --max_prompts 2000 --samples_per_prompt 4
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.distill.build_srt_data import (
    REVISION_PROMPTS, build_prompt_dispatch, format_ok,
)
from src.reward.outcome_reward import OutcomeReward
from src.reward.topo_reward import TopoReward
from src.reward.continuity_reward import ContinuityReward
from src.data.build_dag import ReasoningDAG
from scripts.bench_transformers import patch_swift_adapter_namespace


def load_prompts(path: Path, max_n: int = 0) -> list[dict[str, Any]]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            messages = d.get("messages", [])
            # Extract user question from last user message
            user_text = ""
            for m in messages:
                if m.get("role") == "user":
                    user_text = m.get("content", "")
            if not user_text:
                continue
            out.append({
                "problem": user_text,
                "solution": d.get("solution", ""),
                "reference_dag": d.get("reference_dag"),
            })
            if max_n and len(out) >= max_n:
                break
    return out


def score_trace(text: str, solution: str, reference_dag=None) -> dict:
    """Compute r_out / r_topo / r_cont for a single completion."""
    completions = [[{"role": "assistant", "content": text}]]
    out_rw = OutcomeReward()
    topo_rw = TopoReward()
    cont_rw = ContinuityReward()

    try:
        r_out = float(out_rw(completions, solution=solution)[0])
    except Exception:
        r_out = 0.0
    try:
        r_topo = float(topo_rw(completions, reference_dag=reference_dag)[0])
    except Exception:
        r_topo = 0.0
    try:
        r_cont = float(cont_rw(completions)[0])
    except Exception:
        r_cont = 0.0

    # Binary r_out
    r_out_bin = 1 if r_out >= 0.5 else 0

    # Orphan step index: try to find first orphan conclusion node in DAG
    orphan_step = None
    try:
        dag = ReasoningDAG.from_trace(text) if hasattr(ReasoningDAG, "from_trace") else None
        if dag:
            orphans = dag.orphan_conclusion_nodes() if hasattr(dag, "orphan_conclusion_nodes") else []
            if orphans:
                orphan_step = int(orphans[0])
    except Exception:
        pass

    return {
        "r_out": r_out_bin,
        "r_out_raw": r_out,
        "r_topo": r_topo,
        "r_cont": r_cont,
        "orphan_step": orphan_step,
    }


SYSTEM_PROMPT = (
    "You are a math reasoning assistant. Think step by step inside "
    "<think>...</think>, then put the final answer inside <answer>...</answer>."
)


@torch.inference_mode()
def sample_response(model, tokenizer, msgs, *, max_new_tokens=2048,
                    temperature=0.8, top_p=0.95) -> str:
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"][0].shape[0]
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max_prompts", type=int, default=2000)
    ap.add_argument("--samples_per_prompt", type=int, default=2)
    ap.add_argument("--topo_threshold", type=float, default=0.5)
    ap.add_argument("--max_new_tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter and Path(args.adapter).is_dir():
        print(f"Loading adapter: {args.adapter}")
        patched = patch_swift_adapter_namespace(Path(args.adapter))
        model = PeftModel.from_pretrained(model, str(patched))
        shutil.rmtree(patched.parent, ignore_errors=True)
        model = model.merge_and_unload()
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = load_prompts(args.input, args.max_prompts)
    print(f"Loaded {len(prompts)} prompts")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with args.output.open("w") as fout:
        for pi, p in enumerate(prompts):
            problem = p["problem"]
            solution = p["solution"]
            ref_dag = p.get("reference_dag")

            for sample_i in range(args.samples_per_prompt):
                # 1) sample y_init
                msgs_init = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": problem},
                ]
                try:
                    y_init = sample_response(
                        model, tokenizer, msgs_init,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                except Exception as e:
                    print(f"  skip prompt {pi} init: {e}")
                    continue

                sc_init = score_trace(y_init, solution, ref_dag)

                # 2) build P_r and sample y_revised
                bucket, P_r = build_prompt_dispatch(
                    sc_init["r_out"], sc_init["r_topo"],
                    args.topo_threshold, sc_init.get("orphan_step"),
                )
                msgs_rev = msgs_init + [
                    {"role": "assistant", "content": y_init},
                    {"role": "user",      "content": P_r},
                ]
                try:
                    y_rev = sample_response(
                        model, tokenizer, msgs_rev,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                except Exception as e:
                    print(f"  skip prompt {pi} rev: {e}")
                    continue

                sc_rev = score_trace(y_rev, solution, ref_dag)

                rec = {
                    "problem": problem,
                    "solution": solution,
                    "y_init": y_init,
                    "P_r": P_r,
                    "y_revised": y_rev,
                    "r_out_init": sc_init["r_out"],
                    "r_topo_init": sc_init["r_topo"],
                    "r_cont_init": sc_init["r_cont"],
                    "r_out_revised": sc_rev["r_out"],
                    "r_topo_revised": sc_rev["r_topo"],
                    "orphan_step": sc_init.get("orphan_step"),
                    "format_ok_revised": format_ok(y_rev),
                    "bucket": bucket,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_written += 1

                if n_written % 20 == 0:
                    print(f"  [{n_written}] prompt={pi}/{len(prompts)} "
                          f"r_out_init={sc_init['r_out']} r_out_rev={sc_rev['r_out']} "
                          f"bucket={bucket}")

    print(f"Wrote {n_written} rollouts to {args.output}")


if __name__ == "__main__":
    main()
