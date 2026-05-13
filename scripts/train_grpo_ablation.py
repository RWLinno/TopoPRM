#!/usr/bin/env python3
"""GRPO ablation training — same as train_grpo.py but with selectable reward function.

Usage:
    CUDA_VISIBLE_DEVICES=4 python3 scripts/train_grpo_ablation.py \
        --reward outcome_only --output_dir output/grpo_outcome_only_dr1_7b
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.reward.composite_reward import (
    OutcomeOnlyReward, NoTopoReward, NoContinuityReward, TopoHierarchicalReward,
)
from src.training.accuracy_callback import (
    EvalAccuracyCallback,
    build_eval_subset,
)

MODEL_ID = os.getenv(
    "TOPOPRM_BASE_MODEL",
    "/Knowin/foundation/weilinruan/hf_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
)
DATA_PATH = "data/grpo_ready/train_public.jsonl"

REWARD_MAP = {
    "outcome_only": OutcomeOnlyReward,
    "no_topo": NoTopoReward,
    "no_continuity": NoContinuityReward,
    "hierarchical": TopoHierarchicalReward,
}


def load_grpo_dataset(path: str) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            records.append({
                "prompt": d["question"],
                "solution": d.get("final_answer", ""),
                "reference_dag": d.get("reference_dag", ""),
            })
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--sft_adapter", default="output/sft_deepseek_r1_7b/final")
    parser.add_argument("--reward", required=True, choices=list(REWARD_MAP.keys()))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_len", type=int, default=4096)
    parser.add_argument(
        "--eval_every",
        type=int,
        default=20,
        help="Run lightweight accuracy eval every N gradient steps; set 0 to disable.",
    )
    parser.add_argument("--eval_size", type=int, default=32,
                        help="Number of held-out prompts used by the in-training eval.")
    parser.add_argument("--eval_max_new_tokens", type=int, default=512,
                        help="Per-sample generation cap during in-training eval.")
    parser.add_argument("--eval_jsonl", default=DATA_PATH,
                        help="JSONL with question / final_answer used to build the eval subset.")
    args = parser.parse_args()

    reward_cls = REWARD_MAP[args.reward]
    reward_fn = reward_cls()
    run_name = f"grpo_dr1_7b_{args.reward}_{datetime.now().strftime('%m%d')}"

    print(f"Reward: {args.reward} ({reward_cls.__name__})")
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    if args.sft_adapter and Path(args.sft_adapter).is_dir():
        print(f"Loading SFT adapter: {args.sft_adapter}")
        model = PeftModel.from_pretrained(model, args.sft_adapter)
        model = model.merge_and_unload()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128,
        target_modules="all-linear", lora_dropout=0.05,
    )

    dataset = load_grpo_dataset(DATA_PATH)
    print(f"  {len(dataset)} prompts loaded")

    def reward_function(completions: list[str], **kwargs) -> list[float]:
        solution = kwargs.get("solution", [None] * len(completions))
        reference_dag = kwargs.get("reference_dag", [None] * len(completions))
        if isinstance(solution, str):
            solution = [solution] * len(completions)
        if isinstance(reference_dag, str):
            reference_dag = [reference_dag] * len(completions)
        return reward_fn(completions, solution=solution, reference_dag=reference_dag)

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        max_completion_length=args.max_completion_len,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        beta=0.04,
        logging_steps=5,
        save_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name=run_name,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model, args=grpo_config, train_dataset=dataset,
        reward_funcs=reward_function, peft_config=lora_config,
        processing_class=tokenizer,
    )

    if args.eval_every and args.eval_size > 0:
        eval_subset = build_eval_subset(
            Path(args.eval_jsonl),
            size=args.eval_size,
            sources=("gsm8k", "math"),
            seed=13,
        )
        if eval_subset:
            print(
                f"Eval callback: {len(eval_subset)} held-out prompts, "
                f"every {args.eval_every} steps, "
                f"max_new_tokens={args.eval_max_new_tokens}"
            )
            trainer.add_callback(
                EvalAccuracyCallback(
                    tokenizer=tokenizer,
                    eval_examples=eval_subset,
                    eval_every=args.eval_every,
                    max_new_tokens=args.eval_max_new_tokens,
                )
            )
        else:
            print("Eval callback skipped: subset builder returned 0 rows.")
    else:
        print("Eval callback disabled (--eval_every=0 or --eval_size=0).")

    print(f"Starting GRPO ablation ({args.reward})...")
    trainer.train()
    trainer.save_model(f"{args.output_dir}/final")
    print(f"Done. Saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
