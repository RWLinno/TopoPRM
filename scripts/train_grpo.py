#!/usr/bin/env python3
"""GRPO training with TopoPRM hierarchical reward.

Uses trl.GRPOTrainer + LoRA. Replaces swift rlhf.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_grpo.py \
        [--sft_adapter output/sft_deepseek_r1_7b/final]
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
from src.reward.composite_reward import TopoHierarchicalReward

MODEL_ID = os.getenv(
    "TOPOPRM_BASE_MODEL",
    os.path.expandvars("${MODEL_ROOT}/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
)
DATA_PATH = "data/grpo_ready/train_public.jsonl"
OUTPUT_DIR = "output/grpo_topoprm_deepseek_r1_7b"
MAX_COMPLETION_LEN = 4096
LORA_RANK = 64
LORA_ALPHA = 128


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


reward_fn = TopoHierarchicalReward()


def reward_function(completions: list[str], **kwargs) -> list[float]:
    solution = kwargs.get("solution", [None] * len(completions))
    reference_dag = kwargs.get("reference_dag", [None] * len(completions))
    if isinstance(solution, str):
        solution = [solution] * len(completions)
    if isinstance(reference_dag, str):
        reference_dag = [reference_dag] * len(completions)
    return reward_fn(completions, solution=solution, reference_dag=reference_dag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID, help="Base model path or HF model id")
    parser.add_argument("--sft_adapter", default="", help="Path to SFT LoRA adapter")
    parser.add_argument("--report_to", default="wandb", help="Trainer report target")
    parser.add_argument("--max_steps", type=int, default=200, help="Training max steps")
    parser.add_argument("--num_generations", type=int, default=4, help="Number of generations per prompt")
    parser.add_argument("--max_completion_len", type=int, default=MAX_COMPLETION_LEN, help="Max completion tokens")
    args = parser.parse_args()

    run_name = f"grpo_dr1_7b_topoprm_{datetime.now().strftime('%m%d')}"

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.sft_adapter and Path(args.sft_adapter).is_dir():
        print(f"Loading SFT adapter: {args.sft_adapter}")
        model = PeftModel.from_pretrained(model, args.sft_adapter)
        model = model.merge_and_unload()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules="all-linear",
        lora_dropout=0.05,
    )

    print(f"Loading data: {DATA_PATH}")
    dataset = load_grpo_dataset(DATA_PATH)
    print(f"  {len(dataset)} prompts loaded")

    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
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
        report_to=args.report_to,
        run_name=run_name,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_function,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Starting GRPO training with TopoPRM hierarchical reward...")
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/final")
    print(f"GRPO complete. Saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
