#!/usr/bin/env python3
"""SFT training for DeepSeek-R1-Distill-Qwen-7B on public DAG math data.

Uses trl.SFTTrainer + LoRA (peft). Replaces swift sft.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_sft.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig

MODEL_ID = os.getenv(
    "TOPOPRM_BASE_MODEL",
    "${HF_MODELS_DIR:-./models}/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
)
DATA_PATH = "data/grpo_ready/train_public.jsonl"
OUTPUT_DIR = "output/sft_deepseek_r1_7b"
MAX_SEQ_LEN = 4096
LORA_RANK = 64
LORA_ALPHA = 128


def load_sft_dataset(path: str) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            question = d["question"]
            answer = d.get("standard_answer", d.get("final_answer", ""))
            text = (
                f"<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n{answer}\n</think>\n"
                f"<answer>{d.get('final_answer', '')}</answer><|im_end|>"
            )
            records.append({"text": text})
    return Dataset.from_list(records)


def main():
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules="all-linear",
        lora_dropout=0.05,
    )

    print(f"Loading data: {DATA_PATH}")
    dataset = load_sft_dataset(DATA_PATH)
    print(f"  {len(dataset)} samples loaded")

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=MAX_SEQ_LEN,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        warmup_steps=50,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Starting SFT training...")
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/final")
    print(f"SFT complete. Saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
