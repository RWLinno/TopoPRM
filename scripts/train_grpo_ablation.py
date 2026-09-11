#!/usr/bin/env python3
"""Shared-stage GRPO comparison for the paper's forward hierarchical reward.

Select outcome_only, outcome_length, or topo_hierarchical. All variants share
the same trainer and default 200-update budget. Saved legacy results are not
regenerated merely by running this entrypoint with new checkpoints or software.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_BASE = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATA_PATH = "data/grpo_ready/train_public_swift.jsonl"
SYSTEM = "Solve the math problem step by step. Put the final answer in \\boxed{}."


def load_dataset(path: str):
    from datasets import Dataset
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            # swift-format record: messages / solution / reference_dag
            if "messages" in d:
                user = next((m["content"] for m in d["messages"] if m["role"] == "user"), "")
                prompt = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
            else:
                prompt = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": d["question"]}]
            records.append({
                "prompt": prompt,
                "solution": str(d.get("solution", d.get("final_answer", ""))),
                "reference_dag": d.get("reference_dag", ""),
            })
    return Dataset.from_list(records)


def build_reward(name: str):
    if name == "outcome_length":
        from src.reward.ablation_rewards import OutcomeLengthReward
        impl = OutcomeLengthReward()
    elif name == "outcome_only":
        from src.reward.ablation_rewards import OutcomeOnlyReward
        impl = OutcomeOnlyReward()
    elif name == "topo_hierarchical":
        from src.reward.composite_reward import TopoHierarchicalReward
        impl = TopoHierarchicalReward()
    else:
        raise ValueError(f"unknown reward {name}")

    def reward_function(completions, **kwargs):
        solution = kwargs.get("solution", [None] * len(completions))
        reference_dag = kwargs.get("reference_dag", [None] * len(completions))
        if isinstance(solution, str):
            solution = [solution] * len(completions)
        if isinstance(reference_dag, str):
            reference_dag = [reference_dag] * len(completions)
        # ORM classes accept solution/reference_dag kwargs; pass through.
        try:
            return impl(completions, solution=solution, reference_dag=reference_dag)
        except TypeError:
            return impl(completions, solution=solution)

    reward_function.__name__ = f"reward_{name}"
    return reward_function


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="outcome_length",
                    choices=["outcome_length", "outcome_only", "topo_hierarchical"])
    ap.add_argument("--model", default=DEFAULT_BASE)
    ap.add_argument("--sft_adapter", required=True)
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--output_dir", default="output/grpo_outcome_length_dr1_7b")
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--max_completion_len", type=int, default=2048)
    ap.add_argument("--report_to", default="none")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Pin the forward hierarchical specification, including optional reference scoring.
    os.environ.update({
        'TOPO_DAG_RAW_DIRECTED': '0',
        'TOPO_DAG_LLM_REFINE': '0',
        'TOPO_DAG_EDGE_CHECKPOINT': '',
        'TOPO_DAG_EDGE_REQUIRED': '0',
        'TOPO_DISABLE_EDGE_ENCODER': '1',
        'TOPO_ABLATION_CONFIG': '',
        'TOPO_DAG_SENTENCE_FALLBACK': '0',
        'TOPO_DAG_EXTRA_STEP_MARKERS': '0',
        'TOPO_DAG_LATEX_EXPR': '0',
        'TOPO_DAG_BARRIER_STRICT': '0',
        'TOPO_DAG_FILTER_FORMATTING': '0',
        'TOPO_DAG_SEQ_WHEN_NO_DEP_ONLY': '0',
        'TOPO_VAR_REF_REQUIRE_MULTI': '0',
        'TOPO_VAR_REF_DISTINCTIVE': '0',
        'TOPO_SEQ_REQUIRE_OVERLAP': '0',
        'TOPO_ORDER_REQUIRE_NUMERIC': '0',
        'TOPO_ENABLE_SEQUENTIAL_WEAK_EDGE': '1',
        'TOPO_SEQ_WEAK_EDGE_MODE': 'adaptive',
        'TOPO_NO_THINK_FALLBACK': '0',
        'TOPO_CONT_REQUIRE_EVIDENCE': '0',
        'TOPO_CONTINUITY_BROKEN_CHAIN_PENALTY': '0.8',
        'TOPO_ORPHAN_LEGACY': '0',
        'TOPO_ORPHAN_W_VIRTUAL': '1',
        'TOPO_ORPHAN_W_DOUBLE_BARRIER': '0.5',
        'TOPO_ORPHAN_W_SOLID': '0.3',
        'TOPO_REQUIRE_VALID_DAG': '1',
        'TOPO_QTOPO_SELF_NORM': '0',
        'TOPO_HIER_ALPHA': '0.60',
        'TOPO_HIER_BASE_FLOOR': '0.05',
        'TOPO_HIER_AGG': 'additive',
        'TOPO_HIER_NOISE_EPS': '0',
        'TOPO_DYNAMIC_REWARD': '0',
        'TOPO_HIER_REWARD_TEMP': '1',
        'TOPO_RESCALE_PATCH': '0',
        'TOPO_LENGTH_UNIT': 'chars',
        'TOPO_LENGTH_LOW': '2000',
        'TOPO_LENGTH_HIGH': '4000',
        'TOPO_LAMBDA_BASE': '0.20',
        'TOPO_LAMBDA_ACYCLIC': '0.15',
        'TOPO_LAMBDA_ORPHAN': '0.15',
        'TOPO_LAMBDA_DELTA': '0.15',
        'TOPO_LAMBDA_KAPPA': '0.25',
    })
    if not Path(args.sft_adapter).is_dir():
        raise FileNotFoundError(f"SFT adapter not found: {args.sft_adapter}")

    import torch
    from peft import LoraConfig, PeftModel, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    run_name = f"grpo_{args.reward}_{datetime.now().strftime('%m%d_%H%M')}"
    print(f"[train] reward={args.reward} model={args.model} steps={args.max_steps}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True
    )
    if args.sft_adapter and Path(args.sft_adapter).is_dir():
        print(f"[train] merging SFT adapter: {args.sft_adapter}")
        model = PeftModel.from_pretrained(model, args.sft_adapter)
        model = model.merge_and_unload()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128,
        target_modules="all-linear", lora_dropout=0.05,
    )

    dataset = load_dataset(args.data)
    print(f"[train] {len(dataset)} prompts")

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        max_completion_length=args.max_completion_len,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        beta=0.04,
        temperature=0.8,
        top_p=0.95,
        logging_steps=5,
        save_steps=50,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=False,
        seed=args.seed,
        report_to=args.report_to,
        run_name=run_name,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        reward_funcs=build_reward(args.reward),
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    print("[train] starting GRPO ...")
    trainer.train()
    trainer.save_model(f"{args.output_dir}/final")
    print(f"[train] done -> {args.output_dir}/final")


if __name__ == "__main__":
    main()
