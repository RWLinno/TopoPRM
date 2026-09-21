#!/usr/bin/env python3
"""Shared-stage GRPO comparison for the paper's forward hierarchical reward.

The paper entrypoint uses a forward rule extractor and hierarchical reward.
Full uses ACE; source-removal and ordinary-GRPO controls are explicit choices.
No stored training configuration is loaded by this entrypoint.
Optional ACE changes the policy-gradient coefficients after GRPO standardizes
returns; it is not a scalar reward proxy or a claim about historical runs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_BASE = "Qwen/Qwen3.5-9B"
DATA_PATH = "data/grpo_ready/train_public_swift.jsonl"
SYSTEM = "Solve the math problem step by step in <think>...</think>, then give the final answer in <answer>\\boxed{...}</answer>."


def configure_paper_reward() -> None:
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


def build_reward(name: str, *, collect_ace: bool = False):
    if collect_ace and name != "topo_hierarchical":
        raise ValueError("ACE requires the hierarchical topology/continuity reward")
    if name == "outcome_length":
        from src.reward.ablation_rewards import OutcomeLengthReward
        impl = OutcomeLengthReward()
    elif name == "outcome_only":
        from src.reward.ablation_rewards import OutcomeOnlyReward
        impl = OutcomeOnlyReward()
    elif name in {"no_topology", "no_continuity"}:
        from src.reward.ablation_rewards import NoTopoReward, NoContinuityReward
        impl = NoTopoReward() if name == "no_topology" else NoContinuityReward()
    elif name == "topo_hierarchical":
        from src.reward.composite_reward import TopoHierarchicalReward
        if collect_ace:
            class CapturedHierarchicalReward(TopoHierarchicalReward):
                def _components(self, *args, **kwargs):
                    self.last_components = super()._components(*args, **kwargs)
                    return self.last_components

            impl = CapturedHierarchicalReward()
        else:
            impl = TopoHierarchicalReward()
    else:
        raise ValueError(f"unknown reward {name}")

    def reward_function(completions, **kwargs):
        reward_function.ace_batch = None
        solution = kwargs.get("solution", [None] * len(completions))
        reference_dag = None  # Rollout and reference step indices are not aligned.
        if isinstance(solution, str):
            solution = [solution] * len(completions)
        if isinstance(reference_dag, str):
            reference_dag = [reference_dag] * len(completions)
        if collect_ace:
            # The paper's generated and reference steps are not aligned.
            # Capture components from exactly the call producing these returns.
            result = impl(completions, solution=solution, reference_dag=None)
            outcome, _, topology, continuity, _ = impl.last_components
            alpha = impl._clip01(impl.ALPHA)
            auxiliary = [
                alpha * topo + (1.0 - alpha) * cont
                for topo, cont in zip(
                    impl._batch_rescale(topology), impl._batch_rescale(continuity), strict=True
                )
            ]
            completion_ids = kwargs.get("completion_ids")
            reward_function.ace_batch = (
                tuple(tuple(ids) for ids in completion_ids) if completion_ids is not None else None,
                list(outcome), auxiliary,
            )
            return result
        # ORM classes accept solution/reference_dag kwargs; pass through.
        try:
            return impl(completions, solution=solution, reference_dag=reference_dag)
        except TypeError:
            return impl(completions, solution=solution)

    reward_function.__name__ = f"reward_{name}"
    reward_function.ace_batch = None
    reward_function.collect_ace = collect_ace
    return reward_function


def ace_advantages(standardized_returns, correctness, auxiliary, num_generations,
                   clip_lower=-1.0, clip_upper=1.0):
    """Apply Eq. ACE to globally ordered prompt groups without recentering.

    ``standardized_returns`` is TRL's group-standardized return, including its
    1e-4 denominator stabilizer. All three tensors have one entry per rollout.
    Missing strata and singleton strata produce zero centered auxiliary credit.
    """
    import torch

    if not (math.isfinite(clip_lower) and math.isfinite(clip_upper)
            and clip_lower <= 0 <= clip_upper):
        raise ValueError("ACE bounds must be finite and satisfy lower <= 0 <= upper")
    if num_generations < 1 or standardized_returns.numel() % num_generations:
        raise ValueError("ACE requires complete prompt groups")
    if any(tensor.ndim != 1 or tensor.shape != standardized_returns.shape
           for tensor in (standardized_returns, correctness, auxiliary)):
        raise ValueError("ACE requires aligned one-dimensional rollout tensors")
    if not all(torch.isfinite(tensor).all() for tensor in (standardized_returns, correctness, auxiliary)):
        raise ValueError("ACE inputs must be finite")
    if not ((correctness == 0) | (correctness == 1)).all():
        raise ValueError("ACE requires binary final-answer correctness")
    if not ((auxiliary >= 0) & (auxiliary <= 1)).all():
        raise ValueError("ACE auxiliary scores must be in [0, 1]")

    z = standardized_returns.reshape(-1, num_generations)
    correct = correctness.reshape_as(z).bool()
    u = auxiliary.reshape_as(z)
    delta = torch.zeros_like(u)
    for mask in (correct, ~correct):
        stratum_mean = (u * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        delta = torch.where(mask, u - stratum_mean, delta)
    adjusted = z + torch.where(correct, delta.clamp_min(0), delta.clamp_max(0))
    return torch.where(
        correct, adjusted.clamp(0, clip_upper), adjusted.clamp(clip_lower, 0)
    ).reshape(-1)


def build_ace_trainer_class(clip_lower=-1.0, clip_upper=1.0):
    """Subclass the verified TRL 0.28.0 generation/scoring boundary.

    Official source: https://github.com/huggingface/trl/blob/v0.28.0/trl/trainer/grpo_trainer.py
    Its loss consumes output['advantages'] directly. Subsequent rollout-buffer
    shuffling/splitting applies the same permutation to every output tensor.
    """
    from importlib.metadata import version
    if version("trl") != "0.28.0":
        raise RuntimeError("The ACE trainer boundary is validated only for trl==0.28.0")
    import torch
    from accelerate.utils import gather
    from trl import GRPOTrainer

    class ACEGRPOTrainer(GRPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.scale_rewards != "group" or self.multi_objective_aggregation != "sum_then_normalize":
                raise ValueError("ACE requires group-scaled, sum-then-normalize GRPO returns")
            if len(self.reward_funcs) != 1 or not getattr(self.reward_funcs[0], "collect_ace", False):
                raise ValueError("ACE requires one reward callable built with collect_ace=True")
            if self.args.use_liger_kernel:
                raise ValueError("The ACE boundary is validated with the standard GRPO loss")
            self._ace_global_batch = None

        def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
            self._ace_global_batch = None
            self.reward_funcs[0].ace_batch = None
            rewards = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
            captured = self.reward_funcs[0].ace_batch
            if captured is None:
                raise RuntimeError("The current reward call did not capture ACE metadata")
            ids, correct, auxiliary = captured
            if ids != tuple(tuple(tokens) for tokens in completion_ids_list):
                raise RuntimeError("ACE metadata does not match the current completion order")
            if len(correct) != len(prompts) or len(auxiliary) != len(prompts):
                raise RuntimeError("ACE metadata has an incorrect local batch size")
            local = torch.tensor(list(zip(correct, auxiliary, strict=True)),
                                 dtype=torch.float32, device=self.accelerator.device)
            # Same gather and rank-major order as TRL's rewards. In particular,
            # do not center strata locally: one prompt group may span devices.
            global_batch = gather(local)
            if global_batch.shape != (rewards.shape[0], 2):
                raise RuntimeError("ACE metadata and gathered rewards are misaligned")
            self._ace_global_batch = global_batch
            self.reward_funcs[0].ace_batch = None
            return rewards

        def _generate_and_score_completions(self, inputs):
            self._ace_global_batch = None
            output = super()._generate_and_score_completions(inputs)
            metadata = self._ace_global_batch
            if metadata is None:
                raise RuntimeError("ACE metadata is missing for the generated batch")
            local_z = output["advantages"]
            global_z = gather(local_z)
            if metadata.shape[0] != global_z.numel():
                raise RuntimeError("ACE coefficients and metadata have different batch sizes")
            mode = "train" if self.model.training else "eval"
            group_size = self.num_generations if mode == "train" else self.num_generations_eval
            coefficients = ace_advantages(
                global_z, metadata[:, 0], metadata[:, 1], group_size,
                clip_lower=clip_lower, clip_upper=clip_upper,
            )
            start = self.accelerator.process_index * local_z.numel()
            output["advantages"] = coefficients[start:start + local_z.numel()].detach()
            # Replace the base trainer's just-recorded z values so displayed
            # advantages agree with the coefficients actually used by its loss.
            logged = self._logs["advantages"]
            for _ in range(min(len(logged), coefficients.numel())):
                logged.pop()
            logged.extend(coefficients.tolist())
            self._ace_global_batch = None
            return output

    return ACEGRPOTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="topo_hierarchical",
                    choices=["outcome_length", "outcome_only", "topo_hierarchical", "no_topology", "no_continuity"])
    ap.add_argument("--model", default=DEFAULT_BASE)
    ap.add_argument("--sft_adapter", required=True)
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--output_dir", default="output/topoprm_stage2")
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--num_train_epochs", type=float, default=1.0,
                    help="Used when --max_steps=-1; positive max_steps takes precedence")
    ap.add_argument("--num_generations", type=int, default=2)
    ap.add_argument("--max_completion_len", type=int, default=4096)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1,
                    help="Per-device optimizer microbatch")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--advantage_mode", choices=["grpo", "ace"], default="ace")
    ap.add_argument("--clip_coeff_lower", type=float, default=-1.0)
    ap.add_argument("--clip_coeff_upper", type=float, default=1.0)
    ap.add_argument("--max_prompt_length", type=int, default=4096)
    ap.add_argument("--report_to", default="none")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.advantage_mode == "ace" and args.reward != "topo_hierarchical":
        ap.error("--advantage_mode ace requires --reward topo_hierarchical")
    if not (math.isfinite(args.clip_coeff_lower) and math.isfinite(args.clip_coeff_upper)
            and args.clip_coeff_lower <= 0 <= args.clip_coeff_upper):
        ap.error("ACE clip bounds must be finite and satisfy lower <= 0 <= upper")
    if args.max_prompt_length < 1 or args.max_completion_len < 1:
        ap.error("Prompt and completion budgets must be positive")
    if args.num_generations < 2 or args.gradient_accumulation_steps < 1:
        ap.error("num_generations must be >=2 and gradient_accumulation_steps >=1")
    if args.per_device_train_batch_size is not None and args.per_device_train_batch_size < 1:
        ap.error("per_device_train_batch_size must be >=1")
    if args.num_train_epochs <= 0 or args.max_steps == 0 or args.max_steps < -1:
        ap.error("Use positive epochs and either positive max_steps or --max_steps=-1")

    configure_paper_reward()
    if not Path(args.sft_adapter).is_dir():
        raise FileNotFoundError(f"SFT adapter not found: {args.sft_adapter}")

    import torch
    from peft import LoraConfig, PeftModel, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    trainer_class = (build_ace_trainer_class(args.clip_coeff_lower, args.clip_coeff_upper)
                     if args.advantage_mode == "ace" else GRPOTrainer)
    run_name = f"{args.advantage_mode}_{args.reward}_{datetime.now().strftime('%m%d_%H%M')}"
    print(f"[train] reward={args.reward} advantages={args.advantage_mode} "
          f"model={args.model} steps={args.max_steps} epochs={args.num_train_epochs}")

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
    # TRL 0.28 no longer truncates prompts. Keep complete problems within the
    # prompt budget, so generation respects the declared total sequence cap.
    count_before = len(dataset)
    dataset = dataset.filter(lambda row: len(tokenizer.apply_chat_template(
        row["prompt"], tokenize=True, add_generation_prompt=True
    )) <= args.max_prompt_length)
    if not len(dataset):
        raise ValueError("No complete prompts fit --max_prompt_length")
    print(f"[train] {len(dataset)} prompts; excluded {count_before - len(dataset)} over budget")

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        max_completion_length=args.max_completion_len,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size or args.num_generations,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
        scale_rewards="group",
        multi_objective_aggregation="sum_then_normalize",
    )

    trainer = trainer_class(
        model=model,
        args=cfg,
        train_dataset=dataset,
        reward_funcs=build_reward(args.reward, collect_ace=args.advantage_mode == "ace"),
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    print("[train] starting GRPO ...")
    trainer.train()
    trainer.save_model(f"{args.output_dir}/final")
    print(f"[train] done -> {args.output_dir}/final")


if __name__ == "__main__":
    main()
