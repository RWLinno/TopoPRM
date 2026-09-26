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
        'TOPO_FORMAT_PROTOCOL': 'legacy',
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
                prompt = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": d.get("question", "")}]
            problem = prompt[-1]["content"]
            answer = d.get("solution", d.get("final_answer"))
            if not isinstance(problem, str) or not problem.strip():
                raise ValueError("Every GRPO prompt must contain a nonempty problem")
            if answer is None or not str(answer).strip():
                raise ValueError("Every GRPO prompt must contain a nonempty reference answer")
            records.append({
                "prompt": prompt,
                "solution": str(answer),
                "reference_dag": d.get("reference_dag", ""),
            })
    return Dataset.from_list(records)


def build_reward(name: str, *, collect_ace: bool = False, tokenizer=None):
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
        class CapturedHierarchicalReward(TopoHierarchicalReward):
            def _components(self, completions, solution=None, reference_dag=None, **kwargs):
                # The paper path must not turn a failed scorer into a valid
                # zero: in particular, zero outcome assigns the wrong ACE stratum.
                self.last_components = None
                raw = (
                    ("outcome", self._outcome(completions, solution=solution, **kwargs)),
                    ("format", self._format(completions, **kwargs)),
                    ("topology", self._topo(completions, reference_dag=reference_dag, **kwargs)),
                    ("continuity", self._continuity(completions, **kwargs)),
                    ("length", self._length(completions, **kwargs)),
                )
                checked = []
                for component, values in raw:
                    if not isinstance(values, (list, tuple)) or len(values) != len(completions):
                        raise ValueError(f"Paper {component} scores must have one value per completion")
                    scores = []
                    for value in values:
                        try:
                            score = float(value)
                        except (TypeError, ValueError, OverflowError) as exc:
                            raise ValueError(f"Paper {component} scorer returned a nonnumeric value") from exc
                        if not math.isfinite(score) or not 0 <= score <= 1:
                            raise ValueError(f"Paper {component} scores must be finite and in [0, 1]")
                        if component == "outcome" and score not in (0, 1):
                            raise ValueError("Paper outcome scores must be binary")
                        scores.append(score)
                    checked.append(scores)
                self.last_components = tuple(checked)
                return self.last_components

        impl = CapturedHierarchicalReward()
    else:
        raise ValueError(f"unknown reward {name}")

    # Use the same explicit-final-answer verifier as evaluation. The legacy
    # research backend also accepts intermediate numeric statements; those
    # must not determine the paper path's correctness strata.
    from src.eval.math_scoring import verify_math_response
    from src.reward.utils import completion_to_text

    def paper_outcome(completions, solution=None, **kwargs):
        solutions = solution if isinstance(solution, list) else [solution] * len(completions)
        if len(solutions) != len(completions):
            raise ValueError("Paper outcome scoring requires one answer per completion")
        if any(value is None or not str(value).strip() for value in solutions):
            raise ValueError("Paper outcome scoring requires nonempty reference answers")
        return [float(verify_math_response(completion_to_text(completion), str(answer)))
                for completion, answer in zip(completions, solutions, strict=True)]

    impl._outcome = paper_outcome

    def reward_function(completions, **kwargs):
        reward_function.ace_batch = None
        if tokenizer is not None:
            from src.reward.utils import restore_response_prefix
            ids = kwargs.get("completion_ids")
            prompts = kwargs.get("prompts")
            if ids is None or prompts is None or len(prompts) != len(completions):
                raise ValueError("Paper reward requires aligned prompts and generated token IDs")
            texts = tokenizer.batch_decode(ids, skip_special_tokens=True)
            completions = [restore_response_prefix(
                tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                if isinstance(prompt, list) else prompt, text
            ) for prompt, text in zip(prompts, texts, strict=True)]
        solution = kwargs.get("solution", [None] * len(completions))
        reference_dag = None  # Rollout and reference step indices are not aligned.
        if isinstance(solution, str):
            solution = [solution] * len(completions)
        if isinstance(reference_dag, str):
            reference_dag = [reference_dag] * len(completions)
        if name == "topo_hierarchical":
            # Raw components must be gathered before rescaling: a prompt group
            # can cross a device boundary, and a call can contain many groups.
            result = impl(completions, solution=solution, reference_dag=None)
            reward_function.latest_texts = [completion_to_text(value) for value in completions]
            completion_ids = kwargs.get("completion_ids")
            reward_function.ace_batch = (
                tuple(tuple(ids) for ids in completion_ids) if completion_ids is not None else None,
                list(zip(*impl.last_components, strict=True)),
            )
            return result  # The trainer replaces local-call scores with group scores.
        # ORM classes accept solution/reference_dag kwargs; pass through.
        try:
            return impl(completions, solution=solution, reference_dag=reference_dag)
        except TypeError:
            return impl(completions, solution=solution)

    reward_function.__name__ = f"reward_{name}"
    reward_function.ace_batch = None
    reward_function.collect_ace = collect_ace
    reward_function.collect_components = name == "topo_hierarchical"
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


def group_hierarchical_rewards(components, num_generations, structural_ablation="none"):
    """Eq. (3), with min–max scaling over each complete global prompt group."""
    import torch

    if components.ndim != 2 or components.shape[1] != 5:
        raise ValueError("Expected [rollouts, outcome/format/topology/continuity/length]")
    if num_generations < 2 or components.shape[0] % num_generations:
        raise ValueError("Hierarchical reward requires complete groups of at least two")
    if not torch.isfinite(components).all() or not ((components >= 0) & (components <= 1)).all():
        raise ValueError("Reward components must be finite and in [0, 1]")
    if not ((components[:, 0] == 0) | (components[:, 0] == 1)).all():
        raise ValueError("Outcome must be binary")
    structural = components[:, 2:4].reshape(-1, num_generations, 2)
    lo = structural.amin(dim=1, keepdim=True)
    span = structural.amax(dim=1, keepdim=True) - lo
    scaled = torch.where(span > 0, (structural - lo) / span.clamp_min(1e-8), 0.5)
    scaled = scaled.reshape(-1, 2)
    if structural_ablation not in {"none", "topology", "continuity", "both"}:
        raise ValueError("Unknown matched structural ablation")
    # Neutralize information at the constant-component value, keeping mixture
    # weights, base reward, ACE, rollout budget and training stage unchanged.
    if structural_ablation in {"topology", "both"}:
        scaled[:, 0] = 0.5
    if structural_ablation in {"continuity", "both"}:
        scaled[:, 1] = 0.5
    auxiliary = 0.6 * scaled[:, 0] + 0.4 * scaled[:, 1]
    base = 0.7 * components[:, 0] + 0.15 * components[:, 1] + 0.15 * components[:, 4]
    rewards = (base.clamp_min(0.05) * (1 + auxiliary)).clamp(0, 1)
    return rewards, auxiliary


def build_ace_trainer_class(clip_lower=-1.0, clip_upper=1.0, *, use_ace=True,
                            structural_ablation="none"):
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
            if len(self.reward_funcs) != 1 or not getattr(self.reward_funcs[0], "collect_components", False):
                raise ValueError("Paper reward requires one callable capturing raw components")
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
            ids, components = captured
            if ids != tuple(tuple(tokens) for tokens in completion_ids_list):
                raise RuntimeError("ACE metadata does not match the current completion order")
            if len(components) != len(prompts):
                raise RuntimeError("ACE metadata has an incorrect local batch size")
            local = torch.tensor(components,
                                 dtype=torch.float32, device=self.accelerator.device)
            # Same gather and rank-major order as TRL's rewards. In particular,
            # do not center strata locally: one prompt group may span devices.
            global_batch = gather(local)
            if global_batch.shape != (rewards.shape[0], 5):
                raise RuntimeError("ACE metadata and gathered rewards are misaligned")
            mode = "train" if self.model.training else "eval"
            group_size = self.num_generations if mode == "train" else self.num_generations_eval
            group_rewards, auxiliary = group_hierarchical_rewards(
                global_batch, group_size, structural_ablation)
            rewards[:, 0] = group_rewards
            self._ace_global_batch = torch.stack((global_batch[:, 0], auxiliary), dim=1)
            start = self.accelerator.process_index * len(prompts)
            self._audit_pending = [{
                "prompt": prompts[i], "solution": inputs[i].get("solution"),
                "response": self.reward_funcs[0].latest_texts[i],
                "tokens": len(completion_ids_list[i]),
                "components": dict(zip(("outcome", "format", "topology", "continuity", "length"), row)),
                "reward": group_rewards[start + i].item(),
                "auxiliary": auxiliary[start + i].item(),
            } for i, row in enumerate(components)]
            self.reward_funcs[0].ace_batch = None
            return rewards

        def _generate_and_score_completions(self, inputs):
            self._ace_global_batch = None
            output = super()._generate_and_score_completions(inputs)
            standard = output["advantages"].detach().clone()
            if not use_ace:
                self._write_rollout_audit(standard, standard)
                self._ace_global_batch = None
                return output
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
            self._write_rollout_audit(standard, output["advantages"])
            # Replace the base trainer's just-recorded z values so displayed
            # advantages agree with the coefficients actually used by its loss.
            logged = self._logs["advantages"]
            for _ in range(min(len(logged), coefficients.numel())):
                logged.pop()
            logged.extend(coefficients.tolist())
            self._ace_global_batch = None
            return output

        def _write_rollout_audit(self, standard, advantages):
            path = Path(self.args.output_dir) / f"rollouts_rank{self.accelerator.process_index}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as out:
                for row, z, advantage in zip(self._audit_pending, standard.tolist(), advantages.tolist(), strict=True):
                    out.write(json.dumps({"step": self.state.global_step,
                        "structural_ablation": structural_ablation,
                        "standardized_return": z, "advantage": advantage, **row},
                        ensure_ascii=False) + "\n")
            self._audit_pending = None

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
    ap.add_argument("--structural_ablation", choices=["none", "topology", "continuity", "both"],
                    default="none", help="Matched information removal; neutralize a channel at 0.5")
    ap.add_argument("--generation_batch_size", type=int, default=0,
                    help="Bound rollout memory independently of optimizer accumulation")
    ap.add_argument("--clip_coeff_lower", type=float, default=-1.0)
    ap.add_argument("--clip_coeff_upper", type=float, default=1.0)
    ap.add_argument("--max_prompt_length", type=int, default=4096)
    ap.add_argument("--report_to", default="none")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.advantage_mode == "ace" and args.reward != "topo_hierarchical":
        ap.error("--advantage_mode ace requires --reward topo_hierarchical")
    if args.structural_ablation != "none" and args.reward != "topo_hierarchical":
        ap.error("Matched structural ablation requires the hierarchical reward")
    if args.generation_batch_size and (args.generation_batch_size < args.num_generations
                                      or args.generation_batch_size % args.num_generations):
        ap.error("generation_batch_size must contain complete prompt groups")
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

    if (Path(args.output_dir) / "final").exists():
        raise FileExistsError("Choose a fresh output directory; its final checkpoint already exists")
    configure_paper_reward()
    if not Path(args.sft_adapter).is_dir():
        raise FileNotFoundError(f"SFT adapter not found: {args.sft_adapter}")

    import torch
    from peft import LoraConfig, TaskType
    from src.training import load_and_merge_adapter, save_merged_checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    trainer_class = (build_ace_trainer_class(
        args.clip_coeff_lower, args.clip_coeff_upper, use_ace=args.advantage_mode == "ace",
        structural_ablation=args.structural_ablation,
    ) if args.reward == "topo_hierarchical" else GRPOTrainer)
    run_name = f"{args.advantage_mode}_{args.reward}_{datetime.now().strftime('%m%d_%H%M')}"
    print(f"[train] reward={args.reward} advantages={args.advantage_mode} "
          f"model={args.model} steps={args.max_steps} epochs={args.num_train_epochs}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    local_device = None
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        local_device = {"": torch.cuda.current_device()}
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=local_device, trust_remote_code=True
    )
    if args.sft_adapter and Path(args.sft_adapter).is_dir():
        print(f"[train] merging SFT adapter: {args.sft_adapter}")
        model = load_and_merge_adapter(model, args.sft_adapter)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128,
        target_modules="all-linear", lora_dropout=0.05,
    )

    dataset = load_dataset(args.data)
    # TRL 0.28 no longer truncates prompts. Keep complete problems within the
    # prompt budget, so generation respects the declared total sequence cap.
    count_before = len(dataset)
    dataset = dataset.filter(lambda row: len(tokenizer.apply_chat_template(
        row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False
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
        generation_batch_size=args.generation_batch_size or None,
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
        loss_type="grpo",
        scale_rewards="group",
        multi_objective_aggregation="sum_then_normalize",
    )

    trainer = trainer_class(
        model=model,
        args=cfg,
        train_dataset=dataset,
        reward_funcs=build_reward(args.reward, collect_ace=args.advantage_mode == "ace", tokenizer=tokenizer),
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    if trainer.accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "run_arguments.json").write_text(
            json.dumps(vars(args), indent=2), encoding="utf-8")
    print("[train] starting GRPO ...")
    trainer.train()
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        save_merged_checkpoint(
            trainer.accelerator.unwrap_model(trainer.model), tokenizer, f"{args.output_dir}/final"
        )
    trainer.accelerator.wait_for_everyone()
    print(f"[train] done -> {args.output_dir}/final")


if __name__ == "__main__":
    main()
