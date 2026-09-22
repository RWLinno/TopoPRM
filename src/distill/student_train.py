"""Online TGD: on-policy initial traces and teacher-revised distillation prefixes.

Sampling and acceptance are stop-gradient operations. Eq. (5) sums reverse KL
only over revised response tokens, with different teacher and target contexts.
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.distill.reverse_kl_loss import reverse_kl_loss


@dataclass
class TrainConfig:
    student_model: str
    teacher_model: str
    prompts: Path
    output_dir: Path
    student_adapter: str = ""
    teacher_adapter: str = ""
    student_device: str = "cuda:0"
    teacher_device: str = "cuda:1"
    num_train_epochs: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.05
    token_budget: int = 1024
    max_length: int = 8192
    max_prompts: int = 0
    student_temperature: float = 0.7
    teacher_temperature: float = 0.0
    top_p: float = 0.95
    seed: int = 0
    gradient_checkpointing: bool = True

    def validate(self):
        if min(self.num_train_epochs, self.gradient_accumulation_steps, self.token_budget) < 1:
            raise ValueError("Epochs, accumulation and response budget must be positive")
        if self.max_length <= self.token_budget or self.max_prompts < 0:
            raise ValueError("Context cap must exceed response budget; max_prompts must be nonnegative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if not 0 <= self.warmup_ratio <= 1 or not 0 < self.top_p <= 1:
            raise ValueError("Invalid warmup_ratio or top_p")
        if any(not math.isfinite(t) or t < 0 for t in (self.student_temperature, self.teacher_temperature)):
            raise ValueError("Sampling temperatures must be finite and nonnegative")


def validate_matching_models(student, teacher, tokenizer, teacher_tokenizer):
    """Check merged checkpoints before adding the target's trainable adapter."""
    if type(student) is not type(teacher):
        raise ValueError("TGD requires matching teacher and target architectures")
    shapes = lambda model: {n: tuple(p.shape) for n, p in model.named_parameters()}
    if shapes(student) != shapes(teacher):
        raise ValueError("TGD requires matching parameter shapes and parameter counts")
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("TGD requires identical token-to-ID vocabularies")
    if tokenizer.chat_template != teacher_tokenizer.chat_template:
        raise ValueError("TGD requires the same chat serialization for teacher and target")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(tokenizer, key) != getattr(teacher_tokenizer, key):
            raise ValueError(f"TGD requires matching {key}")


def trace_text(tokenizer, prefix, response):
    from src.reward.utils import restore_response_prefix
    return restore_response_prefix(
        tokenizer.decode(prefix, skip_special_tokens=False),
        tokenizer.decode(response, skip_special_tokens=True),
    )


def revision_messages(messages, initial_trace, instruction):
    # Some model templates strip reasoning from historical assistant messages.
    # Keep the complete candidate in user content so the teacher sees y*.
    return messages + [{"role": "user", "content":
                        "Original solution to revise:\n" + initial_trace +
                        "\n\nRevision instruction:\n" + instruction}]


def prefix_ids(tokenizer, messages):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    if not ids:
        raise ValueError("The chat template produced an empty prefix")
    return list(ids)


@torch.no_grad()
def sample_tokens(model, tokenizer, prefix, cfg, temperature):
    """Generate from the complete prefix; never silently truncate the problem."""
    if len(prefix) + cfg.token_budget > cfg.max_length:
        return None
    ids = torch.tensor([prefix], dtype=torch.long, device=model.device)
    kwargs = dict(max_new_tokens=cfg.token_budget, do_sample=temperature > 0,
                  pad_token_id=tokenizer.pad_token_id, use_cache=True)
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=cfg.top_p)
    model.eval()
    response = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **kwargs)
    return response[0, len(prefix):].detach().cpu().tolist()


def response_logits(model, prefix, response):
    """Align p/q at the same response tokens, despite unequal prefix lengths."""
    import inspect

    if not prefix or not response:
        raise ValueError("A nonempty prefix and response are required")
    # The final sampled token needs a prediction but no subsequent context.
    ids = torch.tensor([prefix + response[:-1]], dtype=torch.long, device=model.device)
    kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if "logits_to_keep" in inspect.signature(base.forward).parameters:
        kwargs["logits_to_keep"] = len(response)
    logits = model(**kwargs).logits
    return logits[:, -len(response):, :]


def revision_loss(student, teacher, student_prefix, teacher_prefix, response):
    with torch.no_grad():
        q = response_logits(teacher, teacher_prefix, response)
    p = response_logits(student, student_prefix, response)
    return reverse_kl_loss(p, q.to(p.device), reduction="sequence_sum")


def run_online_reverse_kl(cfg: TrainConfig) -> int:
    cfg.validate()
    # This import pins the forward extractor before importing its reward classes.
    from scripts.rollout_srt import _load_model, load_prompts, score_trace
    from src.distill.build_srt_data import (
        SYSTEM_PROMPT, DistillationRecord, build_revision_instruction,
        format_ok, revision_rejection_reason,
    )

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    student, tokenizer = _load_model(cfg.student_model, cfg.student_adapter, cfg.student_device)
    teacher, teacher_tokenizer = _load_model(cfg.teacher_model, cfg.teacher_adapter, cfg.teacher_device)
    validate_matching_models(student, teacher, tokenizer, teacher_tokenizer)
    for model in (student, teacher):
        context_cap = getattr(model.config, "max_position_embeddings", cfg.max_length)
        if cfg.max_length > context_cap:
            raise ValueError(f"max_length={cfg.max_length} exceeds checkpoint context cap {context_cap}")
    teacher.eval()
    teacher.requires_grad_(False)
    if cfg.gradient_checkpointing:
        student.gradient_checkpointing_enable()
        student.enable_input_require_grads()
    student = get_peft_model(student, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128,
        target_modules="all-linear", lora_dropout=0.0, bias="none",
    ))
    # Generation and KL use the same conditional distributions. Keep train()
    # for gradient checkpointing, with dropout disabled during optimization.
    for module in student.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    prompts = load_prompts(cfg.prompts, cfg.max_prompts)
    if not prompts:
        raise ValueError("No nonempty problems found in prompt data")
    optimizer = AdamW((p for p in student.parameters() if p.requires_grad),
                      lr=cfg.learning_rate, weight_decay=0.0)
    total_batches = math.ceil(len(prompts) / cfg.gradient_accumulation_steps) * cfg.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_batches * cfg.warmup_ratio),
        num_training_steps=total_batches,
    )
    attempted = accepted = updates = 0
    rng = random.Random(cfg.seed)
    for epoch in range(cfg.num_train_epochs):
        rng.shuffle(prompts)
        for start in range(0, len(prompts), cfg.gradient_accumulation_steps):
            batch = prompts[start:start + cfg.gradient_accumulation_steps]
            optimizer.zero_grad(set_to_none=True)
            kept = 0
            loss_sum = 0.0
            for example in batch:
                attempted += 1
                messages = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": example["problem"]}]
                s_prefix = prefix_ids(tokenizer, messages)
                # This is the live target, including every preceding optimizer update.
                initial_ids = sample_tokens(student, tokenizer, s_prefix, cfg, cfg.student_temperature)
                if not initial_ids:
                    continue
                y_init = trace_text(tokenizer, s_prefix, initial_ids)
                original = score_trace(y_init, str(example["solution"]), strict=True)
                kind, instruction = build_revision_instruction(
                    "topology", score=original, token_budget=cfg.token_budget,
                )
                teacher_messages = revision_messages(messages, y_init, instruction)
                t_prefix = prefix_ids(tokenizer, teacher_messages)
                response = sample_tokens(teacher, tokenizer, t_prefix, cfg, cfg.teacher_temperature)
                if not response:
                    continue
                revised_text = trace_text(tokenizer, t_prefix, response)
                revised = score_trace(revised_text, str(example["solution"]), strict=True)
                record = DistillationRecord(
                    problem=example["problem"], solution=str(example["solution"]),
                    y_init=y_init, P_r=instruction, y_revised=revised_text, defect_type=kind,
                    r_out_init=original["r_out"], r_out_revised=revised["r_out"],
                    q_topo_init=original["r_topo"], q_topo_revised=revised["r_topo"],
                    q_cont_init=original["r_cont"], q_cont_revised=revised["r_cont"],
                    q_dir_init=original["q_dir"], q_dir_revised=revised["q_dir"],
                    q_acyc_init=original["q_acyc"], q_acyc_revised=revised["q_acyc"],
                    revised_tokens=len(response), format_ok_revised=format_ok(revised_text),
                )
                if revision_rejection_reason(record, topo_threshold=0.0, token_budget=cfg.token_budget):
                    continue
                student.train()
                loss = revision_loss(student, teacher, s_prefix, t_prefix, response)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite online reverse-KL loss")
                # E[a_r * sum_t KL]: rejected attempts contribute zero, and the
                # denominator counts attempts, not only accepted revisions.
                (loss / len(batch)).backward()
                loss_sum += loss.detach().item()
                kept += 1
            if kept:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                updates += 1
                accepted += kept
            print(f"[tgd] epoch={epoch + 1} attempts={attempted} accepted={accepted} "
                  f"updates={updates} batch_loss={loss_sum / len(batch):.6f}", flush=True)
    if not updates:
        raise RuntimeError("No revision passed the gate; no trained checkpoint was saved")
    destination = cfg.output_dir / "final"
    destination.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    print(f"[tgd] final adapter -> {destination}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("student_model", "teacher_model"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("output/distill_topology"))
    for name in ("student_adapter", "teacher_adapter"):
        parser.add_argument("--" + name, default="")
    parser.add_argument("--student_device", default="cuda:0")
    parser.add_argument("--teacher_device", default="cuda:1")
    for name, default in (("num_train_epochs", 1), ("gradient_accumulation_steps", 8),
                          ("token_budget", 1024), ("max_length", 8192), ("max_prompts", 0), ("seed", 0)):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("learning_rate", 2e-5), ("warmup_ratio", 0.05),
                          ("student_temperature", 0.7), ("teacher_temperature", 0.0), ("top_p", 0.95)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    raise SystemExit(run_online_reverse_kl(TrainConfig(**vars(parser.parse_args()))))


if __name__ == "__main__":
    main()
