"""Online reverse-KL distillation trainer (teacher 32B -> student ~7/8B)."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from src.distill.reverse_kl_loss import reverse_kl_loss


@dataclass
class TrainConfig:
    teacher_model: str
    student_model: str
    dataset: list[str]
    output_dir: str
    teacher_adapter: str | None = None
    student_init_adapter: str | None = None
    max_length: int = 2048
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 200
    bf16: bool = True
    gradient_checkpointing: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    temperature: float = 1.0
    rkl_weight: float = 1.0
    ce_weight: float = 0.0
    max_train_samples: int = 0


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_config(raw: dict[str, Any]) -> TrainConfig:
    teacher_model = raw.get("teacher_model", "Qwen/Qwen3-32B")
    student_model = raw.get("student_model", raw.get("model", "/mnt/users/rwl/models/Qwen3-8B"))
    dataset = raw.get("dataset", [])
    if isinstance(dataset, str):
        dataset = [dataset]
    if not dataset:
        dataset = ["data/sft_ready/train_augmented.jsonl"]
    output_dir = raw.get("output_dir", "output/distill_rkl_8b")
    return TrainConfig(
        teacher_model=teacher_model,
        student_model=student_model,
        dataset=dataset,
        output_dir=output_dir,
        teacher_adapter=raw.get("teacher_adapter"),
        student_init_adapter=raw.get("student_init_adapter"),
        max_length=int(raw.get("max_length", 2048)),
        per_device_train_batch_size=int(raw.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 4)),
        learning_rate=float(raw.get("learning_rate", 2e-5)),
        num_train_epochs=int(raw.get("num_train_epochs", 1)),
        warmup_ratio=float(raw.get("warmup_ratio", 0.05)),
        weight_decay=float(raw.get("weight_decay", 0.01)),
        logging_steps=int(raw.get("logging_steps", 10)),
        save_steps=int(raw.get("save_steps", 200)),
        bf16=bool(raw.get("bf16", True)),
        gradient_checkpointing=bool(raw.get("gradient_checkpointing", True)),
        lora_rank=int(raw.get("lora_rank", 64)),
        lora_alpha=int(raw.get("lora_alpha", 128)),
        lora_dropout=float(raw.get("lora_dropout", 0.05)),
        target_modules=str(raw.get("target_modules", "all-linear")),
        temperature=float(raw.get("temperature", 1.0)),
        rkl_weight=float(raw.get("rkl_weight", 1.0)),
        ce_weight=float(raw.get("ce_weight", 0.0)),
        max_train_samples=int(raw.get("max_train_samples", 0)),
    )


def _extract_text(record: dict[str, Any], tokenizer: AutoTokenizer) -> str:
    msgs = record.get("messages")
    if isinstance(msgs, list) and msgs:
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            return "\n".join(str(x.get("content", "")) for x in msgs if isinstance(x, dict))
    for key in ("text", "response", "prediction", "output", "solution"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


class JsonlTextDataset(Dataset):
    def __init__(self, paths: list[Path], tokenizer: AutoTokenizer, max_length: int, max_samples: int = 0) -> None:
        self.samples: list[list[int]] = []
        for path in paths:
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    text = _extract_text(rec, tokenizer)
                    if not text:
                        continue
                    ids = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=True)["input_ids"]
                    if len(ids) < 4:
                        continue
                    self.samples.append(ids)
                    if max_samples > 0 and len(self.samples) >= max_samples:
                        break
            if max_samples > 0 and len(self.samples) >= max_samples:
                break
        if not self.samples:
            raise RuntimeError("No valid samples found for reverse-KL training.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> list[int]:
        return self.samples[idx]


def _collate(batch: list[list[int]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(x) for x in batch)
    input_ids = []
    attn = []
    for ids in batch:
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attn.append([1] * len(ids) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def _build_lora_config(cfg: TrainConfig) -> LoraConfig:
    target_modules = cfg.target_modules
    if target_modules == "all-linear":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif isinstance(target_modules, str):
        target_modules = [x.strip() for x in target_modules.split(",") if x.strip()]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )


def run_online_reverse_kl(config_path: Path, gpus: str, project_root: Path) -> int:
    raw = _load_yaml(config_path)
    cfg = _normalize_config(raw)
    output_dir = project_root / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_list = [x.strip() for x in gpus.split(",") if x.strip()]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 32B teacher online reverse-KL training.")
    teacher_gpu = gpu_list[0] if gpu_list else "0"
    student_gpu = gpu_list[1] if len(gpu_list) > 1 else teacher_gpu
    teacher_device = f"cuda:{teacher_gpu}"
    student_device = f"cuda:{student_gpu}"

    dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    print(f"[distill.rkl] teacher_device={teacher_device} student_device={student_device}", flush=True)
    print(f"[distill.rkl] teacher_model={cfg.teacher_model}", flush=True)
    print(f"[distill.rkl] student_model={cfg.student_model}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.student_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    student = AutoModelForCausalLM.from_pretrained(
        cfg.student_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": student_device},
    )
    if cfg.gradient_checkpointing:
        student.gradient_checkpointing_enable()
    student.enable_input_require_grads()
    student = get_peft_model(student, _build_lora_config(cfg))
    student.train()

    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": teacher_device},
    )
    if cfg.teacher_adapter:
        teacher_adapter_path = (project_root / cfg.teacher_adapter).resolve()
        if teacher_adapter_path.is_dir():
            teacher = PeftModel.from_pretrained(teacher, str(teacher_adapter_path), is_trainable=False)
            teacher = teacher.merge_and_unload()
            teacher.to(teacher_device)
            print(f"[distill.rkl] loaded teacher adapter={teacher_adapter_path}", flush=True)
        else:
            print(f"[distill.rkl] teacher adapter not found, ignore: {teacher_adapter_path}", flush=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    if student.config.vocab_size != teacher.config.vocab_size:
        raise RuntimeError(
            f"Vocab mismatch: student={student.config.vocab_size}, teacher={teacher.config.vocab_size}. "
            "Use same tokenizer family for reverse-KL."
        )

    ds_paths = [project_root / p for p in cfg.dataset]
    dataset = JsonlTextDataset(ds_paths, tokenizer, cfg.max_length, cfg.max_train_samples)
    loader = DataLoader(
        dataset,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate(b, tokenizer.pad_token_id),
        drop_last=False,
    )

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = math.ceil(len(loader) * cfg.num_train_epochs / max(cfg.gradient_accumulation_steps, 1))
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    metrics: list[dict[str, float]] = []
    progress = tqdm(total=total_steps, desc="reverse-kl-train")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.num_train_epochs):
        for it, batch in enumerate(loader):
            ids_s = batch["input_ids"].to(student_device)
            attn_s = batch["attention_mask"].to(student_device)
            ids_t = batch["input_ids"].to(teacher_device)
            attn_t = batch["attention_mask"].to(teacher_device)

            student_out = student(input_ids=ids_s, attention_mask=attn_s, use_cache=False)
            with torch.no_grad():
                teacher_out = teacher(input_ids=ids_t, attention_mask=attn_t, use_cache=False)

            # Shift for next-token objective.
            s_logits = student_out.logits[:, :-1, :]
            t_logits = teacher_out.logits[:, :-1, :].to(student_device)
            token_mask = attn_s[:, 1:].float()
            temp = max(cfg.temperature, 1e-4)
            rkl = reverse_kl_loss(s_logits / temp, t_logits / temp, token_mask) * (temp * temp)

            if cfg.ce_weight > 0:
                labels = ids_s[:, 1:]
                ce = F.cross_entropy(
                    s_logits.reshape(-1, s_logits.size(-1)),
                    labels.reshape(-1),
                    reduction="none",
                )
                ce = (ce.reshape(labels.shape) * token_mask).sum() / token_mask.sum().clamp_min(1.0)
            else:
                ce = torch.zeros((), device=student_device, dtype=s_logits.dtype)

            loss = cfg.rkl_weight * rkl + cfg.ce_weight * ce
            (loss / cfg.gradient_accumulation_steps).backward()

            if (it + 1) % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)

                record = {
                    "step": float(global_step),
                    "loss": float(loss.detach().item()),
                    "rkl_loss": float(rkl.detach().item()),
                    "ce_loss": float(ce.detach().item()),
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                metrics.append(record)
                if global_step % cfg.logging_steps == 0:
                    print(f"[distill.rkl] {record}", flush=True)
                if global_step % cfg.save_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    student.save_pretrained(str(ckpt_dir))
                    tokenizer.save_pretrained(str(ckpt_dir))
                    print(f"[distill.rkl] saved {ckpt_dir}", flush=True)

                if global_step >= total_steps:
                    break
        if global_step >= total_steps:
            break

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "teacher_model": cfg.teacher_model,
        "student_model": cfg.student_model,
        "num_samples": len(dataset),
        "train_steps": global_step,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[distill.rkl] done, final adapter -> {final_dir}", flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Online reverse-KL distillation trainer")
    parser.add_argument("--config", type=Path, default=Path("configs/distill_7b_compact.yaml"))
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--project_root", type=Path, default=Path("."))
    args = parser.parse_args()

    rc = run_online_reverse_kl(args.config, args.gpus, args.project_root.resolve())
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
