"""On-Policy Self-Distillation trainer for TVSD Phase III-B.

For each problem x in D2:
    y ~ pi_theta(.|x)                        # student on-policy
    r_out, r_topo, r_cont = score(y, ...)
    P_r = build_prompt_dispatch(r_out, r_topo)
    teacher_logits_t = pi_SRT(.|x, y, P_r, y_{<t})
    L = sum_t KL( pi_theta(.|x, y_{<t})  ||  teacher_logits_t )

We use a lightweight PyTorch training loop (not Swift) because Swift's
GRPO trainer is tightly coupled to its reward format; OPSD needs direct
access to teacher logits per token.

Usage:
    CUDA_VISIBLE_DEVICES=4,5 python3 -m src.distill.opsd_trainer \
        --config configs/opsd_9b.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup,
)

from src.distill.build_srt_data import build_prompt_dispatch
from src.reward.continuity_reward import ContinuityReward
from src.reward.outcome_reward import OutcomeReward
from src.reward.topo_reward import TopoReward


SYSTEM_PROMPT = (
    "You are a math reasoning assistant. Think step by step inside "
    "<think>...</think>, then put the final answer inside <answer>...</answer>."
)


def load_cfg(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_peft(base: str, adapter: str, dtype=torch.bfloat16, device_map="auto"):
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, device_map=device_map, trust_remote_code=True,
    )
    if adapter and Path(adapter).is_dir():
        # Patch swift namespace inline for safety
        try:
            from scripts.bench_transformers import patch_swift_adapter_namespace
            patched = patch_swift_adapter_namespace(Path(adapter))
            model = PeftModel.from_pretrained(model, str(patched))
            shutil.rmtree(patched.parent, ignore_errors=True)
            model = model.merge_and_unload()
        except Exception as e:
            print(f"  peft load fallback ({e})")
            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload()
    return model


def load_prompts(path: Path, split: str, max_n: int = 0) -> list[dict]:
    out = []
    with open(path) as f:
        lines = f.readlines()
    # Split D1/D2: first half vs second half
    n = len(lines)
    if split == "D1":
        use = lines[: n // 2]
    elif split == "D2":
        use = lines[n // 2 :]
    else:
        use = lines
    for line in use:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        msgs = d.get("messages", [])
        user_text = next((m["content"] for m in msgs if m.get("role") == "user"), "")
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


@torch.inference_mode()
def sample_student(model, tok, problem: str, max_new_tokens: int,
                   temperature: float = 0.7) -> tuple[str, torch.Tensor, torch.Tensor]:
    """Return (text, prompt_ids, full_ids)."""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(model.device)
    out_ids = model.generate(
        input_ids=ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    full = out_ids[0]
    text = tok.decode(full[ids.shape[1]:], skip_special_tokens=True)
    return text, ids[0], full


@torch.inference_mode()
def compute_teacher_logits(teacher, tok, problem: str, y: str, P_r: str,
                           device) -> torch.Tensor:
    """Teacher distribution over y tokens conditioned on (x, y, P_r).

    Standard OPSD: teacher sees the full y plus P_r in its context.
    We compute per-token logits teacher_logits[t] over the vocabulary at
    each position in y_{<=t}.
    """
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": y},
        {"role": "user", "content": P_r},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=6144).input_ids.to(device)
    out = teacher(input_ids=ids)
    return out.logits.squeeze(0)  # [T, V]


def score_response(text: str, solution: str, reference_dag=None) -> dict:
    """Batch-safe score."""
    completions = [[{"role": "assistant", "content": text}]]
    r_out = 0.0
    r_topo = 0.0
    r_cont = 0.0
    try:
        r_out = float(OutcomeReward()(completions, solution=solution)[0])
    except Exception:
        pass
    try:
        r_topo = float(TopoReward()(completions, reference_dag=reference_dag)[0])
    except Exception:
        pass
    try:
        r_cont = float(ContinuityReward()(completions)[0])
    except Exception:
        pass

    # Find first orphan conclusion step index, mirroring rollout_srt.score_trace.
    # Used by build_prompt_dispatch to fill P_r's {k} with the real index instead
    # of the legacy default k=1 (handoff §C.1).
    orphan_step = None
    try:
        from src.data.build_dag import build_dag_from_answer
        from src.dag.node import StepType

        dag = build_dag_from_answer(text)
        if dag is not None and getattr(dag, "nodes", None):
            for sid in sorted(dag.nodes.keys()):
                node = dag.nodes[sid]
                if node.step_type != StepType.CONCLUSION:
                    continue
                has_virtual_pred = any(
                    dag.is_virtual_edge(
                        dag.graph.edges[u, sid].get("edge_type", "")
                    )
                    for u in dag.graph.predecessors(sid)
                )
                if not has_virtual_pred:
                    orphan_step = int(sid)
                    break
    except Exception:
        orphan_step = None

    return {
        "r_out": 1 if r_out >= 0.5 else 0,
        "r_topo": r_topo,
        "r_cont": r_cont,
        "orphan_step": orphan_step,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--max_steps", type=int, default=0,
                    help="Override num_train_epochs with hard step cap (0 = no cap)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    print(f"Config: {cfg}")

    # Load teacher (frozen) and student (trainable LoRA)
    print("Loading teacher...")
    teacher = _load_peft(
        cfg["teacher_base"], cfg.get("teacher_adapter", ""),
        device_map={"": 0},  # teacher on GPU 0 (CUDA_VISIBLE_DEVICES maps to physical GPU 4 if set)
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("Loading student...")
    student = _load_peft(
        cfg["student_base"], cfg.get("student_adapter", ""),
        device_map={"": 1 if torch.cuda.device_count() > 1 else 0},
    )

    # Wrap student with new LoRA for training
    lora = LoraConfig(
        r=64, lora_alpha=128, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                              "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    student = get_peft_model(student, lora)
    student.train()

    tok = AutoTokenizer.from_pretrained(cfg["student_base"], padding_side="left", trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    prompts = load_prompts(Path(cfg["dataset"]), cfg.get("dataset_split", "D2"))
    print(f"Loaded {len(prompts)} prompts for OPSD training")

    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=float(cfg.get("learning_rate", 1e-5)),
        weight_decay=0.01,
    )
    total_steps = len(prompts) * cfg.get("num_train_epochs", 1) // max(cfg.get("gradient_accumulation_steps", 1), 1)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    sched = get_cosine_schedule_with_warmup(
        opt, int(cfg.get("warmup_ratio", 0.03) * total_steps), total_steps,
    )

    ga_steps = cfg.get("gradient_accumulation_steps", 1)
    step = 0
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.get("num_train_epochs", 1)):
        for pi, p in enumerate(prompts):
            if args.max_steps and step >= args.max_steps:
                break

            # 1) Student on-policy rollout
            try:
                y, _prompt_ids, _ = sample_student(
                    student, tok, p["problem"],
                    max_new_tokens=cfg.get("max_new_tokens", 1024),
                    temperature=0.7,
                )
            except Exception as e:
                print(f"  skip {pi}: student sample err {e}")
                continue

            # 2) Score y and build P_r (P_r placeholder {k} now uses the
            # actual orphan step index when available, mirroring rollout_srt
            # post-D3; previously P_r always degraded to k=1).
            sc = score_response(y, p["solution"], p.get("reference_dag"))
            if cfg.get("use_topo_aware_pr", True):
                _bucket, P_r = build_prompt_dispatch(
                    sc["r_out"],
                    sc["r_topo"],
                    cfg.get("topo_threshold", 0.5),
                    orphan_step=sc.get("orphan_step"),
                )
            else:
                P_r = ("Let me rephrase the above solution." if sc["r_out"]
                       else "Wait, this response is not correct, let me start over.")

            # 3) Compute teacher distribution over y (as y tokens)
            try:
                teacher_logits = compute_teacher_logits(
                    teacher, tok, p["problem"], y, P_r, teacher.device,
                )  # [T_teacher, V]
            except Exception as e:
                print(f"  skip {pi}: teacher logits err {e}")
                continue

            # 4) Build student forward over same (x, y)
            msgs_student = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p["problem"]},
                {"role": "assistant", "content": y},
            ]
            student_prompt = tok.apply_chat_template(
                msgs_student, tokenize=False, add_generation_prompt=False,
            )
            s_ids = tok(student_prompt, return_tensors="pt", truncation=True, max_length=5120
                        ).input_ids.to(student.device)
            s_out = student(input_ids=s_ids)
            student_logits = s_out.logits.squeeze(0)  # [T_s, V]

            # 5) Align the y-token positions and compute KL
            # Simplification: use the minimum length, last n tokens should approximate y
            T = min(student_logits.shape[0], teacher_logits.shape[0])
            sl = student_logits[-T:]
            tl = teacher_logits[-T:].to(sl.device)
            temp = float(cfg.get("kl_temperature", 1.0))
            log_p = F.log_softmax(sl / temp, dim=-1)
            log_q = F.log_softmax(tl / temp, dim=-1)
            # KL( p || q ) = sum p*(log_p - log_q)
            p_dist = log_p.exp()
            kl = (p_dist * (log_p - log_q)).sum(dim=-1).mean()

            loss = kl / ga_steps
            loss.backward()

            if (pi + 1) % ga_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad], 1.0,
                )
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1

                if step % cfg.get("logging_steps", 10) == 0:
                    print(f"  step {step}/{total_steps} loss={loss.item()*ga_steps:.4f} "
                          f"r_out={sc['r_out']} r_topo={sc['r_topo']:.3f}")

                if step % cfg.get("save_steps", 100) == 0:
                    ckpt = output_dir / f"checkpoint-{step}"
                    ckpt.mkdir(exist_ok=True, parents=True)
                    student.save_pretrained(str(ckpt))
                    print(f"  saved {ckpt}")

    # Final save
    final = output_dir / "final"
    final.mkdir(exist_ok=True, parents=True)
    student.save_pretrained(str(final))
    print(f"Training complete. Saved to {final}")


if __name__ == "__main__":
    main()
