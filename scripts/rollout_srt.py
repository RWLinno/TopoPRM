#!/usr/bin/env python3
"""Generate topology-guided teacher revisions from compact-student traces.

For each problem, a compact student produces an initial trace. The frozen
Stage-II teacher then revises a localized backward, cyclic, orphan, or
continuity defect. Raw candidates are filtered by build_srt_data.py.

Usage:
    python -m scripts.rollout_srt \
        --student_model /path/to/Qwen3.5-4B \
        --teacher_model /path/to/Qwen3.5-9B \
        --teacher_adapter /path/to/stage2-adapter \
        --input data/grpo_ready/train_public_swift.jsonl \
        --output ${EXP_ROOT}/.../distill_candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import networkx as nx
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.distill.build_srt_data import (
    SYSTEM_PROMPT,
    build_revision_instruction,
    derive_record_seed,
    format_ok,
)
from src.reward.outcome_reward import OutcomeReward
from src.reward.topo_reward import TopoReward
from src.reward.continuity_reward import ContinuityReward
from scripts.bench_transformers import patch_swift_adapter_namespace


def _set_record_seed(seed: int, record_id: str, stream: str) -> int:
    record_seed = derive_record_seed(seed, record_id, stream)
    random.seed(record_seed)
    torch.manual_seed(record_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(record_seed)
    return record_seed


def load_prompts(path: Path, max_n: int = 0) -> list[dict[str, Any]]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            messages = d.get("messages", [])
            # Extract user question from last user message.
            user_text = ""
            for m in messages:
                if m.get("role") == "user":
                    user_text = m.get("content", "")
            if not user_text:
                # Support public GRPO records with a direct "question" field.
                user_text = d.get("question", "")
            if not user_text:
                continue
            out.append({
                "record_id": str(d.get("record_id", f"record_{len(out)}")),
                "problem": user_text,
                "solution": d.get("solution", d.get("standard_answer", d.get("final_answer", ""))),
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
    topo_diag: dict[str, float] = {}
    try:
        r_topo = float(topo_rw(completions, reference_dag=reference_dag)[0])
        if topo_rw.last_diagnostics:
            topo_diag = topo_rw.last_diagnostics[0]
    except Exception:
        r_topo = 0.0
    continuity_breaks: list[int] = []
    try:
        r_cont, continuity_breaks = cont_rw.diagnose(text)
        r_cont = float(r_cont)
    except Exception:
        r_cont = 0.0

    # Binary r_out
    r_out_bin = 1 if r_out >= 0.5 else 0

    defect = {
        "backward_edges": [],
        "cycle_components": [],
        "orphan_steps": [],
        "continuity_breaks": continuity_breaks,
    }
    try:
        from src.dag.node import StepType

        dag = topo_rw.last_dags[0] if topo_rw.last_dags else None
        if dag is not None and getattr(dag, "nodes", None):
            raw_edges = [
                edge for edge in dag.graph.graph.get("raw_dependency_edges", [])
                if dag.is_virtual_edge(str(edge.get("edge_type", "")))
            ]
            defect["backward_edges"] = [
                edge for edge in raw_edges
                if int(edge.get("source", -1)) >= int(edge.get("target", -1))
            ]
            raw_graph = nx.DiGraph()
            raw_graph.add_nodes_from(dag.nodes)
            raw_graph.add_edges_from(
                (int(edge["source"]), int(edge["target"])) for edge in raw_edges
            )
            defect["cycle_components"] = [
                sorted(int(node) for node in component)
                for component in nx.strongly_connected_components(raw_graph)
                if len(component) > 1
            ]
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
                    defect["orphan_steps"].append(int(sid))
    except Exception:
        pass

    return {
        "r_out": r_out_bin,
        "r_out_raw": r_out,
        "r_topo": r_topo,
        "r_cont": r_cont,
        "q_dir": float(topo_diag.get("direction_consistency", 0.0)),
        "q_acyc": float(topo_diag.get("acyclic", 0.0)),
        "defect": defect,
    }


@torch.inference_mode()
def sample_response(model, tokenizer, msgs, *, max_new_tokens=2048,
                    temperature=0.8, top_p=0.95) -> str:
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    generation = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation.update(temperature=temperature, top_p=top_p)
    out = model.generate(**inputs, **generation)
    prompt_len = inputs["input_ids"][0].shape[0]
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


def _load_model(model_name: str, adapter: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left",
    )
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    if adapter:
        if not Path(adapter).is_dir():
            raise FileNotFoundError(f"Adapter not found: {adapter}")
        patched = patch_swift_adapter_namespace(Path(adapter))
        model = PeftModel.from_pretrained(model, str(patched))
        shutil.rmtree(patched.parent, ignore_errors=True)
        model = model.merge_and_unload()
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_model", required=True)
    ap.add_argument("--student_adapter", default="")
    ap.add_argument("--student_device", default="cuda:0")
    ap.add_argument("--teacher_model", required=True)
    ap.add_argument("--teacher_adapter", required=True)
    ap.add_argument("--teacher_device", default="cuda:1")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max_prompts", type=int, default=2000)
    ap.add_argument("--samples_per_prompt", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--student_temperature", type=float, default=0.7)
    ap.add_argument("--teacher_temperature", type=float, default=0.0)
    ap.add_argument(
        "--revision_strategy",
        choices=["topology", "generic", "length", "static"],
        default="topology",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading student: {args.student_model}")
    student, student_tokenizer = _load_model(
        args.student_model, args.student_adapter, args.student_device,
    )
    print(f"Loading teacher: {args.teacher_model} + {args.teacher_adapter}")
    teacher, teacher_tokenizer = _load_model(
        args.teacher_model, args.teacher_adapter, args.teacher_device,
    )

    prompts = load_prompts(args.input, args.max_prompts)
    print(
        f"Loaded {len(prompts)} prompts; strategy={args.revision_strategy}; "
        f"seed={args.seed}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with args.output.open("w") as fout:
        for pi, p in enumerate(prompts):
            problem = p["problem"]
            solution = p["solution"]
            ref_dag = p.get("reference_dag")

            for sample_i in range(args.samples_per_prompt):
                record_id = f"{p['record_id']}:sample-{sample_i}"
                # 1) sample y_init
                msgs_init = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": problem},
                ]
                try:
                    student_seed = _set_record_seed(
                        args.seed, record_id, "student"
                    )
                    y_init = sample_response(
                        student, student_tokenizer, msgs_init,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.student_temperature,
                    )
                except Exception as e:
                    print(f"  skip prompt {pi} init: {e}")
                    continue

                sc_init = score_trace(y_init, solution, ref_dag)

                # 2) build P_r and sample y_revised
                defect_type, P_r = build_revision_instruction(
                    args.revision_strategy,
                    score=sc_init,
                    token_budget=args.max_new_tokens,
                )
                if args.revision_strategy == "static":
                    msgs_rev = msgs_init
                else:
                    msgs_rev = msgs_init + [
                        {"role": "assistant", "content": y_init},
                        {"role": "user", "content": P_r},
                    ]
                try:
                    teacher_seed = _set_record_seed(
                        args.seed,
                        record_id,
                        f"teacher:{args.revision_strategy}",
                    )
                    y_rev = sample_response(
                        teacher, teacher_tokenizer, msgs_rev,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.teacher_temperature,
                    )
                except Exception as e:
                    print(f"  skip prompt {pi} rev: {e}")
                    continue

                sc_rev = score_trace(y_rev, solution, ref_dag)

                rec = {
                    "record_id": record_id,
                    "revision_strategy": args.revision_strategy,
                    "seed": args.seed,
                    "student_seed": student_seed,
                    "teacher_seed": teacher_seed,
                    "generation": {
                        "max_new_tokens": args.max_new_tokens,
                        "student_temperature": args.student_temperature,
                        "teacher_temperature": args.teacher_temperature,
                        "top_p": 0.95,
                    },
                    "problem": problem,
                    "solution": solution,
                    "y_init": y_init,
                    "P_r": P_r,
                    "y_revised": y_rev,
                    "r_out_init": sc_init["r_out"],
                    "r_topo_init": sc_init["r_topo"],
                    "r_cont_init": sc_init["r_cont"],
                    "q_dir_init": sc_init["q_dir"],
                    "q_acyc_init": sc_init["q_acyc"],
                    "r_out_revised": sc_rev["r_out"],
                    "r_topo_revised": sc_rev["r_topo"],
                    "r_cont_revised": sc_rev["r_cont"],
                    "q_dir_revised": sc_rev["q_dir"],
                    "q_acyc_revised": sc_rev["q_acyc"],
                    "defect": sc_init["defect"],
                    "defect_type": defect_type,
                    "revised_tokens": len(
                        teacher_tokenizer(y_rev, add_special_tokens=False)["input_ids"]
                    ),
                    "format_ok_revised": format_ok(y_rev),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_written += 1

                if n_written % 20 == 0:
                    print(f"  [{n_written}] prompt={pi}/{len(prompts)} "
                          f"r_out_init={sc_init['r_out']} r_out_rev={sc_rev['r_out']} "
                          f"defect={defect_type}")

    print(f"Wrote {n_written} rollouts to {args.output}")


if __name__ == "__main__":
    main()
