#!/usr/bin/env python3
"""Lightweight GSM8K + MATH-500 benchmark using transformers generate (no vLLM server).

Usage:
    CUDA_VISIBLE_DEVICES=4 python3 scripts/bench_transformers.py \
        --model /mnt/data/huggingface_downloads/models/qwen/Qwen3.5-9B \
        --adapter output/sft_qwen35_9b/v0-20260407-011328/checkpoint-626 \
        --label sft_9b --benchmarks gsm8k math500
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
import shutil
import tempfile
from collections import Counter

import torch
from datasets import load_dataset
from peft import PeftModel
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.unified_benchmark import evaluate_predictions
from src.reward.topo_reward import TopoReward


def extract_number(text: str) -> str | None:
    """Extract the final numeric answer from a GSM8K/MATH-style response."""
    # Look for #### pattern (GSM8K gold format)
    m = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    # Look for \\boxed{...}
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    # Last number in text
    nums = re.findall(r"[+-]?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def normalize_answer(ans: str) -> str:
    ans = ans.strip().replace(",", "").replace("$", "").replace("%", "")
    # Remove trailing period
    if ans.endswith("."):
        ans = ans[:-1]
    return ans.lower()


def extract_mcq(text: str) -> str | None:
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1) if m else None


def answer_extractor_for_benchmark(bench: str):
    if bench in {"gsm8k", "math500", "olympiadbench", "omni_math", "aime2024", "cnmo2024"}:
        return extract_number
    if bench in {"mmlu", "gpqa_diamond"}:
        return extract_mcq
    if bench == "livecode":
        return lambda x: x
    return extract_number


def compute_prm_at_k(correct_flags_per_item: list[list[bool]], prm_scores_per_item: list[list[float]], k: int) -> float:
    if not correct_flags_per_item:
        return 0.0
    hit = 0
    total = 0
    for flags, scores in zip(correct_flags_per_item, prm_scores_per_item):
        if not flags or not scores:
            continue
        kk = min(k, len(flags), len(scores))
        if kk <= 0:
            continue
        best_idx = max(range(kk), key=lambda i: scores[i])
        hit += 1 if flags[best_idx] else 0
        total += 1
    return hit / total if total else 0.0


def load_gsm8k() -> list[dict]:
    ds = load_dataset(
        "openai/gsm8k", "main", split="test",
        cache_dir="/root/.cache/huggingface/datasets",
    )
    items = []
    for row in ds:
        q = row["question"]
        # Gold answer after ####
        m = re.search(r"####\s*(.+)", row["answer"])
        gold = m.group(1).strip() if m else row["answer"].strip()
        items.append({"question": q, "gold": gold, "source": "gsm8k"})
    return items


def load_math500() -> list[dict]:
    """Load MATH-500 from evalscope's parquet cache or HF."""
    try:
        ds = load_dataset(
            "HuggingFaceH4/MATH-500", split="test",
            cache_dir="/root/.cache/huggingface/datasets",
        )
    except Exception:
        ds = load_dataset(
            "lighteval/MATH", split="test",
            cache_dir="/root/.cache/huggingface/datasets",
        )
        # Sample 500
        import random
        random.seed(42)
        indices = random.sample(range(len(ds)), min(500, len(ds)))
        ds = ds.select(indices)
    items = []
    for row in ds:
        q = row.get("problem", row.get("question", ""))
        gold = row.get("answer", row.get("solution", ""))
        # Extract boxed answer if present
        m = re.search(r"\\boxed\{([^}]+)\}", gold)
        if m:
            gold = m.group(1).strip()
        items.append({"question": q, "gold": gold, "source": "math500"})
    return items


def load_generic_hf(
    hf_path: str,
    split: str,
    question_key: str,
    answer_key: str,
    source: str,
    hf_name: str | None = None,
) -> list[dict]:
    if hf_name:
        ds = load_dataset(hf_path, hf_name, split=split, cache_dir="/root/.cache/huggingface/datasets")
    else:
        ds = load_dataset(hf_path, split=split, cache_dir="/root/.cache/huggingface/datasets")
    items = []
    for row in ds:
        q = row.get(question_key, "")
        gold = row.get(answer_key, "")
        items.append({"question": str(q), "gold": str(gold), "source": source})
    return items


def load_benchmark(bench: str) -> list[dict]:
    if bench == "gsm8k":
        return load_gsm8k()
    if bench == "math500":
        return load_math500()
    if bench == "aime2024":
        return load_generic_hf(
            "AI-MO/aimo-validation-aime",
            "train",
            "problem",
            "answer",
            bench,
            None,
        )
    if bench == "mmlu":
        return load_generic_hf("cais/mmlu", "test", "question", "answer", bench, "all")
    if bench == "gpqa_diamond":
        return load_generic_hf("Idavidrein/gpqa", "train", "Question", "Correct Answer", bench, "gpqa_diamond")
    if bench == "omni_math":
        return load_generic_hf("KbsdJames/Omni-MATH", "test", "problem", "answer", bench, None)
    if bench == "olympiadbench":
        return load_generic_hf("lmms-lab/OlympiadBench", "test_en", "question", "final_answer", bench, None)
    if bench == "livecode":
        return load_generic_hf("livecodebench/code_generation_lite", "test", "question_content", "test", bench, None)
    if bench == "cnmo2024":
        local = Path("data/benchmarks/cnmo2024.jsonl")
        items = []
        if local.exists():
            for line in local.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                items.append(
                    {
                        "question": str(obj.get("problem", obj.get("question", ""))),
                        "gold": str(obj.get("answer", "")),
                        "source": bench,
                    }
                )
        return items
    raise ValueError(f"Unknown benchmark: {bench}")


def build_prompt(question: str, source: str) -> str:
    if source == "gsm8k":
        return (
            f"Solve the following math problem step by step. "
            f"Put your final answer after ####.\n\n"
            f"Question: {question}\n\nAnswer:"
        )
    else:
        return (
            f"Solve the following math problem. "
            f"Put your final answer in \\boxed{{}}.\n\n"
            f"Problem: {question}\n\nSolution:"
        )


def patch_swift_adapter_namespace(adapter_dir: Path) -> Path:
    """Patch swift LoRA adapter namespace for vanilla transformers loading."""
    tmp_adapter_root = Path(tempfile.mkdtemp(prefix="adapter_fix_"))
    patched_dir = tmp_adapter_root / "adapter"
    shutil.copytree(adapter_dir, patched_dir, dirs_exist_ok=True)

    cfg_path = patched_dir / "adapter_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        tm = cfg.get("target_modules", "")
        if isinstance(tm, str) and "language_model" in tm:
            cfg["target_modules"] = tm.replace("model\\.language_model(?=\\.)", "model")
            cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            print(f"  Fixed target_modules: {tm} -> {cfg['target_modules']}")

    # Swift checkpoints may keep `...model.language_model...` in tensor keys.
    # We must rename state dict keys as well, otherwise LoRA weights are silently missed.
    safetensor_path = patched_dir / "adapter_model.safetensors"
    if safetensor_path.exists():
        state = safe_load_file(str(safetensor_path))
        needs_patch = any(".language_model." in k for k in state.keys())
        if needs_patch:
            patched = {}
            for k, v in state.items():
                nk = k.replace(".language_model.", ".")
                patched[nk] = v
            safe_save_file(patched, str(safetensor_path))
            print(f"  Fixed adapter tensor namespace in {safetensor_path.name}")

    bin_path = patched_dir / "adapter_model.bin"
    if bin_path.exists():
        state = torch.load(bin_path, map_location="cpu")
        if isinstance(state, dict):
            needs_patch = any(".language_model." in k for k in state.keys())
            if needs_patch:
                patched = {}
                for k, v in state.items():
                    nk = k.replace(".language_model.", ".")
                    patched[nk] = v
                torch.save(patched, bin_path)
                print(f"  Fixed adapter tensor namespace in {bin_path.name}")

    return patched_dir


@torch.inference_mode()
def run_benchmark(
    model,
    tokenizer,
    items: list[dict],
    bench_name: str,
    batch_size: int = 4,
    max_new_tokens: int = 1024,
    num_samples_per_item: int = 1,
    k_values: list[int] | None = None,
) -> tuple[dict, list[dict]]:
    if k_values is None:
        k_values = [1, 5]

    total = len(items)
    results = []
    predictions_per_item: list[list[str]] = [[] for _ in range(total)]
    token_counts_per_item: list[list[int]] = [[] for _ in range(total)]
    extractor = answer_extractor_for_benchmark(bench_name)
    topo_reward = TopoReward()

    for i in range(0, total, batch_size):
        batch = items[i : i + batch_size]
        prompts = [build_prompt(it["question"], it["source"]) for it in batch]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048,
        ).to(model.device)

        # sample 0 = greedy for pass@1; samples >=1 optionally stochastic
        for sample_idx in range(num_samples_per_item):
            do_sample = sample_idx > 0
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs.update({"temperature": 0.8, "top_p": 0.95})

            outputs = model.generate(**inputs, **gen_kwargs)
            for j, out_ids in enumerate(outputs):
                prompt_len = inputs["input_ids"][j].shape[0]
                gen_ids = out_ids[prompt_len:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                global_idx = i + j
                predictions_per_item[global_idx].append(gen_text)
                token_counts_per_item[global_idx].append(len(gen_ids))

        # streaming status by current pass@1
        done = min(i + batch_size, total)
        pass1_correct = 0
        for idx in range(done):
            pred0 = extractor(predictions_per_item[idx][0]) if predictions_per_item[idx] else None
            gold = normalize_answer(items[idx]["gold"])
            pred_norm = normalize_answer(pred0) if pred0 else ""
            pass1_correct += 1 if pred_norm == gold else 0
        acc_so_far = pass1_correct / done * 100
        print(f"  [{done}/{total}] acc={acc_so_far:.1f}%", flush=True)

    # build per-item results and compute PRM scores
    correct_flags_per_item: list[list[bool]] = []
    prm_scores_per_item: list[list[float]] = []
    for idx, item in enumerate(items):
        raw_preds = predictions_per_item[idx]
        extracted = [extractor(p) for p in raw_preds]
        gold_norm = normalize_answer(item["gold"])
        flags = [(normalize_answer(p) if p else "") == gold_norm for p in extracted]
        correct_flags_per_item.append(flags)

        # PRM proxy: use topo reward per sampled completion
        completion_objs = [[{"role": "assistant", "content": p}] for p in raw_preds]
        prm_scores = topo_reward(completion_objs) if completion_objs else []
        prm_scores_per_item.append(prm_scores)

        results.append(
            {
                "question": item["question"][:200],
                "gold": item["gold"],
                "pred_pass1": extracted[0] if extracted else None,
                "correct_pass1": flags[0] if flags else False,
                "num_samples": len(raw_preds),
                "correct_count": sum(flags),
                "avg_gen_tokens": round(sum(token_counts_per_item[idx]) / max(len(token_counts_per_item[idx]), 1), 1),
                "prm_scores": prm_scores,
            }
        )

    gold_answers = [str(it["gold"]) for it in items]
    metrics = evaluate_predictions(
        predictions_per_item=predictions_per_item,
        gold_answers=gold_answers,
        answer_extractor=extractor,
        k_values=k_values,
        token_counts=token_counts_per_item,
    )

    # Add PRM@k
    for k in k_values:
        metrics[f"prm@{k}"] = round(compute_prm_at_k(correct_flags_per_item, prm_scores_per_item, k), 4)

    # Backward-compatible fields
    metrics["accuracy"] = metrics.get("pass@1", 0.0)
    metrics["accuracy_pct"] = round(metrics["accuracy"] * 100.0, 2)
    metrics["correct"] = int(metrics.get("correct_count", sum(1 for r in results if r["correct_pass1"])))
    metrics["error"] = int(metrics.get("error_count", len(items) - metrics["correct"]))
    metrics["num_samples"] = len(items)
    metrics["pass_at_k"] = {str(k): metrics.get(f"pass@{k}", 0.0) for k in k_values}
    metrics["maj_at_k"] = {str(k): metrics.get(f"maj@{k}", 0.0) for k in k_values}
    metrics["prm_at_k"] = {str(k): metrics.get(f"prm@{k}", 0.0) for k in k_values}

    return metrics, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--label", required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "math500"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--num_samples_per_item", type=int, default=1)
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--output_dir", default="output/eval")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.adapter and Path(args.adapter).is_dir():
        print(f"Loading adapter: {args.adapter}")
        patched_adapter = patch_swift_adapter_namespace(Path(args.adapter))
        model = PeftModel.from_pretrained(model, str(patched_adapter))
        shutil.rmtree(patched_adapter.parent, ignore_errors=True)
        model = model.merge_and_unload()

    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for bench in args.benchmarks:
        print(f"\n{'='*60}")
        print(f"Benchmark: {bench} | Label: {args.label}")
        print(f"{'='*60}")

        try:
            items = load_benchmark(bench)
        except Exception as exc:
            print(f"Unknown/failed benchmark {bench}, skipping: {exc}")
            continue
        if not items:
            print(f"No data for benchmark {bench}, skipping")
            continue
        if args.max_items > 0:
            items = items[: args.max_items]

        t0 = time.time()
        metrics, results = run_benchmark(
            model, tokenizer, items,
            bench_name=bench,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            num_samples_per_item=args.num_samples_per_item,
            k_values=args.k_values,
        )
        elapsed = time.time() - t0

        metrics["elapsed_sec"] = round(elapsed, 1)
        metrics["backend"] = "transformers"
        metrics["label"] = args.label
        metrics["num_samples_per_item"] = args.num_samples_per_item
        metrics["k_values"] = args.k_values

        metrics_path = out_dir / f"{args.label}_{bench}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

        details_path = out_dir / f"{args.label}_{bench}_details.jsonl"
        with details_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(
            f"\n  {bench}: pass@1={metrics.get('pass@1', 0.0)*100:.1f}% "
            f"pass@5={metrics.get('pass@5', 0.0)*100:.1f}% "
            f"maj@5={metrics.get('maj@5', 0.0)*100:.1f}% "
            f"prm@5={metrics.get('prm@5', 0.0)*100:.1f}% "
            f"in {elapsed:.0f}s"
        )
        print(f"  Saved: {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
