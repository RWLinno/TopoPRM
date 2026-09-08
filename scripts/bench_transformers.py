#!/usr/bin/env python3
"""Unified benchmark runner using transformers generate (no vLLM server).

Supports 10+ benchmarks with chat-template prompting, few-shot CoT, and
pass@k / maj@k / prm@k metrics via multi-sample generation.

Usage:
    CUDA_VISIBLE_DEVICES=4 python3 scripts/bench_transformers.py \
        --model /mnt/data/huggingface_downloads/models/qwen/Qwen3.5-9B \
        --adapter output/sft_qwen35_9b/v0-20260407-011328/checkpoint-626 \
        --label sft_9b_v2 --benchmarks gsm8k math500 \
        --use_chat_template --num_samples_per_item 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
import shutil
import tempfile

import torch
from datasets import load_dataset
from peft import PeftModel
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.unified_benchmark import bootstrap_item_metrics, evaluate_predictions
from src.reward.outcome_reward import OutcomeReward


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_fingerprint(root: str, *, include_adapter_weights: bool) -> dict:
    """Fingerprint small metadata plus the exact LoRA weights when present."""
    root_path = Path(root).resolve()
    records: list[dict] = []
    metadata_names = (
        "config.json",
        "tokenizer_config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "adapter_config.json",
    )
    for name in metadata_names:
        path = root_path / name
        if path.is_file():
            records.append({
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            })

    adapter_path = root_path / "adapter_model.safetensors"
    if include_adapter_weights and adapter_path.is_file():
        records.append({
            "path": adapter_path.name,
            "bytes": adapter_path.stat().st_size,
            "sha256": _sha256_file(adapter_path),
        })

    # Hashing every dense-model shard in each evaluation worker would add large
    # redundant I/O. The index is content-hashed above; shard names and sizes
    # still make accidental path reuse visible in the manifest.
    shard_records = [
        {"path": path.name, "bytes": path.stat().st_size}
        for path in sorted(root_path.glob("model-*.safetensors"))
    ]
    payload = {"files": records, "weight_shards": shard_records}
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _evaluator_source_fingerprints() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    sources = (
        Path(__file__).resolve(),
        repo_root / "src" / "eval" / "unified_benchmark.py",
        repo_root / "src" / "eval" / "math_scoring.py",
        repo_root / "src" / "reward" / "outcome_reward.py",
    )
    return {
        str(path.relative_to(repo_root)): _sha256_file(path)
        for path in sources
    }


_LOCAL_BENCHMARK_FILES = {
    "gsm8k": "data/benchmarks/GSM8K/test.jsonl",
    "math500": "data/benchmarks/MATH-500/test.jsonl",
    "math_500": "data/benchmarks/MATH-500/test.jsonl",
    "aime2024": "data/benchmarks/AIME2024/train.jsonl",
    "aime2025": "data/benchmarks/AIME2025/train.jsonl",
    "aime2026": "data/benchmarks/AIME2026/train.jsonl",
    "olympiadbench": "data/benchmarks/OlympiadBench/test_en_oe_to_math.jsonl",
    "gpqa_diamond": "data/benchmarks/GPQA_Diamond/test.jsonl",
    "mmlu": "data/benchmarks/MMLU/test.jsonl",
}


def _benchmark_source_fingerprint(bench: str) -> dict:
    """Identify the exact local benchmark artifact consumed by a formal run."""
    relative = _LOCAL_BENCHMARK_FILES.get(bench)
    if relative is None:
        return {"kind": "registered_loader", "benchmark": bench}
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / relative
    if not path.is_file():
        return {"kind": "registered_loader", "benchmark": bench, "local_path": relative}
    with path.open("rb") as handle:
        row_count = sum(1 for line in handle if line.strip())
    return {
        "kind": "local_jsonl",
        "path": relative,
        "rows": row_count,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _strip_answer_tag(text: str) -> str:
    """Prefer <answer>...</answer> block content if present."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def extract_number(text: str) -> str | None:
    """Extract the final numeric answer from a math-style response.

    Priority: <answer> tag > #### > \\boxed{} > explicit answer phrases > fallback.
    The returned value may be a symbolic expression (e.g. "\\frac{14}{3}").
    """
    text = _strip_answer_tag(text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)

    candidates: list[str] = []
    # GSM8K #### gold format
    m = re.search(r"####\s*([^\n]+)", text)
    if m:
        candidates.append(m.group(1).strip())
    # MATH \boxed{...}; use the same balanced-brace extraction as training.
    boxed = OutcomeReward._extract_last_boxed(text)
    if boxed is not None:
        candidates.append(boxed)
    # Phrase patterns ("The answer is 42.", "answer = 42")
    for pat in (
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:=]?\s*([^\n;]+)",
        r"[Aa]nswer\s*[:=]\s*([^\n;]+)",
    ):
        m = re.search(pat, text)
        if m:
            candidates.append(m.group(1).strip())
    # Last symbolic/number-like token anywhere
    exprs = re.findall(
        r"\\frac\{[^{}]+\}\{[^{}]+\}|"
        r"\([^\(\)\n]*,[^\(\)\n]*\)|"
        r"[+-]?\d[\d,]*\.?\d*(?:/\d+)?",
        text,
    )
    if exprs:
        candidates.append(exprs[-1].strip())
    for c in candidates:
        # Keep tuple separators while still normalizing thousands separators.
        out = re.sub(r"(?<=\d),(?=\d)", "", c).strip().rstrip(".")
        if out:
            return out
    return None


def normalize_answer(ans: str) -> str:
    """Light normalization for comparing numeric or short-form answers."""
    if ans is None:
        return ""
    a = str(ans).strip()
    a = a.replace(",", "").replace("$", "").replace("%", "").replace("\\,", "")
    # Strip surrounding whitespace and quotes
    a = a.strip().strip('"').strip("'")
    if a.endswith("."):
        a = a[:-1]
    # Fraction a/b -> decimal if possible
    m = re.fullmatch(r"([+-]?\d+)/(\d+)", a)
    if m:
        try:
            a = str(float(m.group(1)) / float(m.group(2)))
        except Exception:
            pass
    # Drop trailing zeros like "42.0" -> "42"
    try:
        f = float(a)
        if f == int(f):
            a = str(int(f))
        else:
            a = f"{f:.6f}".rstrip("0").rstrip(".")
    except Exception:
        pass
    return a.lower()


def answers_match_numeric(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return OutcomeReward._verify_equivalence(str(pred), str(gold))
    except Exception:
        return normalize_answer(pred) == normalize_answer(gold)


def answers_match_math_response(response: str, gold: str) -> bool:
    try:
        return OutcomeReward.verify_math_response(response, gold)
    except Exception:
        return False


def extract_mcq(text: str) -> str | None:
    """Extract multi-choice (A/B/C/D/E) answer from model output.

    Order:
      1) <answer>X</answer> block content (sft_style outputs land here).
      2) Explicit phrase/boxed patterns.
      3) Last single-letter token A-E.

    NOTE (TODO 2026-05-11): When sft_style=True the model wraps reasoning in
    <think>...</think><answer>...</answer>; if the <answer> block contains a
    full sentence like "The answer is A.", we now recurse so the explicit-
    pattern stage catches it. MMLU still has corner cases (multiple letters
    cited inside reasoning when <answer> is absent), tracked in
    docs/2026-05-11-experiment-resync.md.
    """
    answer_block = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_block:
        block = answer_block.group(1).strip()
        for pat in (
            r"^\s*\(?([A-E])\)?\s*$",
            r"\\boxed\{\s*([A-E])\s*\}",
            r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:=]?\s*\(?([A-E])\)?",
            r"[Aa]nswer\s*[:=]\s*\(?([A-E])\)?",
            r"\b([A-E])\b",
        ):
            m = re.search(pat, block, re.MULTILINE)
            if m:
                return m.group(1).upper()

    text = _strip_answer_tag(text)
    # Search concluding patterns from the tail first (last ~400 chars),
    # so a CoT listing like "A is wrong... the answer is C" picks C not A.
    tail = text[-400:] if len(text) > 400 else text
    for pat in (
        r"\\boxed\{\s*([A-E])\s*\}",
        r"[Ff]inal\s+answer\s*(?:is)?\s*[:=]?\s*\(?([A-E])\)?",
        r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:=]?\s*\(?([A-E])\)?",
        r"[Aa]nswer\s*[:=]\s*\(?([A-E])\)?",
        r"(?:choose|pick|select|option)\s+\(?([A-E])\)?",
    ):
        matches = list(re.finditer(pat, tail, re.MULTILINE))
        if matches:
            return matches[-1].group(1).upper()
    for pat in (
        r"\\boxed\{\s*([A-E])\s*\}",
        r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:=]?\s*\(?([A-E])\)?",
        r"[Aa]nswer\s*[:=]\s*\(?([A-E])\)?",
        r"^\s*\(?([A-E])\)?\s*$",
        r"\b([A-E])\)\s",
    ):
        matches = list(re.finditer(pat, text, re.MULTILINE))
        if matches:
            return matches[-1].group(1).upper()
    # As a last resort, take the LAST A-E letter in the output rather than
    # the first, since reasoning traces often restate wrong options earlier.
    letters = re.findall(r"\b([A-E])\b", text)
    return letters[-1].upper() if letters else None


def answers_match_mcq(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    g = str(gold).strip().upper()
    # Gold may be index (0-3) or letter A-E
    if g in list("ABCDE"):
        return pred.upper() == g
    try:
        idx = int(g)
        letter = "ABCDE"[idx] if 0 <= idx < 5 else None
        return letter is not None and pred.upper() == letter
    except Exception:
        pass
    return pred.upper() == g


def answers_match_text(pred: str | None, gold: str) -> bool:
    """Lenient text matcher for code/output style tasks."""
    if pred is None:
        return False
    p = " ".join(str(pred).strip().split()).lower()
    g = " ".join(str(gold).strip().split()).lower()
    if not p or not g:
        return False
    return p == g or g in p


def answer_extractor_for_benchmark(bench: str):
    if bench in {"gsm8k", "math500", "olympiadbench", "omni_math",
                 "aime2024", "aime2025", "aime2026", "cnmo2024"}:
        return extract_number
    if bench in {"mmlu", "gpqa_diamond"}:
        return extract_mcq
    if bench == "livecode":
        return lambda x: x
    return extract_number


def matcher_for_benchmark(bench: str):
    if bench in {"mmlu", "gpqa_diamond"}:
        return answers_match_mcq
    if bench == "livecode":
        return answers_match_text
    return answers_match_numeric


def raw_matcher_for_benchmark(bench: str):
    if bench in {
        "gsm8k", "math500", "math_500", "olympiadbench", "omni_math",
        "aime2024", "aime2025", "aime2026", "cnmo2024",
    }:
        return answers_match_math_response
    return None


# ---------------------------------------------------------------------------
# pass@k / maj@k / prm@k
# ---------------------------------------------------------------------------

def compute_prm_at_k(correct_flags_per_item: list[list[bool]],
                     prm_scores_per_item: list[list[float]], k: int) -> float:
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


# ---------------------------------------------------------------------------
# Benchmark loaders
# ---------------------------------------------------------------------------

def load_gsm8k() -> list[dict]:
    local = Path("data/benchmarks/GSM8K/test.jsonl")
    if local.exists():
        return [{
            "question": str(r.get("Problem", r.get("question", ""))),
            "gold": str(r.get("Answer", r.get("answer", ""))),
            "source": "gsm8k",
        } for r in _load_jsonl(local)]
    ds = load_dataset(
        "openai/gsm8k", "main", split="test",
        cache_dir="/root/.cache/huggingface/datasets",
    )
    items = []
    for row in ds:
        q = row["question"]
        m = re.search(r"####\s*(.+)", row["answer"])
        gold = m.group(1).strip() if m else row["answer"].strip()
        items.append({"question": q, "gold": gold, "source": "gsm8k"})
    return items


def load_math500() -> list[dict]:
    local = Path("data/benchmarks/MATH-500/test.jsonl")
    if local.exists():
        return [{
            "question": str(r.get("Problem", r.get("problem", ""))),
            "gold": str(r.get("Answer", r.get("answer", ""))),
            "source": "math500",
        } for r in _load_jsonl(local)]
    ds = load_dataset(
        "HuggingFaceH4/MATH-500", split="test",
        cache_dir=os.environ.get("HF_DATASETS_CACHE", "/root/.cache/huggingface/datasets"),
    )
    items = []
    for row in ds:
        q = row.get("problem", row.get("question", ""))
        gold = row.get("answer", row.get("solution", ""))
        items.append({"question": q, "gold": gold, "source": "math500"})
    return items


def _safe_load_dataset(hf_path, *, name=None, split=None, trust_remote_code=False):
    """Wrap load_dataset with multiple fallbacks."""
    last_exc = None
    for s in [split] if split else [None]:
        for n in ([name] if name else [None, "default", "all"]):
            try:
                kwargs = {
                    "cache_dir": "/root/.cache/huggingface/datasets",
                    "trust_remote_code": trust_remote_code,
                }
                if n:
                    return load_dataset(hf_path, n, split=s, **kwargs)
                else:
                    return load_dataset(hf_path, split=s, **kwargs)
            except Exception as e:
                last_exc = e
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed to load {hf_path}")


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_aime(year: str) -> list[dict]:
    """Load an AIME yearly set. Prefers local files and falls back to HF."""
    source_name = f"aime{year}"
    local = Path(f"data/benchmarks/AIME{year}/train.jsonl")
    if local.exists():
        rows = _load_jsonl(local)
        items = []
        for r in rows:
            q = r.get("Problem", r.get("problem", r.get("question", "")))
            gold = r.get("Answer", r.get("answer", ""))
            items.append({"question": str(q), "gold": str(gold), "source": source_name})
        return items
    # HF fallback (may be slow, keep as last resort)
    try:
        ds = _safe_load_dataset("AI-MO/aimo-validation-aime", split="train")
        items = []
        for row in ds:
            row_id = str(row.get("id", ""))
            if year not in row_id:
                continue
            q = row.get("problem", row.get("question", ""))
            gold = row.get("answer", "")
            items.append({"question": str(q), "gold": str(gold), "source": source_name})
        if items:
            return items
    except Exception:
        pass
    raise ValueError(f"aime{year} loader failed (no local + HF unavailable)")


def load_cnmo() -> list[dict]:
    """Load CNMO 2024 only when the actual benchmark is available locally."""
    for name in ("CNMO2024", "cnmo2024"):
        for fn in ("train.jsonl", "test.jsonl"):
            p = Path(f"data/benchmarks/{name}/{fn}")
            if p.exists():
                rows = _load_jsonl(p)
                return [{
                    "question": str(r.get("Problem", r.get("problem", r.get("question", "")))),
                    "gold": str(r.get("Answer", r.get("answer", ""))),
                    "source": "cnmo2024",
                } for r in rows]
    local = Path("data/benchmarks/cnmo2024.jsonl")
    if local.exists():
        rows = _load_jsonl(local)
        return [{
            "question": str(r.get("problem", r.get("question", ""))),
            "gold": str(r.get("answer", "")),
            "source": "cnmo2024",
        } for r in rows]
    return []


def load_olympiadbench() -> list[dict]:
    """Load the official 675-item English text-only math subset."""
    local = Path("data/benchmarks/OlympiadBench/test_en_oe_to_math.jsonl")
    if local.exists():
        rows = _load_jsonl(local)
    else:
        ds = _safe_load_dataset("lmms-lab/OlympiadBench", split="test_en")
        rows = [dict(row) for row in ds]

    items = []
    for row in rows:
        if row.get("source") != "OE_TO_maths_en_COMP":
            continue
        gold = row.get("final_answer", row.get("Answer", row.get("answer", "")))
        if isinstance(gold, list):
            gold = gold[0] if len(gold) == 1 else ""
        question = row.get("question", row.get("Problem", row.get("problem", "")))
        if question and str(gold).strip():
            items.append({
                "question": str(question),
                "gold": str(gold),
                "source": "olympiadbench",
            })
    if len(items) != 675:
        raise ValueError(
            f"Expected 675 official OE_TO_maths_en_COMP items, found {len(items)}"
        )
    return items


def load_omni_math() -> list[dict]:
    """Load Omni-MATH itself; never substitute a MATH difficulty slice."""
    local = Path("data/benchmarks/Omni-MATH/test.jsonl")
    if local.exists():
        return [{
            "question": str(r.get("Problem", r.get("problem", ""))),
            "gold": str(r.get("Answer", r.get("answer", ""))),
            "source": "omni_math",
        } for r in _load_jsonl(local)]
    try:
        ds = _safe_load_dataset("KbsdJames/Omni-MATH", split="test")
        items = []
        for row in ds:
            q = row.get("problem", "")
            gold = row.get("answer", "")
            items.append({"question": str(q), "gold": str(gold), "source": "omni_math"})
        return items
    except Exception:
        return []


def _format_mmlu_q(row) -> tuple[str, str]:
    choices = row.get("choices", [])
    if not choices and "options" in row:
        choices = row["options"]
    q = row.get("question") or row.get("Problem") or row.get("problem") or ""
    opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    q_text = f"{q}\n\n{opts}"
    gold = row.get("answer")
    if gold is None:
        gold = row.get("Answer", "")
    if isinstance(gold, int):
        gold = "ABCDE"[gold] if 0 <= gold < 5 else str(gold)
    return q_text, str(gold)


def load_mmlu(subset_ratio: float = 1.0) -> list[dict]:
    # Prefer local
    local = Path("data/benchmarks/MMLU/test.jsonl")
    if local.exists():
        rows = _load_jsonl(local)
        items = []
        for row in rows:
            q_text, gold = _format_mmlu_q(row)
            items.append({"question": q_text, "gold": gold, "source": "mmlu"})
        if subset_ratio < 1.0:
            import random
            random.seed(42)
            n = max(1, int(len(items) * subset_ratio))
            items = random.sample(items, n)
        return items
    ds = _safe_load_dataset("cais/mmlu", name="all", split="test")
    items = []
    for row in ds:
        q_text, gold = _format_mmlu_q(row)
        items.append({"question": q_text, "gold": gold, "source": "mmlu"})
    if subset_ratio < 1.0:
        import random
        random.seed(42)
        n = max(1, int(len(items) * subset_ratio))
        items = random.sample(items, n)
    return items


def load_gpqa_diamond() -> list[dict]:
    def _to_mcq(rows):
        """Convert raw GPQA rows into shuffled ABCD multiple-choice items."""
        import random
        items = []
        for r in rows:
            q = str(
                r.get("question")
                or r.get("Question")
                or r.get("Problem")
                or r.get("problem")
                or ""
            )
            correct = str(r.get("answer", r.get("Correct Answer", "")))
            opts = [
                correct,
                str(r.get("Incorrect Answer 1", "")),
                str(r.get("Incorrect Answer 2", "")),
                str(r.get("Incorrect Answer 3", "")),
            ]
            rng = random.Random(hash(q) & 0xFFFF)
            order = list(range(4))
            rng.shuffle(order)
            letters = "ABCD"
            shuffled = [opts[i] for i in order]
            gold_letter = letters[order.index(0)]
            opt_text = "\n".join(f"{letters[i]}. {shuffled[i]}" for i in range(4))
            q_text = f"{q}\n\n{opt_text}"
            items.append({"question": q_text, "gold": gold_letter, "source": "gpqa_diamond"})
        return items

    local = Path("data/benchmarks/GPQA_Diamond/test.jsonl")
    if local.exists():
        rows = _load_jsonl(local)
        return _to_mcq(rows)
    for path, name, split in [
        ("Idavidrein/gpqa", "gpqa_diamond", "train"),
    ]:
        try:
            ds = _safe_load_dataset(path, name=name, split=split)
            return _to_mcq([dict(row) for row in ds])
        except Exception:
            continue
    return []


def load_livecode() -> list[dict]:
    """LiveCode is permanently disabled (2026-04-21).

    Our TopoPRM models are trained purely for math reasoning. Empirical
    validation (see docs/exp_roadmap_2026-04-20.md) confirmed that every
    candidate model scores 0% on LiveCode under strict text-match scoring,
    so we no longer report this column in the main paper. The loader now
    returns [] unconditionally so any previously-launched shell pipeline
    that still has ``livecode`` in its benches list skips it instantly.
    """
    return []

    # (legacy loader retained below for reference; unreachable.)
    local_paths = [
        Path("data/benchmarks/LiveCode/test.jsonl"),
        Path("data/benchmarks/LiveCode/train.jsonl"),
        Path("data/benchmarks/livecodebench/test.jsonl"),
        Path("data/benchmarks/livecodebench/train.jsonl"),
        Path("data/benchmarks/livecode/test.jsonl"),
        Path("data/benchmarks/livecode/train.jsonl"),
    ]
    for p in local_paths:
        if p.exists():
            rows = _load_jsonl(p)
            return [{
                "question": str(r.get("question_content", r.get("question", ""))),
                "gold": str(r.get("expected_output", r.get("answer", r.get("test", ""))))[:120],
                "source": "livecode",
            } for r in rows]

    # HF fallback: keep best-effort to avoid "No data for benchmark livecode".
    for hf_path, split in [
        ("livecodebench/code_generation_lite", "test"),
        ("livecodebench/code_generation_lite", "train"),
    ]:
        try:
            ds = _safe_load_dataset(hf_path, split=split, trust_remote_code=True)
            items = []
            for r in ds:
                q = r.get("question_content", r.get("question", r.get("prompt", "")))
                gold = r.get("expected_output", r.get("answer", r.get("test", "")))
                if not str(q).strip():
                    continue
                items.append({
                    "question": str(q),
                    "gold": str(gold)[:120],
                    "source": "livecode",
                })
            if items:
                return items
        except Exception:
            continue
    return []


def load_benchmark(bench: str) -> list[dict]:
    loaders = {
        "gsm8k": load_gsm8k,
        "math500": load_math500,
        "math_500": load_math500,
        "aime2024": lambda: load_aime("2024"),
        "aime2025": lambda: load_aime("2025"),
        "aime2026": lambda: load_aime("2026"),
        "cnmo2024": load_cnmo,
        "olympiadbench": load_olympiadbench,
        "omni_math": load_omni_math,
        "mmlu": load_mmlu,
        "gpqa_diamond": load_gpqa_diamond,
        "livecode": load_livecode,
    }
    if bench not in loaders:
        raise ValueError(f"Unknown benchmark: {bench}")
    return loaders[bench]()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

FEWSHOT_GSM8K = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold "
        "half as many clips in May. How many clips did Natalia sell altogether "
        "in April and May?",
        "Natalia sold 48 clips in April. In May, she sold 48/2 = 24 clips. "
        "In total, she sold 48 + 24 = 72 clips. #### 72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
        "minutes of babysitting. How much did she earn?",
        "50 minutes is 50/60 hours. She earned 12 * 50/60 = 10. #### 10",
    ),
]

FEWSHOT_MATH = [
    (
        "What is the value of $2^3 + 3^2$?",
        "We have $2^3 = 8$ and $3^2 = 9$. Therefore $2^3 + 3^2 = 8 + 9 = 17$. "
        "The final answer is $\\boxed{17}$.",
    ),
]


def build_text_prompt(question: str, source: str, fewshot: bool = False) -> str:
    """Bare-text prompt (legacy path, no chat template)."""
    if source == "gsm8k":
        prefix = (
            "Solve the following math problem step by step. "
            "Put your final answer after ####.\n\n"
        )
        if fewshot:
            shots = "".join(
                f"Question: {q}\nAnswer: {a}\n\n" for q, a in FEWSHOT_GSM8K
            )
            return prefix + shots + f"Question: {question}\n\nAnswer:"
        return prefix + f"Question: {question}\n\nAnswer:"

    if source in {"mmlu", "gpqa_diamond"}:
        return (
            "Answer the following multiple choice question. "
            "End your response with 'The answer is X.' where X is one of A, B, C, or D.\n\n"
            f"Question: {question}\n\nAnswer:"
        )

    if source == "livecode":
        return (
            "Solve the following programming problem. Provide a Python solution.\n\n"
            f"Problem: {question}\n\nSolution:"
        )

    # Default math
    prefix = (
        "Solve the following math problem. "
        "Put your final answer in \\boxed{}.\n\n"
    )
    if fewshot:
        shots = "".join(f"Problem: {q}\nSolution: {a}\n\n" for q, a in FEWSHOT_MATH)
        return prefix + shots + f"Problem: {question}\n\nSolution:"
    return prefix + f"Problem: {question}\n\nSolution:"


def build_chat_messages(
    question: str,
    source: str,
    *,
    sft_style: bool,
    fewshot: bool,
    system_control: str = "",
    empty_system_prompt: bool = False,
    user_suffix: str = "",
) -> list[dict]:
    """Build messages with one task instruction across matched checkpoints."""
    if source in {"mmlu", "gpqa_diamond"}:
        sys_msg = (
            "Answer the multiple choice question and end with "
            "'The answer is X.' where X is the option letter."
        )
    else:
        sys_msg = (
            "Solve the math problem step by step. Put the final answer in \\boxed{}."
        )

    system_content = "" if empty_system_prompt else (system_control or sys_msg)
    messages = [{"role": "system", "content": system_content}]

    # Few-shot demonstrations for GSM8K / MATH
    if fewshot and source == "gsm8k":
        for q, a in FEWSHOT_GSM8K:
            messages.append({"role": "user", "content": q})
            if sft_style:
                messages.append({
                    "role": "assistant",
                    "content": f"<think>{a.split('####')[0].strip()}</think><answer>{a.split('####')[-1].strip()}</answer>",
                })
            else:
                messages.append({"role": "assistant", "content": a})
    elif fewshot and source == "math500":
        for q, a in FEWSHOT_MATH:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

    move_instruction_to_user = bool(system_control) or empty_system_prompt
    final_question = f"{sys_msg}\n\n{question}" if move_instruction_to_user else question
    final_question += user_suffix
    messages.append({"role": "user", "content": final_question})
    return messages


def _fold_system_into_user(messages: list[dict]) -> list[dict]:
    """Return a copy of messages with any leading system message merged into the
    first user turn. Used for chat templates that do not support a system role
    (e.g. Gemma). The instruction content is preserved verbatim."""
    if not messages or messages[0].get("role") != "system":
        return messages
    sys_content = messages[0]["content"]
    rest = messages[1:]
    folded: list[dict] = []
    injected = False
    for m in rest:
        if not injected and m.get("role") == "user":
            folded.append({"role": "user", "content": f"{sys_content}\n\n{m['content']}"})
            injected = True
        else:
            folded.append(m)
    if not injected:  # no user turn at all; prepend as a user message
        folded.insert(0, {"role": "user", "content": sys_content})
    return folded


def _prompt_profile(
    *,
    use_chat_template: bool,
    fold_system_into_user: bool,
    system_control: str,
    empty_system_prompt: bool,
    user_suffix: str,
) -> str:
    if not use_chat_template:
        return "bare_text"
    if fold_system_into_user:
        return "task_user"
    if empty_system_prompt:
        return "empty_system_task_user"
    if system_control:
        return "control_system_task_user"
    if user_suffix:
        return "task_system_user_suffix"
    return "task_system"


def _response_envelope(rendered_prompt: str, *, force_think_prefix: bool) -> str:
    """Classify the response exactly as it is saved and token-counted."""
    if force_think_prefix:
        return "full_think"
    if re.search(r"<think>\s*$", rendered_prompt):
        return "prefilled_think"
    return "full_think"


# ---------------------------------------------------------------------------
# Adapter namespace patch (swift -> transformers)
# ---------------------------------------------------------------------------

def patch_swift_adapter_namespace(adapter_dir: Path) -> Path:
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

    safetensor_path = patched_dir / "adapter_model.safetensors"
    if safetensor_path.exists():
        state = safe_load_file(str(safetensor_path))
        needs_patch = any(".language_model." in k for k in state.keys())
        if needs_patch:
            patched = {k.replace(".language_model.", "."): v for k, v in state.items()}
            safe_save_file(patched, str(safetensor_path))
            print(f"  Fixed adapter tensor namespace in {safetensor_path.name}")

    bin_path = patched_dir / "adapter_model.bin"
    if bin_path.exists():
        state = torch.load(bin_path, map_location="cpu")
        if isinstance(state, dict) and any(".language_model." in k for k in state.keys()):
            patched = {k.replace(".language_model.", "."): v for k, v in state.items()}
            torch.save(patched, bin_path)
            print(f"  Fixed adapter tensor namespace in {bin_path.name}")

    return patched_dir


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

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
    *,
    use_chat_template: bool = False,
    sft_style: bool = False,
    fewshot: bool = False,
    pass1_do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    eval_seed: int = 0,
    fold_system_into_user: bool = False,
    force_think_prefix: bool = False,
    system_control: str = "",
    empty_system_prompt: bool = False,
    user_suffix: str = "",
    use_kv_cache: bool = True,
    score_topology: bool = False,
    save_solutions: bool = False,
) -> tuple[dict, list[dict]]:
    if k_values is None:
        k_values = [1, 5]

    total = len(items)
    results = []
    predictions_per_item: list[list[str]] = [[] for _ in range(total)]
    token_counts_per_item: list[list[int]] = [[] for _ in range(total)]
    extractor = answer_extractor_for_benchmark(bench_name)
    matcher = matcher_for_benchmark(bench_name)
    raw_matcher = raw_matcher_for_benchmark(bench_name)
    topo_reward = None
    if score_topology:
        from src.reward.topo_reward import TopoReward

        topo_reward = TopoReward()
    response_prefix = "<think>\n" if force_think_prefix else ""
    prompt_profile = _prompt_profile(
        use_chat_template=use_chat_template,
        fold_system_into_user=fold_system_into_user,
        system_control=system_control,
        empty_system_prompt=empty_system_prompt,
        user_suffix=user_suffix,
    )
    observed_response_envelopes: set[str] = set()

    # Sampling is reproducible for a fixed manifest, including batch size and
    # backend.  We reset once per benchmark rather than once per batch so every
    # item receives a distinct draw from the same seeded stream.
    random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_seed)

    def tokenize_prompts(prompts_text: list[str]) -> dict:
        return tokenizer(
            prompts_text, return_tensors="pt",
            padding=True, truncation=True, max_length=4096,
        ).to(model.device)

    def prepare_prompts(batch: list[dict]) -> list[str]:
        texts = []
        for it in batch:
            if use_chat_template:
                msgs = build_chat_messages(
                    it["question"], it["source"],
                    sft_style=sft_style, fewshot=fewshot,
                    system_control=system_control,
                    empty_system_prompt=empty_system_prompt,
                    user_suffix=user_suffix,
                )
                if fold_system_into_user:
                    msgs = _fold_system_into_user(msgs)
                try:
                    t = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Chat template rejected the configured message roles. "
                        "Use --fold_system_into_user only when the model's "
                        "official evaluation protocol requires user-only instructions."
                    ) from exc
                observed_response_envelopes.add(
                    _response_envelope(
                        t, force_think_prefix=force_think_prefix
                    )
                )
                t += response_prefix
                texts.append(t)
            else:
                observed_response_envelopes.add("plain_text")
                texts.append(build_text_prompt(
                    it["question"], it["source"], fewshot=fewshot,
                ))
        return texts

    for i in range(0, total, batch_size):
        batch = items[i : i + batch_size]
        prompts = prepare_prompts(batch)
        inputs = tokenize_prompts(prompts)

        for sample_idx in range(num_samples_per_item):
            do_sample = pass1_do_sample or sample_idx > 0
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "use_cache": use_kv_cache,
                "pad_token_id": (
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
                "eos_token_id": tokenizer.eos_token_id,
                "repetition_penalty": repetition_penalty,
            }
            if do_sample:
                gen_kwargs.update(
                    {
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "min_p": min_p,
                    }
                )

            try:
                outputs = model.generate(**inputs, **gen_kwargs)
            except Exception as exc:
                print(f"    generate failed batch {i}: {exc}")
                outputs = None

            if outputs is None:
                # Fill empty predictions so indexing stays consistent
                for j in range(len(batch)):
                    predictions_per_item[i + j].append("")
                    token_counts_per_item[i + j].append(0)
                continue

            for j, out_ids in enumerate(outputs):
                prompt_len = inputs["input_ids"][j].shape[0]
                gen_ids = out_ids[prompt_len:]
                gen_text = response_prefix + tokenizer.decode(
                    gen_ids, skip_special_tokens=True
                )
                global_idx = i + j
                predictions_per_item[global_idx].append(gen_text)
                token_counts_per_item[global_idx].append(
                    len(tokenizer.encode(gen_text, add_special_tokens=False))
                )

        done = min(i + batch_size, total)
        pass1_correct = 0
        for idx in range(done):
            if predictions_per_item[idx]:
                pred0 = extractor(predictions_per_item[idx][0])
            else:
                pred0 = None
            is_correct = (
                raw_matcher(predictions_per_item[idx][0], items[idx]["gold"])
                if raw_matcher is not None
                else matcher(pred0, items[idx]["gold"])
            )
            if is_correct:
                pass1_correct += 1
        acc_so_far = pass1_correct / done * 100 if done else 0.0
        print(f"  [{done}/{total}] acc={acc_so_far:.1f}%", flush=True)

    correct_flags_per_item: list[list[bool]] = []
    prm_scores_per_item: list[list[float]] = []
    for idx, item in enumerate(items):
        raw_preds = predictions_per_item[idx]
        extracted = [extractor(p) for p in raw_preds]
        flags = (
            [raw_matcher(raw, item["gold"]) for raw in raw_preds]
            if raw_matcher is not None
            else [matcher(p, item["gold"]) for p in extracted]
        )
        correct_flags_per_item.append(flags)

        prm_scores: list[float] = []
        if topo_reward is not None:
            completion_objs = [[{"role": "assistant", "content": p}] for p in raw_preds]
            prm_scores = topo_reward(completion_objs) if completion_objs else []
        prm_scores_per_item.append(prm_scores)

        results.append({
            "item_id": hashlib.sha256(
                f"{item['source']}\n{item['question']}".encode("utf-8")
            ).hexdigest()[:20],
            "question": item["question"],
            "gold": item["gold"],
            "pred_pass1": extracted[0] if extracted else None,
            "correct_pass1": flags[0] if flags else False,
            "num_samples": len(raw_preds),
            "correct_count": sum(flags),
            "avg_gen_tokens": round(
                sum(token_counts_per_item[idx]) / max(len(token_counts_per_item[idx]), 1), 1
            ),
            "gen_tokens_pass1": token_counts_per_item[idx][0]
            if token_counts_per_item[idx] else 0,
            "response": raw_preds[0] if raw_preds else "",
        })
        if score_topology:
            results[-1]["prm_scores"] = prm_scores
        if save_solutions:
            # Retain every sample for pass@k and structural analysis. The
            # configured pass@1 response is always present above.
            if len(raw_preds) > 1:
                results[-1]["responses_all"] = list(raw_preds)
                results[-1]["correct_flags_all"] = list(flags)
                results[-1]["gen_tokens_all"] = list(token_counts_per_item[idx])

    gold_answers = [str(it["gold"]) for it in items]
    metrics = evaluate_predictions(
        predictions_per_item=predictions_per_item,
        gold_answers=gold_answers,
        answer_extractor=extractor,
        k_values=k_values,
        token_counts=token_counts_per_item,
        answer_matcher=matcher,
        raw_answer_matcher=raw_matcher,
    )

    dataset_payload = "\n".join(
        f"{item['source']}\t{item['question']}\t{item['gold']}" for item in items
    )
    metrics["dataset_fingerprint"] = hashlib.sha256(
        dataset_payload.encode("utf-8")
    ).hexdigest()

    if score_topology:
        for k in k_values:
            metrics[f"prm@{k}"] = round(
                compute_prm_at_k(correct_flags_per_item, prm_scores_per_item, k), 4
            )

    # Override pass@1 with matcher-based computation to handle MCQ correctly
    metrics["pass@1"] = round(
        sum(1 for flags in correct_flags_per_item if flags and flags[0]) / max(total, 1), 4
    )

    metrics["accuracy"] = metrics.get("pass@1", 0.0)
    metrics["accuracy_pct"] = round(metrics["accuracy"] * 100.0, 2)
    metrics["correct"] = int(
        sum(1 for flags in correct_flags_per_item if flags and flags[0])
    )
    metrics["error"] = int(total - metrics["correct"])
    metrics["correct_count"] = metrics["correct"]
    metrics["error_count"] = metrics["error"]
    metrics["num_samples"] = total
    metrics["pass_at_k"] = {str(k): metrics.get(f"pass@{k}", 0.0) for k in k_values}
    metrics["maj_at_k"] = {str(k): metrics.get(f"maj@{k}", 0.0) for k in k_values}
    if score_topology:
        metrics["prm_at_k"] = {
            str(k): metrics.get(f"prm@{k}", 0.0) for k in k_values
        }
    # Treat pass@1 as precision/recall/F1 for these single-answer benchmarks.
    # The generic evaluator's error-tag fields are not a separate task here.
    p1 = metrics["pass@1"]
    metrics["precision"] = p1
    metrics["recall"] = p1
    metrics["f1"] = p1

    bootstrap = bootstrap_item_metrics(results, n_resamples=10_000, seed=0)
    metrics["bootstrap_n_resamples"] = bootstrap["n_resamples"]
    metrics["bootstrap_seed"] = bootstrap["seed"]
    metrics["accuracy_pct_ci95"] = [
        round(value, 4) for value in bootstrap["accuracy_pct_ci95"]
    ]
    metrics["accuracy_pct_wilson95"] = [
        round(value, 4) for value in bootstrap["accuracy_pct_wilson95"]
    ]
    metrics["mean_tokens_pass1"] = round(bootstrap["mean_tokens"], 4)
    metrics["mean_tokens_pass1_ci95"] = [
        round(value, 4) for value in bootstrap["mean_tokens_ci95"]
    ]
    metrics["interval_scope"] = "evaluation_items_not_training_variance"
    if len(observed_response_envelopes) != 1:
        raise RuntimeError(
            "Inconsistent response envelopes across rendered prompts: "
            f"{sorted(observed_response_envelopes)}"
        )
    metrics["prompt_profile"] = prompt_profile
    metrics["response_envelope"] = next(iter(observed_response_envelopes))

    return metrics, results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--label", required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "math500"],
                        help="Benchmarks to evaluate. mmlu is intentionally NOT in the default. "
                             "When using --sft_style, mmlu/gpqa_diamond require a different MCQ "
                             "extractor (TODO: parse <answer>X</answer> blocks); the runner will "
                             "auto-skip mmlu in sft_style mode unless --allow_mmlu_sft_style is set.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--num_samples_per_item", type=int, default=1)
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument(
        "--item_sample_size",
        type=int,
        default=0,
        help="Uniformly sample this many benchmark items without replacement. "
             "Use for frozen diagnostic pools, not canonical full-set evaluation.",
    )
    parser.add_argument("--item_sample_seed", type=int, default=27)
    parser.add_argument("--output_dir", default="output/eval")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Use tokenizer.apply_chat_template for prompting")
    parser.add_argument("--sft_style", action="store_true",
                        help="Use SFT/GRPO-style system prompt with <think>/<answer>")
    parser.add_argument("--fewshot", action="store_true",
                        help="Prepend 1-3 few-shot CoT demonstrations")
    parser.add_argument(
        "--pass1_do_sample",
        action="store_true",
        help="Sample the single pass@1 response. Canonical ICLR evaluation uses "
             "one seed-0 sample per item, not repeated training or decoding runs.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=float(os.environ.get("TOPO_EVAL_REP_PENALTY", "1.0")),
        help="Generation repetition penalty. Canonical pass@1 uses 1.0.",
    )
    parser.add_argument(
        "--fold_system_into_user",
        action="store_true",
        help="Move the identical system instruction into the first user turn. "
             "Use only when required by the model's official protocol.",
    )
    parser.add_argument(
        "--force_think_prefix",
        action="store_true",
        help="Prefill <think> for DeepSeek-R1-family evaluation and include the "
             "prefill in saved responses and completion-token counts.",
    )
    parser.add_argument(
        "--system_control",
        default="",
        help="Optional model-native system control such as '/think'. The common "
             "task instruction is then prepended to the user turn.",
    )
    parser.add_argument(
        "--empty_system_prompt",
        action="store_true",
        help="Keep an explicit empty system turn and move the common task "
             "instruction into the user turn, as required by MiMo-style protocols.",
    )
    parser.add_argument(
        "--user_suffix",
        default="",
        help="Exact model-native suffix appended to the final user question, "
             "for example ' /think' for Nemotron-Cascade.",
    )
    parser.add_argument(
        "--disable_kv_cache",
        action="store_true",
        help="Disable the generation KV cache. Canonical evaluation enables it "
             "explicitly so training-time model configs cannot disable efficient inference.",
    )
    parser.add_argument(
        "--score_topology",
        action="store_true",
        help="Compute TopoPRM reranking scores/prm@k. Disabled for canonical "
             "answer-accuracy evaluation.",
    )
    parser.add_argument(
        "--save_solutions",
        action="store_true",
        help="Also write the full model response (raw <think>...<answer>) to "
        "each row of <label>_<bench>_details.jsonl as 'response'. Required "
        "for tutorials/render_dag_cases.py --from-rollout.",
    )
    parser.add_argument("--force_overwrite", action="store_true",
                        help="Re-run even if <label>_<bench>_metrics.json already exists")
    parser.add_argument("--allow_mmlu_sft_style", action="store_true",
                        help="Override the safety skip for MMLU when sft_style=True. "
                             "Known issue: chat+sft_style on MCQ benches produces verbose <think>/<answer> "
                             "outputs that extract_mcq cannot parse, yielding ~0%% accuracy. "
                             "Until the MCQ extractor is taught to consume <answer>...</answer> blocks, "
                             "we skip mmlu by default in sft_style runs. See "
                             "docs/2026-05-11-experiment-resync.md for the open TODO.")
    args = parser.parse_args()

    if args.repetition_penalty <= 0:
        parser.error("--repetition_penalty must be greater than zero")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero")
    if not 0 < args.top_p <= 1:
        parser.error("--top_p must be in (0, 1]")
    if args.top_k < 0:
        parser.error("--top_k must be non-negative")
    if not 0 <= args.min_p <= 1:
        parser.error("--min_p must be in [0, 1]")
    if args.eval_seed < 0:
        parser.error("--eval_seed must be non-negative")
    if args.item_sample_size < 0 or args.item_sample_seed < 0:
        parser.error("item sample size and seed must be non-negative")
    if args.max_items > 0 and args.item_sample_size > 0:
        parser.error("--max_items and --item_sample_size are mutually exclusive")
    chat_protocol_flags = (
        args.fold_system_into_user
        or args.force_think_prefix
        or args.system_control
        or args.empty_system_prompt
        or args.user_suffix
    )
    if chat_protocol_flags and not args.use_chat_template:
        parser.error(
            "chat-role and model-native prompt controls require --use_chat_template"
        )
    if args.system_control and args.empty_system_prompt:
        parser.error("--system_control and --empty_system_prompt are mutually exclusive")

    def metrics_match_current_protocol(payload: dict, bench: str) -> bool:
        expected = {
            "label": args.label,
            "num_samples_per_item": args.num_samples_per_item,
            "k_values": args.k_values,
            "use_chat_template": args.use_chat_template,
            "fold_system_into_user": args.fold_system_into_user,
            "force_think_prefix": args.force_think_prefix,
            "system_control": args.system_control,
            "empty_system_prompt": args.empty_system_prompt,
            "user_suffix": args.user_suffix,
            "sft_style": args.sft_style,
            "fewshot": args.fewshot,
            "model": str(Path(args.model).resolve()),
            "adapter": str(Path(args.adapter).resolve()) if args.adapter else "",
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "pass1_do_sample": args.pass1_do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "repetition_penalty": args.repetition_penalty,
            "kv_cache_enabled": not args.disable_kv_cache,
            "eval_seed": args.eval_seed,
            "score_topology": args.score_topology,
            "bootstrap_n_resamples": 10_000,
            "bootstrap_seed": 0,
            "pass1_decoding": "sampled" if args.pass1_do_sample else "greedy",
            "provenance_schema_version": 3,
            "visible_token_counting": "retokenized_decoded_response_no_special_tokens",
            "prompt_profile": _prompt_profile(
                use_chat_template=args.use_chat_template,
                fold_system_into_user=args.fold_system_into_user,
                system_control=args.system_control,
                empty_system_prompt=args.empty_system_prompt,
                user_suffix=args.user_suffix,
            ),
        }
        for key, value in expected.items():
            if key == "empty_system_prompt":
                observed = payload.get(key, False)
            elif key == "user_suffix":
                observed = payload.get(key, "")
            else:
                if key not in payload:
                    return False
                observed = payload[key]
            if observed != value:
                return False
        provenance = payload.get("provenance", {})
        if provenance.get("benchmark_source") != _benchmark_source_fingerprint(bench):
            return False
        if not payload.get("response_envelope"):
            return False
        if args.item_sample_size > 0:
            sampling = payload.get("item_sampling", {})
            if sampling.get("method") != "uniform_without_replacement":
                return False
            if sampling.get("sample_items") != args.item_sample_size:
                return False
            if sampling.get("seed") != args.item_sample_seed:
                return False
        return (
            provenance.get("prompt_profile") == payload.get("prompt_profile")
            and provenance.get("response_envelope")
            == payload.get("response_envelope")
        )

    if args.sft_style and not args.allow_mmlu_sft_style:
        mmlu_in = [b for b in args.benchmarks if b == "mmlu"]
        if mmlu_in:
            print("[warn] sft_style=True + benchmark=mmlu is known to extract 0%% with the "
                  "current MCQ extractor; skipping. Pass --allow_mmlu_sft_style to override.")
            args.benchmarks = [b for b in args.benchmarks if b != "mmlu"]
            if not args.benchmarks:
                print("[skip-all] only mmlu was requested and it was filtered out; exiting.")
                return

    # Skip-if-exists pre-filter: drop benches whose metrics.json already exists,
    # unless --force_overwrite is set.  If this empties the list we exit early
    # and never pay the model-load cost.
    if not args.force_overwrite:
        filtered = []
        out_dir_pre = Path(args.output_dir)
        for bench in args.benchmarks:
            mf = out_dir_pre / f"{args.label}_{bench}_metrics.json"
            if mf.exists():
                try:
                    old_metrics = json.loads(mf.read_text(encoding="utf-8"))
                except Exception:
                    old_metrics = {}
                if metrics_match_current_protocol(old_metrics, bench):
                    print(f"[skip-compatible] {mf}")
                    continue
                print(f"[rerun-incompatible] {mf}")
            filtered.append(bench)
        if not filtered:
            print(f"[skip-all] all benchmarks for label={args.label} already saved; nothing to do")
            return
        args.benchmarks = filtered

    provenance = {
        "schema_version": 3,
        "evaluator_sources": _evaluator_source_fingerprints(),
        "model_artifact": _artifact_fingerprint(
            args.model, include_adapter_weights=False
        ),
        "adapter_artifact": (
            _artifact_fingerprint(args.adapter, include_adapter_weights=True)
            if args.adapter else None
        ),
    }

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model_default_use_cache = getattr(model.config, "use_cache", None)

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
        print(f"Benchmark: {bench} | Label: {args.label} | "
              f"chat={args.use_chat_template} sft_style={args.sft_style} "
              f"fold_system={args.fold_system_into_user} "
              f"think_prefix={args.force_think_prefix} "
              f"system_control={args.system_control or '<none>'} "
              f"empty_system={args.empty_system_prompt} "
              f"user_suffix={args.user_suffix or '<none>'} "
              f"fewshot={args.fewshot} k={args.num_samples_per_item} "
              f"pass1={'sampled' if args.pass1_do_sample else 'greedy'} "
              f"seed={args.eval_seed} temp={args.temperature} "
              f"top_p={args.top_p} top_k={args.top_k} min_p={args.min_p} "
              f"rep_penalty={args.repetition_penalty} "
              f"kv_cache={not args.disable_kv_cache} "
              f"topology_scoring={args.score_topology}")
        print(f"{'='*60}")

        try:
            items = load_benchmark(bench)
        except Exception as exc:
            print(f"Unknown/failed benchmark {bench}, skipping: {exc}")
            continue
        if not items:
            print(f"No data for benchmark {bench}, skipping")
            continue
        population_items = len(items)
        if args.item_sample_size > 0:
            if args.item_sample_size > population_items:
                raise ValueError(
                    f"Cannot sample {args.item_sample_size} items from "
                    f"{bench} population of {population_items}"
                )
            item_rng = random.Random(args.item_sample_seed)
            selected_indices = sorted(
                item_rng.sample(range(population_items), args.item_sample_size)
            )
            items = [items[index] for index in selected_indices]
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
            use_chat_template=args.use_chat_template,
            sft_style=args.sft_style,
            fewshot=args.fewshot,
            pass1_do_sample=args.pass1_do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            eval_seed=args.eval_seed,
            fold_system_into_user=args.fold_system_into_user,
            force_think_prefix=args.force_think_prefix,
            system_control=args.system_control,
            empty_system_prompt=args.empty_system_prompt,
            user_suffix=args.user_suffix,
            use_kv_cache=not args.disable_kv_cache,
            score_topology=args.score_topology,
            save_solutions=args.save_solutions,
        )
        elapsed = time.time() - t0

        metrics["elapsed_sec"] = round(elapsed, 1)
        metrics["backend"] = "transformers"
        metrics["label"] = args.label
        metrics["num_samples_per_item"] = args.num_samples_per_item
        metrics["k_values"] = args.k_values
        metrics["use_chat_template"] = args.use_chat_template
        metrics["fold_system_into_user"] = args.fold_system_into_user
        metrics["force_think_prefix"] = args.force_think_prefix
        metrics["system_control"] = args.system_control
        metrics["empty_system_prompt"] = args.empty_system_prompt
        metrics["user_suffix"] = args.user_suffix
        metrics["prompt_role_protocol"] = (
            "user_only" if args.fold_system_into_user else "system_user"
        )
        metrics["sft_style"] = args.sft_style
        metrics["fewshot"] = args.fewshot
        metrics["model"] = str(Path(args.model).resolve())
        metrics["adapter"] = str(Path(args.adapter).resolve()) if args.adapter else ""
        metrics["max_new_tokens"] = args.max_new_tokens
        metrics["batch_size"] = args.batch_size
        metrics["pass1_do_sample"] = args.pass1_do_sample
        metrics["temperature"] = args.temperature
        metrics["top_p"] = args.top_p
        metrics["top_k"] = args.top_k
        metrics["min_p"] = args.min_p
        metrics["repetition_penalty"] = args.repetition_penalty
        metrics["kv_cache_enabled"] = not args.disable_kv_cache
        metrics["model_default_use_cache"] = model_default_use_cache
        metrics["eval_seed"] = args.eval_seed
        metrics["score_topology"] = args.score_topology
        if args.item_sample_size > 0:
            selected_item_ids = sorted(str(row["item_id"]) for row in results)
            metrics["item_sampling"] = {
                "method": "uniform_without_replacement",
                "population_items": population_items,
                "sample_items": len(results),
                "seed": args.item_sample_seed,
                "item_id_set_sha256": hashlib.sha256(
                    "\n".join(selected_item_ids).encode("utf-8")
                ).hexdigest(),
            }
        metrics["pass1_decoding"] = (
            "sampled" if args.pass1_do_sample else "greedy"
        )
        metrics["provenance_schema_version"] = 3
        metrics["visible_token_counting"] = (
            "retokenized_decoded_response_no_special_tokens"
        )
        metrics["math_scoring_protocol"] = (
            "last_nonempty_box_else_explicit_final_answer_math_verify_gold_first"
            if raw_matcher_for_benchmark(bench) is not None
            else "extracted_answer_matcher"
        )
        metrics["provenance"] = {
            **provenance,
            "benchmark_source": _benchmark_source_fingerprint(bench),
            "prompt_profile": metrics["prompt_profile"],
            "response_envelope": metrics["response_envelope"],
            "response_envelope_evidence": {
                "source": "generation_rendered_prompt_suffix",
                "template_ends_with_think": (
                    metrics["response_envelope"] == "prefilled_think"
                ),
                "forced_think_prefix": args.force_think_prefix,
            },
        }

        metrics_path = out_dir / f"{args.label}_{bench}_metrics.json"
        metrics_tmp = metrics_path.with_suffix(metrics_path.suffix + ".partial")
        metrics_tmp.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        metrics_tmp.replace(metrics_path)

        details_path = out_dir / f"{args.label}_{bench}_details.jsonl"
        details_tmp = details_path.with_suffix(details_path.suffix + ".partial")
        with details_tmp.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        details_tmp.replace(details_path)

        print(
            f"\n  {bench}: pass@1={metrics.get('pass@1', 0.0)*100:.1f}% "
            f"pass@5={metrics.get('pass@5', 0.0)*100:.1f}% "
            f"maj@5={metrics.get('maj@5', 0.0)*100:.1f}% "
            f"prm@5={metrics.get('prm@5', float('nan'))*100:.1f}% "
            f"tok={metrics.get('avg_tokens', 0):.0f} "
            f"in {elapsed:.0f}s"
        )
        print(f"  Saved: {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
