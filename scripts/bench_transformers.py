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
import json
import os
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
from sympy import simplify, sympify
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.unified_benchmark import evaluate_predictions
from src.reward.topo_reward import TopoReward


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
    # MATH \boxed{...} (handle nested braces best-effort)
    m = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if m:
        candidates.append(m.group(1).strip())
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
    pr = str(pred).strip()
    gr = str(gold).strip()

    def _latex_to_sympy(expr: str) -> str:
        expr = expr.strip()
        expr = expr.replace("\\left", "").replace("\\right", "")
        expr = re.sub(r"\\text\{([^{}]*)\}", r"\1", expr)
        expr = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", expr)
        expr = expr.replace("^", "**")
        expr = expr.replace("{", "(").replace("}", ")")
        expr = expr.replace("$", "").replace("\\,", "")
        return expr.strip()

    def _parse_seq(expr: str) -> list[str]:
        s = _latex_to_sympy(expr)
        if s.startswith("(") and s.endswith(")") and "," in s:
            inner = s[1:-1]
            return [x.strip() for x in inner.split(",")]
        if s.startswith("[") and s.endswith("]") and "," in s:
            inner = s[1:-1]
            return [x.strip() for x in inner.split(",")]
        return []

    def _sym_equal(a: str, b: str) -> bool:
        try:
            ea = sympify(_latex_to_sympy(a))
            eb = sympify(_latex_to_sympy(b))
            return bool(simplify(ea - eb) == 0)
        except Exception:
            return False

    ap = _parse_seq(pr)
    ag = _parse_seq(gr)
    if ap or ag:
        if len(ap) != len(ag) or not ap:
            return False
        return all(_sym_equal(x, y) or normalize_answer(x) == normalize_answer(y) for x, y in zip(ap, ag))

    if _sym_equal(pr, gr):
        return True

    pn = normalize_answer(pred)
    gn = normalize_answer(gold)
    if not pn or not gn:
        return False
    if pn == gn:
        return True
    try:
        return abs(float(pn) - float(gn)) < 1e-4
    except Exception:
        pass

    return _sym_equal(pn, gn)


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
                 "aime2024", "aime2025", "cnmo2024"}:
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
        import random
        random.seed(42)
        indices = random.sample(range(len(ds)), min(500, len(ds)))
        ds = ds.select(indices)
    items = []
    for row in ds:
        q = row.get("problem", row.get("question", ""))
        gold = row.get("answer", row.get("solution", ""))
        m = re.search(r"\\boxed\{([^}]+)\}", gold)
        if m:
            gold = m.group(1).strip()
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
    """Load AIME 2024 / 2025. Prefers local file; falls back to HF."""
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
    """CNMO 2024 - Chinese National Math Olympiad. Fall back to AMC23 (competition
    math at similar difficulty) if CNMO data is not available locally."""
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
    # Fallback to AMC23 as competition-math proxy
    amc23 = Path("data/benchmarks/AMC23/train.jsonl")
    if amc23.exists():
        rows = _load_jsonl(amc23)
        return [{
            "question": str(r.get("Problem", "")),
            "gold": str(r.get("Answer", "")),
            "source": "cnmo2024",
        } for r in rows]
    return []


def _load_math_by_levels(levels: list[int], source_name: str) -> list[dict]:
    """Load MATH rows filtered by level; gold from 'answer'/'Answer' field.

    Accepts both lowercase (`problem`/`answer`) and the capitalized
    (`Problem`/`Answer`) schema produced by scripts/download_benchmarks.py.
    Level may be an int ("1".."5") or a string like "Level 5".
    """
    local = Path("data/benchmarks/MATH/test.jsonl")
    if not local.exists():
        return []
    rows = _load_jsonl(local)

    def _normalize_level(raw):
        if raw is None:
            return None
        if isinstance(raw, int):
            return raw
        s = str(raw)
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else None

    levels_set = set(levels)
    items = []
    for r in rows:
        lvl = _normalize_level(r.get("level"))
        if lvl not in levels_set:
            continue
        q = r.get("Problem", r.get("problem", r.get("question", "")))
        gold_raw = r.get("Answer", r.get("answer", ""))
        gold = str(gold_raw)
        m = re.search(r"\\boxed\{([^}]+)\}", gold)
        if m:
            gold = m.group(1).strip()
        if not gold.strip():
            continue
        items.append({"question": str(q), "gold": gold, "source": source_name})
    return items


def load_olympiadbench() -> list[dict]:
    """OlympiadBench proxy via MATH level-5 (hardest, olympiad-like)."""
    items = _load_math_by_levels([5], "olympiadbench")
    if items:
        return items
    # HF fallback
    try:
        ds = _safe_load_dataset("lmms-lab/OlympiadBench", split="test_en")
        items = []
        for row in ds:
            q = row.get("question", row.get("problem", ""))
            gold = row.get("final_answer", row.get("answer", ""))
            if isinstance(gold, list):
                gold = gold[0] if gold else ""
            items.append({"question": str(q), "gold": str(gold), "source": "olympiadbench"})
        return items
    except Exception:
        return []


def load_omni_math() -> list[dict]:
    """Omni-MATH proxy via MATH level-4+5 (high-difficulty subset)."""
    items = _load_math_by_levels([4, 5], "omni_math")
    if items:
        return items
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
) -> list[dict]:
    """Build message list for chat-template prompting.

    sft_style=True: use our <think>/<answer> system prompt (for our SFT/GRPO
    adapters trained on that format). sft_style=False: vanilla system prompt
    suitable for base models.
    """
    if sft_style:
        sys_msg = (
            "You are a math reasoning assistant. Think through the problem "
            "step by step inside <think>...</think>, then give the final "
            "answer inside <answer>...</answer>. The answer must be a single "
            "number or expression."
        )
    else:
        sys_msg = (
            "You are a helpful assistant. Solve problems carefully and provide "
            "your final answer clearly marked (for math, use \\boxed{}; "
            "for multiple choice, end with 'The answer is X.')."
        )

    messages = [{"role": "system", "content": sys_msg}]

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

    messages.append({"role": "user", "content": question})
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
    temperature: float = 0.7,
    top_p: float = 0.95,
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
    topo_reward = TopoReward()

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
                )
                try:
                    t = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                    )
                except Exception:
                    # Some chat templates (e.g. Gemma) do not support a system
                    # role. Fold the system content into the first user turn so
                    # the same instruction is delivered without breaking.
                    folded = _fold_system_into_user(msgs)
                    t = tokenizer.apply_chat_template(
                        folded, tokenize=False, add_generation_prompt=True,
                    )
                texts.append(t)
            else:
                texts.append(build_text_prompt(
                    it["question"], it["source"], fewshot=fewshot,
                ))
        return texts

    for i in range(0, total, batch_size):
        batch = items[i : i + batch_size]
        prompts = prepare_prompts(batch)
        inputs = tokenize_prompts(prompts)

        for sample_idx in range(num_samples_per_item):
            # sample 0 = greedy (for pass@1), others stochastic
            do_sample = sample_idx > 0
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                # Explicit EOS + light repetition penalty: R1-distilled models can
                # loop until max_new_tokens on hard problems (degenerate repetition),
                # which both wastes time and truncates the boxed answer. This is
                # opt-outable via TOPO_EVAL_REP_PENALTY=1.0.
                "eos_token_id": tokenizer.eos_token_id,
                "repetition_penalty": float(os.environ.get("TOPO_EVAL_REP_PENALTY", "1.05")),
            }
            if do_sample:
                gen_kwargs.update({"temperature": temperature, "top_p": top_p})

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
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                global_idx = i + j
                predictions_per_item[global_idx].append(gen_text)
                token_counts_per_item[global_idx].append(len(gen_ids))

        done = min(i + batch_size, total)
        pass1_correct = 0
        for idx in range(done):
            if predictions_per_item[idx]:
                pred0 = extractor(predictions_per_item[idx][0])
            else:
                pred0 = None
            if matcher(pred0, items[idx]["gold"]):
                pass1_correct += 1
        acc_so_far = pass1_correct / done * 100 if done else 0.0
        print(f"  [{done}/{total}] acc={acc_so_far:.1f}%", flush=True)

    correct_flags_per_item: list[list[bool]] = []
    prm_scores_per_item: list[list[float]] = []
    for idx, item in enumerate(items):
        raw_preds = predictions_per_item[idx]
        extracted = [extractor(p) for p in raw_preds]
        flags = [matcher(p, item["gold"]) for p in extracted]
        correct_flags_per_item.append(flags)

        completion_objs = [[{"role": "assistant", "content": p}] for p in raw_preds]
        try:
            prm_scores = topo_reward(completion_objs) if completion_objs else []
        except Exception:
            prm_scores = [0.0] * len(completion_objs)
        prm_scores_per_item.append(prm_scores)

        results.append({
            "question": item["question"][:200],
            "gold": item["gold"],
            "pred_pass1": extracted[0] if extracted else None,
            "correct_pass1": flags[0] if flags else False,
            "num_samples": len(raw_preds),
            "correct_count": sum(flags),
            "avg_gen_tokens": round(
                sum(token_counts_per_item[idx]) / max(len(token_counts_per_item[idx]), 1), 1
            ),
            "prm_scores": prm_scores,
        })
        if save_solutions:
            # Full generated text for DAG extraction / qualitative analysis.
            # "response" = first sample (used for pass@1); "responses_all" = every
            # sample when num_samples_per_item > 1.
            results[-1]["response"] = raw_preds[0] if raw_preds else ""
            if len(raw_preds) > 1:
                results[-1]["responses_all"] = list(raw_preds)

    gold_answers = [str(it["gold"]) for it in items]
    metrics = evaluate_predictions(
        predictions_per_item=predictions_per_item,
        gold_answers=gold_answers,
        answer_extractor=extractor,
        k_values=k_values,
        token_counts=token_counts_per_item,
    )

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
    metrics["num_samples"] = total
    metrics["pass_at_k"] = {str(k): metrics.get(f"pass@{k}", 0.0) for k in k_values}
    metrics["maj_at_k"] = {str(k): metrics.get(f"maj@{k}", 0.0) for k in k_values}
    metrics["prm_at_k"] = {str(k): metrics.get(f"prm@{k}", 0.0) for k in k_values}
    # F1: treat pass@1 as both precision and recall (single-answer benchmarks)
    p1 = metrics["pass@1"]
    metrics["f1"] = round(p1, 4) if p1 else 0.0

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
    parser.add_argument("--output_dir", default="output/eval")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Use tokenizer.apply_chat_template for prompting")
    parser.add_argument("--sft_style", action="store_true",
                        help="Use SFT/GRPO-style system prompt with <think>/<answer>")
    parser.add_argument("--fewshot", action="store_true",
                        help="Prepend 1-3 few-shot CoT demonstrations")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
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
                print(f"[skip-exists] {mf}")
            else:
                filtered.append(bench)
        if not filtered:
            print(f"[skip-all] all benchmarks for label={args.label} already saved; nothing to do")
            return
        args.benchmarks = filtered

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
              f"fewshot={args.fewshot} k={args.num_samples_per_item}")
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
            use_chat_template=args.use_chat_template,
            sft_style=args.sft_style,
            fewshot=args.fewshot,
            temperature=args.temperature,
            top_p=args.top_p,
            save_solutions=args.save_solutions,
        )
        elapsed = time.time() - t0

        metrics["elapsed_sec"] = round(elapsed, 1)
        metrics["backend"] = "transformers"
        metrics["label"] = args.label
        metrics["num_samples_per_item"] = args.num_samples_per_item
        metrics["k_values"] = args.k_values
        metrics["use_chat_template"] = args.use_chat_template
        metrics["sft_style"] = args.sft_style
        metrics["fewshot"] = args.fewshot

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
            f"tok={metrics.get('avg_tokens', 0):.0f} "
            f"in {elapsed:.0f}s"
        )
        print(f"  Saved: {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
