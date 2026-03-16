"""Merge Chinese math-grading data with English HuggingFace datasets.

Produces a single shuffled JSONL file that mixes Chinese and English
samples at a configurable ratio.

Usage::

    python -m src.data.merge_datasets \\
        --zh_path     data/sft_ready/train.jsonl \\
        --output_path data/merged/train.jsonl \\
        --en_ratio    0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_HF_DATASETS: List[str] = [
    "AI-MO/NuminaMath-CoT",
    "meta-math/MetaMathQA",
]


# ---------------------------------------------------------------------------
# HuggingFace loader
# ---------------------------------------------------------------------------

def load_hf_dataset(
    name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Load a HuggingFace dataset and convert rows to message dicts.

    Requires the ``datasets`` library.  Each row is normalised into the
    ``{"messages": [...]}`` format used by the rest of the pipeline.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        logger.error(
            "The `datasets` library is required to load HF datasets. "
            "Install it with: pip install datasets"
        )
        return []

    logger.info("Loading HF dataset %s (split=%s) …", name, split)
    try:
        ds = load_dataset(name, split=split, trust_remote_code=True)
    except Exception as exc:
        logger.error("Failed to load %s: %s", name, exc)
        return []

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    records: List[Dict] = []
    columns = set(ds.column_names)

    for row in ds:
        # NuminaMath-CoT style: problem / solution
        if "problem" in columns and "solution" in columns:
            messages = [
                {"role": "system", "content": "You are a helpful math assistant."},
                {"role": "user", "content": row["problem"]},
                {"role": "assistant", "content": row["solution"]},
            ]
        # MetaMathQA style: query / response
        elif "query" in columns and "response" in columns:
            messages = [
                {"role": "system", "content": "You are a helpful math assistant."},
                {"role": "user", "content": row["query"]},
                {"role": "assistant", "content": row["response"]},
            ]
        # Generic fallback
        elif "question" in columns and "answer" in columns:
            messages = [
                {"role": "system", "content": "You are a helpful math assistant."},
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ]
        else:
            logger.debug("Skipping row from %s – unknown schema", name)
            continue

        records.append({"messages": messages, "source": name, "lang": "en"})

    logger.info("Loaded %d records from %s", len(records), name)
    return records


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_datasets(
    zh_path: str,
    output_path: str,
    hf_datasets: Optional[List[str]] = None,
    en_ratio: float = 0.3,
    seed: int = 42,
) -> int:
    """Merge Chinese JSONL with English HF datasets.

    Parameters
    ----------
    zh_path:
        Path to the Chinese SFT JSONL.
    output_path:
        Destination merged JSONL.
    hf_datasets:
        List of HF dataset identifiers.  Defaults to
        :data:`DEFAULT_HF_DATASETS`.
    en_ratio:
        Target fraction of English samples in the output.
    seed:
        Random seed for shuffling.

    Returns
    -------
    int
        Total number of records written.
    """
    hf_datasets = hf_datasets or DEFAULT_HF_DATASETS

    # Load Chinese data
    zh_records: List[Dict] = []
    with open(zh_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rec.setdefault("lang", "zh")
                rec.setdefault("source", "zh_math_grading")
                zh_records.append(rec)
            except json.JSONDecodeError:
                continue

    logger.info("Loaded %d Chinese records from %s", len(zh_records), zh_path)

    # Determine how many English samples we need
    n_zh = len(zh_records)
    if n_zh == 0:
        logger.warning("No Chinese records found – nothing to merge")
        return 0

    n_en_target = int(n_zh * en_ratio / (1.0 - en_ratio)) if en_ratio < 1.0 else n_zh

    # Load English data from HF
    en_records: List[Dict] = []
    per_ds = max(1, n_en_target // len(hf_datasets)) if hf_datasets else 0
    for ds_name in hf_datasets:
        en_records.extend(load_hf_dataset(ds_name, max_samples=per_ds))

    if len(en_records) > n_en_target:
        rng = random.Random(seed)
        en_records = rng.sample(en_records, n_en_target)

    logger.info(
        "Merge plan: %d zh + %d en → target en_ratio=%.2f",
        n_zh,
        len(en_records),
        en_ratio,
    )

    all_records = zh_records + en_records
    random.Random(seed).shuffle(all_records)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for rec in all_records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Wrote %d merged records → %s", len(all_records), output_path)
    return len(all_records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Chinese math data with English HuggingFace datasets."
    )
    parser.add_argument(
        "--zh_path",
        type=str,
        default="data/sft_ready/train.jsonl",
        help="Chinese SFT JSONL.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/merged/train.jsonl",
    )
    parser.add_argument(
        "--en_ratio",
        type=float,
        default=0.3,
        help="Target fraction of English samples (0–1).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hf_datasets",
        nargs="*",
        default=None,
        help="HuggingFace dataset names to use as English sources.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    n = merge_datasets(
        args.zh_path,
        args.output_path,
        hf_datasets=args.hf_datasets,
        en_ratio=args.en_ratio,
        seed=args.seed,
    )
    logger.info("Total merged: %d", n)


if __name__ == "__main__":
    main()
