#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.." && export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

RAW_INPUT_DIR="${1:-/mnt/users/postrain/ms-swift/data/数学0305}"

echo "============================================"
echo "  TopoPRM Data Pipeline"
echo "============================================"

mkdir -p data/{processed,sft_ready,grpo_ready,dag}

# Step 1: Parse raw data
echo ""
echo "[1/6] Parsing raw data..."
python3 -m src.data.parse_raw \
    --input_dir "${RAW_INPUT_DIR}" \
    --output_path data/processed/parsed.jsonl
echo "  -> data/processed/parsed.jsonl"

# Step 2: Clean data
echo ""
echo "[2/6] Cleaning data..."
python3 -m src.data.clean \
    --input_path data/processed/parsed.jsonl \
    --output_path data/processed/cleaned.jsonl
echo "  -> data/processed/cleaned.jsonl"

# Step 3: Build DAG
echo ""
echo "[3/6] Building DAG structures..."
python3 -m src.data.build_dag \
    --input_path data/processed/cleaned.jsonl \
    --output_dir data/dag
echo "  -> data/dag/"

# Step 4: Prepare SFT data
echo ""
echo "[4/6] Preparing SFT training data..."
python3 -m src.data.prepare_sft \
    --input_path data/processed/cleaned.jsonl \
    --output_path data/sft_ready/train.jsonl
echo "  -> data/sft_ready/train.jsonl"

# Step 5: Merge datasets
echo ""
echo "[5/6] Merging datasets (ZH + EN mix)..."
python3 -m src.data.merge_datasets \
    --zh_path data/sft_ready/train.jsonl \
    --output_path data/sft_ready/train_mixed.jsonl
echo "  -> data/sft_ready/train_mixed.jsonl"

# Step 6: Prepare GRPO data
echo ""
echo "[6/6] Preparing GRPO training data..."
python3 -m src.data.prepare_grpo \
    --input_path data/processed/cleaned.jsonl \
    --dag_dir data/dag \
    --output_path data/grpo_ready/train.jsonl
echo "  -> data/grpo_ready/train.jsonl"

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "============================================"
echo ""
echo "Summary:"
echo "  Parsed:    data/processed/parsed.jsonl"
echo "  Cleaned:   data/processed/cleaned.jsonl"
echo "  DAG:       data/dag/"
echo "  SFT data:  data/sft_ready/train.jsonl"
echo "  SFT mixed: data/sft_ready/train_mixed.jsonl"
echo "  GRPO data: data/grpo_ready/train.jsonl"
