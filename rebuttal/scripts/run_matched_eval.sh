#!/usr/bin/env bash
# Matched eval for the three TRL-trained DR1-7B GRPO variants.
# Each adapter was trained on base+SFT(merged); so eval must stack SFT too.
# Uses the deepseek-r1 <think>/<answer> format (--sft_style) and full budget.
#
# Usage: run_matched_eval.sh <label> <grpo_adapter_dir> [gpu]
set -euo pipefail
cd "$(dirname "$0")/../.."
source rebuttal/scripts/env.sh

LABEL="${1:?label}"
GRPO="${2:?grpo adapter dir}"
BASE=/Knowin/foundation/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
SFT=rebuttal/ckpts/sft-dr1-7b-final
MERGED="output/merged_${LABEL}"

# 1. Stack base + SFT + GRPO into a standalone model (skip if present)
if [ ! -f "${MERGED}/model.safetensors" ] && [ ! -f "${MERGED}/model.safetensors.index.json" ]; then
  "$TOPOPRM_PY" rebuttal/scripts/merge_stacked_adapter.py \
    --base "$BASE" --sft "$SFT" --grpo "$GRPO" --out "$MERGED"
fi

# 2. Eval pass@1 on primary math benchmarks with the training-matched format.
"$TOPOPRM_PY" scripts/bench_transformers.py \
  --model "$MERGED" --label "$LABEL" \
  --benchmarks gsm8k \
  --num_samples_per_item 1 --k_values 1 --max_items 200 \
  --max_new_tokens 4096 --batch_size 8 --use_chat_template --sft_style \
  --output_dir rebuttal/outputs/eval_tables
"$TOPOPRM_PY" scripts/bench_transformers.py \
  --model "$MERGED" --label "$LABEL" \
  --benchmarks math500 \
  --num_samples_per_item 1 --k_values 1 --max_items 200 \
  --max_new_tokens 8192 --batch_size 8 --use_chat_template --sft_style \
  --output_dir rebuttal/outputs/eval_tables
"$TOPOPRM_PY" scripts/bench_transformers.py \
  --model "$MERGED" --label "$LABEL" \
  --benchmarks aime2024 \
  --num_samples_per_item 1 --k_values 1 \
  --max_new_tokens 8192 --batch_size 8 --use_chat_template --sft_style \
  --output_dir rebuttal/outputs/eval_tables

echo "[run_matched_eval] $LABEL done"
