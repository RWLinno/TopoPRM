#!/usr/bin/env bash
# Evaluate the extra-seed matched DR1-7B GRPO adapters (seeds 123, 777) for
# outcome_only and topo_hierarchical, so we can report per-seed pass@1 +
# Wilson intervals ([[SEEDS]] in response.md).  Seed 42 is the already-evaluated
# outcome_only_matched / topo_hier_matched pair.
# Usage: run_seed_eval.sh <reward> <seed> <gpu>
set -euo pipefail
cd /Knowin/foundation/weilinruan/TopoPRM
source rebuttal/scripts/env.sh

REWARD="${1:?reward: outcome_only|topo_hierarchical}"
SEED="${2:?seed}"
GPU="${3:?gpu}"
BASE=/Knowin/foundation/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
SFT=rebuttal/ckpts/sft-dr1-7b-final
GRPO="output/grpo_${REWARD}_dr1_7b_seed${SEED}/final"
LABEL="${REWARD}_seed${SEED}"
MERGED="output/merged_${LABEL}"

if [ ! -f "${MERGED}/model.safetensors" ] && [ ! -f "${MERGED}/model.safetensors.index.json" ]; then
  CUDA_VISIBLE_DEVICES="$GPU" "$TOPOPRM_PY" rebuttal/scripts/merge_stacked_adapter.py \
    --base "$BASE" --sft "$SFT" --grpo "$GRPO" --out "$MERGED"
fi

CUDA_VISIBLE_DEVICES="$GPU" "$TOPOPRM_PY" scripts/bench_transformers.py \
  --model "$MERGED" --label "$LABEL" \
  --benchmarks gsm8k \
  --num_samples_per_item 1 --k_values 1 --max_items 200 \
  --max_new_tokens 4096 --batch_size 8 --use_chat_template --sft_style \
  --output_dir rebuttal/outputs/eval_tables

CUDA_VISIBLE_DEVICES="$GPU" "$TOPOPRM_PY" scripts/bench_transformers.py \
  --model "$MERGED" --label "$LABEL" \
  --benchmarks math500 \
  --num_samples_per_item 1 --k_values 1 --max_items 200 \
  --max_new_tokens 8192 --batch_size 8 --use_chat_template --sft_style \
  --output_dir rebuttal/outputs/eval_tables

CUDA_VISIBLE_DEVICES="$GPU" "$TOPOPRM_PY" scripts/bench_transformers.py \
  --model "$MERGED" --label "$LABEL" \
  --benchmarks aime2024 \
  --num_samples_per_item 1 --k_values 1 \
  --max_new_tokens 8192 --batch_size 8 --use_chat_template --sft_style \
  --output_dir rebuttal/outputs/eval_tables

echo "[run_seed_eval] $LABEL done"
