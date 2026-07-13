#!/usr/bin/env bash
# Eval a Llama-3.1-8B GRPO adapter (no SFT stack; adapter sits on base instruct).
# Usage: run_llama_eval.sh <label> <adapter_dir> [gpu]
set -uo pipefail
cd /Knowin/foundation/weilinruan/TopoPRM
source rebuttal/scripts/env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
LABEL="${1:?label}"
ADAPTER="${2:?adapter dir}"
BASE=/Knowin/foundation/models/meta-llama/Llama-3.1-8B-Instruct_ef

"$TOPOPRM_PY" scripts/bench_transformers.py \
  --model "$BASE" --adapter "$ADAPTER" --label "$LABEL" \
  --benchmarks gsm8k \
  --num_samples_per_item 1 --k_values 1 --max_items 200 \
  --max_new_tokens 3072 --batch_size 16 --use_chat_template \
  --output_dir rebuttal/outputs/eval_tables

"$TOPOPRM_PY" scripts/bench_transformers.py \
  --model "$BASE" --adapter "$ADAPTER" --label "$LABEL" \
  --benchmarks math500 \
  --num_samples_per_item 1 --k_values 1 --max_items 200 \
  --max_new_tokens 4096 --batch_size 16 --use_chat_template \
  --output_dir rebuttal/outputs/eval_tables

echo "[run_llama_eval] $LABEL done"
