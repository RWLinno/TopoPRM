#!/usr/bin/env bash
# Fast pass@1 eval for rebuttal (num_samples=1). One adapter, primary benchmarks.
# Usage: run_fast_eval.sh <label> <adapter_path_or_empty> [base_model]
set -euo pipefail
cd "$(dirname "$0")/../.."
source rebuttal/scripts/env.sh

LABEL="${1:?label}"
ADAPTER="${2:-}"
BASE="${3:-/Knowin/foundation/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
ADP_ARG=""
[ -n "$ADAPTER" ] && ADP_ARG="--adapter $ADAPTER"

# GSM8K (cap 200) + MATH-500 (cap 200) + AIME'24 (full 30), pass@1 only.
"$TOPOPRM_PY" scripts/bench_transformers.py \
  --model "$BASE" $ADP_ARG --label "$LABEL" \
  --benchmarks gsm8k \
  --num_samples_per_item 1 --k_values 1 --max_items 200 \
  --max_new_tokens 3072 --batch_size 16 --use_chat_template \
  --output_dir rebuttal/outputs/eval_tables

"$TOPOPRM_PY" scripts/bench_transformers.py \
  --model "$BASE" $ADP_ARG --label "$LABEL" \
  --benchmarks math500 \
  --num_samples_per_item 1 --k_values 1 --max_items 200 \
  --max_new_tokens 4096 --batch_size 16 --use_chat_template \
  --output_dir rebuttal/outputs/eval_tables

"$TOPOPRM_PY" scripts/bench_transformers.py \
  --model "$BASE" $ADP_ARG --label "$LABEL" \
  --benchmarks aime2024 \
  --num_samples_per_item 1 --k_values 1 \
  --max_new_tokens 8192 --batch_size 8 --use_chat_template \
  --output_dir rebuttal/outputs/eval_tables

echo "[run_fast_eval] $LABEL done"
