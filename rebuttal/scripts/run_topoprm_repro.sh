#!/usr/bin/env bash
# Reproducibility check: evaluate the released full-TopoPRM DR1-7B adapter
# on the primary math benchmarks to confirm the paper numbers (TsKG trust).
set -euo pipefail
cd "$(dirname "$0")/../.."
source rebuttal/scripts/env.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
BASE=/Knowin/foundation/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
ADAPTER=rebuttal/ckpts/grpo-topoprm-dr1-7b

"$TOPOPRM_PY" scripts/bench_transformers.py \
  --model "$BASE" --adapter "$ADAPTER" \
  --label topoprm_dr1_7b_repro \
  --benchmarks gsm8k \
  --num_samples_per_item 5 --k_values 1 5 --max_items 300 \
  --max_new_tokens 4096 --batch_size 8 --use_chat_template \
  --output_dir rebuttal/outputs/eval_tables
"$TOPOPRM_PY" scripts/bench_transformers.py \
  --model "$BASE" --adapter "$ADAPTER" \
  --label topoprm_dr1_7b_repro \
  --benchmarks math500 aime2024 \
  --num_samples_per_item 5 --k_values 1 5 \
  --max_new_tokens 8192 --batch_size 8 --use_chat_template \
  --output_dir rebuttal/outputs/eval_tables

echo "[run_topoprm_repro] done"
