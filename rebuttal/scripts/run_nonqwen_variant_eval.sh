#!/usr/bin/env bash
# Eval a non-Qwen GRPO adapter (LoRA on the base, no SFT stack) on the three
# core math benchmarks, matching the base-model eval protocol.
# Usage: run_nonqwen_variant_eval.sh <label> <base_model> <adapter_dir> <gpu>
set -euo pipefail
cd /Knowin/foundation/weilinruan/TopoPRM
source rebuttal/scripts/env.sh
LABEL="${1:?}"; BASE="${2:?}"; ADAPTER="${3:?}"; export CUDA_VISIBLE_DEVICES="${4:?}"

"$TOPOPRM_PY" scripts/bench_transformers.py \
  --model "$BASE" --adapter "$ADAPTER" --label "$LABEL" \
  --benchmarks gsm8k math500 aime2024 \
  --num_samples_per_item 1 --k_values 1 --max_items 200 \
  --max_new_tokens 4096 --batch_size 16 --use_chat_template \
  --output_dir rebuttal/outputs/eval_tables
echo "[variant_eval] $LABEL done"
