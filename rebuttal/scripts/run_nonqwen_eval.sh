#!/usr/bin/env bash
# Non-Qwen generality sanity check (HxUk W3).
# Evaluates DeepSeek-R1-Distill-Llama-8B (genuinely non-Qwen tokenizer/family)
# on the four primary math benchmarks, as a base-model generality reference.
set -euo pipefail
cd "$(dirname "$0")/../.."
source rebuttal/scripts/env.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
LLAMA=/Knowin/foundation/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B

"$TOPOPRM_PY" scripts/bench_transformers.py \
  --model "$LLAMA" --label dr1_llama8b_base \
  --benchmarks gsm8k math500 olympiadbench omni_math \
  --num_samples_per_item 5 --k_values 1 5 \
  --max_new_tokens 8192 --use_chat_template \
  --output_dir rebuttal/outputs/eval_tables

echo "[run_nonqwen_eval] done"
