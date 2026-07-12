#!/usr/bin/env bash
# BoN candidate generation: GSM8K (neutral policy Qwen2.5-Math-7B-Instruct).
set -uo pipefail
cd /Knowin/foundation/weilinruan/TopoPRM
source rebuttal/scripts/env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
"$TOPOPRM_PY" rebuttal/scripts/prm_bon.py generate \
  --model /Knowin/foundation/models/Qwen/Qwen2.5-Math-7B-Instruct \
  --benchmarks gsm8k --n 80 --num_samples 8 --max_new_tokens 2048 --temperature 0.8 \
  --out rebuttal/outputs/prm_bon_pool.jsonl
