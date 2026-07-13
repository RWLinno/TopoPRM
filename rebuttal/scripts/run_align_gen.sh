#!/usr/bin/env bash
# Generate traces from a trained policy for structure->correctness alignment (A1).
# Usage: run_align_gen.sh <label> <merged_model_dir> [gpu]
set -uo pipefail
cd /Knowin/foundation/weilinruan/TopoPRM
source rebuttal/scripts/env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LABEL="${1:?label}"
MODEL="${2:?model dir}"
"$TOPOPRM_PY" rebuttal/scripts/semantic_gap.py generate \
  --model "$MODEL" \
  --benchmarks gsm8k math500 --n 80 --max-tokens 2048 --temperature 0.0 \
  --out "rebuttal/outputs/align_pool_${LABEL}.jsonl"
