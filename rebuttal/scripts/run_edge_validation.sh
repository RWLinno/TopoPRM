#!/usr/bin/env bash
# Edge validation: independent-judge (Qwen3-32B) support-edge labeling + scoring.
# Answers HxUk W1 / B5w7 W1 / TsKG W1.
set -euo pipefail
cd "$(dirname "$0")/../.."
source rebuttal/scripts/env.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
JUDGE="${EDGE_JUDGE_MODEL:-/Knowin/foundation/models/Qwen/Qwen3-32B}"

# 1. sample the annotation pack (skip if present)
if [ ! -s rebuttal/outputs/edge_validation_pack.jsonl ]; then
  "$TOPOPRM_PY" rebuttal/scripts/edge_validation.py sample \
    --n 120 --out rebuttal/outputs/edge_validation_pack.jsonl
fi

# 2. annotate with the local independent judge
"$TOPOPRM_PY" rebuttal/scripts/edge_validation.py annotate \
  --local --model "$JUDGE" --max-tokens 2048 \
  --pack rebuttal/outputs/edge_validation_pack.jsonl \
  --out rebuttal/outputs/edge_validation_annotations.jsonl

# 3. score P/R/F1 + per-edge-type reliability
"$TOPOPRM_PY" rebuttal/scripts/edge_validation.py score \
  --pack rebuttal/outputs/edge_validation_pack.jsonl \
  --ann rebuttal/outputs/edge_validation_annotations.jsonl \
  --model "$JUDGE" \
  --out rebuttal/outputs/edge_validation_results.json

echo "[run_edge_validation] done"
