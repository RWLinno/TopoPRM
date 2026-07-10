#!/usr/bin/env bash
# Structure-semantic gap: generate traces, score q_topo/q_cont/r_out, cross-tab.
# Answers B5w7 W2 / TsKG comment.
set -euo pipefail
cd "$(dirname "$0")/../.."
source rebuttal/scripts/env.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
GEN_MODEL="${GAP_GEN_MODEL:-/Knowin/foundation/models/Qwen/Qwen3.5-9B}"

# 1. generate traces (100/benchmark on gsm8k + math500)
"$TOPOPRM_PY" rebuttal/scripts/semantic_gap.py generate \
  --model "$GEN_MODEL" \
  --benchmarks gsm8k math500 --n 100 --max-tokens 2048 --temperature 0.0 \
  --out rebuttal/outputs/semantic_gap_pool.jsonl

# 2. score topology/continuity/outcome and the gap probabilities
"$TOPOPRM_PY" rebuttal/scripts/semantic_gap.py score \
  --pool rebuttal/outputs/semantic_gap_pool.jsonl \
  --hi 0.8 --lo 0.5 \
  --out rebuttal/outputs/semantic_gap_table.csv

echo "[run_semantic_gap] done"
