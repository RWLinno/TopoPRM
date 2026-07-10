#!/usr/bin/env bash
# DAG extractor visualization: render success/failure cases for the rebuttal
# (all three reviewers ask to interpret the extractor). CPU-only (matplotlib).
set -euo pipefail
cd "$(dirname "$0")/../.."
source rebuttal/scripts/env.sh

OUT=rebuttal/outputs/dag_cases
mkdir -p "$OUT"

# Gold-solution DAGs (clean structure) from training data
"$TOPOPRM_PY" tutorials/render_dag_cases.py \
  --n 8 --source mixed --seed 7 \
  --jsonl data/grpo_ready/train_public.jsonl \
  --out-dir "$OUT/gold" || echo "[warn] gold render failed"

# Model-generated traces (success + failure mix) if the pool exists
if [ -s rebuttal/outputs/semantic_gap_pool.jsonl ]; then
  "$TOPOPRM_PY" tutorials/render_dag_cases.py \
    --n 8 --from-rollout rebuttal/outputs/semantic_gap_pool.jsonl \
    --out-dir "$OUT/model" || echo "[warn] model render failed"
fi

echo "[run_dag_visualization] done -> $OUT"
