#!/bin/bash
# =============================================================
#  TopoPRM Ablation: Outcome-only reward (no topo/continuity)
#  Multi-GPU deployment
# =============================================================
set -euo pipefail

PROJECT_ROOT="/mnt/users/rwl/topoprm"
SWIFT="${SWIFT:-/mnt/users/conda_env/swift/bin/swift}"
PYTHON="${SWIFT%/*}/python"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export NPROC_PER_NODE

echo "=========================================="
echo "  Ablation: Outcome-only (${NPROC_PER_NODE} GPUs)"
echo "=========================================="

echo "[GRPO] Training with outcome-only reward..."
$SWIFT rlhf \
    --rlhf_type grpo \
    --config "${PROJECT_ROOT}/experiments/configs/grpo_outcome_only.yaml" \
    2>&1 | tee "${PROJECT_ROOT}/output/grpo_outcome_only.log"

echo "[Eval] Evaluating outcome-only model..."
$PYTHON -m src.eval.benchmark_runner \
    --model "${PROJECT_ROOT}/output/grpo_outcome_only" \
    --output_dir "${PROJECT_ROOT}/output/eval_outcome_only" \
    --benchmarks MATH GSM8K CMATH GaoKao 2>&1 || true

echo "  Ablation (outcome-only) complete!"
