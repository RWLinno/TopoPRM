#!/bin/bash
# =============================================================
#  TopoPRM Ablation: No continuity reward
#  Multi-GPU deployment
# =============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SWIFT="${SWIFT:-swift}"
PYTHON="${SWIFT%/*}/python"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export NPROC_PER_NODE

echo "=========================================="
echo "  Ablation: No-continuity (${NPROC_PER_NODE} GPUs)"
echo "=========================================="

echo "[GRPO] Training without continuity reward..."
$SWIFT rlhf \
    --rlhf_type grpo \
    --config "${PROJECT_ROOT}/experiments/configs/grpo_no_continuity.yaml" \
    2>&1 | tee "${PROJECT_ROOT}/output/grpo_no_continuity.log"

echo "[Eval] Evaluating no-continuity model..."
$PYTHON -m src.eval.benchmark_runner \
    --model "${PROJECT_ROOT}/output/grpo_no_continuity" \
    --output_dir "${PROJECT_ROOT}/output/eval_no_continuity" \
    --benchmarks MATH GSM8K CMATH GaoKao 2>&1 || true

echo "  Ablation (no-continuity) complete!"
