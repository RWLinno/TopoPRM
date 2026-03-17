#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SWIFT="swift"
PYTHON="${SWIFT%/*}/python"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export NPROC_PER_NODE

echo "=========================================="
echo "  TopoPRM Main Experiment (${NPROC_PER_NODE} GPUs)"
echo "  Model: Qwen3-32B"
echo "=========================================="

echo "[GRPO] Training with full composite reward..."
$SWIFT rlhf \
    --rlhf_type grpo \
    --config "${PROJECT_ROOT}/experiments/configs/grpo_main.yaml" \
    2>&1 | tee "${PROJECT_ROOT}/output/grpo_main.log"

echo "[Eval] Evaluating main model..."
$PYTHON -m src.eval.benchmark_runner \
    --model "${PROJECT_ROOT}/output/grpo_main" \
    --output_dir "${PROJECT_ROOT}/output/eval_main" \
    --benchmarks MATH GSM8K CMATH GaoKao 2>&1 || true

echo "=========================================="
echo "  Main experiment complete!"
echo "=========================================="
