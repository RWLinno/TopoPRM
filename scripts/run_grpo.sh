#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/mnt/users/rwl/topoprm"
SWIFT="/mnt/users/conda_env/swift/bin/swift"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export NPROC_PER_NODE

echo "=========================================="
echo "  GRPO Training: Qwen3-32B (${NPROC_PER_NODE} GPUs)"
echo "=========================================="

$SWIFT rlhf \
    --rlhf_type grpo \
    --config "${PROJECT_ROOT}/configs/grpo_qwen3_32b.yaml" \
    2>&1 | tee "${PROJECT_ROOT}/output/grpo_qwen3_32b.log"

echo "GRPO training complete."
