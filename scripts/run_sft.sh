#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/mnt/users/rwl/topoprm"
SWIFT="/mnt/users/conda_env/swift/bin/swift"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export NPROC_PER_NODE

echo "=========================================="
echo "  SFT Training: Qwen3-32B (${NPROC_PER_NODE} GPUs)"
echo "=========================================="

$SWIFT sft \
    --config "${PROJECT_ROOT}/configs/sft_qwen3_32b.yaml" \
    2>&1 | tee "${PROJECT_ROOT}/output/sft_qwen3_32b.log"

echo "SFT training complete."
