#!/bin/bash
set -euo pipefail

###############################################################################
# Resume GRPO main experiment from latest checkpoint (v5, checkpoint-150)
# 
# This script specifically handles the OOM crash recovery by:
#   1. Disabling vLLM prefix caching (root cause of shared memory leak)
#   2. Resuming from checkpoint-150
#   3. Clearing /dev/shm before starting
###############################################################################

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PATH="$(dirname $(which swift)):$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Clean up leftover shared memory from previous vLLM crash
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaning shared memory..."
rm -f /dev/shm/nccl-* /dev/shm/vllm-* /dev/shm/cuda-* 2>/dev/null || true

CKPT="$(cd "$(dirname "$0")/.." && pwd)/output/grpo_main/v5-20260316-110039/checkpoint-150"

if [ ! -d "$CKPT" ]; then
    echo "[ERROR] Checkpoint not found: $CKPT"
    echo "Available checkpoints:"
    find output/grpo_main -name "checkpoint-*" -type d 2>/dev/null
    exit 1
fi

echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Resuming GRPO main from checkpoint-150"
echo "Checkpoint: $CKPT"
echo "GPUs: $CUDA_VISIBLE_DEVICES (NPROC=$NPROC_PER_NODE)"
echo "Fix: vllm_enable_prefix_caching=false"
echo "=========================================="

swift rlhf \
    --rlhf_type grpo \
    --config experiments/configs/grpo_main.yaml \
    --resume_from_checkpoint "$CKPT" \
    2>&1 | tee "output/grpo_main_resume_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GRPO main resume completed"
