#!/bin/bash
set -euo pipefail

###############################################################################
# Run all 3 ablation experiments sequentially
# Each has prefix_caching disabled to prevent OOM
#
# Usage: bash scripts/run_ablations.sh
###############################################################################

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PATH="$(dirname $(which swift)):$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

ABLATIONS=("grpo_outcome_only" "grpo_no_topo" "grpo_no_continuity")

echo "=========================================="
echo " TopoPRM Ablation Experiments"
echo " GPUs: $CUDA_VISIBLE_DEVICES  NPROC: $NPROC_PER_NODE"
echo " Experiments: ${ABLATIONS[*]}"
echo "=========================================="

for exp in "${ABLATIONS[@]}"; do
    echo ""
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Starting: $exp"

    # Clean shared memory before each experiment
    rm -f /dev/shm/nccl-* /dev/shm/vllm-* /dev/shm/cuda-* 2>/dev/null || true

    bash scripts/run_grpo.sh "$exp"

    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Finished: $exp"
    echo ""
    sleep 30  # brief cooldown between experiments
done

echo "=========================================="
echo " All ablations completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
