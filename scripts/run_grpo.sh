#!/bin/bash
set -euo pipefail

###############################################################################
# Run GRPO training for a single config
# Usage: bash scripts/run_grpo.sh [config_name]
#   config_name: grpo_main (default), grpo_outcome_only, grpo_no_topo, grpo_no_continuity
###############################################################################

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PATH="/mnt/users/conda_env/swift/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

CONFIG_NAME="${1:-grpo_main}"
CONFIG_PATH="experiments/configs/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: config not found: $CONFIG_PATH"
    echo "Available configs:"
    ls experiments/configs/grpo_*.yaml
    exit 1
fi

echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting GRPO: $CONFIG_NAME"
echo "Config: $CONFIG_PATH"
echo "GPUs: $CUDA_VISIBLE_DEVICES (NPROC=$NPROC_PER_NODE)"
echo "=========================================="

swift rlhf \
    --rlhf_type grpo \
    --config "$CONFIG_PATH" \
    2>&1 | tee "output/${CONFIG_NAME}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GRPO $CONFIG_NAME completed"
