#!/bin/bash
set -euo pipefail

###############################################################################
# Run GRPO training in server mode (ms-swift paradigm)
#
# Prerequisites: rollout server must be running on GPUs 0-3
#   bash scripts/start_rollout_server.sh
#
# Usage: bash scripts/run_grpo.sh [config_name]
###############################################################################

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PATH="$(dirname $(which swift)):$PATH"
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NPROC_PER_NODE=4

CONFIG_NAME="${1:-grpo_main}"
CONFIG_PATH="experiments/configs/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "[ERROR] Config not found: $CONFIG_PATH"
    exit 1
fi

# Verify rollout server is accessible
if ! curl -s http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
    echo "[ERROR] Rollout server not running on port 8000"
    echo "Start it first: bash scripts/start_rollout_server.sh"
    exit 1
fi

echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GRPO: $CONFIG_NAME (server mode)"
echo "Config: $CONFIG_PATH"
echo "Training GPUs: $CUDA_VISIBLE_DEVICES (NPROC=$NPROC_PER_NODE)"
echo "Rollout server: http://127.0.0.1:8000"
echo "=========================================="

swift rlhf \
    --rlhf_type grpo \
    --config "$CONFIG_PATH" \
    2>&1 | tee "output/${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GRPO $CONFIG_NAME completed"
