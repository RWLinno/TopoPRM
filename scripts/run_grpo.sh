#!/bin/bash
set -euo pipefail

###############################################################################
# Run a single GRPO experiment with auto-resume and memory safety
#
# Usage: bash scripts/run_grpo.sh [config_name] [--resume]
#   config_name: grpo_main | grpo_outcome_only | grpo_no_topo | grpo_no_continuity
#   --resume: auto-detect and resume from latest checkpoint
#
# Env vars:
#   CUDA_VISIBLE_DEVICES (default: 0,1,2,3,4,5,6,7)
#   NPROC_PER_NODE      (default: 8)
###############################################################################

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PATH="/mnt/users/conda_env/swift/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# Prevent shared memory accumulation from vLLM prefix caching
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CONFIG_NAME="${1:-grpo_main}"
RESUME_FLAG="${2:-}"
CONFIG_PATH="experiments/configs/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "[ERROR] Config not found: $CONFIG_PATH"
    ls experiments/configs/grpo_*.yaml
    exit 1
fi

# Find latest checkpoint for auto-resume
RESUME_ARG=""
OUTPUT_DIR=$(grep "^output_dir:" "$CONFIG_PATH" | awk '{print $2}')
if [ "$RESUME_FLAG" = "--resume" ] && [ -n "$OUTPUT_DIR" ]; then
    LATEST_VERSION=$(ls -dt "${OUTPUT_DIR}"/v*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_VERSION" ]; then
        LATEST_CKPT=$(ls -dt "${LATEST_VERSION}"checkpoint-* 2>/dev/null | head -1)
        if [ -n "$LATEST_CKPT" ]; then
            RESUME_ARG="--resume_from_checkpoint $LATEST_CKPT"
            echo "[RESUME] Found checkpoint: $LATEST_CKPT"
        fi
    fi
fi

mkdir -p output

echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GRPO: $CONFIG_NAME"
echo "Config: $CONFIG_PATH"
echo "GPUs: $CUDA_VISIBLE_DEVICES (NPROC=$NPROC_PER_NODE)"
[ -n "$RESUME_ARG" ] && echo "Resume: $RESUME_ARG"
echo "=========================================="

swift rlhf \
    --rlhf_type grpo \
    --config "$CONFIG_PATH" \
    $RESUME_ARG \
    2>&1 | tee "output/${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GRPO $CONFIG_NAME completed"
