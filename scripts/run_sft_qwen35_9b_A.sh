#!/bin/bash
set -euo pipefail
###############################################################################
# Server A — Stage-I SFT cold start (Qwen3.5-9B)
# Date: 2026-05-14
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
if [[ "$BRANCH" != "exp_May14" ]]; then
    echo "[FATAL] Expected branch exp_May14, got: $BRANCH"
    exit 1
fi

export ALL_PROXY=http://accelerator-cname-hnpmnhnmdul3rmxrwhgend.c.vegalb.com:80
export HF_TOKEN=${HF_TOKEN}
export WANDB_API_KEY=${WANDB_API_KEY}
export WANDB_PROJECT=topoprm
export WANDB_RUN_GROUP=sft_A
export WANDB_NAME="sft_qwen35_9b_A_$(date +%Y%m%d)"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NPROC_PER_NODE=6

echo "══════════════════════════════════════════════════════════════"
echo "[Server A] $(date '+%Y-%m-%d %H:%M:%S') Stage-I SFT (Qwen3.5-9B)"
echo "  Branch:  $BRANCH"
echo "  Config:  configs/sft_qwen35_9b_A.yaml"
echo "  GPUs:    $CUDA_VISIBLE_DEVICES (NPROC=$NPROC_PER_NODE)"
echo "  Output:  output/sft_qwen35_9b_A"
echo "══════════════════════════════════════════════════════════════"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

swift sft configs/sft_qwen35_9b_A.yaml
