#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

SFT_FINAL_DIR="output/sft_qwen25_7b_longcot_exp_may18/final"
POLL_SEC="${POLL_SEC:-300}"

echo "[exp_May18] Waiting for SFT final adapter: $SFT_FINAL_DIR"
while [ ! -d "$SFT_FINAL_DIR" ]; do
  sleep "$POLL_SEC"
done

LOG="logs/grpo_topoprm_qwen25_7b_longcot_exp_may18_$(date +%Y%m%d_%H%M%S).log"
echo "[exp_May18] SFT ready, launching GRPO on GPU6. log=$LOG"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-1}" \
MASTER_PORT="${MASTER_PORT:-29533}" \
WANDB_PROJECT="${WANDB_PROJECT:-topoprm}" \
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-exp_May18}" \
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
/mnt/users/conda_env/topoprm/bin/swift rlhf --rlhf_type grpo \
  --config configs/grpo_topoprm_qwen25_7b_longcot_exp_may18.yaml \
  > "$LOG" 2>&1
