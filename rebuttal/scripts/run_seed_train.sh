#!/usr/bin/env bash
# Extra-seed matched DR1-7B GRPO for significance (B4).
# Usage: run_seed_train.sh <reward> <seed> [gpu]
set -uo pipefail
cd /Knowin/foundation/weilinruan/TopoPRM
source rebuttal/scripts/env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_PROJECT=topoprm-rebuttal
REWARD="${1:?reward}"
SEED="${2:?seed}"
export PYTHONHASHSEED="$SEED"
"$TOPOPRM_PY" rebuttal/scripts/train_grpo_rebuttal.py \
  --reward "$REWARD" \
  --model /Knowin/foundation/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --sft_adapter rebuttal/ckpts/sft-dr1-7b-final \
  --output_dir "output/grpo_${REWARD}_dr1_7b_seed${SEED}" \
  --seed "$SEED" \
  --max_steps 150 --num_generations 4 --max_completion_len 2048 --report_to wandb
