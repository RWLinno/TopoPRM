#!/usr/bin/env bash
# Llama-3.1-8B-Instruct GRPO with outcome-only reward (generality baseline, HxUk W3).
set -uo pipefail
cd /Knowin/foundation/weilinruan/TopoPRM
source rebuttal/scripts/env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export WANDB_PROJECT=topoprm-rebuttal
"$TOPOPRM_PY" rebuttal/scripts/train_grpo_rebuttal.py \
  --reward outcome_only \
  --model /Knowin/foundation/models/meta-llama/Llama-3.1-8B-Instruct_ef \
  --sft_adapter "" \
  --output_dir output/grpo_outcome_only_llama8b \
  --max_steps 150 --num_generations 4 --max_completion_len 2048 --report_to wandb
