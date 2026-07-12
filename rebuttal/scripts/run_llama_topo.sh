#!/usr/bin/env bash
# Llama-3.1-8B-Instruct GRPO with full TopoPRM reward (generality test, HxUk W3).
set -uo pipefail
cd /Knowin/foundation/weilinruan/TopoPRM
source rebuttal/scripts/env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export WANDB_PROJECT=topoprm-rebuttal
"$TOPOPRM_PY" rebuttal/scripts/train_grpo_rebuttal.py \
  --reward topo_hierarchical \
  --model /Knowin/foundation/models/meta-llama/Llama-3.1-8B-Instruct_ef \
  --sft_adapter "" \
  --output_dir output/grpo_topo_hier_llama8b \
  --max_steps 150 --num_generations 4 --max_completion_len 2048 --report_to wandb
