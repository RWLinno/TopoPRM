#!/usr/bin/env bash
# Matched GRPO on DeepSeek-R1-Distill-Llama-8B — a genuinely non-Qwen
# (Llama-architecture) reasoning base with real headroom (GSM8K ~52, MATH ~50).
# Unlike Llama-3.1-8B-Instruct (GSM8K-saturated at ~85), this base leaves room
# for a process reward to show accuracy gains, giving HxUk W3 a complete
# non-Qwen base -> outcome-only -> TopoPRM comparison.
#
# DR1-Distill emits <think>...</think> natively, so the released TopoReward
# reads the think block directly (no prose-fallback flags needed).
#
# Usage: run_dr1llama_grpo.sh <reward> <gpu>   reward in {outcome_only, topo_hierarchical}
set -uo pipefail
cd /Knowin/foundation/weilinruan/TopoPRM
source rebuttal/scripts/env.sh
REWARD="${1:?reward: outcome_only|topo_hierarchical}"
export CUDA_VISIBLE_DEVICES="${2:?gpu}"
export WANDB_PROJECT=topoprm-rebuttal

"$TOPOPRM_PY" rebuttal/scripts/train_grpo_rebuttal.py \
  --reward "$REWARD" \
  --model /Knowin/foundation/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --sft_adapter "" \
  --output_dir "output/grpo_${REWARD}_dr1llama8b" \
  --max_steps 150 --num_generations 4 --max_completion_len 2048 --report_to wandb
