#!/usr/bin/env bash
# Outcome+length GRPO baseline (DR1-7B): isolates topology from brevity pressure.
# Trains from the released SFT adapter for 200 steps, then evaluates.
# Answers HxUk W2 / B5w7 W3.
set -euo pipefail
cd "$(dirname "$0")/../.."
source rebuttal/scripts/env.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export WANDB_PROJECT="${WANDB_PROJECT:-topoprm-rebuttal}"

BASE=/Knowin/foundation/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
OUT=output/grpo_outcome_length_dr1_7b

# 1. Train via TRL (no vLLM; broken in this env). Skip if a checkpoint exists.
if ! ls -d ${OUT}/checkpoint-* >/dev/null 2>&1 && [ ! -d "${OUT}/final" ]; then
  "$TOPOPRM_PY" rebuttal/scripts/train_grpo_rebuttal.py \
    --reward outcome_length \
    --sft_adapter rebuttal/ckpts/sft-dr1-7b-final \
    --output_dir "$OUT" --max_steps 200 --num_generations 4 \
    --max_completion_len 2048 --report_to wandb
fi

# 2. Locate trained adapter
ADAPTER=$(ls -d ${OUT}/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
[ -z "$ADAPTER" ] && [ -d "${OUT}/final" ] && ADAPTER="${OUT}/final"
echo "[run_outcome_length_baseline] eval adapter: $ADAPTER"

# 3. Evaluate on the primary public math benchmarks.
#    GSM8K capped to 300 items (pass@1 +-3%); MATH-500 full; AIME'24 full.
"$TOPOPRM_PY" scripts/bench_transformers.py \
  --model "$BASE" --adapter "$ADAPTER" \
  --label outcome_length_dr1_7b \
  --benchmarks gsm8k \
  --num_samples_per_item 5 --k_values 1 5 --max_items 300 \
  --max_new_tokens 4096 --batch_size 8 --use_chat_template \
  --output_dir rebuttal/outputs/eval_tables
"$TOPOPRM_PY" scripts/bench_transformers.py \
  --model "$BASE" --adapter "$ADAPTER" \
  --label outcome_length_dr1_7b \
  --benchmarks math500 aime2024 \
  --num_samples_per_item 5 --k_values 1 5 \
  --max_new_tokens 8192 --batch_size 8 --use_chat_template \
  --output_dir rebuttal/outputs/eval_tables

echo "[run_outcome_length_baseline] done"
