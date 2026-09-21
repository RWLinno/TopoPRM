#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
TOPOPRM_ENV_BIN="${TOPOPRM_ENV_BIN:-${PYTHON_ENV_BIN:-}}"
TOPOPRM_SWIFT="${TOPOPRM_ENV_BIN:+${TOPOPRM_ENV_BIN}/}swift"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
cmd=("$TOPOPRM_SWIFT" sft
     --model "${SFT_MODEL:-Qwen/Qwen3.5-9B}"
     --dataset "${SFT_DATA:-data/sft_ready/train_public_swift.jsonl}"
     --output_dir "${SFT_OUTPUT_DIR:-output/topoprm_stage1}"
     --tuner_type lora --lora_rank 64 --lora_alpha 128 --lora_dropout 0.05
     --target_modules all-linear --learning_rate 5e-5
     --lr_scheduler_type cosine --warmup_ratio 0.03 --num_train_epochs 2
     --per_device_train_batch_size 2 --gradient_accumulation_steps 8
     --max_length 4096 --torch_dtype bfloat16 "$@")
if [ "${TOPOPRM_DRY_RUN:-0}" = "1" ]; then
    printf '%q ' "${cmd[@]}"; printf '\n'; exit 0
fi
exec "${cmd[@]}"
