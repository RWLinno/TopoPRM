#!/usr/bin/env bash
# Run extended 9-benchmark suite on the 3 best-performing models.
# Waits for GPU 4-5 to be free, then launches serially.
#
# Usage: bash scripts/run_extended_benchmarks.sh [model_label]

set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="${PYTHON_ENV_BIN}:$PATH"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
mkdir -p logs

ALLOWED_GPUS="4 5"
BENCHMARKS=(aime2024 aime2025 cnmo2024 mmlu gpqa_diamond olympiadbench omni_math)

# label -> "model adapter sft_style"
declare -A JOBS=(
  ["sft_9b_v2_ext"]="${MODEL_ROOT}/qwen/Qwen3.5-9B output/sft_qwen35_9b/v0-20260407-011328/checkpoint-626 sft_style"
  ["topoprm_gated_9b_v2_ext"]="${MODEL_ROOT}/qwen/Qwen3.5-9B output/grpo_gated_qwen35_9b_mcl4096/v4-20260407-111747/checkpoint-79 sft_style"
  ["topoprm_hier_9b_v2_ext"]="${MODEL_ROOT}/qwen/Qwen3.5-9B output/grpo_hierarchical_qwen35_9b_mcl4096/v2-20260407-162048/checkpoint-79 sft_style"
)

wait_for_gpu() {
  while true; do
    for g in $ALLOWED_GPUS; do
      mem=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 999999)
      nproc=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
      if [ "$nproc" = "0" ] && [ "$mem" -lt 1024 ]; then
        echo $g
        return
      fi
    done
    sleep 60
  done
}

run_one() {
  local label=$1
  local info=${JOBS[$label]}
  IFS=' ' read -r model adapter sft_flag <<< "$info"
  local gpu=$(wait_for_gpu)
  local flags=""
  [ "$sft_flag" = "sft_style" ] && flags="--use_chat_template --sft_style"
  echo "[$(date '+%H:%M')] $label -> GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/bench_transformers.py \
    --model "$model" \
    --adapter "$adapter" \
    --label "$label" \
    --benchmarks "${BENCHMARKS[@]}" \
    $flags \
    --num_samples_per_item 3 \
    --k_values 1 3 \
    --batch_size 4 \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    > "logs/eval_${label}.log" 2>&1 &
  echo $! > "logs/eval_${label}.pid"
}

# Launch one job at a time (fall back to both GPUs when free)
for label in sft_9b_v2_ext topoprm_gated_9b_v2_ext topoprm_hier_9b_v2_ext; do
  run_one "$label"
  sleep 30
done

wait
echo "All extended benchmark jobs launched. Use tail -f logs/eval_*_ext.log to watch."
