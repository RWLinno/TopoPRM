#!/usr/bin/env bash
# Re-run key models with num_samples_per_item=5 to measure pass@5 / maj@5 / prm@5.
# Only uses GPUs 0-5 (never 6/7).

set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="${PYTHON_ENV_BIN}:$PATH"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

QWEN35_9B=${MODEL_ROOT}/qwen/Qwen3.5-9B

# Model matrix (label -> "model_path adapter_path")
declare -A MODELS=(
  ["base_9b_k5"]="$QWEN35_9B"
  ["sft_9b_k5"]="$QWEN35_9B output/sft_qwen35_9b/v0-20260407-011328/checkpoint-626"
  ["topoprm_hier_9b_k5"]="$QWEN35_9B output/grpo_hierarchical_qwen35_9b_mcl4096/v2-20260407-162048/checkpoint-79"
  ["topoprm_gated_9b_k5"]="$QWEN35_9B output/grpo_gated_qwen35_9b_mcl4096/v4-20260407-111747/checkpoint-79"
  ["no_topo_9b_k5"]="$QWEN35_9B output/grpo_no_topo_qwen35_9b_mcl4096/v1-20260407-191217/checkpoint-79"
)

ALLOWED_GPUS="0 1 2 3 4 5"

wait_for_gpu() {
  while true; do
    for g in $ALLOWED_GPUS; do
      mem=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 999999)
      nproc=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
      if [ "$nproc" = "0" ] && [ "$mem" -lt 1024 ]; then
        echo $g; return
      fi
    done
    sleep 60
  done
}

launch_eval() {
  local label="$1"
  local model="$2"
  local adapter="${3:-}"
  local gpu=$(wait_for_gpu)
  local adapter_arg=""
  [ -n "$adapter" ] && adapter_arg="--adapter $adapter"
  local log="logs/eval_${label}.log"
  echo "[$label] GPU $gpu -> $log"
  nohup env CUDA_VISIBLE_DEVICES=$gpu python3 scripts/bench_transformers.py \
    --model "$model" \
    $adapter_arg \
    --label "$label" \
    --benchmarks gsm8k math500 \
    --batch_size 4 \
    --max_new_tokens 2048 \
    --num_samples_per_item 5 \
    --k_values 1 5 \
    > "$log" 2>&1 &
  echo "[$label] PID=$!"
  sleep 30  # stagger launches so GPU detection works
}

for label in "${!MODELS[@]}"; do
  IFS=' ' read -r model adapter <<< "${MODELS[$label]}"
  launch_eval "$label" "$model" "${adapter:-}"
done

echo "All pass@5 jobs launched. Use 'tail -f logs/eval_*_k5.log' to monitor."
