#!/usr/bin/env bash
# watch_and_launch_grpo_B.sh ? ? GPU 0 ?? smoke / ref_qwen25_7b_instruct_B ????
# ?? grpo_outcome_only_qwen35_9b_B ?? 9 benchmark ???
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
cd "$REPO_ROOT"

source /mnt/users/miniconda3/etc/profile.d/conda.sh
conda activate topoprm
export ALL_PROXY=http://accelerator-cname-hnpmnhnmdul3rmxrwhgend.c.vegalb.com:80
export HF_TOKEN=${HF_TOKEN:-""}
export WANDB_API_KEY=${WANDB_API_KEY:-""}

LOG_DIR="$REPO_ROOT/results/baseline/logs_B"
mkdir -p "$LOG_DIR"
WATCH_LOG="$LOG_DIR/watch_and_launch_grpo_B.log"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$WATCH_LOG"
}

log "watch start ? will launch grpo_outcome_only_qwen35_9b_B when GPU 0 is free (<3 GiB used)"

while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  if [[ -n "$used" && "$used" -lt 3000 ]]; then
    log "GPU 0 free (${used} MiB used), launching grpo_outcome_only_qwen35_9b_B"
    break
  fi
  sleep 60
done

ADAPTER=$(ls -dt output/grpo_outcome_only_qwen35_9b/*/checkpoint-* 2>/dev/null | head -1)
if [[ -z "$ADAPTER" || ! -d "$ADAPTER" ]]; then
  log "ERROR: grpo_outcome_only adapter not found"
  exit 1
fi

log "launching with adapter=$ADAPTER"
python -u scripts/unified_eval_orchestrator.py \
  --model /mnt/data/huggingface_downloads/models/qwen/Qwen3.5-9B \
  --adapter "$ADAPTER" \
  --label grpo_outcome_only_qwen35_9b_B \
  --benchmarks all \
  --gpus 0 \
  --output_dir output/eval \
  --log_dir "$LOG_DIR" \
  --num_samples_per_item 5 \
  --k_values 1 5 \
  --use_chat_template \
  >>"$LOG_DIR/orch_grpo_outcome_only_qwen35_9b_B.nohup.log" 2>&1

log "grpo_outcome_only_qwen35_9b_B orchestrator exited rc=$?"
