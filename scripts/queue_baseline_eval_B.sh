#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
cd "$REPO_ROOT"

source /mnt/users/miniconda3/etc/profile.d/conda.sh
conda activate topoprm

export ALL_PROXY=http://accelerator-cname-hnpmnhnmdul3rmxrwhgend.c.vegalb.com:80
export HF_TOKEN=${HF_TOKEN:-""}
export WANDB_API_KEY=${WANDB_API_KEY:-""}

LOG_DIR="$REPO_ROOT/results/baseline/logs_B"
mkdir -p "$LOG_DIR"
QLOG="$LOG_DIR/queue_baseline_eval_B.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"
}

# format: label|model_path|gpu
JOBS=(
  "ref_llama31_8b_instruct_B|/mnt/data/models/Llama-3.1-8B-Instruct|1"
  "ref_deepseek_math_7b_instruct_B|/mnt/data/models/deepseek-math-7b-instruct|2"
  "ref_deepseek_math_7b_rl_B|/mnt/data/models/deepseek-math-7b-rl|3"
  "ref_qwen25_math_1p5b_instruct_B|/mnt/data/models/Qwen2.5-Math-1.5B-Instruct|4"
  "ref_deepseek_r1_distill_llama_8b_B|/mnt/data/models/DeepSeek-R1-Distill-Llama-8B|1"
  "ref_qwen3_8b_B|/mnt/data/models/Qwen3-8B|2"
)

declare -A done
for j in "${JOBS[@]}"; do done["$j"]=0; done

log "queue started with ${#JOBS[@]} jobs"

while true; do
  pending=0
  for j in "${JOBS[@]}"; do
    if [[ "${done[$j]}" -eq 1 ]]; then
      continue
    fi
    pending=$((pending+1))

    IFS='|' read -r label model gpu <<<"$j"

    # already has at least one metric file => job considered launched/completed previously
    if ls output/eval/${label}_*_metrics.json >/dev/null 2>&1; then
      log "skip $label (metrics already present)"
      done["$j"]=1
      continue
    fi

    # if orchestrator already running for this label, mark done in queue
    if pgrep -af "unified_eval_orchestrator.py.*--label ${label}( |$)" >/dev/null 2>&1; then
      log "already running: $label"
      done["$j"]=1
      continue
    fi

    # wait model ready
    if [[ ! -d "$model" || ! -f "$model/config.json" ]]; then
      log "model not ready for $label: $model"
      continue
    fi

    # wait target gpu free enough
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    if [[ -z "$used" || "$used" -ge 3000 ]]; then
      log "gpu $gpu busy (${used:-NA} MiB), wait for $label"
      continue
    fi

    log "launch $label on gpu $gpu model=$model"
    nohup python -u scripts/unified_eval_orchestrator.py \
      --model "$model" \
      --label "$label" \
      --benchmarks all \
      --gpus "$gpu" \
      --output_dir output/eval \
      --log_dir "$LOG_DIR" \
      --num_samples_per_item 5 \
      --k_values 1 5 \
      --use_chat_template \
      > "$LOG_DIR/orch_${label}.nohup.log" 2>&1 &

    done["$j"]=1
  done

  if [[ "$pending" -eq 0 ]]; then
    log "all queued jobs launched or skipped"
    break
  fi

  sleep 60
done
