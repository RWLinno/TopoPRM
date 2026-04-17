#!/usr/bin/env bash
# Launch SFT distillation to 4B / 2B / 0.8B students when GPUs become available.
# Usage: bash scripts/launch_distill_when_ready.sh [min_free_gpus]

set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="/mnt/users/conda_env/topoprm/bin:$PATH"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

MIN_FREE=${1:-3}
ALLOWED_GPUS="0 1 2 3 4 5"  # never use GPU 6 or 7

# Wait for at least MIN_FREE GPUs to be free (no process + mem < 1GB)
wait_for_gpus() {
  while true; do
    free_gpus=()
    for g in $ALLOWED_GPUS; do
      mem=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 999999)
      nproc=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
      if [ "$nproc" = "0" ] && [ "$mem" -lt 1024 ]; then
        free_gpus+=($g)
      fi
    done
    echo "$(date '+%H:%M:%S') free GPUs: ${free_gpus[*]:-none} (need $MIN_FREE)"
    if [ ${#free_gpus[@]} -ge $MIN_FREE ]; then
      echo "Free GPUs: ${free_gpus[*]}"
      export FREE_GPUS="${free_gpus[*]}"
      return
    fi
    sleep 120
  done
}

launch_one() {
  local cfg="$1"
  local gpu="$2"
  local log="logs/distill_${cfg}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$cfg] launching on GPU $gpu -> $log"
  nohup env CUDA_VISIBLE_DEVICES=$gpu NPROC_PER_NODE=1 swift sft \
    --config configs/${cfg}.yaml \
    > "$log" 2>&1 &
  echo "[$cfg] PID=$!"
  echo "$!" > "logs/distill_${cfg}.pid"
}

wait_for_gpus
read -ra FREE <<< "$FREE_GPUS"

# Launch in order: 4B (priority), 2B, 0.8B
CONFIGS=(sft_distill_4b sft_distill_2b sft_distill_0p8b)
for i in "${!CONFIGS[@]}"; do
  if [ $i -ge ${#FREE[@]} ]; then
    echo "Not enough GPUs for ${CONFIGS[$i]}, will launch later"
    break
  fi
  launch_one "${CONFIGS[$i]}" "${FREE[$i]}"
done

echo "Launched $((${#CONFIGS[@]} < ${#FREE[@]} ? ${#CONFIGS[@]} : ${#FREE[@]})) distillation jobs"
