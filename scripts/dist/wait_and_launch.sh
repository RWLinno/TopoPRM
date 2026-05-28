#!/usr/bin/env bash
# ============================================================================
# wait_and_launch.sh
#
# Wait until specific GPUs (or all GPUs the worker would land on) are idle,
# then launch the given TASK_IDs through launch_local.sh.
#
# Usage:
#   bash scripts/dist/wait_and_launch.sh TASK_IDS...
#   GPU_FREE_MEM_MB=5000 IDLE_POLL_SEC=60 bash scripts/dist/wait_and_launch.sh 8 9
#
# Default: a GPU is considered "free" if memory.used < 5000 MiB. Used so that
# we don't kick off a heavy bench job onto a card that still has the previous
# bench's model resident.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

NUM_GPUS_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l)"
[[ -z "$NUM_GPUS_PER_NODE" || "$NUM_GPUS_PER_NODE" == "0" ]] && NUM_GPUS_PER_NODE=1
GPU_FREE_MEM_MB="${GPU_FREE_MEM_MB:-5000}"
IDLE_POLL_SEC="${IDLE_POLL_SEC:-60}"

if [[ $# -eq 0 ]]; then
    echo "usage: $0 TASK_ID [TASK_ID ...]" >&2
    exit 2
fi

TASKS=("$@")

is_gpu_free() {
    local gpu_id=$1
    # The pipe-to-awk pattern is safe because we don't have pipefail here
    # (this script intentionally does not `set -o pipefail`).
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | head -1 | tr -d ' MiB' )
    if [[ -z "$used" ]]; then
        return 1
    fi
    [[ "$used" -lt "$GPU_FREE_MEM_MB" ]]
}

# For each task, derive its target GPU and wait until that GPU is free.
PENDING=("${TASKS[@]}")
while [[ ${#PENDING[@]} -gt 0 ]]; do
    READY=()
    STILL=()
    for tid in "${PENDING[@]}"; do
        gpu=$(( tid % NUM_GPUS_PER_NODE ))
        if is_gpu_free "$gpu"; then
            READY+=("$tid")
        else
            STILL+=("$tid")
        fi
    done

    if [[ ${#READY[@]} -gt 0 ]]; then
        echo "[wait_and_launch] ready tasks: ${READY[*]}"
        bash scripts/dist/launch_local.sh "${READY[@]}" &
    fi

    PENDING=("${STILL[@]}")
    if [[ ${#PENDING[@]} -eq 0 ]]; then
        break
    fi

    echo "[wait_and_launch] still pending: ${PENDING[*]}  (sleep ${IDLE_POLL_SEC}s)"
    sleep "$IDLE_POLL_SEC"
done

wait
echo "[wait_and_launch] all tasks dispatched"
