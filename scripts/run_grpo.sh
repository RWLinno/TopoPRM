#!/bin/bash
set -euo pipefail
###############################################################################
# GRPO Training — single experiment with safety
#
# Usage: bash scripts/run_grpo.sh <config_name> [extra swift args...]
#   e.g.: bash scripts/run_grpo.sh grpo_main
#         bash scripts/run_grpo.sh grpo_outcome_only --num_train_epochs 2
#
# Env vars:
#   CUDA_VISIBLE_DEVICES  — GPUs to use (default: 0,1,2,3,4,5,6,7)
#   NPROC_PER_NODE        — number of training processes (default: 8)
#   GUARD_SHM_LIMIT_GB    — shared memory kill threshold in GB (default: 400)
#   SKIP_PREFLIGHT        — set 1 to skip GPU health check
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
source "$SCRIPT_DIR/gpu_guard.sh"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PATH="/mnt/users/conda_env/topoprm/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export WANDB_PROJECT="${WANDB_PROJECT:-topoprm}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-grpo}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export TOPO_REWARD_LOG_EVERY="${TOPO_REWARD_LOG_EVERY:-10}"

CONFIG_NAME="${1:?Usage: $0 <config_name> [extra args...]}"
shift
CONFIG="configs/${CONFIG_NAME}.yaml"
[ ! -f "$CONFIG" ] && echo "[ERROR] Config not found: $CONFIG" && exit 1
mkdir -p output

# Safety: env + pre-flight + cleanup
export_safe_env
[ "${SKIP_PREFLIGHT:-0}" != "1" ] && { gpu_preflight || exit 1; }
shm_cleanup

# Auto-resume from latest checkpoint
OUTPUT_DIR=$(grep -E '^\s*output_dir:' "$CONFIG" | awk '{print $2}' | tr -d '"' | tr -d "'")
RESUME_ARG=""
if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
    LATEST=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    [ -n "$LATEST" ] && RESUME_ARG="--resume_from_checkpoint $LATEST" && echo "[run_grpo] Resuming from $LATEST"
fi

LOG="output/${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S).log"
register_cleanup "$CONFIG_NAME"

# Auto-resolve SFT adapter to avoid stale config paths.
SFT_ADAPTER="${SFT_ADAPTER:-}"
if [ -z "$SFT_ADAPTER" ]; then
    SFT_ADAPTER=$(find output/sft -maxdepth 3 -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1 || true)
fi
ADAPTER_ARG=""
if [ -n "$SFT_ADAPTER" ] && [ -d "$SFT_ADAPTER" ]; then
    ADAPTER_ARG="--adapters $SFT_ADAPTER"
fi

echo "══════════════════════════════════════════"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GRPO: $CONFIG_NAME"
echo "  GPUs: $CUDA_VISIBLE_DEVICES  NPROC: $NPROC_PER_NODE"
echo "  SHM limit: ${GUARD_SHM_LIMIT_GB}GB"
echo "  Adapter: ${SFT_ADAPTER:-from-config}"
if [ -n "${WANDB_API_KEY:-}" ]; then
  echo "  W&B: enabled (project=${WANDB_PROJECT}, group=${WANDB_RUN_GROUP})"
else
  echo "  W&B: WANDB_API_KEY not set, relying on existing wandb login/session"
fi
echo "  Reward component log every: ${TOPO_REWARD_LOG_EVERY} calls"
echo "  Log: $LOG"
echo "══════════════════════════════════════════"

setsid swift rlhf --rlhf_type grpo --config "$CONFIG" $RESUME_ARG $ADAPTER_ARG "$@" 2>&1 | tee "$LOG" &
GUARDED_PID=$!
save_pid_file "$CONFIG_NAME" "$GUARDED_PID"
start_shm_watchdog "$GUARDED_PID"
start_gpu_watchdog "$GUARDED_PID"

echo "[run_grpo] PID=$GUARDED_PID, watchdogs active."
wait $GUARDED_PID
EXIT_CODE=$?
GUARDED_PID=""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] $CONFIG_NAME exited with code $EXIT_CODE"
exit $EXIT_CODE
