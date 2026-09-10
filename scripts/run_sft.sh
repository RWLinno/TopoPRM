#!/bin/bash
set -euo pipefail
###############################################################################
# SFT Training — Qwen3-32B
#
# Usage: bash scripts/run_sft.sh [extra swift args...]
#
# Env vars:
#   CUDA_VISIBLE_DEVICES  — GPUs to use (default: 0,1,2,3,4,5,6,7)
#   NPROC_PER_NODE        — number of training processes (default: 8)
#   SKIP_PREFLIGHT        — set 1 to skip GPU health check
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
source "$SCRIPT_DIR/gpu_guard.sh"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PATH="${PYTHON_ENV_BIN}:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

CONFIG="configs/sft.yaml"
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
    LATEST=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    [ -n "$LATEST" ] && RESUME_ARG="--resume_from_checkpoint $LATEST" && echo "[run_sft] Resuming from $LATEST"
fi

LOG="output/sft_$(date +%Y%m%d_%H%M%S).log"
register_cleanup "sft"

echo "══════════════════════════════════════════"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SFT Training"
echo "  GPUs: $CUDA_VISIBLE_DEVICES  NPROC: $NPROC_PER_NODE"
echo "  Log:  $LOG"
echo "══════════════════════════════════════════"

setsid swift sft --config "$CONFIG" $RESUME_ARG "$@" 2>&1 | tee "$LOG" &
GUARDED_PID=$!
save_pid_file "sft" "$GUARDED_PID"
start_shm_watchdog "$GUARDED_PID"
start_gpu_watchdog "$GUARDED_PID"

wait $GUARDED_PID
EXIT_CODE=$?
GUARDED_PID=""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SFT exited with code $EXIT_CODE"
exit $EXIT_CODE
