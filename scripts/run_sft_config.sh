#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# SFT Training by named config (same safety behavior as run_sft.sh)
#
# Usage: bash scripts/run_sft_config.sh <config_name> [extra swift args...]
# Example: bash scripts/run_sft_config.sh sft_qwen25_7b
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
source "$SCRIPT_DIR/gpu_guard.sh"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
TOPOPRM_ENV_BIN="${TOPOPRM_ENV_BIN:-${PYTHON_ENV_BIN}}"
if [ -x "$TOPOPRM_ENV_BIN/swift" ]; then
  export PATH="$TOPOPRM_ENV_BIN:$PATH"
fi
export DS_IGNORE_CUDA_DETECTION="${DS_IGNORE_CUDA_DETECTION:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

CONFIG_NAME="${1:?Usage: $0 <config_name> [extra args...]}"
shift
CONFIG="configs/${CONFIG_NAME}.yaml"
[ ! -f "$CONFIG" ] && echo "[ERROR] Config not found: $CONFIG" && exit 1

mkdir -p output
export_safe_env
[ "${SKIP_PREFLIGHT:-0}" != "1" ] && { gpu_preflight || exit 1; }
shm_cleanup

OUTPUT_DIR="${SFT_OUTPUT_DIR:-$(grep -E '^\s*output_dir:' "$CONFIG" | awk '{print $2}' | tr -d '"' | tr -d "'")}"
RESUME_ARG=""
if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
  LATEST=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
  [ -n "$LATEST" ] && RESUME_ARG="--resume_from_checkpoint $LATEST" && echo "[run_sft_config] Resuming from $LATEST"
fi

LOG="output/${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S).log"
register_cleanup "$CONFIG_NAME"
OUTPUT_ARG=()
if [ -n "${SFT_OUTPUT_DIR:-}" ]; then
  OUTPUT_ARG+=(--output_dir "$SFT_OUTPUT_DIR")
fi

echo "══════════════════════════════════════════"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SFT: $CONFIG_NAME"
echo "  GPUs: $CUDA_VISIBLE_DEVICES  NPROC: $NPROC_PER_NODE"
echo "  Output: $OUTPUT_DIR"
echo "  Log:  $LOG"
echo "══════════════════════════════════════════"

setsid "$TOPOPRM_ENV_BIN/python" -m swift.cli.main sft "$CONFIG" $RESUME_ARG "${OUTPUT_ARG[@]}" "$@" > >(tee "$LOG") 2>&1 &
GUARDED_PID=$!
save_pid_file "$CONFIG_NAME" "$GUARDED_PID"
start_shm_watchdog "$GUARDED_PID"
start_gpu_watchdog "$GUARDED_PID"

set +e
wait "$GUARDED_PID"
EXIT_CODE=$?
set -e
GUARDED_PID=""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] $CONFIG_NAME exited with code $EXIT_CODE"
exit $EXIT_CODE
