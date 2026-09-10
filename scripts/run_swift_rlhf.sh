#!/bin/bash
set -euo pipefail
###############################################################################
# Generic wrapper around `swift rlhf` for GRPO / DAPO / GKD / OPD / OPSD.
#
# Usage:
#   bash scripts/run_swift_rlhf.sh <config_name> [rlhf_type] [extra swift args...]
#
# Examples:
#   bash scripts/run_swift_rlhf.sh grpo_hierarchical_qwen25_7b grpo
#   bash scripts/run_swift_rlhf.sh dapo_dr1_7b                  grpo   # DAPO
#   bash scripts/run_swift_rlhf.sh opd_dr1_7b_to_qwen3_4b       gkd    # OPD
#   bash scripts/run_swift_rlhf.sh opsd_dr1_7b                  gkd    # OPSD
#
# Notes:
#   * DAPO shares the `grpo` rlhf_type; the DAPO mode is activated inside the
#     yaml via loss_type: dapo + epsilon_high + dynamic_sampling.
#   * For OPD / OPSD we use rlhf_type=gkd (same as ms-swift's
#     on_policy_distillation.sh and rlhf/opsd/opsd.sh).
#   * TEACHER_ADAPTER=<path> will be forwarded as --teacher_adapters so you can
#     distill from "DR1-7B + TopoPRM" instead of vanilla DR1-7B.
#   * SFT_ADAPTER=<path> (same as run_grpo.sh) auto-loads the student's SFT
#     starting point.
#
# Env vars (same conventions as run_grpo.sh / run_sft.sh):
#   CUDA_VISIBLE_DEVICES   (default: 0,1,2,3,4,5,6,7)
#   NPROC_PER_NODE         (default: 8)
#   WANDB_PROJECT          (default: topoprm)
#   WANDB_RUN_GROUP        (default: <rlhf_type>)
#   SKIP_PREFLIGHT=1       skip GPU health check
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
source "$SCRIPT_DIR/gpu_guard.sh"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PATH="${PYTHON_ENV_BIN}:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export TOPO_REWARD_LOG_EVERY="${TOPO_REWARD_LOG_EVERY:-10}"

CONFIG_NAME="${1:?Usage: $0 <config_name> [rlhf_type] [extra swift args...]}"
shift || true
RLHF_TYPE="${1:-grpo}"
# Allow the caller to skip rlhf_type if they passed only "--foo bar"
if [[ "$RLHF_TYPE" == -* ]]; then
    RLHF_TYPE="grpo"
else
    shift || true
fi
export WANDB_PROJECT="${WANDB_PROJECT:-topoprm}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$RLHF_TYPE}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"

CONFIG="configs/${CONFIG_NAME}.yaml"
[ ! -f "$CONFIG" ] && echo "[ERROR] Config not found: $CONFIG" && exit 1
mkdir -p output

export_safe_env
[ "${SKIP_PREFLIGHT:-0}" != "1" ] && { gpu_preflight || exit 1; }
shm_cleanup

OUTPUT_DIR=$(grep -E '^\s*output_dir:' "$CONFIG" | awk '{print $2}' | tr -d '"' | tr -d "'")
RESUME_ARG=""
if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
    LATEST=$(find "$OUTPUT_DIR" -maxdepth 3 -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1 || true)
    if [ -n "$LATEST" ]; then
        RESUME_ARG="--resume_from_checkpoint $LATEST"
    fi
fi

# Auto-attach student SFT adapter when present (only meaningful for GRPO/DAPO).
SFT_ADAPTER="${SFT_ADAPTER:-}"
if [ -z "$SFT_ADAPTER" ] && [[ "$RLHF_TYPE" == "grpo" ]]; then
    SFT_ADAPTER=$(find output/sft -maxdepth 3 -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1 || true)
fi
ADAPTER_ARG=""
if [ -n "$SFT_ADAPTER" ] && [ -d "$SFT_ADAPTER" ]; then
    ADAPTER_ARG="--adapters $SFT_ADAPTER"
fi

TEACHER_ADAPTER_ARG=""
if [ -n "${TEACHER_ADAPTER:-}" ]; then
    TEACHER_ADAPTER_ARG="--teacher_adapters $TEACHER_ADAPTER"
fi

LOG="output/${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S).log"
register_cleanup "$CONFIG_NAME" 

echo "══════════════════════════════════════════"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] swift rlhf type=$RLHF_TYPE  config=$CONFIG_NAME"
echo "  GPUs: $CUDA_VISIBLE_DEVICES   NPROC: $NPROC_PER_NODE"
echo "  Student adapter: ${SFT_ADAPTER:-from-config}"
echo "  Teacher adapter: ${TEACHER_ADAPTER:-from-config}"
echo "  W&B project: $WANDB_PROJECT   group: $WANDB_RUN_GROUP"
echo "  Log: $LOG"
echo "══════════════════════════════════════════"

setsid swift rlhf \
    "$CONFIG" \
    --rlhf_type "$RLHF_TYPE" \
    $RESUME_ARG $ADAPTER_ARG $TEACHER_ADAPTER_ARG "$@" \
    2>&1 | tee "$LOG" &
GUARDED_PID=$!
save_pid_file "$CONFIG_NAME" "$GUARDED_PID"
start_shm_watchdog "$GUARDED_PID"
start_gpu_watchdog "$GUARDED_PID"

echo "[run_swift_rlhf] PID=$GUARDED_PID, watchdogs active."
wait $GUARDED_PID
EXIT_CODE=$?
GUARDED_PID=""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] $CONFIG_NAME exited with code $EXIT_CODE"
exit $EXIT_CODE
