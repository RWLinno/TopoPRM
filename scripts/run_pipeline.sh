#!/bin/bash
set -euo pipefail
###############################################################################
# Full GRPO Experiment Pipeline — runs all experiments sequentially with safety
#
# Usage:
#   bash scripts/run_pipeline.sh [mode]
#     mode: "colocate" (default) — vLLM colocated with training on same GPUs
#           "server"             — separate vLLM server (GPUs 0-3) + training (GPUs 4-7)
#
#   nohup bash scripts/run_pipeline.sh colocate > output/pipeline.log 2>&1 &
#
# Env vars:
#   CUDA_VISIBLE_DEVICES  — GPUs for colocate mode (default: 0,1,2,3,4,5,6,7)
#   NPROC_PER_NODE        — training processes (default: 8)
#   SERVER_GPUS           — rollout server GPUs in server mode (default: 0,1,2,3)
#   TRAIN_GPUS            — training GPUs in server mode (default: 4,5,6,7)
#   EXPERIMENTS           — space-separated list (default: grpo_main grpo_outcome_only grpo_no_topo grpo_no_continuity)
#   GUARD_SHM_LIMIT_GB    — SHM kill threshold in GB (default: 400)
#   SKIP_PREFLIGHT        — set 1 to skip GPU check
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
source "$SCRIPT_DIR/gpu_guard.sh"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PATH="/mnt/users/conda_env/topoprm/bin:$PATH"

MODE="${1:-colocate}"
SERVER_GPUS="${SERVER_GPUS:-0,1,2,3}"
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
SERVER_PORT=8000
SERVER_PID=""

EXPERIMENTS=(${EXPERIMENTS:-grpo_main grpo_outcome_only grpo_no_topo grpo_no_continuity})
COMPLETED=()
FAILED=()

# Cleanup handler — kills server + training + watchdogs
pipeline_cleanup() {
    echo ""
    echo "[pipeline] ════════════════════════════════════════"
    echo "[pipeline] Shutdown @ $(date '+%Y-%m-%d %H:%M:%S')"
    for wpid in "${_WATCHDOG_PIDS[@]}"; do kill "$wpid" 2>/dev/null; done
    _WATCHDOG_PIDS=()
    if [ -n "$GUARDED_PID" ]; then
        kill -- -"$GUARDED_PID" 2>/dev/null || kill "$GUARDED_PID" 2>/dev/null
        sleep 3; kill -9 -- -"$GUARDED_PID" 2>/dev/null; wait "$GUARDED_PID" 2>/dev/null
    fi
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null; sleep 2; kill -9 "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null
    fi
    shm_cleanup
    remove_pid_file "pipeline"; remove_pid_file "rollout_server"
    echo "[pipeline] Results: OK=[${COMPLETED[*]:-}] FAIL=[${FAILED[*]:-}]"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null || true
    echo "[pipeline] ════════════════════════════════════════"
}
trap pipeline_cleanup EXIT INT TERM

# ── Pre-flight ──
export_safe_env
mkdir -p output

if [ "$MODE" = "server" ]; then
    export CUDA_VISIBLE_DEVICES="$SERVER_GPUS,$TRAIN_GPUS"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

[ "${SKIP_PREFLIGHT:-0}" != "1" ] && { gpu_preflight || exit 1; }
shm_cleanup

echo "══════════════════════════════════════════"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] TopoPRM GRPO Pipeline"
echo "  Mode:        $MODE"
echo "  Experiments: ${EXPERIMENTS[*]}"
if [ "$MODE" = "server" ]; then
    echo "  Server GPUs: $SERVER_GPUS  |  Train GPUs: $TRAIN_GPUS"
else
    echo "  GPUs: $CUDA_VISIBLE_DEVICES  NPROC: ${NPROC_PER_NODE:-8}"
fi
echo "══════════════════════════════════════════"

# ── Launch rollout server (server mode only) ──
if [ "$MODE" = "server" ]; then
    SFT_ADAPTER=$(ls -d output/sft*/checkpoint-* 2>/dev/null | sort -V | tail -1)
    [ -z "$SFT_ADAPTER" ] && echo "[ERROR] No SFT checkpoint found" && exit 1

    echo "[pipeline] Launching vLLM server on GPUs $SERVER_GPUS..."
    TP=$(echo "$SERVER_GPUS" | tr ',' '\n' | wc -l)
    CUDA_VISIBLE_DEVICES="$SERVER_GPUS" swift rollout \
        --model Qwen/Qwen3-32B --adapters "$SFT_ADAPTER" \
        --vllm_tensor_parallel_size "$TP" \
        --vllm_gpu_memory_utilization 0.9 --vllm_max_model_len 4096 \
        --vllm_enable_prefix_caching false \
        --vllm_enable_lora true --vllm_max_lora_rank 64 \
        --port "$SERVER_PORT" --max_new_tokens 2048 \
        > output/rollout_server.log 2>&1 &
    SERVER_PID=$!
    save_pid_file "rollout_server" "$SERVER_PID"

    MAX_WAIT=600; WAITED=0
    while ! curl -s "http://127.0.0.1:${SERVER_PORT}/v1/models" > /dev/null 2>&1; do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "[ERROR] Server crashed"; tail -20 output/rollout_server.log; exit 1; }
        sleep 10; WAITED=$((WAITED+10))
        [ "$WAITED" -ge "$MAX_WAIT" ] && echo "[ERROR] Server timeout" && exit 1
        echo "  ... waiting (${WAITED}s)"
    done
    echo "[pipeline] Server ready."
fi

# ── Run experiments sequentially ──
for exp in "${EXPERIMENTS[@]}"; do
    CONFIG="configs/${exp}.yaml"
    [ ! -f "$CONFIG" ] && echo "[pipeline] SKIP $exp (no config)" && FAILED+=("$exp") && continue

    echo ""
    echo "────────────────────────────────────────"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $exp"
    echo "────────────────────────────────────────"
    shm_cleanup

    # Auto-resume
    OUTPUT_DIR=$(grep -E '^\s*output_dir:' "$CONFIG" | awk '{print $2}' | tr -d '"' | tr -d "'")
    RESUME_ARG=""
    if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
        LATEST=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
        [ -n "$LATEST" ] && RESUME_ARG="--resume_from_checkpoint $LATEST" && echo "[pipeline] Resume: $LATEST"
    fi

    if [ "$MODE" = "server" ]; then
        export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
        export NPROC_PER_NODE=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)
    else
        export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
    fi

    # Prefer latest SFT adapter unless user explicitly sets SFT_ADAPTER.
    SFT_ADAPTER_RESOLVED="${SFT_ADAPTER:-}"
    if [ -z "$SFT_ADAPTER_RESOLVED" ]; then
        SFT_ADAPTER_RESOLVED=$(ls -d output/sft/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    fi
    ADAPTER_ARG=""
    if [ -n "$SFT_ADAPTER_RESOLVED" ] && [ -d "$SFT_ADAPTER_RESOLVED" ]; then
        ADAPTER_ARG="--adapters $SFT_ADAPTER_RESOLVED"
    fi

    setsid swift rlhf --rlhf_type grpo --config "$CONFIG" $RESUME_ARG $ADAPTER_ARG 2>&1 | tee "output/${exp}_$(date +%Y%m%d_%H%M%S).log" &
    GUARDED_PID=$!
    save_pid_file "$exp" "$GUARDED_PID"
    start_shm_watchdog "$GUARDED_PID"
    start_gpu_watchdog "$GUARDED_PID"

    wait $GUARDED_PID; EXP_EXIT=$?; GUARDED_PID=""

    for wpid in "${_WATCHDOG_PIDS[@]}"; do kill "$wpid" 2>/dev/null; done
    _WATCHDOG_PIDS=()
    remove_pid_file "$exp"

    if [ "$EXP_EXIT" -eq 0 ]; then
        COMPLETED+=("$exp"); echo "[$(date '+%H:%M:%S')] $exp OK"
    else
        FAILED+=("$exp"); echo "[$(date '+%H:%M:%S')] $exp FAILED (exit=$EXP_EXIT)"
    fi
    sleep 10
done

echo ""
echo "══════════════════════════════════════════"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline Complete"
echo "  OK:   ${COMPLETED[*]:-none}"
echo "  FAIL: ${FAILED[*]:-none}"
echo "══════════════════════════════════════════"
[ ${#FAILED[@]} -gt 0 ] && exit 1
exit 0
