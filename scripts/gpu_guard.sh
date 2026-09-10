#!/bin/bash
###############################################################################
# gpu_guard.sh — GPU & shared-memory safety utilities
#
# Source this file in any training script:
#   source "$(dirname "$0")/gpu_guard.sh"
#
# Provides:
#   gpu_preflight         — verify GPUs are clean before training
#   shm_cleanup           — report SHM usage; remove stale handles only when opted in
#   start_shm_watchdog    — background monitor, kills $GUARDED_PID on SHM overflow
#   start_gpu_watchdog    — background monitor, warns on near-full GPU memory
#   register_cleanup      — install EXIT/INT/TERM trap for graceful shutdown
#   save_pid_file / remove_pid_file — PID-file management
#   export_safe_env       — export memory-safety env vars
###############################################################################

GUARD_SHM_LIMIT_GB="${GUARD_SHM_LIMIT_GB:-400}"
GUARD_GPU_LEAK_MB="${GUARD_GPU_LEAK_MB:-1000}"
GUARD_POLL_INTERVAL="${GUARD_POLL_INTERVAL:-30}"
GUARD_PID_DIR="${GUARD_PID_DIR:-/tmp/topoprm_pids}"
GUARD_CLEAN_STALE_SHM="${GUARD_CLEAN_STALE_SHM:-0}"
GUARDED_PID=""
_WATCHDOG_PIDS=()

gpu_preflight() {
    local devices="${CUDA_VISIBLE_DEVICES:-all}"
    echo "[gpu_guard] Pre-flight check on GPUs: $devices"

    local indices
    if [ "$devices" = "all" ]; then
        indices=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')
    else
        indices=$(echo "$devices" | tr ',' ' ')
    fi

    local dirty=0
    for idx in $indices; do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')
        if [ -n "$used" ] && [ "$used" -gt "$GUARD_GPU_LEAK_MB" ]; then
            echo "[gpu_guard] WARNING: GPU $idx has ${used} MiB in use (threshold: ${GUARD_GPU_LEAK_MB} MiB)"
            local pids
            pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$idx" 2>/dev/null | tr -d ' ')
            if [ -n "$pids" ] && [ "$pids" != "[N/A]" ]; then
                echo "[gpu_guard]   Processes: $pids"
            else
                echo "[gpu_guard]   No process found — likely leaked (requires restart)"
            fi
            dirty=1
        else
            echo "[gpu_guard]   GPU $idx: ${used:-0} MiB OK"
        fi
    done

    if [ "$dirty" -eq 1 ]; then
        echo "[gpu_guard] Some GPUs have residual memory. Training may OOM."
        echo "[gpu_guard]   Fix: (1) kill PIDs  (2) exclude via CUDA_VISIBLE_DEVICES  (3) restart"
        return 1
    fi
    echo "[gpu_guard] All GPUs clean."
    return 0
}

shm_cleanup() {
    if [ "$GUARD_CLEAN_STALE_SHM" != "1" ]; then
        echo "[gpu_guard] Shared-memory cleanup disabled (set GUARD_CLEAN_STALE_SHM=1 to opt in)."
        df -h /dev/shm 2>/dev/null | tail -1 || true
        return 0
    fi

    echo "[gpu_guard] Cleaning stale IPC handles in /dev/shm..."
    local count=0
    shopt -s nullglob
    for pattern in "nccl-*" "vllm-*" "cuda-*" "torch_*"; do
        local matches=(/dev/shm/${pattern})
        local found=${#matches[@]}
        if [ "$found" -gt 0 ]; then
            rm -f "${matches[@]}"
            count=$((count + found))
        fi
    done
    shopt -u nullglob
    echo "[gpu_guard] Removed $count stale handle(s)."
    local shm_used_kb
    shm_used_kb=$(df /dev/shm 2>/dev/null | awk 'NR==2{print $3}')
    local shm_used_gb=$((shm_used_kb / 1024 / 1024))
    echo "[gpu_guard] /dev/shm usage: ${shm_used_gb} GB"
}

start_shm_watchdog() {
    local target_pid="${1:?start_shm_watchdog requires a PID}"
    (
        while kill -0 "$target_pid" 2>/dev/null; do
            local shm_kb
            shm_kb=$(df /dev/shm 2>/dev/null | awk 'NR==2{print $3}')
            local shm_gb=$((shm_kb / 1024 / 1024))
            if [ "$shm_gb" -gt "$GUARD_SHM_LIMIT_GB" ]; then
                echo "[gpu_guard] CRITICAL: /dev/shm=${shm_gb}GB exceeds ${GUARD_SHM_LIMIT_GB}GB!"
                echo "[gpu_guard] Killing process group $target_pid to prevent system OOM"
                kill -- -"$target_pid" 2>/dev/null || kill "$target_pid" 2>/dev/null
                sleep 5
                kill -9 -- -"$target_pid" 2>/dev/null || kill -9 "$target_pid" 2>/dev/null
                break
            fi
            sleep "$GUARD_POLL_INTERVAL"
        done
    ) &
    _WATCHDOG_PIDS+=($!)
}

start_gpu_watchdog() {
    local target_pid="${1:?start_gpu_watchdog requires a PID}"
    local devices="${CUDA_VISIBLE_DEVICES:-all}"
    (
        while kill -0 "$target_pid" 2>/dev/null; do
            local indices
            if [ "$devices" = "all" ]; then
                indices=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')
            else
                indices=$(echo "$devices" | tr ',' ' ')
            fi
            for idx in $indices; do
                local used total pct
                used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')
                total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')
                if [ -n "$used" ] && [ -n "$total" ] && [ "$total" -gt 0 ]; then
                    pct=$((used * 100 / total))
                    if [ "$pct" -gt 98 ]; then
                        echo "[gpu_guard] WARNING: GPU $idx at ${pct}% (${used}/${total} MiB)"
                    fi
                fi
            done
            sleep "$GUARD_POLL_INTERVAL"
        done
    ) &
    _WATCHDOG_PIDS+=($!)
}

save_pid_file() {
    local name="${1:?save_pid_file requires a name}"
    local pid="${2:?save_pid_file requires a PID}"
    mkdir -p "$GUARD_PID_DIR"
    echo "$pid" > "${GUARD_PID_DIR}/${name}.pid"
    echo "[gpu_guard] PID file: ${GUARD_PID_DIR}/${name}.pid (PID=$pid)"
}

remove_pid_file() {
    local name="${1:?remove_pid_file requires a name}"
    rm -f "${GUARD_PID_DIR}/${name}.pid"
}

register_cleanup() {
    local name="${1:-training}"
    trap "_do_cleanup '$name'" EXIT INT TERM
}

_do_cleanup() {
    local name="$1"
    echo ""
    echo "[gpu_guard] ================================================"
    echo "[gpu_guard] Cleanup: $name @ $(date '+%Y-%m-%d %H:%M:%S')"

    for wpid in "${_WATCHDOG_PIDS[@]}"; do
        kill "$wpid" 2>/dev/null
    done
    _WATCHDOG_PIDS=()

    if [ -n "$GUARDED_PID" ]; then
        echo "[gpu_guard] Terminating process group $GUARDED_PID..."
        kill -- -"$GUARDED_PID" 2>/dev/null || kill "$GUARDED_PID" 2>/dev/null
        sleep 3
        kill -9 -- -"$GUARDED_PID" 2>/dev/null || kill -9 "$GUARDED_PID" 2>/dev/null
        wait "$GUARDED_PID" 2>/dev/null
    fi

    shm_cleanup
    remove_pid_file "$name"

    echo "[gpu_guard] Post-cleanup GPU status:"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null || true
    echo "[gpu_guard] ================================================"
}

export_safe_env() {
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export TOKENIZERS_PARALLELISM=false
    echo "[gpu_guard] Safe env vars exported (PYTORCH_CUDA_ALLOC_CONF, NCCL, etc.)"
}
