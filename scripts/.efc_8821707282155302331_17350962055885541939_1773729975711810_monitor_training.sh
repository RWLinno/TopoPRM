#!/bin/bash
###############################################################################
# Monitor training: GPU usage, system memory, shared memory
# Usage: bash scripts/monitor_training.sh [interval_seconds]
#   interval: polling interval in seconds (default: 60)
# Output: prints to stdout, also appends to output/monitor.log
###############################################################################

INTERVAL="${1:-60}"
LOG="output/monitor.log"
mkdir -p output

echo "Monitoring started (interval=${INTERVAL}s). Ctrl+C to stop."
echo "Log: $LOG"

while true; do
    TS=$(date '+%Y-%m-%d %H:%M:%S')

    # System memory
    MEM_TOTAL=$(free -g | awk '/Mem:/{print $2}')
    MEM_USED=$(free -g | awk '/Mem:/{print $3}')
    MEM_FREE=$(free -g | awk '/Mem:/{print $4}')
    SHM_USED=$(df -BG /dev/shm 2>/dev/null | awk 'NR==2{print $3}' || echo "?")

    # GPU summary
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | tr '\n' '/' | sed 's/\/$//')

    # Training process check
    TRAIN_PID=$(pgrep -f "swift.*rlhf" 2>/dev/null | head -1 || echo "none")

    LINE="[$TS] RAM: ${MEM_USED}/${MEM_TOTAL}G (free:${MEM_FREE}G) SHM:${SHM_USED} GPU_MiB:${GPU_MEM} PID:${TRAIN_PID}"
    echo "$LINE"
    echo "$LINE" >> "$LOG"

    # Warn if shared memory > 400GB
    SHM_NUM=$(echo "$SHM_USED" | tr -dc '0-9')
    if [ -n "$SHM_NUM" ] && [ "$SHM_NUM" -gt 400 ] 2>/dev/null; then
        echo "[WARNING] Shared memory > 400GB! Risk of OOM!"
        echo "[WARNING] $TS Shared memory > 400GB!" >> "$LOG"
    fi

    sleep "$INTERVAL"
done
