#!/usr/bin/env bash
# Usage: v3b_wait_and_run.sh GPU JOBFILE WAIT_FOR_PID
#
# Polls WAIT_FOR_PID. When that process (a still-running v3 short-group bench
# on the target GPU) exits, launches the v3b queue for GPU.
#
# Also kills any lingering rerun_unified_v3.sh ${GPU} orchestrator shell so
# it won't spawn a new legacy bench after our wait completes.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:?usage: $0 GPU JOBFILE WAIT_PID}"
JOBFILE="${2:?missing jobfile}"
WAIT_PID="${3:?missing wait pid}"

echo "[$(date '+%H:%M:%S')] watchdog GPU${GPU} waiting for PID=${WAIT_PID} to exit"
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
done
echo "[$(date '+%H:%M:%S')] watchdog GPU${GPU} PID=${WAIT_PID} is gone, cleaning legacy orchestrator"

# kill the legacy rerun_unified_v3.sh <gpu> orchestrator so it can't queue
# a new bench onto this GPU
pgrep -f "rerun_unified_v3.sh ${GPU}$" | xargs -r kill 2>/dev/null || true
sleep 2

echo "[$(date '+%H:%M:%S')] watchdog GPU${GPU} launching v3b queue <- ${JOBFILE}"
exec bash scripts/rerun_unified_v3b.sh queue "$GPU" "$JOBFILE"
