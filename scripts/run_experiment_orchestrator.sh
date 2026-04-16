#!/usr/bin/env bash
set -euo pipefail

# Unified orchestrator:
# - GPUs run in parallel (default: 0,1,2,3)
# - per-GPU tasks run serially
#
# Usage:
#   bash scripts/run_experiment_orchestrator.sh
#   bash scripts/run_experiment_orchestrator.sh --dry-run
#   bash scripts/run_experiment_orchestrator.sh --gpus 0,1 --matrix scripts/experiment_matrix.jsonl

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MATRIX_PATH="scripts/experiment_matrix.jsonl"
GPU_LIST="0,1,2,3"
DRY_RUN=0
REGISTRY_PATH="output/analysis/experiment_registry.jsonl"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --matrix)
      MATRIX_PATH="$2"
      shift 2
      ;;
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --registry)
      REGISTRY_PATH="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "[orchestrator] unknown arg: $1"
      exit 1
      ;;
  esac
done

mkdir -p output/analysis logs
: > "$REGISTRY_PATH"

echo "══════════════════════════════════════════"
echo " TopoPRM Unified Orchestrator"
echo " Matrix:   $MATRIX_PATH"
echo " GPUs:     $GPU_LIST"
echo " Dry-run:  $DRY_RUN"
echo " Registry: $REGISTRY_PATH"
echo "══════════════════════════════════════════"

PIDS=()
for gpu in $(echo "$GPU_LIST" | tr ',' ' '); do
  worker_log="logs/orchestrator_gpu${gpu}.log"
  nohup bash "$SCRIPT_DIR/queue_worker.sh" "$gpu" "$MATRIX_PATH" "$REGISTRY_PATH" "$DRY_RUN" > "$worker_log" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  echo "[orchestrator] worker gpu${gpu} pid=${pid} log=${worker_log}"
done

overall_code=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    overall_code=1
  fi
done

python3 - <<'PY' "$REGISTRY_PATH"
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
counter = Counter()
latest = {}

if path.exists():
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[obj.get("task_id")] = obj.get("status")

for status in latest.values():
    counter[status] += 1

print("══════════════════════════════════════════")
print(" Orchestrator summary")
print("══════════════════════════════════════════")
for key in sorted(counter):
    print(f" {key}: {counter[key]}")
print(f" total_tasks: {sum(counter.values())}")
PY

exit "$overall_code"

