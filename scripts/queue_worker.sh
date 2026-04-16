#!/usr/bin/env bash
set -euo pipefail

# Single-GPU serial worker for experiment_matrix.jsonl.
# Usage:
#   bash scripts/queue_worker.sh <gpu_id> <matrix_jsonl> <registry_jsonl> [dry_run]

GPU_ID="${1:?need gpu_id}"
MATRIX_PATH="${2:?need matrix jsonl path}"
REGISTRY_PATH="${3:?need registry jsonl path}"
DRY_RUN="${4:-0}"

LOCK_PATH="${REGISTRY_PATH}.lock"
WORKER_NAME="gpu${GPU_ID}"

append_event() {
  local task_id="$1"
  local status="$2"
  local attempt="$3"
  local exit_code="$4"
  local duration_sec="$5"
  local stage="$6"
  local command="$7"
  local artifacts_json="$8"

  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  local line
  line="$(python3 - <<'PY' "$ts" "$WORKER_NAME" "$task_id" "$status" "$attempt" "$exit_code" "$duration_sec" "$GPU_ID" "$stage" "$command" "$artifacts_json"
import json
import sys

ts, worker, task_id, status, attempt, exit_code, duration_sec, gpu, stage, command, artifacts_json = sys.argv[1:]
try:
    artifacts = json.loads(artifacts_json)
except Exception:
    artifacts = []
payload = {
    "ts": ts,
    "worker": worker,
    "task_id": task_id,
    "status": status,
    "attempt": int(attempt),
    "exit_code": int(exit_code) if exit_code not in {"", "NA"} else None,
    "duration_sec": float(duration_sec) if duration_sec not in {"", "NA"} else None,
    "gpu": int(gpu),
    "stage": stage,
    "command": command,
    "artifacts": artifacts,
}
print(json.dumps(payload, ensure_ascii=False))
PY
)"
  {
    flock 9
    echo "$line" >> "$REGISTRY_PATH"
  } 9>"$LOCK_PATH"
}

deps_state() {
  local deps_json="$1"
  python3 - <<'PY' "$REGISTRY_PATH" "$deps_json"
import json
import sys
from pathlib import Path

registry = Path(sys.argv[1])
deps = json.loads(sys.argv[2]) if sys.argv[2] else []
if not deps:
    print("ready")
    raise SystemExit(0)

latest = {}
if registry.exists():
    for line in registry.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[obj.get("task_id")] = obj.get("status")

for dep in deps:
    status = latest.get(dep)
    if status in {"failed_final", "skipped_dep_failed"}:
        print("failed")
        raise SystemExit(0)
    if status not in {"success", "success_cached"}:
        print("pending")
        raise SystemExit(0)

print("ready")
PY
}

outputs_ready() {
  local outputs_json="$1"
  python3 - <<'PY' "$outputs_json"
import json
import sys
from pathlib import Path
import glob

outputs = json.loads(sys.argv[1]) if sys.argv[1] else []
if not outputs:
    print("no")
    raise SystemExit(0)
all_ok = True
for item in outputs:
    if any(ch in item for ch in ["*", "?", "["]):
        if not glob.glob(item):
            all_ok = False
            break
    else:
        if not Path(item).exists():
            all_ok = False
            break
print("yes" if all_ok else "no")
PY
}

# Stagger vLLM initialization across GPUs to avoid memory-profiling race
STAGGER_SEC=$((GPU_ID * 30))
if [ "$DRY_RUN" != "1" ] && [ "$STAGGER_SEC" -gt 0 ]; then
  echo "[worker:$WORKER_NAME] stagger wait ${STAGGER_SEC}s to avoid vLLM memory-profiling race"
  sleep "$STAGGER_SEC"
fi

echo "[worker:$WORKER_NAME] start (dry_run=$DRY_RUN)"

while IFS=$'\t' read -r task_id stage gpu deps_json retries outputs_json tags_json command; do
  [ -z "${task_id:-}" ] && continue
  [ "$gpu" != "$GPU_ID" ] && continue

  dep_status="$(deps_state "$deps_json")"
  while [ "$dep_status" = "pending" ]; do
    sleep 20
    dep_status="$(deps_state "$deps_json")"
  done
  if [ "$dep_status" = "failed" ]; then
    append_event "$task_id" "skipped_dep_failed" 1 "NA" "NA" "$stage" "$command" "$outputs_json"
    echo "[worker:$WORKER_NAME] skip $task_id (dependency failed)"
    continue
  fi

  if [ "$(outputs_ready "$outputs_json")" = "yes" ]; then
    append_event "$task_id" "success_cached" 1 0 0 "$stage" "$command" "$outputs_json"
    echo "[worker:$WORKER_NAME] cache-hit $task_id"
    continue
  fi

  max_retries="${retries:-1}"
  max_attempts=$((max_retries + 1))
  attempt=1
  while [ "$attempt" -le "$max_attempts" ]; do
    append_event "$task_id" "running" "$attempt" "NA" "NA" "$stage" "$command" "$outputs_json"
    start_ts="$(date +%s)"
    if [ "$DRY_RUN" = "1" ]; then
      sleep 1
      exit_code=0
    else
      MASTER_PORT=$((29500 + GPU_ID)) CUDA_VISIBLE_DEVICES="$GPU_ID" bash -lc "$command"
      exit_code=$?
    fi
    end_ts="$(date +%s)"
    duration=$((end_ts - start_ts))

    if [ "$exit_code" -eq 0 ]; then
      append_event "$task_id" "success" "$attempt" "$exit_code" "$duration" "$stage" "$command" "$outputs_json"
      echo "[worker:$WORKER_NAME] success $task_id (attempt=$attempt)"
      break
    fi

    if [ "$attempt" -lt "$max_attempts" ]; then
      append_event "$task_id" "failed_retry" "$attempt" "$exit_code" "$duration" "$stage" "$command" "$outputs_json"
      echo "[worker:$WORKER_NAME] failed retrying $task_id (attempt=$attempt/$max_attempts)"
      sleep 15
    else
      append_event "$task_id" "failed_final" "$attempt" "$exit_code" "$duration" "$stage" "$command" "$outputs_json"
      echo "[worker:$WORKER_NAME] failed_final $task_id"
    fi
    attempt=$((attempt + 1))
  done
done < <(python3 - <<'PY' "$MATRIX_PATH"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"matrix not found: {path}")

for line in path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    obj = json.loads(line)
    fields = [
        obj["id"],
        obj.get("stage", "unknown"),
        str(obj["gpu"]),
        json.dumps(obj.get("deps", []), ensure_ascii=False),
        str(obj.get("retries", 1)),
        json.dumps(obj.get("expected_outputs", []), ensure_ascii=False),
        json.dumps(obj.get("tags", []), ensure_ascii=False),
        obj["command"],
    ]
    print("\t".join(fields))
PY
)

echo "[worker:$WORKER_NAME] done"

