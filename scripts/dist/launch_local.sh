#!/usr/bin/env bash
# ============================================================================
# launch_local.sh
#
# Fan a slice of the eval manifest out across the local GPUs of one node.
# Each picked TASK_ID is dispatched to run_eval_worker.sh (which pins itself
# to a single GPU based on TASK_ID % NUM_GPUS_PER_NODE).
#
# Usage:
#   bash scripts/dist/launch_local.sh                  # all tasks in manifest
#   bash scripts/dist/launch_local.sh 0 1 2 3 4 5      # subset (e.g. Wave A)
#   TASK_IDS="0 1 2 3 4 5" bash scripts/dist/launch_local.sh
#   PARALLEL=6 bash scripts/dist/launch_local.sh 0..5  # 6-way parallel
#
# Range syntax `N..M` is expanded inclusively.
#
# This is also the recommended dry-run target before invoking the Slurm/Volc/Ray
# wrappers — the worker contract is identical.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="${MANIFEST:-configs/dist/eval_manifest.tsv}"
if [[ -z "${NUM_GPUS_PER_NODE:-}" ]]; then
    NUM_GPUS_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l)"
    if [[ -z "$NUM_GPUS_PER_NODE" || "$NUM_GPUS_PER_NODE" == "0" ]]; then
        NUM_GPUS_PER_NODE=1
    fi
fi
PARALLEL="${PARALLEL:-$NUM_GPUS_PER_NODE}"
LOG_DIR="${LOG_DIR:-logs/dist}"
mkdir -p "$LOG_DIR"

expand_range() {
    local tok="$1"
    if [[ "$tok" =~ ^([0-9]+)\.\.([0-9]+)$ ]]; then
        seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    else
        echo "$tok"
    fi
}

if [[ $# -gt 0 ]]; then
    TASK_TOKENS=("$@")
elif [[ -n "${TASK_IDS:-}" ]]; then
    read -r -a TASK_TOKENS <<<"$TASK_IDS"
else
    mapfile -t TASK_TOKENS < <(awk -F'\t' '$0 !~ /^#/ && NF >= 7 {print $1}' "$MANIFEST")
fi

TASK_LIST=()
for tok in "${TASK_TOKENS[@]}"; do
    while IFS= read -r x; do TASK_LIST+=("$x"); done < <(expand_range "$tok")
done

echo "[launch_local] manifest=$MANIFEST"
echo "[launch_local] tasks (${#TASK_LIST[@]}): ${TASK_LIST[*]}"
echo "[launch_local] parallel=$PARALLEL  gpus_per_node=$NUM_GPUS_PER_NODE"
echo "[launch_local] logs -> $LOG_DIR/"

# xargs -P fans tasks out across the configured parallelism. Each invocation
# writes its own log; we keep top-level stdout clean and just print exit codes.
printf '%s\n' "${TASK_LIST[@]}" | xargs -n 1 -P "$PARALLEL" -I {} \
    bash -c 'bash scripts/dist/run_eval_worker.sh "$1" || echo "[launch_local] task $1 FAILED" >&2' _ {}

echo "[launch_local] all done"
