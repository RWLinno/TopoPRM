#!/usr/bin/env bash
# ============================================================================
# submit_volc_pytorch.sh
#
# Worker-side entrypoint for a Volcengine MLP "PyTorch distributed" custom task.
# When you submit a PyTorchJob via the MLP console (or `volc ml_task submit`),
# every worker container is started with these env vars injected:
#   - MLP_WORKER_NUM    (total workers, == number of pods)
#   - MLP_ROLE_INDEX    (0-based index of this worker)
#   - MLP_WORKER_GPU    (GPUs per worker)
#   - MLP_HOST_NODE_ADDR, MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK ...
#
# Each worker takes one chunk of the manifest. If MLP_WORKER_GPU > 1 we fan
# the chunk out across local GPUs (still via launch_local.sh, since
# evaluation jobs are embarrassingly parallel and don't need NCCL).
#
# Submitter (template, run on your local machine that has `volc` CLI):
#   volc ml_task submit \
#     --name topoprm_eval_dr1_7b \
#     --framework PyTorchDDP \
#     --task-role-num 1 \
#     --task-role worker --replica 6 --gpu-per-replica 1 \
#     --image registry.../topoprm:latest \
#     --working-dir /Knowin/foundation/weilinruan/TopoPRM \
#     --entrypoint "bash scripts/dist/submit_volc_pytorch.sh"
#
# If you want one pod to own multiple manifest rows (e.g. fat 8-GPU pod), set
#   TASK_IDS_PER_WORKER="0,1,2,3,4,5,6,7"
# in the task env vars, and this script will dispatch them via xargs.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

WORKER_NUM="${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}"
WORKER_IDX="${MLP_ROLE_INDEX:-${RANK:-0}}"
if [[ -z "${MLP_WORKER_GPU:-}" ]]; then
    GPUS_PER_WORKER="$(nvidia-smi -L 2>/dev/null | wc -l)"
    [[ -z "$GPUS_PER_WORKER" || "$GPUS_PER_WORKER" == "0" ]] && GPUS_PER_WORKER=1
else
    GPUS_PER_WORKER="$MLP_WORKER_GPU"
fi

export NUM_GPUS_PER_NODE="$GPUS_PER_WORKER"

# Strategy 1: explicit list of TASK_IDs for this worker (preferred for
# heterogeneous manifests where you want a specific worker to own a known
# chunk).
if [[ -n "${TASK_IDS_PER_WORKER:-}" ]]; then
    TASK_IDS_FOR_ME="${TASK_IDS_PER_WORKER//,/ }"
else
    # Strategy 2: auto-shard the manifest into WORKER_NUM contiguous chunks
    # and pick this worker's chunk.
    MANIFEST="${MANIFEST:-configs/dist/eval_manifest.tsv}"
    ALL_IDS=( $(awk -F'\t' '$0 !~ /^#/ && NF >= 7 {print $1}' "$MANIFEST") )
    TOTAL=${#ALL_IDS[@]}
    CHUNK=$(( (TOTAL + WORKER_NUM - 1) / WORKER_NUM ))
    START=$(( WORKER_IDX * CHUNK ))
    END=$(( START + CHUNK ))
    if [[ $END -gt $TOTAL ]]; then END=$TOTAL; fi
    TASK_IDS_FOR_ME=""
    for ((i=START; i<END; i++)); do
        TASK_IDS_FOR_ME="$TASK_IDS_FOR_ME ${ALL_IDS[$i]}"
    done
fi

echo "[volc] worker=$WORKER_IDX/$WORKER_NUM host=$(hostname) gpus=$GPUS_PER_WORKER tasks=[$TASK_IDS_FOR_ME]"

if [[ -z "$TASK_IDS_FOR_ME" ]]; then
    echo "[volc] no tasks assigned to this worker; exiting cleanly."
    exit 0
fi

# Reuse the local launcher to fan tasks across this worker's local GPUs.
TASK_IDS="$TASK_IDS_FOR_ME" PARALLEL="$GPUS_PER_WORKER" \
    bash scripts/dist/launch_local.sh
