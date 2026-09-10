#!/usr/bin/env bash
# ============================================================================
# run_eval_worker.sh
#
# A single worker entrypoint for the TopoPRM evaluation manifest.
#
# Behavior:
#   1. Pick a TASK_ID from one of (priority order):
#        - 1st CLI arg ($1)
#        - $TASK_ID
#        - $SLURM_ARRAY_TASK_ID   (Slurm job array)
#        - $RAY_TASK_INDEX        (Ray submit_ray.py)
#        - $MLP_ROLE_INDEX        (managed MLP PyTorch distributed task)
#        - $RANK                  (generic torchrun fallback)
#   2. Read the corresponding row from $MANIFEST
#      (default: configs/dist/eval_manifest.tsv).
#   3. Pin the worker to a local GPU via
#         CUDA_VISIBLE_DEVICES = TASK_ID % NUM_GPUS_PER_NODE
#      (NUM_GPUS_PER_NODE comes from nvidia-smi unless overridden).
#   4. Invoke scripts/bench_transformers.py with the row's args, redirecting
#      stdout+stderr to logs/dist/<LABEL>_<BENCHMARKS>.log.
#
# This script is the single point of contract for every launcher
# (local / Slurm / managed PyTorch DDP / Ray). Platform-specific submitters only
# need to translate their native task index into TASK_ID.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="${MANIFEST:-configs/dist/eval_manifest.tsv}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
LOG_DIR="${LOG_DIR:-logs/dist}"
OUTPUT_DIR="${OUTPUT_DIR:-output/eval}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
BATCH_SIZE="${BATCH_SIZE:-8}"
if [[ -z "${NUM_GPUS_PER_NODE:-}" ]]; then
    # Avoid `nvidia-smi ... | head -1` here: with `set -o pipefail`, the
    # SIGPIPE that head triggers on nvidia-smi turns into a failed pipe,
    # which then aborts the worker before it can pin a GPU.
    NUM_GPUS_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l)"
    if [[ -z "$NUM_GPUS_PER_NODE" || "$NUM_GPUS_PER_NODE" == "0" ]]; then
        NUM_GPUS_PER_NODE=1
    fi
fi
PYTHON_BIN="${PYTHON_BIN:-${PYTHON_ENV_BIN}/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    # Fallback to PATH python3 if the pinned env is missing (e.g. on cloud node).
    PYTHON_BIN="python3"
fi
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

TASK_ID="${1:-${TASK_ID:-${SLURM_ARRAY_TASK_ID:-${RAY_TASK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-}}}}}}"
if [[ -z "${TASK_ID:-}" ]]; then
    echo "ERROR: TASK_ID is unset. Pass it as the 1st arg or via env." >&2
    exit 2
fi

# Read the manifest, drop comment/blank lines, and select the row whose first
# TSV column matches our task id.
ROW="$(awk -v tid="$TASK_ID" -F'\t' \
    '$0 !~ /^#/ && NF >= 7 && $1 == tid {print; found=1; exit} END{if(!found) exit 1}' \
    "$MANIFEST")"
if [[ -z "$ROW" ]]; then
    echo "ERROR: no row in $MANIFEST with TASK_ID=$TASK_ID" >&2
    exit 3
fi

IFS=$'\t' read -r _ LABEL ADAPTER SFT_STYLE BENCHMARKS NUM_SAMPLES MAX_ITEMS NOTES <<<"$ROW"

if [[ "$ADAPTER" == "-" || -z "$ADAPTER" ]]; then
    ADAPTER_FLAG=""
else
    ADAPTER_FLAG="--adapter $ADAPTER"
fi
if [[ "$SFT_STYLE" == "1" ]]; then
    SFT_STYLE_FLAG="--sft_style"
else
    SFT_STYLE_FLAG=""
fi
if [[ "${MAX_ITEMS:-0}" -gt 0 ]]; then
    MAX_ITEMS_FLAG="--max_items $MAX_ITEMS"
else
    MAX_ITEMS_FLAG=""
fi

GPU_ID=$(( TASK_ID % NUM_GPUS_PER_NODE ))
BENCH_LIST="${BENCHMARKS//+/ }"
LOG_TAG="${LABEL}_$(echo "$BENCHMARKS" | tr '+' '-')"
LOG_FILE="$LOG_DIR/${LOG_TAG}.log"

echo "[worker] task=$TASK_ID label=$LABEL adapter=$ADAPTER sft_style=$SFT_STYLE benches=$BENCH_LIST gpu=$GPU_ID log=$LOG_FILE notes=$NOTES"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
    $PYTHON_BIN -u scripts/bench_transformers.py \
        --model "$MODEL_PATH" \
        $ADAPTER_FLAG \
        --label "$LABEL" \
        --benchmarks $BENCH_LIST \
        $SFT_STYLE_FLAG \
        --use_chat_template \
        --num_samples_per_item "$NUM_SAMPLES" \
        --k_values 1 5 \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --batch_size "$BATCH_SIZE" \
        $MAX_ITEMS_FLAG \
        --output_dir "$OUTPUT_DIR" \
        $EXTRA_FLAGS \
        >>"$LOG_FILE" 2>&1

echo "[worker] task=$TASK_ID done"
