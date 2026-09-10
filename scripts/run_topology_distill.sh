#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
TOPOPRM_ENV_BIN="${TOPOPRM_ENV_BIN:-${PYTHON_ENV_BIN}}"
DISTILL_VARIANT="${DISTILL_VARIANT:-topology}"
DISTILL_PHASE="${DISTILL_PHASE:-all}"
case "$DISTILL_VARIANT" in
    topology|generic|length|static) ;;
    *) echo "[ERROR] Unknown distillation variant: $DISTILL_VARIANT" >&2; exit 2 ;;
esac
case "$DISTILL_PHASE" in
    rollout|data|train|all) ;;
    *) echo "[ERROR] Unknown distillation phase: $DISTILL_PHASE" >&2; exit 2 ;;
esac

STUDENT_MODEL="${STUDENT_MODEL:-${MODEL_ROOT}/Qwen3.5-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-${MODEL_ROOT}/Qwen/Qwen3.5-9B}"
TEACHER_ADAPTER="${TEACHER_ADAPTER:-}"
DISTILL_ROOT="${DISTILL_ROOT:-${EXP_ROOT}/canonical/distill_data/${DISTILL_VARIANT}}"
PROMPT_DATA="${PROMPT_DATA:-${EXP_ROOT}/canonical/grpo_data/train_no_eval_overlap_max512.jsonl}"
MAX_PROMPTS="${MAX_PROMPTS:-8000}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
TOKEN_BUDGET="${TOKEN_BUDGET:-1024}"
DISTILL_SEED="${DISTILL_SEED:-0}"
MAX_ACCEPTED_TARGETS="${MAX_ACCEPTED_TARGETS:-0}"
if [ -z "${DISTILL_SELECTION:-}" ]; then
    if [ "$DISTILL_VARIANT" = "topology" ]; then
        DISTILL_SELECTION=topology
    else
        DISTILL_SELECTION=basic
    fi
fi
case "$DISTILL_SELECTION" in
    topology|basic) ;;
    *) echo "[ERROR] Unknown selection contract: $DISTILL_SELECTION" >&2; exit 2 ;;
esac

mkdir -p "$DISTILL_ROOT"
CANDIDATES="$DISTILL_ROOT/candidates.jsonl"
PARTIAL="$DISTILL_ROOT/candidates.partial.jsonl"
TRAIN="$DISTILL_ROOT/train.jsonl"
VALIDATION="$DISTILL_ROOT/validation.jsonl"

echo "[distill] variant=$DISTILL_VARIANT phase=$DISTILL_PHASE selection=$DISTILL_SELECTION"
echo "[distill] root=$DISTILL_ROOT seed=$DISTILL_SEED max_accepted=$MAX_ACCEPTED_TARGETS"
if [ "${DISTILL_DRY_RUN:-0}" = "1" ]; then
    echo "[distill] candidates=$CANDIDATES"
    echo "[distill] train=$TRAIN validation=$VALIDATION"
    echo "[distill] student_output=${SFT_OUTPUT_DIR:-/tmp/TopoPRM_ICLR27/canonical/distill_${DISTILL_VARIANT}_qwen35_4b}"
    exit 0
fi

if [ "$DISTILL_PHASE" = "rollout" ] || [ "$DISTILL_PHASE" = "all" ]; then
    if [ -z "$TEACHER_ADAPTER" ]; then
        echo "[ERROR] Set TEACHER_ADAPTER to the canonical Stage-II adapter" >&2
        exit 1
    fi
    for path in "$STUDENT_MODEL" "$TEACHER_MODEL" "$TEACHER_ADAPTER" "$PROMPT_DATA"; do
        if [ ! -e "$path" ]; then
            echo "[ERROR] Required path not found: $path" >&2
            exit 1
        fi
    done
    set -a
    source configs/grpo_topoprm_iclr27.env
    set +a
    export TOPO_DAG_EDGE_DEVICE="${DISTILL_EDGE_DEVICE:-cuda:2}"

    if [ ! -f "$CANDIDATES" ]; then
        if [ -e "$PARTIAL" ]; then
            echo "[ERROR] Partial candidate file exists: $PARTIAL"
            echo "Inspect it, then move or remove it explicitly before retrying."
            exit 1
        fi
        CUDA_VISIBLE_DEVICES="${DISTILL_ROLLOUT_GPUS:-0,1,2}" \
            "$TOPOPRM_ENV_BIN/python" -m scripts.rollout_srt \
            --student_model "$STUDENT_MODEL" \
            --student_device cuda:0 \
            --teacher_model "$TEACHER_MODEL" \
            --teacher_adapter "$TEACHER_ADAPTER" \
            --teacher_device cuda:1 \
            --input "$PROMPT_DATA" \
            --output "$PARTIAL" \
            --max_prompts "$MAX_PROMPTS" \
            --samples_per_prompt "$SAMPLES_PER_PROMPT" \
            --max_new_tokens "$TOKEN_BUDGET" \
            --student_temperature 0.7 \
            --teacher_temperature 0.0 \
            --revision_strategy "$DISTILL_VARIANT" \
            --seed "$DISTILL_SEED"
        mv "$PARTIAL" "$CANDIDATES"
    else
        echo "[distill] Reusing complete candidates: $CANDIDATES"
    fi
fi

if [ "$DISTILL_PHASE" = "data" ] || [ "$DISTILL_PHASE" = "all" ]; then
    if [ ! -s "$CANDIDATES" ]; then
        echo "[ERROR] Complete candidate file not found: $CANDIDATES" >&2
        exit 1
    fi
    "$TOPOPRM_ENV_BIN/python" -m src.distill.build_srt_data \
        --raw_rollouts "$CANDIDATES" \
        --output "$TRAIN" \
        --validation_output "$VALIDATION" \
        --validation_fraction 0.05 \
        --topo_threshold 0.5 \
        --token_budget "$TOKEN_BUDGET" \
        --selection "$DISTILL_SELECTION" \
        --max_samples "$MAX_ACCEPTED_TARGETS"
fi

if [ "$DISTILL_PHASE" = "train" ] || [ "$DISTILL_PHASE" = "all" ]; then
    if [ ! -s "$TRAIN" ] || [ ! -s "$VALIDATION" ]; then
        echo "[ERROR] Accepted target split is empty; refusing to launch student training."
        exit 1
    fi
    GUARD_GPU_LEAK_MB="${GUARD_GPU_LEAK_MB:-4096}" \
    CUDA_VISIBLE_DEVICES="${DISTILL_TRAIN_GPUS:-0,1,2,3}" \
    NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
    SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-/tmp/TopoPRM_ICLR27/canonical/distill_${DISTILL_VARIANT}_qwen35_4b}" \
        bash scripts/run_sft_config.sh sft_distill_4b \
        --dataset "$TRAIN" \
        --val_dataset "$VALIDATION"
fi
