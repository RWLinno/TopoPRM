#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
TOPOPRM_ENV_BIN="${TOPOPRM_ENV_BIN:-${PYTHON_ENV_BIN:-}}"
TOPOPRM_PYTHON="${TOPOPRM_ENV_BIN:+${TOPOPRM_ENV_BIN}/}python"
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

STUDENT_MODEL="${STUDENT_MODEL:-${MODEL_ROOT:-models}/Qwen/Qwen3.5-9B}"
STUDENT_ADAPTER="${STUDENT_ADAPTER:-}"
TEACHER_MODEL="${TEACHER_MODEL:-$STUDENT_MODEL}"
TEACHER_ADAPTER="${TEACHER_ADAPTER:-}"
DISTILL_ROOT="${DISTILL_ROOT:-${EXP_ROOT:-output}/distill_data/${DISTILL_VARIANT}}"
PROMPT_DATA="${PROMPT_DATA:-data/grpo_ready/train_public_swift.jsonl}"
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

CANDIDATES="$DISTILL_ROOT/candidates.jsonl"
PARTIAL="$DISTILL_ROOT/candidates.partial.jsonl"
TRAIN="$DISTILL_ROOT/train.jsonl"
VALIDATION="$DISTILL_ROOT/validation.jsonl"

echo "[distill] variant=$DISTILL_VARIANT phase=$DISTILL_PHASE selection=$DISTILL_SELECTION"
echo "[distill] root=$DISTILL_ROOT seed=$DISTILL_SEED max_accepted=$MAX_ACCEPTED_TARGETS"
if [ "${DISTILL_DRY_RUN:-0}" = "1" ]; then
    echo "[distill] candidates=$CANDIDATES"
    echo "[distill] train=$TRAIN validation=$VALIDATION"
    echo "[distill] student_output=${DISTILL_OUTPUT_DIR:-output/distill_${DISTILL_VARIANT}}"
    echo "[distill] objective=fixed-corpus reverse-KL; python=$TOPOPRM_PYTHON"
    exit 0
fi

mkdir -p "$DISTILL_ROOT"

if [ "$DISTILL_PHASE" = "rollout" ] || [ "$DISTILL_PHASE" = "all" ]; then
    if [ -z "$TEACHER_ADAPTER" ] || [ -z "$STUDENT_ADAPTER" ]; then
        echo "[ERROR] Set STUDENT_ADAPTER and TEACHER_ADAPTER to their Stage-II adapters" >&2
        exit 1
    fi
    for path in "$STUDENT_MODEL" "$TEACHER_MODEL" "$TEACHER_ADAPTER" "$STUDENT_ADAPTER" "$PROMPT_DATA"; do
        if [ ! -e "$path" ]; then
            echo "[ERROR] Required path not found: $path" >&2
            exit 1
        fi
    done
    export TOPO_DAG_RAW_DIRECTED=0 TOPO_DAG_LLM_REFINE=0
    export TOPO_DAG_EDGE_CHECKPOINT= TOPO_DAG_EDGE_REQUIRED=0 TOPO_DISABLE_EDGE_ENCODER=1

    if [ ! -f "$CANDIDATES" ]; then
        if [ -e "$PARTIAL" ]; then
            echo "[ERROR] Partial candidate file exists: $PARTIAL"
            echo "Inspect it, then move or remove it explicitly before retrying."
            exit 1
        fi
        CUDA_VISIBLE_DEVICES="${DISTILL_ROLLOUT_GPUS:-0,1,2}" \
            "$TOPOPRM_PYTHON" -m scripts.rollout_srt \
            --student_model "$STUDENT_MODEL" \
            --student_adapter "$STUDENT_ADAPTER" \
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
    "$TOPOPRM_PYTHON" -m src.distill.build_srt_data \
        --raw_rollouts "$CANDIDATES" \
        --output "$TRAIN" \
        --validation_output "$VALIDATION" \
        --validation_fraction 0.05 \
        --topo_threshold "${TOPO_THRESHOLD:-0.5}" \
        --token_budget "$TOKEN_BUDGET" \
        --selection "$DISTILL_SELECTION" \
        --max_samples "$MAX_ACCEPTED_TARGETS"
fi

if [ "$DISTILL_PHASE" = "train" ] || [ "$DISTILL_PHASE" = "all" ]; then
    if [ ! -s "$TRAIN" ] || [ ! -s "$VALIDATION" ]; then
        echo "[ERROR] Accepted target split is empty; refusing to launch student training."
        exit 1
    fi
    if [ -z "$STUDENT_ADAPTER" ] || [ -z "$TEACHER_ADAPTER" ]; then
        echo "[ERROR] Set both initialization adapters before reverse-KL training." >&2
        exit 1
    fi
    RKL_CONFIG="$DISTILL_ROOT/reverse_kl_config.yaml"
    export STUDENT_MODEL STUDENT_ADAPTER TEACHER_MODEL TEACHER_ADAPTER TRAIN RKL_CONFIG
    export DISTILL_OUTPUT_DIR="${DISTILL_OUTPUT_DIR:-output/distill_${DISTILL_VARIANT}}"
    "$TOPOPRM_PYTHON" - <<'PYCONFIG'
import os
from pathlib import Path
import yaml
cfg = dict(
    student_model=os.environ["STUDENT_MODEL"],
    student_init_adapter=os.environ["STUDENT_ADAPTER"],
    teacher_model=os.environ["TEACHER_MODEL"],
    teacher_adapter=os.environ["TEACHER_ADAPTER"],
    dataset=[str(Path(os.environ["TRAIN"]).resolve())],
    output_dir=os.environ["DISTILL_OUTPUT_DIR"],
    max_length=int(os.environ.get("DISTILL_MAX_LENGTH", "8192")),
    max_train_samples=0,
    rkl_weight=1.0,
    ce_weight=0.0,
    temperature=1.0,
    learning_rate=float(os.environ.get("DISTILL_LEARNING_RATE", "2e-5")),
    num_train_epochs=float(os.environ.get("DISTILL_EPOCHS", "1")),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    bf16=True,
)
Path(os.environ["RKL_CONFIG"]).write_text(yaml.safe_dump(cfg, sort_keys=False))
PYCONFIG
    CUDA_VISIBLE_DEVICES="${DISTILL_TRAIN_GPUS:-0,1}" \
        "$TOPOPRM_PYTHON" -m src.distill.student_train \
        --config "$RKL_CONFIG" --gpus 0,1 --project_root "$PROJECT_ROOT"
fi
