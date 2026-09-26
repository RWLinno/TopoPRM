#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
TOPOPRM_ENV_BIN="${TOPOPRM_ENV_BIN:-${PYTHON_ENV_BIN:-}}"
TOPOPRM_PYTHON="${TOPOPRM_ENV_BIN:+${TOPOPRM_ENV_BIN}/}python"
if [ "${DISTILL_VARIANT:-topology}" != topology ] || [ "${DISTILL_PHASE:-online}" != online ]; then
    echo "[ERROR] The paper entrypoint requires DISTILL_VARIANT=topology and DISTILL_PHASE=online." >&2
    exit 2
fi
: "${STUDENT_MODEL:?Set STUDENT_MODEL to the Stage-II checkpoint or its merged Stage-I base}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL to the fixed matching-architecture teacher}"
: "${PROMPT_DATA:?Set PROMPT_DATA to problems with reference final answers}"
args=( -m src.distill.student_train
    --student_model "$STUDENT_MODEL" --student_adapter "${STUDENT_ADAPTER:-}"
    --teacher_model "$TEACHER_MODEL" --teacher_adapter "${TEACHER_ADAPTER:-}"
    --student_device "${STUDENT_DEVICE:-cuda:0}" --teacher_device "${TEACHER_DEVICE:-cuda:1}"
    --prompts "$PROMPT_DATA" --output_dir "${DISTILL_OUTPUT_DIR:-output/distill_topology}"
    --token_budget "${TOKEN_BUDGET:-2048}" --max_length "${DISTILL_MAX_LENGTH:-8192}"
    --max_prompts "${MAX_PROMPTS:-0}" --seed "${DISTILL_SEED:-0}"
    --learning_rate "${DISTILL_LEARNING_RATE:-2e-5}" --num_train_epochs "${DISTILL_EPOCHS:-1}"
    --gradient_accumulation_steps "${DISTILL_ACCUMULATION:-8}"
    --student_temperature "${STUDENT_TEMPERATURE:-0.7}"
    --teacher_temperature "${TEACHER_TEMPERATURE:-0.0}"
    --revision_strategy "${REVISION_STRATEGY:-topology}" --selection "${REVISION_SELECTION:-topology}"
)
if [ "${DISTILL_DRY_RUN:-0}" = 1 ]; then
    printf '[tgd] current-target sampling; fixed same-scale teacher; revised-prefix token-summed RKL\n'
    printf '%q ' "$TOPOPRM_PYTHON" "${args[@]}" "$@"
    printf '\n'
    exit 0
fi
exec "$TOPOPRM_PYTHON" "${args[@]}" "$@"
