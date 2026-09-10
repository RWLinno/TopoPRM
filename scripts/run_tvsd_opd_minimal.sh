#!/bin/bash
set -euo pipefail

# Minimal TVSD smoke path:
# 1) Phase III-A rollout + filtering on a tiny subset
# 2) Phase III-B OPSD trainer for a few steps

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

TEACHER_MODEL="${TEACHER_MODEL:-${MODEL_ROOT}/Qwen/Qwen3.5-9B}"
TEACHER_ADAPTER="${TEACHER_ADAPTER:-}"
INPUT_JSONL="${INPUT_JSONL:-data/grpo_ready/train_public.jsonl}"
RAW_OUT="${RAW_OUT:-data/srt_raw/rollouts_minimal.jsonl}"
SRT_READY="${SRT_READY:-data/srt_ready/train_minimal.jsonl}"
OPSD_CONFIG="${OPSD_CONFIG:-configs/opsd_student_4b.yaml}"

MAX_PROMPTS="${MAX_PROMPTS:-4}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
OPSD_MAX_STEPS="${OPSD_MAX_STEPS:-1}"

echo "=== TVSD minimal smoke run ==="
echo "Teacher model: ${TEACHER_MODEL}"
echo "Teacher adapter: ${TEACHER_ADAPTER:-<none>}"
echo "Input: ${INPUT_JSONL}"

mkdir -p data/srt_raw data/srt_ready

ROLL_ARGS=(
  --model "${TEACHER_MODEL}"
  --input "${INPUT_JSONL}"
  --output "${RAW_OUT}"
  --max_prompts "${MAX_PROMPTS}"
  --samples_per_prompt "${SAMPLES_PER_PROMPT}"
  --max_new_tokens 512
)
if [ -n "${TEACHER_ADAPTER}" ]; then
  ROLL_ARGS+=(--adapter "${TEACHER_ADAPTER}")
fi

python3 scripts/rollout_srt.py "${ROLL_ARGS[@]}"
python3 -m src.distill.build_srt_data --raw_rollouts "${RAW_OUT}" --output "${SRT_READY}" --max_samples 8
python3 -m src.distill.opsd_trainer --config "${OPSD_CONFIG}" --max_steps "${OPSD_MAX_STEPS}"

echo "=== TVSD minimal smoke run finished ==="
