#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${PYTHON_ENV_BIN}:$PATH"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_PROJECT=topoprm
export WANDB_RUN_GROUP=distill_reverse_kl

echo "[distill_reverse_kl] skip sample distillation data pipeline by design"
echo "[distill_reverse_kl] run online teacher-student reverse-KL training directly"

GPUS="${DISTILL_GPUS:-2,3}"
python -m src.distill.student_train \
  --config configs/distill_7b_compact.yaml \
  --gpus "$GPUS" \
  --project_root ${REPO_ROOT}

bash scripts/run_distill_rkl_eval.sh \
  "${MODEL_ROOT}/Qwen3-8B" \
  "output/distill_7b_compact_rkl/final" \
  "distill_rkl_8b_compact"
