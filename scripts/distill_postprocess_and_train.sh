#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/mnt/users/conda_env/topoprm/bin:$PATH"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_PROJECT=topoprm
export WANDB_RUN_GROUP=distill_reverse_kl

echo "[distill_reverse_kl] skip sample distillation data pipeline by design"
echo "[distill_reverse_kl] run online teacher-student reverse-KL training directly"

GPUS="${DISTILL_GPUS:-2,3}"
python -m src.distill.student_train \
  --config configs/distill_7b_compact.yaml \
  --gpus "$GPUS" \
  --project_root /mnt/users/rwl/topoprm

bash scripts/run_distill_rkl_eval.sh \
  "/mnt/users/rwl/models/Qwen3-8B" \
  "output/distill_7b_compact_rkl/final" \
  "distill_rkl_8b_compact"
