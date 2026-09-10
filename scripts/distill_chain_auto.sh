#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${PYTHON_ENV_BIN}:$PATH"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_PROJECT=topoprm
export WANDB_RUN_GROUP=distill_reverse_kl

python - <<'PY2'
import subprocess,time
while True:
    out=subprocess.check_output(['ps','-eo','cmd'],text=True,errors='ignore')
    running=any('output/sft_private_boost' in ln and 'swift' in ln for ln in out.splitlines())
    if not running:
        break
    time.sleep(60)
print('[distill_chain] sft_private_boost finished, start reverse-KL distillation')
PY2

GPUS="${DISTILL_GPUS:-2,3}"
python -m src.distill.student_train \
  --config configs/distill_7b.yaml \
  --gpus "$GPUS" \
  --project_root ${REPO_ROOT}

bash scripts/run_distill_rkl_eval.sh \
  "${MODEL_ROOT}/Qwen3-8B" \
  "output/distill_8b_rkl/final" \
  "distill_rkl_8b"
