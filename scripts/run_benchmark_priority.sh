#!/usr/bin/env bash
set -euo pipefail

# Priority benchmark runner (independent of opencompass)
# Runs private light-200 benchmark first for paper-ready early signal.

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PATH="/mnt/users/conda_env/topoprm/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

mkdir -p output/eval logs

echo "[benchmark_priority] started at $(date '+%F %T')"
echo "[benchmark_priority] GPUs=$CUDA_VISIBLE_DEVICES MAX_NEW_TOKENS=$MAX_NEW_TOKENS"

# 1) Main model
bash scripts/run_eval_light_private.sh \
  "Qwen/Qwen3-32B" \
  "output/grpo_main/v3-20260318-211524/checkpoint-79" \
  "grpo_main_light200"

# 2) Outcome-only ablation
bash scripts/run_eval_light_private.sh \
  "Qwen/Qwen3-32B" \
  "output/grpo_outcome_only/v0-20260319-171419/checkpoint-212" \
  "grpo_outcome_light200"

# 3) SFT baseline (latest)
SFT_CKPT=$(find output/sft -maxdepth 3 -name 'checkpoint-*' -type d 2>/dev/null | sort -V | tail -1 || true)
if [ -n "$SFT_CKPT" ]; then
  bash scripts/run_eval_light_private.sh \
    "Qwen/Qwen3-32B" \
    "$SFT_CKPT" \
    "sft_light200"
fi

# 4) Qwen2.5 baseline
bash scripts/run_eval_light_private.sh \
  "${HF_MODELS_DIR:-./models}/Qwen2.5-7B-Instruct" \
  "none" \
  "baseline_qwen25_7b_light200"

# Summarize
python3 - << 'PY'
import json,glob,os,csv
rows=[]
for fp in sorted(glob.glob('output/eval/*light200*_metrics.json')):
    d=json.load(open(fp))
    rows.append({
        'file': os.path.basename(fp),
        'num_samples': d.get('num_samples'),
        'format_compliance': d.get('format_compliance'),
        'score_accuracy': d.get('score_accuracy'),
        'error_identification_f1': d.get('error_identification_f1'),
        'step_coverage': d.get('step_coverage'),
    })
if rows:
    with open('output/eval/benchmark_priority_summary.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('wrote output/eval/benchmark_priority_summary.csv')
else:
    print('no light200 metrics produced')
PY

echo "[benchmark_priority] finished at $(date '+%F %T')"
