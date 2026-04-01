#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/mnt/users/conda_env/topoprm/bin:$PATH"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python - <<'PY2'
import subprocess,time
while True:
    out=subprocess.check_output(['ps','-eo','cmd'],text=True,errors='ignore')
    running=any('output/sft_private_boost' in ln and 'swift' in ln for ln in out.splitlines())
    if not running:
        break
    time.sleep(60)
print('[benchmark_retry] sft done, start math500 retry on gpu4')
PY2

export CUDA_VISIBLE_DEVICES=4
swift eval --model "Qwen/Qwen3-32B"   --adapters "output/grpo_main/v3-20260318-211524/checkpoint-79"   --eval_dataset math_500 --eval_limit 100   --max_new_tokens 512 --timeout 600   --eval_output_dir "output/eval/benchmark_light/grpo_main_light_math_500_retry"

swift eval --model "Qwen/Qwen3-32B"   --adapters "output/grpo_outcome_only/v0-20260319-171419/checkpoint-212"   --eval_dataset math_500 --eval_limit 100   --max_new_tokens 512 --timeout 600   --eval_output_dir "output/eval/benchmark_light/grpo_outcome_light_math_500_retry"
