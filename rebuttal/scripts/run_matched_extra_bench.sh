#!/usr/bin/env bash
# Extra hard-benchmark columns for the matched DR1-7B comparison, so the
# rebuttal table shows separation on the benchmarks that actually discriminate
# (GSM8K/MATH are saturated; competition/olympiad sets separate the policies).
# Usage: run_matched_extra_bench.sh <label> <merged_dir> <gpu>
set -euo pipefail
cd /Knowin/foundation/weilinruan/TopoPRM
source rebuttal/scripts/env.sh
LABEL="${1:?}"; MERGED="${2:?}"; GPU="${3:?}"

for B in olympiadbench omni_math; do
  CUDA_VISIBLE_DEVICES="$GPU" "$TOPOPRM_PY" scripts/bench_transformers.py \
    --model "$MERGED" --label "$LABEL" \
    --benchmarks "$B" \
    --num_samples_per_item 1 --k_values 1 --max_items 120 \
    --max_new_tokens 8192 --batch_size 8 --use_chat_template --sft_style \
    --output_dir rebuttal/outputs/eval_tables
done
echo "[extra_bench] $LABEL done"
