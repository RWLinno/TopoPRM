#!/usr/bin/env bash
# Req-3: launch token-efficiency resampling for the 3 matched DR1-7B policies.
# Each policy -> one free GPU; 5 stochastic draws, GSM8K 200 items, sft_style.
set -uo pipefail
cd /Knowin/foundation/weilinruan/TopoPRM
source rebuttal/scripts/env.sh
L=rebuttal/outputs/logs
PY="$TOPOPRM_PY"

run() {  # <gpu> <merged_dir> <label>
  CUDA_VISIBLE_DEVICES="$1" nohup "$PY" rebuttal/scripts/efficiency_resample.py \
    --model "$2" --label "$3" --draws 5 --max_items 200 \
    --max_new_tokens 4096 --batch_size 8 --sft_style \
    --out "rebuttal/outputs/efficiency_${3}.json" \
    > "$L/efficiency_${3}.log" 2>&1 &
  echo "  $3 on GPU $1 PID $!"
}

run 5 output/merged_outcome_only_matched   outcome_only
run 6 output/merged_outcome_length_matched outcome_length
run 7 output/merged_topo_hier_matched      topo_hier
echo "[efficiency] launched 3 jobs"
jobs -l
