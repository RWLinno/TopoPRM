#!/bin/bash
set -euo pipefail

###############################################################################
# Evaluate all trained models: SFT baseline + GRPO main + ablations
# Automatically finds latest checkpoint for each experiment
#
# Usage: bash scripts/run_eval_all.sh
###############################################################################

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PATH="$(dirname $(which swift)):$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

SFT_CKPT="output/sft_qwen3_32b/v1-20260316-154443/checkpoint-120"
EXPERIMENTS=("grpo_main" "grpo_outcome_only" "grpo_no_topo" "grpo_no_continuity")

echo "=========================================="
echo " TopoPRM Evaluation Pipeline"
echo "=========================================="

# Helper: find best checkpoint in an experiment output dir
find_best_ckpt() {
    local exp_dir="output/$1"
    find "$exp_dir" -maxdepth 3 -name "checkpoint-*" -type d 2>/dev/null \
        | sort -t- -k2 -n | tail -1
}

# 1. Evaluate SFT baseline
if [ -d "$SFT_CKPT" ]; then
    echo ">>> Evaluating SFT baseline: $SFT_CKPT"
    bash scripts/run_eval.sh "$SFT_CKPT" "sft_baseline"
else
    echo ">>> SKIP SFT baseline (checkpoint not found)"
fi

# 2. Evaluate each GRPO variant
for exp in "${EXPERIMENTS[@]}"; do
    CKPT=$(find_best_ckpt "$exp")
    if [ -n "$CKPT" ]; then
        echo ">>> Evaluating $exp: $CKPT"
        bash scripts/run_eval.sh "$CKPT" "$exp"
    else
        echo ">>> SKIP $exp (no checkpoint found in output/$exp)"
    fi
done

echo ""
echo "=========================================="
echo " Evaluation complete. Results in output/eval/"
echo "=========================================="
# Print summary
echo ""
echo "=== Results Summary ==="
for f in output/eval/*_metrics.json; do
    [ -f "$f" ] && echo "$(basename $f): $(cat $f)"
done
