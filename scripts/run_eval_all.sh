#!/bin/bash
set -euo pipefail

###############################################################################
# Evaluate all trained models: SFT baseline + GRPO variants
# Automatically finds latest checkpoint for each experiment
#
# Usage: bash scripts/run_eval_all.sh
###############################################################################

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PATH="/mnt/users/conda_env/topoprm/bin:$PATH"
export PATH="$(dirname $(which swift)):$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

EXPERIMENTS=(
  "grpo_main"
  "grpo_outcome_only"
  "grpo_no_topo"
  "grpo_no_continuity"
  "grpo_clipped"
  "grpo_confgate"
  "grpo_mulgate"
  "grpo_scae"
)

echo "=========================================="
echo " TopoPRM Evaluation Pipeline"
echo "=========================================="

find_best_ckpt() {
    local exp_dir="output/$1"
    find "$exp_dir" -maxdepth 3 -name "checkpoint-*" -type d 2>/dev/null \
        | sort -V | tail -1
}

find_sft_ckpt() {
    local ckpt=""
    ckpt=$(find "output/sft" -maxdepth 3 -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1 || true)
    if [ -n "$ckpt" ]; then
        echo "$ckpt"
        return 0
    fi
    ckpt=$(find "output" -maxdepth 4 -path "*/sft*/*" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1 || true)
    echo "$ckpt"
}

# 1) SFT baseline
SFT_CKPT="$(find_sft_ckpt)"
if [ -n "${SFT_CKPT:-}" ] && [ -d "$SFT_CKPT" ]; then
    echo ">>> Evaluating SFT baseline: $SFT_CKPT"
    bash scripts/run_eval.sh "$SFT_CKPT" "sft_baseline"
else
    echo ">>> SKIP SFT baseline (checkpoint not found)"
fi

# 2) GRPO variants
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

echo ""
echo "=== Results Summary ==="
for f in output/eval/*_metrics.json; do
    [ -f "$f" ] && echo "$(basename "$f"): $(cat "$f")"
done
