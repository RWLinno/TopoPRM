#!/bin/bash
set -euo pipefail

###############################################################################
# Run the full experiment pipeline:
#   1. GRPO main (TopoPRM full composite reward)
#   2. GRPO ablation: outcome_only
#   3. GRPO ablation: no_topo
#   4. GRPO ablation: no_continuity
#   5. Evaluation on all variants
#
# Usage: bash scripts/run_all_experiments.sh
# Env vars: CUDA_VISIBLE_DEVICES (default: 0,1,2,3,4,5,6,7)
#           NPROC_PER_NODE (default: 8)
###############################################################################

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PATH="/mnt/users/conda_env/swift/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

SFT_CKPT="output/sft_qwen3_32b/v1-20260316-154443/checkpoint-120"

echo "============================================================"
echo " TopoPRM Full Experiment Pipeline"
echo " GPUs: $CUDA_VISIBLE_DEVICES  NPROC: $NPROC_PER_NODE"
echo " SFT Checkpoint: $SFT_CKPT"
echo "============================================================"

# --- Phase 1: GRPO Training ---
EXPERIMENTS=("grpo_main" "grpo_outcome_only" "grpo_no_topo" "grpo_no_continuity")

for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Starting GRPO: $exp"
    bash scripts/run_grpo.sh "$exp"
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Finished GRPO: $exp"
done

# --- Phase 2: Evaluation ---
echo ""
echo "============================================================"
echo " Phase 2: Evaluation"
echo "============================================================"

# Evaluate SFT baseline
echo ">>> Evaluating SFT baseline"
bash scripts/run_eval.sh "$SFT_CKPT" "sft_baseline"

# Evaluate GRPO variants - find latest checkpoints
for exp in "${EXPERIMENTS[@]}"; do
    CKPT_DIR="output/${exp}"
    LATEST=$(find "$CKPT_DIR" -maxdepth 2 -name "checkpoint-*" -type d 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -n "$LATEST" ]; then
        echo ">>> Evaluating $exp: $LATEST"
        bash scripts/run_eval.sh "$LATEST" "$exp"
    else
        echo ">>> WARNING: No checkpoint found for $exp, skipping eval"
    fi
done

echo ""
echo "============================================================"
echo " All experiments completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
