#!/bin/bash
set -euo pipefail

###############################################################################
# Run remaining pipeline after outcome_only ablation completes:
#   1. grpo_no_topo ablation
#   2. grpo_no_continuity ablation
#   3. Evaluate all models
#
# Usage: nohup bash scripts/run_remaining_pipeline.sh > output/pipeline.log 2>&1 &
###############################################################################

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PATH="/mnt/users/conda_env/swift/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Remaining Pipeline"
echo "=========================================="

# Ablation 2: no_topo
echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Starting: grpo_no_topo"
swift rlhf --rlhf_type grpo --config experiments/configs/grpo_no_topo.yaml \
    2>&1 | tee "output/grpo_no_topo_$(date +%Y%m%d_%H%M%S).log"
echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Finished: grpo_no_topo"
sleep 30

# Ablation 3: no_continuity
echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Starting: grpo_no_continuity"
swift rlhf --rlhf_type grpo --config experiments/configs/grpo_no_continuity.yaml \
    2>&1 | tee "output/grpo_no_continuity_$(date +%Y%m%d_%H%M%S).log"
echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Finished: grpo_no_continuity"
sleep 30

# Evaluate all models
echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Starting evaluation"
bash scripts/run_eval_all.sh
echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Pipeline complete!"
