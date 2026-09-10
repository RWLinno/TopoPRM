#!/bin/bash
set -euo pipefail
###############################################################################
# Server A — Stage-II GRPO + TopoPRM v2 (DR1-7B)
# All outputs/logs carry _A suffix for Server A identification.
# Date: 2026-05-14
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ---- Branch guard ----
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
if [[ "$BRANCH" != "exp_May14" ]]; then
    echo "[FATAL] Expected branch exp_May14, got: $BRANCH"
    exit 1
fi

# ---- Environment ----
# Optional egress proxy for clusters without direct internet access.
export ALL_PROXY=${ALL_PROXY:-}
export HF_TOKEN=${HF_TOKEN}
export WANDB_API_KEY=${WANDB_API_KEY}
export WANDB_PROJECT=topoprm
export WANDB_RUN_GROUP=grpo_v2_A
export WANDB_NAME="grpo_topoprm_v2_A_$(date +%Y%m%d)"

# Use GPU 0-5 (leave 6-7 for ongoing eval)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NPROC_PER_NODE=6

# ---- Load v2 patch env vars ----
set -a
source <(grep -v '^\s*#' "$PROJECT_ROOT/configs/grpo_topoprm_v2.env" | grep -v '^\s*$')
set +a

echo "══════════════════════════════════════════════════════════════"
echo "[Server A] $(date '+%Y-%m-%d %H:%M:%S') Stage-II GRPO+TopoPRM v2"
echo "  Branch:  $BRANCH"
echo "  Config:  configs/grpo_topoprm_v2_A.yaml"
echo "  GPUs:    $CUDA_VISIBLE_DEVICES (NPROC=$NPROC_PER_NODE)"
echo "  Output:  output/grpo_topoprm_v2_A"
echo "  W&B:     $WANDB_PROJECT / $WANDB_RUN_GROUP / $WANDB_NAME"
echo "  Patches: P0=$TOPO_RESCALE_PATCH P2=$TOPO_HIER_AGG P3=$TOPO_CONT_REQUIRE_EVIDENCE"
echo "           P4=$TOPO_DAG_SENTENCE_FALLBACK P5=$TOPO_LENGTH_UNIT"
echo "══════════════════════════════════════════════════════════════"

# ---- Pre-flight: reward invariants ----
python3 "$SCRIPT_DIR/check_reward_invariants.py"
echo

# ---- Launch training ----
exec bash "$SCRIPT_DIR/run_grpo.sh" grpo_topoprm_v2_A "$@"
