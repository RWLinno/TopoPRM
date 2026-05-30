#!/usr/bin/env bash
# ==============================================================================
# Sequential evaluation runner for the 5 DR1-7B family adapters.
# - Skips benchmarks that already have *_metrics.json (default behavior).
# - Uses all 8 GPUs in parallel; one bench per GPU.
# - Logs per-label to logs/unified/<label>.dispatch.log.
# Generated 2026-05-13 by the planning agent.
# ==============================================================================
set -uo pipefail

REPO_ROOT="${TOPOPRM_ROOT:-.}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
PYBIN="${TOPOPRM_PYTHON:-python3}"

BASE_MODEL="${HF_MODELS_DIR:-./models}/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
LOG_DIR="logs/unified"
mkdir -p "$LOG_DIR"

# label -> adapter path
declare -A ADAPTERS=(
    [sft_dr1_7b]="$REPO_ROOT/output/sft_deepseek_r1_7b/final"
    [grpo_outcome_only]="$REPO_ROOT/output/grpo_outcome_only_dr1_7b/final"
    [grpo_no_topo]="$REPO_ROOT/output/grpo_no_topo_dr1_7b/final"
    [grpo_no_continuity]="$REPO_ROOT/output/grpo_no_continuity_dr1_7b/final"
    [topoprm_full_dr1_7b]="$REPO_ROOT/output/grpo_topoprm_deepseek_r1_7b/final"
)

# Order: SFT first (cheapest gain), then GRPO ablations, then TopoPRM full.
ORDER=( sft_dr1_7b grpo_outcome_only grpo_no_topo grpo_no_continuity topoprm_full_dr1_7b )

# Also evaluate baseline DR1-7B for any missing benches (should be none).
BASELINE_LABEL="baseline_dr1_7b_chat"

START_TS=$(date +%s)
echo "[INFO] Starting sequential eval at $(date)"
echo "[INFO] GPUs=$GPUS  base_model=$BASE_MODEL"

for LABEL in "${ORDER[@]}"; do
    ADAPTER="${ADAPTERS[$LABEL]}"
    if [[ ! -d "$ADAPTER" ]]; then
        echo "[WARN] adapter not found for $LABEL: $ADAPTER (skipping)"
        continue
    fi
    DISPATCH_LOG="$LOG_DIR/${LABEL}.dispatch.log"
    echo "──────────────────────────────────────────────"
    echo "[INFO] $(date +%H:%M:%S) running $LABEL  (adapter=$ADAPTER)"
    echo "[INFO]   log=$DISPATCH_LOG"

    # SFT/GRPO adapters need --sft_style for the chat-template wrapping.
    SFT_STYLE_FLAG="--sft_style"

    "$PYBIN" scripts/unified_eval_orchestrator.py \
        --model "$BASE_MODEL" \
        --adapter "$ADAPTER" \
        --label "$LABEL" \
        --gpus "$GPUS" \
        --benchmarks all \
        --num_samples_per_item 5 \
        --k_values 1 5 \
        --use_chat_template \
        $SFT_STYLE_FLAG \
        > "$DISPATCH_LOG" 2>&1
    RC=$?

    if [[ $RC -ne 0 ]]; then
        echo "[WARN] $LABEL exited with code $RC; continuing"
    else
        echo "[INFO] $LABEL done"
    fi
done

ELAPSED=$(( $(date +%s) - START_TS ))
echo "[INFO] All labels finished. Total elapsed: $((ELAPSED / 60)) min"
