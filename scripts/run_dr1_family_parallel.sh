#!/usr/bin/env bash
# Parallel launcher: remaining 4 DR1-7B family labels on GPUs 2-7.
# sft_dr1_7b is assumed to already be running on GPU 0,1.
set -uo pipefail

REPO_ROOT="${TOPOPRM_ROOT:-.}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
PYBIN="${TOPOPRM_PYTHON:-python3}"
BASE_MODEL="${HF_MODELS_DIR:-./models}/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
LOG_DIR="logs/unified"
mkdir -p "$LOG_DIR"

# label, adapter, GPU-pair, missing-benches
LABELS=(
    "grpo_outcome_only|output/grpo_outcome_only_dr1_7b/final|2,3|cnmo2024 mmlu"
    "topoprm_full_dr1_7b|output/grpo_topoprm_deepseek_r1_7b/final|4,5|cnmo2024 mmlu"
    "grpo_no_topo|output/grpo_no_topo_dr1_7b/final|6|olympiadbench omni_math cnmo2024 mmlu"
    "grpo_no_continuity|output/grpo_no_continuity_dr1_7b/final|7|olympiadbench omni_math cnmo2024 mmlu"
)

for entry in "${LABELS[@]}"; do
    IFS='|' read -r LABEL ADAPTER_REL GPUS BENCHES <<< "$entry"
    ADAPTER="$REPO_ROOT/$ADAPTER_REL"
    if [[ ! -d "$ADAPTER" ]]; then
        echo "[WARN] adapter not found for $LABEL: $ADAPTER (skipping)"
        continue
    fi
    DISPATCH_LOG="$LOG_DIR/${LABEL}.dispatch.log"
    echo "[INFO] launching $LABEL on GPUs $GPUS (benches: $BENCHES)"
    nohup "$PYBIN" scripts/unified_eval_orchestrator.py \
        --model "$BASE_MODEL" \
        --adapter "$ADAPTER" \
        --label "$LABEL" \
        --gpus "$GPUS" \
        --benchmarks $BENCHES \
        --num_samples_per_item 5 \
        --k_values 1 5 \
        --use_chat_template \
        --sft_style \
        > "$DISPATCH_LOG" 2>&1 &
    echo "  pid=$!"
    sleep 2
done

echo "[INFO] all orchestrators launched"
echo "[INFO] monitor with: tail -f $LOG_DIR/*.dispatch.log"
