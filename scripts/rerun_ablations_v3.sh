#!/usr/bin/env bash
# Unified v3 benchmark re-run for GRPO ablation variants (RQ2 support).
#
# Adapters (all on Qwen3.5-9B base):
#   outcome_only:  output/grpo_outcome_only_qwen35_9b_mcl4096/v1-20260407-191217/checkpoint-79
#   no_topo:       output/grpo_no_topo_qwen35_9b_mcl4096/v1-20260407-191217/checkpoint-79
#   no_continuity: output/grpo_no_continuity_qwen35_9b/v0-20260404-163247/checkpoint-79
#
# Default GPU: 0 (does not overlap with the main v3 pipelines on 1/2/7).
#
# Usage: bash scripts/rerun_ablations_v3.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${PYTHON_ENV_BIN}:$PATH"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
mkdir -p logs output/eval

GPU="${1:-0}"
MODEL_BASE_9B="${MODEL_ROOT}/qwen/Qwen3.5-9B"

ADAPTER_OUTCOME_ONLY="output/grpo_outcome_only_qwen35_9b_mcl4096/v1-20260407-191217/checkpoint-79"
ADAPTER_NO_TOPO="output/grpo_no_topo_qwen35_9b_mcl4096/v1-20260407-191217/checkpoint-79"
ADAPTER_NO_CONTINUITY="output/grpo_no_continuity_qwen35_9b/v0-20260404-163247/checkpoint-79"

LONG_COT_BENCHES=(olympiadbench omni_math aime2024 aime2025 cnmo2024)
MEDIUM_BENCHES=(gsm8k math500)
SHORT_BENCHES=(mmlu gpqa_diamond)   # livecode dropped 2026-04-21 (math models always 0%)
MMLU_MAX_ITEMS=1500

run_group() {
    local label="$1" adapter="$2" gpu="$3" max_new_tokens="$4" max_items="$5"
    shift 5
    local benches=("$@")

    local cmd=(python3 scripts/bench_transformers.py
        --model "$MODEL_BASE_9B"
        --label "$label"
        --benchmarks "${benches[@]}"
        --num_samples_per_item 5
        --k_values 1 5
        --batch_size 2
        --max_new_tokens "$max_new_tokens"
        --temperature 0.7
        --top_p 0.95
        --use_chat_template
        --sft_style
        --adapter "$adapter"
    )
    [[ "$max_items" -gt 0 ]] && cmd+=(--max_items "$max_items")

    local stamp
    stamp=$(date +%H%M%S)
    local log="logs/eval_${label}_mnt${max_new_tokens}_${stamp}.log"
    echo "[$(date '+%H:%M:%S')] RUN $label gpu=$gpu mnt=$max_new_tokens benches=${benches[*]} -> $log"
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$log" 2>&1 || {
        echo "[WARN] $label group (mnt=$max_new_tokens) non-zero exit"
    }
}

run_model() {
    local label="$1" adapter="$2" gpu="$3"
    run_group "$label" "$adapter" "$gpu" 2560 0 "${LONG_COT_BENCHES[@]}"
    run_group "$label" "$adapter" "$gpu" 1536 0 "${MEDIUM_BENCHES[@]}"
    run_group "$label" "$adapter" "$gpu" 512  "$MMLU_MAX_ITEMS" "${SHORT_BENCHES[@]}"
}

echo "Launching ablation v3 pipeline on GPU $GPU (outcome_only -> no_topo -> no_continuity)"
run_model "outcome_only_9b_v3"  "$ADAPTER_OUTCOME_ONLY"  "$GPU"
run_model "no_topo_9b_v3"       "$ADAPTER_NO_TOPO"       "$GPU"
run_model "no_continuity_9b_v3" "$ADAPTER_NO_CONTINUITY" "$GPU"

echo "[$(date '+%H:%M:%S')] ablation v3 pipeline on GPU $GPU complete"
