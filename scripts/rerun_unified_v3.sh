#!/usr/bin/env bash
# Unified benchmark re-run (v3) with corrected max_new_tokens per group.
#
# Groups:
#   long_cot  (olympiadbench, omni_math, aime2024, aime2025, cnmo2024) -> 2560
#   medium    (gsm8k, math500)                                         -> 1536
#   short     (mmlu[subset], gpqa_diamond, livecode)                   -> 512
#
# GPU allocation (serial per GPU, no stacking):
#   GPU 1: base_9b, topoprm_hier_9b_v3
#   GPU 2: sft_9b_v3, topoprm_gated_9b_v3
#   GPU 7: topoprm_hier_qwen25_7b_v3, base_4b_v3, student_4b_sft_distill_v3
#
# Usage:
#   bash scripts/rerun_unified_v3.sh [GPU]    # run all models assigned to a GPU
#   bash scripts/rerun_unified_v3.sh all      # launch all three GPU pipelines in parallel

set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${PYTHON_ENV_BIN}:$PATH"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
mkdir -p logs output/eval

# ---------------------------------------------------------------------------
# Model registry: label | model_path | adapter_path | sft_style_flag
# ---------------------------------------------------------------------------
MODEL_BASE_9B="${MODEL_ROOT}/qwen/Qwen3.5-9B"
MODEL_BASE_4B="${MODEL_ROOT}/qwen/Qwen3.5-4B"
MODEL_BASE_Q25_7B="${MODEL_ROOT}/qwen/Qwen2.5-7B-Instruct"

ADAPTER_SFT_9B="output/sft_qwen35_9b/v0-20260407-011328/checkpoint-626"
ADAPTER_HIER_9B="output/grpo_hierarchical_qwen35_9b_mcl4096/v2-20260407-162048/checkpoint-79"
ADAPTER_GATED_9B="output/grpo_gated_qwen35_9b_mcl4096/v4-20260407-111747/checkpoint-79"
ADAPTER_HIER_Q25_7B="output/grpo_hierarchical_qwen25_7b/v3-20260406-134423/checkpoint-318"
ADAPTER_DISTILL_4B="output/sft_distill_4b/v0-20260417-121952/checkpoint-2034"

# ---------------------------------------------------------------------------
# Benchmark groups
# ---------------------------------------------------------------------------
LONG_COT_BENCHES=(olympiadbench omni_math aime2024 aime2025 cnmo2024)
MEDIUM_BENCHES=(gsm8k math500)
SHORT_BENCHES=(mmlu gpqa_diamond)   # livecode dropped 2026-04-21 (math models always 0%)

MMLU_MAX_ITEMS=1500    # fixed subset for tractable runtime (~10x faster than full)

# ---------------------------------------------------------------------------
# One benchmark group runner
# Args: $1=label $2=model $3=adapter $4=sft_style($true|false)
#       $5=gpu $6=max_new_tokens $7=max_items $8...=benches
# ---------------------------------------------------------------------------
run_group() {
    local label="$1" model="$2" adapter="$3" sft_style="$4"
    local gpu="$5" max_new_tokens="$6" max_items="$7"
    shift 7
    local benches=("$@")

    local cmd=(python3 scripts/bench_transformers.py
        --model "$model"
        --label "$label"
        --benchmarks "${benches[@]}"
        --num_samples_per_item 5
        --k_values 1 5
        --batch_size 2
        --max_new_tokens "$max_new_tokens"
        --temperature 0.7
        --top_p 0.95
        --use_chat_template
    )
    if [[ "$sft_style" == "true" ]]; then
        cmd+=(--sft_style)
    fi
    [[ -n "$adapter" ]] && cmd+=(--adapter "$adapter")
    [[ "$max_items" -gt 0 ]] && cmd+=(--max_items "$max_items")

    local stamp group_tag log
    stamp=$(date +%H%M%S)
    group_tag="mnt${max_new_tokens}"
    log="logs/eval_${label}_${group_tag}_${stamp}.log"

    echo "[$(date '+%H:%M:%S')] RUN $label gpu=$gpu mnt=$max_new_tokens benches=${benches[*]} -> $log"
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$log" 2>&1 || {
        echo "[WARN] $label group (mnt=$max_new_tokens) exited non-zero, continuing"
    }
}

# ---------------------------------------------------------------------------
# One model runner: runs long -> medium -> short serially
# Args: $1=label $2=model $3=adapter $4=sft_style $5=gpu
# ---------------------------------------------------------------------------
run_model() {
    local label="$1" model="$2" adapter="$3" sft_style="$4" gpu="$5"
    run_group "$label" "$model" "$adapter" "$sft_style" "$gpu" 2560 0 "${LONG_COT_BENCHES[@]}"
    run_group "$label" "$model" "$adapter" "$sft_style" "$gpu" 1536 0 "${MEDIUM_BENCHES[@]}"
    run_group "$label" "$model" "$adapter" "$sft_style" "$gpu" 512  "$MMLU_MAX_ITEMS" "${SHORT_BENCHES[@]}"
}

# ---------------------------------------------------------------------------
# GPU pipelines: each function runs the models assigned to that GPU in series
# ---------------------------------------------------------------------------
run_gpu1() {
    run_model "base_9b_v3"          "$MODEL_BASE_9B" ""                   false 1
    run_model "topoprm_hier_9b_v3"  "$MODEL_BASE_9B" "$ADAPTER_HIER_9B"   true  1
}

run_gpu2() {
    run_model "sft_9b_v3"           "$MODEL_BASE_9B" "$ADAPTER_SFT_9B"    true  2
    run_model "topoprm_gated_9b_v3" "$MODEL_BASE_9B" "$ADAPTER_GATED_9B"  true  2
}

run_gpu7() {
    run_model "topoprm_hier_qwen25_7b_v3" "$MODEL_BASE_Q25_7B" "$ADAPTER_HIER_Q25_7B" true  7
    run_model "base_4b_v3"                "$MODEL_BASE_4B"     ""                     false 7
    run_model "student_4b_sft_distill_v3" "$MODEL_BASE_4B"     "$ADAPTER_DISTILL_4B"  true  7
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
action="${1:-all}"
case "$action" in
    1) run_gpu1 ;;
    2) run_gpu2 ;;
    7) run_gpu7 ;;
    all)
        echo "launching GPU1/2/7 pipelines in background"
        nohup bash "$0" 1 >logs/pipeline_gpu1.log 2>&1 &
        echo "gpu1 pid=$!"
        nohup bash "$0" 2 >logs/pipeline_gpu2.log 2>&1 &
        echo "gpu2 pid=$!"
        nohup bash "$0" 7 >logs/pipeline_gpu7.log 2>&1 &
        echo "gpu7 pid=$!"
        wait
        ;;
    *)
        echo "Usage: $0 [1|2|7|all]"
        exit 1
        ;;
esac
