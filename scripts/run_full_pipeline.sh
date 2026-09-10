#!/bin/bash
# TopoPRM Full Pipeline: DAG construction -> Training -> Evaluation -> PRM Reranking
# Usage: bash scripts/run_full_pipeline.sh [--phase 1|2|3|4|5|all]
set -euo pipefail
cd "$(dirname "$0")/.." && export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

PHASE="${1:---phase}"
PHASE_VAL="${2:-all}"
if [[ "$PHASE" == "--phase" ]]; then PHASE_VAL="${2:-all}"; fi

GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "========================================"
echo "TopoPRM Full Pipeline"
echo "Phase: ${PHASE_VAL}  GPUs: ${GPUS}"
echo "========================================"

# ========================================
# Phase 1: Large-scale DAG Construction
# ========================================
phase_1_dag() {
    echo ""
    echo "[Phase 1] Building DAGs from public math datasets..."
    mkdir -p data/dag_public data/grpo_ready

    python3 scripts/build_dag_public.py \
        --datasets gsm8k math \
        --output_dir data/dag_public \
        --output_jsonl data/grpo_ready/train_public.jsonl \
        --max_samples -1

    echo "[Phase 1] DAG construction complete."
    echo "  DAGs: data/dag_public/"
    echo "  GRPO data: data/grpo_ready/train_public.jsonl"
    wc -l data/grpo_ready/train_public.jsonl
}

# ========================================
# Phase 2: Baseline Evaluation (DeepSeek-R1-Distill)
# ========================================
phase_2_baseline() {
    echo ""
    echo "[Phase 2] Evaluating baseline DeepSeek-R1-Distill-Qwen-7B..."

    BENCHMARKS="gsm8k math500 aime2024 cnmo2024 mmlu"
    LABEL="baseline_dr1_7b_chat"
    for bench in $BENCHMARKS; do
        echo "  Evaluating ${bench}..."
        NS=5
        MTOK=4096
        EXTRA=(--k_values 1 5)
        BATCH=4
        if [[ "$bench" == "mmlu" ]] || [[ "$bench" == "gpqa_diamond" ]]; then
            NS=1
            EXTRA=(--k_values 1)
            MTOK=512
            BATCH=8
        fi
        python3 scripts/bench_transformers.py \
            --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
            --label "$LABEL" \
            --benchmarks "$bench" \
            --num_samples_per_item "$NS" \
            "${EXTRA[@]}" \
            --batch_size "$BATCH" \
            --max_new_tokens "$MTOK" \
            --use_chat_template \
            --output_dir output/eval || echo "  WARN: ${bench} failed"
    done
    echo "[Phase 2] Baseline evaluation complete."
}

# ========================================
# Phase 3: SFT + GRPO Training
# ========================================
phase_3_train() {
    echo ""
    echo "[Phase 3a] SFT on public DAG data..."
    swift sft --config configs/sft_deepseek_r1_7b.yaml

    echo ""
    echo "[Phase 3b] GRPO with TopoPRM reward..."
    swift rlhf --config configs/grpo_topoprm_deepseek_r1_7b.yaml

    echo "[Phase 3] Training complete."
}

# ========================================
# Phase 4: Evaluation with PRM Reranking
# ========================================
phase_4_eval() {
    echo ""
    echo "[Phase 4] Evaluating trained model with PRM reranking..."

    MODEL_PATH="output/grpo_topoprm_deepseek_r1_7b/checkpoint-best"
    BENCHMARKS="gsm8k math500 aime2024 cnmo2024 mmlu"
    LABEL="topoprm_deepseek_r1_7b"
    for bench in $BENCHMARKS; do
        echo "  Evaluating ${bench} (k=5 for math, k=1 for MCQ)..."
        NS=5
        MTOK=4096
        EXTRA=(--k_values 1 5)
        BATCH=4
        if [[ "$bench" == "mmlu" ]] || [[ "$bench" == "gpqa_diamond" ]]; then
            NS=1
            EXTRA=(--k_values 1)
            MTOK=512
            BATCH=8
        fi
        python3 scripts/bench_transformers.py \
            --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
            --adapter "$MODEL_PATH" \
            --label "$LABEL" \
            --benchmarks "$bench" \
            --num_samples_per_item "$NS" \
            "${EXTRA[@]}" \
            --batch_size "$BATCH" \
            --max_new_tokens "$MTOK" \
            --use_chat_template \
            --sft_style \
            --output_dir output/eval || echo "  WARN: ${bench} failed"
    done
    echo "[Phase 4] Evaluation complete."
}

# ========================================
# Phase 5: Sync results to paper tables
# ========================================
phase_5_sync() {
    echo ""
    echo "[Phase 5] Syncing results to paper tables..."
    python3 scripts/fill_rft_csv.py || echo "  fill_rft_csv skipped"
    echo "[Phase 5] Run scripts/sync_all.sh to update LaTeX tables."
}

# ========================================
# Dispatch
# ========================================
case "$PHASE_VAL" in
    1) phase_1_dag ;;
    2) phase_2_baseline ;;
    3) phase_3_train ;;
    4) phase_4_eval ;;
    5) phase_5_sync ;;
    all)
        phase_1_dag
        phase_2_baseline
        phase_3_train
        phase_4_eval
        phase_5_sync
        ;;
    *) echo "Unknown phase: $PHASE_VAL (use 1-5 or all)"; exit 1 ;;
esac

echo ""
echo "========================================"
echo "Pipeline phase ${PHASE_VAL} complete!"
echo "========================================"
