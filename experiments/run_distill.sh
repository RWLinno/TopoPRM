#!/bin/bash
# =============================================================
#  Distillation: Reverse-KL reasoning compression
#  Multi-GPU deployment
# =============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SWIFT="${SWIFT:-swift}"
PYTHON="${SWIFT%/*}/python"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export NPROC_PER_NODE

TEACHER_MODEL="${1:-${PROJECT_ROOT}/output/grpo_main}"

echo "=========================================="
echo "  TopoPRM Distillation (${NPROC_PER_NODE} GPUs)"
echo "  Teacher: ${TEACHER_MODEL}"
echo "=========================================="

# Step 1: Generate teacher traces
echo "[Step 1] Generating teacher reasoning traces..."
$SWIFT infer \
    --model "${TEACHER_MODEL}" \
    --val_dataset "${PROJECT_ROOT}/data/sft_ready/train.jsonl" \
    --max_new_tokens 4096 \
    --temperature 0.3 \
    --result_path "${PROJECT_ROOT}/output/teacher_traces.jsonl" \
    2>&1 | tee "${PROJECT_ROOT}/output/teacher_infer.log" || true

# Step 2: Filter & score traces
echo "[Step 2] Filtering compact traces..."
$PYTHON -m src.data.generate_distill_data \
    --input_path "${PROJECT_ROOT}/data/sft_ready/train.jsonl" \
    --output_path "${PROJECT_ROOT}/data/sft_ready/distill_train.jsonl" \
    --teacher_traces "${PROJECT_ROOT}/output/teacher_traces.jsonl" \
    --max_steps 15 \
    --max_chars 2000

# Step 3: Distill to 7B
echo "[Step 3] Distilling to Qwen3-7B..."
$SWIFT sft \
    --config "${PROJECT_ROOT}/experiments/configs/distill_7b.yaml" \
    2>&1 | tee "${PROJECT_ROOT}/output/distill_7b.log"

# Step 4: Distill to 1.5B
echo "[Step 4] Distilling to Qwen3-1.5B..."
$SWIFT sft \
    --config "${PROJECT_ROOT}/experiments/configs/distill_1_5b.yaml" \
    2>&1 | tee "${PROJECT_ROOT}/output/distill_1_5b.log"

# Step 5: Evaluate
echo "[Step 5] Evaluating distilled models..."
for size in 7b 1_5b; do
    $PYTHON -m src.eval.benchmark_runner \
        --model "${PROJECT_ROOT}/output/distill_${size}" \
        --output_dir "${PROJECT_ROOT}/output/eval_distill_${size}" \
        --benchmarks MATH GSM8K CMATH GaoKao 2>&1 || true
done

echo "  Distillation complete!"
