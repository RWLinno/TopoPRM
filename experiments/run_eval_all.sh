#!/bin/bash
# =============================================================
#  Evaluate ALL trained TopoPRM models on benchmarks
#  Multi-GPU deployment
# =============================================================
set -euo pipefail

PROJECT_ROOT="/mnt/users/rwl/topoprm"
SWIFT="${SWIFT:-/mnt/users/conda_env/swift/bin/swift}"
PYTHON="${SWIFT%/*}/python"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export NPROC_PER_NODE

BENCHMARKS="MATH GSM8K CMATH GaoKao"

echo "=========================================="
echo "  TopoPRM: Evaluate All Models"
echo "  Benchmarks: ${BENCHMARKS}"
echo "=========================================="

# SFT baseline
SFT_CKPT="${PROJECT_ROOT}/output/sft_qwen3_14b/v0-20260313-195147/checkpoint-120"
if [ -d "${SFT_CKPT}" ]; then
    echo ""
    echo "[Eval] SFT baseline..."
    $PYTHON -m src.eval.benchmark_runner \
        --model "${SFT_CKPT}" \
        --output_dir "${PROJECT_ROOT}/output/eval_sft_baseline" \
        --benchmarks ${BENCHMARKS} 2>&1 || true
fi

# GRPO experiments
for exp in main outcome_only no_topo no_continuity; do
    MODEL_DIR="${PROJECT_ROOT}/output/grpo_${exp}"
    if [ -d "${MODEL_DIR}" ]; then
        echo ""
        echo "[Eval] GRPO ${exp}..."
        $PYTHON -m src.eval.benchmark_runner \
            --model "${MODEL_DIR}" \
            --output_dir "${PROJECT_ROOT}/output/eval_${exp}" \
            --benchmarks ${BENCHMARKS} 2>&1 || true
    else
        echo "[SKIP] ${MODEL_DIR} not found"
    fi
done

# Distilled models
for size in 7b 1_5b; do
    MODEL_DIR="${PROJECT_ROOT}/output/distill_${size}"
    if [ -d "${MODEL_DIR}" ]; then
        echo ""
        echo "[Eval] Distill ${size}..."
        $PYTHON -m src.eval.benchmark_runner \
            --model "${MODEL_DIR}" \
            --output_dir "${PROJECT_ROOT}/output/eval_distill_${size}" \
            --benchmarks ${BENCHMARKS} 2>&1 || true
    else
        echo "[SKIP] ${MODEL_DIR} not found"
    fi
done

echo ""
echo "=========================================="
echo "  All evaluations complete!"
echo "=========================================="

# Print summary if results exist
echo ""
echo "Results directories:"
for d in "${PROJECT_ROOT}"/output/eval_*; do
    [ -d "$d" ] && echo "  $d"
done
