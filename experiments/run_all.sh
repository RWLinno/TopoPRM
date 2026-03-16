#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/mnt/users/rwl/topoprm"
SWIFT="/mnt/users/conda_env/swift/bin/swift"
PYTHON="${SWIFT%/*}/python"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export SWIFT NPROC_PER_NODE

echo "=========================================="
echo "  TopoPRM: Full NeurIPS Experiment Suite"
echo "  Model: Qwen3-32B"
echo "  GPUs per node: ${NPROC_PER_NODE}"
echo "=========================================="

# 0. Data pipeline
if [ ! -f "${PROJECT_ROOT}/data/grpo_ready/train.jsonl" ]; then
    echo "[0] Running data pipeline..."
    bash "${PROJECT_ROOT}/scripts/run_data_pipeline.sh"
fi

# 1. SFT (skip if checkpoint already exists)
SFT_DIR="${PROJECT_ROOT}/output/sft_qwen3_32b"
if [ ! -d "${SFT_DIR}" ] || [ -z "$(ls -A ${SFT_DIR}/*/checkpoint-* 2>/dev/null)" ]; then
    echo "[SFT] Training Qwen3-32B..."
    $SWIFT sft \
        --config "${PROJECT_ROOT}/configs/sft_qwen3_32b.yaml" \
        2>&1 | tee "${PROJECT_ROOT}/output/sft_qwen3_32b.log"
fi

# Find latest SFT checkpoint
SFT_CKPT=$(find "${SFT_DIR}" -maxdepth 2 -name "checkpoint-*" -type d | sort -t- -k2 -n | tail -1)
echo "Using SFT checkpoint: ${SFT_CKPT}"

# 2. GRPO experiments
for exp in main outcome_only no_topo no_continuity; do
    echo ""
    echo "[GRPO] ${exp}..."
    CONFIG="${PROJECT_ROOT}/experiments/configs/grpo_${exp}.yaml"
    sed -i "s|PLACEHOLDER_SFT_CHECKPOINT|${SFT_CKPT}|g" "${CONFIG}"
    $SWIFT rlhf \
        --rlhf_type grpo \
        --config "${CONFIG}" \
        2>&1 | tee "${PROJECT_ROOT}/output/grpo_${exp}.log"
done

# 3. Evaluate all GRPO models
echo ""
echo "[Eval] Evaluating all models..."
for exp in main outcome_only no_topo no_continuity; do
    echo "  Evaluating ${exp}..."
    $PYTHON -m src.eval.benchmark_runner \
        --model "${PROJECT_ROOT}/output/grpo_${exp}" \
        --output_dir "${PROJECT_ROOT}/output/eval_${exp}" \
        --benchmarks MATH GSM8K CMATH GaoKao 2>&1 || true
done

echo ""
echo "=========================================="
echo "  All experiments complete!"
echo "=========================================="
