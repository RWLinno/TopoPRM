#!/bin/bash
set -euo pipefail

###############################################################################
# Run public math benchmarks (GSM8K + MATH-500) for a given model.
#
# Usage:
#   bash scripts/run_public_benchmarks.sh <model_name_or_path> [adapter_path] [label]
#
# Examples:
#   bash scripts/run_public_benchmarks.sh Qwen/Qwen3-32B "" base_qwen3_32b
#   bash scripts/run_public_benchmarks.sh Qwen/Qwen3-8B output/distill_7b_compact/checkpoint-500 distill_8b
#   bash scripts/run_public_benchmarks.sh Qwen/Qwen2.5-7B-Instruct "" qwen25_7b_instruct
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source scripts/gpu_guard.sh 2>/dev/null || true

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PATH="/mnt/users/conda_env/topoprm/bin:$PATH"

MODEL="${1:?Usage: run_public_benchmarks.sh <model> [adapter] [label]}"
ADAPTER="${2:-}"
LABEL="${3:-$(basename "$MODEL" | tr '/' '_')}"

EVAL_DIR="output/eval"
mkdir -p "$EVAL_DIR"

GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
TP_SIZE="${TP_SIZE:-$NUM_GPUS}"

echo "══════════════════════════════════════════"
echo " Public Benchmark Evaluation"
echo "══════════════════════════════════════════"
echo "  Model:   $MODEL"
echo "  Adapter: ${ADAPTER:-none}"
echo "  Label:   $LABEL"
echo "  GPUs:    $GPU_IDS (TP=$TP_SIZE)"
echo "══════════════════════════════════════════"

# Build adapter flag
ADAPTER_FLAG=""
if [ -n "$ADAPTER" ] && [ -d "$ADAPTER" ]; then
    ADAPTER_FLAG="--adapters $ADAPTER"
fi

# ─── GSM8K ───
echo ""
echo "[1/2] Evaluating on GSM8K (1,319 problems)..."
GSM8K_OUTPUT="$EVAL_DIR/${LABEL}_gsm8k_metrics.json"

if [ -f "$GSM8K_OUTPUT" ]; then
    echo "  Already exists: $GSM8K_OUTPUT (delete to re-run)"
else
    python3 -m src.eval.benchmark_runner \
        --model "$MODEL" \
        $ADAPTER_FLAG \
        --benchmark gsm8k \
        --split test \
        --output "$GSM8K_OUTPUT" \
        --tensor_parallel_size "$TP_SIZE" \
        --max_new_tokens 2048 \
        --temperature 0.0 \
        --batch_size 32 \
        2>&1 | tee "logs/benchmark_${LABEL}_gsm8k.log"

    if [ -f "$GSM8K_OUTPUT" ]; then
        ACC=$(python3 -c "import json; d=json.load(open('$GSM8K_OUTPUT')); print(f\"{d.get('accuracy', d.get('acc', '?')):.1f}%\")" 2>/dev/null || echo "?")
        echo "  GSM8K accuracy: $ACC"
    fi
fi

# ─── MATH-500 ───
echo ""
echo "[2/2] Evaluating on MATH-500 (500 problems)..."
MATH_OUTPUT="$EVAL_DIR/${LABEL}_math500_metrics.json"

if [ -f "$MATH_OUTPUT" ]; then
    echo "  Already exists: $MATH_OUTPUT (delete to re-run)"
else
    python3 -m src.eval.benchmark_runner \
        --model "$MODEL" \
        $ADAPTER_FLAG \
        --benchmark math \
        --split test \
        --num_samples 500 \
        --output "$MATH_OUTPUT" \
        --tensor_parallel_size "$TP_SIZE" \
        --max_new_tokens 2048 \
        --temperature 0.0 \
        --batch_size 16 \
        2>&1 | tee "logs/benchmark_${LABEL}_math500.log"

    if [ -f "$MATH_OUTPUT" ]; then
        ACC=$(python3 -c "import json; d=json.load(open('$MATH_OUTPUT')); print(f\"{d.get('accuracy', d.get('acc', '?')):.1f}%\")" 2>/dev/null || echo "?")
        echo "  MATH-500 accuracy: $ACC"
    fi
fi

echo ""
echo "══════════════════════════════════════════"
echo " Public benchmark evaluation complete."
echo " Results: $EVAL_DIR/${LABEL}_gsm8k_metrics.json"
echo "          $EVAL_DIR/${LABEL}_math500_metrics.json"
echo "══════════════════════════════════════════"
