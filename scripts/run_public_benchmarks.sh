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

# ─── GSM8K ───
echo ""
echo "[1/2] Evaluating on GSM8K (1,319 problems)..."
GSM8K_OUTPUT="$EVAL_DIR/${LABEL}_gsm8k_metrics.json"

if [ -f "$GSM8K_OUTPUT" ]; then
    echo "  Already exists: $GSM8K_OUTPUT (delete to re-run)"
else
    GSM_DIR="$EVAL_DIR/benchmark_${LABEL}_gsm8k"
    mkdir -p "$GSM_DIR" logs
    if [ -n "$ADAPTER" ] && [ -d "$ADAPTER" ]; then
        swift eval --model "$MODEL" --adapters "$ADAPTER" --eval_dataset gsm8k \
            --eval_output_dir "$GSM_DIR" --max_new_tokens 2048 \
            2>&1 | tee "logs/benchmark_${LABEL}_gsm8k.log"
    else
        swift eval --model "$MODEL" --eval_dataset gsm8k \
            --eval_output_dir "$GSM_DIR" --max_new_tokens 2048 \
            2>&1 | tee "logs/benchmark_${LABEL}_gsm8k.log"
    fi
    python3 - <<PY
import json, glob
from pathlib import Path
reports = sorted(Path("$GSM_DIR").rglob("reports/*/gsm8k.json"))
if not reports:
    raise SystemExit("No gsm8k.json under $GSM_DIR")
rep = reports[-1]
acc = float(json.loads(rep.read_text(encoding="utf-8"))["score"])
Path("$GSM8K_OUTPUT").write_text(
    json.dumps({"accuracy": acc, "source": str(rep)}, indent=2) + "\\n",
    encoding="utf-8",
)
print("wrote", "$GSM8K_OUTPUT", "acc=", acc)
PY
    ACC=$(python3 -c "import json; d=json.load(open('$GSM8K_OUTPUT')); print(f\"{d.get('accuracy', d.get('acc', '?')):.1f}%\")" 2>/dev/null || echo "?")
    echo "  GSM8K accuracy: $ACC"
fi

# ─── MATH-500 ───
echo ""
echo "[2/2] Evaluating on MATH-500 (500 problems)..."
MATH_OUTPUT="$EVAL_DIR/${LABEL}_math500_metrics.json"

if [ -f "$MATH_OUTPUT" ]; then
    echo "  Already exists: $MATH_OUTPUT (delete to re-run)"
else
    MATH_DIR="$EVAL_DIR/benchmark_${LABEL}_math500"
    mkdir -p "$MATH_DIR" logs
    if [ -n "$ADAPTER" ] && [ -d "$ADAPTER" ]; then
        swift eval --model "$MODEL" --adapters "$ADAPTER" --eval_dataset math_500 \
            --eval_output_dir "$MATH_DIR" --max_new_tokens 2048 \
            2>&1 | tee "logs/benchmark_${LABEL}_math500.log"
    else
        swift eval --model "$MODEL" --eval_dataset math_500 \
            --eval_output_dir "$MATH_DIR" --max_new_tokens 2048 \
            2>&1 | tee "logs/benchmark_${LABEL}_math500.log"
    fi
    python3 - <<PY
import json
from pathlib import Path
reports = sorted(Path("$MATH_DIR").rglob("reports/*/math_500.json"))
if not reports:
    raise SystemExit("No math_500.json under $MATH_DIR")
rep = reports[-1]
acc = float(json.loads(rep.read_text(encoding="utf-8"))["score"])
Path("$MATH_OUTPUT").write_text(
    json.dumps({"accuracy": acc, "source": str(rep)}, indent=2) + "\\n",
    encoding="utf-8",
)
print("wrote", "$MATH_OUTPUT", "acc=", acc)
PY
    ACC=$(python3 -c "import json; d=json.load(open('$MATH_OUTPUT')); print(f\"{d.get('accuracy', d.get('acc', '?')):.1f}%\")" 2>/dev/null || echo "?")
    echo "  MATH-500 accuracy: $ACC"
fi

echo ""
echo "══════════════════════════════════════════"
echo " Public benchmark evaluation complete."
echo " Results: $EVAL_DIR/${LABEL}_gsm8k_metrics.json"
echo "          $EVAL_DIR/${LABEL}_math500_metrics.json"
echo "══════════════════════════════════════════"
