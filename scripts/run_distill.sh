#!/bin/bash
set -euo pipefail

###############################################################################
# Run distillation pipeline: generate teacher traces -> filter -> train student
#
# Usage: bash scripts/run_distill.sh [config_name]
#   config_name: distill config in configs/ (default: distill_7b_compact)
#
# Prerequisites:
#   - Teacher checkpoint exists (output/grpo_hierarchical or grpo_main)
#   - SFT data exists in data/
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PATH="/mnt/users/conda_env/topoprm/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

CONFIG_NAME="${1:-distill_7b_compact}"
CONFIG="configs/${CONFIG_NAME}.yaml"
[ ! -f "$CONFIG" ] && echo "[ERROR] Config not found: $CONFIG" && exit 1

TEACHER_MODEL="Qwen/Qwen3-32B"
STUDENT_MODEL="Qwen/Qwen3-8B"
QUALITY_THRESHOLD="${DISTILL_QUALITY_THRESHOLD:-0.7}"

# Find teacher checkpoint (prefer hierarchical, fall back to main)
find_teacher_ckpt() {
    local ckpt=""
    for exp in grpo_hierarchical grpo_main grpo_clipped; do
        ckpt=$(find "output/$exp" -maxdepth 3 -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1 || true)
        [ -n "$ckpt" ] && echo "$ckpt" && return 0
    done
    echo ""
}

TEACHER_CKPT="$(find_teacher_ckpt)"
if [ -z "$TEACHER_CKPT" ]; then
    echo "[ERROR] No teacher checkpoint found. Run GRPO training first."
    exit 1
fi

echo "══════════════════════════════════════════"
echo " TopoPRM Distillation Pipeline"
echo "══════════════════════════════════════════"
echo "  Teacher: $TEACHER_MODEL + $TEACHER_CKPT"
echo "  Student: $STUDENT_MODEL"
echo "  Config:  $CONFIG"
echo "  Quality threshold: $QUALITY_THRESHOLD"
echo "══════════════════════════════════════════"

mkdir -p output/distill_data output/distill_logs

# ─── Step 1: Generate teacher traces ───
echo ""
echo "[Step 1/4] Generating teacher traces..."
TRACE_OUTPUT="output/distill_data/teacher_traces.jsonl"

if [ -f "$TRACE_OUTPUT" ] && [ -s "$TRACE_OUTPUT" ]; then
    echo "  Teacher traces already exist ($TRACE_OUTPUT), skipping generation."
    echo "  Delete the file to regenerate."
else
    python3 -m src.data.generate_distill_data \
        --model "$TEACHER_MODEL" \
        --adapter "$TEACHER_CKPT" \
        --dataset "data/grpo_ready/train.jsonl" \
        --output "$TRACE_OUTPUT" \
        --num_samples_per_prompt 4 \
        --temperature 0.7 \
        --max_new_tokens 2048 \
        2>&1 | tee "output/distill_logs/generate_traces.log"
fi

TOTAL_TRACES=$(wc -l < "$TRACE_OUTPUT" 2>/dev/null || echo 0)
echo "  Total teacher traces: $TOTAL_TRACES"

# ─── Step 2: Filter by process-aware quality ───
echo ""
echo "[Step 2/4] Filtering traces (threshold=$QUALITY_THRESHOLD)..."
FILTERED_OUTPUT="output/distill_data/filtered_traces.jsonl"

python3 -m src.distill.teacher_trace_filter \
    --input "$TRACE_OUTPUT" \
    --output "$FILTERED_OUTPUT" \
    --quality_threshold "$QUALITY_THRESHOLD" \
    2>&1 | tee "output/distill_logs/filter_traces.log"

FILTERED_TRACES=$(wc -l < "$FILTERED_OUTPUT" 2>/dev/null || echo 0)
echo "  Filtered traces: $FILTERED_TRACES / $TOTAL_TRACES ($(( FILTERED_TRACES * 100 / (TOTAL_TRACES + 1) ))%)"

if [ "$FILTERED_TRACES" -lt 100 ]; then
    echo "[WARNING] Very few traces passed filtering. Consider lowering threshold."
fi

# ─── Step 3: Train student via reverse-KL ───
echo ""
echo "[Step 3/4] Training student model..."

swift sft \
    --config "$CONFIG" \
    --model "$STUDENT_MODEL" \
    --dataset "$FILTERED_OUTPUT" \
    2>&1 | tee "output/distill_logs/student_train.log"

# ─── Step 4: Evaluate student ───
echo ""
echo "[Step 4/4] Evaluating student..."

# Find student checkpoint
STUDENT_CKPT=$(find "output/${CONFIG_NAME}" -maxdepth 3 -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1 || true)

if [ -n "$STUDENT_CKPT" ]; then
    echo "  Student checkpoint: $STUDENT_CKPT"

    # Private benchmarks
    bash scripts/run_eval.sh "$STUDENT_CKPT" "distill_${CONFIG_NAME}"

    # Public benchmarks
    if [ -f scripts/run_public_benchmarks.sh ]; then
        bash scripts/run_public_benchmarks.sh "$STUDENT_MODEL" "$STUDENT_CKPT" "distill_${CONFIG_NAME}"
    fi
else
    echo "[WARNING] No student checkpoint found. Check training logs."
fi

echo ""
echo "══════════════════════════════════════════"
echo " Distillation complete."
echo " Results: output/eval/distill_${CONFIG_NAME}_*_metrics.json"
echo "══════════════════════════════════════════"
