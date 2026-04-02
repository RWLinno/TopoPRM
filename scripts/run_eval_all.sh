#!/bin/bash
set -euo pipefail

###############################################################################
# Evaluate ALL trained models on private + public benchmarks.
# Covers: base model, SFT, all GRPO variants, distilled student.
#
# Usage: bash scripts/run_eval_all.sh
###############################################################################

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PATH="/mnt/users/conda_env/topoprm/bin:$PATH"
export PATH="$(dirname $(which swift)):$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# All GRPO experiments to evaluate
GRPO_EXPERIMENTS=(
  "grpo_hierarchical"
  "grpo_main"
  "grpo_outcome_only"
  "grpo_no_topo"
  "grpo_no_continuity"
  "grpo_clipped"
  "grpo_confgate"
  "grpo_mulgate"
  "grpo_scae"
)

# Distillation experiments
DISTILL_EXPERIMENTS=(
  "distill_7b_compact_rkl"
  "distill_7b_compact"
)

# Public benchmark models (model_path:adapter_path:label)
PUBLIC_MODELS=(
  "Qwen/Qwen3-32B::base_qwen3_32b"
  "Qwen/Qwen2.5-7B-Instruct::qwen25_7b_instruct"
  "meta-llama/Llama-3.1-8B-Instruct::llama31_8b_instruct"
)

echo "══════════════════════════════════════════"
echo " TopoPRM Full Evaluation Pipeline"
echo "══════════════════════════════════════════"

find_best_ckpt() {
    local exp_dir="output/$1"
    find "$exp_dir" -maxdepth 3 -name "checkpoint-*" -type d 2>/dev/null \
        | sort -V | tail -1
}

find_sft_ckpt() {
    find "output/sft" -maxdepth 3 -name "checkpoint-*" -type d 2>/dev/null \
        | sort -V | tail -1 || true
}

EVAL_COUNT=0
SKIP_COUNT=0

# ─── 1) SFT baseline ───
echo ""
echo "─── SFT Baseline ───"
SFT_CKPT="$(find_sft_ckpt)"
if [ -n "${SFT_CKPT:-}" ] && [ -d "$SFT_CKPT" ]; then
    echo "  [EVAL] SFT baseline: $SFT_CKPT"
    bash scripts/run_eval.sh "$SFT_CKPT" "sft_baseline" && ((EVAL_COUNT++)) || true
else
    echo "  [SKIP] SFT baseline (checkpoint not found)"
    ((SKIP_COUNT++))
fi

# ─── 2) GRPO variants (private benchmarks) ───
echo ""
echo "─── GRPO Variants (Private Benchmarks) ───"
for exp in "${GRPO_EXPERIMENTS[@]}"; do
    CKPT=$(find_best_ckpt "$exp")
    if [ -n "$CKPT" ]; then
        echo "  [EVAL] $exp: $CKPT"
        bash scripts/run_eval.sh "$CKPT" "$exp" && ((EVAL_COUNT++)) || true
    else
        echo "  [SKIP] $exp (no checkpoint)"
        ((SKIP_COUNT++))
    fi
done

# ─── 3) Distillation models ───
echo ""
echo "─── Distillation Models ───"
for exp in "${DISTILL_EXPERIMENTS[@]}"; do
    CKPT=$(find_best_ckpt "$exp")
    if [ -n "$CKPT" ]; then
        echo "  [EVAL] $exp: $CKPT"
        bash scripts/run_eval.sh "$CKPT" "$exp" && ((EVAL_COUNT++)) || true
    else
        echo "  [SKIP] $exp (no checkpoint)"
        ((SKIP_COUNT++))
    fi
done

# ─── 4) Public benchmarks (GSM8K + MATH-500) ───
echo ""
echo "─── Public Benchmarks (GSM8K + MATH-500) ───"

# TopoPRM teacher (hierarchical)
HIER_CKPT=$(find_best_ckpt "grpo_hierarchical")
if [ -n "$HIER_CKPT" ]; then
    echo "  [EVAL] TopoPRM teacher (public)"
    bash scripts/run_public_benchmarks.sh "Qwen/Qwen3-32B" "$HIER_CKPT" "grpo_hierarchical" && ((EVAL_COUNT++)) || true
fi

# Distilled student
for exp in "${DISTILL_EXPERIMENTS[@]}"; do
    CKPT=$(find_best_ckpt "$exp")
    if [ -n "$CKPT" ]; then
        echo "  [EVAL] $exp (public)"
        bash scripts/run_public_benchmarks.sh "Qwen/Qwen3-8B" "$CKPT" "$exp" && ((EVAL_COUNT++)) || true
    fi
done

# External reference models
for entry in "${PUBLIC_MODELS[@]}"; do
    IFS=':' read -r model adapter label <<< "$entry"
    echo "  [EVAL] $label (public)"
    bash scripts/run_public_benchmarks.sh "$model" "$adapter" "$label" && ((EVAL_COUNT++)) || true
done

# ─── 5) DAG structural metrics ───
echo ""
echo "─── DAG Structural Metrics ───"
if [ -f scripts/run_dag_metrics.sh ]; then
    bash scripts/run_dag_metrics.sh && ((EVAL_COUNT++)) || true
fi

# ─── 6) Export paper tables ───
echo ""
echo "─── Exporting Paper Tables ───"
python3 -m src.eval.export_paper_tables \
    --eval_dir output/eval \
    --output output/eval/paper_table_summary.csv || true

# ─── Summary ───
echo ""
echo "══════════════════════════════════════════"
echo " Evaluation complete."
echo " Evaluated: $EVAL_COUNT models"
echo " Skipped:   $SKIP_COUNT models"
echo " Results:   output/eval/"
echo "══════════════════════════════════════════"

echo ""
echo "=== Results Summary ==="
for f in output/eval/*_metrics.json; do
    [ -f "$f" ] || continue
    name=$(basename "$f" _metrics.json)
    acc=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d.get('score_accuracy', d.get('accuracy', '?')):.1f}%\")" 2>/dev/null || echo "?")
    echo "  $name: $acc"
done
