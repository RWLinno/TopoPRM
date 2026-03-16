#!/bin/bash
set -euo pipefail

###############################################################################
# Run evaluation on test sets using swift infer + critique evaluator
# Usage: bash scripts/run_eval.sh <adapter_path> <output_name>
#   adapter_path: path to LoRA adapter checkpoint
#   output_name:  e.g. "grpo_main", "grpo_outcome_only", "sft_baseline"
###############################################################################

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PATH="/mnt/users/conda_env/swift/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MODEL="/mnt/users/rwl/models/Qwen/Qwen3-32B"
ADAPTER="${1:?Usage: $0 <adapter_path> <output_name>}"
OUTPUT_NAME="${2:?Usage: $0 <adapter_path> <output_name>}"

mkdir -p output/eval

echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluating: $OUTPUT_NAME"
echo "Model: $MODEL"
echo "Adapter: $ADAPTER"
echo "=========================================="

# Middle school test set (2181 samples)
echo "--- Middle school test set (2181 samples) ---"
swift infer \
    --model "$MODEL" \
    --adapters "$ADAPTER" \
    --val_dataset "data/test/infer_初中测试集全量_2181_20260122.json" \
    --max_new_tokens 2048 \
    --temperature 0.1 \
    --infer_backend vllm \
    --gpu_memory_utilization 0.9 \
    --max_model_len 4096 \
    --result_path "output/eval/${OUTPUT_NAME}_middle.jsonl" \
    2>&1 | tee "output/eval/${OUTPUT_NAME}_middle.log"

# High school test set (5414 samples)
echo "--- High school test set (5414 samples) ---"
swift infer \
    --model "$MODEL" \
    --adapters "$ADAPTER" \
    --val_dataset "data/test/infer_高中测试集全量_5414_20260204.json" \
    --max_new_tokens 2048 \
    --temperature 0.1 \
    --infer_backend vllm \
    --gpu_memory_utilization 0.9 \
    --max_model_len 4096 \
    --result_path "output/eval/${OUTPUT_NAME}_high.jsonl" \
    2>&1 | tee "output/eval/${OUTPUT_NAME}_high.log"

# Run critique evaluation
echo "--- Computing critique metrics ---"
python3 -m src.eval.critique_eval \
    --predictions "output/eval/${OUTPUT_NAME}_middle.jsonl" \
    --ground_truth "data/test/infer_初中测试集全量_2181_20260122.json" \
    --output "output/eval/${OUTPUT_NAME}_middle_metrics.json"

python3 -m src.eval.critique_eval \
    --predictions "output/eval/${OUTPUT_NAME}_high.jsonl" \
    --ground_truth "data/test/infer_高中测试集全量_5414_20260204.json" \
    --output "output/eval/${OUTPUT_NAME}_high_metrics.json"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluation $OUTPUT_NAME done"
