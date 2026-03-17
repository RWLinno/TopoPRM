#!/bin/bash
set -euo pipefail

###############################################################################
# Launch vLLM rollout server for GRPO training (ms-swift server mode)
#
# This runs on GPUs 0-3 with tensor_parallel=4
# The GRPO training connects from GPUs 4-7
#
# Usage: bash scripts/start_rollout_server.sh
###############################################################################

cd "$(dirname "$0")/.."
export PATH="$(dirname $(which swift)):$PATH"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL="Qwen/Qwen3-32B"
SFT_ADAPTER="output/sft_qwen3_32b/v1-20260316-154443/checkpoint-120"
PORT="${1:-8000}"

echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting vLLM Rollout Server"
echo "Model: $MODEL"
echo "Adapter: $SFT_ADAPTER"
echo "GPUs: $CUDA_VISIBLE_DEVICES (tensor_parallel=4)"
echo "Port: $PORT"
echo "=========================================="

swift rollout \
    --model "$MODEL" \
    --adapters "$SFT_ADAPTER" \
    --vllm_tensor_parallel_size 4 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 4096 \
    --vllm_enable_prefix_caching false \
    --vllm_enable_lora true \
    --vllm_max_lora_rank 64 \
    --port "$PORT" \
    --max_new_tokens 2048 \
    2>&1 | tee "output/rollout_server.log"
