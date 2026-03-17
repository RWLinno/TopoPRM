#!/bin/bash
set -euo pipefail

###############################################################################
# Complete GRPO pipeline with vLLM server mode (ms-swift paradigm)
#
# Architecture:
#   GPUs 0-3: vLLM rollout server (tensor_parallel=4)
#   GPUs 4-7: GRPO training (NPROC_PER_NODE=4)
#
# This script:
#   1. Launches vLLM rollout server in background
#   2. Waits for server to be ready
#   3. Runs GRPO experiments sequentially
#   4. Shuts down server when done
#
# Usage: nohup bash scripts/run_full_grpo_pipeline.sh > output/pipeline.log 2>&1 &
###############################################################################

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PATH="$(dirname $(which swift)):$PATH"

PORT=8000
SERVER_PID=""

cleanup() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Shutting down rollout server (PID: $SERVER_PID)"
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

echo "=========================================="
echo " TopoPRM Full GRPO Pipeline (Server Mode)"
echo " Architecture: GPUs 0-3 (rollout) + GPUs 4-7 (training)"
echo "=========================================="

# Step 1: Launch rollout server
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching vLLM rollout server on GPUs 0-3..."
CUDA_VISIBLE_DEVICES=0,1,2,3 swift rollout \
    --model Qwen/Qwen3-32B \
    --adapters $(cd "$(dirname "$0")/.." && pwd)/output/sft_qwen3_32b/v1-20260316-154443/checkpoint-120 \
    --vllm_tensor_parallel_size 4 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 4096 \
    --vllm_enable_prefix_caching false \
    --vllm_enable_lora true \
    --vllm_max_lora_rank 64 \
    --port $PORT \
    > output/rollout_server.log 2>&1 &
SERVER_PID=$!
echo "Rollout server PID: $SERVER_PID"

# Step 2: Wait for server to be ready
echo "Waiting for rollout server to start..."
MAX_WAIT=600
WAITED=0
while ! curl -s http://127.0.0.1:${PORT}/v1/models > /dev/null 2>&1; do
    sleep 10
    WAITED=$((WAITED + 10))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "[ERROR] Server failed to start within ${MAX_WAIT}s"
        cat output/rollout_server.log | tail -20
        exit 1
    fi
    echo "  ... waiting (${WAITED}s / ${MAX_WAIT}s)"
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rollout server ready!"

# Step 3: Run experiments sequentially
EXPERIMENTS=("grpo_main" "grpo_outcome_only" "grpo_no_topo" "grpo_no_continuity")

for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Starting: $exp"
    bash scripts/run_grpo.sh "$exp"
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Finished: $exp"
    sleep 10
done

echo ""
echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All GRPO experiments completed!"
echo "=========================================="
