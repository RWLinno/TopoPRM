#!/bin/bash
set -euo pipefail

###############################################################################
# Generate distillation training data from the TopoPRM-trained 32B teacher
# 
# This script:
#   1. Uses the GRPO-trained teacher (checkpoint-150) to generate responses
#      on the training prompts
#   2. Filters responses by composite reward threshold
#   3. Outputs distillation-ready JSONL for student training
###############################################################################

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PATH="/mnt/users/conda_env/swift/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MODEL="/mnt/users/rwl/models/Qwen/Qwen3-32B"
TEACHER_ADAPTER="output/grpo_main/v5-20260316-110039/checkpoint-150"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generating distillation data from teacher..."
echo "Model: $MODEL"
echo "Teacher adapter: $TEACHER_ADAPTER"

# Generate teacher responses on GRPO training data
swift infer \
    --model "$MODEL" \
    --adapters "$TEACHER_ADAPTER" \
    --val_dataset "data/grpo_ready/train.jsonl" \
    --max_new_tokens 2048 \
    --temperature 0.3 \
    --do_sample true \
    --num_return_sequences 3 \
    --infer_backend vllm \
    --gpu_memory_utilization 0.9 \
    --max_model_len 4096 \
    --result_path "data/sft_ready/teacher_responses.jsonl" \
    2>&1 | tee "output/distill_data_gen.log"

# Filter and format for distillation
python3 -c "
import json, sys
sys.path.insert(0, '.')
from src.reward.composite_reward import TopoCompositeReward
reward_fn = TopoCompositeReward()

input_path = 'data/sft_ready/teacher_responses.jsonl'
output_path = 'data/sft_ready/distill_train.jsonl'
threshold = 0.07

kept = 0
total = 0
with open(input_path) as fin, open(output_path, 'w') as fout:
    for line in fin:
        record = json.loads(line)
        total += 1
        text = record.get('prediction', record.get('output', ''))
        if not text:
            continue
        score = reward_fn([text])[0]
        if score >= threshold:
            msg = record.get('messages', [])
            if msg:
                msg_out = msg + [{'role': 'assistant', 'content': text}]
                fout.write(json.dumps({'messages': msg_out}, ensure_ascii=False) + '\n')
                kept += 1

print(f'Distillation data: {kept}/{total} samples passed threshold {threshold}')
"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Distillation data ready: data/sft_ready/distill_train.jsonl"
