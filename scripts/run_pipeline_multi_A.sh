#!/bin/bash
set -euo pipefail
###############################################################################
# Multi-Node Pipeline: 16xA800 (2 nodes x 8 GPUs)
# Usage on MASTER node:
#   conda activate topoprm
#   MASTER_ADDR=<master_ip> NNODES=2 NODE_RANK=0 \
#     bash scripts/run_pipeline_multi_A.sh
#
# Usage on WORKER node:
#   conda activate topoprm
#   MASTER_ADDR=<master_ip> NNODES=2 NODE_RANK=1 \
#     bash scripts/run_pipeline_multi_A.sh
#
# Key env vars:
#   MASTER_ADDR   — IP of master node (required)
#   MASTER_PORT   — default 29500
#   NNODES        — default 2
#   NODE_RANK     — 0 for master, 1 for worker
#   NPROC_PER_NODE — default 8
###############################################################################
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ─── Distributed Config ───
export MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to master node IP}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NNODES="${NNODES:-2}"
export NODE_RANK="${NODE_RANK:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# ─── Auth & Proxy ───
export ALL_PROXY=http://accelerator-cname-hnpmnhnmdul3rmxrwhgend.c.vegalb.com:80
export HF_TOKEN=${HF_TOKEN}
export WANDB_API_KEY=${WANDB_API_KEY}
export WANDB_PROJECT=topoprm
export WANDB_ENTITY=rwlinno
export WANDB_MODE=offline

# ─── v2 Reward Patches ───
export TOPO_RESCALE_PATCH=1
export TOPO_RESCALE_MIN_SPAN=0.05
export TOPO_HIER_AGG=multiplicative
export TOPO_CONT_REQUIRE_EVIDENCE=1
export TOPO_DAG_SENTENCE_FALLBACK=1
export TOPO_DAG_SENTENCE_MIN_LEN=20
export TOPO_LENGTH_UNIT=tokens
export TOPO_LENGTH_LOW=512
export TOPO_LENGTH_HIGH=8192
export TOPO_SCAE_PRESERVE_OUTCOME=1

# ─── Paths ───
BASE_MODEL="/Knowin/foundation/weilinruan/hf_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
STUDENT_MODEL="/Knowin/foundation/weilinruan/hf_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET="data/grpo_ready/train_public_swift.jsonl"
SFT_ADAPTER="output/sft_deepseek_r1_7b/final"
GRPO_OUT="output/grpo_topoprm_v2_multi_A"
OPD_OUT="output/tg_opd_multi_A"
mkdir -p "$GRPO_OUT" "$OPD_OUT"

echo "==============================================================="
echo "[Multi-Node] $(date) node=$NODE_RANK/$NNODES master=$MASTER_ADDR:$MASTER_PORT"
echo "  NPROC=$NPROC_PER_NODE  GPUs=$CUDA_VISIBLE_DEVICES"
echo "==============================================================="

# ─── Data check (all nodes need data) ───
if [ ! -f "$DATASET" ]; then
  python3 -c "
import json
with open('data/grpo_ready/train_public.jsonl') as f: lines=f.readlines()
with open('$DATASET','w') as o:
  for l in lines:
    d=json.loads(l)
    o.write(json.dumps({'query':d['question'],'solution':d.get('standard_answer',''),'answer':d.get('final_answer','')},ensure_ascii=False)+'\n')
print(len(lines),'records')
"
fi

# ─── STAGE 2: GRPO (multi-node) ───
echo ""
echo "====== STAGE 2: GRPO + TopoPRM v2 (${NNODES}x${NPROC_PER_NODE} GPUs) ======"

cat > configs/grpo_topoprm_v2_multi_A.yaml << EOF
model: $BASE_MODEL
tuner_type: lora
lora_rank: 64
lora_alpha: 128
adapters:
  - $SFT_ADAPTER
rlhf_type: grpo
num_generations: 4
max_completion_length: 4096
beta: 0.04
use_vllm: false
external_plugins:
  - src/reward/composite_reward.py
reward_funcs:
  - topo_hierarchical
dataset:
  - $DATASET
max_steps: 200
learning_rate: 5.0e-6
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
save_steps: 50
output_dir: $GRPO_OUT
report_to:
  - tensorboard
logging_steps: 5
EOF

# ms-swift reads NPROC_PER_NODE, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT
# from env and auto-uses torchrun with those values
swift rlhf configs/grpo_topoprm_v2_multi_A.yaml 2>&1 | tee "$GRPO_OUT/train_node${NODE_RANK}.log"
GRPO_EXIT=$?
echo "[Stage2] exit=$GRPO_EXIT"

# Only master proceeds to Stage 3
if [ "$NODE_RANK" != "0" ]; then
    echo "[Worker node $NODE_RANK] Stage 2 done. Exiting."
    exit $GRPO_EXIT
fi

TEACHER=$(find "$GRPO_OUT" -name "adapter_config.json" -printf '%h\n' 2>/dev/null | sort -V | tail -1)
[ -z "$TEACHER" ] && TEACHER="output/grpo_topoprm_deepseek_r1_7b/final"

# ─── STAGE 3: TG-OPD (master only, or multi-node) ───
echo ""
echo "====== STAGE 3: TG-OPD (16 GPUs) ======"

cat > configs/tg_opd_multi_A.yaml << EOF
model: $STUDENT_MODEL
teacher_model: $BASE_MODEL
teacher_adapters:
  - $TEACHER
tuner_type: lora
lora_rank: 64
lora_alpha: 128
rlhf_type: gkd
use_vllm: false
dataset:
  - $DATASET
max_steps: 150
learning_rate: 3.0e-6
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
max_completion_length: 4096
beta: 0.1
save_steps: 50
output_dir: $OPD_OUT
report_to:
  - tensorboard
logging_steps: 5
EOF

swift rlhf configs/tg_opd_multi_A.yaml 2>&1 | tee "$OPD_OUT/train.log"
OPD_EXIT=$?

echo ""
echo "====== MULTI-NODE PIPELINE COMPLETE ======"
echo "  Stage2 GRPO: $GRPO_OUT exit=$GRPO_EXIT"
echo "  Stage3 OPD:  $OPD_OUT exit=$OPD_EXIT"
