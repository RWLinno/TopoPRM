#!/bin/bash
set -euo pipefail
###############################################################################
# Server A Full 3-Stage Pipeline: SFT -> GRPO+TopoPRM v2 -> TG-OPD
# Usage: conda activate topoprm && bash scripts/run_pipeline_A.sh
###############################################################################
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ─── Config ───
BASE_MODEL="${MODEL_ROOT}/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
STUDENT_MODEL="${MODEL_ROOT}/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET="data/grpo_ready/train_public_swift.jsonl"

# Optional egress proxy for clusters without direct internet access.
export ALL_PROXY=${ALL_PROXY:-}
export HF_TOKEN=${HF_TOKEN}
export WANDB_API_KEY=${WANDB_API_KEY}
export WANDB_PROJECT=topoprm
export WANDB_ENTITY=${WANDB_ENTITY:-anonymous}
export WANDB_MODE=online

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
export MASTER_PORT="${MASTER_PORT:-29501}"

# v2 reward patches (all ON)
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
export TOPO_SCAE_FLOOR_POS=0.3
export TOPO_SCAE_FLOOR_NEG=0.3

SFT_OUT="output/sft_dr1_7b_A"
GRPO_OUT="output/grpo_topoprm_v2_A"
OPD_OUT="output/tg_opd_A"
mkdir -p "$SFT_OUT" "$GRPO_OUT" "$OPD_OUT" results/method_v2

TRACE="results/method_v2/train_eval_trace_A_$(date +%Y-%m-%d).md"
echo "# Pipeline Trace $(date)" > "$TRACE"

log() { echo "[$(date '+%H:%M:%S')] $*"; echo "| $(date '+%H:%M') | $* |" >> "$TRACE"; }

# ─── Branch guard ───
BRANCH=$(git branch --show-current 2>/dev/null || echo "?")
[[ "$BRANCH" != "exp_May14" ]] && echo "[FATAL] branch=$BRANCH" && exit 1

# ─── Reward invariants ───
log "Reward invariant check"
python3 scripts/check_reward_invariants.py || { log "FAIL invariants"; exit 1; }

# ─── Data prep ───
if [ ! -f "$DATASET" ]; then
  log "Converting dataset"
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

# ═══════ STAGE 1: SFT ═══════
echo ""; echo "====== STAGE 1: SFT ======"
SFT_ADAPTER=""
for cand in output/sft_deepseek_r1_7b/final "$SFT_OUT/final" "$SFT_OUT"/checkpoint-*; do
  [ -f "$cand/adapter_config.json" ] && SFT_ADAPTER="$cand" && break
done

if [ -n "$SFT_ADAPTER" ]; then
  log "Stage1 SKIP (adapter=$SFT_ADAPTER)"
else
  log "Stage1 START SFT"
  swift sft \
    --model "$BASE_MODEL" --tuner_type lora --lora_rank 64 --lora_alpha 128 \
    --dataset "$DATASET" --max_length 4096 --num_train_epochs 2 \
    --learning_rate 5e-5 --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 --bf16 true --gradient_checkpointing true \
    --save_steps 100 --save_total_limit 2 --report_to tensorboard \
    --logging_steps 10 --output_dir "$SFT_OUT" 2>&1 | tee "$SFT_OUT/train.log"
  SFT_ADAPTER=$(find "$SFT_OUT" -name "adapter_config.json" -printf '%h\n' 2>/dev/null | sort -V | tail -1)
  [ -z "$SFT_ADAPTER" ] && log "Stage1 FAIL" && exit 1
  log "Stage1 DONE adapter=$SFT_ADAPTER"
fi
echo "SFT_ADAPTER=$SFT_ADAPTER"

# ═══════ STAGE 2: GRPO + TopoPRM ═══════
echo ""; echo "====== STAGE 2: GRPO + TopoPRM v2 ======"
log "Stage2 START GRPO+TopoPRM"

cat > configs/grpo_topoprm_v2_A.yaml << EOF
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
gradient_accumulation_steps: 4
save_steps: 50
output_dir: $GRPO_OUT
report_to:
  - tensorboard
logging_steps: 5
EOF

swift rlhf configs/grpo_topoprm_v2_A.yaml 2>&1 | tee "$GRPO_OUT/train.log"
GRPO_EXIT=$?
if [ $GRPO_EXIT -ne 0 ]; then
  log "Stage2 FAIL (exit=$GRPO_EXIT) retrying..."
  sleep 10
  swift rlhf configs/grpo_topoprm_v2_A.yaml 2>&1 | tee -a "$GRPO_OUT/train.log"
  GRPO_EXIT=$?
fi
[ $GRPO_EXIT -ne 0 ] && log "Stage2 FAIL after retry" || log "Stage2 DONE"

TEACHER=$(find "$GRPO_OUT" -name "adapter_config.json" -printf '%h\n' 2>/dev/null | sort -V | tail -1)
[ -z "$TEACHER" ] && TEACHER="output/grpo_topoprm_deepseek_r1_7b/final"
echo "TEACHER=$TEACHER"

# ═══════ STAGE 3: TG-OPD ═══════
echo ""; echo "====== STAGE 3: TG-OPD ======"
log "Stage3 START TG-OPD teacher=$TEACHER"

cat > configs/tg_opd_A.yaml << EOF
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
gradient_accumulation_steps: 8
max_completion_length: 4096
beta: 0.1
save_steps: 50
output_dir: $OPD_OUT
report_to:
  - tensorboard
logging_steps: 5
EOF

swift rlhf configs/tg_opd_A.yaml 2>&1 | tee "$OPD_OUT/train.log"
OPD_EXIT=$?
[ $OPD_EXIT -ne 0 ] && log "Stage3 FAIL exit=$OPD_EXIT" || log "Stage3 DONE"

echo ""
echo "====== PIPELINE COMPLETE ======"
echo "  Stage1 SFT:  $SFT_ADAPTER"
echo "  Stage2 GRPO: $GRPO_OUT (exit=$GRPO_EXIT)"
echo "  Stage3 OPD:  $OPD_OUT (exit=$OPD_EXIT)"
echo "  Trace:       $TRACE"
