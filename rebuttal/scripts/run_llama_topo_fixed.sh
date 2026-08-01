#!/usr/bin/env bash
# Llama-3.1-8B-Instruct GRPO with full TopoPRM reward, REQ-2 FIX applied.
#
# Root cause (rebuttal/outputs/llama_reward_diagnosis.json): the released
# TopoReward scores only text inside <think>...</think>.  Llama-3.1-8B-Instruct
# (no SFT adapter) emits plain prose, so the topology reward was 0.0 for every
# rollout and "Full TopoPRM" collapsed to outcome+format+length ~= outcome-only.
#
# Fix (all opt-in, released-checkpoint behaviour unchanged when unset):
#   TOPO_TOPO_NO_THINK_FALLBACK=1  score whole completion when no <think> block
#   TOPO_DAG_SENTENCE_FALLBACK=1   segment marker-free prose into steps
#   TOPO_DAG_EXTRA_STEP_MARKERS=1  recognise "First/Then/So/Therefore" markers
#   TOPO_DAG_FILTER_FORMATTING=1   drop LaTeX/markdown chrome from step list
#   TOPO_VAR_REF_REQUIRE_MULTI=1 + seq-overlap guards (Req-1 precision guards)
#   TOPO_RESCALE_PATCH=1           avoid stretching near-constant topo batches
set -uo pipefail
cd /Knowin/foundation/weilinruan/TopoPRM
source rebuttal/scripts/env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export WANDB_PROJECT=topoprm-rebuttal

# --- Req-2 method fix (prose-aware topology signal) ---
export TOPO_TOPO_NO_THINK_FALLBACK=1
export TOPO_DAG_SENTENCE_FALLBACK=1
export TOPO_DAG_EXTRA_STEP_MARKERS=1
export TOPO_DAG_FILTER_FORMATTING=1
export TOPO_DAG_SENTENCE_MIN_LEN=20
# --- Req-1 precision guards (clean prose edges) ---
export TOPO_VAR_REF_REQUIRE_MULTI=1
export TOPO_SEQ_WEAK_EDGE_MODE=full
export TOPO_SEQ_REQUIRE_OVERLAP=1
export TOPO_SEQ_MIN_OVERLAP=0.06
# --- keep topology from dominating on saturated easy batches ---
export TOPO_RESCALE_PATCH=1
export TOPO_RESCALE_MIN_SPAN=0.05
# surface per-component reward means every 20 calls for auditability
export TOPO_REWARD_LOG_EVERY=20

"$TOPOPRM_PY" rebuttal/scripts/train_grpo_rebuttal.py \
  --reward topo_hierarchical \
  --model /Knowin/foundation/models/meta-llama/Llama-3.1-8B-Instruct_ef \
  --sft_adapter "" \
  --output_dir output/grpo_topo_hier_llama8b_fixed \
  --max_steps 150 --num_generations 4 --max_completion_len 2048 --report_to wandb
