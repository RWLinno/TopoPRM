#!/usr/bin/env bash
# Matched GRPO on a non-Qwen base model, to give HxUk W3 a complete
# base -> +GRPO(outcome-only) -> +Full TopoPRM comparison across families.
#
# Usage: run_nonqwen_grpo.sh <reward> <gpu> <model_path> <tag>
#   reward in {outcome_only, topo_hierarchical}
set -uo pipefail
cd /Knowin/foundation/weilinruan/TopoPRM
source rebuttal/scripts/env.sh
REWARD="${1:?reward}"
export CUDA_VISIBLE_DEVICES="${2:?gpu}"
MODEL="${3:?model path}"
TAG="${4:?tag}"
export WANDB_PROJECT=topoprm-rebuttal

# Prose-CoT models (no native <think>) need the topology prose-fallback flags so
# the topology reward is not identically zero; harmless for <think> models.
if [ "$REWARD" = "topo_hierarchical" ]; then
  export TOPO_TOPO_NO_THINK_FALLBACK=1
  export TOPO_DAG_SENTENCE_FALLBACK=1
  export TOPO_DAG_EXTRA_STEP_MARKERS=1
  export TOPO_DAG_FILTER_FORMATTING=1
  export TOPO_VAR_REF_REQUIRE_MULTI=1
  export TOPO_SEQ_WEAK_EDGE_MODE=full
  export TOPO_SEQ_REQUIRE_OVERLAP=1
  export TOPO_SEQ_MIN_OVERLAP=0.06
  export TOPO_RESCALE_PATCH=1
fi

"$TOPOPRM_PY" rebuttal/scripts/train_grpo_rebuttal.py \
  --reward "$REWARD" \
  --model "$MODEL" \
  --sft_adapter "" \
  --output_dir "output/grpo_${REWARD}_${TAG}" \
  --max_steps 150 --num_generations 4 --max_completion_len 2048 --report_to wandb
