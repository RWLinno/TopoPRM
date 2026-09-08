#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?Usage: $0 <full|outcome|outcome_length|laser_d|hero|sarl|equal|additive|multiplicative|no_dir_acyc_interaction|no_direction|no_acyclicity|no_raw_support|rules_only|no_continuity>}"
shift

if [ -n "${TOPO_FORMAT_PROTOCOL:-}" ] && [ "$TOPO_FORMAT_PROTOCOL" != "prefilled_think" ]; then
  echo "[ERROR] Canonical Qwen3.5 runs require TOPO_FORMAT_PROTOCOL=prefilled_think" >&2
  exit 4
fi
export TOPO_FORMAT_PROTOCOL=prefilled_think

unset TOPO_LAMBDA_DELTA TOPO_LAMBDA_ACYCLIC TOPO_DISABLE_EDGE_ENCODER
unset TOPO_DISABLE_DIRECTION_SOURCE TOPO_DISABLE_ACYCLICITY_SOURCE
unset TOPO_DISABLE_CONTINUITY_SOURCE
unset TOPO_DISABLE_DIRECTION_ACYCLICITY_INTERACTION

case "$VARIANT" in
  full)
    GRPO_REWARD_FUNC=topo_independent_choquet
    ;;
  outcome)
    GRPO_REWARD_FUNC=topo_outcome
    ;;
  outcome_length)
    GRPO_REWARD_FUNC=outcome_format_length
    ;;
  laser_d)
    GRPO_REWARD_FUNC=matched_laser_d
    ;;
  hero)
    GRPO_REWARD_FUNC=topo_independent_hero
    ;;
  sarl)
    GRPO_REWARD_FUNC=sarl_structure
    export TOPO_DISABLE_EDGE_ENCODER=1
    export SARL_EMBED_MODEL="${SARL_EMBED_MODEL:-/knowin-oss/weilinruan/models/Qwen3-Embedding-0.6B}"
    export SARL_CLUSTER_METHOD="${SARL_CLUSTER_METHOD:-hdbscan}"
    export SARL_EMBED_MAX_LENGTH="${SARL_EMBED_MAX_LENGTH:-4096}"
    export SARL_EMBED_BATCH_SIZE="${SARL_EMBED_BATCH_SIZE:-8}"
    export SARL_EMBED_DTYPE="${SARL_EMBED_DTYPE:-bfloat16}"
    export SARL_REFERENCE_COMMIT="${SARL_REFERENCE_COMMIT:-9fb8d397fe2c4f06e169b30d95730520fef392ce}"
    ;;
  equal)
    GRPO_REWARD_FUNC=topo_independent_equal_additive
    ;;
  additive)
    GRPO_REWARD_FUNC=topo_independent_matched_additive
    ;;
  multiplicative)
    GRPO_REWARD_FUNC=topo_independent_matched_multiplicative
    ;;
  no_dir_acyc_interaction)
    GRPO_REWARD_FUNC=topo_independent_choquet
    export TOPO_DISABLE_DIRECTION_ACYCLICITY_INTERACTION=1
    ;;
  no_direction)
    GRPO_REWARD_FUNC=topo_independent_choquet
    export TOPO_LAMBDA_DELTA=0
    export TOPO_DISABLE_DIRECTION_SOURCE=1
    ;;
  no_acyclicity)
    GRPO_REWARD_FUNC=topo_independent_choquet
    export TOPO_LAMBDA_ACYCLIC=0
    export TOPO_DISABLE_ACYCLICITY_SOURCE=1
    ;;
  no_raw_support|projected_only)
    if [ "$VARIANT" = "projected_only" ]; then
      echo "[WARN] projected_only is a legacy alias; use no_raw_support." >&2
    fi
    GRPO_REWARD_FUNC=topo_independent_choquet
    export TOPO_DISABLE_DIRECTION_SOURCE=1
    export TOPO_DISABLE_ACYCLICITY_SOURCE=1
    ;;
  rules_only)
    GRPO_REWARD_FUNC=topo_independent_choquet
    export TOPO_DISABLE_EDGE_ENCODER=1
    ;;
  no_continuity)
    GRPO_REWARD_FUNC=topo_independent_choquet
    export TOPO_DISABLE_CONTINUITY_SOURCE=1
    ;;
  *)
    echo "[ERROR] Unknown Stage-II variant: $VARIANT"
    exit 2
    ;;
esac

export GRPO_REWARD_FUNC
case "$GRPO_REWARD_FUNC" in
  topo_independent_*)
    case "${TOPO_CHOQUET_CAPACITY_PROFILE:-}" in
      balanced|structure_forward) ;;
      *)
        echo "[ERROR] Set TOPO_CHOQUET_CAPACITY_PROFILE to an author-approved profile: balanced or structure_forward" >&2
        exit 4
        ;;
    esac
    ;;
esac
OUTPUT_TAG="$VARIANT"
if [ "$VARIANT" = "projected_only" ]; then
  OUTPUT_TAG=no_raw_support
fi
case "$GRPO_REWARD_FUNC" in
  topo_independent_*)
    OUTPUT_TAG="${VARIANT}_${TOPO_CHOQUET_CAPACITY_PROFILE}"
    ;;
esac
# Direct safetensors writes to /knowin-oss reproducibly produced zero-filled
# files. Train on the writable demo filesystem, then archive verified artifacts.
export GRPO_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-/Knowin/demo/weilinruan/TopoPRM_ICLR27/canonical/grpo_${OUTPUT_TAG}_qwen35_9b}"
export GUARD_GPU_LEAK_MB="${GUARD_GPU_LEAK_MB:-4096}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export AUTO_RESOLVE_SFT_ADAPTER=0
export TOPOPRM_ENV_BIN="${TOPOPRM_ENV_BIN:-/Knowin/foundation/weilinruan/env/qwen35/bin}"

STAGE1_ROOT="/knowin-oss/weilinruan/TopoPRM_ICLR27/canonical/sft_qwen35_9b"
if [ -z "${SFT_ADAPTER:-}" ]; then
  LATEST_STAGE1_STATE="$(find "$STAGE1_ROOT" -maxdepth 2 -path '*/checkpoint-*/trainer_state.json' -print 2>/dev/null | sort -V | tail -1)"
  if [ -z "$LATEST_STAGE1_STATE" ]; then
    echo "[ERROR] No Stage-I trainer state found under $STAGE1_ROOT" >&2
    exit 3
  fi
  read -r SFT_ADAPTER STAGE1_STEP STAGE1_MAX_STEP < <(
    /Knowin/foundation/weilinruan/env/qwen35/bin/python -c '
import json, os, sys
state_path = sys.argv[1]
state = json.load(open(state_path, encoding="utf-8"))
best = state.get("best_model_checkpoint")
if not best:
    evaluated = [
        entry for entry in state.get("log_history", [])
        if entry.get("eval_loss") is not None and entry.get("step") is not None
    ]
    if evaluated:
        best_step = min(evaluated, key=lambda entry: float(entry["eval_loss"]))["step"]
        best = os.path.join(os.path.dirname(os.path.dirname(state_path)), f"checkpoint-{best_step}")
print(best or "", state.get("global_step", -1), state.get("max_steps", -1))
' "$LATEST_STAGE1_STATE"
  )
  if [ "$STAGE1_STEP" -ne "$STAGE1_MAX_STEP" ]; then
    echo "[ERROR] Stage-I is incomplete: step $STAGE1_STEP/$STAGE1_MAX_STEP" >&2
    exit 3
  fi
fi
STAGE1_ROOT_REAL="$(realpath -e "$STAGE1_ROOT")"
SFT_ADAPTER_REAL="$(realpath -e "$SFT_ADAPTER" 2>/dev/null || true)"
case "$SFT_ADAPTER_REAL" in
  "$STAGE1_ROOT_REAL"/checkpoint-*) ;;
  *)
    echo "[ERROR] Stage-I adapter is outside the canonical run: $SFT_ADAPTER" >&2
    exit 3
    ;;
esac
SFT_ADAPTER="$SFT_ADAPTER_REAL"
if [ ! -f "$SFT_ADAPTER/adapter_config.json" ] || \
   { [ ! -f "$SFT_ADAPTER/adapter_model.safetensors" ] && [ ! -f "$SFT_ADAPTER/adapter_model.bin" ]; }; then
  echo "[ERROR] Invalid Stage-I adapter: $SFT_ADAPTER" >&2
  exit 3
fi
if [ -f "$SFT_ADAPTER/adapter_model.safetensors" ]; then
  if ! "$TOPOPRM_ENV_BIN/python" -c '
from safetensors import safe_open
import sys

with safe_open(sys.argv[1], framework="pt", device="cpu") as handle:
    if not list(handle.keys()):
        raise ValueError("adapter contains no tensors")
' "$SFT_ADAPTER/adapter_model.safetensors"; then
    echo "[ERROR] Unreadable Stage-I safetensors adapter: $SFT_ADAPTER" >&2
    exit 3
  fi
fi
export SFT_ADAPTER
echo "[stage2] Locked Stage-I adapter: $SFT_ADAPTER"

if [ "${STAGE2_DRY_RUN:-0}" = "1" ]; then
  echo "variant=$VARIANT"
  echo "reward=$GRPO_REWARD_FUNC"
  echo "capacity_profile=${TOPO_CHOQUET_CAPACITY_PROFILE:-unset}"
  echo "format_protocol=$TOPO_FORMAT_PROTOCOL"
  echo "output=$GRPO_OUTPUT_DIR"
  echo "adapter=$SFT_ADAPTER"
  echo "lambda_direction=${TOPO_LAMBDA_DELTA:-default}"
  echo "lambda_acyclicity=${TOPO_LAMBDA_ACYCLIC:-default}"
  echo "disable_edge_encoder=${TOPO_DISABLE_EDGE_ENCODER:-0}"
  if [ "${TOPO_DISABLE_EDGE_ENCODER:-0}" = "1" ]; then
    echo "edge_encoder_required=0 (applied after canonical env is sourced)"
  fi
  echo "disable_direction=${TOPO_DISABLE_DIRECTION_SOURCE:-0}"
  echo "disable_acyclicity=${TOPO_DISABLE_ACYCLICITY_SOURCE:-0}"
  echo "disable_continuity=${TOPO_DISABLE_CONTINUITY_SOURCE:-0}"
  echo "disable_direction_acyclicity_interaction=${TOPO_DISABLE_DIRECTION_ACYCLICITY_INTERACTION:-0}"
  echo "sarl_embed_model=${SARL_EMBED_MODEL:-unused}"
  echo "sarl_cluster_method=${SARL_CLUSTER_METHOD:-unused}"
  echo "sarl_embed_max_length=${SARL_EMBED_MAX_LENGTH:-unused}"
  echo "sarl_embed_batch_size=${SARL_EMBED_BATCH_SIZE:-unused}"
  echo "sarl_embed_dtype=${SARL_EMBED_DTYPE:-unused}"
  echo "sarl_reference_commit=${SARL_REFERENCE_COMMIT:-unused}"
  echo "env_bin=$TOPOPRM_ENV_BIN"
  printf 'forwarded_swift_args='
  printf ' %q' "$@"
  printf '\n'
  exit 0
fi

exec bash scripts/run_grpo.sh grpo_topoprm_iclr27 "$@"
