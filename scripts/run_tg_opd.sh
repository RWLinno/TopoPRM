#!/bin/bash
set -euo pipefail
###############################################################################
# TopoPRM Stage III: Topology-Guided On-Policy Distillation (TG-OPD)
#
# This script launches the OPSD trainer via ms-swift's `gkd` rlhf_type,
# using the Stage-II TopoPRM teacher checkpoint to supervise a compact
# student through token-level reverse-KL with topology-conditioned
# revision prompts.
#
# Prerequisites:
#   - Stage-II teacher checkpoint exists (output/grpo_topoprm_full/final
#     or output/grpo_topoprm_deepseek_r1_7b/final)
#   - Student base model accessible (e.g. Qwen2.5-4B or DR1-Distill-1.5B)
#   - data/grpo_ready/train_public.jsonl exists
#
# Usage:
#   bash scripts/run_tg_opd.sh                          # default: opsd_dr1_7b
#   bash scripts/run_tg_opd.sh opsd_student_4b          # 4B student
#   bash scripts/run_tg_opd.sh opsd_student_2b          # 2B student
#   TEACHER_ADAPTER=output/grpo_topoprm_full/final \
#     bash scripts/run_tg_opd.sh opsd_student_4b
#
# Env vars:
#   CUDA_VISIBLE_DEVICES   (default: 0,1,2,3,4,5,6,7)
#   NPROC_PER_NODE         (default: 8)
#   TEACHER_ADAPTER        (default: auto-detect from output/)
#   OPSD_MAX_STEPS         (override max_steps in config)
#   TOPO_RESCALE_PATCH     (default: 1 — enable P0 dead-band rescaling)
#   TOPO_HIER_AGG          (default: multiplicative — enable P2)
#   TOPO_CONT_REQUIRE_EVIDENCE  (default: 1 — enable P3)
#   TOPO_DAG_SENTENCE_FALLBACK  (default: 1 — enable P4)
#   TOPO_LENGTH_UNIT       (default: tokens — enable P5)
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CONFIG_NAME="${1:-opsd_dr1_7b}"
shift 2>/dev/null || true

# ---- Auto-detect teacher adapter if not set ----
if [ -z "${TEACHER_ADAPTER:-}" ]; then
    for cand in \
        output/grpo_topoprm_full/final \
        output/grpo_topoprm_deepseek_r1_7b/final \
        output/grpo_hierarchical/final \
        output/grpo_main/final; do
        if [ -d "$cand" ]; then
            TEACHER_ADAPTER="$cand"
            break
        fi
    done
fi
export TEACHER_ADAPTER="${TEACHER_ADAPTER:-}"

# ---- Enable all v2 patches by default for Stage III ----
export TOPO_RESCALE_PATCH="${TOPO_RESCALE_PATCH:-1}"
export TOPO_HIER_AGG="${TOPO_HIER_AGG:-multiplicative}"
export TOPO_CONT_REQUIRE_EVIDENCE="${TOPO_CONT_REQUIRE_EVIDENCE:-1}"
export TOPO_DAG_SENTENCE_FALLBACK="${TOPO_DAG_SENTENCE_FALLBACK:-1}"
export TOPO_LENGTH_UNIT="${TOPO_LENGTH_UNIT:-tokens}"

# ---- Build extra args ----
EXTRA_ARGS=()
if [ -n "${OPSD_MAX_STEPS:-}" ]; then
    EXTRA_ARGS+=(--max_steps "$OPSD_MAX_STEPS")
fi

echo "══════════════════════════════════════════"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] TG-OPD (Stage III) launch"
echo "  config:              $CONFIG_NAME"
echo "  teacher adapter:     ${TEACHER_ADAPTER:-<none, using base>}"
echo "  TOPO_RESCALE_PATCH:  $TOPO_RESCALE_PATCH"
echo "  TOPO_HIER_AGG:       $TOPO_HIER_AGG"
echo "  TOPO_CONT_REQUIRE_EVIDENCE: $TOPO_CONT_REQUIRE_EVIDENCE"
echo "  TOPO_DAG_SENTENCE_FALLBACK: $TOPO_DAG_SENTENCE_FALLBACK"
echo "  TOPO_LENGTH_UNIT:    $TOPO_LENGTH_UNIT"
echo "══════════════════════════════════════════"

# Delegate to the generic swift RLHF launcher with rlhf_type=gkd
exec bash "$SCRIPT_DIR/run_swift_rlhf.sh" "$CONFIG_NAME" gkd "${EXTRA_ARGS[@]}" "$@"
