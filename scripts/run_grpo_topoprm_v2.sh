#!/bin/bash
set -euo pipefail
###############################################################################
# TopoPRM v2 (P0 + P2 patches) — Stage-II GRPO runner
#
# This wrapper re-runs the TopoPRM-full GRPO training with the two
# aggregation fixes enabled (see the method notes):
#
#   * P0  TOPO_RESCALE_PATCH=1
#         dead-band batch rescaling when within-group topology spread is
#         below TOPO_RESCALE_MIN_SPAN (0.05). Prevents rescale noise from
#         dominating the outcome signal on saturated benchmarks
#         (GSM8K / MATH500).
#
#   * P2  TOPO_HIER_AGG=multiplicative
#         r_total = outcome * f_gate * l_gate * (1 + a*q_topo + (1-a)*q_cont)
#         outcome = 0  =>  r_total = 0. Restores the correctness-primacy
#         contract from Eq. 2 in sections/3_method.tex.
#
# Both flags default to OFF when unset, so `bash scripts/run_grpo.sh
# grpo_topoprm_full` still produces byte-level-identical rewards to the
# v3b released checkpoint. This wrapper only flips the flags ON.
#
# Usage:
#   bash scripts/run_grpo_topoprm_v2.sh                         # default config
#   bash scripts/run_grpo_topoprm_v2.sh grpo_topoprm_full       # explicit
#   CONFIG_OVERRIDE=grpo_topoprm_v2 bash scripts/run_grpo_topoprm_v2.sh
#
# Verify default preservation before running:
#   python3 scripts/check_reward_invariants.py
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG_NAME="${1:-${CONFIG_OVERRIDE:-grpo_topoprm_full}}"
shift 2>/dev/null || true

# Load all P0–P5 patch flags from the v2 env file.
V2_ENV="${PROJECT_ROOT}/configs/grpo_topoprm_v2.env"
if [[ -f "$V2_ENV" ]]; then
    set -a
    # shellcheck disable=SC1090
    source <(grep -v '^\s*#' "$V2_ENV" | grep -v '^\s*$')
    set +a
fi

# Allow per-invocation overrides (env vars set before calling this script
# take precedence over the file).
export TOPO_RESCALE_PATCH="${TOPO_RESCALE_PATCH:-1}"
export TOPO_RESCALE_MIN_SPAN="${TOPO_RESCALE_MIN_SPAN:-0.05}"
export TOPO_HIER_AGG="${TOPO_HIER_AGG:-multiplicative}"
export TOPO_CONT_REQUIRE_EVIDENCE="${TOPO_CONT_REQUIRE_EVIDENCE:-1}"
export TOPO_DAG_SENTENCE_FALLBACK="${TOPO_DAG_SENTENCE_FALLBACK:-1}"
export TOPO_DAG_SENTENCE_MIN_LEN="${TOPO_DAG_SENTENCE_MIN_LEN:-20}"
export TOPO_LENGTH_UNIT="${TOPO_LENGTH_UNIT:-tokens}"
export TOPO_LENGTH_LOW="${TOPO_LENGTH_LOW:-512}"
export TOPO_LENGTH_HIGH="${TOPO_LENGTH_HIGH:-8192}"
export TOPO_SCAE_PRESERVE_OUTCOME="${TOPO_SCAE_PRESERVE_OUTCOME:-1}"
export TOPO_SCAE_FLOOR_POS="${TOPO_SCAE_FLOOR_POS:-0.3}"
export TOPO_SCAE_FLOOR_NEG="${TOPO_SCAE_FLOOR_NEG:-0.3}"

echo "[topoprm-v2] All P0–P5 patches enabled:"
echo "  TOPO_RESCALE_PATCH         = ${TOPO_RESCALE_PATCH}"
echo "  TOPO_RESCALE_MIN_SPAN      = ${TOPO_RESCALE_MIN_SPAN}"
echo "  TOPO_HIER_AGG              = ${TOPO_HIER_AGG}"
echo "  TOPO_CONT_REQUIRE_EVIDENCE = ${TOPO_CONT_REQUIRE_EVIDENCE}"
echo "  TOPO_DAG_SENTENCE_FALLBACK = ${TOPO_DAG_SENTENCE_FALLBACK}"
echo "  TOPO_DAG_SENTENCE_MIN_LEN  = ${TOPO_DAG_SENTENCE_MIN_LEN}"
echo "  TOPO_LENGTH_UNIT           = ${TOPO_LENGTH_UNIT}"
echo "  TOPO_LENGTH_LOW            = ${TOPO_LENGTH_LOW}"
echo "  TOPO_LENGTH_HIGH           = ${TOPO_LENGTH_HIGH}"
echo "  TOPO_SCAE_PRESERVE_OUTCOME = ${TOPO_SCAE_PRESERVE_OUTCOME}"
echo "  TOPO_SCAE_FLOOR_POS        = ${TOPO_SCAE_FLOOR_POS}"
echo "  TOPO_SCAE_FLOOR_NEG        = ${TOPO_SCAE_FLOOR_NEG}"
echo "[topoprm-v2] config: ${CONFIG_NAME}"
echo

python3 "$SCRIPT_DIR/check_reward_invariants.py"
echo

exec bash "$SCRIPT_DIR/run_grpo.sh" "$CONFIG_NAME" "$@"
