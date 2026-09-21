#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
TOPOPRM_ENV_BIN="${TOPOPRM_ENV_BIN:-${PYTHON_ENV_BIN:-}}"
TOPOPRM_PYTHON="${TOPOPRM_ENV_BIN:+${TOPOPRM_ENV_BIN}/}python"
variant="${1:-full}"
if [ "$#" -gt 0 ]; then shift; fi
case "$variant" in
    full) opts=(--reward topo_hierarchical --advantage_mode ace) ;;
    without_ace) opts=(--reward topo_hierarchical --advantage_mode grpo) ;;
    outcome_only) opts=(--reward outcome_only --advantage_mode grpo) ;;
    outcome_length) opts=(--reward outcome_length --advantage_mode grpo) ;;
    no_topology) opts=(--reward no_topology --advantage_mode grpo) ;;
    no_continuity) opts=(--reward no_continuity --advantage_mode grpo --max_completion_len 1024 --max_prompt_length 3072) ;;
    -h|--help) exec "$TOPOPRM_PYTHON" "$SCRIPT_DIR/train_grpo_ablation.py" --help ;;
    *) echo "Usage: $0 {full|without_ace|outcome_only|outcome_length|no_topology|no_continuity} --sft_adapter PATH --data PATH [trainer options]" >&2; exit 2 ;;
esac
cmd=("$TOPOPRM_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-1}"
     "$SCRIPT_DIR/train_grpo_ablation.py" "${opts[@]}" "$@")
if [ "${TOPOPRM_DRY_RUN:-0}" = "1" ]; then
    printf '%q ' "${cmd[@]}"; printf '\n'; exit 0
fi
exec "${cmd[@]}"
