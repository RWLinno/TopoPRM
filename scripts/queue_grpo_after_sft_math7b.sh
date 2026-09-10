#!/usr/bin/env bash
# Wait for first SFT checkpoint under output/sft_qwen25_math_7b, then launch GRPO hierarchical.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${PYTHON_ENV_BIN}:$PATH"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

SFT_DIR="${SFT_DIR:-output/sft_qwen25_math_7b}"
POLL_SEC="${POLL_SEC:-120}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-86400}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

echo "[queue] waiting for checkpoint under $SFT_DIR ..."
elapsed=0
while true; do
  ckpt=$(find "$SFT_DIR" -maxdepth 3 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -1 || true)
  if [[ -n "${ckpt:-}" ]]; then
    echo "[queue] found $ckpt"
    export SFT_ADAPTER="$ckpt"
    exec bash scripts/run_grpo.sh grpo_hierarchical_qwen25_math_7b
  fi
  if (( elapsed >= MAX_WAIT_SEC )); then
    echo "[queue] timeout after ${MAX_WAIT_SEC}s" >&2
    exit 1
  fi
  sleep "$POLL_SEC"
  elapsed=$((elapsed + POLL_SEC))
done
