#!/bin/bash
set -euo pipefail
###############################################################################
# GRPO fallback runner without vLLM.
#
# Default usage (use GPU 2-7):
#   bash scripts/train_wo_vllm.sh grpo_main
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_NAME="${1:-grpo_main}"
shift || true
BASE_CONFIG="configs/${CONFIG_NAME}.yaml"

if [ ! -f "$BASE_CONFIG" ]; then
  echo "[ERROR] Config not found: $BASE_CONFIG"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-6}"

TMP_NAME="${CONFIG_NAME}_wo_vllm_tmp_$$_$(date +%s)"
TMP_CONFIG="configs/${TMP_NAME}.yaml"

python3 - <<'PY' "$BASE_CONFIG" "$TMP_CONFIG"
from pathlib import Path
import sys
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
lines = src.read_text(encoding='utf-8').splitlines()
out = []
for line in lines:
    key = line.split(':', 1)[0].strip()
    if key in {
        'use_vllm',
        'vllm_mode',
        'vllm_gpu_memory_utilization',
        'vllm_max_model_len',
        'vllm_enable_prefix_caching',
        'sleep_level',
        'offload_model',
        'offload_optimizer',
    }:
        continue
    out.append(line)

inserted = False
for i, line in enumerate(out):
    if line.strip().startswith('top_p:'):
        out.insert(i + 1, 'use_vllm: false')
        inserted = True
        break
if not inserted:
    out.append('use_vllm: false')

dst.write_text('\n'.join(out) + '\n', encoding='utf-8')
print(f'[train_wo_vllm] temp config -> {dst}')
PY

cleanup() {
  rm -f "$TMP_CONFIG"
}
trap cleanup EXIT

echo "[train_wo_vllm] Launching without vLLM on GPUs ${CUDA_VISIBLE_DEVICES}"
SFT_ADAPTER="${SFT_ADAPTER:-}" bash "$SCRIPT_DIR/run_grpo.sh" "$TMP_NAME" "$@"
