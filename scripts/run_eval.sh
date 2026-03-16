#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.." && export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CONFIG="${1:-configs/eval_config.yaml}"
echo "=== Running benchmark evaluation ==="
echo "Config: ${CONFIG}"

python3 -m src.eval.benchmark_runner --config "${CONFIG}"

echo "=== Evaluation complete ==="
