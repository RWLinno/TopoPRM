#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT="${1:-8765}"
DATA_PATH="${2:-data/grpo_ready/train.jsonl}"

echo "Launching DAG viewer on port ${PORT} with data: ${DATA_PATH}"
streamlit run src/gui/dag_reward_viewer.py --server.port "${PORT}" --server.headless true
