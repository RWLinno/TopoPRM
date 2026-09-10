#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT="${1:-8765}"
DATA_PATH="${2:-data/grpo_ready/train.jsonl}"
WATCHER_TYPE="${3:-none}"
MAX_PORT_SCAN="${4:-30}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export STREAMLIT_SERVER_FILE_WATCHER_TYPE="${WATCHER_TYPE}"

port_available() {
  local port="$1"
  python3 - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", port))
except OSError:
    print("0")
else:
    print("1")
finally:
    s.close()
PY
}

START_PORT="$PORT"
if [[ "$(port_available "$PORT")" != "1" ]]; then
  found=0
  for ((offset=1; offset<=MAX_PORT_SCAN; offset++)); do
    candidate=$((PORT + offset))
    if [[ "$(port_available "$candidate")" == "1" ]]; then
      PORT="$candidate"
      found=1
      break
    fi
  done
  if [[ "$found" -ne 1 ]]; then
    echo "Error: port ${START_PORT} is busy, and no free port found in [${START_PORT}, $((START_PORT + MAX_PORT_SCAN))]." >&2
    echo "Hint: try 'bash scripts/run_dag_gui.sh <port>' with a larger custom port." >&2
    exit 1
  fi
  echo "Port ${START_PORT} is not available. Auto-switched to ${PORT}."
fi

echo "Launching DAG viewer on port ${PORT} with data: ${DATA_PATH}"
echo "URL: http://localhost:${PORT}"

streamlit run src/gui/dag_reward_viewer.py \
  --server.port "${PORT}" \
  --server.headless true
