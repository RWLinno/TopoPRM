#!/bin/bash
set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen3-14B}"
CACHE_DIR="${2:-models/}"

echo "=== Downloading model: ${MODEL_ID} ==="
echo "Cache directory: ${CACHE_DIR}"

mkdir -p "${CACHE_DIR}"

python3 -c "
from modelscope import snapshot_download
import sys

model_id = '${MODEL_ID}'
cache_dir = '${CACHE_DIR}'

print(f'Downloading {model_id} to {cache_dir} ...')
path = snapshot_download(model_id, cache_dir=cache_dir)
print(f'Model saved to: {path}')
"

echo "=== Download complete ==="
