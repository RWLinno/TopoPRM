#!/bin/bash
set -euo pipefail

OUTPUT_DIR="${1:-data/benchmarks}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=== Downloading benchmark datasets (full list) ==="
python3 src/data/download_benchmarks.py \
  --output_dir "$OUTPUT_DIR" \
  --manifest "$OUTPUT_DIR/manifest.json" \
  --status_md "$OUTPUT_DIR/status.md"

echo "=== Done. See $OUTPUT_DIR/manifest.json and $OUTPUT_DIR/status.md ==="
