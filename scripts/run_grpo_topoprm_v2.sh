#!/usr/bin/env bash
set -euo pipefail
# Compatibility name for the paper Stage-II implementation.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/run_grpo.sh" full "$@"
