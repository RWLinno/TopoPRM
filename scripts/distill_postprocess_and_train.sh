#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ "$#" -gt 0 ]; then
    echo "Configure fixed teacher, target and input paths through the documented DISTILL_* environment variables; stored training configs are not used." >&2
    exit 2
fi
exec bash "$SCRIPT_DIR/run_topology_distill.sh"
