#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
variant="${1:-full}"
if [ "$#" -gt 0 ]; then shift; fi
case "$variant" in
    outcome) variant=outcome_only ;;
    full|without_ace|outcome_only|outcome_length|no_topology|no_continuity|-h|--help) ;;
    *) echo "[ERROR] Variant $variant is not part of the forward hierarchical paper pipeline." >&2; exit 2 ;;
esac
exec bash "$SCRIPT_DIR/run_grpo.sh" "$variant" "$@"
