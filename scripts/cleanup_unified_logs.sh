#!/usr/bin/env bash
# Safely clean failed / interrupted evaluation logs and stale artifacts.
#
# Policy:
#   - REMOVE: orchestrator *.log/*.out + per-bench *.log older than RETAIN_DAYS
#     (default 7) that have no matching metrics.json, i.e. never completed.
#   - REMOVE: per-bench *_details.jsonl whose matching metrics.json is missing
#     (evidence of a mid-run kill).
#   - PRESERVE: any metrics.json, and any *.log whose matching metrics.json
#     exists (so paper-quality runs stay auditable).
#
# Usage:
#   bash scripts/cleanup_unified_logs.sh            # dry-run
#   CONFIRM=1 bash scripts/cleanup_unified_logs.sh  # actually delete
#
set -euo pipefail
cd "$(dirname "$0")/.."

RETAIN_DAYS="${RETAIN_DAYS:-7}"
DRY="[dry-run]"
[[ "${CONFIRM:-0}" == "1" ]] && DRY=""

echo "Scanning logs/unified + output/eval (retain_days=$RETAIN_DAYS)..."
echo ""

# 1. per-bench log with missing metrics.json
declare -a RM_LOGS
for lf in logs/unified/*_*.log; do
    [[ -f "$lf" ]] || continue
    base="$(basename "$lf" .log)"
    # skip orchestrator logs
    [[ "$base" == orchestrator_* ]] && continue
    mp="output/eval/${base}_metrics.json"
    alt="output/eval/${base}.json"
    if [[ ! -f "$mp" && ! -f "$alt" ]]; then
        age_days=$(( ( $(date +%s) - $(stat -c %Y "$lf") ) / 86400 ))
        if (( age_days >= RETAIN_DAYS )); then
            RM_LOGS+=("$lf")
        fi
    fi
done

# 2. orchestrator logs older than retain_days with no matching status dashboard update
declare -a RM_ORCH
for lf in logs/unified/orchestrator_*.{log,out}; do
    [[ -f "$lf" ]] || continue
    age_days=$(( ( $(date +%s) - $(stat -c %Y "$lf") ) / 86400 ))
    if (( age_days >= RETAIN_DAYS )); then
        RM_ORCH+=("$lf")
    fi
done

# 3. orphan details.jsonl
declare -a RM_JSONL
for jl in output/eval/*_details.jsonl; do
    [[ -f "$jl" ]] || continue
    mp="${jl%_details.jsonl}_metrics.json"
    [[ -f "$mp" ]] || RM_JSONL+=("$jl")
done

set +u
nlogs=${#RM_LOGS[@]}
norch=${#RM_ORCH[@]}
njsonl=${#RM_JSONL[@]}
set -u
total=$(( nlogs + norch + njsonl ))
echo "==> Candidates for removal: $total"
(( nlogs  > 0 )) && for f in "${RM_LOGS[@]}";  do echo "  $DRY LOG     $f"; done
(( norch  > 0 )) && for f in "${RM_ORCH[@]}";  do echo "  $DRY ORCHLOG $f"; done
(( njsonl > 0 )) && for f in "${RM_JSONL[@]}"; do echo "  $DRY JSONL   $f"; done

if [[ -z "$DRY" ]]; then
    (( nlogs  > 0 )) && for f in "${RM_LOGS[@]}";  do rm -f "$f"; done
    (( norch  > 0 )) && for f in "${RM_ORCH[@]}";  do rm -f "$f"; done
    (( njsonl > 0 )) && for f in "${RM_JSONL[@]}"; do rm -f "$f"; done
    echo "[done] removed $total files."
else
    echo ""
    echo "Dry-run only. Rerun with CONFIRM=1 to delete."
fi
