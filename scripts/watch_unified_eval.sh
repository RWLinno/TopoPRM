#!/usr/bin/env bash
# Monitor unified evaluation for a given LABEL.
# Usage: bash scripts/watch_unified_eval.sh <LABEL>
set -euo pipefail
LABEL="${1:?Usage: watch_unified_eval.sh <LABEL>}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${TOPOPRM_PYTHON:-/Knowin/foundation/weilinruan/env/topoprm/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3)"

STATUS="logs/unified/status_${LABEL}.json"

render() {
    clear 2>/dev/null || true
    echo "============================================================"
    echo " Label: $LABEL   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    "$PYTHON_BIN" - "$LABEL" "$STATUS" <<'PY'
import json, os, sys
from pathlib import Path

label, status_path = sys.argv[1], sys.argv[2]
status_path = Path(status_path)
benches = [
    "gsm8k","math500","aime2024","aime2025","cnmo2024",
    "olympiadbench","omni_math","gpqa_diamond","mmlu",
]

if status_path.exists():
    try:
        d = json.loads(status_path.read_text())
    except Exception as e:
        d = {}
        print(f"(status parse error: {e})")
    else:
        print(f"model:   {d.get('model','?')}")
        ad = d.get('adapter') or '<none>'
        print(f"adapter: {ad}")
        print("--- running ---")
        for gpu, info in (d.get("running") or {}).items():
            print(f"  GPU {gpu}: {info['bench']} (attempt {info['attempt']})")
        print("--- completed (orchestrator view) ---")
        for r in (d.get("completed") or []):
            p1 = r.get("pass@1")
            ps = f"{p1*100:5.1f}%" if isinstance(p1, (int, float)) else "  --  "
            print(f"  {r['bench']:<15} pass@1={ps}  {r['duration_min']:>6.1f}min  rc={r['exit_code']}  attempt={r.get('attempt',1)}")
else:
    print(f"(no orchestrator status yet at {status_path})")

print()
print("--- per-bench metrics.json ---")
for b in benches:
    mp = Path("output/eval") / f"{label}_{b}_metrics.json"
    if not mp.exists():
        print(f"  {b:<15} : missing")
        continue
    try:
        m = json.loads(mp.read_text())
        p1 = m.get("pass@1") or m.get("accuracy") or 0.0
        tok = m.get("avg_tokens", 0) or 0
        el = m.get("elapsed_sec", 0) or 0
        print(f"  {b:<15} : pass@1={p1*100:5.1f}%   tok={float(tok):.0f}   elapsed={float(el):.0f}s")
    except Exception as e:
        print(f"  {b:<15} : parse error: {e}")
PY
    echo ""
    echo "--- latest orchestrator log tail ---"
    BG_LOG="$(ls -t logs/unified/orchestrator_${LABEL}_*.log logs/unified/orchestrator_${LABEL}_*.out 2>/dev/null | head -1 || true)"
    if [[ -n "$BG_LOG" && -f "$BG_LOG" ]]; then
        echo "[$BG_LOG]"
        tail -15 "$BG_LOG"
    else
        echo "(no orchestrator log)"
    fi
    echo ""
    echo "(Ctrl-C to exit; refreshing every ${INTERVAL:-10}s)"
}

INTERVAL="${INTERVAL:-10}"
if [[ "${ONCE:-0}" == "1" ]]; then
    render
    exit 0
fi
while true; do
    render
    sleep "$INTERVAL"
done
