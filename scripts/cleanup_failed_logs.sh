#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_DIR="logs/failed_archive/${TS}"
MANIFEST="${ARCHIVE_DIR}/manifest.tsv"

mkdir -p "$ARCHIVE_DIR"

echo -e "source_path\tsize_bytes\treason" > "$MANIFEST"

# Failure signals (any match => candidate failed log)
FAIL_PATTERNS='SIGABRT|SIGTERM|Traceback|OOM|out of memory|killed|exited with code [1-9]|exitcode\s*:\s*-[0-9]+'
# Success signals (if strong success and no failure, keep)
SUCCESS_PATTERNS='End time of running main|exited with code 0|Train:\s*100%\|██████████\|'

moved=0
kept=0
checked=0

collect_logs() {
  local dir="$1"
  [ -d "$dir" ] || return 0
  find "$dir" -type f -name "*.log"
}

while IFS= read -r log; do
  checked=$((checked + 1))

  has_fail=0
  has_success=0
  reason=""

  if grep -Eiq "$FAIL_PATTERNS" "$log"; then
    has_fail=1
    reason=$(grep -Eio "$FAIL_PATTERNS" "$log" | head -n 1 || true)
    reason=${reason:-pattern_match}
  fi

  if grep -Eiq "$SUCCESS_PATTERNS" "$log"; then
    has_success=1
  fi

  # Archive+delete if failed. If both fail and success markers present, still archive when explicit failure exists.
  if [ "$has_fail" -eq 1 ]; then
    rel="${log#${ROOT_DIR}/}"
    dst="${ARCHIVE_DIR}/${rel}"
    mkdir -p "$(dirname "$dst")"
    cp "$log" "$dst"
    size=$(stat -c%s "$log" 2>/dev/null || wc -c < "$log")
    echo -e "${rel}\t${size}\t${reason}" >> "$MANIFEST"
    rm -f "$log"
    moved=$((moved + 1))
  else
    kept=$((kept + 1))
  fi

done < <(
  {
    collect_logs "output"
    collect_logs "logs"
  } | sort -u
)

echo "[cleanup_failed_logs] checked=${checked}, archived_and_deleted=${moved}, kept=${kept}"
echo "[cleanup_failed_logs] archive_dir=${ARCHIVE_DIR}"
echo "[cleanup_failed_logs] manifest=${MANIFEST}"
