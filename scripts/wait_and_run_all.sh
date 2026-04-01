#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

while ps -ef | awk '/python3 -m pip install -r \/mnt\/users\/rwl\/topoprm\/requirements.txt/ && !/awk/ {found=1} END{exit found?0:1}'; do
  sleep 20
done

bash scripts/run_all.sh > logs/run_all_after_install.log 2>&1
