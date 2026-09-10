#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

while ps -ef | awk '/python3 -m pip install -r \/mnt\/users\/rwl\/topoprm\/requirements.txt/ && !/awk/ {found=1} END{exit found?0:1}'; do
  sleep 20
done

RUN_SFT=1 RUN_GRPO=1 RUN_ABLATIONS=1 RUN_AGGREGATORS=1 RUN_SCAE=1 RUN_EVAL=1 \
  bash scripts/run_all.sh > logs/run_all_heavy.log 2>&1
