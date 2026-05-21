#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
cd "$REPO_ROOT"

source /mnt/users/miniconda3/etc/profile.d/conda.sh
conda activate topoprm

export ALL_PROXY=http://accelerator-cname-hnpmnhnmdul3rmxrwhgend.c.vegalb.com:80
export HF_TOKEN=${HF_TOKEN:-""}
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ETAG_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=120

LOG_DIR="$REPO_ROOT/logs/server_B"
mkdir -p "$LOG_DIR" /mnt/data/models
LOG_FILE="$LOG_DIR/download_baseline_models_B.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# format: repo_id|target_dir_name
MODELS=(
  "Qwen/Qwen2.5-Math-1.5B-Instruct|Qwen2.5-Math-1.5B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct|Llama-3.1-8B-Instruct"
  "deepseek-ai/deepseek-math-7b-instruct|deepseek-math-7b-instruct"
  "deepseek-ai/deepseek-math-7b-rl|deepseek-math-7b-rl"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B|DeepSeek-R1-Distill-Llama-8B"
  "Qwen/Qwen3-8B|Qwen3-8B"
)

for item in "${MODELS[@]}"; do
  IFS='|' read -r REPO TARGET <<<"$item"
  DST="/mnt/data/models/$TARGET"

  if [[ -d "$DST" && -f "$DST/config.json" ]]; then
    log "SKIP $REPO already exists at $DST"
    continue
  fi

  log "START download $REPO -> $DST"

  # try ModelScope first (usually stable on this network)
  if python - <<PY >>"$LOG_FILE" 2>&1
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(model_id="$REPO", local_dir="$DST")
print("MS_OK")
PY
  then
    log "OK via ModelScope: $REPO"
    continue
  fi

  # fallback to HF mirror with retry
  ok=0
  for i in $(seq 1 20); do
    if python - <<PY >>"$LOG_FILE" 2>&1
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$REPO", local_dir="$DST", token="$HF_TOKEN", max_workers=8)
print("HF_OK")
PY
    then
      log "OK via HF mirror: $REPO"
      ok=1
      break
    fi
    log "RETRY $REPO attempt=$i failed"
    sleep $((i<12?i*5:60))
  done

  if [[ "$ok" -ne 1 ]]; then
    log "FAIL download $REPO (manual check needed; might require agree or network retry)"
  fi

done

log "ALL download queue done"
