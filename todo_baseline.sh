#!/usr/bin/env bash
# ============================================================================
# todo_baseline.sh ? Server B baseline orchestrator (TopoPRM, exp_May14)
# ----------------------------------------------------------------------------
# ??? baseline ??????SFT/GRPO/DAPO/DPO ??????? +TopoPRM ?
# TGSD-Distilled??? server_B ??/????? _B ???????
# server_A ? results/method_v2/* ???
#
# ??:
#   bash todo_baseline.sh prepare         # 1. ?????????????
#   bash todo_baseline.sh eval            # 2. ???? baseline ??
#   bash todo_baseline.sh aggregate       # 3. ?? metrics ? results/baseline/
#   bash todo_baseline.sh sync            # 4. ????? manifest
#   bash todo_baseline.sh all             # ???? prepare ? eval ? aggregate ? sync
#
# ????:
#   * GPU ???: ????? GPU_POOL ?? (?? 0,1,2,3,4,5,6,7)
#   * ?????? 1 ? (orchestrator ??)
#   * ????: ??? metrics.json ? (label, bench) ????
#   * ??? (label, bench) ???? results/baseline/logs_B/
#
# ??:
#   conda activate topoprm
#   export ALL_PROXY=http://accelerator-cname-hnpmnhnmdul3rmxrwhgend.c.vegalb.com:80
#   export HF_TOKEN=hf_xxx
#   export WANDB_API_KEY=wandb_v1_xxx
# ============================================================================

set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$REPO_ROOT"

GPU_POOL="${GPU_POOL:-0,1,2,3,4,5,6,7}"
BASELINE_OUT="$REPO_ROOT/results/baseline"
BASELINE_LOGS="$BASELINE_OUT/logs_B"
EVAL_OUT="$REPO_ROOT/output/eval"
TODAY="$(date +%Y%m%d)"
SERVER_TAG="_B"

# ????? model + adapter ???Server B ? baseline ????
# ??: "label|model_path|adapter_path|use_chat_template|sft_style|num_samples"
# adapter ??????????????? server_B ?????
BASELINE_TARGETS=(
  # ---- Reference baselines?????????? ----
  "ref_qwen25_7b_instruct${SERVER_TAG}|/mnt/data/huggingface_downloads/models/qwen/Qwen2.5-7B-Instruct||true|false|5"
  "ref_qwen35_9b_base${SERVER_TAG}|/mnt/data/huggingface_downloads/models/qwen/Qwen3.5-9B-Base||false|false|5"
  "ref_qwen35_9b${SERVER_TAG}|/mnt/data/huggingface_downloads/models/qwen/Qwen3.5-9B||true|false|5"

  # ---- Qwen3.5-9B family: SFT / GRPO baselines??????? adapter? ----
  "sft_qwen35_9b${SERVER_TAG}|/mnt/data/huggingface_downloads/models/qwen/Qwen3.5-9B|__AUTO_SFT_QWEN35_9B__|true|true|5"
  "grpo_outcome_only_qwen35_9b${SERVER_TAG}|/mnt/data/huggingface_downloads/models/qwen/Qwen3.5-9B|__AUTO_GRPO_OUTCOME_ONLY_9B__|true|false|5"
)

LOG_FILE="$BASELINE_LOGS/todo_baseline_${TODAY}${SERVER_TAG}.log"

log() {
  local ts msg
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  msg="[$ts] $*"
  printf '%s\n' "$msg" | tee -a "$LOG_FILE" >&2
}

resolve_adapter() {
  local marker="$1"
  case "$marker" in
    "")
      printf ''
      ;;
    __AUTO_SFT_QWEN35_9B__)
      ls -dt output/sft_qwen35_9b/*/checkpoint-* 2>/dev/null | head -1
      ;;
    __AUTO_GRPO_OUTCOME_ONLY_9B__)
      ls -dt output/grpo_outcome_only_qwen35_9b/*/checkpoint-* 2>/dev/null | head -1
      ;;
    *)
      printf '%s' "$marker"
      ;;
  esac
}

# --------------------------------------------------------------------------
# Phase 1: prepare ? env / ?? / ?? sanity check
# --------------------------------------------------------------------------
phase_prepare() {
  log "=== prepare: env & assets sanity check ==="

  if ! command -v python &>/dev/null; then
    log "ERROR: python not found in PATH (activate topoprm env first)."
    return 1
  fi
  log "python: $(python -V 2>&1)"

  python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import importlib, sys
for mod in ("torch", "transformers", "swift", "peft"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod}: {getattr(m, '__version__', 'unknown')}")
    except Exception as e:
        print(f"  {mod}: MISSING ({e})")
        sys.exit(2)
PY

  log "GPU pool: $GPU_POOL"
  python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import torch
print(f"  cuda_avail={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
PY

  log "checking baseline targets..."
  local idx=0 ok=0 missing=0
  for entry in "${BASELINE_TARGETS[@]}"; do
    idx=$((idx+1))
    IFS='|' read -r label model adapter_marker chat sft_style nspi <<<"$entry"
    local adapter
    adapter="$(resolve_adapter "$adapter_marker")"
    local model_status="OK"
    [[ -d "$model" ]] || model_status="MISSING"
    local adapter_status="(none)"
    if [[ -n "$adapter_marker" ]]; then
      if [[ -n "$adapter" && -d "$adapter" ]]; then
        adapter_status="OK ($adapter)"
      else
        adapter_status="MISSING (marker=$adapter_marker)"
      fi
    fi
    log "  [$idx] $label: model=$model_status adapter=$adapter_status"
    if [[ "$model_status" != "OK" ]]; then missing=$((missing+1)); else ok=$((ok+1)); fi
  done
  log "summary: ${ok} ok, ${missing} missing"

  log "benchmark data:"
  for d in GSM8K MATH-500 AIME2024 AIME2025 'CN Middle' Olympiad MMLU GPQA_Diamond; do
    local p="data/benchmarks/${d}"
    if [[ -d "$p" ]]; then log "  ? ${p}"; else log "  ? ${p} MISSING"; fi
  done

  return 0
}

# --------------------------------------------------------------------------
# Phase 2: eval ? ??? baseline ??????????
# --------------------------------------------------------------------------
phase_eval() {
  log "=== eval: launching baseline evaluations ==="
  mkdir -p "$EVAL_OUT" "$BASELINE_LOGS"

  local idx=0
  for entry in "${BASELINE_TARGETS[@]}"; do
    idx=$((idx+1))
    IFS='|' read -r label model adapter_marker chat sft_style nspi <<<"$entry"
    local adapter
    adapter="$(resolve_adapter "$adapter_marker")"

    if [[ ! -d "$model" ]]; then
      log "[$idx] SKIP $label (model missing: $model)"
      continue
    fi
    if [[ -n "$adapter_marker" ]]; then
      if [[ -z "$adapter" || ! -d "$adapter" ]]; then
        log "[$idx] SKIP $label (adapter missing for marker=$adapter_marker)"
        continue
      fi
    fi

    local label_log="$BASELINE_LOGS/${label}_orchestrator_${TODAY}.log"
    local cmd=(
      python -u scripts/unified_eval_orchestrator.py
      --model "$model"
      --label "$label"
      --benchmarks all
      --gpus "$GPU_POOL"
      --output_dir "$EVAL_OUT"
      --log_dir "$BASELINE_LOGS"
      --num_samples_per_item "$nspi"
      --k_values 1 5
    )
    [[ "$chat" == "true" ]] && cmd+=(--use_chat_template)
    [[ "$sft_style" == "true" ]] && cmd+=(--sft_style)
    [[ -n "$adapter" ]] && cmd+=(--adapter "$adapter")

    log "[$idx] START $label"
    log "    cmd: ${cmd[*]}"
    if "${cmd[@]}" >>"$label_log" 2>&1; then
      log "[$idx] DONE  $label (log: $label_log)"
    else
      log "[$idx] FAIL  $label (rc=$? log: $label_log) ? orchestrator handles per-bench retry internally"
    fi
  done

  log "eval phase done."
  return 0
}

# --------------------------------------------------------------------------
# Phase 3: aggregate ? ??? server_B label ? metrics ???
# results/baseline/{leaderboard_baseline.csv, metrics_full_baseline.json}
# --------------------------------------------------------------------------
phase_aggregate() {
  log "=== aggregate: collecting metrics into results/baseline/ ==="
  mkdir -p "$BASELINE_OUT"
  local expected=()
  for entry in "${BASELINE_TARGETS[@]}"; do
    IFS='|' read -r label _ _ _ _ _ <<<"$entry"
    expected+=("$label")
  done
  python -u scripts/aggregate_baseline_results_B.py \
    --eval_dir "$EVAL_OUT" \
    --output_dir "$BASELINE_OUT" \
    --label_suffix "$SERVER_TAG" \
    --expected_labels "${expected[@]}" 2>&1 | tee -a "$LOG_FILE"
  log "aggregate phase done."
  return 0
}

# --------------------------------------------------------------------------
# Phase 4: sync ? ?? missing_cells / merge_manifest
# --------------------------------------------------------------------------
phase_sync() {
  log "=== sync: generating manifest & missing-cell report ==="
  local expected=()
  for entry in "${BASELINE_TARGETS[@]}"; do
    IFS='|' read -r label _ _ _ _ _ <<<"$entry"
    expected+=("$label")
  done
  python -u scripts/aggregate_baseline_results_B.py \
    --eval_dir "$EVAL_OUT" \
    --output_dir "$BASELINE_OUT" \
    --label_suffix "$SERVER_TAG" \
    --expected_labels "${expected[@]}" \
    --emit_manifest 2>&1 | tee -a "$LOG_FILE"
  log "sync phase done."
  return 0
}

mkdir -p "$BASELINE_LOGS"

case "${1:-help}" in
  prepare)   phase_prepare ;;
  eval)      phase_eval ;;
  aggregate) phase_aggregate ;;
  sync)      phase_sync ;;
  all)       phase_prepare && phase_eval && phase_aggregate && phase_sync ;;
  help|*)
    cat <<'USAGE'
Usage:
  todo_baseline.sh prepare       # check env / models / data
  todo_baseline.sh eval          # launch baseline evaluations (resumable)
  todo_baseline.sh aggregate     # build leaderboard / metrics_full
  todo_baseline.sh sync          # build manifest / missing-cell report
  todo_baseline.sh all           # all four phases in order

Env knobs:
  GPU_POOL=0,1,2,3,4,5,6,7   GPU whitelist (default: all 8)
  HF_TOKEN, WANDB_API_KEY, ALL_PROXY    standard auth/proxy
USAGE
    ;;
esac
