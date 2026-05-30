#!/usr/bin/env bash
# ============================================================================
# run_unified_eval.sh — one-click 9-benchmark evaluation.
#
# Usage:
#   bash scripts/run_unified_eval.sh [MODEL_PATH] [ADAPTER|""] [LABEL]
#
# Env overrides:
#   GPUS=0,1,2,3,4,5,6,7   CUDA ids pool (default: all visible)
#   BENCHMARKS="gsm8k math500 ..."   default: "all" (9 benches)
#   SFT_STYLE=1            enable --sft_style (for SFT/GRPO adapters)
#   USE_CHAT=1             enable --use_chat_template (default on)
#   NUM_SAMPLES=1          --num_samples_per_item
#   KS="1"                 --k_values (space-sep)
#   FORCE=1                re-run even if metrics.json exists
#   RUN_IN_BACKGROUND=1    launch orchestrator with nohup & echo PID
#
# Examples:
#   # Base model
#   bash scripts/run_unified_eval.sh \
#       ${HF_MODELS_DIR:-./models}/Qwen/Qwen3.5-9B "" qwen35_9b_base
#
#   # +SFT (auto-detects latest checkpoint if ADAPTER="")
#   SFT_STYLE=1 bash scripts/run_unified_eval.sh \
#       ${HF_MODELS_DIR:-./models}/Qwen/Qwen3.5-9B "" qwen35_9b_sft
#
#   # Background
#   RUN_IN_BACKGROUND=1 bash scripts/run_unified_eval.sh \
#       ${HF_MODELS_DIR:-./models}/Qwen/Qwen3.5-9B "" qwen35_9b_base
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${1:-${HF_MODELS_DIR:-./models}/Qwen/Qwen3.5-9B}"
ADAPTER_ARG="${2:-}"
LABEL_ARG="${3:-}"

PYTHON_BIN="${TOPOPRM_PYTHON:-${TOPOPRM_PYTHON:-python3}}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] model path not found: $MODEL_PATH" >&2
    exit 2
fi

derive_label() {
    local base
    base="$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_' '_')"
    if [[ -n "$ADAPTER_ARG" ]]; then
        echo "${base}_sft"
    else
        echo "${base}_base"
    fi
}
LABEL="${LABEL_ARG:-$(derive_label)}"

GPUS="${GPUS:-}"
if [[ -z "$GPUS" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        GPUS="$CUDA_VISIBLE_DEVICES"
    else
        NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
        [[ -z "$NGPU" || "$NGPU" -le 0 ]] && NGPU=1
        GPUS="$(seq -s, 0 $((NGPU-1)))"
    fi
fi

BENCHMARKS="${BENCHMARKS:-all}"
SFT_STYLE_FLAG=""
if [[ "${SFT_STYLE:-0}" == "1" ]]; then
    SFT_STYLE_FLAG="--sft_style"
fi
USE_CHAT_FLAG="--use_chat_template"
if [[ "${USE_CHAT:-1}" == "0" ]]; then
    USE_CHAT_FLAG="--no_chat_template"
fi
NUM_SAMPLES="${NUM_SAMPLES:-1}"
KS="${KS:-1}"
FORCE_FLAG=""
[[ "${FORCE:-0}" == "1" ]] && FORCE_FLAG="--force"

# Auto-discover adapter if requested SFT but no path given.
if [[ "${SFT_STYLE:-0}" == "1" && -z "$ADAPTER_ARG" ]]; then
    CAND="$(ls -td output/sft_qwen* output/sft_*/final output/sft_*/checkpoint-* 2>/dev/null | head -1 || true)"
    if [[ -n "$CAND" && -d "$CAND" ]]; then
        # Verify adapter matches this base model family; warn on mismatch.
        CFG="$CAND/adapter_config.json"
        if [[ -f "$CFG" ]]; then
            BASE_IN_CFG="$(grep -oP '"base_model_name_or_path"\s*:\s*"\K[^"]+' "$CFG" 2>/dev/null || true)"
            MODEL_BASENAME="$(basename "$MODEL_PATH")"
            if [[ -n "$BASE_IN_CFG" && "$BASE_IN_CFG" != *"$MODEL_BASENAME"* ]]; then
                echo "[WARN] auto-discovered adapter '$CAND' was trained on '$BASE_IN_CFG'," >&2
                echo "       which does not match the base model '$MODEL_BASENAME'." >&2
                echo "       Pass an explicit adapter path or skip SFT." >&2
                exit 3
            fi
        fi
        ADAPTER_ARG="$CAND"
        echo "[INFO] auto-discovered SFT adapter: $ADAPTER_ARG"
    else
        echo "[ERROR] SFT_STYLE=1 requested but no adapter path given and none found under output/sft_*" >&2
        echo "        pass adapter path as 2nd arg, or unset SFT_STYLE to run base only." >&2
        exit 3
    fi
fi

ADAPTER_FLAG=""
if [[ -n "$ADAPTER_ARG" ]]; then
    if [[ ! -d "$ADAPTER_ARG" ]]; then
        echo "[ERROR] adapter dir not found: $ADAPTER_ARG" >&2
        exit 2
    fi
    ADAPTER_FLAG="--adapter $ADAPTER_ARG"
fi

mkdir -p logs/unified output/eval
export TOKENIZERS_PARALLELISM=false

# Flatten BENCHMARKS which may be quoted as "all" or "gsm8k math500 ..."
read -r -a BENCH_ARR <<< "$BENCHMARKS"
KS_ARR=()
for k in $KS; do KS_ARR+=("$k"); done

CMD=(
    "$PYTHON_BIN" -u "$REPO_ROOT/scripts/unified_eval_orchestrator.py"
    --model "$MODEL_PATH"
    --label "$LABEL"
    --gpus "$GPUS"
    --benchmarks "${BENCH_ARR[@]}"
    --num_samples_per_item "$NUM_SAMPLES"
    --k_values "${KS_ARR[@]}"
    $USE_CHAT_FLAG
    $SFT_STYLE_FLAG
    $FORCE_FLAG
    --output_dir output/eval
    --log_dir logs/unified
)
if [[ -n "$ADAPTER_FLAG" ]]; then
    CMD+=(--adapter "$ADAPTER_ARG")
fi

echo "══════════════════════════════════════════"
echo " TopoPRM unified evaluation"
echo "══════════════════════════════════════════"
echo "  Python:     $PYTHON_BIN"
echo "  Model:      $MODEL_PATH"
echo "  Adapter:    ${ADAPTER_ARG:-<none>}"
echo "  Label:      $LABEL"
echo "  GPUs:       $GPUS"
echo "  Benchmarks: ${BENCH_ARR[*]}"
echo "  chat=${USE_CHAT:-1} sft_style=${SFT_STYLE:-0} samples=$NUM_SAMPLES ks=$KS force=${FORCE:-0}"
echo "  Output:     output/eval/${LABEL}_<bench>_metrics.json"
echo "  Logs:       logs/unified/${LABEL}_<bench>.log"
echo "  Monitor:    bash scripts/watch_unified_eval.sh $LABEL"
echo "══════════════════════════════════════════"

if [[ "${RUN_IN_BACKGROUND:-0}" == "1" ]]; then
    BG_LOG="logs/unified/orchestrator_${LABEL}_$(date +%Y%m%d_%H%M%S).out"
    nohup "${CMD[@]}" >"$BG_LOG" 2>&1 &
    BG_PID=$!
    echo "$BG_PID" > "logs/unified/orchestrator_${LABEL}.pid"
    echo "[INFO] orchestrator backgrounded as PID $BG_PID (log: $BG_LOG)"
    echo "[INFO] watch:  tail -f $BG_LOG"
    echo "[INFO] status: cat logs/unified/status_${LABEL}.json"
else
    exec "${CMD[@]}"
fi
