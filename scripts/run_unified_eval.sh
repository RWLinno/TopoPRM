#!/usr/bin/env bash
# ============================================================================
# run_unified_eval.sh — one-click registered-benchmark evaluation.
#
# Usage:
#   bash scripts/run_unified_eval.sh [MODEL_PATH] [ADAPTER|""] [LABEL]
#
# Env overrides:
#   GPUS=0,1,2,3,4,5,6,7   CUDA ids pool (default: all visible)
#   BENCHMARKS="gsm8k math500 ..."   default: "all" (10 benches)
#   SFT_STYLE=1            enable --sft_style (for SFT/GRPO adapters)
#   USE_CHAT=1             enable --use_chat_template (default on)
#   NUM_SAMPLES=1          --num_samples_per_item
#   KS="1"                 --k_values (space-sep)
#   PASS1_DO_SAMPLE=1      one fixed-seed sampled response per item (canonical)
#   TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20 MIN_P=0.0
#   EVAL_SEED=0            generation seed (not a training seed)
#   PAIRED_BASELINES="label1 label2" compute paired deltas from saved details
#   FORCE=1                re-run even if metrics.json exists
#   RUN_IN_BACKGROUND=1    launch orchestrator with nohup & echo PID
#   EVAL_OUTPUT_DIR=...    canonical per-item/metric output directory
#   SAVE_SOLUTIONS=1       retain all sampled responses (pass@1 is always retained)
#   REPETITION_PENALTY=1.0 canonical decoder value
#   FOLD_SYSTEM_INTO_USER=1 use a user-only prompt protocol (e.g. DeepSeek-R1)
#   FORCE_THINK_PREFIX=1  prefill <think> for DeepSeek-R1-family evaluation
#   SCORE_TOPOLOGY=1       compute TopoPRM prm@k reranking metrics
#
# Examples:
#   # Base model
#   bash scripts/run_unified_eval.sh \
#       /Knowin/foundation/weilinruan/hf_models/Qwen/Qwen3.5-9B "" qwen35_9b_base
#
#   # +SFT (auto-detects latest checkpoint if ADAPTER="")
#   SFT_STYLE=1 bash scripts/run_unified_eval.sh \
#       /Knowin/foundation/weilinruan/hf_models/Qwen/Qwen3.5-9B "" qwen35_9b_sft
#
#   # Background
#   RUN_IN_BACKGROUND=1 bash scripts/run_unified_eval.sh \
#       /Knowin/foundation/weilinruan/hf_models/Qwen/Qwen3.5-9B "" qwen35_9b_base
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${1:-/Knowin/foundation/weilinruan/hf_models/Qwen/Qwen3.5-9B}"
ADAPTER_ARG="${2:-}"
LABEL_ARG="${3:-}"

PYTHON_BIN="${TOPOPRM_PYTHON:-/Knowin/foundation/weilinruan/env/qwen35/bin/python}"
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
SAVE_SOLUTIONS_FLAG=""
[[ "${SAVE_SOLUTIONS:-1}" == "1" ]] && SAVE_SOLUTIONS_FLAG="--save_solutions"
FOLD_SYSTEM_FLAG=""
[[ "${FOLD_SYSTEM_INTO_USER:-0}" == "1" ]] && FOLD_SYSTEM_FLAG="--fold_system_into_user"
THINK_PREFIX_FLAG=""
[[ "${FORCE_THINK_PREFIX:-0}" == "1" ]] && THINK_PREFIX_FLAG="--force_think_prefix"
EMPTY_SYSTEM_FLAG=""
[[ "${EMPTY_SYSTEM_PROMPT:-0}" == "1" ]] && EMPTY_SYSTEM_FLAG="--empty_system_prompt"
SCORE_TOPOLOGY_FLAG=""
[[ "${SCORE_TOPOLOGY:-0}" == "1" ]] && SCORE_TOPOLOGY_FLAG="--score_topology"
PASS1_SAMPLE_FLAG=""
[[ "${PASS1_DO_SAMPLE:-1}" == "1" ]] && PASS1_SAMPLE_FLAG="--pass1_do_sample"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
EVAL_SEED="${EVAL_SEED:-0}"
PAIRED_BASELINES="${PAIRED_BASELINES:-}"
SYSTEM_CONTROL="${SYSTEM_CONTROL:-}"
USER_SUFFIX="${USER_SUFFIX:-}"

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

EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-/knowin-oss/weilinruan/TopoPRM_ICLR27/canonical/eval}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-output/eval_logs}"
mkdir -p "$EVAL_LOG_DIR" "$EVAL_OUTPUT_DIR"
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
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --top_k "$TOP_K"
    --min_p "$MIN_P"
    --repetition_penalty "$REPETITION_PENALTY"
    --eval_seed "$EVAL_SEED"
    $USE_CHAT_FLAG
    $SFT_STYLE_FLAG
    $FORCE_FLAG
    $SAVE_SOLUTIONS_FLAG
    $FOLD_SYSTEM_FLAG
    $THINK_PREFIX_FLAG
    $EMPTY_SYSTEM_FLAG
    $SCORE_TOPOLOGY_FLAG
    $PASS1_SAMPLE_FLAG
    --output_dir "$EVAL_OUTPUT_DIR"
    --log_dir "$EVAL_LOG_DIR"
)
if [[ -n "$ADAPTER_FLAG" ]]; then
    CMD+=(--adapter "$ADAPTER_ARG")
fi
if [[ -n "$PAIRED_BASELINES" ]]; then
    read -r -a PAIRED_BASELINE_ARR <<< "$PAIRED_BASELINES"
    CMD+=(--paired_baseline_labels "${PAIRED_BASELINE_ARR[@]}")
fi
if [[ -n "$SYSTEM_CONTROL" ]]; then
    CMD+=(--system_control "$SYSTEM_CONTROL")
fi
if [[ -n "$USER_SUFFIX" ]]; then
    CMD+=(--user_suffix "$USER_SUFFIX")
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
echo "  chat=${USE_CHAT:-1} fold_system=${FOLD_SYSTEM_INTO_USER:-0} think_prefix=${FORCE_THINK_PREFIX:-0} empty_system=${EMPTY_SYSTEM_PROMPT:-0} system_control=${SYSTEM_CONTROL:-<none>} user_suffix=${USER_SUFFIX:-<none>} sft_style=${SFT_STYLE:-0}"
echo "  samples=$NUM_SAMPLES pass1_sample=${PASS1_DO_SAMPLE:-1} seed=$EVAL_SEED temp=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K min_p=$MIN_P"
echo "  ks=$KS rep_penalty=$REPETITION_PENALTY topology_scoring=${SCORE_TOPOLOGY:-0} force=${FORCE:-0}"
echo "  paired_baselines=${PAIRED_BASELINES:-<none>}"
echo "  Output:     $EVAL_OUTPUT_DIR/${LABEL}_<bench>_metrics.json"
echo "  Logs:       $EVAL_LOG_DIR/${LABEL}_<bench>.log"
echo "  Monitor:    bash scripts/watch_unified_eval.sh $LABEL"
echo "══════════════════════════════════════════"

if [[ "${RUN_IN_BACKGROUND:-0}" == "1" ]]; then
    BG_LOG="$EVAL_LOG_DIR/orchestrator_${LABEL}_$(date +%Y%m%d_%H%M%S).out"
    nohup "${CMD[@]}" >"$BG_LOG" 2>&1 &
    BG_PID=$!
    echo "$BG_PID" > "$EVAL_LOG_DIR/orchestrator_${LABEL}.pid"
    echo "[INFO] orchestrator backgrounded as PID $BG_PID (log: $BG_LOG)"
    echo "[INFO] watch:  tail -f $BG_LOG"
    echo "[INFO] status: cat $EVAL_LOG_DIR/status_${LABEL}.json"
else
    exec "${CMD[@]}"
fi
