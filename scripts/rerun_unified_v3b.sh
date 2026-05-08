#!/usr/bin/env bash
# Unified benchmark re-run v3b (2026-04-21):
#   * label+bench-list precise dispatch (no more full-model pipelines; you
#     just tell it which benches a given label still misses).
#   * Larger batches to utilise 143 GB L20X cards (batch=6 long / 8 medium
#     / 16 short) -> ~2-3x throughput over v3.
#   * MMLU subset 1500 -> 500 (still ?<3pp, saves 3x on that bench).
#   * Skip-if-exists relies on bench_transformers.py's new default behaviour
#     (see --force_overwrite).
#   * LiveCode permanently dropped (load_livecode returns []).
#
# Usage:
#   bash scripts/rerun_unified_v3b.sh run GPU LABEL MODEL ADAPTER SFT_STYLE BENCH_CSV
#   bash scripts/rerun_unified_v3b.sh queue GPU JOBFILE
#
#   LABEL       : e.g. topoprm_hier_9b_v3
#   MODEL       : path to base model
#   ADAPTER     : path to adapter, or "" for none
#   SFT_STYLE   : "sft" or "nosft"
#   BENCH_CSV   : comma-separated benches, e.g.
#                 "olympiadbench,omni_math,aime2024,aime2025,cnmo2024,gsm8k,math500,mmlu,gpqa_diamond"
#
#   JOBFILE     : text file, one job per line: "LABEL MODEL ADAPTER SFT BENCH_CSV"

set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/mnt/users/conda_env/topoprm/bin:$PATH"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
mkdir -p logs output/eval

MMLU_MAX_ITEMS=500

# Benchmark groupings.
is_long()   { case "$1" in olympiadbench|omni_math|aime2024|aime2025|cnmo2024) return 0 ;; *) return 1 ;; esac; }
is_medium() { case "$1" in gsm8k|math500) return 0 ;; *) return 1 ;; esac; }
is_short()  { case "$1" in mmlu|gpqa_diamond) return 0 ;; *) return 1 ;; esac; }

group_params() {
    # Echoes: "max_new_tokens batch_size max_items"
    local bench="$1"
    if is_long "$bench"; then
        echo "2560 6 0"
    elif is_medium "$bench"; then
        echo "1536 8 0"
    elif is_short "$bench"; then
        echo "512 16 ${MMLU_MAX_ITEMS}"
    else
        # livecode or unknown: skip silently at caller level
        echo "SKIP 0 0"
    fi
}

run_one_group() {
    local gpu="$1" label="$2" model="$3" adapter="$4" sft_style="$5"
    shift 5
    local benches=("$@")
    [[ "${#benches[@]}" -eq 0 ]] && return 0

    # All benches in `benches` must belong to the same group.
    local first_bench="${benches[0]}"
    local params
    params=$(group_params "$first_bench")
    local mnt bs items
    read -r mnt bs items <<< "$params"
    [[ "$mnt" == "SKIP" ]] && { echo "[skip-group] $label ${benches[*]}"; return 0; }

    local cmd=(python3 scripts/bench_transformers.py
        --model "$model"
        --label "$label"
        --benchmarks "${benches[@]}"
        --num_samples_per_item 5
        --k_values 1 5
        --batch_size "$bs"
        --max_new_tokens "$mnt"
        --temperature 0.7
        --top_p 0.95
        --use_chat_template
    )
    [[ "$sft_style" == "sft" ]] && cmd+=(--sft_style)
    [[ -n "$adapter" && "$adapter" != "-" ]] && cmd+=(--adapter "$adapter")
    [[ "$items" -gt 0 ]] && cmd+=(--max_items "$items")

    local stamp
    stamp=$(date +%H%M%S)
    local log="logs/eval_${label}_v3b_mnt${mnt}_${stamp}.log"
    echo "[$(date '+%H:%M:%S')] GPU${gpu} $label mnt=$mnt bs=$bs items=$items benches=${benches[*]} -> $log"
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$log" 2>&1 || {
        echo "[WARN] $label group (mnt=$mnt) non-zero exit, continuing"
    }
}

run_one_label() {
    # Split the bench list into long / medium / short groups and run them
    # serially, so each group gets its correct max_new_tokens/batch_size.
    local gpu="$1" label="$2" model="$3" adapter="$4" sft_style="$5" bench_csv="$6"

    local all_benches=()
    IFS=',' read -ra all_benches <<< "$bench_csv"

    local long_list=() medium_list=() short_list=()
    for b in "${all_benches[@]}"; do
        if is_long "$b"; then long_list+=("$b")
        elif is_medium "$b"; then medium_list+=("$b")
        elif is_short "$b"; then short_list+=("$b")
        fi
    done

    [[ "${#long_list[@]}"   -gt 0 ]] && run_one_group "$gpu" "$label" "$model" "$adapter" "$sft_style" "${long_list[@]}"
    [[ "${#medium_list[@]}" -gt 0 ]] && run_one_group "$gpu" "$label" "$model" "$adapter" "$sft_style" "${medium_list[@]}"
    [[ "${#short_list[@]}"  -gt 0 ]] && run_one_group "$gpu" "$label" "$model" "$adapter" "$sft_style" "${short_list[@]}"
}

main() {
    local action="${1:?usage: $0 [run|queue] ...}"
    shift
    case "$action" in
        run)
            local gpu="$1" label="$2" model="$3" adapter="$4" sft_style="$5" bench_csv="$6"
            run_one_label "$gpu" "$label" "$model" "$adapter" "$sft_style" "$bench_csv"
            ;;
        queue)
            local gpu="$1" jobfile="$2"
            [[ -f "$jobfile" ]] || { echo "job file not found: $jobfile"; exit 1; }
            echo "[$(date '+%H:%M:%S')] GPU${gpu} queue start <- $jobfile"
            while IFS= read -r line || [[ -n "$line" ]]; do
                line="${line%%#*}"   # strip comments
                line="${line#"${line%%[![:space:]]*}"}" # ltrim
                [[ -z "$line" ]] && continue
                # expected 5 space-separated fields: LABEL MODEL ADAPTER SFT BENCHCSV
                read -r LBL MDL ADP SFT BCS <<< "$line"
                [[ -z "$BCS" ]] && { echo "[skip-malformed] $line"; continue; }
                echo "[$(date '+%H:%M:%S')] GPU${gpu} --> $LBL"
                run_one_label "$gpu" "$LBL" "$MDL" "$ADP" "$SFT" "$BCS"
            done < "$jobfile"
            echo "[$(date '+%H:%M:%S')] GPU${gpu} queue complete"
            ;;
        *)
            echo "Usage: $0 run GPU LABEL MODEL ADAPTER SFT_STYLE BENCH_CSV"
            echo "       $0 queue GPU JOBFILE"
            exit 1
            ;;
    esac
}

main "$@"
