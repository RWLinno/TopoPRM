#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

PHASE="all"
MODE="${MODE:-ours}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
CONDA_ENV="${CONDA_ENV:-topoprm}"
RUN_STUDENT_VARIANTS="${RUN_STUDENT_VARIANTS:-1}"
NUM_SAMPLES_PER_ITEM="${NUM_SAMPLES_PER_ITEM:-5}"
K_VALUES="${K_VALUES:-1 5}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)
            PHASE="${2:?missing phase value}"
            shift 2
            ;;
        --mode)
            MODE="${2:?missing mode value}"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash todo_exp_ours.sh [--phase all|invariants|archive|data|train|eval|sync] [--mode ours|legacy|v2]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

export PATH="/mnt/users/conda_env/${CONDA_ENV}/bin:$PATH"
mkdir -p logs output/eval output/analysis

# ─── v2 mode: load all P0–P5 patch env vars ───────────────────────────────
if [[ "$MODE" == "v2" ]]; then
    V2_ENV="configs/grpo_topoprm_v2.env"
    if [[ -f "$V2_ENV" ]]; then
        set -a
        # shellcheck disable=SC1090
        source <(grep -v '^\s*#' "$V2_ENV" | grep -v '^\s*$')
        set +a
    fi
    echo "[mode=v2] All P0–P5 patches loaded from $V2_ENV"
fi
# ───────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "TopoPRM Unified Orchestrator"
echo "phase=${PHASE} mode=${MODE} gpus=${GPUS}"
echo "============================================================"

latest_ckpt() {
    local pattern="$1"
    ls -d $pattern 2>/dev/null | sort -V | tail -1 || true
}

phase_invariants() {
    echo ""
    echo "[Phase] invariants"
    python3 scripts/check_reward_invariants.py
}

phase_archive() {
    echo ""
    echo "[Phase] archive"
    local archive_dir="archive/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$archive_dir"

    if [ -d "logs" ] && [ -n "$(ls -A logs 2>/dev/null)" ]; then
        cp -r logs "$archive_dir/logs_snapshot"
    fi
    if [ -d "output/eval" ] && [ -n "$(ls -A output/eval 2>/dev/null)" ]; then
        cp -r output/eval "$archive_dir/eval_snapshot"
    fi
    if [ -d "output/analysis" ] && [ -n "$(ls -A output/analysis 2>/dev/null)" ]; then
        cp -r output/analysis "$archive_dir/analysis_snapshot"
    fi
    if [ -d "wandb" ]; then
        ls -la wandb/ > "$archive_dir/wandb_manifest.txt" 2>/dev/null || true
        echo "[NOTE] wandb dry-run only. To clean project-local runs: rm -rf wandb/run-*"
    fi
    echo "archive written to $archive_dir"
}

phase_data() {
    echo ""
    echo "[Phase] data"
    if [[ "$MODE" == "legacy" ]]; then
        USE_AUGMENTED=0 bash scripts/run_data_pipeline.sh data/raw
    else
        USE_AUGMENTED=1 VARIANTS="${VARIANTS:-original correct_short local_wrong}" \
            bash scripts/run_data_pipeline.sh data/raw
    fi
}

phase_train() {
    echo ""
    echo "[Phase] train"

    bash scripts/run_sft_config.sh sft_qwen35_9b 2>&1 | tee logs/todo_ours_sft_main.log
    if [[ "$RUN_STUDENT_VARIANTS" == "1" ]]; then
        for cfg in sft_student_8b sft_student_4b sft_student_0p5b; do
            bash scripts/run_sft_config.sh "$cfg" 2>&1 | tee "logs/todo_ours_${cfg}.log"
        done
    fi

    bash scripts/run_grpo.sh grpo_hierarchical_qwen35_9b 2>&1 | tee logs/todo_ours_grpo_hier_main.log
    if [[ "$RUN_STUDENT_VARIANTS" == "1" ]]; then
        for cfg in grpo_hierarchical_student_8b grpo_hierarchical_student_4b grpo_hierarchical_student_0p5b; do
            bash scripts/run_grpo.sh "$cfg" 2>&1 | tee "logs/todo_ours_${cfg}.log"
        done
    fi

    if [ -f scripts/run_distill.sh ]; then
        bash scripts/run_distill.sh distill_7b_compact_rkl 2>&1 | tee logs/todo_ours_distill_8b.log
        if [[ "$RUN_STUDENT_VARIANTS" == "1" ]]; then
            bash scripts/run_distill.sh distill_student_4b_compact_rkl 2>&1 | tee logs/todo_ours_distill_4b.log
            bash scripts/run_distill.sh distill_student_0p5b_compact_rkl 2>&1 | tee logs/todo_ours_distill_0p5b.log
        fi
    else
        echo "[WARN] scripts/run_distill.sh not found, skip distill"
    fi
}

phase_eval() {
    echo ""
    echo "[Phase] eval"
    IFS=',' read -r -a GPU_LIST <<< "$GPUS"
    local gpu_count="${#GPU_LIST[@]}"
    if [[ "$gpu_count" -eq 0 ]]; then
        echo "[ERROR] no GPU in CUDA_VISIBLE_DEVICES"
        exit 1
    fi

    declare -A MODELS=(
        ["base_9b"]="/mnt/data/huggingface_downloads/models/qwen/Qwen3.5-9B"
        ["sft_9b"]="/mnt/data/huggingface_downloads/models/qwen/Qwen3.5-9B"
        ["grpo_hier_9b"]="/mnt/data/huggingface_downloads/models/qwen/Qwen3.5-9B"
        ["base_8b"]="/mnt/users/rwl/models/Qwen3-8B"
        ["base_4b"]="/mnt/data/huggingface_downloads/models/qwen/Qwen3.5-4B"
        ["base_0p5b"]="/mnt/users/rwl/models/Qwen2.5-0.5B-doctor"
    )
    declare -A ADAPTERS=(
        ["base_9b"]=""
        ["sft_9b"]="$(latest_ckpt "output/sft_qwen35_9b/v*/checkpoint-*")"
        ["grpo_hier_9b"]="$(latest_ckpt "output/grpo_hierarchical_qwen35_9b*/v*/checkpoint-*")"
        ["base_8b"]=""
        ["base_4b"]=""
        ["base_0p5b"]=""
    )

    local benchmarks=(gsm8k math500 olympiadbench omni_math aime2024 cnmo2024 livecode mmlu gpqa_diamond)
    local idx=0
    local running=0
    for label in "${!MODELS[@]}"; do
        local model="${MODELS[$label]}"
        local adapter="${ADAPTERS[$label]}"
        if [[ -z "$model" || ! -d "$model" ]]; then
            echo "[SKIP] $label model not found: $model"
            continue
        fi

        local gpu="${GPU_LIST[$((idx % gpu_count))]}"
        idx=$((idx + 1))
        local cmd=(python3 scripts/bench_transformers.py
            --model "$model"
            --label "$label"
            --benchmarks "${benchmarks[@]}"
            --num_samples_per_item "$NUM_SAMPLES_PER_ITEM"
            --k_values $K_VALUES
            --batch_size 4
            --max_new_tokens 1536
        )
        if [[ -n "$adapter" && -d "$adapter" ]]; then
            cmd+=(--adapter "$adapter")
        fi

        echo "[EVAL] label=$label gpu=$gpu adapter=${adapter:-none}"
        CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" 2>&1 | tee "logs/eval_${label}.log" &
        running=$((running + 1))
        if [[ "$running" -ge "$gpu_count" ]]; then
            wait -n
            running=$((running - 1))
        fi
    done
    wait
}

phase_sync() {
    echo ""
    echo "[Phase] sync"
    local summary_path="output/analysis/experiment_summary.json"
    python3 -m src.eval.collect_experiment_results --output_dir output/analysis --eval_dir output/eval
    python3 -m src.eval.sync_paper_tables --summary "$summary_path" --paper_dir topoprm_paper --eval_dir output/eval --progress_file output/analysis/experiment_progress.md
    if python3 -m src.eval.export_paper_tables --help >/dev/null 2>&1; then
        python3 -m src.eval.export_paper_tables || true
    fi
    python3 scripts/check_reward_invariants.py || echo "[WARN] reward invariant check failed"
}

case "$PHASE" in
    all)
        phase_invariants
        phase_archive
        phase_data
        phase_train
        phase_eval
        phase_sync
        ;;
    invariants) phase_invariants ;;
    archive) phase_archive ;;
    data) phase_data ;;
    train) phase_train ;;
    eval) phase_eval ;;
    sync) phase_sync ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Available: all invariants archive data train eval sync"
        exit 1
        ;;
esac
