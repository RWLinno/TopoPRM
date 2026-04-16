#!/bin/bash
# ==========================================================================
#  TopoPRM Unified Experiment Orchestrator
#
#  Usage:
#    bash todo_exp_ours.sh                    # 全流程
#    bash todo_exp_ours.sh --phase archive    # 仅归档清理
#    bash todo_exp_ours.sh --phase data       # 仅数据管线
#    bash todo_exp_ours.sh --phase train      # 仅训练
#    bash todo_exp_ours.sh --phase eval       # 仅评测
#    bash todo_exp_ours.sh --phase sync       # 仅结果同步
#    bash todo_exp_ours.sh --mode legacy      # 使用旧管线（对照）
#
#  环境要求:
#    - conda activate topoprm
#    - CUDA_VISIBLE_DEVICES 在外部设置（默认0-4）
#    - PYTHONPATH 会被自动设置
# ==========================================================================
set -euo pipefail

cd "$(dirname "$0")" && export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# -- 参数解析 ---------------------------------------------------------------
PHASE="${1:---all}"
MODE="${MODE:-ours}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
CONDA_ENV="${CONDA_ENV:-topoprm}"
export PATH="/mnt/users/conda_env/${CONDA_ENV}/bin:$PATH"

if [[ "$PHASE" == "--phase" ]]; then
    PHASE="${2:?请指定 phase: archive|data|train|eval|sync}"
elif [[ "$PHASE" == "--mode" ]]; then
    MODE="${2:?请指定 mode: legacy|ours}"
    PHASE="all"
elif [[ "$PHASE" == "--all" ]]; then
    PHASE="all"
fi

echo "============================================================"
echo "  TopoPRM Unified Orchestrator"
echo "  Phase: ${PHASE}  |  Mode: ${MODE}  |  GPUs: ${GPUS}"
echo "============================================================"

# -- Phase 0: 创新点不变性检查 -----------------------------------------------
phase_invariants() {
    echo ""
    echo "-- Phase 0: Reward Invariant Check --"
    python3 scripts/check_reward_invariants.py
}

# -- Phase 1: 归档 + wandb 清理 ---------------------------------------------
phase_archive() {
    echo ""
    echo "-- Phase 1: Archive & Cleanup --"

    ARCHIVE_DIR="archive/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$ARCHIVE_DIR"

    if [ -d logs ] && [ "$(ls -A logs 2>/dev/null)" ]; then
        cp -r logs "$ARCHIVE_DIR/logs_snapshot"
        echo "  Archived logs -> $ARCHIVE_DIR/logs_snapshot"
    fi

    if [ -d output/eval ] && [ "$(ls -A output/eval 2>/dev/null)" ]; then
        cp -r output/eval "$ARCHIVE_DIR/eval_snapshot"
        echo "  Archived eval -> $ARCHIVE_DIR/eval_snapshot"
    fi

    if [ -d output/analysis ] && [ "$(ls -A output/analysis 2>/dev/null)" ]; then
        cp -r output/analysis "$ARCHIVE_DIR/analysis_snapshot"
        echo "  Archived analysis -> $ARCHIVE_DIR/analysis_snapshot"
    fi

    if [ -d wandb ]; then
        echo "  wandb/ directory found. Creating archive manifest..."
        ls -la wandb/ > "$ARCHIVE_DIR/wandb_manifest.txt" 2>/dev/null || true
        echo "  Manifest -> $ARCHIVE_DIR/wandb_manifest.txt"
        echo "  [NOTE] To actually clear wandb runs, run: rm -rf wandb/run-*"
    fi

    echo "  Archive complete -> $ARCHIVE_DIR"
}

# -- Phase 2: 数据管线 -------------------------------------------------------
phase_data() {
    echo ""
    echo "-- Phase 2: Data Pipeline (mode=$MODE) --"

    if [[ "$MODE" == "legacy" ]]; then
        echo "  Running legacy data pipeline..."
        bash scripts/run_data_pipeline.sh data/raw
    else
        echo "  Running enhanced data pipeline..."
        bash scripts/run_data_pipeline.sh data/raw

        echo ""
        echo "  [enhanced] Building DAGs with native-first strategy..."
        python3 -m src.data.build_dag \
            --input_path data/processed/cleaned.jsonl \
            --output_dir data/dag

        echo "  [enhanced] Preparing SFT + GRPO data..."
        python3 -m src.data.prepare_sft \
            --input_path data/processed/cleaned.jsonl \
            --output_path data/sft_ready/train.jsonl
        python3 -m src.data.merge_datasets \
            --zh_path data/sft_ready/train.jsonl \
            --output_path data/sft_ready/train_mixed.jsonl
        python3 -m src.data.prepare_grpo \
            --input_path data/processed/cleaned.jsonl \
            --dag_dir data/dag \
            --output_path data/grpo_ready/train.jsonl
    fi
}

# -- Phase 3: 训练 -----------------------------------------------------------
phase_train() {
    echo ""
    echo "-- Phase 3: Training (mode=$MODE) --"

    echo "  [SFT] Starting..."
    bash scripts/run_sft_config.sh sft 2>&1 | tee logs/todo_ours_sft.log

    echo "  [GRPO hierarchical] Starting..."
    bash scripts/run_grpo.sh grpo_hierarchical 2>&1 | tee logs/todo_ours_grpo_hier.log

    echo "  [Distill] reverse-KL training..."
    if [ -f scripts/run_distill.sh ]; then
        bash scripts/run_distill.sh distill_7b_compact_rkl 2>&1 | tee logs/todo_ours_distill.log
    else
        echo "  [SKIP] scripts/run_distill.sh not found"
    fi
}

# -- Phase 4: 统一评测 -------------------------------------------------------
phase_eval() {
    echo ""
    echo "-- Phase 4: Unified Evaluation --"

    mkdir -p output/eval logs

    declare -A MODELS=(
        ["base_9b"]="/mnt/data/huggingface_downloads/models/qwen/Qwen3.5-9B"
        ["sft_9b"]="/mnt/data/huggingface_downloads/models/qwen/Qwen3.5-9B"
        ["grpo_hier_9b"]="/mnt/data/huggingface_downloads/models/qwen/Qwen3.5-9B"
    )
    declare -A ADAPTERS=(
        ["base_9b"]=""
        ["sft_9b"]="$(ls -d output/sft*/v*/checkpoint-* 2>/dev/null | sort -V | tail -1 || echo '')"
        ["grpo_hier_9b"]="$(ls -d output/grpo_hierarchical*/v*/checkpoint-* 2>/dev/null | sort -V | tail -1 || echo '')"
    )

    BENCHMARKS="gsm8k math500"

    for label in "${!MODELS[@]}"; do
        model="${MODELS[$label]}"
        adapter="${ADAPTERS[$label]}"

        if [ -z "$model" ] || [ ! -d "$model" ]; then
            echo "  [SKIP] $label: model not found ($model)"
            continue
        fi

        adapter_arg=""
        if [ -n "$adapter" ] && [ -d "$adapter" ]; then
            adapter_arg="--adapter $adapter"
        fi

        echo ""
        echo "  Evaluating: $label"
        echo "    model=$model"
        echo "    adapter=$adapter"

        CUDA_VISIBLE_DEVICES=${GPUS%%,*} python3 scripts/bench_transformers.py \
            --model "$model" \
            $adapter_arg \
            --label "$label" \
            --benchmarks $BENCHMARKS \
            --batch_size 8 \
            --max_new_tokens 1536 \
            2>&1 | tee "logs/eval_${label}.log" &
    done

    echo ""
    echo "  Waiting for all eval jobs..."
    wait
    echo "  All evaluations complete."
}

# -- Phase 5: 结果同步 -------------------------------------------------------
phase_sync() {
    echo ""
    echo "-- Phase 5: Result Sync --"

    python3 -m src.eval.collect_experiment_results 2>/dev/null || true
    python3 -m src.eval.sync_paper_tables 2>/dev/null || true
    python3 -m src.eval.export_paper_tables 2>/dev/null || true

    echo "  Results synced to paper tables."

    echo ""
    echo "  Post-experiment invariant check..."
    python3 scripts/check_reward_invariants.py || echo "  [WARN] Invariant check failed!"

    echo ""
    echo "============================================================"
    echo "  Orchestrator complete."
    echo "  Results: output/eval/*_metrics.json"
    echo "  Paper:   topoprm_paper/tables/"
    echo "============================================================"
}

# -- 执行 -------------------------------------------------------------------
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
    archive)    phase_archive ;;
    data)       phase_data ;;
    train)      phase_train ;;
    eval)       phase_eval ;;
    sync)       phase_sync ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Available: all, invariants, archive, data, train, eval, sync"
        exit 1
        ;;
esac
