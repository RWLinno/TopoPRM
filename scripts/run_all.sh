#!/bin/bash
set -u -o pipefail

# NeurIPS26 ordered pipeline runner.
# Safe defaults: expensive training stages are disabled unless explicitly enabled.
#
# Usage:
#   nohup bash scripts/run_all.sh > logs/run_all.log 2>&1 &
#   # enable heavy stages explicitly
#   RUN_SFT=1 RUN_GRPO=1 RUN_ABLATIONS=1 RUN_EVAL=1 nohup bash scripts/run_all.sh > logs/run_all.log 2>&1 &

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p logs output/eval

log_step() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_stage() {
  local name="$1"
  shift
  log_step "Stage: ${name}"
  if "$@"; then
    log_step "Stage OK: ${name}"
  else
    log_step "Stage FAIL (continue): ${name}"
  fi
}

RUN_BENCHMARK_DOWNLOAD="${RUN_BENCHMARK_DOWNLOAD:-1}"
RUN_DAG_METRICS="${RUN_DAG_METRICS:-1}"
RUN_SFT="${RUN_SFT:-0}"
RUN_GRPO="${RUN_GRPO:-0}"
RUN_ABLATIONS="${RUN_ABLATIONS:-0}"
RUN_AGGREGATORS="${RUN_AGGREGATORS:-0}"
RUN_SCAE="${RUN_SCAE:-0}"
RUN_EVAL="${RUN_EVAL:-0}"
RUN_DISTILL="${RUN_DISTILL:-0}"

if [ "$RUN_BENCHMARK_DOWNLOAD" = "1" ]; then
  run_stage "benchmark download + manifest" bash scripts/download_benchmarks.sh
fi

if [ "$RUN_DAG_METRICS" = "1" ]; then
  run_stage "DAG explainability metrics" python3 -m src.eval.dag_metrics --dag_dir data/dag --output output/eval/dag_metrics.json
fi

if [ "$RUN_SFT" = "1" ]; then
  run_stage "SFT" bash scripts/run_sft.sh
fi

if [ "$RUN_GRPO" = "1" ]; then
  run_stage "GRPO main" bash scripts/run_grpo.sh grpo_main
fi

if [ "$RUN_ABLATIONS" = "1" ]; then
  run_stage "GRPO outcome_only" bash scripts/run_grpo.sh grpo_outcome_only
  run_stage "GRPO no_topo" bash scripts/run_grpo.sh grpo_no_topo
  run_stage "GRPO no_continuity" bash scripts/run_grpo.sh grpo_no_continuity
fi

if [ "$RUN_AGGREGATORS" = "1" ]; then
  run_stage "GRPO mulgate" bash scripts/run_grpo.sh grpo_mulgate
  run_stage "GRPO confgate" bash scripts/run_grpo.sh grpo_confgate
  run_stage "GRPO clipped" bash scripts/run_grpo.sh grpo_clipped
fi

if [ "$RUN_SCAE" = "1" ]; then
  run_stage "GRPO SCAE-style" bash scripts/run_grpo.sh grpo_scae
fi

if [ "$RUN_EVAL" = "1" ]; then
  run_stage "evaluation all" bash scripts/run_eval_all.sh
  run_stage "export paper summary" python3 -m src.eval.export_paper_tables --eval_dir output/eval --output output/eval/paper_table_summary.csv
fi

if [ "$RUN_DISTILL" = "1" ]; then
  run_stage "distill data generation" bash scripts/generate_distill_data.sh
  run_stage "distill 7b" swift sft --config configs/distill_7b.yaml
  run_stage "distill 1.5b" swift sft --config configs/distill_1_5b.yaml
fi

log_step "Pipeline finished"
