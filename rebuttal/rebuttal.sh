#!/usr/bin/env bash
###############################################################################
# TopoPRM rebuttal master launcher.
#
# Each experiment runs as a detached job:  nohup run_xxx.sh > run_xxx.log 2>&1 &
# All heavy scripts are pinned to GPUs 4,5,6,7 (GPUs 0-3 left for other users).
#
# GPU allocation (avoid contention):
#   edge validation      -> GPU 4,5   (Qwen3-32B judge)
#   semantic gap         -> GPU 6,7   (Qwen3.5-9B generation)
#   outcome+length GRPO  -> GPU 4,5,6,7 (run AFTER the two generation jobs)
#   topoprm repro eval   -> GPU 4,5
#   non-qwen eval        -> GPU 6,7
#   dag visualization    -> CPU
#
# Usage:
#   bash rebuttal/rebuttal.sh p0      # edge validation + semantic gap + dag viz
#   bash rebuttal/rebuttal.sh p1      # outcome+length baseline + evals (needs GPUs free)
#   bash rebuttal/rebuttal.sh all     # p0 then p1
###############################################################################
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
S=rebuttal/scripts
L=rebuttal/outputs/logs
mkdir -p "$L"
chmod +x "$S"/run_*.sh 2>/dev/null || true

PHASE="${1:-p0}"

run_p0() {
  echo "[rebuttal] launching P0 generation jobs (GPU 4,5 + 6,7)"
  CUDA_VISIBLE_DEVICES=4,5 nohup bash "$S/run_edge_validation.sh" > "$L/run_edge_validation.log" 2>&1 &
  echo "  edge_validation PID $!"
  CUDA_VISIBLE_DEVICES=6,7 nohup bash "$S/run_semantic_gap.sh"    > "$L/run_semantic_gap.log" 2>&1 &
  echo "  semantic_gap PID $!"
  # DAG visualization is CPU-only and can run immediately
  nohup bash "$S/run_dag_visualization.sh" > "$L/run_dag_visualization.log" 2>&1 &
  echo "  dag_visualization PID $!"
}

run_p1() {
  echo "[rebuttal] launching P1 training + eval jobs (GPU 4,5,6,7)"
  CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash "$S/run_outcome_length_baseline.sh" > "$L/run_outcome_length_baseline.log" 2>&1 &
  echo "  outcome_length_baseline PID $!"
}

run_evals() {
  echo "[rebuttal] launching eval jobs (GPU 4,5 + 6,7)"
  CUDA_VISIBLE_DEVICES=4,5 nohup bash "$S/run_topoprm_repro.sh" > "$L/run_topoprm_repro.log" 2>&1 &
  echo "  topoprm_repro PID $!"
  CUDA_VISIBLE_DEVICES=6,7 nohup bash "$S/run_nonqwen_eval.sh"  > "$L/run_nonqwen_eval.log" 2>&1 &
  echo "  nonqwen_eval PID $!"
}

case "$PHASE" in
  p0)    run_p0 ;;
  p1)    run_p1 ;;
  evals) run_evals ;;
  all)   run_p0; wait; run_p1; wait; run_evals ;;
  *)     echo "usage: $0 {p0|p1|evals|all}"; exit 1 ;;
esac

echo "[rebuttal] launched phase=$PHASE. Logs in $L/"
jobs -l
