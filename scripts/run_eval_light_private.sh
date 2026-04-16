#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash scripts/run_eval_light_private.sh <model_path> <adapter_or_none> <output_name> [middle_jsonl] [high_jsonl]

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PATH="/mnt/users/conda_env/topoprm/bin:$PATH"

MODEL="${1:?need model_path}"
ADAPTER="${2:-none}"
NAME="${3:?need output_name}"
MIDDLE="${4:-data/test/light_middle_200.jsonl}"
HIGH="${5:-data/test/light_high_200.jsonl}"

mkdir -p output/eval

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
COMMON_ARGS=(--model "$MODEL" --max_new_tokens "$MAX_NEW_TOKENS" --temperature 0.1)
if [ "$ENABLE_THINKING" = "false" ]; then
  COMMON_ARGS+=(--enable_thinking false)
fi
if [ "$ADAPTER" != "none" ]; then
  COMMON_ARGS+=(--adapters "$ADAPTER")
fi

echo "[light_eval] name=$NAME model=$MODEL adapter=$ADAPTER"
echo "[light_eval] middle=$MIDDLE high=$HIGH"

# Clean stale results to prevent append-contamination across reruns
for stale in "output/eval/${NAME}_middle.jsonl" "output/eval/${NAME}_high.jsonl" \
             "output/eval/${NAME}_middle_metrics.json" "output/eval/${NAME}_high_metrics.json"; do
  [ -f "$stale" ] && rm -f "$stale" && echo "[light_eval] removed stale $stale"
done

swift infer "${COMMON_ARGS[@]}" \
  --val_dataset "$MIDDLE" \
  --result_path "output/eval/${NAME}_middle.jsonl" \
  2>&1 | tee "output/eval/${NAME}_middle.log"

swift infer "${COMMON_ARGS[@]}" \
  --val_dataset "$HIGH" \
  --result_path "output/eval/${NAME}_high.jsonl" \
  2>&1 | tee "output/eval/${NAME}_high.log"

python3 -m src.eval.critique_eval \
  --predictions "output/eval/${NAME}_middle.jsonl" \
  --ground_truth "$MIDDLE" \
  --output "output/eval/${NAME}_middle_metrics.json"

python3 -m src.eval.critique_eval \
  --predictions "output/eval/${NAME}_high.jsonl" \
  --ground_truth "$HIGH" \
  --output "output/eval/${NAME}_high_metrics.json"

echo "[light_eval] done: $NAME"
