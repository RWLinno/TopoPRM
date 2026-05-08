#!/usr/bin/env bash
# One-stop sync: regenerate CSV/LaTeX/progress from current output/eval state.
# Run this after each major batch of v3b results lands.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/mnt/users/conda_env/topoprm/bin:$PATH"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=== [1/3] fill_rft_csv (rft_ours.csv + rft_bestof_ours.csv) ==="
python3 scripts/fill_rft_csv.py \
    --eval_dir output/eval \
    --output docs/rft_ours.csv \
    --bestof_output docs/rft_bestof_ours.csv \
    --only_with_data

echo
echo "=== [2/3] collect_experiment_results ==="
python3 -m src.eval.collect_experiment_results \
    --output_dir output/analysis \
    --eval_dir output/eval

echo
echo "=== [3/3] sync_paper_tables ==="
python3 -m src.eval.sync_paper_tables \
    --summary output/analysis/experiment_summary.json \
    --paper_dir topoprm_paper \
    --eval_dir output/eval \
    --progress_file output/analysis/experiment_progress.md

echo
echo "Done. Review:"
echo "  docs/rft_ours.csv"
echo "  docs/rft_bestof_ours.csv"
echo "  output/analysis/experiment_summary.csv"
echo "  topoprm_paper/tables/public_results.tex (AUTO_SYNC block)"
