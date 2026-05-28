#!/bin/bash
# ============================================================================
# submit_aliyun_slurm.sbatch
#
# Submitter for an Aliyun PAI / Lingjun Slurm cluster.
# Each Slurm array task evaluates one manifest row on one GPU.
#
# Usage:
#   sbatch --array=0-5 \
#          --gres=gpu:1 \
#          --cpus-per-task=8 \
#          --mem=64G \
#          -t 12:00:00 \
#          -o logs/dist/slurm_%A_%a.log \
#          scripts/dist/submit_aliyun_slurm.sbatch
#
# Notes:
#   - --array indices MUST be a subset of TASK_IDs in
#     configs/dist/eval_manifest.tsv.
#   - The cluster image must already have python + transformers installed at
#     the expected path; otherwise set CONDA_ENV / PYTHON_BIN env vars on the
#     sbatch command line.
# ============================================================================
#SBATCH --job-name=topoprm_eval
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
cd "$REPO_ROOT"

# Optional conda activation (Aliyun Lingjun images usually ship miniconda).
if [[ -n "${CONDA_ENV:-}" && -f "${CONDA_PREFIX:-/opt/conda}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_PREFIX:-/opt/conda}/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# Slurm hands the manifest row index to us via SLURM_ARRAY_TASK_ID; the worker
# script also reads it directly, but exporting TASK_ID makes the flow explicit.
export TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
export NUM_GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-1}"

echo "[slurm] job=$SLURM_JOB_ID array=$SLURM_ARRAY_TASK_ID host=$(hostname) gpus=$NUM_GPUS_PER_NODE"

bash scripts/dist/run_eval_worker.sh "$TASK_ID"
