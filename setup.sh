#!/bin/bash
# TopoPRM environment setup + data download from HuggingFace
set -euo pipefail

HF_CKPT_REPO="rwlinno/topoprm-ckpts"
HF_DATA_REPO="rwlinno/topoprm-data"

echo "=========================================="
echo "TopoPRM Setup"
echo "=========================================="

# 1. Conda environment
if ! conda env list | grep -q "topoprm"; then
    echo "[1/4] Creating conda env..."
    conda create -n topoprm python=3.12 -y
else
    echo "[1/4] Conda env 'topoprm' already exists"
fi

echo "[2/4] Installing dependencies..."
eval "$(conda shell.bash hook)"
conda activate topoprm
pip install -r requirements.txt --quiet

# 3. Download data from HuggingFace
echo "[3/4] Downloading data from HuggingFace..."
mkdir -p data/grpo_ready data/dag_public data/benchmarks output/eval

if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "$HF_DATA_REPO" --repo-type dataset --local-dir data/hf_download 2>/dev/null || echo "  WARN: HF data download failed (may need 'huggingface-cli login')"
    # Move files to expected locations
    [ -f data/hf_download/train_public.jsonl ] && cp data/hf_download/train_public.jsonl data/grpo_ready/
    [ -d data/hf_download/eval ] && cp -r data/hf_download/eval/* output/eval/ 2>/dev/null || true
    echo "  Data downloaded to data/"
else
    echo "  WARN: huggingface-cli not found. Install with: pip install huggingface_hub"
    echo "  Then run: huggingface-cli download $HF_DATA_REPO --repo-type dataset --local-dir data/hf_download"
fi

# 4. Download checkpoints
echo "[4/4] Downloading checkpoints from HuggingFace..."
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "$HF_CKPT_REPO" --local-dir output/hf_ckpts 2>/dev/null || echo "  WARN: HF checkpoint download failed"
    echo "  Checkpoints downloaded to output/hf_ckpts/"
else
    echo "  WARN: huggingface-cli not found. Checkpoints not downloaded."
fi

# 5. Verify
echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="
python -c "from src.reward.topo_reward import TopoReward; print('  [OK] TopoReward importable')"
python -c "from src.dag.graph import ReasoningDAG; print('  [OK] ReasoningDAG importable')"
python -c "from src.reward.composite_reward import TopoHierarchicalReward; print('  [OK] TopoHierarchicalReward importable')"
python -c "import trl; print(f'  [OK] trl {trl.__version__}')"
python -c "import torch; print(f'  [OK] torch {torch.__version__}')"

echo ""
echo "Setup complete! Next steps:"
echo "  1. Read docs/HANDOFF.md for experiment continuation plan"
echo "  2. Run: python3 scripts/build_dag_public.py  (if DAG data not downloaded)"
echo "  3. Run: python3 scripts/train_grpo.py --sft_adapter output/sft_deepseek_r1_7b/final"
echo ""
