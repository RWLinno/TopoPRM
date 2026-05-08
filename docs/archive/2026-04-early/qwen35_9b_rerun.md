# Qwen3.5-9B Re-run Notes (TopoPRM v2)

## Goal
Re-run TopoPRM hierarchical GRPO with `Qwen3.5-9B` as the base model.

Model path:
- `/mnt/data/huggingface_downloads/models/qwen/Qwen3.5-9B`

## Launch Command
Use conda env `topoprm` and GPUs `0,1,2,3`:

```bash
source /mnt/users/miniconda3/etc/profile.d/conda.sh
conda activate topoprm
nohup env SKIP_PREFLIGHT=1 CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 SFT_ADAPTER=/tmp/no_adapter \
  bash scripts/run_grpo.sh grpo_hierarchical_qwen35_9b > ./logs/run_grpo_hierarchical_qwen35_9b.log 2>&1 &
```

## Config
- `configs/grpo_hierarchical_qwen35_9b.yaml`
- Reward: `topo_hierarchical`
- Output dir: `output/grpo_hierarchical_qwen35_9b`

## Runtime Fixes Applied
1. `transformers` did not recognize `qwen3_5`:
   - Installed latest source build:
   - `pip install -U "git+https://github.com/huggingface/transformers.git"`
2. Missing Qwen dependency package:
   - `pip install -U qwen_vl_utils decord`

## Logs
- Launcher log: `logs/run_grpo_hierarchical_qwen35_9b.log`
- Training stream log: `output/grpo_hierarchical_qwen35_9b_20260403_225216.log`

## Status
- Training workers are successfully spawned via `torch.distributed.run` on `0-3` GPUs.
- Keep this run in background; follow-up benchmark export should be done after checkpoint generation.
