# TopoPRM 完整实验复现指南

本文档提供从零开始复现 TopoPRM 所有实验结果的完整步骤。

## 前置条件

```bash
conda create -n topoprm python=3.12 -y && conda activate topoprm
pip install -r requirements.txt
export HF_MODELS_DIR=/path/to/your/hf_models
export TOPOPRM_PYTHON=python3
python3 -c "from src.reward.topo_reward import TopoReward; print('OK')"
```

## Phase 0: 数据准备

```bash
bash scripts/run_data_pipeline.sh
bash scripts/download_benchmarks.sh
python3 scripts/check_reward_invariants.py
```

## Phase 1: SFT 冷启动

```bash
bash scripts/run_sft_config.sh sft_deepseek_r1_7b
bash scripts/run_sft_config.sh sft_qwen35_9b
```

## Phase 2: GRPO + TopoPRM 训练

```bash
# TopoPRM Full (hierarchical)
python3 scripts/train_grpo_ablation.py --reward hierarchical \
    --output_dir output/grpo_topoprm_hierarchical --eval_every 20 --eval_size 32

# Outcome-only (ablation baseline)
python3 scripts/train_grpo_ablation.py --reward outcome_only \
    --output_dir output/grpo_outcome_only --eval_every 20 --eval_size 32

# No-topo ablation
python3 scripts/train_grpo_ablation.py --reward no_topo \
    --output_dir output/grpo_no_topo --eval_every 20 --eval_size 32

# No-continuity ablation
python3 scripts/train_grpo_ablation.py --reward no_continuity \
    --output_dir output/grpo_no_continuity --eval_every 20 --eval_size 32
```

## Phase 3: OPD 蒸馏

```bash
bash scripts/run_swift_rlhf.sh opd_topoprm_dr1_7b_stage3_v2 gkd
bash scripts/run_swift_rlhf.sh opd_topoprm_qwen35_9b_stage3_v2 gkd
```

## Phase 4: 评估

```bash
# 统一 9-Benchmark 评估
bash scripts/run_unified_eval.sh \
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    output/grpo_topoprm_hierarchical/final topoprm_hier

# 填充论文表格
python3 scripts/fill_paper_table.py --label topoprm_hier --row "TopoPRM" --write
```

## Phase 5: 可视化

```bash
python3 tutorials/training_curve.py
python3 tutorials/render_dag_cases.py --from-rollout output/eval/
python3 tutorials/efficiency_barplot.py
```

## 关键超参数

| 参数 | 值 |
|------|-----|
| LoRA r | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Target modules | q/k/v/o/gate/up/down_proj |
| Max context length | 4096 |
| Reward alpha | 0.5 |
