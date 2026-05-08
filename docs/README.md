# TopoPRM 文档导航

> Last refreshed: **2026-05-07 17:50 CST** — Phase 2 完成；Phase 3a SFT 训练进行中。

## 当前实验状态

### Phase 1: DAG 数据构建 -- 完成

- 19,472 条公开数学题 DAG（GSM8K + MATH 训练集），100% valid, 100% acyclic, 40.9% virtual edges
- 产物：`data/grpo_ready/train_public.jsonl`

### Phase 2: DeepSeek-R1-Distill-Qwen-7B 基线评测 -- 完成

使用 `--use_chat_template`（label: `baseline_dr1_7b_chat`）：

| Benchmark | pass@1 | pass@5 | maj@5 | prm@5 | avg_tok | 模型卡 |
|-----------|--------|--------|-------|-------|---------|--------|
| GSM8K     | **83.5%** | 90.8% | 82.6% | 83.5% | 599  | ~95%   |
| MATH-500  | **55.6%** | 62.8% | 58.8% | 55.6% | 3412 | 92.8%  |
| AIME 2024 | **23.3%** | 33.3% | 30.0% | 23.3% | 4096 | 55.5%  |
| CNMO 2024 | **23.3%** | 40.0% | 30.0% | 23.3% | 4094 | --     |
| MMLU      | **42.5%** | --    | --    | --    | 511  | ~70%   |

注意事项：
- MATH-500 绝对数字低于模型卡，主因：answer extraction 不支持 LaTeX 复杂答案（如 `\frac{14}{3}`），实际模型能力被低估
- AIME/CNMO avg_tok≈4096 = 触顶截断，导致答案被截
- prm@5 = pass@1 是预期行为（baseline 无 TopoPRM 训练，PRM scorer 给出 0 分）
- 关键对比：**同一协议下** TopoPRM 训练后 vs baseline 的提升幅度

### Phase 3a: SFT 训练 -- 进行中

- 脚本：`scripts/train_sft.py`（trl SFTTrainer + LoRA）
- 数据：`data/grpo_ready/train_public.jsonl`（19,472 条）
- 模型：DeepSeek-R1-Distill-Qwen-7B + LoRA (r=64, α=128)
- 进度：~11/3651 steps，预计约 3 小时完成
- 日志：`logs/phase3_sft.log`
- 产物将在：`output/sft_deepseek_r1_7b/final/`

### Phase 3b: GRPO 训练 -- 待 SFT 完成

- 脚本：`scripts/train_grpo.py`（trl GRPOTrainer + TopoPRM hierarchical reward）
- 加载 SFT adapter → GRPO 200 steps

### Phase 4/5: 后续评测与论文同步 -- 待 Phase 3

## 论文当前状态

| 文件 | 状态 |
|------|------|
| `tables/public_results.tex` | 12 列表（p@1/m@5/prm@5），baseline 行已填，其余 `--` 等训练后数据 |
| `sections/4_experiments.tex` | Setup/Benchmarks/Baselines 已重写为 DR1-7B 主线 |
| `tables/dag_structural_public.tex` | TBD 模板，等 Phase 3 产物 |
| `tables/dag_scaling.tex` | TBD 模板 |

## 管线命令

```bash
# Phase 3a: SFT（当前正在运行）
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_sft.py

# Phase 3b: GRPO（SFT 完成后）
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_grpo.py --sft_adapter output/sft_deepseek_r1_7b/final

# Phase 4: 评测训练后模型
# (同 bench_transformers.py，加 --adapter 和 --sft_style)
```

## 关键文件

### 训练脚本（新，替代无法响应的 swift CLI）
- `scripts/train_sft.py` — trl SFTTrainer + LoRA
- `scripts/train_grpo.py` — trl GRPOTrainer + TopoPRM reward
- `scripts/bench_transformers.py` — 统一评测

### 配置（参考，实际参数已内嵌脚本）
- `configs/sft_deepseek_r1_7b.yaml`
- `configs/grpo_topoprm_deepseek_r1_7b.yaml`

### 核心方法文档
- [dag_schema.md](dag_schema.md) — DAG 节点/边/虚边定义
- [reward_design.md](reward_design.md) — 五分量奖励设计
- [training_pipeline.md](training_pipeline.md) — SFT→GRPO→TVSD 三阶段

## 归档
- `archive/2026-03/` — 早期方案文档
- `archive/2026-04-early/` — 4 月上半月日志
- `archive/2026-04-late/` — 4 月下旬不达标结果
