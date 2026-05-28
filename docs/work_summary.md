# TopoPRM 工作总结

> 最后更新: 2026-05-28

## 项目概述

TopoPRM 是一个基于拓扑结构的过程奖励模型（Process Reward Model），用于提升大语言模型的数学推理能力。核心思想是从推理链中提取依赖 DAG（有向无环图），利用图结构信息设计 reward 信号，指导 GRPO 训练。

## 方法框架

### 三阶段训练流程

1. **Stage I — SFT**: 对齐格式和基础推理行为
2. **Stage II — GRPO + TopoPRM**: 注入拓扑感知的过程监督（hierarchical reward + ACE）
3. **Stage III — TG-OPD**: 拓扑引导的在线策略蒸馏

### 核心组件

- **TopoHierarchicalReward**: 乘法聚合的层次化 reward（outcome × format_gate × length_gate × topo_gain）
- **TopoSCAEReward (ACE)**: 分层裁剪优势估计，保证 correctness-first
- **OutcomeReward**: 数学答案匹配（支持 `\boxed{}` 提取 + math_verify 验证）
- **TopoReward**: DAG 结构质量评分（有效性、无环性、方向性、步骤对齐）
- **ContinuityReward**: 推理连续性评分

## 实验结果

### Qwen2.5-7B + TopoPRM (opd_stage3_ckpt200)

| Benchmark | pass@1 | pass@5 |
|-----------|--------|--------|
| MATH-500 | 66.8% | 72.6% |
| AIME'24 | 13.3% | 13.3% |
| AIME'25 | 10.0% | 26.7% |
| CNMO'24 | 47.0% | — |

### Qwen3.5-9B + TopoPRM v2 (from ckpt79 + 120 steps)

| Benchmark | pass@1 | pass@5 |
|-----------|--------|--------|
| AIME'24 | 3.3% | 6.7% |
| AIME'25 | 3.3% | 13.3% |
| Omni-MATH | 55.2% | 76.2% |
| CNMO'24 | 18.1% | — |

### DeepSeek-R1-7B + TopoPRM

| Benchmark | pass@1 |
|-----------|--------|
| GSM8K | 62.0% |
| MATH-500 | 66.0% |
| OlympiadBench | 58.0% |
| Omni-MATH | 54.0% |
| MMLU | 84.0% |
| GPQA-D | 36.0% |
| CNMO'24 | 36.0% |

### Baselines

| Model | Benchmark | pass@1 |
|-------|-----------|--------|
| DR1-7B (chat) | AIME'24 | 46.7% |
| DR1-7B (chat) | Omni-MATH | 72.8% |
| DR1-7B (chat) | OlympiadBench | 57.8% |
| Qwen3.5-9B (base) | AIME'24 | 6.7% |
| Qwen3.5-9B (base) | Omni-MATH | 51.4% |
| Qwen3.5-2B | CNMO'24 | 36.1% |
| Qwen3.5-4B | GPQA-D | 29.8% |

## 训练配置

### 9B TopoPRM v2 (最终配置)

```yaml
model: Qwen/Qwen3.5-9B
adapters: grpo_hier_9b_ckpt79  # 从 GRPO baseline 继续训
reward_funcs: topo_hierarchical
num_generations: 2
temperature: 0.8
max_completion_length: 4096
learning_rate: 3.0e-6
gradient_accumulation_steps: 16
max_steps: 120
```

环境变量:
```bash
TOPO_SCAE_PRESERVE_OUTCOME=1
TOPO_HIER_AGG=multiplicative
TOPO_RESCALE_PATCH=1
TOPO_DAG_SENTENCE_FALLBACK=1
TOPO_LENGTH_UNIT=tokens
```

### Qwen2.5-7B TopoPRM (Stage 3 OPD)

- Checkpoint: `opd_qwen25_7b_stage3_ckpt200`
- Base: Qwen/Qwen2.5-7B-Instruct
- LoRA rank: 64, alpha: 128

## 关键发现与经验

1. **Correctness-first 至关重要**: multiplicative 聚合保证 outcome=0 时 reward=0，防止结构 bonus 提升错误答案
2. **SCAE bug 修复**: neg 组映射公式有反转 bug，修复后保证 min(correct) > max(wrong)
3. **max_completion_length 影响巨大**: 1024→4096 对 competition math 至关重要（50%+ 截断率降到 34%）
4. **从 GRPO checkpoint 继续训比从头训好得多**: 直接从 base model 训 TopoPRM reward 信号为负，从 ckpt79 继续训则稳定在 +0.4
5. **reference_dag 对 topo reward 有帮助**: 训练数据包含 DAG 时 topo reward 更准确
6. **Competition math (AIME/CNMO) 仍是难点**: TopoPRM 在 general reasoning (MMLU/GPQA) 上提升明显，但在极难竞赛题上提升有限

## 文件结构

```
TopoPRM/
├── src/reward/           # Reward 函数实现
│   ├── composite_reward.py   # TopoHierarchical, TopoSCAE, TopoComposite
│   ├── outcome_reward.py     # 答案匹配 (math_verify)
│   ├── format_reward.py      # 格式检查
│   ├── topo_reward.py        # DAG 结构评分
│   ├── continuity_reward.py  # 推理连续性
│   └── reward_config.py      # 环境变量配置
├── configs/              # 训练配置 YAML
├── scripts/              # 评测和训练脚本
├── data/grpo_ready/      # 训练数据
├── output/               # 训练产物和 checkpoints
├── results/              # 评测结果
├── tests/                # 单元测试
└── tutorials/            # 可视化和分析工具
```

## Checkpoints (HuggingFace)

| Name | Base Model | Description |
|------|-----------|-------------|
| `grpo_hier_9b_ckpt79` | Qwen3.5-9B | GRPO + topo_hierarchical, 79 steps from SFT |
| `grpo_9b_from_ckpt79_ckpt120` | Qwen3.5-9B | TopoPRM v2, 120 steps from ckpt79 |
| `grpo_topoprm_dr1_7b_ckpt949` | DR1-7B | TopoPRM + SCAE, 949 steps |
| `opd_qwen25_7b_stage3_ckpt200` | Qwen2.5-7B-Instruct | Stage 3 OPD, 200 steps |

## 复现步骤

```bash
# 1. 环境安装
pip install -r requirements.txt

# 2. 评测
CUDA_VISIBLE_DEVICES=0 python scripts/bench_transformers.py \
  --model /path/to/Qwen3.5-9B \
  --adapter output/hf_ckpts/grpo_hier_9b_ckpt79 \
  --label topoprm_9b \
  --benchmarks aime2024 math500 \
  --use_chat_template --num_samples_per_item 5

# 3. 训练 (GRPO + TopoPRM)
export TOPO_HIER_AGG=multiplicative
export TOPO_SCAE_PRESERVE_OUTCOME=1
swift rlhf configs/grpo_9b_from_ckpt79.yaml
```
