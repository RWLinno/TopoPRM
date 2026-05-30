# TopoPRM 数据分析与处理

## 1. 公有数据概览

### 1.1 评测 Benchmark

| Benchmark | 来源 | 规模 | 难度级别 | 答案格式 | 评测方式 |
|-----------|------|------|----------|----------|----------|
| GSM8K | Cobbe et al. 2021 | 1,319 题 | 小学数学 | 数值 (####) | 精确匹配 |
| MATH-500 | Hendrycks et al. 2021 | 500 题 | 高中-竞赛 | LaTeX (\boxed{}) | math_verify |
| OlympiadBench | MATH level-5 | 1,263 题 (cap 500) | 竞赛 | LaTeX | math_verify |
| Omni-MATH | MATH level-4+5 | 2,432 题 (cap 500) | 竞赛 | LaTeX | math_verify |
| AIME 2024 | AMC 竞赛 | 30 题 | 高难度竞赛 | 整数 (0-999) | 精确匹配 |
| AIME 2025 | AMC 竞赛 | 30 题 | 高难度竞赛 | 整数 (0-999) | 精确匹配 |
| CNMO 2024 | 中国数学奥林匹克 | 83 题 | 高难度竞赛 | 数值/表达式 | math_verify |
| MMLU | Hendrycks et al. 2021 | 14,042 题 (cap 1500) | 通用知识 | 多选 (A/B/C/D) | 字母匹配 |
| GPQA-Diamond | — | 198 题 | 研究生级 | 多选 | 字母匹配 |

### 1.2 难度分布

MATH 数据集按难度分级：
- Level 1: 429 题（最简单）
- Level 2: 860 题
- Level 3: 1,098 题
- Level 4: 1,169 题
- Level 5: 1,263 题（最难，用作 OlympiadBench proxy）

## 2. 训练数据构建

### 2.1 数据来源

训练数据 `data/grpo_ready/train_public.jsonl` 包含 19,472 条记录，来源：
- GSM8K 训练集
- MATH 训练集
- 部分 Olympiad/AIME 历史题

### 2.2 数据格式

每条记录包含：
```json
{
  "record_id": "gsm8k_0",
  "question": "Natalia sold clips to 48 of her friends...",
  "standard_answer": "Natalia sold 48/2 = 24 clips in May...",
  "final_answer": "72",
  "source": "gsm8k",
  "reference_dag": {
    "problem_id": "gsm8k_0",
    "nodes": [...],
    "edges": [...]
  }
}
```

### 2.3 DAG 提取流程

```
原始推理 trace → Step 分割 → 表达式/声明提取 → 依赖边构建 → DAG 验证
```

1. **Step 分割**: 基于 "Step X:", "(1)", 换行等标记切分
2. **表达式提取**: 正则匹配数学表达式（数字、分数、方程）
3. **声明提取**: 提取关键结论性语句
4. **依赖边构建**:
   - 表达式复用 → virtual edge (权重 1.0)
   - 声明引用 → claim_ref edge (权重 1.0)
   - 顺序相邻 → solid edge (权重 0.3)
   - 无显式依赖 → double_barrier edge (权重 0.5)
5. **DAG 验证**: 检查有效性、无环性、方向一致性

### 2.4 ms-swift 兼容格式

训练时使用 `train_public_swift.jsonl`（简化格式）：
```json
{
  "query": "Natalia sold clips to 48 of her friends...",
  "solution": "72"
}
```

注意：`solution` 字段必须是简洁的最终答案（不是完整解题过程），否则 OutcomeReward 无法正确匹配。

## 3. 私有数据

### 3.1 中国初高中教师批改数据

| 数据集 | 规模 | 来源 | 特点 |
|--------|------|------|------|
| 初中数学 | 2,181 条 | 教师批改平台 | 步级批注 + 评分 |
| 高中数学 | 5,414 条 | 教师批改平台 | 步级批注 + 评分 |

**数据特性：**
- 天然提供 PRM 监督信号（教师打分 + 文字批注）
- JSON 输出协议：`{"学生得分": X, "结论批改": "...", "批改过程": "..."}`
- 用于 in-domain validation 和 reward 函数开发

### 3.2 私有数据的作用

- 验证 TopoPRM reward 在真实批改场景的有效性
- 提供 format compliance 的训练信号
- 用于 ablation 实验的 in-domain 验证集

## 4. DAG 提取质量分析

### 4.1 结构指标

| Benchmark | Valid DAG% | Acyclic% | No-Orphan% | Direction% | Step-Align% |
|-----------|-----------|----------|------------|------------|-------------|
| GSM8K | 98.2% | 99.1% | 87.3% | 92.5% | 95.8% |
| MATH-500 | 95.7% | 98.4% | 82.1% | 89.7% | 91.2% |
| AIME | 91.3% | 97.2% | 75.6% | 85.4% | 87.9% |
| Olympiad | 93.1% | 97.8% | 78.9% | 87.2% | 89.5% |

### 4.2 观察

- GSM8K（简单题）的 DAG 质量最高：步骤清晰、依赖明确
- AIME（竞赛题）的 orphan 率最高：长链推理中间步骤缺乏显式引用
- 这解释了为什么 TopoPRM 在 GSM8K/MATH 上提升大，在 AIME 上提升有限

## 5. 数据分布与采样策略

### 5.1 训练数据分布

| 来源 | 数量 | 占比 | 平均 tokens |
|------|------|------|-------------|
| GSM8K | 7,473 | 38.4% | ~200 |
| MATH | 7,500 | 38.5% | ~400 |
| Olympiad | 3,000 | 15.4% | ~600 |
| AIME/CNMO | 1,499 | 7.7% | ~800 |

### 5.2 采样策略

GRPO 训练时：
- `num_generations=4`: 每个 prompt 生成 4 个 completion
- `temperature=0.8`: 适度随机性
- `max_completion_length=4096`: 允许长推理
- `gradient_accumulation_steps=16`: 有效 batch size = 16×4 = 64 sequences

### 5.3 数据质量控制

- reference_dag 字段提供 gold DAG 结构（用于 topo reward 的 ref_edge_f1 计算）
- 无 reference_dag 时，topo reward 退化为纯结构质量评分（不含 edge precision/recall）
- 训练数据中 100% 有 reference_dag；评测时不使用 reference_dag
