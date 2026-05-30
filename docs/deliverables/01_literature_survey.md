# 大模型数学推理后训练：文献调研与技术路线分析

> 本文档整理了大语言模型数学推理能力提升相关的核心文献，覆盖强化学习优化、过程奖励建模、推理结构化表示、在线策略蒸馏、推理链压缩五大方向，共计 70+ 篇论文。适合作为该领域的入门调研材料和技术选型参考。

---

## 1. 强化学习与可验证奖励 (RLVR)

### 1.1 核心方法

| 论文 | 作者 | 年份/会议 | 核心贡献 |
|------|------|-----------|----------|
| DeepSeek-R1 | DeepSeek-AI | 2025 | 首次证明纯 RL（无 SFT）可激发 LLM 推理能力，提出 GRPO 的大规模应用范式 |
| DeepSeekMath (GRPO) | Shao et al. | 2024 | 提出 Group Relative Policy Optimization，用组内相对优势替代 critic，降低训练成本 |
| DAPO | Yu et al. | 2025 | 开源大规模 LLM RL 系统，引入动态采样和裁剪策略，解决 GRPO 的 reward hacking |
| ProRL | Liu et al. | 2025, NeurIPS | 证明延长 RL 训练（1000+ steps）可持续扩展推理边界 |
| VinePPO | Kazemnejad et al. | 2025, ICML | 用 Monte Carlo 树搜索细化 credit assignment，解决稀疏奖励问题 |
| REINFORCE++ | Hu et al. | 2025 | 简化 PPO/GRPO 为单步 REINFORCE + baseline，证明复杂算法未必优于简单基线 |
| Tulu 3 | Lambert et al. | 2024 | 开源后训练全流程（SFT→DPO→RL），提供可复现的训练 recipe |
| SimpleRL-Zoo | Zeng et al. | 2025 | 系统研究零 RL（无 SFT 直接 RL）在开源模型上的效果 |

### 1.2 关键技术细节

**GRPO 的核心公式：**
```
A_i = (R_i - mean(R_group)) / std(R_group)
L = -E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)] + β * KL(π_θ || π_ref)
```

**DAPO 的改进：**
- Dynamic sampling：根据 reward 分布动态调整采样温度
- Clip-higher：对高 advantage 样本使用更宽的 clip 范围
- Token-level KL：替代 sequence-level KL，更精细的正则化

**关键观察：**
- RL 训练的 reward 曲线通常在 100-200 步内快速上升，之后进入平台期
- `num_generations` 越大，advantage 估计越准确，但训练越慢
- `temperature` 对 exploration-exploitation 平衡至关重要（0.7-1.0 为常用范围）
- `beta`（KL 系数）过大会限制模型偏离 reference policy，过小会导致 reward hacking

### 1.3 其他相关工作

| 论文 | 核心贡献 |
|------|----------|
| Negative Reinforcement (Zhu et al. 2025) | 发现负样本的惩罚信号比正样本的奖励信号更有效 |
| SFT Memorizes, RL Generalizes (Chu et al. 2025) | 实证证明 SFT 倾向记忆训练分布，RL 能泛化到 OOD |
| Entropy Regularization (Cui et al. 2025) | 用熵正则化防止 RL 训练中的 mode collapse |
| Sycophancy to Subterfuge (Denison et al. 2024) | 揭示 reward tampering 风险，强调 reward 设计的鲁棒性 |

---

## 2. 过程奖励模型 (PRM)

### 2.1 发展脉络

```
人工标注 PRM (2023) → 自动化标注 (2024) → 隐式 PRM (2025) → 生成式 PRM (2026)
```

### 2.2 核心论文

| 论文 | 作者 | 年份 | 核心贡献 | 局限性 |
|------|------|------|----------|--------|
| Let's Verify Step by Step | Lightman et al. | 2023 | 首次大规模人工标注步骤级 reward | 标注成本极高（800K 步标注） |
| OmegaPRM | Luo et al. | 2024 | 用 Monte Carlo 树搜索自动生成步骤标注 | 需要大量 rollout 计算 |
| Implicit PRM | Yuan et al. | 2025, ICML | 从 outcome feedback 推导步骤级信号，无需步骤标注 | 信号质量依赖 outcome 的密度 |
| ThinkPRM | Khalifa et al. | 2025 | PRM 自身也进行推理（"thinking PRM"） | 推理开销大 |
| GenPRM | Zhao et al. | 2026, AAAI | 用生成式推理替代判别式打分，支持 test-time scaling | 需要训练专门的生成式验证器 |
| R-PRM | She et al. | 2025, EMNLP | 推理驱动的 PRM，显式建模验证推理过程 | 训练复杂度高 |
| Qwen2.5-Math-PRM | Zhang et al. | 2025 | 总结 PRM 开发经验：数据质量 > 模型规模 | — |
| PRMBench | Song et al. | 2025, ACL | 细粒度 PRM 评测基准，揭示现有 PRM 的系统性弱点 | — |

### 2.3 PRM 的核心挑战

1. **标注成本**：人工标注每步正确性极其昂贵
2. **步骤粒度**：什么算"一步"没有统一定义
3. **跨步依赖**：现有 PRM 大多是 step-local 的，忽略步骤间的依赖关系
4. **Reward hacking**：模型可能学会产生"看起来正确"但实际错误的步骤
5. **泛化性**：在一个数据集上训练的 PRM 难以泛化到其他领域

### 2.4 替代方案：确定性过程奖励

与学习型 PRM 不同，确定性方法直接从推理结构中计算奖励信号：
- **优势**：零标注成本、完全可解释、无 reward hacking 风险
- **劣势**：依赖规则质量、对非结构化推理效果有限
- **适用场景**：数学推理（步骤结构明确）、代码生成（可执行验证）

---

## 3. 推理结构化表示

### 3.1 从线性到图结构

| 论文 | 表示形式 | 核心思想 | 应用场景 |
|------|----------|----------|----------|
| Chain-of-Thought (Wei et al. 2022) | 线性链 | 逐步推理提升准确率 | 通用推理 |
| Tree of Thoughts (Yao et al. 2023) | 树 | 搜索多个推理路径 | 规划/搜索 |
| Graph of Thoughts (Besta et al. 2024) | 图 | 允许分支合并和回溯 | 复杂问题 |
| Chains to DAGs (Zhong et al. 2026) | DAG | 证明 LLM 内部推理是 DAG 结构 | 可解释性 |
| PASC-GRPO (Liu et al. 2026) | 拓扑图 | 用图结构指导 RL 训练 | 数学推理 |

### 3.2 关键洞察

**推理链不是线性的：**
- 数学推理中经常存在：跨步依赖、分支合并、公式复用、中间结论复用
- CoT 只是 DAG 的一种线性序列化（topological sort）
- 恢复隐式 DAG 结构可以提供更密集的监督信号

**DAG 提取方法：**
1. **规则方法**：正则匹配表达式/变量引用，构建依赖边（快速、确定性）
2. **LLM 方法**：让 LLM 标注步骤间依赖关系（准确但昂贵）
3. **混合方法**：规则提取 + LLM 辅助验证（平衡成本和质量）

### 3.3 结构化奖励的设计原则

1. **Correctness-first**：最终答案正确性必须是主导信号
2. **结构作为增益**：结构信号只能在正确答案内部做 re-ranking，不能翻转正确/错误的排序
3. **归一化**：结构信号需要 batch-level 归一化，避免噪声主导 advantage
4. **Anti-collapse**：当所有样本结构分相同时，需要 dead-zone 机制避免噪声放大

---

## 4. 在线策略蒸馏 (OPD)

### 4.1 方法演进

```
离线蒸馏 (Hinton 2015) → 在线蒸馏 (GKD 2024) → 推理压缩 (OPSDC 2026)
```

### 4.2 核心论文

| 论文 | 作者 | 年份 | 核心贡献 |
|------|------|------|----------|
| Knowledge Distillation | Hinton et al. | 2015 | 奠基性工作：soft label 蒸馏 |
| MiniLLM | Gu et al. | 2024, ICLR | 用 reverse KL 替代 forward KL，更适合 LLM |
| GKD | Agarwal et al. | 2024, ICLR | 在线策略蒸馏：student 自己生成，teacher 监督 |
| BOND | Sessa et al. | 2024 | Best-of-N 蒸馏：用 rejection sampling 选最优 |
| KDRL | Xu et al. | 2025 | 统一 KD 和 RL：同时优化蒸馏和奖励目标 |
| GAD | Ye et al. | 2025 | 黑盒蒸馏：不需要 teacher logits |
| OPSDC | Sang et al. | 2026 | 推理压缩专用的在线自蒸馏 |
| ExOPD | Yang et al. | 2026 | 超越 teacher：用 reward extrapolation 突破 teacher 上界 |
| OPD Survey | Song et al. | 2026 | 系统综述 OPD 方法 |

### 4.3 OPD 的关键设计选择

| 设计维度 | 选项 | 权衡 |
|----------|------|------|
| 采样策略 | Student on-policy vs Teacher off-policy | On-policy 减少 train-test mismatch |
| 损失函数 | Forward KL vs Reverse KL vs JSD | Reverse KL 更适合 mode-seeking |
| Teacher 信号 | Token-level logits vs Sequence-level reward | Token-level 更密集但计算量大 |
| 条件化 | 无条件 vs 结构条件化 | 条件化可以聚焦结构缺陷 |

---

## 5. 推理链压缩

### 5.1 核心方法

| 论文 | 方法 | 压缩率 | 精度保持 |
|------|------|--------|----------|
| TokenSkip (Xia et al. 2025) | Token 级剪枝 | 30-50% | 95%+ |
| O1-Pruner (Luo et al. 2025) | 推理步骤剪枝 | 40-60% | 90%+ |
| S1 (Muennighoff et al. 2025) | Budget-aware 生成 | 可控 | 随 budget 变化 |
| L1 (Aggarwal et al. 2025) | 长度正则化 | 20-40% | 95%+ |
| Training for Shorter (Arora et al. 2026) | RL + 长度惩罚 | 30-50% | 93%+ |

### 5.2 压缩与结构的关系

关键洞察：**有效的压缩应该保留 DAG 的关键路径，而非随机删除 token。**

- 冗余分支：DAG 中不通向最终结论的分支可以安全删除
- 重复推导：DAG 中被多次引用的中间结论只需保留一次
- 缩点缩环：DAG 中的强连通分量可以合并为单个节点

---

## 6. 技术路线总结

### 6.1 各方向的互补关系

```
RLVR (训练框架)
  ├── PRM (奖励信号来源)
  │     ├── 学习型 PRM (高质量但昂贵)
  │     └── 确定性 PRM (零成本但依赖规则)
  ├── 结构化表示 (信号载体)
  │     └── DAG 提取 → 拓扑奖励 + 连续性奖励
  └── 蒸馏 (效率优化)
        └── OPD + 结构条件化 → 保留依赖结构的压缩
```

### 6.2 开放问题

1. **Competition math 的瓶颈**：极难题（AIME/IMO）需要创造性推理，结构化方法的提升有限
2. **Reward collapse**：RL 训练中 reward 方差趋零的问题仍未完全解决
3. **跨领域泛化**：数学推理的结构化方法能否迁移到代码、科学推理等领域
4. **Scaling law**：结构化奖励的收益是否随模型规模增大而变化
5. **在线 DAG 提取的效率**：如何在不增加训练时间的前提下提取高质量 DAG

### 6.3 推荐阅读路径

**入门（5 篇）：**
1. Wei et al. 2022 — Chain-of-Thought 原始论文
2. Shao et al. 2024 — GRPO 方法
3. Lightman et al. 2023 — PRM 奠基
4. Agarwal et al. 2024 — GKD (OPD)
5. Besta et al. 2024 — Graph of Thoughts

**进阶（5 篇）：**
1. DeepSeek-R1 2025 — 大规模 RL 实践
2. Yuan et al. 2025 — Implicit PRM
3. Zhong et al. 2026 — Chains to DAGs
4. Liu et al. 2026 — PASC-GRPO
5. Song et al. 2026 — OPD Survey

---

## 参考资源

- [awesome-RLVR](https://github.com/opendilab/awesome-RLVR) — RLVR 论文列表
- [Awesome-Process-Reward-Models](https://github.com/RyanLiu112/Awesome-Process-Reward-Models) — PRM 论文列表
- [awesome-on-policy-distillation](https://github.com/chrisliu298/awesome-on-policy-distillation) — OPD 论文列表
