# Proposal（Reframed）

## 研究问题

Outcome-only RLVR 可提升最终答案准确率，但对中间推理结构约束弱；learned PRM 成本高且稳定性不确定。我们需要可验证、确定性、插件式兼容的过程监督，同时需要把这种能力迁移给更小模型。

## 双主贡献

### 贡献 A：Deterministic Verifiable Process Reward Model

- trace -> dependency DAG
- topology reward（全局依赖规则性）
- continuity reward（局部支持可追溯性）
- correctness-first reward shaping

### 贡献 B：Reverse-KL Reasoning Distillation

- teacher 先通过可验证奖励训练
- process-aware trace filtering 选高质量轨迹
- reverse KL（mode-seeking）将图增强推理压缩为 student 紧凑链式推理

## 关键边界

- 提取图不是真实证明图
- 奖励可验证不等于语义验证完备
- 不宣称 step correctness guarantee
