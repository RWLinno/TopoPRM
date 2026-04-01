# TopoPRM 终版宣讲稿（5-8 分钟）

## 0. 开场（20s）

我们这项工作聚焦两件事：
1) 如何把过程监督做成确定性、可审计、可复现；
2) 如何把这种能力高效迁移到更小模型。

## 1. 问题背景（60s）

Outcome-only RLVR 能提升最终答案准确率，但对中间推理链约束弱。模型可能出现：
- 局部看起来合理、全局依赖紊乱；
- 结构冗余，推理过长；
- 稀疏奖励导致过程质量难优化。

learned PRM 可以给过程监督，但成本高、可靠性需要额外校准。

## 2. 贡献 A：Deterministic Verifiable PRM（120s）

核心做法：
- 从自由推理 trace 提取 dependency DAG；
- 计算两个互补奖励：
  - topology reward：全局依赖规则性；
  - continuity reward：局部支持可追溯性。

这里的“verifiable”是操作性定义：
- 一旦生成 trace，奖励由确定性程序计算；
- 可复现、可审计、可程序化检查；
- 不宣称语义完备证明验证。

同时用 correctness-first shaping 保证：正确性始终是主优化目标。

## 3. 贡献 B：Reverse-KL Distillation（120s）

Teacher 先用稀疏但可验证奖励训练，然后：
- 做 process-aware filtering，保留高质量轨迹；
- 用 reverse KL 训练 student。

为什么 reverse KL：
- mode-seeking，更偏向集中高质量模式；
- 适合把 teacher 的图增强推理行为压缩成 student 的紧凑链式推理。

## 4. 实验与价值（60s）

结果上我们关注三类收益：
- 准确率（任务性能）；
- 结构连贯性（过程质量）；
- 推理成本（压缩效率）。

蒸馏后 student 在显著降低成本的同时，保留 teacher 大部分收益。

## 5. 边界与下一步（40s）

边界：
- 提取图不是 ground-truth proof graph；
- 可验证奖励不等于语义完备验证。

下一步：
- 做 extractor fidelity 分析；
- 做 reverse-KL 下 mode collapse 风险控制与对比。
