# Deterministic Verifiable PRM 设计说明

## 1. 设计目标

我们希望在 outcome-only RLVR 之外，加入可复现、可审计的过程监督信号，避免模型仅靠结果对错进行稀疏优化。

本文的“可验证奖励（verifiable reward）”定义为：

- 给定模型输出轨迹，奖励可由确定性程序直接计算；
- 计算过程可复现、可审计、可检查；
- **不等价于**“语义上完备的 step correctness verification”。

---

## 2. 从 Trace 到 dependency DAG

输入：`<think>` 中自由形式推理文本。

处理流程：

1. Step segmentation（步骤切分）
2. Expression/claim mining（表达式与命题抽取）
3. Step typing（步骤类型规则分类）
4. Dependency building（依赖边构建）

输出：提取图 `G=(V,E)`，用于奖励计算。

> 注意：提取图是结构代理，不是 ground-truth proof graph。

---

## 3. 两类过程奖励

### 3.1 Topology reward（全局结构信号）

关注：

- 无环性
- 结论节点依赖支持情况
- 依赖方向一致性
- （可选）与参考依赖边覆盖率

作用：提供全局依赖规则性的可验证监督。

### 3.2 Continuity reward（局部可追溯信号）

关注：

- 当前步骤是否可追溯到前文表达式/命题
- 是否显式引用题设已知条件
- 是否存在明显跳步

作用：提供局部支持可追溯性的可验证监督。

---

## 4. 多源奖励聚合

组合信号：

- outcome reward（主目标）
- topology reward
- continuity reward
- format reward
- length reward

### 为什么不能朴素线性混合

朴素混合可能将“结构好但答案错”的样本排在“答案正确但结构一般”的样本前面，破坏主目标。

### 解决：correctness-first shaping

先按 outcome 正确性分层，再在层内利用过程奖励做细分排序，保证正确性优先。

---

## 5. 工程映射

- 兼容 `ms-swift` 的 reward plugin 接口
- 保留旧接口名 + 新术语别名，支持平滑迁移
- 关键实现入口：
  - `src/reward/composite_reward.py`
  - `src/prm/model.py`
  - `src/prm/rewards/*`
