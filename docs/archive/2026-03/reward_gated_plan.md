# Reward Aggregation Redesign Plan (Draft for Confirmation)

## 背景
当前 TopoPRM 的聚合方式主要是固定加权和（weighted sum）。这在科研叙事上容易被质疑“权重手工写死”，且会带来 reward hacking 风险。

## 候选方案

### 方案A：Hierarchical-Gated（推荐，最可落地）

\[
R = R_{out} \cdot (1 + \alpha \cdot R_{topo} + \beta \cdot R_{cont})
    + \gamma \cdot R_{fmt} + \delta \cdot R_{len}
\]

并加门控：仅当 `R_out >= tau` 时才放大结构奖励，否则结构项衰减或截断。

优点：
- 保留 RLVR 的 outcome-first 原则；
- 避免“纯加权和”叙事；
- 对现有代码改动小，易做 ablation。

风险：
- 仍有超参（`alpha,beta,tau`），但可通过网格搜索/课程策略减敏。

### 方案B：Curriculum Gate
训练前期重 outcome，后期逐步提高 topo/cont 影响。

优点：稳定；
缺点：训练脚本复杂度上升，解释性不如方案A直观。

### 方案C：Learnable Mixer
用小网络动态学习 reward 组合权重。

优点：理论灵活；
缺点：引入学习模块，偏离“纯可验证程序奖励”的简洁叙事，不建议作为当前论文主线。

## 推荐结论
优先落地 **方案A Hierarchical-Gated**，并做三组 ablation：
1. `tau = 0`（无门控，仅乘性）
2. `tau = 0.5`（中等门控）
3. `tau = 1.0`（严格 correctness-first）

## 代码接入点
- 新文件：`src/reward/gated_composite_reward.py`
- 配置新增：`configs/grpo_hier_gated.yaml`
- 评测比较：与 `grpo_main`（加权和）和 `grpo_outcome_only` 对照

## 预期论文表述
- 创新重心从“SCAE 新算法”转向“拓扑可验证过程奖励 + DAG视角RFT范式”。
- SCAE/stratified clipping 作为稳定训练策略描述，不作为唯一创新点。
