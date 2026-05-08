# TopoPRM 阶段工作总结（聚焦双主贡献）

## 一、核心产出

我们将项目收敛为两条主贡献：

1. **Deterministic Verifiable PRM**：将自由推理映射为 dependency DAG，并计算 topology/continuity 两类可验证过程奖励。
2. **Reverse-KL Distillation**：通过 process-aware trace filtering，将 teacher 的图增强推理压缩为 student 的紧凑链式推理。

## 二、关键原则

- correctness-first：正确性始终是主优化目标
- verifiable != semantic proof correctness
- distillation 不是附属步骤，而是并列主贡献

## 三、工程进展

- 论文与 README 叙事已统一
- 新建 `src/prm/*` 与 `src/distill/*` 模块骨架
- 奖励聚合接口添加论文术语别名并保留兼容
- 测试新增 reverse-KL 与 process filtering 验证

## 四、风险与对策

- 风险：规则 DAG 与真实依赖不一致
  - 对策：做 extractor fidelity 抽样评估与误差分类
- 风险：多奖励混合偏离正确性
  - 对策：correctness-first shaping + 阈值敏感性分析
- 风险：student 过度 mode collapse
  - 对策：调温度/样本多样性约束并监控压缩-性能曲线
