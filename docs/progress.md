# TopoPRM 工作进展文档

> 最后更新：2026-03-16

---

## 时间线

### Phase 0：环境搭建与技术选型（2026-03-10 ~ 03-11）

- [x] 确定技术路线：Qwen3-14B + ms-swift + GRPO + 拓扑奖励
- [x] 创建项目骨架：`src/`、`tests/`、`configs/`、`scripts/`、`docs/`
- [x] 安装依赖：ms-swift 4.x、networkx、deepspeed 等
- [x] 解决 ms-swift 4.x 兼容性问题（见下方技术问题记录）
- [x] 下载基座模型 Qwen3-14B

### Phase 1：数据管线（2026-03-11 ~ 03-12）

- [x] 实现原始数据解析模块 `src.data.parse_raw`
  - 输入：原始数学批改 JSON 文件（691 条）
  - 输出：标准化 JSONL（stem, standard_answer, student_answer, llm_result, score, category 等字段）
- [x] 实现数据清洗模块 `src.data.clean`
  - 过滤无效记录（缺少必要字段、score 异常等）
  - 691 → 637 条（清洗率 7.8%）
- [x] 实现 DAG 构建模块 `src.data.build_dag`
  - 从 LLM 批改结果中提取步骤，构建推理 DAG
  - 生成 637 个 DAG JSON 文件
- [x] 实现 SFT 数据准备 `src.data.prepare_sft`
  - 生成 system + user + assistant 对话格式
  - 637 条原始 × 2（正向 + 反向）= 1274 条
- [x] 实现 GRPO 数据准备 `src.data.prepare_grpo`
  - 生成 prompt-only 格式 + solution 标签 + reference_dag
  - 637 条
- [x] 实现数据集合并 `src.data.merge_datasets`

### Phase 2：核心模块开发（2026-03-11 ~ 03-12）

- [x] 实现 DAG 数据结构
  - `Node`：步骤节点（8 种 StepType：definition, derivation, computation, conclusion, auxiliary, substitution, case_analysis, unknown）
  - `Edge`：边（sequential / dependency）
  - `ReasoningDAG`：基于 networkx.DiGraph 的完整 DAG 实现
  - `compress_dag`：DAG 压缩算法
- [x] 实现 5 种奖励函数
  - `OutcomeReward`：评分准确性 (w=0.40)
  - `FormatReward`：格式合规性 (w=0.15)
  - `TopoReward`：拓扑结构质量 (w=0.20)
  - `ContinuityReward`：推理连续性 (w=0.15)
  - `LengthReward`：长度惩罚 (w=0.10)
  - `TopoCompositeReward`：加权复合奖励
- [x] 注册 ms-swift ORM 插件（`orms["topo_composite"]`）
- [x] 编写单元测试（86 个），全部通过

### Phase 3：SFT 训练（2026-03-13）

- [x] 编写 SFT 训练配置 `configs/sft_qwen3_14b.yaml`
  - Qwen3-14B + LoRA (rank=64, alpha=128, all-linear)
  - DeepSpeed ZeRO Stage 3
  - batch_size=2, grad_accum=8, lr=1e-4
  - 3 epochs = 120 steps
- [x] 执行 SFT 训练
  - 训练耗时：49 分钟
  - 最终 loss：0.250（从 0.945 下降）
  - 最终 token_acc：0.916（从 0.777 上升）
  - 显存峰值：107.1 GiB
- [x] 保存检查点 `output/sft_qwen3_14b/v0-20260313-195147/checkpoint-120`

### Phase 4：实验配置（2026-03-13 ~ 03-14）

- [x] 编写 GRPO 训练配置 `configs/grpo_qwen3_14b.yaml`
- [x] 编写消融实验脚本
  - `experiments/run_ablation_outcome_only.sh`
  - `experiments/run_ablation_no_topo.sh`
  - `experiments/run_ablation_no_continuity.sh`
- [x] 编写蒸馏实验脚本 `experiments/run_distill.sh`
- [x] 编写评估配置 `configs/eval_config.yaml`
- [x] 编写全流程启动脚本 `experiments/run_all.sh`

### Phase 5：GRPO 训练（2026-03-14 ~ 进行中）

- [x] 启动 GRPO 主实验
- [ ] 等待 GRPO 主实验完成
- [ ] 启动消融实验
- [ ] 启动蒸馏实验

### Phase 6：评估（待开始）

- [ ] 在 MATH, GSM8K, CMATH, GaoKao, C-Eval-Math 上评估
- [ ] 对比 base / SFT / GRPO / 消融 / 蒸馏 结果
- [ ] 生成评估报告

---

## 已解决的技术问题

### 1. ms-swift 4.x ORM 接口变更

**问题**：ms-swift 从 3.x 升级到 4.x 后，自定义奖励模型（ORM）的注册方式发生了变化。旧版的 `reward_model_cls` 参数不再可用。

**解决方案**：使用 `external_plugins` 机制加载自定义奖励代码：

```yaml
# configs/grpo_qwen3_14b.yaml
external_plugins: src/reward/composite_reward.py
reward_funcs:
  - topo_composite
```

在 `composite_reward.py` 中通过模块级代码注册：

```python
from swift.rewards import ORM, orms
orms["topo_composite"] = TopoCompositeReward
```

### 2. PYTHONPATH 依赖

**问题**：`src.*` 模块的导入要求项目根目录在 `PYTHONPATH` 上，但 ms-swift 的 `swift sft` / `swift rlhf` 命令不会自动设置。

**解决方案**：在所有 shell 脚本中统一添加：

```bash
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"
```

### 3. DeepSpeed ZeRO3 与 LoRA 的显存管理

**问题**：SFT 训练初期显存仅需 37 GiB，但随着梯度累积和优化器状态的增长，峰值达到 107 GiB。在单卡 80G A100 上会 OOM。

**解决方案**：
- 使用 DeepSpeed ZeRO Stage 3 进行参数分片
- 启用 `gradient_checkpointing: true` 减少激活内存
- 多卡训练（实际使用 2×A100 80G）

### 4. DAG 构建中的步骤分割

**问题**：LLM 批改结果中的推理步骤有时跨多行，简单的换行分割会导致步骤不完整。

**解决方案**：
- 基于标记模式（`**小题X**`、`步骤块行序号范围` 等）进行结构化分割
- 对短行（<10字符）进行合并处理
- 仍存在少量边界情况（见已知问题）

### 5. 中文数学表达式的正则匹配

**问题**：中文数学批改中混合使用 LaTeX（`$...$`、`\(...\)`）和纯文本数学表达式，正则匹配容易遗漏或误匹配。

**解决方案**：
- 多模式匹配：先匹配 LaTeX 定界符，再匹配数学符号模式（`=`、`±`、`≥` 等）
- 中文数学关键词白名单（`方程`、`等式`、`解得` 等）

### 6. 奖励函数的数值稳定性

**问题**：当模型输出不包含有效的 `<answer>` JSON 时，解析失败可能导致 NaN/Inf 传播到 GRPO 的 advantage 计算中。

**解决方案**：
- 在每个奖励组件中添加 try/except，解析失败返回 0.0
- 复合奖励的最终值 clamp 到 [0, 1]
- 在 GRPO 配置中设置 `reward_clip_range` 作为额外保护

---

## 下一步计划

### 短期（1-2 周）

1. **完成 GRPO 主实验**
   - 等待当前 GRPO 训练完成（预计 1-2 天）
   - 检查训练曲线，确认收敛
   - 保存最终检查点

2. **消融实验**
   - 依次运行 3 个消融配置（outcome-only, no-topo, no-continuity）
   - 每个预计 1-2 天训练时间

3. **基准评估**
   - 在 5 个基准数据集上评估所有模型变体
   - 生成对比表格

### 中期（2-4 周）

4. **蒸馏实验**
   - 实现逆向 KL 蒸馏代码
   - 蒸馏到 Qwen3-7B 和 Qwen3-1.5B
   - 评估蒸馏模型

5. **论文写作**
   - 补充实验结果表格
   - 完善 GRP_paper/ 中的论文内容

### 长期

6. **模型部署**
   - VLLM 推理优化
   - API 服务封装

7. **数据扩充**
   - 收集更多数学批改数据
   - 探索英文数学数据的迁移学习

---

## 已知问题

### 1. 拓扑提取的局限性

步骤分割依赖换行边界，多行步骤或行内子论证可能被遗漏。目前通过合并短行进行缓解，但仍有约 5% 的边界情况未覆盖。

### 2. 依赖检测为规则驱动

`build_dependency_edges_by_llm` 目前回退到规则匹配，而非真正的 LLM 调用。这限制了拓扑奖励中 reference-overlap 子指标的质量。

### 3. 中文专属模式

表达式/命题提取和步骤类型分类的正则模式针对中文数学符号调优。扩展到英文或其他语言需要额外的正则规则。

### 4. 格式奖励粒度较粗

格式奖励仅有 3 个离散值（0.0, 0.3, 1.0），更细粒度的 JSON schema 验证可提供更丰富的信号。

### 5. 完成级别的奖励

尽管提取了单个步骤，所有奖励仍在整个 completion 级别计算。真正的步骤级 PRM 需要逐节点的信用分配机制。

### 6. 参考 DAG 的可用性

拓扑奖励的 reference-overlap 子指标需要预构建的 DAG 文件。没有匹配 DAG 的记录会使用空 DAG，导致该子信号失效。

---

## 关键指标汇总

| 指标 | 值 |
|------|-----|
| 原始数据量 | 691 条 |
| 清洗后数据量 | 637 条 |
| SFT 训练数据 | 1274 条 |
| GRPO 训练数据 | 637 条 |
| DAG 文件数 | 637 个 |
| 数据类别 | 4 类（初中代数/几何, 高中代数/几何） |
| SFT 最终 Loss | 0.250 |
| SFT 最终 Token Acc | 0.916 |
| 训练步数 | 120 |
| 训练耗时 | 49 分钟 |
| 单元测试 | 86/86 通过 |
| 奖励组件 | 5 个 |
| 消融实验 | 3 组已配置 |
