# TopoPRM 工作进展文档

> 最后更新：2026-03-16 17:30

---

## 时间线

### Phase 0：环境搭建与技术选型（2026-03-10 ~ 03-11）

- [x] 确定技术路线：Qwen3-14B + ms-swift + GRPO + 拓扑奖励
- [x] 创建项目骨架：`src/`、`tests/`、`configs/`、`scripts/`、`docs/`
- [x] 安装依赖：ms-swift 4.x、networkx、deepspeed 等
- [x] 解决 ms-swift 4.x 兼容性问题
- [x] 下载基座模型 Qwen3-14B

### Phase 1：数据管线（2026-03-11 ~ 03-12）

- [x] 实现原始数据解析模块 `src.data.parse_raw` (691 → 637 条)
- [x] 实现数据清洗模块 `src.data.clean`
- [x] 实现 DAG 构建模块 `src.data.build_dag` (637 个 DAG)
- [x] 实现 SFT / GRPO 数据准备
- [x] 实现数据集合并 `src.data.merge_datasets`

### Phase 2：核心模块开发（2026-03-11 ~ 03-12）

- [x] 实现 DAG 数据结构 (Node, Edge, ReasoningDAG)
- [x] 实现 5 种奖励函数 + 消融变体
- [x] 注册 ms-swift ORM 插件
- [x] 编写单元测试（86 个），全部通过

### Phase 3：SFT 训练 Qwen3-14B（2026-03-13）

- [x] 完成 SFT 训练 (loss 0.945→0.250, token_acc 0.777→0.916, 49分钟)

### Phase 4：模型升级 Qwen3-32B（2026-03-16）

- [x] **下载 Qwen3-32B** (62GB, 17 shards, 从 ModelScope)
- [x] **更新所有配置文件** 从 14B → 32B
- [x] **修复 DeepSpeed lr_scheduler 兼容性问题**
  - 错误：`ValueError: zip() argument 2 is longer than argument 1`
  - 原因：DeepSpeed ZeRO2 optimizer wrapper 与 lr_scheduler param_groups 不匹配
  - 解决：移除 DeepSpeed，使用原生 DDP（8x L20X 143GB 有足够显存）
- [x] **初始化 Git 仓库**
  - 本地 repo：`/mnt/users/rwl/topoprm/.git`
  - 远程：`https://github.com/rwlinno/topoprm.git`（需用户创建后 push）
  - 分支：main

### Phase 5：数据增强（2026-03-16）

- [x] **SFT 数据增强**：637 中文批改 + 3000 NuminaMath-CoT + 2000 MetaMathQA = 5000 条
- [x] 输出文件：`data/sft_ready/train_augmented.jsonl`
- [ ] GRPO 数据保持 637 条中文批改（因 reward function 依赖批改格式）

### Phase 6：SFT 训练 Qwen3-32B（2026-03-16）

- [x] **SFT 训练完成**
  - 模型：Qwen3-32B + LoRA (rank=64, alpha=128, all-linear)
  - 数据：5000 条混合数据
  - 参数：batch=2, grad_accum=8, lr=5e-5, 3 epochs = 120 steps
  - 最终 loss：**0.311**
  - 最终 token_acc：**0.897**
  - 训练时间：**36 分钟**
  - 峰值显存：135 GiB / GPU
  - Checkpoint：`output/sft_qwen3_32b/v1-20260316-154443/checkpoint-120`

### Phase 7：GRPO 训练 Qwen3-32B（2026-03-16 ~ 进行中）

- [x] 启动 GRPO 主实验（159 步，topo_composite reward）
- [ ] 等待 GRPO 完成
- [ ] 启动消融实验 (outcome_only, no_topo, no_continuity)

### Phase 8：论文更新（2026-03-16 ~ 进行中）

- [x] **重写 Method 节**：增加 SCAE（分层裁剪优势估计），更精确的数学公式
- [x] **更新 Abstract**：更新为 32B 模型，增加 SCAE 贡献
- [x] **更新 Introduction**：三个核心贡献重新整理
- [x] **更新 Related Work**：增加 Proof-RM, DAPO, difficulty-GRPO 引用
- [x] **更新 Conclusion**：更新限制和未来工作
- [x] **更新 Experiments 框架**：为真实结果预留表格
- [x] **更新 references.bib**：补全所有引用
- [ ] 填充实验数据表格（需要训练完成后的结果）

---

## 已解决的技术问题

### 1. DeepSpeed ZeRO2 lr_scheduler 不兼容（新增）

**问题**：ms-swift 4.0.1 + torch 2.10.0 + DeepSpeed ZeRO2 组合下，`lr_scheduler.step()` 报错 `ValueError: zip() argument 2 is longer than argument 1`。

**根因**：DeepSpeed 的 optimizer wrapper 修改了 `param_groups` 数量，导致与 cosine lr_scheduler 的 `values` 列表长度不一致。

**解决方案**：移除 DeepSpeed，使用原生 PyTorch DDP。8x L20X (143GB) 对 32B LoRA 训练有充足显存。

### 2. Qwen3-32B 首次 GRPO Generation 极慢

**问题**：不使用 vLLM 时，32B 模型的首次 generation（8x4096 tokens）非常慢。

**状态**：已知问题。vLLM 0.17.1 与 TRL 0.10.2-0.12.0 不兼容，无法使用。考虑后续降级 vLLM 或使用 colocate 模式。

### 3-6. （保留之前的技术问题记录）

同上一版本。

---

## 硬件配置

| 项目 | 规格 |
|------|------|
| GPU | 8× NVIDIA L20X (143 GB VRAM each) |
| 总 GPU 显存 | 1.1 TB |
| 磁盘 | 2.0 TB 可用 |
| Python | 3.12 |
| ms-swift | 4.0.1 |
| PyTorch | 2.10.0+cu129 |
| vLLM | 0.17.1 |
| Transformers | 4.57.6 |

---

## 关键指标汇总

| 指标 | 值 |
|------|-----|
| 基座模型 | Qwen3-32B (64层, hidden=5120, 33.3B params) |
| SFT 数据量 | 5000 条 (637 中文 + 4363 英文) |
| GRPO 数据量 | 637 条 |
| 测试数据量 | 7595 条 (2181 初中 + 5414 高中) |
| SFT 最终 Loss | 0.311 |
| SFT 最终 Token Acc | 0.897 |
| SFT 训练步数 | 120 |
| SFT 训练耗时 | 36 分钟 |
| GRPO 总步数 | 159 |
| 奖励组件 | 5 个 (outcome/format/topo/continuity/length) |
| 消融实验 | 3 组 (outcome_only, no_topo, no_continuity) |
| 单元测试 | 86/86 通过 |

---

## 下一步计划

1. **等待 GRPO 训练完成** → 检查收敛曲线
2. **运行消融实验** (3 组)
3. **基准评估** (MATH500, GSM8K, CMATH, GaoKao)
4. **填充论文表格** → 完善实验分析
5. **蒸馏实验** (32B → 14B/7B)
6. **推送到 GitHub**
