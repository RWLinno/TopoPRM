# TopoPRM 工作进展文档

> 最后更新：2026-03-16 21:00

---

## 时间线

### Phase 0：环境搭建与技术选型（2026-03-10 ~ 03-11）

- [x] 确定技术路线：Qwen3-32B + ms-swift + GRPO + 拓扑奖励
- [x] 创建项目骨架：`src/`、`tests/`、`configs/`、`scripts/`、`docs/`
- [x] 安装依赖：ms-swift 4.x、networkx 等
- [x] 下载基座模型 Qwen3-14B（后升级为 32B）

### Phase 1：数据管线（2026-03-11 ~ 03-12）

- [x] 实现原始数据解析模块 `src.data.parse_raw` (691 → 637 条)
- [x] 实现数据清洗模块 `src.data.clean`
- [x] 实现 DAG 构建模块 `src.data.build_dag` (637 个 DAG)
- [x] 实现 SFT / GRPO 数据准备
- [x] 实现数据集合并 `src.data.merge_datasets`

### Phase 2：核心模块开发（2026-03-11 ~ 03-12）

- [x] 实现 DAG 数据结构 (Node, Edge, ReasoningDAG)
- [x] 实现 5 种奖励函数 (outcome, format, topo, continuity, length)
- [x] 实现 5 种消融奖励 (outcome_only, no_topo, no_continuity, no_format, topo_only)
- [x] 注册 ms-swift ORM 插件
- [x] 编写单元测试（86 个），全部通过

### Phase 3：模型升级 Qwen3-32B + SFT（2026-03-16）

- [x] 下载 Qwen3-32B (62GB, 17 shards, ModelScope)
- [x] 更新所有配置文件 14B → 32B
- [x] 修复 DeepSpeed lr_scheduler 兼容性问题（改用原生 DDP）
- [x] SFT 数据增强：637 中文 + 3000 NuminaMath-CoT + 2000 MetaMathQA = 5000 条
- [x] **SFT 训练完成**
  - Loss: 0.311, Token Acc: 89.7%, 120 steps, 36 分钟
  - Checkpoint: `output/sft_qwen3_32b/v1-20260316-154443/checkpoint-120`

### Phase 4：GRPO 训练调试（2026-03-16 ~ 进行中）

- [x] 配置 GRPO 主实验 + 3 组消融实验（统一 vLLM colocate 设置）
- [x] 修复 vLLM KV cache 内存问题 (`vllm_max_model_len: 4096`)
- [x] 修复 GPU 内存冲突（`vllm_gpu_memory_utilization: 0.5`）
- [ ] **运行 GRPO 主实验** (grpo_main, 159 steps)
- [ ] **运行 3 组消融实验** (outcome_only, no_topo, no_continuity)

### Phase 5：评估（待 GRPO 完成后）

- [x] 评估脚本完成 (`scripts/run_eval.sh`)
- [x] 评估框架完成 (`src/eval/critique_eval.py`, `src/eval/benchmark_runner.py`)
- [ ] 评估 SFT baseline
- [ ] 评估 GRPO 各变体
- [ ] 填充论文表格

### Phase 6：论文撰写（2026-03-16 ~ 进行中）

- [x] 重写 Abstract：32B 模型，SCAE 贡献
- [x] 重写 Introduction：三个核心贡献
- [x] 重写 Related Work：PRM、RLVR、graph reasoning、reward hacking
- [x] 重写 Method：SCAE、DAG 提取、拓扑/连续性奖励
- [x] 更新 Experiments 结构：匹配 32B 设置，中文测试集
- [x] 更新 Conclusion：限制和未来工作
- [x] 更新表格：main_results、ablation、structural、compression
- [x] 添加附录：训练细节、Case Study、奖励函数细节
- [x] 更新 references.bib
- [ ] 填入实验结果数据
- [ ] 添加训练曲线图

---

## 已解决的技术问题

### 1. DeepSpeed ZeRO2 lr_scheduler 不兼容

**问题**：`ValueError: zip() argument 2 is longer than argument 1` in `lr_scheduler.step()`

**根因**：DeepSpeed optimizer wrapper 修改 `param_groups` 数量，与 cosine lr_scheduler 不匹配

**解决**：移除 DeepSpeed，使用原生 DDP。8x L20X (143GB) 对 32B LoRA 有足够显存

### 2. vLLM KV Cache 内存不足

**问题**：`ValueError: max_seq_len (40960) needs 10GB KV cache, only 7.35GB available`

**根因**：Qwen3-32B 默认 max_seq_len=40960，vLLM 按此分配 KV cache

**解决**：设置 `vllm_max_model_len: 4096`（我们的 max_completion_length=2048）

### 3. vLLM GPU 内存溢出（SIGKILL）

**问题**：`vllm_gpu_memory_utilization: 0.9` 导致 vLLM 尝试分配 129GB，超出可用空间

**根因**：colocate 模式下训练模型占用 64GB，vLLM 按总显存 0.9 分配

**解决**：设置 `vllm_gpu_memory_utilization: 0.5`，确保 vLLM 分配不超过可用空间

---

## 硬件配置

| 项目 | 规格 |
|------|------|
| GPU | 8× NVIDIA L20X (143 GB VRAM each) |
| 总 GPU 显存 | 1.1 TB |
| 系统内存 | 700 GB |
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
| SFT Loss | 0.311 |
| SFT Token Acc | 89.7% |
| SFT 训练步数/耗时 | 120 steps / 36 min |
| GRPO 总步数 | 159 |
| 奖励组件 | 5 个 (outcome/format/topo/continuity/length) |
| 消融实验 | 3 组 (outcome_only, no_topo, no_continuity) |

---

## 项目文件结构

```
topoprm/
├── configs/                    # SFT 配置
│   ├── sft_qwen3_32b.yaml
│   └── grpo_qwen3_32b.yaml
├── experiments/configs/        # GRPO 实验配置
│   ├── grpo_main.yaml          # 完整 TopoPRM 奖励
│   ├── grpo_outcome_only.yaml  # 消融：仅 outcome
│   ├── grpo_no_topo.yaml       # 消融：去掉拓扑
│   ├── grpo_no_continuity.yaml # 消融：去掉连续性
│   ├── distill_7b.yaml         # 蒸馏到 7B
│   └── distill_1_5b.yaml      # 蒸馏到 1.5B
├── scripts/
│   ├── run_sft.sh              # SFT 训练
│   ├── run_grpo.sh             # 单个 GRPO 实验
│   ├── run_eval.sh             # 评估脚本
│   └── run_all_experiments.sh  # 完整实验流程
├── src/
│   ├── data/                   # 数据管线
│   ├── dag/                    # DAG 核心模块
│   ├── reward/                 # 奖励函数
│   │   ├── composite_reward.py # 完整复合奖励
│   │   └── ablation_rewards.py # 消融变体
│   └── eval/                   # 评估模块
├── paper/                      # NeurIPS 论文
│   ├── main.tex
│   ├── sections/
│   ├── tables/
│   └── references.bib
├── data/
│   ├── sft_ready/              # SFT 训练数据
│   ├── grpo_ready/             # GRPO 训练数据
│   └── test/                   # 测试数据
└── output/                     # 训练输出
```

---

## 下一步操作

1. 运行 GRPO 主实验：`bash scripts/run_grpo.sh grpo_main`
2. 运行消融实验：依次运行 `outcome_only`, `no_topo`, `no_continuity`
3. 运行评估：`bash scripts/run_eval.sh <checkpoint> <name>`
4. 填充论文表格数据
5. 推送到 GitHub
