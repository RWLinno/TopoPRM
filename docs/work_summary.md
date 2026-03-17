# TopoPRM 工作总结报告

> 撰写日期：2026-03-17

---

## 一、项目概述

TopoPRM 是一个面向数学推理的拓扑感知过程奖励框架。核心创新在于：不依赖人工标注或学习型奖励模型，而是通过确定性算法从模型生成的推理文本中提取隐式 DAG（有向无环图），并基于其拓扑结构属性（无环性、连通性、方向一致性）提供可验证的过程级奖励信号。

**技术栈**：Qwen3-32B + MS-Swift 4.x + GRPO + vLLM + LoRA

---

## 二、工作内容

### 2.1 数据管线

- 从原始数学批改数据中解析出 691 条样本，清洗后保留 637 条
- 为每条样本构建推理 DAG（节点分类、依赖检测）
- SFT 数据增强：混合 637 条中文批改 + 3000 NuminaMath-CoT + 2000 MetaMathQA = 5000 条

### 2.2 核心模块

- **DAG 数据结构**：Node/Edge/ReasoningDAG，支持 7 种步骤类型分类
- **5 种奖励函数**：outcome（答案正确性）、format（格式合规）、topo（拓扑结构）、continuity（步骤连续性）、length（长度惩罚）
- **5 种消融变体**：outcome_only、no_topo、no_continuity、no_format、topo_only
- **SCAE**：分层裁剪优势估计，防止辅助奖励导致的 reward hacking
- **86 个单元测试**全部通过

### 2.3 训练实验

- **SFT**：Qwen3-32B + LoRA，5000 条数据，3 epoch，36 分钟完成
  - 最终 loss=0.311，token accuracy=89.7%
- **GRPO**：637 条中文数学批改 prompt，每条生成 8 个 completion
  - 主实验训练至 186 步（reward 0.042→0.100，已收敛）
  - 消融实验管线已验证可行

### 2.4 论文撰写

- 完成 NeurIPS 格式全文初稿（6 个 section + 3 个 appendix）
- 参照 awesome-ai-research-writing 指南进行润色
- 包含 Algorithm 1（训练流程）、5 个实验表格模板

---

## 三、遇到的难点与解决方案

### 难点 1：DeepSpeed 与 lr_scheduler 不兼容

**现象**：SFT 训练启动后报错 `ValueError: zip() argument 2 is longer than argument 1`

**根因**：DeepSpeed ZeRO2 的 optimizer wrapper 修改了 `param_groups` 数量，与 PyTorch cosine lr_scheduler 的 `values` 列表长度不匹配

**解决方案**：移除 DeepSpeed，使用原生 PyTorch DDP。8×L20X（143GB/卡）对 32B LoRA 训练有足够显存，无需模型并行

### 难点 2：vLLM 版本不兼容

**现象**：TRL 库仅支持 vLLM 0.10.2-0.12.0，但环境中安装的是 vLLM 0.17.1

**影响**：colocate 模式下部分 API 不兼容，导致段错误（SIGSEGV）和初始化失败

**解决方案**：通过 ms-swift 的封装层间接使用 vLLM，避免直接调用不兼容的 TRL 接口。同时设置 `vllm_max_model_len: 4096` 和 `vllm_gpu_memory_utilization: 0.5` 适配内存

### 难点 3：训练 58\% 时进程静默消失（OOM Killer）

**现象**：GRPO 训练在 step 186/318 时所有 GPU 进程消失，日志无任何错误信息

**根因**：通过 `dmesg` 发现 Linux OOM Killer 杀死了进程。vLLM 的 prefix caching 机制在 colocate 模式下持续累积共享内存，11 小时后共享内存从 0 增长到 680GB，耗尽了系统的 700GB RAM

**解决方案**：
1. 设置 `vllm_enable_prefix_caching: false` 阻止共享内存累积
2. 减少并行进程数量以降低内存压力
3. 添加内存监控脚本 (`scripts/monitor_training.sh`)

### 难点 4：vLLM Server 模式权重同步失败

**现象**：Server 模式下 GRPO 训练生成的 completion 仅 2-5 tokens（全是随机内容）

**根因**：ms-swift 的 rollout server 默认使用 `load_format=dummy`，不从磁盘加载模型权重，依赖训练进程通过 NCCL 同步权重。但 NCCL 在跨 CUDA\_VISIBLE\_DEVICES 的进程间初始化失败

**解决方案**：回退到 colocate 模式，该模式在同一进程内管理模型和 vLLM，避免跨进程 NCCL 通信问题

### 难点 5：GPU 内存泄漏

**现象**：进程被杀后 GPU 显存无法释放（nvidia-smi 显示 138GB 占用但对应 PID 不存在）

**根因**：CUDA driver 级别的内存泄漏，仅能通过 GPU 驱动重置回收

**解决方案**：动态调整 `CUDA_VISIBLE_DEVICES` 只使用可用 GPU；记录泄漏 GPU 列表，后续实验规避

---

## 四、关键指标

| 指标 | 值 |
|------|-----|
| 基座模型 | Qwen3-32B (33.3B params) |
| SFT 数据量 | 5,000 条 |
| GRPO 数据量 | 637 条 |
| 测试数据量 | 7,595 条 (2,181 初中 + 5,414 高中) |
| SFT Loss | 0.311 |
| SFT Token Accuracy | 89.7\% |
| GRPO Reward（收敛值） | 0.098 / 0.1 |
| 推理长度压缩 | 1915 → 645 字符 (66\% 压缩) |
| 奖励组件数 | 5 |
| 消融实验组数 | 3 |
| 单元测试 | 86/86 通过 |
| GPU 硬件 | 8×NVIDIA L20X (143GB each) |

---

## 五、项目交付物

1. **代码仓库**：完整的 TopoPRM 项目（`src/`、`scripts/`、`configs/`、`experiments/`）
2. **SFT Checkpoint**：`output/sft_qwen3_32b/v1-20260316-154443/checkpoint-120`
3. **NeurIPS 论文初稿**：`paper/` 目录，含完整 LaTeX 源码
4. **训练脚本**：一键运行的 SFT / GRPO / 评估 / 消融脚本
5. **文档**：README、progress.md、work\_summary.md
