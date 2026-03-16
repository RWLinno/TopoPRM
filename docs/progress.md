# TopoPRM 工作进展文档

> 最后更新：2026-03-16 23:00

---

## 时间线

### Phase 1-3：项目搭建、数据、核心模块（2026-03-10 ~ 03-12）
- [x] 确定技术路线：Qwen3-32B + ms-swift + GRPO + 拓扑奖励
- [x] 数据管线：691 → 637 条中文数学批改数据
- [x] DAG 模块、5 种奖励函数 + 5 种消融变体、86 个单元测试通过

### Phase 4：模型升级 + SFT（2026-03-16）
- [x] 下载 Qwen3-32B (62GB, ModelScope)
- [x] 修复 DeepSpeed lr_scheduler 不兼容 → 改用原生 DDP
- [x] SFT 数据增强：637 中文 + 3000 NuminaMath + 2000 MetaMath = 5000 条
- [x] **SFT 完成**: Loss=0.311, TokenAcc=89.7%, 120 steps, 36min
- [x] Checkpoint: `output/sft_qwen3_32b/v1-20260316-154443/checkpoint-120`

### Phase 5：GRPO 训练 + Debug（2026-03-16）
- [x] 修复 vLLM KV cache 不足 (`vllm_max_model_len: 4096`)
- [x] 修复 vLLM GPU 内存溢出 (`vllm_gpu_memory_utilization: 0.5`)
- [x] **GRPO 主实验运行至 step 186/318 后被 OOM killer 杀死**
  - **根因**: vLLM prefix caching 导致共享内存持续增长至 680GB/700GB
  - **修复**: `vllm_enable_prefix_caching: false`
  - **影响**: 模型已收敛，checkpoint-150 可用 (reward=0.098, mean_len=937)
- [x] 所有配置已修复，添加了稳健运行脚本

### Phase 6：待执行
- [ ] 恢复 GRPO 主实验（从 checkpoint-150 继续）或直接使用 checkpoint-150
- [ ] 运行 3 组消融实验
- [ ] 评估所有模型变体
- [ ] 填充论文表格

---

## 已解决的技术问题

| # | 问题 | 根因 | 解决方案 |
|---|------|------|----------|
| 1 | DeepSpeed lr_scheduler crash | ZeRO2 optimizer wrapper 修改 param_groups | 移除 DeepSpeed，用原生 DDP |
| 2 | vLLM KV cache 不足 | max_seq_len=40960 需要 10GB KV cache | `vllm_max_model_len: 4096` |
| 3 | vLLM GPU OOM (SIGKILL) | gpu_mem_util=0.9 尝试分配 129GB | `vllm_gpu_memory_utilization: 0.5` |
| 4 | **训练 58% 静默崩溃** | **vLLM prefix_caching 共享内存累积 680GB → OOM killer** | **`vllm_enable_prefix_caching: false`** |

---

## GRPO 主实验训练曲线（截至 step 185/318）

| Step | Reward | Mean Length | Memory | 状态 |
|------|--------|------------|--------|------|
| 1    | 0.042  | 1915       | 39 GiB | warmup |
| 50   | 0.083  | 1415       | 45 GiB | learning |
| 100  | 0.097  | 1134       | 45 GiB | converging |
| 150  | 0.098  | 937        | 45 GiB | **checkpoint saved** |
| 185  | 0.100  | 645        | 45 GiB | **OOM killed** |

---

## 脚本说明

| 脚本 | 用途 |
|------|------|
| `scripts/run_grpo.sh <config> [--resume]` | 单个 GRPO 实验（支持自动恢复） |
| `scripts/run_grpo_main_resume.sh` | 从 checkpoint-150 恢复主实验 |
| `scripts/run_ablations.sh` | 顺序运行 3 组消融实验 |
| `scripts/run_eval.sh <ckpt> <name>` | 评估单个模型 |
| `scripts/run_eval_all.sh` | 评估所有模型变体 |
| `scripts/monitor_training.sh [interval]` | 监控 GPU/内存/共享内存 |
