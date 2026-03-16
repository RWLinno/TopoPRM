# TopoPRM 工作进展文档

> 最后更新：2026-03-17 01:00

---

## 时间线

### Phase 1-3：项目搭建、数据、核心模块（2026-03-10 ~ 03-12）
- [x] 技术路线、项目骨架、依赖安装
- [x] 数据管线 (691→637 条)、DAG 模块、5 种奖励 + 5 种消融、86 个测试通过

### Phase 4：模型升级 + SFT（2026-03-16）
- [x] Qwen3-32B 下载 (62GB)、配置文件全部更新
- [x] 修复 DeepSpeed lr_scheduler → 改用原生 DDP
- [x] SFT 数据增强：5000 条 (637 中文 + 3000 NuminaMath + 2000 MetaMath)
- [x] **SFT 完成**: Loss=0.311, TokenAcc=89.7%, 36min

### Phase 5：GRPO 训练（2026-03-16 ~ 03-17）
- [x] GRPO 主实验运行至 step 186/318
  - reward: 0.042 → 0.100 (收敛), completion_length: 1915 → 645
  - **被 OOM killer 杀死** (vLLM prefix_caching 共享内存累积 680GB/700GB)
  - checkpoint-150 可用 (reward=0.098, 已收敛)
- [x] **根因修复**: 消融实验改用无 vLLM 模式 (稳定但较慢)
- [🔄] **消融实验 outcome_only 正在运行** (step 1/79, ~7.8h 预计)
- [ ] 消融实验 no_topo
- [ ] 消融实验 no_continuity

### Phase 6：论文（持续更新）
- [x] 全部 6 个 section 已完成初稿
- [x] 5 个表格模板 (待填数据)
- [x] Algorithm 1: TopoPRM Training Pipeline
- [x] 附录: Training Details, Case Study, Reward Details
- [ ] 填充实验结果
- [ ] 训练曲线图

### Phase 7：蒸馏（待消融完成后）
- [x] 蒸馏数据生成脚本 (`scripts/generate_distill_data.sh`)
- [x] 蒸馏配置 (7B, 1.5B)
- [ ] 生成蒸馏数据 → 训练 student models

---

## 已解决的技术问题

| # | 问题 | 解决方案 |
|---|------|----------|
| 1 | DeepSpeed lr_scheduler crash | 移除 DeepSpeed, 用 DDP |
| 2 | vLLM KV cache 不足 | `vllm_max_model_len: 4096` |
| 3 | vLLM GPU OOM (SIGKILL) | `vllm_gpu_memory_utilization: 0.5` |
| 4 | 训练 58% 静默崩溃 | vLLM prefix_caching 关闭 |
| 5 | vLLM colocate 初始化 OOM | 消融实验改用无 vLLM 模式 |
| 6 | max_completion_length=1024 太短 | 改为 2048 (模型需 ~1500 tokens) |

---

## 当前实验状态

| 实验 | 状态 | 最佳 Checkpoint | Reward |
|------|------|-----------------|--------|
| GRPO main (TopoPRM) | ✅ 收敛 | checkpoint-150 | 0.098 |
| Ablation: outcome_only | 🔄 运行中 | - | - |
| Ablation: no_topo | ⏳ 排队 | - | - |
| Ablation: no_continuity | ⏳ 排队 | - | - |

---

## 运行命令

```bash
# 监控训练
tail -f output/grpo_outcome_only_run2.log | grep "Train:\|reward"

# 下一个消融实验 (outcome_only 完成后)
bash scripts/run_grpo.sh grpo_no_topo

# 评估所有模型
bash scripts/run_eval_all.sh

# 蒸馏 (消融完成后)
bash scripts/generate_distill_data.sh
swift sft --config experiments/configs/distill_7b.yaml
```
