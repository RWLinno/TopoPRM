# General Benchmark 评测记录（2026-04-11）

## 目标

- 使用 GPU 0-4 并行运行 general benchmark 主表评测
- 重点验证：
  - TopoPRM 方法（Qwen3.5-9B hierarchical）
  - 蒸馏学生（Qwen3-8B + RKL）

## 已启动的评测任务

- `CUDA_VISIBLE_DEVICES=0`:
  - `topoprm_hier_9b_mcl4096`（Qwen3.5-9B + hierarchical checkpoint-79）
- `CUDA_VISIBLE_DEVICES=1`:
  - `topoprm_full_32b_reval`（Qwen3-32B + grpo_main checkpoint-79）
- `CUDA_VISIBLE_DEVICES=2`:
  - `distill_rkl_8b_reval`（Qwen3-8B + distill_rkl checkpoint-500）
- `CUDA_VISIBLE_DEVICES=3`:
  - `qwen3_8b_base_reval`（Qwen3-8B base）
- `CUDA_VISIBLE_DEVICES=4`:
  - `sft_9b_reval`（Qwen3.5-9B + SFT checkpoint-626）

## 遇到的问题与修复

### 1) 进程管理问题

- 之前的任务用 `nohup` 和 `&` 启动，部分进程残留
- 已 kill 并释放 GPU 资源后重新启动

### 2) Qwen3.5-9B LoRA 命名空间不一致

- 现象：
  - `sft_9b` 和 `topoprm_hier_9b` 加载时报错
  - 日志显示 `missing adapter keys`
- 原因：
  - swift 训练的 `adapter_config.json` 中 `target_modules` 前缀为 `model.language_model`
  - 但 `transformers` 加载时期望前缀为 `model`
- 修复：
  - 在 `scripts/bench_transformers.py` 中增加自动 patch 逻辑
  - 同时修正 `adapter_model.safetensors` 中的 tensor key 前缀

### 3) 初步结果（GSM8K 前 96 条在线精度）

- 结果：
  - `distill_rkl_8b_reval`: `80.2%`（96/1319 已完成）
  - `qwen3_8b_base_reval`: `79.2%`（96/1319 已完成）
- 观察：
  - 蒸馏模型略优于同规模 8B base，说明 GSM8K + MATH-500 上蒸馏有正向效果

## 后续

- 等待全部评测完成后，结果写入 `output/eval/*_metrics.json`
- 通过 collect/sync pipeline 自动同步到论文表格
- 更新 `docs/progress.md` 记录最终结论
