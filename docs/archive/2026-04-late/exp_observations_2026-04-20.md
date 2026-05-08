# 实验观察与分析（2026-04-20）

## 今日执行事项

1. 清理重复评测进程：同一 `label` 仅保留 1 个 `bench_transformers.py` 进程，避免重复抢占 GPU 和日志互相覆盖。
2. 修复 LiveCode 数据加载问题：在 `scripts/bench_transformers.py` 中为 `load_livecode()` 增加本地多路径搜索与 HuggingFace 回退加载，解决 `No data for benchmark livecode`。
3. 修复 LiveCode 匹配逻辑：新增 `answers_match_text()`，并将 `livecode` benchmark 的 matcher 改为文本匹配，避免代码任务被错误按数值题判分。

## 当前在跑任务快照（记录时刻）

- `sft_9b_v2_ext_k5_gpu1`: `[386/14042] acc=86.3%`
- `topoprm_gated_9b_v2_ext_k5_gpu2`: `[324/14042] acc=85.2%`
- `topoprm_hier_9b_v2_ext_k5_gpu7`: `[96/14042] acc=79.2%`
- `student_4b_sft_distill_ext_k5_gpu1`: `[80/14042] acc=36.2%`
- `base_4b_ext_k5_gpu2`: `[502/1319] acc=59.8%`
- `topoprm_hier_qwen25_7b_ext_k5_gpu7`: `[3868/14042] acc=64.7%`

## 关键观察（Observations）

- **MMLU 是主要时延瓶颈**：多数任务处在 `mmlu (14042)` 阶段，单任务耗时远高于 AIME/CNMO。
- **9B 主干模型稳定性更好**：SFT/TopoPRM 9B 在线 acc 曲线总体稳定上升，波动相对较小。
- **4B 蒸馏模型在通用任务上偏弱**：4B 蒸馏在 AIME/CNMO 已落盘结果中明显偏低，且 MMLU 早期表现也偏弱。
- **LiveCode 之前“无数据”主要是加载链路问题**：修复后可从 HF 数据源加载样本，不再直接跳过。

## 低表现实验结论（后续决策）

- **Legacy RKL distill 路线不建议继续投入**：历史结果显示 GSM8K/MATH-500 相比 9B 主干路线明显落后。
- **小模型（<=4B）应优先聚焦 TVSD 和数据质量**：不建议沿用旧蒸馏设置直接扩量跑全表。
- **主表优先级建议**：先保证 9B 主线（SFT/TopoPRM）统一指标完整，再回补 4B 以下学生模型。

## 日志处理说明

- 不保留无用/低价值的完整历史日志文件。
- 关键结论与分析在本文件中保留，便于论文写作与后续复盘。
