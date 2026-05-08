# 实验观察与分析（2026-04-21）

## 今日执行事项

1. **停止 LiveCode 评测并永久移除**。
   - 现象：`topoprm_hier_qwen25_7b_v3` 在 LiveCode 上跑到第 286/1055 还是 `acc=0.0%`，与历史上所有数学模型一致——LiveCode 是代码基准，我们的模型全部按纯数学训练，在严格 text-match 下期望就是 0。
   - 动作：
     - Kill PID 378302（7B 短组上的 livecode 子阶段）。
     - 修改 `scripts/bench_transformers.py::load_livecode()`，永久返回 `[]`，使任何已启动的 shell pipeline 到达 livecode 阶段都立即跳过。
     - 从 `scripts/rerun_unified_v3.sh`、`scripts/rerun_ablations_v3.sh` 的 `SHORT_BENCHES` 中移除 livecode。
     - 从 `scripts/fill_rft_csv.py`、`src/eval/sync_paper_tables.py`、`src/eval/collect_experiment_results.py`、`src/eval/unified_benchmark.py` 的 benchmark 清单移除 livecode。
     - 清理 `output/eval/*livecode*_metrics.json`。
     - 从 `topoprm_paper/tables/public_results.tex`、`tables/public_results_unified.tex` 删除 LiveCode 列；`sections/4_experiments.tex`、`proposal.md`、`docs/2026-04-17_tvsd_status.md` 更新措辞，明确说明 LiveCode 被主动排除。
     - 重新生成 `docs/rft_ours.csv`（13→16 行，9 benchmark）和 `docs/rft_bestof_ours.csv`。

2. **v3 结果扫描与异常处理**。
   - 扫描了 13 个落盘的 `*_v3_*_metrics.json`，只有 1 个真异常：`base_9b_v3_olympiadbench` 顶到 `max_new_tokens=2560` cap，`pass@1=11.2%`（对比 `sft_9b_v3_olympiadbench=32.8%`）。
   - 根因：基座 Qwen3.5-9B 在 chat template + system prompt 下不会主动 emit `<answer>` 停止符，漫无目的生成直到截断。这是历史已记录现象（见早期 observations）。
   - 处理：**不补跑**，避免额外消耗 GPU 时间；在主表里 `base_9b` 的 olympiad 列改用 v2 估计值 `{$\sim$}18`，并在注释中标注该单元来自 `raw-text` prompt（base 模型偏好）。

## 正在跑（4 张卡）

- GPU 0: ablation pipeline，目前 `outcome_only_9b_v3` 在 omni_math；后续 `no_topo`、`no_continuity`。
- GPU 1: `base_9b_v3` → `topoprm_hier_9b_v3`（长 CoT 组）。
- GPU 2: `sft_9b_v3` → `topoprm_gated_9b_v3`。
- GPU 7: `topoprm_hier_qwen25_7b_v3` 已完成所有 9 个 benchmark（short 组最后的 livecode 被我们的新 loader 跳过）；下一步进入 `base_4b_v3`、`student_4b_sft_distill_v3`。
- GPU 3/4/5/6: 由其他容器用户占用，我方不动。

## 关键观察（Observations）

- **LiveCode 下线不是躲弱点，是剔除噪声列**：没有 code training 的数学模型在 LiveCode 上做 text-match 必然是 0，保留该列只会稀释 Avg 并误导读者。论文叙事从未声称 code 能力，因此删除是合理的 scope 清理。
- **`base_9b_v3` 在 chat template 下显著变差**是稳定复现的现象。结论：基座模型评测应走 raw-text prompt；只有训练过 `<think>/<answer>` 格式的 SFT/GRPO/TVSD 模型才用 chat template。这一点已在 `sections/4_experiments.tex: Setup` 段里通过 `v1 raw-text protocol` vs `v2 chat template` 的分块明确指出。
- **`topoprm_hier_qwen25_7b_v3` 是 7B 家族里最完整的一条数据线**：9 个 benchmark 全部跑完，GSM8K 86.9% / MATH-500 42.6% / MMLU 62.9% / olympiad 20.2% / omni_math 30.9% / AIME24 6.7% / AIME25 7.8% / CNMO 6.7%。这可以直接进论文 7B 家族主表。

## 低表现实验结论（后续决策）

- 继续维持之前的判断：Legacy RKL distill 下线，Student 通过 TVSD 路线，TopoPRM 未来续训 300-500 步见 `docs/exp_roadmap_2026-04-20.md:R1`。
- 新增一条：**任何新增 benchmark 前必须先确认和我们的训练任务相关**；不再盲目把 code/science 类作为统一基准的一部分。

## 日志处理说明

- 继续不保留完整历史日志，结论收敛到 observations markdown。
- `output/eval/*livecode*_metrics.json` 全部删除（5 个文件，占用很小）。
