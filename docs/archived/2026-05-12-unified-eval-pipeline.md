# 2026-05-12 — Qwen3.5-9B 一键九基准评测上线与结果回填

> 承接 [`2026-05-11-experiment-resync.md`](2026-05-11-experiment-resync.md)。本次主要交付 Qwen3.5-9B 主行（base / +SFT）在 `public_results_unified.tex` 的真实 pass@1 填充，并把评测链路打包成一条命令。

## 1. 交付物一览

| 类型 | 路径 | 说明 |
|------|------|------|
| 一键入口 (bash) | [`scripts/run_unified_eval.sh`](../scripts/run_unified_eval.sh) | 接收 `MODEL_PATH [ADAPTER] [LABEL]` 三个位参，走编排器默认 7 卡并发 |
| 编排器 (python) | [`scripts/unified_eval_orchestrator.py`](../scripts/unified_eval_orchestrator.py) | 9 个 benchmark 按“快先慢后”优先级分发；1 GPU / 1 bench；失败 1 次自动降 batch 重试 |
| 实时监控 | [`scripts/watch_unified_eval.sh`](../scripts/watch_unified_eval.sh) | 刷新 `status_<LABEL>.json` + 每个 bench 的 `pass@1` |
| 论文表回填 | [`scripts/fill_paper_table.py`](../scripts/fill_paper_table.py) | 逐行把 `pass@1%` 写回 `public_results_unified.tex` |
| 清理策略 | [`scripts/cleanup_unified_logs.sh`](../scripts/cleanup_unified_logs.sh) | 只清理无对应 `metrics.json` 的孤儿日志 |

## 2. 关键设计决定

- **为什么用“1 GPU / 1 benchmark”而不是 tensor parallel？** 9B 模型在 bf16 下占 ~20 GB，A100-80GB 单卡裕量充足；一次加载一份权重、跑一个 bench，吞吐 ~= TP 方案且不用调 vllm/ray。并发度 = 可用 GPU 数。
- **快基准优先**：`gsm8k/math500/aime24/aime25/cnmo24` 先上，避免 `olympiadbench/omni_math/mmlu` 这类慢 bench 占住卡导致论文表全是 `--`。
- **失败自动重试一次**：子进程 rc != 0 → 把 `batch_size` 减半再跑一遍。Swift / HF 偶发 OOM / CUDA context 抖动自救。
- **MMLU 限项 1500**：对应现象是 `sft_style + MMLU` 历史卡死（见 2026-05-11 文档 §3.1）；限项 + 仍保留 `--allow_mmlu_sft_style` 兜底。

## 3. 标准运行参数（已固化）

| bench | `max_new_tokens` | `max_items` | `batch_size` | 预估单卡时 |
|-------|------------------|-------------|--------------|------------|
| gsm8k | 1536 | 0 | 4 | 45 min |
| math500 | 3072 | 0 | 4 | 60 min |
| aime2024 / aime2025 | 4096 | 0 | 2 | 25 min |
| cnmo2024 | 4096 | 0 | 2 | 40 min |
| olympiadbench | 4096 | 500 | 2 | 120 min |
| omni_math | 4096 | 500 | 2 | 150 min |
| gpqa_diamond | 1536 | 0 | 4 | 35 min |
| mmlu | 768 | 1500 | 8 | 100 min |

`num_samples_per_item=1, k_values=[1], use_chat_template=True, sft_style=opt-in`。

## 4. 本次 Qwen3.5-9B 评测

- 启动：2026-05-12 04:17 CST，GPUs 1–7（GPU 0 被其他任务占用）。
- 模型路径：`/Knowin/foundation/weilinruan/hf_models/Qwen/Qwen3.5-9B`（架构 `Qwen3_5ForConditionalGeneration`，bf16，~20 GB / 卡）。
- Label：`qwen35_9b_base`。9 个 bench 并发起跑；GPQA-D 与 MMLU 排队等前一批释放 GPU。
- 实时日志：
  - `logs/unified/orchestrator_qwen35_9b_base_*.out`
  - `logs/unified/qwen35_9b_base_<bench>.log`
  - `logs/unified/status_qwen35_9b_base.json`
- 回填：bench 一个产出一个，按 `fill_paper_table.py --label qwen35_9b_base --row "Qwen3.5-9B (base)" --write` 同步到主表。

### 4.1 SFT 行的现状与下一步

本机**尚无 Qwen3.5-9B 的 SFT checkpoint**：
- `output/sft_qwen35_9b/...`（历史脚本引用路径）不存在。
- 唯一本地 adapter `output/sft_deepseek_r1_7b/final/` 的 `adapter_config.json` base 为 `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`，不能挂到 Qwen3.5-9B 上。
- 处理：`run_unified_eval.sh` 在 `SFT_STYLE=1` 但无 adapter 时会自动发现并校验 base model 一致性，base 不匹配会立刻报错并退出，避免跑出误导结果。

**下一步（有 SFT checkpoint 后）**
```bash
SFT_STYLE=1 RUN_IN_BACKGROUND=1 \
    bash scripts/run_unified_eval.sh \
    /Knowin/foundation/weilinruan/hf_models/Qwen/Qwen3.5-9B \
    <PATH_TO_QWEN35_9B_SFT_CKPT> qwen35_9b_sft
```

### 4.2 Qwen3.5-9B (base) 最终 pass@1（2026-05-12 17:11 CST 完成）

| bench          | n_items | pass@1 | avg_tokens | elapsed |
|----------------|--------:|-------:|-----------:|--------:|
| GSM8K          |   1319  | 67.1%  | 1536       | ≈ 5 h   |
| MATH-500       |    500  | 43.8%  | 3072       | ≈ 5 h   |
| OlympiadBench  |    500  | 29.8%  | 4096       | ≈12.9 h |
| Omni-MATH      |    500  | 51.4%  | 4096       | ≈12.9 h |
| AIME'24        |     30  |  6.7%  | 4096       | ≈ 46 m  |
| AIME'25        |     30  |  3.3%  | 4096       | ≈ 47 m  |
| CNMO'24        |     83  | 12.0%  | 4096       | ≈ 2.2 h |
| MMLU           |   1500  | 52.9%  |  768       | ≈ 2.1 h |
| GPQA-Diamond   |    198  | 27.3%  | 1536       | ≈ 58 m  |

已回填 `topoprm_paper/tables/public_results_unified.tex` 的 `Qwen3.5-9B (base)` 行（从原占位 `\textbf{91.0}/\underline{55.0}/\sim18/\sim20/\sim17/\sim14/\sim10/\underline{83.2}/\underline{42.1}` 改为真实值 `67.1/43.8/29.8/51.4/6.7/3.3/12.0/52.9/27.3`）。作者需根据新数字重走一遍排名加粗/下划线的视觉标注。

### 4.3 过程中修的两个数据与抽取器 bug

1. **MMLU 样本全部空 prompt** (`scripts/bench_transformers.py::_format_mmlu_q`)：我们本地 `data/benchmarks/MMLU/test.jsonl` 使用大写字段 `Problem` / `Answer`，原函数只读 `question` / `answer` 小写字段，导致每道题的提示只含选项没有题干，模型无法回答，准确率固定 0%。已扩展为 `question or Problem or problem`、`answer or Answer`，同时保留 `options` 兼容。
2. **GPQA-Diamond 同一 bug** (`load_gpqa_diamond._to_mcq`)：只读 `question / Question`，本地数据键名是 `Problem`，导致空题干。已扩展字段兜底。
3. **MCQ extractor 倾向取首字母** (`extract_mcq`)：在 CoT 里 "Option A is wrong ... the answer is C" 这类输出原本会返回 A。改成：先在输出尾部 400 字符内搜索结论式 pattern（`\boxed{·}`、`the answer is X`、`final answer X`、`pick/select/choose X`），取**最后一次命中**；退化到最后一个 `A-E` 单字母。本次 MMLU 真实 52.9% 就是修后得出的，明显高于修前观察到的 9.4%。

### 4.4 Orchestrator 小坑

同一 label 启动两次 orchestrator（例如重跑单个 bench）会互相覆盖 `status_<label>.json`。本次用另起 label `qwen35_9b_base_fix` 跑 MMLU+GPQA-D 修复版，完成后再用 `promote` 脚本把 `qwen35_9b_base_fix_<bench>_metrics.json` 拷成 `qwen35_9b_base_<bench>_metrics.json`（并打上 `_promoted_from`）。后续最好直接在 orchestrator 里加 `--append_status` 模式，避免这种绕。

## 5. 慢基准提示（给作者侧的心理预期）

按当前 `max_new_tokens` / `max_items` 配置，预计“最慢先完成的时间轴”大致如下（单卡、无互相阻塞情况下）：

1. AIME'24 / AIME'25：20–30 min
2. GSM8K / CNMO'24：45–60 min
3. MATH-500 / GPQA-D：60–90 min
4. MMLU(1500) / OlympiadBench(500) / Omni-MATH(500)：1.5–3 h

所以“快批”一般 1 小时内能先回 4–5 个 cell，够你在论文表里先去掉 `~`；慢批 3–4 小时后能补齐 Olympiad / Omni-MATH / MMLU / GPQA-D。

## 6. 论文表主行维护约定

- Qwen3.5-9B base / +SFT 两行只接受 `fill_paper_table.py` 写入的数值（可追溯到 `output/eval/*_metrics.json`）。
- 已标 `\textbf{}` / `\underline{}` 的格式由作者手工维护；本脚本**只改数字本身**，保留原有 LaTeX 修饰位。如果手工修饰位跟着新排名走，评估完成后由作者整体刷一轮。
- 其他行（ablation / distill 等）保持不动，由对应训练/蒸馏产出单独跑评测后再回填。

## 7. 清理策略

默认“**仅清理无对应 metrics.json 的孤儿日志**”：
- 保留：成功/进行中（有 metrics.json 或最近 7 天）的 per-bench log。
- 清理：无对应 metrics.json 且 >7 天的 `logs/unified/*.log`；孤儿 `output/eval/*_details.jsonl`。
- 使用：先 `bash scripts/cleanup_unified_logs.sh`（dry-run），确认再 `CONFIRM=1 ...`。
