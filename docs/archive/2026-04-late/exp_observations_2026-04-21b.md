# TopoPRM v3 验收与补跑计划（2026-04-21b）

## 0. 背景

用户反馈上一轮 v3 评测"太慢"，kill 掉了部分实验。本文件正式入档当前
`*_v3_*_metrics.json` 的真实 coverage、被 kill 的条目、下一步补跑计划，
以及加速方案。

## 1. 当前 coverage（10 模型 × 9 benchmark，只看 `*_v3_*_metrics.json`）

| 模型 label | gsm8k | math500 | olympiad | omni_math | aime2024 | aime2025 | cnmo2024 | mmlu | gpqa_d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_9b_v3              | -    | -    | **11.2**★ | 16.4 | 0.0 | -    | -    | -    | -    |
| sft_9b_v3               | -    | -    | 32.8  | 42.4 | 30.0 | 15.6 | 30.0 | -    | -    |
| topoprm_hier_9b_v3      | -    | -    | -     | -    | -    | -    | -    | (跑中) | (跑中) |
| topoprm_gated_9b_v3     | -    | -    | -     | -    | -    | -    | -    | (跑中) | (跑中) |
| outcome_only_9b_v3      | -    | -    | 31.3  | 41.6 | 16.7 | 13.3 | 16.7 | -    | -    |
| no_topo_9b_v3           | -    | -    | -     | -    | -    | -    | -    | -    | -    |
| no_continuity_9b_v3     | -    | -    | -     | -    | -    | -    | -    | -    | -    |
| topoprm_hier_qwen25_7b_v3 | 86.9 | 42.6 | 20.2 | 30.9 | 6.7 | 7.8 | 6.7 | 62.9 | -   |
| base_4b_v3              | -    | -    |  9.7  | -    | -    | -    | -    | -    | -    |
| student_4b_sft_distill_v3 | -  | -    | -     | -    | -    | -    | -    | -    | -    |

★ 标注 `base_9b_v3_olympiadbench`: 顶到 `max_new_tokens=2560` cap，
`pass@1=11.2%`；根因是 base 模型在 chat template + SFT-style system prompt
下不会主动 emit `<answer>` 停止符，漫无目的生成直到截断。此单元不采用，
主表继续沿用 v2 估计（~18）。

## 2. 分类：采用 / 截断受害 / 待补

### 2.1 采用
- `topoprm_hier_qwen25_7b_v3`: 8/9 完成，直接入 7B 家族主表；仅 `gpqa_diamond` 待补
- `sft_9b_v3` 长组 5/5 完整
- `outcome_only_9b_v3` 长组 5/5 完整
- `base_9b_v3` omni=16.4 / aime24=0.0 入参考

### 2.2 截断受害（不采用 v3，标注 v2 估计代替）
- `base_9b_v3_olympiadbench` = 11.2 → 主表填 v2 估计 ~18

### 2.3 待补（下一轮 v3b）
所有在 §1 中为 `-` 的单元格都需要补跑。统计:
- 9B 家族 (7 模型): 各缺 2-9 benchmark
- 7B hier: 缺 1 benchmark (gpqa_diamond)
- 4B 家族 (2 模型): base 缺 8，student 缺 9
- 合计需要补 ~55 个 (模型, benchmark) pair

## 3. 为什么上一轮慢 & v3b 的加速方案

根因:
- `--batch_size 2`（在 143GB 卡上太保守，9B 模型占 20GB，剩 120GB 全浪费）
- `--max_items 1500` for MMLU（14042 全集跑不起，但 1500 仍偏大）
- 每次补跑都要从头加载模型 + 重复已完成的 benchmark（没有 skip-if-exists）

v3b 的三项改动:
1. **batch_size 硬升**: 长 CoT 组 2→6, 中组 2→8, 短组 2→16（MCQ 极快，bs=16 不占显存）
2. **MMLU 子集**: 1500 → 500（5 samples × 500 items = 2500 generations，置信区间仍 <±3pp）
3. **skip-if-exists**: `bench_transformers.py` 读写 `*_metrics.json` 时检查文件存在即跳过；`--force_overwrite` 反向开关

预估单 9B pipeline 完工: ~12h → ~4-5h（~2.5-3x 加速）

## 4. 接下来的执行计划（只用 GPU 0/1/2/3）

### 4.1 现在（t=0）
- Kill GPU7 上 `student_4b_sft_distill_v3`（迁移到 GPU3）
- 不 kill GPU1/2 上正在跑的短组（已接近尾声）
- GPU0 启动: `no_topo_9b_v3` → `no_continuity_9b_v3`
- GPU3 启动: `student_4b_sft_distill_v3` → `base_4b_v3` 补跑 → 7B gpqa 补测

### 4.2 当 GPU1/2 短组结束（t+1-2h）
- GPU1 启动: `topoprm_hier_9b_v3` long+medium 补跑 → `base_9b_v3` 补跑
- GPU2 启动: `topoprm_gated_9b_v3` long+medium 补跑 → `sft_9b_v3` backfill → `outcome_only_9b_v3` backfill

### 4.3 ETA: ~10-12 小时完成所有补跑

## 5. 验收标准
- `docs/rft_ours.csv` 除 `base_9b_v3_olympiad` 标注外，所有 v3 单元格有数
- `docs/rft_bestof_ours.csv` 同步刷新
- `topoprm_paper/tables/public_results_unified.tex` 的 v2 blue 行替换为 v3 数字

## 5.1 Final sync（v3b 跑完后必做）

v3b 跑完后的一键同步:

```bash
cd /mnt/users/rwl/topoprm && bash scripts/sync_all.sh
```

该脚本依次运行:

1. `scripts/fill_rft_csv.py` → `docs/rft_ours.csv` + `docs/rft_bestof_ours.csv`
2. `src/eval/collect_experiment_results.py` → `output/analysis/experiment_summary.*`
3. `src/eval/sync_paper_tables.py` → 更新 `topoprm_paper/tables/public_results.tex` 的 AUTO_SYNC 注释块

## 6. 不在本次范围（已在 exp_roadmap_2026-04-20.md 中登记）
- TopoPRM 续训到 300-500 步（R1）
- TVSD 端到端重训（R2）
- DAG 结构指标脚本（R3）
- Pareto 图（R4）
