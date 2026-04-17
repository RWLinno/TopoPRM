# TopoPRM 项目时间线总结（2026-03-13 → 2026-04-17）

> 这份文档把 proposal 提出到今天为止所有关键阶段的工作、结果、分析、踩过的坑和解法按时间顺序收拢到一处。读完这一份就可以不用再翻零散的 progress / session_summary / exp_completion 文档。文末给出把 SD-Zero 自蒸馏思想融入我们蒸馏模块的实验计划。


## 目录

- [2026.3.13 — 2026.3.20：把底子搭起来](#2026313--2026320把底子搭起来)
- [2026.3.23：论文叙事重构 + 首轮评测铺开](#2026323论文叙事重构--首轮评测铺开)
- [2026.3.24 — 2026.3.27：主基线回填 + 评测卡点反复](#2026324--2026327主基线回填--评测卡点反复)
- [2026.4.6：跨尺度消融 —— TopoPRM 在小模型上失效](#202646跨尺度消融--topoprm-在小模型上失效)
- [2026.4.7：9B own-SFT 冷启动 + gated 奖励重设计 + 蒸馏意外爆发](#202647-9b-own-sft-冷启动--gated-奖励重设计--蒸馏意外爆发)
- [2026.4.8：DAG 闭环升级](#202648dag-闭环升级)
- [2026.4.10 — 2026.4.11：公开 benchmark 自动化 + Math-7B 对照](#2026410--2026411公开-benchmark-自动化--math-7b-对照)
- [2026.4.16 — 2026.4.17：统一评测 + 投稿润色 + 终盘决策](#2026416--2026417统一评测--投稿润色--终盘决策)
- [贯穿全项目的失败模式与解法](#贯穿全项目的失败模式与解法一张表看完)
- [未来方向：把 SD-Zero 融进蒸馏](#未来方向把-sd-zero-融进蒸馏)
- [当前状态快照](#当前状态快照2026417)

---

## 2026.3.13 — 2026.3.20：把底子搭起来

这段时间在把"把文本推理变成 DAG 再打分"这套流水线的骨架做出来，还没有正式跑对比实验。

**DAG 流水线定型**（见 [dag_pipeline.md](dag_pipeline.md)、[dag_schema.md](dag_schema.md)）：

| 步骤 | 做什么 | 关键规则 |
|---|---|---|
| 1. 步骤切分 | `extract_steps_from_answer` | 换行为界；正则识别 `【小题N】`/`(N)`/`（N）`/`第N问` 等子题标记 |
| 2. 表达式/命题抽取 | `extract_expressions / extract_claims` | 先抓 `$...$` 和 `\(...\)`；再抓含 Unicode 运算符的等式；命题分四类：代数关系、几何 ∥/⊥/≅/∽、角度 ∠、∵/∴ 因果 |
| 3. 步骤类型 | `classify_step_type` | 7 种：definition / derivation / computation / conclusion / auxiliary / substitution / case_analysis |
| 4. 建边 | 首次出现原则 | 顺序边 weight=0.5（不参与打分）；依赖边 weight=1.0，类型分 `expr_ref / claim_ref / implicit` |
| 5. 打分 | `TopoReward` | 基础 0.4 + 无环 0.2 + 无孤儿结论 0.2 + 方向一致性 ×0.1 + 参考覆盖率 ×0.1 |

**工程脚手架**：`src/prm/` 和 `src/distill/` 模块骨架；ms-swift ORM 插件接入规范（见 [ms_swift_custom_reward.md](ms_swift_custom_reward.md)）；`TopoSCAEReward` 注册为 `topo_composite_scae`（reward-level 分层 clipping，见 [scae_implementation.md](scae_implementation.md)）。

**proposal 的三大挑战已明确**（见 [proposal_zh.md](proposal_zh.md)）：
- C1 过程奖励必须可验证（不能依赖 learned PRM）；
- C2 多源奖励聚合会导致 reward collapse —— 线性加权下 20 步后就有 93% batch 奖励方差归零；
- C3 推理链冗余要压缩。

这段时间**没有硬性模型对比结论**，重点是"管道跑通 + 奖励单测通过"。

---

## 2026.3.23：论文叙事重构 + 首轮评测铺开

这天是关键转折。论文叙事收敛到**两个并列主贡献**：

1. Deterministic Verifiable PRM（DAG + 拓扑/连续性奖励）；
2. Reverse-KL Reasoning Distillation（把 teacher 图增强推理压成 student 紧凑链式）。

当天同时在 8 张卡上铺开 light-200 评测（`data/test/light_middle_200.jsonl` 和 `light_high_200.jsonl` 各 200 条），结果陆续落盘到 `output/eval/*_metrics.json`。

**问题：数据集名不被 swift eval 认识**。`scripts/run_benchmark_light.py` 原本写的是 `math/cmath`，但 swift eval 只支持 `math_500/gsm8k`。把公开评测脚本的数据源换掉，问题解决。

**问题：`grpo_clipped` 起不来**，报 `ModuleNotFoundError: weave / vllm`。
**解决**：在 topoprm 环境里 `pip install weave==0.52.35 vllm 0.18.0`，三次 retry 后稳定进入训练（对应 `MASTER_PORT=29531`）。

---

## 2026.3.24 — 2026.3.27：主基线回填 + 评测卡点反复

### Private light-200 首版主表

| 模型 | Mid | High | Overall | Fmt |
|---|---|---|---|---|
| Qwen3-32B (zero-shot) | 16.0 | 7.0 | 11.5 | 64.5 |
| + SFT | 39.0 | 26.5 | 32.8 | 48.7 |
| + GRPO (outcome-only) | 19.5 | 13.0 | 16.3 | 89.0 |
| + GRPO (TopoPRM) | **37.2** | 22.4 | 29.8 | **94.6** |

TopoPRM 相对 outcome-only 在 32B 上拉开 +13.5 点，论文主表立得住。GRPO 加了格式和过程奖励后 Fmt 从 48.7 涨到 94.6，说明**格式奖励对可评测性非常关键**——SFT 的 Fmt 只有 48.7 时其实有近一半样本根本没法被 critique_eval 解析。

### Public benchmark 首批回填

| 模型 | GSM8K | MATH-500 |
|---|---|---|
| Llama-3.1-8B-Instruct | 84.5 | 52.2 |
| Qwen2.5-7B-Instruct | 93.0 | 74.2 |
| Qwen3-32B + grpo_clipped | 93.5 | — |
| Qwen3-32B + TopoPRM | 90.0 | **73.6** |
| Qwen3-32B + outcome-only | 85.0 | 75.4 |

### 评测链路本身比模型更难伺候

| 现象 | 根因 | 做法 |
|---|---|---|
| swift eval 长时间 `num_samples=0` | vLLM 线程/端口竞争 | `eval_num_proc=1`、`max_new_tokens=1536`、每次换独立端口（8133/8151/8166 等） |
| `LoRA rank 64 > max_lora_rank 16` | vLLM 默认 LoRA rank 上限 | 命令级追加 `--vllm_max_lora_rank 64` |
| GPU0 VRAM 残留 128GB | 驱动级僵尸进程（PID 2136374 在 `ps` 里看不见） | `kill -9` / `nvidia-smi --gpu-reset` 都无效，最终等驱动 GC 自释 |
| `grpo_mulgate / grpo_scae / grpo_confgate` 多次 SIGTERM | 资源抢占 / 主节点退出 | 记录到台账、清理失败日志、用空闲卡 retry2/3 |
| `grpo_clipped_light200` 全 0 分 | `MAX_NEW_TOKENS=128` 把 `<answer>` 截断，没闭合标签 | 放宽 `MAX_NEW_TOKENS` 到 1024+，并把截断检测改为语义判断（见 2026.4.7） |

**3.24 新增 5 指标**：`src/eval/critique_eval.py` 输出 `error_identification_precision/recall/avg_prediction_tokens`，以后可以稳定汇报 Acc/P/R/F1/#Tokens。

**3.27 `R_topo` 可审计化重构**：奖励拆成 `valid / acyclic / no_orphan / direction / step_align / ref_edge_f1` 六个子项加权，并保留 invalid-DAG hard gate。这是为了审稿人能对着公式一项一项核数。

---

## 2026.4.6：跨尺度消融 —— TopoPRM 在小模型上失效

我们发现在 Qwen3-32B 下面做实验有明显消融证据（+13.0 vs Outcome Only），但是转移到 9B 或者 7B 之后结论失效（原文见 [2026-04-06_cross_scale_analysis.md](2026-04-06_cross_scale_analysis.md)）。

### Cross-Scale Ablation Table

| Config | 32B-Mid | 32B-High | 32B-Avg | 9B-Mid | 9B-High | 9B-Avg | 7B-Mid | 7B-High | 7B-Avg |
|---|---|---|---|---|---|---|---|---|---|
| TopoPRM (full) | **0.330** | **0.255** | **0.292** | 0.305 | 0.270 | 0.287 | 0.140 | 0.045 | 0.092 |
| w/o Topology | 0.230 | 0.165 | 0.198 | **0.340** | 0.280 | **0.310** | 0.210 | 0.050 | 0.130 |
| w/o Continuity | 0.275 | 0.200 | 0.238 | 0.295 | **0.290** | 0.292 | 0.120 | 0.080 | 0.100 |
| Outcome Only | 0.195 | 0.130 | 0.163 | 0.315 | 0.250 | 0.282 | **0.220** | **0.080** | **0.150** |

| Config | 32B Δ | 9B Δ | 7B Δ |
|---|---|---|---|
| w/o Topology | **−9.5** | +2.3 | +3.8 |
| w/o Continuity | **−5.5** | +0.5 | +0.8 |
| Outcome Only | **−13.0** | −0.5 | **+5.8** |

对于小模型上 Topology 的奖励具有危害性，而 continuity 保持其作用。总结为 Topo 奖励函数设计有问题，接下来继续修改。我们后续使用分层门控的形式来融合 Topo，代替加权门控融合。

### 训练动态诊断

| Model | R_mean | R_std | ZeroStd | Clip | KL | Eval-Avg | 诊断 |
|---|---|---|---|---|---|---|---|
| 9B hier | 0.099 | 0.026 | 0.077 | 1.000 | 0.003 | 0.287 | 截断 + 信号弱 |
| 9B outcome | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.282 | 完全无梯度 |
| 7B hier | 0.225 | 0.010 | 0.017 | 0.461 | 0.586 | 0.092 | 有梯度但方向错 |
| 7B outcome | 0.000 | 0.000 | 1.000 | 0.034 | 0.056 | 0.150 | 无梯度 |

三种失败模式：9B 所有配置 Clip=1.0（**截断噪声**：所有生成都顶格到 `max_completion_length=1024`，不完整 DAG 污染 topo 分）；7B 所有消融 ZeroStd≈1.0（**奖励方差坍缩**：batch 内 4 个生成输出几乎相同，GRPO advantage=0 → 梯度为零，模型只靠 KL 漂回 SFT）；7B hier 奖励 0.225 最高但 eval 0.092 最差（**结构奖励 hacking**：模型学会把输出做长做"有结构"但答案是错的）。

### Format Compliance

| Config | 32B-Fmt | 9B-Fmt | 7B-FmtM | 7B-FmtH |
|---|---|---|---|---|
| TopoPRM | 0.91 | 0.86 | 0.380 | 0.230 |
| w/o Topology | 0.89 | 0.84 | 0.260 | 0.355 |
| w/o Continuity | 0.88 | 0.83 | 0.475 | 0.520 |
| Outcome Only | 0.87 | 0.83 | 0.195 | 0.165 |

惩罚格式错误对结果来说相当重要，7B 下模型的格式兼容性非常低（19–52%），模型根本没法稳定产出要求的输出结构。

### 当天就落地的修正

- 把乘性公式 `R = R_base · (1 + α·topo + (1-α)·cont)` 改成 outcome-gated：`R = R_out · (1 + gate(R_out)·(α·topo + (1-α)·cont))`，`gate = sigmoid(β·(R_out − τ))`。
- Truncation-robust topology：检测到生成顶格时把 topo 权重按 `(1 − clip_ratio)` 衰减。
- `max_completion_length` 从 1024 放到 4096（9B）。

---

## 2026.4.7：9B own-SFT 冷启动 + gated 奖励重设计 + 蒸馏意外爆发

原文见 [2026-04-07_session_summary.md](2026-04-07_session_summary.md)。

### 一个长期被忽略的坑：9B 一直在用 32B 的 SFT adapter

之前所有 9B GRPO 实验都是套 32B 的 `checkpoint-120` 做 LoRA adapter。32B 已经学好了格式，9B 套上这个 adapter 等于根本没学过输出格式，所以 gated reward 一直是 0、reward_std 一直是 0、梯度全无。

**解决**：新增 `configs/sft_qwen35_9b.yaml`（2 epochs，产出 checkpoint-626）和 `configs/sft_qwen25_7b_boost.yaml`（5 epochs，lr=1e-4，产出 checkpoint-1560）分别给 9B / 7B 做各自的 SFT 冷启动。**9B 用自己的 SFT adapter 后，gated reward_std 首次出现 0.0006 的非零值**，梯度终于能传到 policy 了。

### 9B mcl=4096 重跑

| Config | reward | reward_std | 解读 |
|---|---|---|---|
| **no_topo** | **0.069** | **0.025** | 梯度最强 |
| hier | 0.020 | 0.008 | 信号弱 |
| gated | 0.0125 | 0 | 格式有部分分但方差 0 |
| outcome | 0 | 0 | 完全无信号 |

和之前 mcl=1024 的结论一致：**9B 上 no_topo 比 full 还好**，topo 奖励在 9B 是负贡献。

### TopoGatedReward：零手调权重的新奖励

旧的 `(0.70, 0.15, 0.15)` 权重一写出来就被审稿人质疑"调参"。新形式直接从 OutcomeReward 的离散结构推出来：

```
R = outcome + δ · format · (1 + ε · q)
```

- `δ = 0.1`：因为 OutcomeReward 最小 gap 是 0.167，要求 `δ · (1+ε) < 0.167`；
- `ε = 0.5`：由上式反推；
- `q = TOPO_W · scale(topo) + (1 − TOPO_W) · scale(cont)`，`TOPO_W` 走环境变量 `TOPO_GATED_TOPO_W`（0 = 仅 continuity，1 = 仅 topo，默认 0.5）；
- 40040 对 Monte Carlo 验证，**0 排序违例**——correctness-primacy 可证。

配套改动：**FormatReward v2** 给课程化部分分（完整 `<think>+<answer>` → 1.0；仅 `<answer>` → 0.5；仅 `<think>` → 0.1；什么都没有 → 0）；**truncation 检测改成语义判断**（原来按 1800 字符硬阈值，在 mcl=4096 上直接把全部样本都标成截断，改成检查 `</answer>` 闭合标签是否存在）。

### 意外：蒸馏把 GRPO 全踩在地上

| 模型 | Mid | High | Overall |
|---|---|---|---|
| **distill_rkl_8b_compact** | **60.0** | **58.0** | **59.0** |
| GRPO 32B (full TopoPRM) | 37.2 | 22.4 | 29.8 |
| SFT 32B | 39.0 | 26.5 | 32.8 |

8B 蒸馏学生把 32B GRPO 甩开 29 点。这是第一次量化证明 **teacher signal ≫ reward shaping**——论文叙事从"TopoPRM 很强"转向"TopoPRM + distillation 两翼齐飞"。

---

## 2026.4.8：DAG 闭环升级

把 DAG 流水线里几处"能跑但不干净"的地方彻底收口：

- **句级 claim 抽取**：`extract_claims` 改为按句切分，过滤不完整残句（例如 "所有有两种运输方案："），避免依赖边被垃圾命题带偏。新增 `extract_claim_keys` 专用于依赖匹配。
- **hybrid verdict**：`parse_answer_to_dag_debug` 接受 `reference_dag`，`local_verdict` 先继承 reference，缺失时回退到规则判定 `correct / incorrect / unverifiable`。
- **adaptive 顺序弱边**：默认不再全量链接相邻步骤，只在缺强依赖时补边，并输出边来源统计。
- **TopoReward 逐项可审计**：暴露 `term_base / term_acyclic / term_orphan / term_delta / term_kappa` 五个分项，外加 `rho_orphan / delta / kappa / lambda_* / denom / r_topo` 诊断字段。写进论文公式时读者能对着数字核。无参考图时自动把 κ 项移除并重算归一化分母。
- **层压缩 + 可视化闭环**：新增 `compress_dag_by_layers`（每个拓扑层聚合成一个节点）；GUI 改成单视图 `layer_boxes`，同图展示顺序边 + 依赖边、层级虚线框、边按类型用不同曲率避免重叠、显示 `Lk → Ck` 层到压缩链映射、压缩率统计——"拓扑奖励 → 推理压缩"整条叙事终于有图可以看。
- 回归：`tests/test_build_dag.py`、`test_graph.py`、`TestTopoReward` 全部通过。

---

## 2026.4.10 — 2026.4.11：公开 benchmark 自动化 + Math-7B 对照

原文见 [exp_completion_20260410.md](exp_completion_20260410.md) 和 [exp_general_benchmark_20260411.md](exp_general_benchmark_20260411.md)。

### 论文表格同步流水线

- `scripts/export_benchmark_metric_json.py`：从 `output/eval/benchmark_light/**/reports/*.json` 物化 metrics；
- `src/eval/collect_experiment_results.py`：过滤全 0 的 private 行（那些格式崩盘导致全 0 的 baseline 不该进主表）；
- `src/eval/sync_paper_tables.py`：自动回填 `public_results.tex / aggregation_ablation.tex / structural_metrics.tex / case_study.tex`。

### 问题：`swift eval` + vLLM + Qwen3.5-9B LoRA 挂死 4h+

**根因**：swift 训出来的 adapter 里 `target_modules` 前缀是 `model.language_model.*`，但 `transformers` 裸加载时模块名是 `model.layers.*`，对不上，加载 hang 死。

**解决**：写 `scripts/bench_transformers.py`，用纯 transformers `model.generate()` + 自动 patch `adapter_config.json` 和 `adapter_model.safetensors` 的 key 前缀。这个脚本后来成了所有公开 benchmark 的主跑路径，swift eval 这条线彻底弃用。

### 公开 benchmark 结果（transformers backend, greedy）

| 模型 | GSM8K | MATH-500 |
|---|---|---|
| Qwen3.5-9B + SFT | **90.4** | **50.8** |
| Qwen3.5-9B + GRPO no_topo mcl4096 | 82.3 | 33.6 |
| Qwen2.5-Math-7B + SFT | 57.2 | 47.4 |
| Qwen3-8B + RKL distill（有 padding bug）| 28.5 | 27.0 |

`aggregation_ablation` 给出关键数据：`frac_reward_zero_std` 均值，linear=75.1%，clipped=67.7%，hierarchical=37.9%——**层次化聚合的主要价值就是压住方差坍缩**，不是在提准确率。

Math-7B 的"数学专用性"在本任务上没赢过通用 9B，SFT 后 GSM8K 才 57.2% vs 9B 的 90.4%，GRPO 不大可能补上差距——这条线放次优先级。

---

## 2026.4.16 — 2026.4.17：统一评测 + 投稿润色 + 终盘决策

### 自动化评测

`todo_exp_ours.sh` 做成多卡轮询 + 并发槽位控制，一条命令跑完 9 个 benchmark 并汇总进 `unified_benchmark_summary.csv`；`collect → sync → invariants` 自动链路校验通过。论文 `4_experiments.tex` 切到统一 benchmark + 统一指标（error / correct / F1 / pass@k / maj@k / prm@k / #Tokens）。

### 论文润色

修 4 个缺失 BibTeX 键、清 80 条未引用条目（102 → 22），修 `main.tex` 包重复、匿名模式、作者占位符；补 `tables/unified_metrics.tex`；写完整中文 [topoprm_paper/proposal.md](../topoprm_paper/proposal.md)；Method 清掉 200+ 行旧注释，补 Appendix Additional Results；生成 3 张图的详细绘制 prompt（`figure_prompts.md`）。

### Batch 2 实测（ablation + 7B family）

| 模型 | 参数 | GSM8K | MATH-500 | Cor/Err (GSM8K) | AvgTok |
|---|---|---|---|---|---|
| outcome_only_9b | 9B | 88.3 | 54.4 | 1165/154 | 287 |
| no_topo_9b | 9B | 88.7 | 54.2 | 1170/149 | 305 |
| **no_continuity_9b** | 9B | **90.9** | 55.0 | 1199/120 | 1006 |
| base_qwen25_7b | 7B | 84.2 | 55.2 | 1110/209 | 1917 |
| topoprm_hier_7b | 7B | 83.8 | 38.8 | 1105/214 | 1753 |
| distill_rkl_8b (MATH500 fix) | 8B | 81.0 | 45.4 | 1069/250 | 2048 |

`no_continuity_9b` 在 GSM8K 反而最高，但 tok=1006 直接回到 base 水平——说明 **continuity reward 是长度压缩的主力**，topology reward 负责结构；把 continuity 去掉模型就退回到冗长的 base 行为。`topoprm_hier_7b` 在 MATH-500 从 base 的 55.2 掉到 38.8，**7B 容量不够同时优化答案和结构**。

### GSM8K + MATH-500 统一评测

| 模型 | GSM8K | MATH-500 | AvgTok(GSM8K) | Time(s) |
|---|---|---|---|---|
| base_9b (Qwen3.5-9B) | **91.0** | 55.0 | 1017 | 5198 |
| sft_9b | 88.0 | 53.0 | 275 | 1439 |
| topoprm_hier_9b | 87.7 | 53.4 | 277 | 1412 |
| **topoprm_gated_9b** | 87.8 | **55.4** | 303 | 1535 |
| distill_rkl_8b | 81.0 | 27.0 | 2048 | 8435 |

几个直接结论：

- **topoprm_gated_9b 在 MATH-500 上 55.4 > base 55.0**，同时 tok 只要 303 vs 1017；Acc/kTok 指标 27.3，全家族最高。这是我们当前的主亮点。
- **SFT/GRPO 变体比 base 短 3–4 倍**，精度只掉 2–3 点——这是格式成本不是过拟合。因为我们的训练强制 `<think>...<answer>`，而 base 可以无限自由思考；速度差就是输出长度差，完全线性。
- **distill_rkl_8b 持续拉胯**：tok 一直顶格 2048，学生从没学会停止。根因定位：32B teacher trace 里只有 **0.4% 有完整 `<answer>` 闭合标签**，学生看到的绝大多数训练样本都是截断的，根本不知道在哪停。

### 战略决策：弃用 Qwen3-8B 的 reverse-KL 蒸馏

改走 SFT-based 蒸馏到 Qwen3.5-4B/2B/0.8B，数据换成 `train_mixed.jsonl`（10847 条，70% 带完整 `<think>/<answer>`）。`sft_distill_4b/2b/0p8b.yaml` 已配好，`scripts/launch_distill_when_ready.sh` 等 GPU 0–5 空闲自启。但从 SD-Zero 阅读后（见下节），我们倾向于把这条 SFT 蒸馏线换成 **self-revision 驱动的 on-policy 自蒸馏**。

---

## 贯穿全项目的失败模式与解法（一张表看完）

| 类型 | 典型表现 | 解法 |
|---|---|---|
| Reward 方差坍缩 | 线性聚合下 93% batch R_std=0，GRPO 无梯度 | batch min-max rescale + 层次化聚合 + gated 公式（δ=0.1, ε=0.5 由 outcome 结构反推，非调参） |
| Reward hacking（7B hier） | 结构分高但 eval 最差 | outcome-gated：`gate = sigmoid(β·(R_out − τ))` |
| 截断噪声（9B Clip=1.0） | 不完整 DAG 污染 topo 分 | 语义截断检测（看 `</answer>`）+ mcl 1024→4096 + truncation-robust 权重衰减 |
| 格式崩盘 | `<answer>` 无闭合 / 全 0 分 | FormatReward v2 课程化 + MAX_NEW_TOKENS 放宽 |
| 跨模型 adapter | 9B 用 32B 的 SFT 权重 → reward 全 0 | 每个 base 独立 SFT 冷启动（9B own, 7B boost） |
| swift eval 卡死 | vLLM LoRA hang 4h+，num_samples=0 | `bench_transformers.py`（纯 transformers 后端）+ adapter key 自动 patch |
| GPU 僵尸进程 | VRAM 128GB 残留，`ps` 看不见 PID | `kill -9` / `--gpu-reset` 不支持，等驱动 GC；台账策略改为"跑过都记" |
| 蒸馏学生不停机 | tok 顶格 2048 | teacher trace 必须有 `<answer>` 闭合率；弃用 RKL，改 on-policy 自蒸馏（见下节） |

---

# 未来方向：把 SD-Zero 融进蒸馏

## 2026.4.17：SD-Zero 阅读要点

SD-Zero（Princeton 等，COLM 2026 投稿，源码在 [topoprm_paper/SD-Zero/](../topoprm_paper/SD-Zero)）的核心思路和我们现在的蒸馏失败点正好对症。

**一句话概括**：**一个模型扮演两个角色** —— generator 产出初稿 $y_\text{init}$，reviser 看着 $(x, y_\text{init}, r)$ 写修订 $y_\text{revised}$；然后用 on-policy 自蒸馏，把 reviser 的 token 分布蒸回 generator。

它分两个 phase：

| 阶段 | 做什么 | 损失 |
|---|---|---|
| Phase 1: SRT | 采 $y_\text{init}$，用 binary reward $r$ 选 prompt：错的加 "Wait, this response is not correct, let me start over."，对的加 "Let me rephrase the above solution."，同一个模型生成 $y_\text{revised}$。过滤出 $D_\text{revision} = \{(x, y_\text{init}, P_r, y_\text{revised}) : r(y_\text{revised})=1\}$。 | $L_\text{SRT} = L_\text{revision} + L_\text{generation}$：前者是 $-\log \pi_\theta(y_\text{rev} \mid x, y_\text{init}, P_r)$（教修订），后者是 $-\log \pi_\theta([y_\text{init}, P_r, y_\text{rev}] \mid x)$（保生成）。两项互补。 |
| Phase 2: OPSD | 冻结 Phase 1 模型当 reviser teacher；generator（正在训的 student）on-policy 产 $y$；teacher 看 $(x, y, P_r)$ 给 token 分布；student 用 reverse KL 匹配。 | $L_\text{OPSD} = \mathbb{E}\sum_t \mathrm{KL}\big(\pi_\theta(\cdot \mid x, y_{<t}) \,\|\, \pi_{\theta_\text{SRT}}(\cdot \mid x, y, P_r, y_{<t})\big)$ |

### 它报告的两个有意思的性质

1. **Token-level self-localization**：reviser 虽然只拿到 binary reward，但 KL 集中在少数几个"需要改"的 token 上——本质是 binary reward 自动做了 token 级 credit assignment。
2. **Iterative self-evolution**：每跑完一轮 OPSD，同步 teacher = 当前 student，再跑一轮还能再涨 3%+。

### 数字

Qwen3-4B / Olmo3-7B 在 8 个 math/code benchmark 上平均涨 10%+，超 RFT / GRPO / SDFT 至少 5%；response 长度只有 SFT 的 ~1/2。

## 为什么这个对我们很合适

对照我们蒸馏的失败根因（32B teacher 0.4% 有闭合 `<answer>`，学生永远学不会停），SD-Zero 能从三个点补上：

1. **不再依赖 32B teacher 的离线轨迹**。我们之前是把 32B 的 trace 批量存下来做 RKL，`max_new_tokens` 一截断 teacher 就烂了。SD-Zero 是 **on-policy**，student 自己产的 $y$ 由同一个模型（或 Phase 1 冻结的 reviser）即时给 token 监督，不需要预先生成大量高质量 trace。
2. **Binary reward → dense token KL 的转换**正是我们当前缺的环节。我们的 `TopoGatedReward` 给的是 *单个标量*，粒度太粗；SD-Zero 通过 reviser 的 token 分布把标量细化到每个 token 的 KL，能放大"错在哪里"的局部信号。
3. **Reviser 的条件输入天然能装我们的过程信号**。SD-Zero 只把 binary reward 塞进 $P_r$；我们可以把 **outcome + topology gate + continuity gate** 都塞进 $P_r$（例如 "Your critique is correct but step 3 has no dependency to earlier steps, rewrite with explicit reference."），让 reviser 的 token 分布天然携带结构信号，得到 outcome-gated + process-aware 的 dense supervision。这是对 SD-Zero 的自然扩展——SD-Zero 只有 binary signal，我们用 DAG 可以给更丰富的 gate 信号。

## 实验计划：TopoSD-Zero

下面是建议的落地方案，分三个里程碑，每个都可以独立验收。

### 里程碑 1（2026.4.18 — 2026.4.22）：复现 SD-Zero 在 critique 任务上

目标：验证 SD-Zero 的 self-revision 路线在 **critique 任务 + 我们现有数据** 上能跑通，先不引入任何新奖励。

- **Base**：Qwen3.5-9B SFT (`output/sft_qwen35_9b/checkpoint-626`)——已经会格式，是现成的 Phase 0 起点。
- **Phase 1 数据构造**：对 `data/grpo_ready/train.jsonl` 里 ~10k 道题，每题采 N=4 个 $y_\text{init}$，用 `OutcomeReward` 算 $r$：
  - $r=0$ → $P_r$ = "Wait, this critique is incorrect, let me redo it."
  - $r=1$ → $P_r$ = "Let me rewrite this critique more concisely."
  
  同模型生成 $y_\text{revised}$，只保留 `r(y_revised)=1` 且格式合规的样本，目标收到 6k 条。
- **Phase 1 训练**：$L_\text{revision} + L_\text{generation}$ 联合，LoRA rank 64，2 epochs。产出 `sft_srt_9b/checkpoint-*`。
- **Phase 2 训练**：$\theta := \theta_\text{SRT}$，teacher 冻结；on-policy 采样 1 response/题；token-wise reverse KL，1 epoch。
- **评测**：`bash todo_exp_ours.sh --phase eval`，和 `distill_rkl_8b / sft_9b / topoprm_gated_9b` 直接对比。
- **验收指标**：
  - private light-200 Overall ≥ 0.50（9B SRT+OPSD 的下限，现在 8B RKL 蒸馏是 0.59）；
  - GSM8K ≥ 88.0（对齐 sft_9b）；
  - AvgTok ≤ 360（比 sft_9b 的 275 膨胀不超过 30%）。

### 里程碑 2（2026.4.23 — 2026.4.27）：Topo-aware Reviser

目标：把我们的 topology / continuity 信号塞进 reviser 的 $P_r$，验证"process-aware $P_r$"能不能让 token KL 更聚焦结构错误的位置。这是 TopoSD-Zero 相对 SD-Zero 的独特贡献。

- **$P_r$ 四档扩展**：根据 $(R_\text{outcome}, R_\text{topo}, R_\text{cont})$ 的象限选模板：

  | 象限 | 判据 | $P_r$ 模板 |
  |---|---|---|
  | Ⅰ outcome ✓ + 结构 ✓ | $R_o=1$ and $R_\text{topo} \ge 0.5$ | "Rephrase this correct critique more concisely." |
  | Ⅱ outcome ✓ + 结构 ✗ | $R_o=1$ and $R_\text{topo} < 0.5$ | "Your answer is correct but step {k} has no dependency to earlier steps. Rewrite with explicit references." |
  | Ⅲ outcome ✗ + 结构 ✓ | $R_o=0$ and $R_\text{topo} \ge 0.5$ | "Your reasoning looks well-structured but the final verdict is wrong. Reconsider." |
  | Ⅳ outcome ✗ + 结构 ✗ | $R_o=0$ and $R_\text{topo} < 0.5$ | "Let me redo this from scratch." |

  其中 Ⅱ 的 $\{k\}$ 直接从 DAG 的 `orphan_nodes()` 取第一个孤儿结论节点的 `step_id`，做成动态模板。

- **Reviser 训练**：Phase 1 数据按这四档均衡采 6–8k，训出 `topo_srt_9b`。
- **Phase 2**：teacher 保持 topo-aware $P_r$ 条件，student 只看 $x$；OPSD 的 KL 天然把 topology 信号分摊到 token。
- **新诊断指标**：按 SD-Zero 的 token-level credit assignment 分析方法，分 20 bucket 看 KL 质量分布；对 Ⅱ 类样本（outcome=1, structure=0），检查高 KL token 是否集中在结构断裂位置（孤儿结论节点所在的 span）。这里我们**有 DAG 作为 ground truth**，可以定量评价 token-level localization 准确率——这个指标是 SD-Zero 没法做的，我们能做。
- **验收指标**：
  - private light-200 Overall ≥ 里程碑 1 结果 + 2.0 点；
  - Ⅱ 类样本上高 KL token 命中孤儿节点 span 的 precision ≥ 0.4（作为 token-level localization 的定量证据）；
  - Token-level KL 分布的 Gini 系数 ≥ 0.6（SD-Zero 的 KL 分布明显偏斜，至少要复现这个性质）。

### 里程碑 3（2026.4.28 — 2026.5.05）：Iterative Self-Evolution + 小模型蒸馏

- **Iteration**：按 SD-Zero 的做法，OPSD 跑完一轮后 `teacher := student`，再跑一轮，预期再涨 2–3 点。上限 3 轮避免过拟合。每轮换用不同的 question split（把 train set 切成 $N_1 / N_2$，每轮重切）。
- **小模型迁移**：把 iteration-2 的 9B student 当新 teacher，蒸到 Qwen3.5-4B / 2B / 0.8B。**这次 teacher 轨迹是 on-policy 产生的、有完整 `<answer>`**（不是原来 32B 那种 0.4% 闭合率），学生能学到 stop。用现有的 `sft_distill_*.yaml` 跑 SFT-based 蒸馏。
- **验收**：4B 学生在 private light-200 Overall ≥ 0.55（逼近现在 8B RKL 蒸馏的 0.59），AvgTok ≤ 500；2B 学生 Overall ≥ 0.45；0.8B 学生作边界案例，看能撑到什么程度。

### 实现上需要加的代码

- `src/distill/self_revision.py`：Phase 1 数据构造（采样、判 reward、建 $P_r$、过滤）。
- `src/distill/opsd_trainer.py`：on-policy rollout + token-wise reverse KL，可以基于 ms-swift 的 GRPO trainer 改。
- `src/distill/process_aware_prompt.py`：把 $(R_\text{out}, R_\text{topo}, R_\text{cont})$ 映射到四档 $P_r$ 模板，动态嵌入孤儿节点 `step_id`。
- `configs/opsd_9b.yaml`、`configs/opsd_9b_topo.yaml`、`configs/opsd_9b_iter.yaml`。
- 评测侧：扩展 `scripts/bench_transformers.py` 支持 Generate-then-Revise 指标（first attempt accuracy vs revised attempt accuracy），对齐 SD-Zero 的 Table 1 口径。

### 风险与对策

| 风险 | 对策 |
|---|---|
| Phase 1 采样预算爆炸（每题 N=4 生成） | 先用 1k 题做 pilot 验证数据质量再放量；SD-Zero 报告 6k 条就够 |
| 9B SRT 后响应变长（SD-Zero 也遇到） | 这是正常现象，Phase 2 正好用来压回去；观察 `behavior_evolution` 曲线做早停 |
| Topo-aware $P_r$ 过度设计导致 reviser 过拟合某一档 | 四档数据强制均衡；$P_r$ 做成短模板，避免 reviser 记死具体文本 |
| Iteration 过拟合训练集 | 每轮重切 $N_1 / N_2$；每轮单独记 eval 曲线，达到 plateau 就停 |
| Reviser teacher 本身给错信号（Ⅱ 类样本 $P_r$ 错指位置） | 在 Phase 1 数据构造阶段就用 `TopoReward` 诊断字段校验 $P_r$ 和 DAG 一致；不一致的样本丢掉 |

---

## 当前状态快照（2026.4.17）

- 9B 家族主表实测值已全部回填论文（见 `topoprm_paper/tables/public_results.tex` 和 `unified_metrics.tex`）。
- **topoprm_gated_9b 是 MATH-500 上 Acc/kTok 与绝对 acc 双料 SOTA**（55.4, Acc/kTok=27.3）。
- Qwen3.5-4B/2B/0.8B SFT 蒸馏配置已就绪，等 GPU 空闲自启；但倾向于换成 TopoSD-Zero 路线。
- 待执行：`bash todo_exp_ours.sh --phase eval → --phase sync`，把论文里还打 `~` 的估计值替换为真实数据。
- 下一步优先级：按里程碑 1 启动 TopoSD-Zero Phase 1 数据构造。
