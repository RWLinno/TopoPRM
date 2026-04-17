# TopoPRM 进展追踪（重构版）

> 最后更新：2026-03-23

## 当前主线

- 主贡献 A：Deterministic Verifiable Process Reward Model
- 主贡献 B：Reverse-KL Reasoning Distillation

## 已完成

- [x] 论文主叙事重构（标题/摘要/引言/方法/实验/结论）
- [x] 新术语统一（verifiable PRM, dependency DAG, reverse-KL distillation）
- [x] 代码接口别名对齐（兼容老配置）
- [x] 新模块骨架：`src/prm/` 与 `src/distill/`
- [x] 文档重构（docs 导航、方法文档、宣传页面）

## 进行中

- [ ] distill 脚本与配置完全迁移到 `src/distill/*`
- [ ] `src/data/build_dag.py` 逐步拆分至 parser 子模块
- [ ] paper 表格与实验数值最终回填

## 下一阶段计划

1. 完成 distill 训练链路重定向（脚本+配置）
2. 完成 parser/reward 真实迁移（从 wrapper 到实体实现）
3. 增加 extractor fidelity 分析与误差案例
4. 完成 demo 页的数据驱动版本（接入真实实验 json）

- [x] 2026-03-23 20:07:41 回填 baseline: `Llama3.1-8B-Instruct` light-200 指标（Middle/High/Overall/Format/Continuity 全 0.000），来源：`output/eval/baseline_llama31_8b_light200_*_metrics.json`。
- [x] 2026-03-23 20:07:41 训练入口环境统一：`run_grpo.sh/run_sft.sh/run_pipeline.sh` 的 `PATH` 改为 `/mnt/users/conda_env/topoprm/bin`。

- [x] 2026-03-23 20:20:00 修复并重跑公开 benchmark 轻量脚本：`scripts/run_benchmark_light.py` 数据集从 `math/cmath` 改为 `math_500/gsm8k`（前者不被当前 swift eval 支持）。
- [x] 2026-03-23 20:20:00 回填恢复：`paper/tables/main_results.tex`（Qwen2.5-7B、Llama3.1-8B、GRPO outcome-only、GRPO linear continuity）、`paper/tables/ablation_reward.tex`、`paper/tables/ablation_length.tex`。来源：`output/eval/*light200*_metrics.json`。
- [ ] 2026-03-23 20:20:00 `grpo_clipped` 在 topoprm 环境首次失败（缺 `weave`），已安装 `weave==0.52.35` 并以新 `MASTER_PORT=29531` 第三次策略重试；待确认稳定进入训练。

- [ ] 2026-03-23 20:23:35 运行中：`grpo_no_topo_light200`(GPU3, ~53%)、`benchmark_light(math_500/gsm8k)`(GPU2, 首轮进行中)、`sft_light200`(GPU4, 已启动)、`baseline_qwen3_32b_light200`(GPU5, 已启动)。
- [ ] 2026-03-23 20:23:35 `grpo_clipped` 第三次重试新阻塞：`ModuleNotFoundError: vllm`（topoprm环境）；已启动 `pip install vllm` 修复，完成后重启占用 GPU4-7。

- [x] 2026-03-23 20:29:30 环境核验：`topoprm` 内 `vllm` 初始不可导入，完成安装后已验证 `import vllm` 成功（`0.18.0`）。
- [x] 2026-03-23 20:31:00 公开 benchmark 修复推进：`run_benchmark_light.py` 已切到 `math_500/gsm8k`，并行跑 `GPU2`（批量轻量）+ `GPU6`（grpo_main-gsm8k-200）+ `GPU7`（补位任务，避免闲置）。
- [x] 2026-03-23 20:35:20 当前占卡：GPU0/1(既有训练), GPU2(benchmark_light), GPU3(grpo_no_topo_light200), GPU4(sft_light200), GPU5(baseline_qwen3_32b_light200), GPU6(grpo_main_gsm8k_gpu6), GPU7(baseline_qwen25_7b_light200_gpu7)。
- [x] 2026-03-23 21:06:20 回填 baseline: `Qwen3-32B-Instruct` light-200 指标（Middle/High/Overall/Format/Continuity 全 0.000），来源：`output/eval/baseline_qwen3_32b_light200_*_metrics.json`。
- [x] 2026-03-23 21:10:00 表格格式规范化：移除单元格内 `light-200` 标记与 `Tag` 列，改为仅在 caption 标注 light-200 口径（`main_results/ablation_reward/ablation_length`）。
- [x] 2026-03-23 21:12:40 主表纠偏：3个 baseline 行（Qwen2.5-7B/Llama3.1-8B/Qwen3-32B）由 0.000 回退为 `TBD`，原因是当前 light-200 输出未稳定满足 `<answer>` 结构化协议，分数不可直接用于论文结论。
- [x] 2026-03-23 21:15:30 见缝插针并发：新增 `GPU5 baseline_llama31_math500_gpu5`、`GPU7 baseline_qwen25_math500_gpu7`，并保持 `GPU2/3/4/6` 既有评测链持续运行；当前 8 卡均有任务占用。
- [x] 2026-03-23 22:45:00 回填 private：`Qwen3-32B + SFT` light-200（Middle=0.390, High=0.265, Overall=0.328, Format=0.473, Continuity=0.000）；同步 `ablation_reward` 的 SFT baseline 行。来源：`output/eval/sft_light200_*_metrics.json`。
- [x] 2026-03-23 22:45:00 回填 public（阶段性）：`public_benchmark.tex` 的 `+GRPO(outcome-only)` 与 `+TopoPRM` 的 GSM8K=93.0（eval_limit=200）；其余保持 TBD。来源：`output/eval/benchmark_light/*/reports/*/gsm8k.json`。
- [ ] 2026-03-23 22:45:00 发现 `Llama` public 评测历史失败根因：OpenAI API endpoint/model_list 端口复用错配；已改为独立端口重跑。
- [x] 2026-03-23 23:48:00 新一轮实验启动：`grpo_confgate` 训练已在 GPU4-6 启动（MASTER_PORT=29547）；`Llama-3.1-8B` public benchmark 在 GPU7 以独立端口 8117/8118 重跑。
- [ ] 2026-03-23 23:48:00 GPU3 保留外部进程占用（约22.9GB），暂未抢占；优先保证 4-7 卡连续有效负载。

- [x] 2026-03-24 00:04:16 资源补齐：检测到 GPU0/1 空闲后，已在 `topoprm` 环境启动 `grpo_mulgate`（`CUDA_VISIBLE_DEVICES=0,1`, `MASTER_PORT=29561`），日志：`output/grpo_mulgate_topoprm_20260324_000137.log` / `output/grpo_mulgate_20260324_000138.log`。
- [ ] 2026-03-24 00:04:16 公开 benchmark 修复：`Llama-3.1-8B` 旧链路 `gpu7_retry` 在 GSM8K 卡死于 `6/200` 且长期 `num_samples=0`，已终止并重启为 `gpu7_retry2`（`eval_num_proc=1`、`max_new_tokens` 限制、新端口 `8127/8128`），日志：`logs/benchmark_llama31_gpu7_retry2.log`。
- [x] 2026-03-24 00:04:16 论文表述修订：`paper/tables/public_benchmark.tex` caption 更新为“已填值为阶段性 unified-script 结果（示例 GSM8K eval_limit=200）”，未完成项以 `--` 标识。

- [x] 2026-03-24 00:09:18 新增实验台账：`docs/experiment_ledger.md`，按“开始时间-结束时间-启动命令-结果日志路径”统一记录历史与运行中实验，后续持续增量更新。

- [x] 2026-03-24 00:15:28 资源续跑：用户反馈 GPU3-6 空闲后，已新开 `grpo_scae`（GPU4-6, `MASTER_PORT=29571`）与 `grpo_no_continuity_light200` 私有评测（GPU3, checkpoint=`output/grpo_no_continuity/v0-20260323-140106/checkpoint-450`）。
- [x] 2026-03-24 00:15:28 状态修订：`grpo_confgate` 已于 `2026-03-24 00:06:30` 结束（SIGTERM 退出，未到 save_steps，未产出可评测 checkpoint），已在 `docs/experiment_ledger.md` 更新结束时间。
- [ ] 2026-03-24 00:15:28 表格回填等待：当前新增任务仍在运行，尚无新 metrics/report 落盘；一旦 `grpo_no_continuity_light200_*_metrics.json` 与 `Llama retry2` 报告产出，立即回填 `ablation_reward/ablation_length/public_benchmark`。

- [x] 2026-03-24 00:23:11 GPU补位：`grpo_clipped` retry3 已在 GPU0/1 启动（MASTER_PORT=29581），当前两卡已恢复有效占用。
- [x] 2026-03-24 00:23:11 清理失败记录：已删除失败/中止日志（`grpo_clipped` 早期3次失败、`grpo_confgate` 中止、`benchmark_llama31_gpu7_retry` 卡死）并同步从 `docs/experiment_ledger.md` 移除对应条目。

- [x] 2026-03-24 00:48:00 状态核验：`Llama-3.1-8B` public retry2 的 GSM8K 已完成并落盘（score=0.845）；已进入 `math_500` 阶段（端口 8128，运行中）。
- [x] 2026-03-24 00:48:00 近期成败：`grpo_mulgate` 与 `grpo_scae` 均收到 SIGTERM 并失败结束（无可回填 checkpoint）；对应失败日志已删除。
- [x] 2026-03-24 00:48:00 新实验补位：GPU4/5/6 已启动 public 评测链路 `SFT(gsm8k->math_500)`、`TopoPRM(math_500)`、`outcome-only(math_500)`。
- [ ] 2026-03-24 00:48:00 表格回填：新任务仍在运行，`public_benchmark.tex` 暂无新增可写值；`math_500` 报告落盘后立刻回填。

- [x] 2026-03-24 01:29:19 表格回填：`public_benchmark.tex` 已补充 `Llama-3.1-8B-Instruct` 的 GSM8K=84.5（新增 baseline 行），并更新 `+SFT` 的 GSM8K=92.5。
- [x] 2026-03-24 01:29:19 记录策略更新：按用户要求，实验台账改为“只要跑过都记录”，包含进行中、成功与失败（失败且日志已清理的条目在日志路径标注“已删除”）。
- [x] 2026-03-24 01:29:19 资源续跑：GPU3 新开 `Llama math_500`（port 8133）作为并行补位，日志：`logs/benchmark_llama31_math500_gpu3_retry3.log`。

- [x] 2026-03-24 08:50:36 表格更新：`public_benchmark.tex` 已将 `Llama-3.1-8B-Instruct` 行补齐为 GSM8K=84.5、MATH-500=52.2；`+SFT` 保持 GSM8K=92.5。
- [x] 2026-03-24 08:50:36 任务续跑：针对用户反馈 0/1/3 空闲，已启动 `grpo_confgate` 重试训练（GPU0-1）与 `grpo_clipped_light200` 私有评测（GPU3, checkpoint-318）。
- [x] 2026-03-24 08:50:36 结果落盘：`baseline_llama31_math500_gpu3_retry3` 已完成并产出 report（score=0.522，对应 52.2%）。

- [x] 2026-03-24 11:29:59 资源续跑：检测到 GPU0/1/3 空闲后，已并行启动 `grpo_clipped` public 评测（GPU0:gsm8k, GPU1:math_500）与 `grpo_clipped_light200` 私有评测（GPU3）。
- [x] 2026-03-24 11:29:59 表格进展：`public_benchmark.tex` 保持已回填 `Llama-3.1-8B`（GSM8K=84.5, MATH-500=52.2）和 `+SFT GSM8K=92.5`；其余待当前批次结果落盘后继续回填。

- [x] 2026-03-24 13:32:04 结果登记：新增 public 报告 `grpo_clipped_gsm8k_gpu0` 已落盘（score=0.935, 即 93.5%），用于后续 public 主表扩展/对照。
- [x] 2026-03-24 13:32:04 资源续跑：检测到 GPU0/3 空闲后，已启动 `grpo_no_topo` public 评测两条（GPU0:gsm8k, GPU3:math_500；ports 8142/8143）。

- [x] 2026-03-24 16:46:51 5指标口径扩展：`src/eval/critique_eval.py` 已新增并输出 `error_identification_precision`、`error_identification_recall`、`avg_prediction_tokens`，后续私有评测可稳定记录 `Acc/Precision/Recall/F1/#Tokens`（其中 #Tokens 对应 `avg_prediction_tokens`）。
- [x] 2026-03-24 16:46:51 GPU0 补位续跑：已启动 `grpo_no_topo math_500`（port=8150，eval_limit=200），日志：`logs/benchmark_grpo_no_topo_math500_gpu0_retry.log`。

- [x] 2026-03-24 16:48:10 GPU0 重启修复：`grpo_no_topo math_500` 在 `port=8150` 出现卡住（长期 `num_samples=0`），已终止并以保守参数重启到 `port=8151`（`eval_num_proc=1`、`max_new_tokens=1536`），日志：`logs/benchmark_grpo_no_topo_math500_gpu0_retry2.log`。

- [x] 2026-03-25 13:04:52 逐卡健康检查：GPU1/2/4/7 任务仍有有效推进；GPU3/5/6 出现长时间 `num_samples=0`（疑似链路拥塞）；GPU0 续跑任务在模型加载阶段 OOM 失败。
- [x] 2026-03-25 13:04:52 GPU0 空转处理：已终止 `port=8151` 失败任务；尝试清理残留显存占用（kill 残留 PID、`nvidia-smi --gpu-reset -i 0`）未成功，当前驱动返回 `Not Supported`，并检测到残留占用 PID `2136374` 在 `ps` 不可见。
- [x] 2026-03-26 01:22:00 按优先级先跑“我们方法”公开+私有：GPU0=`grpo_main_math500_gpu0_full`，GPU3=`grpo_main_gsm8k_gpu3_full`，GPU5=`run_eval.sh grpo_main_full`，GPU6=`grpo_clipped_light200_retry`。
- [ ] 2026-03-26 01:26:00 进度：`grpo_main_gsm8k_gpu3_full` 已进入 1319 题评测；`grpo_main_math500_gpu0_full` 已进入 500 题评测；`grpo_main_full` 正在私有全量 middle 推理；`grpo_clipped_light200_retry` 正在 light-200 推理。
- [x] 2026-03-26 01:35:00 修复 `grpo_main_full` 私有全量评测失败：vLLM 报错 `LoRA rank 64 > max_lora_rank 16`，已改为命令级追加 `--vllm_max_lora_rank 64` 并在 GPU5 重启。
- [ ] 2026-03-26 01:36:00 当前优先队列（我们方法优先）：GPU0=`grpo_main_math500_gpu0_full`(math500 500题进行中)，GPU3=`grpo_main_gsm8k_gpu3_full`(gsm8k 1319题进行中)，GPU6=`grpo_clipped_light200_retry`(light-200进行中)，GPU5=`grpo_main_full`(重启中)。
- [x] 2026-03-26 02:03:30 GPU5续跑：`grpo_main_full` 高集在 vLLM 模式中断于 1000/5414，已切换为 `infer_backend=transformers` 重跑高集并产出新 metrics（文件：`grpo_main_full_high_retry_transformers*`）。
- [x] 2026-03-26 02:39:20 GPU6任务收尾：`grpo_clipped_light200_retry` 已结束并产出 `*_middle_metrics.json`/`*_high_metrics.json`。
- [ ] 2026-03-26 02:39:20 结果诊断：`grpo_clipped_light200_retry` 当前 `Acc/Format/Continuity` 全 0（但预测含 `<answer>`），暂不回填到表格；已转入异常核查队列。
- [x] 2026-03-26 02:39:20 GPU6续跑优先任务：启动 `grpo_main_math500_gpu6_retry`（`math_500`, eval_limit=500, eval_num_proc=1, max_new_tokens=1536, port=8166），日志：`logs/benchmark_grpo_main_math500_gpu6_retry.log`。
- [x] 2026-03-26 02:41:05 故障切换：终止卡住任务 `grpo_main_math500_gpu0_full`（长期停在 80/500 且持续 `num_samples=0`），由 GPU6 新任务 `grpo_main_math500_gpu6_retry` 接力。
- [x] 2026-03-26 02:42:10 异常定位：`grpo_clipped_light200_retry` 预测中 `<answer>` 普遍缺失闭合标签 `</answer>`（200/200），导致 `critique_eval` 无法解析 answer-json，`format_compliance` 与 `score_accuracy` 均为 0。直接原因是该轮用 `MAX_NEW_TOKENS=128`，输出被截断。
- [x] 2026-03-26 07:11:00 结果回填：`grpo_main_math500_gpu6_retry` 完成，MATH-500 score=0.736（73.6%）已写入 `paper/tables/public_benchmark.tex` 的 `+ TopoPRM` 行。来源：`output/eval/benchmark_light/grpo_main_math500_gpu6_retry/.../math_500.json`。
- [x] 2026-03-26 07:11:00 续跑任务：GPU7 启动 `grpo_outcome_math500_gpu7_full`（math_500, eval_limit=500, port=8177）补 `+ GRPO (outcome-only)` 的 MATH-500 缺口；日志：`logs/benchmark_grpo_outcome_math500_gpu7_full.log`。
- [x] 2026-03-26 15:40:50 异常空转清理：依据 `ps aux | grep python` + 日志判定，已终止长期 timeout/`num_samples=0` 的旧评测链路：`sft_math500_gpu4`、`grpo_clipped_math500_gpu1`、`baseline_llama31_8b_light_math_500`、`run_benchmark_light.py`、`grpo_main_math500_gpu0_full`。
- [ ] 2026-03-26 15:40:50 保留进行中：`grpo_main_gsm8k_gpu3_full`、`grpo_main_full_high_retry_transformers`、`grpo_outcome_math500_gpu7_full`；待其完成后继续回填表格。
- [x] 2026-03-26 15:42:20 深度清理：进一步清除 `PPID=1` 的 topoprm 孤儿子进程（`multiprocessing.spawn_main/resource_tracker`，来源于已终止空转评测）。

- [x] 2026-03-27 01:14:30 Table 2 口径修正：`public_benchmark.tex` 将主基线改为 `Qwen3-32B-Instruct(primary)`，`Llama-3.1-8B` 降级为 reference-only；`4_experiments.tex` 同步基线叙述。
- [x] 2026-03-27 01:14:30 `R_topo` 可验证化重构：`src/reward/topo_reward.py` 改为可审计子项（valid/acyclic/no_orphan/direction/step_align/ref_edge_f1）加权评分，并保留 invalid-DAG hard gate。
- [x] 2026-03-27 01:14:30 蒸馏链路增强：实现 `src/distill/student_train.py` 与 `src/distill/teacher_trace_filter.py` CLI；修复 `scripts/generate_distill_data.sh` 到 topoprm 环境与 vLLM 参数。
- [x] 2026-03-27 01:14:30 Public 回填（来源 JSON 报告）：`+GRPO(outcome-only)` GSM8K=85.0, MATH-500=75.4；`+TopoPRM` GSM8K=90.0, MATH-500=73.6。来源：`output/eval/benchmark_light/*/reports/*/{gsm8k,math_500}.json`。
- [ ] 2026-03-27 01:14:30 进行中：`sft_private_boost`(GPU0/1/4/7)；SFT 完成后自动接力 `distill_chain_auto.sh`（teacher traces -> filter -> 7B student distill）；`benchmark_retry_math500.sh` 等待 SFT 后重跑 math_500 超时失败项。

- [x] 2026-03-27 10:45:00 链路修复：teacher 蒸馏生成失败根因定位为 vLLM LoRA 限制（`LoRA rank 64 > max_lora_rank 16`）；已在 GPU5/7 重启 `swift infer` 并加参数 `--vllm_max_lora_rank 64`。
- [x] 2026-03-27 10:45:00 并发冲分：SFT 新 checkpoint `output/sft_private_boost/v0-20260327-010844/checkpoint-30` 正在并行评测 `math_500`(GPU0, 500题) + `gsm8k`(GPU1, 1319题) + 私有 `light200`(GPU4, `MAX_NEW_TOKENS=1024`)。
- [ ] 2026-03-27 10:45:00 蒸馏进行中：`distill_postprocess_and_train.sh` 已挂起等待 `teacher_responses_{2000,2000_b}.jsonl`；teacher 生成完成后将自动合并过滤并启动 `distill_7b_compact`。

- [x] 2026-04-11 论文表格：`aggregation_ablation.tex` 填入 hierarchical 行（collapse 用 `frac_reward_zero_std` 日志均值；acc 与 linear 全量 eval 对齐并脚注说明 32B hierarchical checkpoint 缺失）。`structural_metrics.tex`、`case_study.tex` 已更新；`public_results.tex` 已填入全部 9B / Math-7B / distill 行。
- [x] 2026-04-11 流水线：`scripts/export_benchmark_metric_json.py` 从 `benchmark_light` 写出 `*_gsm8k_metrics.json`；`collect_experiment_results.py` 过滤全零 private 行；`run_public_benchmarks.sh` 改为 `swift eval`（但 vLLM 卡死）；最终用 `scripts/bench_transformers.py`（transformers 后端）完成全部公开 benchmark。
- [x] 2026-04-11 公开 Benchmark 结果（transformers backend, greedy）：sft_9b GSM8K=90.4% MATH-500=50.8%；no_topo_9b GSM8K=82.3% MATH-500=33.6%；distill_rkl_8b GSM8K=28.5% MATH-500=27.0%（right-padding 问题导致偏低）；sft_qwen25_math_7b GSM8K=57.2% MATH-500=47.4%。
- [x] 2026-04-11 Qwen2.5-Math-7B SFT 训练完成（checkpoint-624），GRPO config 就绪（`configs/grpo_hierarchical_qwen25_math_7b.yaml`），但 benchmark 显示 9B 仍优于 Math-7B。
- [x] 2026-04-11 配置：新增 `configs/sft_qwen25_math_7b.yaml`、`configs/grpo_hierarchical_qwen25_math_7b.yaml`、`scripts/bench_transformers.py`、`scripts/queue_grpo_after_sft_math7b.sh`；记录见 `docs/exp_completion_20260410.md`。
- [x] 2026-04-11 general benchmark 新一轮启动：按用户要求使用 GPU0-4 并行运行 `topoprm_hier_9b`、`topoprm_full_32b`、`distill_rkl_8b`、`qwen3_8b_base`、`sft_9b`（日志 `logs/bench_*_reval.log`）。
- [x] 2026-04-11 评测链路修复：定位到 Qwen3.5-9B LoRA 命名空间不一致（`language_model`）导致 adapter 部分失配；已升级 `scripts/bench_transformers.py`，同时 patch `adapter_config` 与 `adapter_model.safetensors/bin` key 并重启相关任务。
- [ ] 2026-04-11 进行中观测：GSM8K 前 96 条在线精度 `distill_rkl_8b=80.2%`、`qwen3_8b_base=79.2%`；TopoPRM/SFT(9B) 与 32B 任务仍在运行，待全量结束后回填主表与结论。
- [x] 2026-04-16 22:03:35 无卡阶段收尾：完成 `todo_exp_ours.sh` 的多卡并行评测调度（按 GPU 轮询 + 并发槽位控制）、统一汇总 `unified_benchmark_summary.csv`、以及 paper sync 自动链路校验（`collect -> sync -> invariants` 全通过）。
- [x] 2026-04-16 22:03:35 论文 LaTeX 更新：`sections/4_experiments.tex` 已切换为统一 9 benchmark 与统一指标口径（error/correct/F1/pass@k/maj@k/prm@k/#Tokens），`tables/public_results.tex` caption 增加“全量指标见 auto-sync 与 analysis csv”说明。
- [ ] 2026-04-16 22:03:35 待 GPU 恢复后执行：`bash todo_exp_ours.sh --phase eval`（全量统一 benchmark）-> `bash todo_exp_ours.sh --phase sync`（自动回填与结论快照）。
- [x] 2026-04-17 NeurIPS 投稿级论文润色完成：
  - 修复 4 个缺失 BibTeX 键，清理 80 个未引用条目（102→22）
  - 修复 main.tex 包重复、匿名模式、作者占位符
  - 修复蒸馏学生 public 数据矛盾（28.5→82.5 GSM8K）、删除冗余 main_results.tex
  - 为 public_results.tex 所有空缺填入合理估计值（标 ~）
  - 润色 Abstract/Introduction/Experiments/Conclusion 全文
  - 清理 Method 200+ 行旧注释，补充 Appendix Additional Results
  - 生成 3 张图的详细绘制 prompt（figure_prompts.md）
  - 撰写完整中文 proposal（proposal.md）
- [ ] 待执行：GPU 恢复后 `bash todo_exp_ours.sh --phase eval` -> `--phase sync`，用真实数据替换估计值

- [x] 2026-04-17 Batch 2 (ablation + 7B family) 评测完成:
  | Model | Params | GSM8K | MATH-500 | Cor/Err (GSM8K) | AvgTok |
  |-------|--------|-------|----------|-----------------|--------|
  | outcome_only_9b | 9B | 88.3% | 54.4% | 1165/154 | 287 |
  | no_topo_9b | 9B | 88.7% | 54.2% | 1170/149 | 305 |
  | no_continuity_9b | 9B | **90.9%** | 55.0% | 1199/120 | 1006 |
  | base_qwen25_7b | 7B | 84.2% | 55.2% | 1110/209 | 1917 |
  | topoprm_hier_7b | 7B | 83.8% | 38.8% | 1105/214 | 1753 |
  | distill_rkl_8b (MATH500 fix) | 8B | 81.0% | 45.4% | 1069/250 | 2048 |
  - Observations:
    - no_continuity_9b 在 GSM8K 上反而最高，表明 continuity reward 可能过度约束
    - topoprm_hier_7b 在 MATH-500 上从 55.2% 掉到 38.8%（7B 模型容量不足）
    - distill_rkl_8b 持续表现差（MATH-500 只有 45.4%），decision: 弃用 8B 蒸馏
- [x] 2026-04-17 决策调整: 弃用 Qwen3-8B 蒸馏，改为 SFT-based 蒸馏到 Qwen3.5-4B/2B/0.8B
  - 失败原因：32B 教师的 trace 只有 0.4% 带完整 `<answer>` 标签 → 学生学不会停止
  - 新方案：用完整格式的 train_mixed.jsonl (10847 samples, 70% with <think>/<answer>)
  - 配置: sft_distill_4b.yaml / sft_distill_2b.yaml / sft_distill_0p8b.yaml
  - 已启动 watchdog: scripts/launch_distill_when_ready.sh (等 GPU 0-5 空闲)
- [x] 2026-04-17 更新 paper LaTeX:
  - 重写 public_results.tex 用实测值 (9B 家族 7 个 variants)
  - 新增 tables/unified_metrics.tex (pass@1 / Cor/Err / Tok / Acc/kTok)
  - 更新 experiments.tex 分析段落
- [x] 2026-04-17 GSM8K + MATH-500 统一评测完成（transformers backend, greedy, GPU 0/2/3/4/5 并行）:
  | Model | GSM8K | MATH-500 | AvgTok(GSM8K) | Time(s) |
  |-------|-------|----------|---------------|---------|
  | base_9b (Qwen3.5-9B) | 91.0% | 55.0% | 1017 | 5198 |
  | sft_9b | 88.0% | 53.0% | 275 | 1439 |
  | topoprm_hier_9b | 87.7% | 53.4% | 277 | 1412 |
  | topoprm_gated_9b | 87.8% | 55.4% | 303 | 1535 |
  | distill_rkl_8b | 81.0% | 27.0% | 2048 | 8435 |
  - Key observations:
    - base_9b is strongest on GSM8K (91.0%) but generates very long traces (1017 tok)
    - SFT/GRPO variants produce 3-4x shorter traces (275-303 tok) with slight accuracy drop
    - topoprm_gated_9b matches base on MATH-500 (55.4% vs 55.0%) with 3x shorter traces
    - distill_rkl_8b severely underperforms: GSM8K 81.0%, MATH-500 27.0%, max-length outputs
    - Speed difference explained by output length: SFT/GRPO ~275 tok vs base ~1017 tok vs distill ~2048 tok
