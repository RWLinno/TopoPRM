你是一位资深 ML 研究工程师，正在接手 TopoPRM（NeurIPS 2026 投稿）项目的最后冲刺。项目根目录 `/mnt/users/rwl/topoprm`。

请严格按以下顺序执行。**初次回复时请先汇报你读完哪些文件、识别到 top-3 风险、你建议的 first move。**

---

## Part A. 阅读上下文（≤ 20 min）

**按优先级读，读完即可开工：**

1. **最关键 — 今天的诊断和观察（必读前 3 个）**
   - [docs/method_diagnosis_2026-04-22.md](docs/method_diagnosis_2026-04-22.md) — 5 条按 ROI 排序的优化动作
   - [docs/exp_observations_2026-04-22.md](docs/exp_observations_2026-04-22.md) — 当前结果评价（TL;DR + coverage + 三类定性）
   - [docs/efficiency_table_2026-04-22.csv](docs/efficiency_table_2026-04-22.csv) — token 效率详表

2. **论文现状（paper draft）**
   - [topoprm_paper/sections/0_abstract.tex](topoprm_paper/sections/0_abstract.tex)
   - [topoprm_paper/sections/1_intro.tex](topoprm_paper/sections/1_intro.tex)
   - [topoprm_paper/sections/3_method.tex](topoprm_paper/sections/3_method.tex) — 重点读 §sec:compression 和 §sec:tvsd
   - [topoprm_paper/sections/4_experiments.tex](topoprm_paper/sections/4_experiments.tex) — 重点读 RQ4
   - [topoprm_paper/sections/6_appendix.tex](topoprm_paper/sections/6_appendix.tex) — Limitations 段已列 TopoPRM 在 AIME/CNMO 的 gap
   - [topoprm_paper/tables/compression.tex](topoprm_paper/tables/compression.tex) — **今天刚改过**，用 7B TopoPRM 作为真正的压缩证据
   - [topoprm_paper/tables/public_results.tex](topoprm_paper/tables/public_results.tex) — 主表（AUTO_SYNC 注释块里有每个 label 的 pass@1/5/maj@5 落盘记录）
   - [topoprm_paper/tables/public_results_unified.tex](topoprm_paper/tables/public_results_unified.tex) — 统一 9-benchmark 视图

3. **历史 observations（了解演进用）**
   - [docs/exp_observations_2026-04-21b.md](docs/exp_observations_2026-04-21b.md) — v3→v3b 加速方案
   - [docs/exp_observations_2026-04-21.md](docs/exp_observations_2026-04-21.md) — LiveCode 下线，异常指标排查
   - [docs/exp_observations_2026-04-20.md](docs/exp_observations_2026-04-20.md) — 原始清理归档
   - [docs/reward_collapse_diagnosis_2026-04-20.md](docs/reward_collapse_diagnosis_2026-04-20.md) — GRPO hier-79/gated-79 的 reward_std 收窄记录
   - [docs/checkpoints_triage_2026-04-20.md](docs/checkpoints_triage_2026-04-20.md) — checkpoint 分诊

4. **路线图**
   - [docs/exp_roadmap_2026-04-20.md](docs/exp_roadmap_2026-04-20.md) — 5 条后续实验：R1 续训 TopoPRM / R2 TVSD 端到端 / R3 DAG 结构评测 / R4 Pareto 图 / R5 AIME 恢复

5. **代码关键文件（按需跳读）**
   - 奖励：`src/reward/topo_reward.py`、`src/reward/composite_reward.py`（含 Hierarchical/Gated/Confidence-Gate 三类实现）
   - DAG：`src/data/build_dag.py`、`src/dag/graph.py`（`ReasoningDAG` 实现）
   - TVSD：`scripts/rollout_srt.py`、`src/distill/opsd_trainer.py`、`src/distill/build_srt_data.py`、`src/distill/reverse_kl_loss.py`
   - 评测：`scripts/bench_transformers.py`（含 `--force_overwrite` 反向开关，skip-if-exists 已默认启用）、`scripts/rerun_unified_v3b.sh`（label+benchlist 精准下发）
   - 一键同步：`scripts/sync_all.sh`

---

## Part B. 当前状态（2026-04-22 snapshot）

- **已 kill 所有旧实验**（用户在切会话前手动执行）
- **白名单 checkpoint（不要删）**：
  - `output/sft_qwen35_9b/v0-20260407-011328/checkpoint-626`
  - `output/grpo_hierarchical_qwen35_9b_mcl4096/v2-20260407-162048/checkpoint-79`
  - `output/grpo_gated_qwen35_9b_mcl4096/v4-20260407-111747/checkpoint-79`
  - `output/grpo_hierarchical_qwen25_7b/v3-20260406-134423/checkpoint-318`
  - `output/sft_distill_4b/v0-20260417-121952/checkpoint-2034`（注意：这是 plain SFT-distill，不是 TVSD）
- **GPU 约束**：只能用 **0/1/2/3**，严禁动 4/5/6/7（其他容器用户）
- **Conda env**：`topoprm`（位于 `/mnt/users/conda_env/topoprm`）

---

## Part C. 已知问题（按优先级，详情见 method_diagnosis_2026-04-22.md）

1. **[代码 bug] `rollout_srt.py` 的 orphan_step 永远是 None** — `ReasoningDAG.from_trace` / `.orphan_conclusion_nodes()` 都不存在（L88-97）
2. **[代码 bug] OPSD KL 对齐错误** — teacher 有 P_r 上下文、student 没有，`opsd_trainer.py:255-288` 用 `[-T:]` 截尾对齐是错的，且没 mask 到 student 生成段
3. **[代码 bug] HierarchicalReward 没有 r_base floor** — `r_base=0` 时 topology gain 被乘成 0，占约 40% rollouts，造成 `frac_reward_zero_std=0.36`
4. **[文档/代码不一致] `TopoGatedReward` 没有 τ 门**（类名误导）— 诊断里有写推荐 fix
5. **[叙事] 4B SFT-distill 不是 TVSD** — token_ratio=1.01、dAcc=-13pp，反例而非压缩证据；7B TopoPRM（token_ratio=0.28）才是

---

## Part D. 你的目标（按 ROI 执行，先 1+2+3 + 续训，再 4+5）

### 第一批（≤ 4h 代码 + ≤ 8h 训练，价值最高）

**D1.** 改 `src/reward/composite_reward.py` 里 `TopoHierarchicalReward._combine`：
- 加 `BASE_FLOOR` env var（默认 0.05），让 `r_base = max(r_base, BASE_FLOOR)` 再乘 gain
- 把 `MIN_STD + NOISE_EPS` 的方差注入从仅 hierarchical 推广到 `TopoGatedReward` 和 `TopoCompositeReward`
- 跑 `scripts/check_reward_invariants.py` 确保原有正确解 > 错误解 的不变量

**D2.** 改 `src/reward/topo_reward.py::_orphan_conclusion_ratio`（L119-135）:
- 把 `double_barrier_edge` 和 `solid_edge` 当作"弱支持"：weight 0.5 和 0.3
- `no_orphan = 1 - weighted_orphan_ratio`（连续分数而不是二值）

**D3.** 修 `scripts/rollout_srt.py:88-97`:
- 删掉对不存在的 `ReasoningDAG.from_trace` 的调用
- 改用 `from src.data.build_dag import build_dag_from_answer` + 从 `_orphan_conclusion_ratio` 里提 orphan step index

**D4.** 从 `checkpoint-79` 续训 TopoPRM hier 到 300-500 步:
- 用 D1 改过的 hierarchical reward
- 建议新 config `configs/grpo_topoprm_hier_continue.yaml`（可以 copy mcl4096 的改 `resume_from_checkpoint` 和 `max_steps`）
- GPU 0/1/2/3（只 1-2 张卡就够 continue training）
- 评测用 `scripts/rerun_unified_v3b.sh queue N logs/v3b_jobs/hier_continue.jobs`

**验收**: 新 hier checkpoint 在 Olympiad/Omni 至少不比 79 步的 32.8/42.8 差，AIME2024 能从 26.7 提到 **≥ 30**（打平 SFT）。

### 第二批（等 D1-D4 看到初步收益再做）

**D5.** 修 `src/distill/opsd_trainer.py`:
- 用 `src/distill/reverse_kl_loss.py::reverse_kl_loss(mask=...)` 替代手写 KL
- 用 tokenizer 精确计算 student 生成段的 mask（`y` 起止 index）
- Teacher / student 的 token 位置对齐：student 用 `(x, y)` 为 context，teacher 用 `(x, P_r, y)` 但只取 teacher 在 `y` token 位置的下一个 token logits
- 在 OPSD training time 也传 `orphan_step` 给 `build_prompt_dispatch`（对齐 rollout）

**D6.** 端到端跑 TVSD 4B/2B:
- Phase III-A: `python scripts/rollout_srt.py ...` → `python -m src.distill.build_srt_data ...` → SFT on `data/srt_ready/train.jsonl` → 产出 `output/srt_9b/final`
- Phase III-B: `python -m src.distill.opsd_trainer configs/opsd_student_4b.yaml`
- 评测 `student_4b_tvsd_v3b` label

**验收**: student_4b_tvsd 在 Olympiad/Omni 的 `token_ratio ≤ 0.6`，`dAcc ≥ -5pp`（相比 9B teacher）。

---

## Part E. 约束与工作流

- **每次改代码前**运行 `python scripts/check_reward_invariants.py` 确保现有 reward 不变量没回归
- **每次训练完**运行 `bash scripts/sync_all.sh` 更新 CSV 和 LaTeX（会自动跑 fill_rft_csv + collect + sync_paper_tables）
- **每次评测前**核对 `nvidia-smi` 确认 GPU 0/1/2/3 无其他用户占用
- **产出 observation**: 每个里程碑后在 `docs/exp_observations_2026-04-XX.md` 加一份新文件，不要改老的
- **论文修改**: `sections/4_experiments.tex` RQ4 段现在写的是 7B 压缩 + 4B 反例，等你训出 TVSD 4B 后再替换
- **不要删** `topoprm_paper/tables/public_results.tex` 里的 `% AUTO_SYNC_PUBLIC_BEGIN ... END` 注释块（sync_all.sh 会覆写里面内容）
- **不要动** GPU 4/5/6/7

---

## Part F. 出错恢复

如果你遇到以下问题：
- **bench_transformers.py 加载不到 adapter**：检查 `patch_swift_adapter_namespace`（`scripts/bench_transformers.py:922-926`），target_modules regex 会自动修 `model\.language_model` 命名空间问题
- **OPSD 训练 OOM**：把 `opsd_student_*.yaml` 的 `per_device_train_batch_size` 降到 1，用 `gradient_accumulation_steps` 8
- **MMLU 太慢**：用 `--max_items 500` （v3b 默认），单卡 9B 约 2h
- **LiveCode**：已永久下线（`load_livecode()` 返回 `[]`），不要重新启用

---

## Part G. 第一步请做这些

1. 读完 Part A 的前 3 个必读文件 + 论文 RQ4 段
2. 告诉我你识别到的 top-3 风险、你建议的 first move 是什么（D1 / D3 / D4 哪个先动）
3. 拿到我确认后再动代码

祝你顺利。我们的主卖点现在定位清晰了：**TopoPRM-hier + 7B 是真正的 compression story，9B 上 TopoPRM 和 SFT on par，TVSD 4B 是 future work 但代码已诊断好 bug**。
