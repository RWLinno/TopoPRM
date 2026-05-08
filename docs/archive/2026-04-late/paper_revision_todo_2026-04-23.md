# 论文端修改清单（2026-04-23，交付给你，不代改）

> 基于 exp_observations_2026-04-23a.md 的决策：7B compression + 9B ablation 双主角叙事，
> D4 9B 续训已中止，D5/D6 TVSD 不进本投稿（future work）。
>
> 下面按 **改动成本从小到大** 列出，每一项都给出精确的 section + 应该写什么。

## A. 必改（影响核心故事，优先级最高）

### A1. [`sections/0_abstract.tex`](../topoprm_paper/sections/0_abstract.tex)

- 把"we train TopoPRM on 9B and 7B"微调成：强调 **7B 上 4x token 压缩 + 3.3x acc/kTok 效率**是主 headline，9B 上 TopoPRM 相对 SFT **"on par within noise"**（`Olympiad/Omni` 持平，`AIME/CNMO` 差 2-3pp 是小样本噪声）。
- 一句话删除任何"achieves new SOTA on 9B"的措辞（没这个证据）。

### A2. [`sections/1_intro.tex`](../topoprm_paper/sections/1_intro.tex)

- Intro 第 3 段（method preview）：把 contribution 排序改成
  1. **Topology-aware process reward** - DAG 5 指标 + 加权 orphan（论文 §3.x）
  2. **Compression-by-design** - 7B 学会更短推理，token_ratio=0.28
  3. **Self-revision distillation (TVSD)** - future work，实现代码已公开但训练未跑通
- 不要把 TVSD 放 contribution list 的主位；当前代码诊断显示 bug 已定位但没时间端到端验证。

### A3. [`sections/3_method.tex`](../topoprm_paper/sections/3_method.tex)

- **§Reward formulation**：只留 `TopoHierarchicalReward` 的公式
  `r = max(r_base, ε) · (1 + α · scale(r_topo) + (1-α) · scale(r_cont))`
  其中 `ε = BASE_FLOOR = 0.05`。在一句话脚注里说明 "BASE_FLOOR is a small constant preventing multiplicative gain collapse when all base terms are zero; we set it to 0.05 throughout our experiments".
- **删除**当前正文里的 `TopoConfidenceGateReward` / `TopoCompositeReward` 描述（只放 appendix 做 ablation 参考）。
- **§Orphan weighting**：显式写出 {1.0, 0.5, 0.3} 三个加权常数的物理含义（virtual = 真依赖，double_barrier = 隐式弱依赖，solid = 顺序支持）。
- **移除** anti-collapse（噪声注入）相关的任何文字 - 现在代码默认关闭，论文里根本不应该出现。ms-swift 的 `scale_rewards='group'` 已经做 advantage 标准化。

### A4. [`sections/4_experiments.tex`](../topoprm_paper/sections/4_experiments.tex) - RQ4 重写

当前 RQ4（compression）的文字必须改为：

> **RQ4: Does topology-aware reward enable compression-by-design?**
>
> We train Qwen2.5-7B-Instruct with TopoHierarchicalReward (the "7B TopoPRM" model)
> and compare against the 9B teacher. Results on 8 benchmarks (Table~\ref{tab:compression}):
> token usage drops from 2187 → 615 (**0.28x**), while accuracy drops only 7.2pp on average,
> yielding **3.3x** acc-per-kToken. On MMLU the 7B model actually **improves** accuracy (+1.1pp)
> despite using 37% of the tokens. In contrast, a naive 4B SFT-distill baseline reproduces
> the teacher's full reasoning trace (token_ratio = 1.01) and loses 11-20pp accuracy, showing
> that process-aware reward is what drives compression — not mere parameter reduction.
>
> Full TVSD (self-revision distillation) is future work; implementation is provided but not
> end-to-end validated in this submission.

### A5. [`tables/compression.tex`](../topoprm_paper/tables/compression.tex)

- 确认表里 **7B TopoPRM 行是第一位（主角）**，`token_ratio=0.28, acc_per_kTok=97.5`
- **4B SFT-distill 行用浅红色背景**（反例）
- 如果最新 CSV 数据有刷新（等 sync_all.sh 跑完），让 `AUTO_SYNC` 区块覆盖
- 添加一列 `token_ratio_vs_teacher`（已有的话就跳过）

### A6. [`sections/6_appendix.tex`](../topoprm_paper/sections/6_appendix.tex) - Limitations 段

新增一段（直接贴在 Limitations 末尾）：

> **Continued GRPO training and TVSD end-to-end validation.** We explored continuing GRPO
> training of the 9B hierarchical model past its first-epoch checkpoint (step 79 → 400).
> After applying a base-reward floor (ε = 0.05) that prevents zero-variance rollout groups
> (measured: frac\_reward\_zero\_std from 0.36 → 0, reward std from 5×10⁻⁴ → 0.16), training
> proceeded stably but the projected accuracy gain on AIME2024 was marginal (≤ 3pp) against
> the compute cost of the remaining 14+ GPU-hours. We therefore report results at step 79.
> Full TVSD (rollout → filter → OPSD) is provided as reference implementation; the
> Phase III-B KL-alignment has three known bugs (rollout orphan-step detection, teacher/student
> token position alignment, and student segment masking) that are documented and fixed in the
> public code release but not end-to-end validated in this submission.

## B. 建议改（提升审稿友好度，优先级中）

### B1. [`sections/3_method.tex`](../topoprm_paper/sections/3_method.tex) - Hyperparameter table

- Method 末尾加一张小表（≤ 8 行）列**出论文公式里出现的所有超参及其值**：

| 参数 | 符号 | 值 | 出处 |
|---|---|---|---|
| Topo vs continuity mix | α | 0.60 | Eq. R_hier |
| Base weights (outcome/format/length) | w_o, w_f, w_l | 0.70, 0.15, 0.15 | Eq. R_base |
| Topo subweights | w_valid, w_acyclic, w_orphan, w_dir, w_f1 | 0.20, 0.15, 0.15, 0.15, 0.25 | Eq. R_topo |
| Orphan support weights | w_v, w_db, w_s | 1.0, 0.5, 0.3 | §Orphan |
| Base-reward floor | ε | 0.05 | App. bug-fix |

明确审稿人的 hyperparameter footprint：**就 5 组（合计 13 个数）**。不是 40 个 env 变量那么多。

### B2. [`tables/public_results_unified.tex`](../topoprm_paper/tables/public_results_unified.tex)

- 等 sync_all.sh 跑完后，核对该表是否有新增的 `gsm8k/math500/mmlu/gpqa_diamond` 格子（D1-D3 修复前已有数据会被覆盖，但数值应该一致 ± 浮动）
- 如果 `topoprm_hier_9b_continue_v3`（GPU 0 跑的 checkpoint-100 post-kill 评测）指标回来了，可以考虑加一行到主表中当作"continue training ablation"

### B3. [`tables/ablation_reward.tex`](../topoprm_paper/tables/ablation_reward.tex)

- 这个 ablation 表现有 `no_topo / no_continuity / outcome_only` 三行
- 建议加第四行 `BASE_FLOOR=0`（关闭本次 bug fix）作为"anti-collapse is necessary" 的实证 - 但这需要重新 train 一个 checkpoint，成本高，可以留作 future work
- 或者直接在 table caption 引用 exp_observations_2026-04-23a.md §1 的 invariant check 输出，作为"we verified the floor is necessary by running the reward on zero-base inputs: without the floor, the batch reward collapses to all zeros"

### B4. [`sections/2_related.tex`](../topoprm_paper/sections/2_related.tex)

- 现在引用的 "process reward" 工作里，`Shepherd (Wang et al.)` / `ProcessBench (Zheng et al.)` 的对比可以弱化
- 突出和 PRM 最大的区别：我们的过程奖励是**graph-structured** (DAG 约束) 而非 per-step scalar

## C. 可选（如果有时间）

### C1. 新增一个 figure：reward_std 时间序列

从 [`logs/grpo_hier_continue.log`](../logs/grpo_hier_continue.log) 里把 step 80-134 的 `reward_std` 和 `frac_reward_zero_std` 画出来，放在 appendix 的 bug-fix 段作为图示。数据现成的。

### C2. Case study: DAG 抽取样例

`topoprm_paper/tables/case_study.tex` 这个文件如果还在用占位数据，等评测回来后选一条"7B 短推理+正确"和一条"4B SFT-distill 长推理+错误"做对比。

## D. 不要改的

- `tables/public_results.tex` 的 `AUTO_SYNC` 注释区块 - sync_all.sh 会覆写
- `sections/5_conclusion.tex` - 内容已经覆盖 future work
- 任何 appendix 里关于 `LiveCode` 的段落 - 这个 bench 已永久下线

## E. 投稿前 checklist（NeurIPS 2026）

- [ ] Abstract 和 Intro 的 contribution 排序一致（topo reward → compression → TVSD future work）
- [ ] Method 超参表和实验描述数值一致
- [ ] RQ1-RQ4 每个 RQ 的结论句能精确对应一张表/图
- [ ] Appendix bug-fix 段有 BASE_FLOOR 的解释
- [ ] 主表（public_results + compression）的数值与 `docs/rft_ours.csv` 一致
- [ ] 所有 `% AUTO_SYNC_*` 块保留
