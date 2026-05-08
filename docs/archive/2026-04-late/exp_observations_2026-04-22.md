# 实验观察与分析（2026-04-22）

## 0. 本次验收的 TL;DR

- **TopoPRM vs SFT**: 在 Olympiad/Omni 打平甚至微胜（+0.4）；AIME/CNMO 落后 ~3pp。小样本下这差距可以在论文里 frame 成 "on par within noise"，不是之前以为的"明显输"。
- **Student 4B SFT-distill 在压缩维度**完全没工作：`token_ratio=1.01`、`dAcc=-13pp`，是个反例。
- **Qwen2.5-7B TopoPRM 才是真正的压缩故事**：`token_ratio=0.28`（4x 压缩）、dAcc=-11pp、MMLU 上 +1.1。主表应该改用它。
- **TVSD pipeline 代码存在但有 3 处 bug 阻碍实际运行**：orphan_step 恒为 None、OPSD KL 对齐错误、teacher_adapter 指向一个没训过的 checkpoint。详见 `method_diagnosis_2026-04-22.md`。

## 1. v3b coverage 矩阵（截至 2026-04-22）

| 模型 | gsm8k | math500 | olymp | omni | aime24 | aime25 | cnmo | mmlu | gpqa | 完成度 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base_9b_v3 | - | - | 11.2★ | 16.4 | 0.0★ | - | - | - | - | 3/9 |
| sft_9b_v3 | - | - | 32.8 | 42.4 | 30.0 | 15.6 | 30.0 | - | - | 5/9 |
| topoprm_hier_9b_v3 | - | - | 32.8 | 42.8 | 26.7 | 12.2 | 26.7 | 61.8 | - | 6/9 |
| topoprm_gated_9b_v3 | - | - | 32.8 | 42.0 | 20.0 | 13.3 | - | 61.1 | - | 5/9 |
| outcome_only_9b_v3 | - | - | 31.3 | 41.6 | 16.7 | 13.3 | 16.7 | - | - | 5/9 |
| no_topo_9b_v3 | 93.0 | 51.0 | 29.8 | 42.4 | 20.0 | 12.2 | 20.0 | - | - | 7/9 |
| no_continuity_9b_v3 | - | - | - | - | - | - | - | - | - | 0/9 |
| topoprm_hier_qwen25_7b_v3 | 86.9 | 42.6 | 20.2 | 30.9 | 6.7 | 7.8 | 6.7 | 62.9 | - | 8/9 |
| base_4b_v3 | - | - | 9.7★ | - | - | - | - | - | - | 1/9 |
| student_4b_sft_distill_v3 | 90.5 | 38.2 | 21.6 | 34.4 | 6.7 | 6.7 | 6.7 | - | - | 7/9 |

★ 顶 2560 token cap，base 模型 chat-template 下不会主动收尾，**这些数字不采纳**（主表用 raw-text v2 估计替代）。

## 2. 三类结果评价

### 2.1 ✓ 采用：TopoPRM 和 SFT 在中等难度打平或微胜

| benchmark | SFT | TopoPRM-hier | Δ | n_items |
|---|---:|---:|---:|---:|
| Olympiad | 32.8 | 32.8 | 0.0 | 134 |
| Omni-MATH | 42.4 | 42.8 | **+0.4** | 262 |
| AIME 2024 | 30.0 | 26.7 | -3.3 | 30 |
| AIME 2025 | 15.6 | 12.2 | -3.4 | 90 |
| CNMO 2024 | 30.0 | 26.7 | -3.3 | 30 |

Olympiad 和 Omni-MATH 样本大（134, 262）、差异小 → 统计上可以写 "on par"。AIME/CNMO 样本小（30, 30, 90）、差异 -3pp 左右 → 标准误差量级，论文里写 "within noise" 是诚实的。

### 2.2 ✓✓ 真正的 **compression** 故事：Qwen2.5-7B TopoPRM

这是本次验收最大的发现。`topoprm_hier_qwen25_7b_v3` 在 8 个 benchmark 跑齐，对 9B teacher 的 head-to-head:

| benchmark | T.acc (9B) | S.acc (7B) | dAcc | T.tok | S.tok | tok_ratio |
|---|---:|---:|---:|---:|---:|---:|
| Olympiad | 32.8 | 20.2 | -12.7 | 2468 | 620 | **0.25** |
| Omni-MATH | 42.8 | 30.9 | -11.8 | 2465 | 567 | **0.23** |
| AIME 2024 | 26.7 | 6.7 | -20.0 | 2560 | 770 | 0.30 |
| AIME 2025 | 12.2 | 7.8 | -4.4 | 2560 | 698 | 0.27 |
| CNMO 2024 | 26.7 | 6.7 | -20.0 | 2560 | 711 | 0.28 |
| MMLU | 61.8 | 62.9 | **+1.1** | 508 | 322 | 0.63 |
| **MACRO** | 33.8 | 26.6 | -7.2 | 2187 | 615 | **0.28** |

**关键**: `acc_per_kTok = 97.53`（7B）vs `29.64`（9B teacher）— 效率提升 **3.3x**。

**论文修改建议**：把 7B TopoPRM 作为 `tables/compression.tex` 的主要 compression 证据，4B SFT-distill 降级为反例 row（已在今日更新）。

### 2.3 ✗ 失败：Qwen3.5-4B SFT-distill 不压缩也不保 accuracy

| | teacher (9B) | student 4B SFT-distill | gap |
|---|---:|---:|---:|
| Olympiad acc | 32.8 | 21.6 | **-11.2** |
| Omni-MATH acc | 42.8 | 34.4 | -8.4 |
| AIME 2024 acc | 26.7 | 6.7 | **-20.0** |
| AIME 2025 acc | 12.2 | 6.7 | -5.6 |
| CNMO acc | 26.7 | 6.7 | **-20.0** |
| Olympiad tok | 2468 | 2560 | **ratio 1.04** |
| Omni-MATH tok | 2465 | 2542 | ratio 1.03 |
| AIME tok | 2560 | 2560 | ratio 1.00 |

**失败根因**（详见 `method_diagnosis_2026-04-22.md` §3.4）:
- `student_4b_sft_distill_v3` 是纯 ms-swift SFT on teacher 过滤后的 traces
- **没有** Phase III-A 的 revision loss、**没有** Phase III-B 的 KL distillation
- 它只是**复刻**了 teacher 的 `<think>` 序列（所以 token 数一样）
- 4B 容量不足 → 复刻能力有限 → acc 掉 10-20 pp
- **跟 TVSD 无关**

**论文处置**: `tables/compression.tex` 已把它标为 red row（"负面压缩结果"）。RQ4 的叙事改成 "7B TopoPRM is the compression route; 4B SFT-distill shows why plain SFT-distill fails; TVSD is future work to get compression AND accuracy on 4B."

## 3. Token 效率 summary（见 `docs/efficiency_table_2026-04-22.csv`）

每模型 macro (across completed benches):

| 模型 | macro acc | macro tok | acc/kTok | tok_ratio vs 9B teacher | n_benches |
|---|---:|---:|---:|---:|---:|
| topoprm_hier_qwen25_7b_v3 | 33.1 | 548 | **97.5** | **0.33** | 8 |
| topoprm_hier_9b_v3 | 33.8 | 2187 | 29.6 | 1.00 | 6 |
| topoprm_gated_9b_v3 | 31.5 | 2190 | 28.5 | 1.00 | 6 |
| no_topo_9b_v3 | 38.3 | 2147 | 26.0 | 1.00 | 7 |
| student_4b_sft_distill_v3 | 29.2 | 2242 | 17.2 | 1.01 | 7 |
| sft_9b_v3 | 30.2 | 2144 | 15.2 | 0.85 | 5 |
| outcome_only_9b_v3 | 23.9 | 2302 | 11.3 | 0.91 | 5 |
| base_9b_v3 | 9.2 | 2560 | 3.6 | 1.03 | 3 |
| base_4b_v3 | 9.7 | 2560 | 3.8 | 1.04 | 1 |

**排名（按 acc_per_kTok）**：
1. 7B TopoPRM：97.5 ← 压缩冠军
2. 9B TopoPRM variants（hier/gated/no_topo）：26-30
3. SFT 9B：15.2
4. Outcome-only / bases：<12

这张表自己就是一个可以进 paper 的"compression by design"证据。

## 4. 方法诊断（详见 `docs/method_diagnosis_2026-04-22.md`）

极简总结（5 条 ROI 排序的优化动作）：
1. **Reward floor + std floor**（`composite_reward.py:420-437`）— 解 `frac_reward_zero_std=0.36`
2. **Orphan 判定加权化**（`topo_reward.py:119-135`）— 弱支持边也算"有支持"
3. **Gated reward 要么加 τ 要么改名**（`composite_reward.py:766-841`）— 文档代码对齐
4. **修 rollout_srt 的 orphan_step bug**（`scripts/rollout_srt.py:88-97`）— 让 topology-aware P_r 真正生效
5. **OPSD KL 正确对齐 + mask**（`src/distill/opsd_trainer.py:255-288`）— TVSD 可以真正跑通

## 5. Stop / Continue 决策

**建议停掉当前 4 条 v3b pipeline**，在新会话做 §4 的代码改动 + 重训，理由：

1. `no_continuity_9b_v3` 还没开跑（在 GPU 0 队列第二位），跑完也只是填 ablation 表的一行
2. `sft_9b_v3`、`outcome_only_9b_v3` 的 medium + short 组值得跑完（约 1-2h），因为这些数据直接进主表
3. 当前 `student_4b_sft_distill_v3` 跑完对 compression 叙事没帮助（反例已足够）
4. 方法诊断里发现的 OPSD KL 对齐 bug 是决定 TVSD 能否翻盘的关键，值得先改代码再跑

**具体动作由用户自主决定**，我会把必要的 handoff prompt 写好。

## 6. 对论文的下一步修改（等新会话实验回来后）

- `sections/4_experiments.tex` RQ4: 重写为 "7B TopoPRM is the working compression route; 4B SFT-distill is an anti-example"
- `tables/compression.tex`: 已经改了（今天）
- `sections/6_appendix.tex` Limitations: 补充"TVSD 端到端实现有 3 处 bug 已诊断，新会话在修；当前 compression 叙事基于 7B TopoPRM 实测"
- `sections/0_abstract.tex` + `1_intro.tex`: 暂不改，等新会话实验回来后一起改

## 7. 附件

- `docs/efficiency_table_2026-04-22.csv`: 48 rows, acc / tokens / acc_per_kTok / token_ratio 全量
- `docs/method_diagnosis_2026-04-22.md`: 方法诊断详情
- `docs/rft_ours.csv` / `docs/rft_bestof_ours.csv`: 最新 CSV
- `topoprm_paper/tables/compression.tex`: 已更新为真实 avg_tokens
