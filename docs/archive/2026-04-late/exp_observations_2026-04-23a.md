# 实验观察 2026-04-23a — Reward 修复 + D4 续训中止

> 接续 [exp_observations_2026-04-22.md](exp_observations_2026-04-22.md) 与 [method_diagnosis_2026-04-22.md](method_diagnosis_2026-04-22.md)。
> 本次按 [handoff_prompt_2026-04-22.md](handoff_prompt_2026-04-22.md) 的 D1-D3 落地 reward 修复，并在 step 134/400 主动中止 D4 续训。

## TL;DR

- **D1-D3 代码修复落地且被训练数据证实有效**：step 80 以后 `frac_reward_zero_std = 0`（此前为 0.36），`reward_std ≈ 0.15`（此前 gated 版 ≈ 5e-4），BASE_FLOOR + 加权 orphan 的效果与设计预期一致。
- **D4 续训 step 134/400 主动中止（释放 GPU 0,1）**。理由：每步 200 s，剩余 ~15 h，最乐观的上限是 AIME +3pp（打平 SFT），不足以成为论文叙事转折点；与其等它，不如把 GPU 0,1 转给评测补齐 7B compression 故事。
- **文档归档与 `docs/README.md` 索引完成**；40 份早期文档迁入 `docs/archive/{2026-03,2026-04-early}/`。
- **下一步**：7B + 9B 双主角叙事；代码做 **moderate 清理**（保留 env 可配置性，默认值去 hack 化）；评测 queue 完成后产出 `exp_observations_2026-04-23b.md` 和主表更新。

## 1. D1-D3 代码修复摘要

三处 `bug fix / 科学性增强`，全部通过 `scripts/check_reward_invariants.py` 7/7 PASS。

### D1. Reward floor + std-floor 推广 ([`src/reward/composite_reward.py`](../src/reward/composite_reward.py))

- `TopoHierarchicalReward.__call__` 在 `r = r_base * gain` 之前对 `r_base` 做 `max(r_base, BASE_FLOOR)`，避免 outcome=format=length=0 时 topology gain 被整组乘成 0。
- 抽取 `_inject_std_floor(rewards, min_std, noise_eps)` helper，同时在 `TopoCompositeReward` / `TopoGatedReward` 里接入（此前只有 Hier 做 anti-collapse）。
- 新增 5 个 env vars：`TOPO_HIER_BASE_FLOOR`、`TOPO_GATED_MIN_STD`、`TOPO_GATED_NOISE_EPS`、`TOPO_COMPOSITE_MIN_STD`、`TOPO_COMPOSITE_NOISE_EPS`。**注意：这些 hack 参数的默认值在本次 moderate 清理中会再次调整（见 §3）**。

### D2. Orphan ratio 加权化 ([`src/reward/topo_reward.py`](../src/reward/topo_reward.py))

旧二值版（virtual 支持=1，其余=0）被重命名 `_orphan_conclusion_ratio_legacy`。新加权版：

```
virtual_edge        -> 1.0   (strong support)
double_barrier_edge -> 0.5   (weak support, auto-added fallback)
solid_edge          -> 0.3   (sequential proximity)
support_i = min(1.0, sum_weights(predecessors(i)))
rho_orphan = mean_i (1 - support_i)
```

把 `build_dag.py` 自动填充的 double_barrier fallback 从 "no support" 纠正为 "weak support"，减少 false-orphan 对 topo reward 的过度惩罚。env `TOPO_ORPHAN_LEGACY=1` 可回退。

### D3. `rollout_srt.py` orphan_step bug ([`scripts/rollout_srt.py`](../scripts/rollout_srt.py))

- 删除对不存在的 `ReasoningDAG.from_trace` / `.orphan_conclusion_nodes()` 的调用（此前 `orphan_step` 永远是 None，`P_r` 里 `{k}` 退化成 1）。
- 改用 `build_dag_from_answer(text)` 构图 + 按 `step_id` 升序找第一个无 virtual predecessor 的 CONCLUSION 节点。
- 同步在 `src/distill/opsd_trainer.py::score_response` 加同款提取，并在 `build_prompt_dispatch` 调用处传 `orphan_step`，为 D5 铺路。

### 不变量验收

`scripts/check_reward_invariants.py` 7 项全绿，新增的 **TopoHierarchicalReward[zero-base floor]** 检查：4 条 NO_TAGS 样本（原本 `r_base = 0`）现在返回 `r ∈ [0.22, 0.24]`、`std = 8.1e-3`，确认 BASE_FLOOR 让零方差 group 重新获得学习信号。

## 2. D4 续训：step 134 主动中止

### 训练期真实指标（来自 `logs/grpo_hier_continue.log` + console tee）

| Step | reward mean | reward std | frac_zero_std | loss | kl |
|---:|---:|---:|---:|---:|---:|
| 80 (resume+1) | 0.2752 | 0.1553 | 0.000 | 0.0102 | 0.181 |
| 132 | 0.2869 | 0.155 | 0.000 | 0.010 | 0.180 |
| 134 | 0.4019 | 0.2855 | 0.000 | 0.010 | 0.180 |

- **D1 BASE_FLOOR 修复被实证**：`frac_reward_zero_std = 0.36 -> 0`（之前 40% 的 rollout group 是零方差）。
- **D2 加权 orphan 侧面有效**：reward std 从 `5e-4` 量级恢复到 `0.15` 量级，GRPO 有足够 advantage 去学。
- **但训练已进入平台期**：`loss` 很小、`kl` 稳定，这是一个 healthy 但温吞的训练状态。

### 为什么 kill

- **耗时**：每步 200 s * 剩余 266 步 = 另外 **14 h 51 min**；GPU 0,1 被占一整夜。
- **收益上限**：对比 step-79 checkpoint，即便训到 400 步最乐观也是：
  - Olympiad / Omni **持平**（本来就已经 on par with SFT）
  - AIME2024 从 26.7 -> ~30（**+3pp**）
  - AIME2025 / CNMO 变化不确定
- **论文定位**：9B TopoPRM vs SFT 的叙事本来就是 "on par within noise, compression-by-design 在 7B 上展示"，AIME +3pp **不是 story turning point**。
- **取舍**：15h 两卡换 +3pp vs 15h 两卡用来补评测缺口 / 做 7B α 消融 / 写论文 —— 后者对投稿更直接。

### 保留物

- `output/grpo_hierarchical_qwen35_9b_mcl4096_continue/v0-20260423-135530/checkpoint-100`
- `configs/grpo_topoprm_hier_continue.yaml` 顶部已加 `[KILLED]` 块注释
- 本文档 §2 成为"为什么不继续训"的可追溯依据

## 3. 代码工程化审计（moderate 清理，本次即将执行）

当前 `src/reward/reward_config.py` + 两份 reward 类一共暴露约 **40 个 env 变量**。对 NeurIPS 方法章来说太多：审稿人会认为"method has too many knobs"。

### 保留（科学意义，写在论文里）

| 类别 | env 变量 | 论文出处 |
|---|---|---|
| 主公式 α | `TOPO_HIER_ALPHA` | Eq. R_hier |
| outcome/format/length 权重 | `BASE_WEIGHTS` 常量 0.70/0.15/0.15 | Table: reward weights |
| DAG 五项指标权重 | `TOPO_W_VALID/ACYCLIC/NO_ORPHAN/DIRECTION/STEP_ALIGN/REF_EDGE_F1` | Eq. R_topo |
| Orphan 加权 | `TOPO_ORPHAN_W_VIRTUAL/DOUBLE_BARRIER/SOLID` (1.0/0.5/0.3) | §DAG orphan |
| BASE_FLOOR | `TOPO_HIER_BASE_FLOOR` (0.05) | 附录：zero-base bug fix |

### 去 hack 化：默认值改为"等于关闭"

| 变量 | 旧默认 | 新默认 | 理由 |
|---|---|---|---|
| `TOPO_DYNAMIC_REWARD` | True | **False** | 动态权重未进论文 |
| `TOPO_HIER_REWARD_TEMP` | 2.0 | **1.0** | 温度缩放无公式依据；=1 等于不缩放 |
| `TOPO_HIER_NOISE_EPS` | 0.01 | **0.0** | 噪声注入是 hack；BASE_FLOOR 已解崩溃 |
| `TOPO_GATED_NOISE_EPS` | 0.005 | **0.0** | 同上 |
| `TOPO_COMPOSITE_NOISE_EPS` | 0.005 | **0.0** | 同上 |

保留类结构 / env 接口不变 — 只改默认，ms-swift 的 `scale_rewards=group`（GRPO 默认）做 advantage 标准化就够，不需要我们再手动注入高斯噪声。

### 验收

- `python scripts/check_reward_invariants.py` 重跑 7/7 PASS（zero-base floor 断言需要从"std > 0"调整为"mean > 0"，因为 `NOISE_EPS=0` 后零方差可能再出现，但 mean 被 BASE_FLOOR 保证非零）
- 训练 / 评测都不会因为这次清理回归（env vars 默认值变更，调用方不动）

## 4. GPU 重分配（kill 之后）

| GPU | 状态 | 任务 |
|---:|---|---|
| 0 | 空闲 -> 上任 | 新建 `logs/v3b_jobs/gpu0.jobs`：接管 `no_continuity_9b_v3` 的 long-cot（原 gpu4.jobs 里的那行） |
| 1 | 空闲 -> 上任 | 新建 `logs/v3b_jobs/gpu1.jobs`：给 7B 补齐 gpqa 并行、或 base_4b 兜底长尾 |
| 2-5 | 继续评测 | IO 压力减半，速度会加快 |

## 5. 下一步

按计划顺序（本会话内串行执行）：

1. 代码 moderate 清理（reward_config + 断言调整）
2. 起 gpu0 / gpu1 评测 queue
3. 监控到主要缺口落盘（目标 8 模型 × 9 bench ≥ 70/72）
4. `bash scripts/sync_all.sh` 更新主表
5. 产出 `docs/exp_observations_2026-04-23b.md` 交付
6. 给出论文段落修改 checklist（不代改）

## 6. 归档/索引变化

- `docs/archive/2026-03/`：5 份早期规格文档
- `docs/archive/2026-04-early/`：20 份 4 月上半月日志 + talk slides + showcase HTML
- `docs/README.md` 重写为索引，本文档将加入"本周活跃"区

