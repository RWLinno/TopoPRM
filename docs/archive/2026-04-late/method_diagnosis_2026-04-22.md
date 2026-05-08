# TopoPRM 方法诊断（2026-04-22）

> 本文件基于 read-only 代码审查。**不改任何代码**，只列 findings + 推荐修法。
> 新会话据此按 ROI 排序进行改动 + 重训。

## 背景：当前 v3b 定量结果要解释的三件事

1. `frac_reward_zero_std ≈ 0.36`（`hier-79`）、`reward_std ≈ 5e-4`（`gated-79`）——为何 GRPO 几乎 update 不动
2. TopoPRM vs SFT 在 AIME/CNMO 差 ~3pp、Olympiad/Omni 打平——topology reward 的信号到底有没有作用
3. Student 4B SFT-distill `token_ratio=1.01`、`dAcc=-13pp`——压缩故事完全没落地

---

## 1. Reward 侧诊断

### 1.1 Hierarchical 的乘法增益 **没有 floor**

`src/reward/composite_reward.py:375-424` `TopoHierarchicalReward`：

```python
r_base = w_o*o + w_f*f + w_l*l        # 0.7*outcome + 0.15*fmt + 0.15*len
gain   = 1 + α * topo_scaled + (1-α) * cont_scaled   # α = 0.6
r      = r_base * gain                 # ← 没有 floor
```

当 `outcome=0, format=0, length=0`（长 CoT 超长或答案错 + 格式破），`r_base=0`，**整组被乘成 0**，topology reward 再高也救不回来。这直接解释 `frac_reward_zero_std=0.36` — 约 40% 的 rollout group 里所有样本都是 `r=0`，GRPO 无 advantage 可学。

**推荐 fix**：`r = max(r_base, floor) * gain`，`floor ≈ 0.05`；或改成加法 `r = r_base + γ * topo_scaled * cont_scaled`。

### 1.2 `TopoGatedReward` **并不是** "阈值门"（文档 vs 代码不一致）

`src/reward/composite_reward.py:766-841`：

```python
DELTA: float = env_float("TOPO_GATED_DELTA", 0.1)     # 没有 τ
EPSILON: float = env_float("TOPO_GATED_EPSILON", 0.5)
...
r = outcome + delta * format * (1 + eps * q)          # 无阈值条件
```

`reward_collapse_diagnosis_2026-04-20.md` 里我们推测过"gate 门槛过紧"——**代码里没有这个 gate**。真实原因更可能是：
- 组内所有 rollout 的 `outcome` 和 `format` 都一样（大多 `o=0, f=0`），`r=0`；
- `q`（batch-wise rescaled topo+cont）几乎常数，外层又被 outcome/format 归零；
- 结果：reward_std ≈ 5e-4 = `delta * format * eps * q_noise` 级别。

**推荐 fix**：要么把 `TopoGatedReward` 改成真正的 `o ≥ τ` 条件 gate（与文档一致），要么把这个类改名为 `TopoAdditiveReward`。无论哪种，**先**改文档还是改代码都必须 2 选 1。

`TopoConfidenceGateReward` (`confgate`, line 515-539) **才**是真正有 τ 的 gate，`CORRECT_THRESHOLD=0.66` 硬写在类属性里（没有 env var）。

### 1.3 组内方差仅在 hierarchical 做 **加性噪声注入**（不是 std floor 正则）

`composite_reward.py:430-437`：

```python
if std(rewards) < MIN_STD and NOISE_EPS > 0:
    rewards = [r + gauss(0, NOISE_EPS) for r in rewards]   # 高斯噪声，不是 floor
```

这只影响 hierarchical；`topo_gated`、`topo_composite` 都没这个。

**推荐 fix**：
- 在所有 reward 类里统一注入 `MIN_STD` 正则；或
- 把 std floor 挪到 GRPO 侧的 advantage 计算（对 `std < 阈值` 的组直接 skip / down-weight）。

---

## 2. DAG 抽取侧诊断

### 2.1 "句级 claim" 只用于展示，**依赖匹配仍用 pattern-based `claim_keys`**

- `src/data/build_dag.py:297-308` `extract_claims`：句级，有 `_is_complete_claim_sentence` 过滤（L183-194），**只**进 `DAGNode.claims` 用于 display
- `src/data/build_dag.py:311-318` `extract_claim_keys`：从全文扫 `_CLAIM_PATTERNS`，**这**是 `build_dependency_edges_by_rules` 实际用的
- 论文 appendix 说"claim extraction is sentence-level rather than phrase-level"，**与依赖边的实际实现矛盾**

**后果**：释义改写（paraphrase）会被拒绝匹配，产生很多 false orphan conclusion。

**推荐 fix**：让 `build_dependency_edges_by_rules` 额外消费 `step.claims` 做 sentence-level 重叠/相似度匹配（至少加简单的 trigram overlap）。

### 2.2 Orphan 定义过严：**忽略了 solid_edge 和 double_barrier_edge**

`src/reward/topo_reward.py:119-135`：

```python
for cid in conclusion_ids:
    has_virtual_pred = any(
        dag.is_virtual_edge(...)          # 只看 virtual (claim_ref / expr_ref / var_ref)
        for u in dag.graph.predecessors(cid)
    )
    if not has_virtual_pred:
        orphan_count += 1
```

但 `build_dag.py` 对没有显式依赖的推导/结论步**会自动加 `double_barrier_edge`**（L402-413）——这个 fallback 本意是"弱支持"，不是"没有支持"。当前代码把它**视为完全没支持**，放大了 false-orphan。

**推荐 fix**：把 orphan 判定改为
```python
support = #virtual + 0.5*#double_barrier + 0.3*#solid
has_support = (support > 0)
```
或给 `no_orphan` 一个连续分数而不是二值。

### 2.3 `rollout_srt.py` 里的 `orphan_step` **永远是 None**（代码 bug）

`scripts/rollout_srt.py:88-97`：

```python
dag = ReasoningDAG.from_trace(text) if hasattr(ReasoningDAG, "from_trace") else None
# ReasoningDAG 没有 from_trace 方法
if dag:
    orphans = dag.orphan_conclusion_nodes() ...
    # ReasoningDAG 没有 orphan_conclusion_nodes 方法
```

两个方法 `ReasoningDAG` 都不存在（`src/dag/graph.py` 只有 `from_dict`/`from_json`），`hasattr` 检查静默返回 `None`。

**后果**：`P_r` 里 `{k}` 永远用 **默认 `k=1`**，topology-aware revision prompt 降级成通用 prompt。

**推荐 fix**：改用 `build_dag_from_answer(text)` + `_orphan_conclusion_ratio` 里的节点列表，把真正的 orphan index 写入 `P_r`。

---

## 3. TVSD pipeline 完整性

### 3.1 Phase III-A（SRT）**部分可跑但关键信号失效**

- `scripts/rollout_srt.py` CLI 有 entry point，能跑，但 `orphan_step` bug（见 §2.3）使 `P_r` 退化。
- 产出需经 `src/distill/build_srt_data.py` → `data/srt_ready/train.jsonl` 才能给 SFT 用。
- **没有**实际运行记录 — `output/srt_9b/final` 不存在（下面 §3.2 的 OPSD 依赖它）。

### 3.2 Phase III-B（OPSD）**三处关键偏差于论文公式**

`src/distill/opsd_trainer.py:255-288`：

```python
teacher_logits = compute_teacher_logits(teacher, tok, problem, y, P_r, ...)  # (x, y, P_r) 上下文
s_out = student(input_ids=s_ids)                                            # (x, y) 上下文
T = min(student_logits.shape[0], teacher_logits.shape[0])
sl = student_logits[-T:]; tl = teacher_logits[-T:]      # ← 粗暴截尾对齐
log_p = F.log_softmax(sl/temp, -1); log_q = F.log_softmax(tl/temp, -1)
kl = (log_p.exp() * (log_p - log_q)).sum(-1).mean()     # 全序列 mean
```

偏差：
1. **token-position 对齐错误**：teacher 上下文里多了 `P_r` 段（user turn），student 没有，两者的 "第 t 个位置" 指的不是同一个 token。直接截最后 T 个让 KL 对齐到**不同的预测目标**。
2. **没对 student 生成段 mask**：`.mean()` 覆盖 prompt + y 所有位置。论文公式里 KL 只跨 `y` token。
3. **没用现成的 `reverse_kl_loss(mask=...)`**（`src/distill/reverse_kl_loss.py`）。

`build_prompt_dispatch` 调用**没传 `orphan_step`**（L247-250），所以 training-time `P_r` 里 `{k}` 也是默认 `k=1`——Phase III-B 同样失去 topology conditioning。

### 3.3 Configs 依赖一个**没跑过的 SRT checkpoint**

- `configs/opsd_9b.yaml`、`configs/opsd_student_4b.yaml` 里 `teacher_adapter: output/srt_9b/final`
- `output/srt_9b/final` 当前**不存在**，因为 Phase III-A 没真正训完
- 所以 Phase III-B 直接启动就会 crash

### 3.4 `student_4b_sft_distill_v3` 是什么？

从 `output/sft_distill_4b/v0-20260417-121952/checkpoint-2034` 路径和 `configs/sft_student_4b.yaml`（如存在）看：
- 这是**纯 ms-swift SFT** on teacher 过滤后的 traces
- 没有 Phase III-A 的 revision loss、没有 Phase III-B 的 KL
- → **跟 TVSD 无关**

这直接解释了 `token_ratio=1.01`：SFT-distill 复刻的是 teacher 的**完整 `<think>` 序列**，不是 TVSD 想要的"学生自己短、teacher logits 每 token 指导"。

---

## 4. 五条按 ROI 排序的优化动作（for 新会话）

| # | 改动 | 文件 + 大致位置 | 预期收益 | 代价 |
|---|---|---|---|---|
| **1** | **Reward floor + std floor 统一注入** | `src/reward/composite_reward.py:420-437`，所有 `Topo*Reward._combine`；把 `MIN_STD + NOISE_EPS` 模式推广到 gated/composite | 直接解 `frac_reward_zero_std=0.36`；`reward_std` 稳定后 GRPO 才能真正继续学 | 低（几十行） |
| **2** | **orphan 定义加权化** | `src/reward/topo_reward.py:119-135`：`no_orphan = 1 - weighted_orphan_ratio`，把 solid/double_barrier 当作弱支持 | 拓扑信号更连续；训练期 topo reward 不再被过严的 orphan 判断压 0 | 低 |
| **3** | **Gated reward 要么加 τ、要么改名** | `composite_reward.py:766-841`：选 A）加 `if outcome >= τ: boost else: no boost`；选 B）改类名 `TopoAdditiveReward` + 更新文档 | 让文档和代码对齐；当前 gated-v2 adapter 本质是"SFT + 噪声"的描述有代码支持 | 低-中（改 + 可能重训） |
| **4** | **修 rollout_srt 的 orphan_step bug** | `scripts/rollout_srt.py:88-97`：用 `build_dag_from_answer` 而不是不存在的 `ReasoningDAG.from_trace`；在 `src/distill/opsd_trainer.py:247-250` 把 `orphan_step` 传给 `build_prompt_dispatch` | `P_r` 里 `{k}` 真正指向 orphan 步骤；topology-aware revision prompt 生效 | 低（<20 行） |
| **5** | **OPSD KL 正确对齐 + mask** | `src/distill/opsd_trainer.py:255-288`：用 `reverse_kl_loss(mask=...)`，用 `tokenizer` 精确得到 student `y` 段的起止 index；重训 4B student | 压缩叙事真能成立 — student 学会每 token 和 teacher 对齐、且只在 `y` 段学 | 中（~1-2 天实验时间，含 Phase III-A 训练 `output/srt_9b/final`） |

**执行顺序建议**：
- 立即（≤ 2h 代码 + 4-6h 训练）：动作 1 + 2 + 4 + 续训 TopoPRM hier 300 步（warm start from `checkpoint-79`）
- 半天（代码 + 2 天训练）：动作 3
- 1-2 天（代码 + 1 天 Phase III-A + 1 天 Phase III-B）：动作 5 全链 TVSD

---

## 5. 论文叙事影响

- `sections/3_method.tex` 和 `sections/6_appendix.tex` 的"sentence-level claim filtering" 和 "topology-aware revision prompt" 描述**当前代码没完全做到**。两条路：
  - A. 修代码到与论文一致（动作 2、4、5）
  - B. 改论文到与代码一致（弱化两处描述，admit implementation simplification）
- 建议 A，因为文档现在说的这两个细节恰好是我们跟 baseline 区别的核心卖点。

- `sections/4_experiments.tex` RQ4 的 "compression" narrative 需要**换标的**：
  - **删除** `student_4b_sft_distill_v3` 作为压缩证据（因为没压缩）
  - **新增** `topoprm_hier_qwen25_7b_v3` 作为压缩证据：`token_ratio=0.28, dAcc=-11.3pp`，在 MMLU 上甚至 `dAcc=+1.1`
  - 新会话跑 TVSD 4B 回来后再替换上去

## 6. 不在本次诊断范围

- Figure 生成（Pareto 图）——在 `docs/exp_roadmap_2026-04-20.md:R4`
- 新训 TVSD → 需要动作 4+5 + GPU 资源 + 1-2 天 → 在新会话做
