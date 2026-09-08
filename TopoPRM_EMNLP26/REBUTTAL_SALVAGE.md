# EMNLP #4012 Rebuttal 挽救材料

本文件包含以下部分：
1. **PART 1 — Author-Editor Confidential Comment**（英文成稿，总体 reconcile + 三条 Suggested Revisions 回应，直接复制到 OpenReview，≤5000 字符）
2. **PART 1B — 针对 AC 个人风险判断的逐点拆解 / Direct rebuttal to the AC's core risk assessment**（英文成稿，沿 AC "Personally, I think..." 的因果链逐环拆解，可作第二条 comment 或口头 argue 素材）
3. **PART 2 — Meta-Review Issue Report 决策与备用文案**（中文说明 + 英文成稿）
4. **PART 3 — 执行顺序建议**；**PART 4 — 数字核对表**

---

## PART 1 — Author-Editor Confidential Comment（主武器，先用这个）

> 收件人建议勾选：Program Chairs / SACs / ACs / Authors（默认即可）
> 说明：这条 comment 的真正读者是 SAC 和 PC，因为 AC 已经给了 2.5。它的作用是让上级看到"reviewer 共识是 3，AC 个人判断偏离共识，且核心 concern 已被 evidence 回应"。语气尊重、就事论事、用数据说话。
> 与 1B 的分工（总-分）：PART 1 是"总"——面向 SAC/PC 的**总体 reconcile**，覆盖 reviewer 共识、noise 有界、correctness gating、实证增益、三条 Suggested Revisions 的逐条回应；PART 1B 是"分"——只沿 AC 那段个人判断的**因果链逐环深挖**。PART 1 的第 2 点（correctness gating）在这里是**总体论证的一环**，在 1B 里被展开为 Link 3 的"决定性断裂点"。两者各自自足、可独立提交，交叉引用见文末推荐用法。

### 英文成稿（复制这段）

**Title:** Reconciling the meta-review's risk assessment with reviewer consensus and audited evidence

We are grateful to the AC for the careful synthesis, and write to respectfully reconcile one gap: all three reviewers rate the paper Findings-acceptable (Overall 3; Soundness 3/3/3.5; Excitement 3/3/3.5), whereas the meta-review lands at 2.5, driven by a personal concern that a heuristic DAG extractor "severely limits practical reliability and real-world deployment potential." We fully agree the extractor is a noisy heuristic, not a proof-level verifier — we narrowed the claim accordingly during discussion — but the following points show the noise concern does not undercut a Findings contribution.

**1. The noise is measured and bounded, not open-ended.** During discussion we added an external edge-level audit on 120 stratified GSM8K/MATH traces against two references computed outside the training pipeline. Against the human-adjudicated reference the extractor reaches P/R/F1 = 0.58/0.71/0.64 (0.48/0.59/0.53 against a held-out Qwen3-32B annotator), and a second, architecturally distinct annotator (Qwen2.5-32B) agrees with the first at 0.90 raw / Cohen's kappa 0.73, so the reference is not one model's artifact. The per-edge-type breakdown localizes error to a single type: variable-only overlap has precision 0.25 and accounts for 52% of all false positives, whereas expression/claim-reuse edges reach 0.59–0.63. We now guard the weak type (variable edges require two shared variables; fallback order edges require local lexical overlap). This is a signal with quantified, non-random discriminative power and one known, mitigated failure mode — not an untested heuristic.

**2. Correctness gating structurally prevents noisy edges from corrupting the reward.** ACE computes advantages *within* the correct and wrong strata, so structural credit can only re-rank traces inside a stratum and never lift a wrong answer over a correct one. A mislabeled edge can thus reorder same-outcome peers but never flip the correctness decision that dominates the reward, so imperfect edge precision does not propagate into the objective. Extractor noise therefore cannot produce the "unstable, biased reward" failure mode an additive scheme would; this is a design choice, not an unaddressed risk. The baseline the concern implicitly favors — outcome-only GRPO — uses a coarser binary signal well-documented to induce reward hacking; TopoPRM layers a graded, correctness-gated structural signal on top of it.

**3. The empirical gains exceed what noise explains.** On the matched DeepSeek-R1-Distill-Qwen-7B setting the nine-benchmark average is 58.3 vs 55.1 (+3.2). The mechanism ablation on Qwen3.5-9B shows full TopoPRM at 45.6 average vs 40.8 for outcome-only GRPO, 41.8 without topology, and 41.7 without ACE — topology contributes beyond length/continuity, and ACE additionally cuts GRPO group-collapse from 57.8% to 37.9%. A purely noisy reward does not produce this consistent, mechanism-attributable gain.

**4. A robustness signature, not a fragility one.** If extractor noise dominated the reward, we would expect gains on only one or two benchmarks and no correlation between topology and correctness. We observe the opposite: gains span GSM8K/MATH/Olympiad/general reasoning, and the high-minus-low-topology correctness gap *widens* under TopoPRM training (0.150 for outcome-only GRPO to 0.171 for full TopoPRM) — structure becomes a stronger predictor of correctness after training, not a decoupled artifact.

**On the meta-review's framing and its three Suggested Revisions.** "Practical reliability and real-world deployment potential" is a bar for a deployed verifier, not the Findings criterion, which asks whether the contribution is sound, novel, and empirically supported — as the three reviews affirm. Each of the AC's three consolidated revision points was answered with new evidence during discussion: (i) *extractor validation* — the external human/LLM edge-level audit above, with per-edge-type failure analysis; (ii) *attribution of gains* — the mechanism-isolation ablation separating topology from length, continuity, and ACE, plus a matched outcome+length baseline; (iii) *presentation/rigor* — Table 1 split into quoted vs. matched-reproduced rows with per-row metadata, the "<500 tokens" wording corrected to "up to 24% fewer tokens on GSM8K, about 13% on the four-primary-benchmark mean," three-seed uncertainty with Wilson intervals, and separated public-benchmark vs. rubric reward definitions. The concern was not left open; it was measured, scoped, and mitigated on the record.

We respectfully ask the AC and SACs to weigh whether a 2.5 reflects the unanimous reviewer consensus and the post-rebuttal evidence, and whether the residual concern — a bounded, correctness-gated, empirically validated structural signal — is more consistent with the Findings bar the three reviewers converged on. We are glad to clarify any further point.

---

**字符数**（口径：从 "We are grateful..." 到 "...clarify any further point." 的正文，含空格、含段间空行）：
- **含 markdown 星号/斜体标记：4,874 字符**（提交到 OpenReview 复制的就是这个口径，因为正文里保留了 `**...**`/`*...*` 标记）；
- 纯文本（去掉 markdown 星号后）：4,844 字符。
- Title 单独字段 90 字符（Title 在 OpenReview 里是独立字段，不计入 Comment 的 5000 字符额度）。

正文两种口径均在 OpenReview 5000 单条上限内（约 120–160 字符缓冲）。逐字数字核对见文末"数字核对表"。

---

## PART 1B — 针对 AC 个人风险判断的逐点拆解 / Direct rebuttal to the AC's core risk assessment

> 用途说明（中文）：PART 1 是"总"——面向 SAC/PC 的总体 reconcile。本节 1B 是"分"——专门沿 AC 那段 "Personally, I think..." 的因果链（surface matching → graph noise → unstable/biased reward → severely limits reliability/deployment）逐环拆解。可作为**第二条 Author-Editor Comment**单独提交，或作为 discussion 中**口头 argue / 追加回复**的素材。语气：先肯定 AC 判断中合理的部分，再用 evidence 把 "severely limits" 降级为 "a bounded, mitigated limitation"。

### 英文成稿（复制这段）

**Title:** On the AC's core risk — why bounded extractor noise does not "severely limit" a correctness-gated structural reward

We are grateful for the AC's precisely articulated concern. As we read it, it forms a four-link causal chain: *surface matching → high graph noise (spurious edges + missing links) → "unstable, biased reward" → "severely limits practical reliability and real-world deployment."* We appreciate that each link is a reasonable prior, and we address them in order — agreeing where the AC is right and showing where the chain does not hold for TopoPRM specifically.

**Link 1 — "surface matching cannot build high-fidelity graphs."** The AC is right that surface matching cannot recover a proof-level dependency graph, and we do not claim it does — during discussion we narrowed the claim to a *surface-evidenced support signal*, not a high-fidelity or logically complete graph. The question is therefore not "is it high-fidelity?" but "is it better than random and measurably discriminative?", and the audit answers yes. Against a human-adjudicated reference the extractor reaches P/R/F1 = 0.58/0.71/0.64, far above chance, and a second, architecturally distinct annotator (Qwen2.5-32B) agrees with that reference at 0.90 raw / Cohen's kappa 0.73, so it is not one model's artifact. "High-fidelity is required" is a bar we never set; the signal only needs quantifiable discriminative power, which it demonstrably has.

**Link 2 — "spurious edges and missing links are systematic."** The AC is right that errors exist; the audit was designed to characterize them, not deny them. Crucially, they are not diffuse — the per-edge-type breakdown localizes them to a single type. Variable-only overlap has precision 0.25 and accounts for 52% of all false positives, whereas expression/claim-reuse edges reach 0.59–0.63 and explicit order/citation edges 0.61. We now guard the one weak type (variable edges require two shared variables; fallback order edges require local lexical overlap). So the failure mode is one identified, dominant, mitigated type — not the pervasive noise the chain assumes. "Prone to errors" is true; "uncontrollably noisy" is not.

**Link 3 — "graph noise introduces unstable, biased reward." This is the decisive break in the chain.** Even granting residual edge noise, it cannot become "biased reward" in TopoPRM, because correctness gating severs this link by design. ACE computes advantages *within* the correct and wrong strata separately, so a structural signal can only re-rank same-outcome peers and can never lift a wrong answer above a correct one. A mislabeled edge may reorder traces that share an outcome, but it cannot flip the correctness decision that dominates the reward — noisy edges are structurally barred from producing a *biased* (correctness-inverting) signal. That failure mode belongs to an *additive* scheme; TopoPRM instead layers a graded structural signal on top of a correctness gate precisely so extractor noise stays bounded to intra-stratum reordering. The AC's third link presupposes a reward TopoPRM does not use.

**Link 4 — "this severely limits practical reliability and real-world deployment potential."** Two responses. On criterion: "deployment potential" is a bar for a shipped verifier, not the Findings criterion, which asks whether the contribution is sound, novel, and empirically supported — which the three reviewers unanimously affirm (Overall 3/3/3; Soundness 3/3/3.5). On evidence: if noise truly dominated, the reward would be near-random, yielding gains on only one or two benchmarks and no structure–correctness coupling. We observe the opposite — consistent gains across nine benchmarks (DR1-7B 58.3 vs 55.1, +3.2), a mechanism ablation attributing them to topology and ACE (full 45.6 vs outcome-only 40.8, w/o topology 41.8, w/o ACE 41.7; ACE cuts group-collapse 57.8%→37.9%), and, most tellingly, a high-minus-low-topology correctness gap that *widens* after training (0.150→0.171). A signal that "severely limits reliability" does not become a *stronger* predictor of correctness through training. The honest characterization is "a bounded, measured, mitigated limitation," not a severe one.

**In sum.** We agree with the two premises that are true — surface matching is not high-fidelity, and the extractor makes errors — and we have measured, localized, and guarded those errors. But the inference from "noisy edges" to "severely limited reliability" passes through a step that TopoPRM's correctness gating structurally blocks: noise can reorder same-outcome traces but cannot bias the correctness signal. With consistent nine-benchmark gains and a *strengthening* topology–correctness coupling, the evidence supports treating this as a bounded, mitigated limitation fit for a Findings contribution, not a severe risk. We thank the AC for pressing on exactly the right mechanism, and are glad to discuss any link further.

---

**字符数**（口径：从 "We are grateful for the AC's precisely articulated concern..." 到 "...discuss any link further." 的正文，含空格、含段间空行）：
- **含 markdown 星号/斜体标记：4,796 字符**（提交口径，正文保留 `**...**`/`*...*`）；
- 纯文本（去掉 markdown 星号后）：4,760 字符。
- Title 单独字段 115 字符。

若作为独立的第二条 Author-Editor Comment 提交，正文两种口径均在 OpenReview 5000 上限内（约 200–240 字符缓冲）。所有数字均与 PART 4 核对表一致，未引入新数字。

### 推荐用法（合并 vs 分两条 comment）

**不能合并成一条 comment。** PART 1 正文 4,874 字符、1B 正文 4,796 字符（均为"含 markdown 星号"口径），二者相加 ≈9,670 字符，远超 OpenReview Author-Editor Comment 的 5,000 单条上限。故有两种落地方式：

- **推荐（首选）——分两条 comment 先后提交**：先发 PART 1（总体 reconcile，面向 SAC/PC），紧接着第二条发 PART 1B（针对 AC 个人判断的深挖），在 1B 开头一句点明"as a focused follow-up to our reconciliation comment"。理由：PART 1 让上级快速看到"共识 3 vs AC 2.5 + concern 已回应"的全局图景，1B 则给愿意深究机制的 AC 一个逐环、尊重式的技术拆解，两者读者与功能不同，分开反而更清晰、每条都在字数内且自足。
- **备选——只发一条**：若希望克制、只投一次，则**只发 PART 1**（它已内含 correctness gating 与三条 revision 的回应，可独立成立），把 1B 留作 discussion 里**口头/追加回复 AC 时的弹药**——当 AC 在线追问 noise 问题时，直接引用 1B 的 Link 3（correctness gating 断裂点）逐点回应。

**一句话建议**：默认走"分两条"；若团队判断 AC 情绪敏感、不宜显得"层层加码"，就走"只发 PART 1 + 1B 留作口头 argue"。两条路都不改动任何数字。

---

## PART 2 — Meta-Review Issue Report（备用武器，谨慎使用）

### 中文决策建议

**先不要提交 Issue Report。** 原因：

- Issue Report 是 ARR 里比较"重"的动作，等于正式质疑 AC 的评审程序，SAC/PC 会介入审查。滥用或理由不充分会给 AC/SAC 留下负面印象，反而不利。
- 你现在的最优先动作是先用 Author-Editor Confidential Comment（上面 PART 1），它同样能被 SAC/PC 看到，风险低、收益接近。
- **只有在满足以下条件时才提交 Issue Report**：
  1. AC 在 comment 后仍不调整，且
  2. 你能指出**明确的程序性/事实性问题**（不是"我不同意打分"，而是"AC 用错了标准 / 误读了数据 / justification 与分数矛盾"）。

**本案确实存在两个可作为 Issue 的正当理由**（如果要提，用这两个，不要用"分数太低"这种主观理由）：

1. **评审标准错位**：AC 的压分理由建立在 "real-world deployment potential" 和 "practical reliability" 上，而这不是 ARR/Findings 的评审维度。ARR 评的是 soundness/novelty/empirical support，三项 reviewer 均给正面。
2. **Meta 分数与 justification 不一致**：AC 的 "Reasons to Publish" 写了三条实打实的优点，"Suggested Revisions" 的三条核心 concern 在 discussion 中都已用新 evidence 回应，但最终分数（2.5）却低于三位 reviewer 的一致评分（3.0），且 meta-review 未说明为何低于 reviewer 共识。

### 英文成稿（若决定提交，复制这段；建议简短、克制）

**Title:** Request to reconcile meta-review score with reviewer consensus and evaluation criteria

We respectfully raise two points for the SACs' consideration regarding the meta-review of Submission 4012.

First, the reduction below the unanimous reviewer assessment (all three: Overall 3, Findings; Soundness 3/3/3.5; Excitement 3/3/3.5) appears to rest on a criterion outside the ARR rubric. The meta-review's decisive concern is stated as the "practical reliability and real-world deployment potential" of the DAG extractor. Findings evaluates soundness, novelty, and empirical support rather than deployment readiness, and the paper does not claim a deployed verifier — it explicitly frames the extractor as a noisy, correctness-gated structural signal.

Second, the three core revision points the meta-review consolidates (extractor validation, attribution of gains, presentation/rigor) were each addressed with new evidence during discussion: an external human/LLM edge-level audit (human P/R/F1 = 0.58/0.71/0.64; second-annotator Cohen's kappa 0.73), a mechanism-isolation ablation separating topology from length and continuity (full 45.6 vs. outcome-only 40.8 vs. no-topology 41.8), and a corrected, split, per-row-documented Table 1 with separated reward definitions. We would be grateful if the panel could confirm whether these responses were weighed, as the final 2.5 sits below the reviewer consensus without an explicit rationale for the gap.

We are not contesting the AC's expertise; we only ask that the score be checked against the reviewer consensus and the post-rebuttal record. We remain happy to clarify any remaining point.

---

## PART 3 — 执行顺序建议（中文）

1. **立即**提交 PART 1 的 Author-Editor Confidential Comment（总体 reconcile）。这是低风险、高杠杆的一步。
2. **紧接着**提交第二条 comment——PART 1B（针对 AC 个人判断的因果链深挖），开头点明"as a focused follow-up to our reconciliation comment"。这是作者已确定的首选提交策略（分两条）。
3. **观察** AC 是否在 discussion 截止前调整（modified 时间戳会变）。
4. 若临近截止 AC 无动作、且你判断 SAC 有介入空间，**再**提交 PART 2 的 Issue Report（用"标准错位 + 分数与 justification 不一致"两个正当理由，不要用主观的"分数太低"）。
5. 全程语气：尊重 AC，不攻击个人判断，一切用 evidence 和 reviewer 共识说话。

## 关于是否值得继续投入的现实判断

- 2.5 Borderline + 三个 Findings，**进 Findings 的概率本来就存在**（很多 SAC 会向 reviewer 共识靠拢）。这份 comment 的目的是把天平推向 reviewer 共识。
- 即使这轮没进 main，Findings 对这篇工作是合理归宿；若连 Findings 都被拒，凭现有 evidence（edge audit + 机制消融 + 修正表格）下一轮 ARR 重投的基础已经相当扎实。

---

## PART 4 — 数字核对表（逐字对齐 PDF / discussion 记录）

下表列出 rebuttal 中用到的每个关键数字及其在 OpenReview PDF（`Rewarding the Graph Behind the Chain ... OpenReview.pdf`）中的出处。全部已逐字核对，无需修改（本次打磨未改动任何数值，仅重新组织表述）。

| 关键数字 | 本文件中的用法 | PDF 出处（作者原始表述） |
|---|---|---|
| Reviewer 评分 HxUk | Soundness 3.5 / Excitement 3.5 / Overall 3 / Confidence 3 | Review HxUk：Soundness 3.5、Excitement 3.5、Overall 3、Confidence 3 |
| Reviewer 评分 B5w7 | Soundness 3 / Excitement 3 / Overall 3 / Confidence 4 | Review B5w7：Soundness 3、Excitement 3、Overall 3、Confidence 4 |
| Reviewer 评分 TsKG | Soundness 3 / Excitement 3 / Overall 3 / Reproducibility 2 | Review TsKG：Soundness 3、Excitement 3、Overall 3、Reproducibility 2 |
| Meta score | 2.5 = Borderline Findings | Meta Review (AC qohD)："Overall Assessment: 2.5 = Borderline Findings" |
| 人工验证 edge P/R/F1 | 0.58 / 0.71 / 0.64 | Response to HxUk 表 R1、Response to TsKG R1："Human annotation P=0.58/R=0.71/F1=0.64" |
| Qwen3-32B annotator P/R/F1 | 0.48 / 0.59 / 0.53 | Response to HxUk 表 R1、Response to TsKG R1："Qwen3-32B annotator 0.48/0.59/0.53" |
| 第二标注者 Qwen2.5-32B | 0.90 raw agreement / Cohen's kappa 0.73 | Response to HxUk R1、B5w7 R1、TsKG R1："Qwen2.5-32B reaches 0.90 raw / kappa 0.73" |
| variable-only overlap precision | 0.25，占 52% false positives | Response to B5w7 表：variable-only overlap 0.25，"52% of all false positives" |
| expression/claim reuse precision | 0.59–0.63 | Response to B5w7 表：expression/claim reuse 0.59–0.63 |
| explicit order/citation precision | 0.61 | Response to B5w7 表：explicit order/citation 0.61 |
| DR1-7B 九基准平均 | 58.3 vs 55.1（+3.2） | Response to HxUk R3："58.3 vs. 55.1 on DR1-7B, Table 1" |
| Qwen3.5-9B 机制消融 | full 45.6 / outcome-only 40.8 / w/o topology 41.8 / w/o ACE 41.7 | Response to B5w7 表 R2、TsKG 表 R1：45.6 / 40.8 / 41.8 / 41.7 |
| 去掉 continuity 崩溃 | 16.3（Collapse% 68.8） | Response to TsKG 表 R1："w/o continuity 16.3 / 68.8" |
| Collapse% full vs outcome-only | 37.9% vs 57.8% | Response to B5w7/TsKG 表：Full 37.9%，Outcome-only 57.8% |
| topology-correctness gap | 0.150（outcome-only）→ 0.171（full） | Response to B5w7 R2："gap widens from 0.150 ... to 0.171" |
| token 减少（修正后） | GSM8K 最高 24%，四主基准均值约 13% | Response to TsKG R2："up to 24% ... on GSM8K, about 13% on the four-primary-benchmark mean" |
| PRM 复用 reranking | 70.0 → 71.2（MATH-500） | Response to B5w7 R3、Confidential Comment 草稿："71.2 vs 70.0" |

**关于 "15–24%" 的重要提示（已同步）**：作者在 Response to TsKG (W2/R2) 中已**当众修正** token 措辞为 "up to 24% on GSM8K, about 13% on the four-primary-benchmark mean"。本次挽救工作已把论文正文 tex 中所有与该承诺冲突的全局声称一并同步为一致措辞，具体如下：
- `sections/0_abstract.tex` L12（英文 abstract 主体）：`up to 24\% fewer tokens` → `up to 24\% fewer tokens on GSM8K and about 13\% fewer on the four-primary-benchmark mean`。
- `sections/1_intro.tex` L26（贡献点）：`compressing max response length by up to 24\%` → `compressing response length by up to 24\% on GSM8K and about 13\% on the four-primary-benchmark mean`。
- `sections/4_experiments.tex` L39（Shorter Chains 段）：`generating 15--24\% fewer tokens` → `generating up to 24\% fewer tokens on GSM8K and about 13\% fewer on the four-primary-benchmark mean`。
- `sections/6_appendix.tex` L270（in-domain critique 段的括注）：`this compression ratio further enlarges to 15\%--24\%` → `this reaches up to 24\% on GSM8K and about 13\% on the four-primary-benchmark mean`。

**未改动且属正常的项**（不同粒度，非冲突，故保留）：
- `6_appendix.tex` L270 句首的 `11.2\% shorter`：这是 **in-domain 教师-批判任务**上的压缩率，与公开基准的 24%/13% 是不同数据集上的不同度量，保留正确。
- `0_abstract.tex` L5 / L9、`1_intro.tex` L59 里的 `11%`/`11.2%`/`+13.4`：均在 `%` 注释行内（未渲染进正文）或为已弃用旧稿，无需改动。

结论：tex 正文的 token 措辞现已与 rebuttal 承诺及 discussion 记录**完全一致**，camera-ready 不再有 text/table 数字自相矛盾的风险。

---

### 表间行标签一致性修复（efficiency.tex ↔ main_detailed.tex）

reviewer TsKG 点名 "internal inconsistencies undermine empirical claims"。本次排查发现 `tables/efficiency.tex`（`tab:efficiency`）与 `tables/main_detailed.tex`（`tab:main_detailed`）存在**行标签不一致**，已按作者确认的真值修正 efficiency 表标签，**未改动任何数值**。

**核对方法**：按 GSM8K 的 `Tok / Acc/kTok` 逐行对齐，并跨全部 4 个基准（GSM8K/MATH-500/AIME'24/Omni-MATH 的 Tok 列）交叉验证，避免单点误判。

**已确定并完成的修正**（均经 4 基准全维度验证，属硬证据）：
- `SFT baseline`（GSM8K 796 / 118.2）→ **`TopoPRM (Full)`**。作者已确认 796=TopoPRM(Full)；与 `main_detailed` 的 Full 行（796/118.2、MATH 1454、Omni 1592）全维度一致。原 "SFT baseline" 标签是错的。
- `TopoPRM (Gated)`（GSM8K 1006 / 93.2）→ **`w/o ACE`**。与 `main_detailed` 的 w/o ACE 行（1006/93.2、MATH 1515/33.5、AIME 2560/7.8、Omni 2463）全维度一致。
- 其余三行（`Outcome Only GRPO` 1047、`w/o Topology` 946、`w/o Continuity` 1528）两表标签本就一致，无需改动。

**留待作者确认（未改，已在 efficiency.tex 内以 `% TODO(author-confirm)` 注释标注）**：
- `TopoPRM (Hier.)` 行（GSM8K 1026 / 91.1）在 `main_detailed` 中**任何行、任何基准都无匹配**；其数据与 Base（1536/35.9）也不符，故既非 Base 也非任何现有消融行。无法凭数据确定其正确身份。**候选方案**：(a) 系陈旧/冗余的旧变体标签，应整行删除；(b) 系 `main_detailed` 中缺失的一个真实配置，应保留并在 main_detailed 补齐。二者需作者按实验记录裁定。
- 附带观察（非本次改动范围）：efficiency 表 Full 行 MATH-500 的 `Acc/kTok` 为 34.9，而 main_detailed 对应值为 35.1；属数值层面的轻微出入，未在本次"只改标签"范围内处理，建议作者一并核对。

**数值改动**：无。本次仅修改行标签文本与加粗，LaTeX 列数、`&` 对齐、`\textbf` 加粗位置均保持正确。
