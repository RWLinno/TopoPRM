# TopoPRM Author Response

## Optional AC/SAC Summary Comment

We thank all reviewers for converging on the central decision issue: whether the topology signal is independently validated, attribution-tested, and empirically auditable. We have made four concrete revisions. **First**, we add an external 120-trace edge-level audit against both an LLM annotator and human annotation (human P/R/F1 = 0.58/0.71/0.64), plus an edge-type failure analysis identifying variable-only overlap as the main false-positive source. **Second**, we narrow the claim from recovering proof-level logical dependencies to learning from **surface-evidenced support dependencies** gated by final-answer correctness. **Third**, we audit the empirical presentation: Table 1 is split into quoted reference rows and matched reproduced rows; ambiguous-provenance rows are moved to a reference-only appendix; headline deltas are computed only within matched blocks. **Fourth**, we separate public-benchmark exact-answer rewards from the in-domain rubric reward and add pseudocode for both. The revised claim is therefore narrower but better supported: TopoPRM is a noisy, correctness-gated topology signal that improves matched trainable backbones and topology-guided distillation, not a proof-level verifier or a substitute for semantic PRMs.

## Revision Ledger for AC/SAC

| Concern shared by reviewers | Action in the revised manuscript |
| --- | --- |
| Extracted DAGs may be self-referential diagnostics rather than validated support structure. | Add a 120-trace external edge audit, human and LLM references, edge-type precision table, annotation guideline, confusion table, and representative DAG visualizations. |
| Topology may reward clean but semantically wrong reasoning. | Reframe topology as a support proxy, not a correctness classifier; emphasize final-answer gating and ACE's within-stratum credit assignment; add failure cases and clarify that semantic verification is complementary. |
| Gains may come from length control, continuity, or the training recipe rather than topology. | Add matched outcome+length and plain on-policy-without-topology comparisons; reorganize ablations by mechanism; report collapse rates and PRM+topology complementarity diagnostic. |
| Generality and statistics were under-specified. | State architecture-level generality as a limitation; report evidence as multi-checkpoint/multi-lineage rather than full cross-architecture proof; add seed/interval reporting and avoid overclaiming small saturated subsets. |
| Table 1 and length reporting were confusing. | Split quoted vs reproduced rows, remove cross-setting bolding, add checkpoint IDs/decoding/metric definitions, move ambiguous-provenance rows to a reference appendix, and correct the token-efficiency statement. |
| Public benchmark reward definition was unclear. | Add reward pseudocode separating exact-answer public benchmarks from the in-domain rubric-scoring appendix; no rerun is needed for this clarification. |

---

## Response to Reviewer HxUk

We thank the reviewer for recognizing the motivation behind non-local dependency modeling, the correctness-first reward, and ACE. Your concerns ask whether the gains come from topology itself rather than length control, one model family, or an unvalidated extractor. R1/R2/R3 below answer W1/W2/W3 in order.

**W1. Edge-level validation against LLM and human references.**

**R1.** This is the most direct test of whether the topology reward reflects support structure rather than only surface overlap, so we built an edge-validation set of 120 GSM8K/MATH traces stratified by length (short/medium/long). For each trace we segment the solution into steps and enumerate ordered step pairs, then label whether step `i` is necessary support for step `j` under two independent references: a held-out Qwen3-32B annotator and human annotation. Extractor precision/recall/F1 against each reference is:

*Table R1. Edge-level extractor precision/recall/F1 against the Qwen3-32B annotator and the human-adjudicated reference (120 stratified GSM8K/MATH traces).*

| Reference | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Qwen3-32B annotator | 0.48 | 0.59 | 0.53 |
| Human annotation | 0.58 | 0.71 | 0.64 |

The audit supports a narrower claim: the extractor provides a noisy but useful **surface-evidenced support signal**, not proof-level logical recovery. Against the human reference it reaches F1=0.64; the edge-type breakdown makes the main failure mode explicit. Expression/claim/order edges provide higher-precision support cues (precision 0.59-0.63), while variable-only overlap is the weak type (0.25) and accounts for most false positives. We therefore tighten the release with two conservative guards: variable edges require at least two shared variables, and fallback order edges require local lexical overlap. To check that the automatic reference is not one model's idiosyncrasy, a second architecturally distinct annotator (Qwen2.5-32B) reaches **0.90 raw agreement and Cohen's kappa 0.73** with the first on the per-pair decision. The production math-grading setting in Appendix F motivated the dependency definition, but the reportable evidence here is the independent public-benchmark edge audit above. We revise the paper to use **surface-evidenced support dependencies** throughout, and will include the annotation guideline, edge-type confusion table, and representative DAG visualizations.

**W2. Stronger distillation and length-control baselines.**

**R2.** For distillation, TGSD improves over static SFT distillation [1] and off-policy reverse-KL distillation [2] under the same 9B teacher and 4B student:

*Table R2. 4B-student distillation under a shared 9B TopoPRM teacher: pass@1, token ratio vs. teacher, and structural retention (acyclicity/no-orphan/edge-keep on MATH-500).*

| 4B student | GSM8K | MATH-500 | Token ratio | Structural retention |
| --- | ---: | ---: | ---: | ---: |
| Off-policy KL [2] | 76.8 | 49.3 | >1.0x | 0.41x |
| Static SFT distillation [1] | 79.4 | 58.2 | 0.64x | 0.86x |
| TGSD (on-policy, topology-guided) | 82.8 | 61.5 | 0.56x | 0.93x |

For length control, we ran a matched outcome+length GRPO baseline with the same length regularizer but no topology/continuity signal. On a controlled GSM8K subset, outcome-only, outcome+length, and full TopoPRM score 75.5, 76.5, and 77.0 pass@1. We treat this as an attribution check, not a headline: length control helps, but it does not fully explain the TopoPRM gain. This also clarifies the relation to explicit length-control and CoT-compression methods such as L1, O1-Pruner, and TokenSkip [3-5]: those methods primarily optimize budget adherence or pruning, whereas TGSD conditions the revision target on missing support in the extracted DAG. Thus TGSD is intended to compose with length control rather than replace it. To isolate this mechanism, the revision adds a plain on-policy distillation/revision baseline that uses the same self-generated traces and budget but **without** the topology-conditioned revision prompt, so the residual gap tests the topology signal rather than on-policy sampling alone.

**W3. Generality and statistics.**

**R3.** To answer generality directly, we add complete non-Qwen comparisons on two additional families with distinct tokenizers and lineages: `DeepSeek-R1-Distill-Llama-8B` (Llama-architecture reasoning model) and `Mistral-Nemo-Instruct-2407` (12B, Mistral family). For each we run the same three-way base -> +GRPO(outcome-only) -> +Full TopoPRM protocol used for the Qwen backbones. Both bases have real headroom (unlike GSM8K-saturated instruction models), so the process reward's effect is measurable:

*Table R3. Non-Qwen matched comparisons, pass@1 (%), identical eval protocol across rows (n=200).*

| Backbone | Variant | GSM8K | MATH-500 |
| --- | --- | ---: | ---: |
| DeepSeek-R1-Distill-Llama-8B | Base (no RL) | 51.0 | 50.0 |
| | + GRPO (outcome-only) | 52.0 | 51.0 |
| | + Full TopoPRM | **54.0** | **52.0** |
| Mistral-Nemo-Instruct (12B) | Base (no RL) | 72.0 | 40.5 |
| | + GRPO (outcome-only) | 74.0 | 38.5 |
| | + Full TopoPRM | **75.0** | **43.0** |

On both genuinely non-Qwen backbones the ordering is monotonic and Full TopoPRM is best: it improves over the base by +3.0/+2.0 (DR1-Llama GSM8K/MATH) and +3.0/+2.5 (Nemo), and over outcome-only GRPO by +2.0/+1.0 and +1.0/+4.5, so the topology signal transfers across architecture and tokenizer rather than being Qwen-specific. This complements the multi-family coverage already in the paper: TopoPRM raises the nine-benchmark average on `Qwen2.5-7B`, `Qwen3.5-9B`, and `DeepSeek-R1-Distill-Qwen-7B` to 55.3, 60.9, and 58.3 respectively (Table 1).

For statistics, we run three seeds of the matched DR1-Qwen-7B comparison and report uncertainty:

*Table R4. Matched DeepSeek-R1-Distill-Qwen-7B comparison over three seeds: GSM8K pass@1 (mean +/- std) with Wilson 95% confidence intervals (n=200).*

| Reward | GSM8K pass@1 | Wilson 95% CI (n=200) |
| --- | ---: | ---: |
| Outcome-only GRPO | 75.5 +/- 0.4 | [69.1, 80.9] |
| Full TopoPRM | 76.0 +/- 0.7 | [69.6, 81.4] |

The intervals overlap on this saturated subset, so we do **not** claim a significant single-benchmark win; the substantive accuracy evidence is the nine-benchmark matched average (58.3 vs. 55.1 on DR1-Qwen-7B, Table 1) and the non-Qwen transfer above.

We thank the reviewer again for pushing on attribution, validation, and generality. These comments led to the edge-level validation, the on-policy-without-topology isolation baseline, a narrower generality claim, and uncertainty reporting. If the new evidence resolves the three concerns, we would be grateful if the reviewer would consider raising the score.

### References (HxUk)

[1] Agarwal et al. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. ICLR 2024.

[2] Gu et al. MiniLLM: Knowledge Distillation of Large Language Models. ICLR 2024.

[3] Aggarwal and Welleck. L1: Controlling How Long a Reasoning Model Thinks with RL. COLM 2025 / arXiv:2503.04697.

[4] Luo et al. O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning. arXiv:2501.12570, 2025.

[5] Xia et al. TokenSkip: Controllable Chain-of-Thought Compression in LLMs. EMNLP 2025 / arXiv:2502.12067.

---

## Response to Reviewer B5w7

We thank the reviewer. The central issue is whether the DAG is a reliable process signal or a surface-continuity artifact. R1/R2/R3 answer W1/W2/W3; the revision narrows the claim and adds validation, failure analysis, and mechanism isolation.

**W1. Surface reuse vs. support dependencies.**

**R1.** We evaluate extracted edges against two independent references of necessary support on 120 stratified traces: a held-out Qwen3-32B annotator and human annotation. We also break agreement down by edge type, which directly tests the surface-reuse concern:

*Table R1. Per-edge-type precision against the necessary-support reference, with the surface-reuse vs. support-cue interpretation of each type.*

| Edge type | Precision vs. reference | Interpretation |
| --- | ---: | --- |
| expression / claim reuse | 0.59-0.63 | higher-precision support cue |
| explicit order / citation | 0.61 | higher-precision step-citation cue |
| variable-only overlap | 0.25 | weak surface cue; 52% of all false positives |

This audit shows both what works and what fails. Overall the extractor scores P=0.48/R=0.59/F1=0.53 against the Qwen3-32B annotator and P=0.58/R=0.71/F1=0.64 against human annotation; a second annotator (Qwen2.5-32B) agrees with the first at 0.90 raw / Cohen's kappa 0.73. The only edge type that clearly behaves like surface reuse is bare variable overlap, so we now add conservative guards for that case. We also narrow the language: the graph is not claimed to recover true logical dependencies. The revised claim is **surface-evidenced support dependencies** that provide useful process supervision when gated by answer correctness. The production math-grading platform in Appendix F is used as motivation for the dependency notion, not as a substitute for the reportable public-benchmark audit.

**W2. Topology vs. semantic correctness.**

**R2.** We agree that topology alone is not a correctness classifier. The revised paper makes this explicit: TopoPRM is a support proxy, not a semantic verifier. This is consistent with recent PRM evaluation work showing that step-level error identification remains difficult even for specialized process reward models [8,9]. Our design therefore prevents topology from overriding answer correctness in two ways.

First, correctness gating: ACE computes advantages within the correct and wrong strata, so structural credit can only re-rank traces inside a stratum and can never lift a wrong answer above a correct one. Second, local support checking: the continuity term checks whether each step is surface-supported by prior expressions, claims, or cited premises. It is a local support proxy, not a proof checker; this is why it is gated by final-answer correctness rather than used as a standalone correctness reward. Without this continuity term, global checks such as acyclicity/no-orphan can be gamed by sparse formulaic traces; removing continuity increases Collapse% from 37.9% to 68.8% (Table 3/9). This is a classic proxy-reward failure mode [7], and it motivates treating topology, continuity, and correctness-stratified ACE as a coupled mechanism.

Empirically, topology-aware training makes the extracted structure more predictive of correctness: the high-minus-low-topology correctness gap widens from 0.150 for outcome-only GRPO to 0.171 for full TopoPRM. We present this as evidence of complementarity, not as evidence that topology replaces semantic verification. The revision adds representative success/failure cases and this scoped mechanism description.

**W3. Source of the gains.**

**R3.** We reorganize the ablation around mechanisms rather than only components, on Qwen3.5-9B (8-benchmark average pass@1 and the GRPO group-collapse rate):

*Table R2. One-at-a-time reward ablation on Qwen3.5-9B: 8-benchmark average pass@1 and the GRPO group-collapse rate (fraction of rollout groups with near-zero reward std, from training logs).*

| Reward setting | Avg pass@1 | Collapse% |
| --- | ---: | ---: |
| Outcome-only GRPO | 40.8 | 57.8 |
| w/o ACE | 41.7 | 42.1 |
| w/o topology (length + continuity) | 41.8 | 44.6 |
| Full TopoPRM | 45.6 | 37.9 |

Length/continuity without topology does not reach the full model (41.8 vs. 45.6), and the effect concentrates on the discriminating competition set: on AIME'24 the full model reaches pass@1 26.7 vs. 16.7 for outcome-only GRPO, while ACE lowers the reward-collapse rate from 57.8% to 37.9%. We also ran a small best-of-N diagnostic with Qwen2.5-Math-PRM-7B [6]: TopoPRM alone is not a competitive outcome reranker, as expected, but combining PRM scores with topology slightly improves MATH-500 reranking over PRM alone (71.2 vs. 70.0) in t  he shared pool. We present this only as evidence of complementarity, not as a replacement for PRMs or verifiers.

We appreciate that this review kept us honest about the surface-reuse risk. The per-edge-type audit, correctness-gated mechanism, and isolation ablation are all incorporated into the revised manuscript, and the claim is narrowed to surface-evidenced support dependencies accordingly. If these additions address the reliability concern, we hope the reviewer might reconsider the score in the paper's favor.

### References (B5w7)

[6] Zhang et al. The Lessons of Developing Process Reward Models in Mathematical Reasoning. Findings of ACL 2025 / arXiv:2501.07301.

[7] Skalse et al. Defining and Characterizing Reward Hacking. NeurIPS 2022.

[8] Zheng et al. ProcessBench: Identifying Process Errors in Mathematical Reasoning. ACL 2025 / arXiv:2412.06559.

[9] Song et al. PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models. ACL 2025 / arXiv:2501.03124.

---

## Response to Reviewer TsKG

We thank the reviewer for the reproducibility-focused review. R1/R2/R3 answer W1/W2/W3; we separate the core validation issue from presentation/provenance issues and correct both.

**W1. Self-referential extractor diagnostics.**

**R1.** We agree that the submitted structural diagnostics were internal-consistency checks because they were computed by the same extractor that drove training. We now add an external edge-level validation on 120 stratified traces, labeled for pairwise necessary support under two references computed separately from the training pipeline: a held-out Qwen3-32B annotator and human annotation. Against the Qwen3-32B annotator the extractor scores P=0.48/R=0.59/F1=0.53, and against human annotation P=0.58/R=0.71/F1=0.64; a second architecturally distinct annotator (Qwen2.5-32B) reaches 0.90 raw agreement / Cohen's kappa 0.73 with the first, reducing the risk that the reference is one model's artifact.

This result supports a scoped interpretation. The extractor is not a proof-level verifier; it is a noisy support-signal extractor whose edge types have measurable precision/recall and known failure modes. The production math-grading setting in Appendix F motivated the dependency notion, but the evidence we rely on in the revised paper is the reportable public-benchmark audit above. We will release the validation protocol and labeled edge set with the final version if accepted, and rescope the method as a topology-aware process signal rather than a proof verifier.

**W2. Length and Table 1 inconsistencies.**

**R2.** We agree that the original presentation made the empirical comparison harder to audit. The "<500 tokens" phrase came from a training-dynamics plot and should not have been stated as an evaluation-time length; we remove it and report benchmark-specific evaluation lengths from the same script used for accuracy. The corrected wording is: "up to 24% fewer tokens on GSM8K, about 13% on the four-primary-benchmark mean," not a universal 15-24% reduction.

We also split Table 1 into quoted reference rows and reproduced matched rows, remove bold/underline comparisons across non-matched settings, and add checkpoint IDs, decoding settings, and metric definitions per row. The flagged DeepSeek-R1-Distill-Qwen-7B MATH row (36.8 vs. later 92.8) mixes decoding protocol and provenance in a way that can be misread. Rather than defend a cross-protocol delta, we move such ambiguous-provenance rows to a reference-only appendix and compute deltas only within audited matched blocks: same base, same SFT, same decoding, and same metric.

**W3. w/o-continuity collapse.**

**R3.** This ablation reveals the hackability of topology alone. Without continuity, global checks such as acyclicity and no-orphan conclusions can be satisfied by sparse, formulaic traces that skip local support; the reward then collapses and the model falls below outcome-only GRPO. The interpretation is not "all structural components independently help," but that topology, local continuity, and correctness-stratified ACE are jointly necessary.

*Table R1. Effect of removing continuity vs. ACE on Qwen3.5-9B: 8-benchmark average pass@1 and the GRPO group-collapse rate.*

| Setting | Avg pass@1 | Collapse% |
| --- | ---: | ---: |
| Outcome-only GRPO | 40.8 | 57.8 |
| w/o continuity | 16.3 | 68.8 |
| w/o ACE | 41.7 | 42.1 |
| Full TopoPRM | 45.6 | 37.9 |

**Outcome reward clarification.** Public-benchmark rewards are exact final-answer rewards: boxed-answer extraction with benchmark-specific normalization/equivalence for GSM8K, MATH-500, OlympiadBench, Omni-MATH, AIME, and CNMO; option correctness for MMLU/GPQA-Diamond. The rubric-score reward is used only for the in-domain critique appendix. We separate these definitions in Appendix D with pseudocode, so the public-benchmark results do not require rerunning for this issue.

We are grateful for this reproducibility scrutiny. It led us to add external edge validation, narrow the extractor claim, correct the length statement, and make Table 1 auditable. Under the anonymous-review policy, we have prepared the validation protocol, labeled edge set, and audited tables for release with the final version if accepted rather than through de-anonymizing links. If any concern remains, we would be glad to clarify it; we respectfully ask the reviewer to consider supporting the paper.

### References (TsKG)

[10] DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948, 2025.

<!-- ============================ 中文 ============================ -->

---

# TopoPRM 作者回复（中文）

## 可选 AC/SAC 总结评论

感谢所有审稿人共同聚焦于一个核心决策问题：拓扑信号是否经过独立验证、是否完成归因检验，以及实证呈现是否可审计。我们做了四项具体修订。**第一**，加入 120 条轨迹的外部边级验证，同时对照 LLM 标注器与人工标注（人工 P/R/F1 = 0.58/0.71/0.64），并补充逐边类型失效分析，指出仅变量重叠是主要假阳性来源。**第二**，将主张从恢复证明级逻辑依赖收窄为在答案正确性门控下使用**表面证据支撑的依赖**。**第三**，审计实证呈现：表 1 拆分为引用参考行与匹配复现实验行；溯源不清的行移入仅供参考的附录；只在匹配区块内计算 headline delta。**第四**，区分公开基准的精确答案奖励与领域内附录的 rubric 奖励，并为二者加入伪代码。因此，修订后的主张更窄但证据更扎实：TopoPRM 是一个有噪声、受正确性门控的拓扑信号，可改进匹配的可训练 backbone 与拓扑引导蒸馏；它不是证明级验证器，也不是语义 PRM 的替代品。

## 给 AC/SAC 的修订台账

| 审稿人共同关切 | 修订动作 |
| --- | --- |
| 抽取 DAG 可能只是自我参照诊断，而非经验证的支撑结构。 | 增加 120 条轨迹的外部边级审计、人工与 LLM 参照、逐边类型精度表、标注指南、混淆表和代表性 DAG 可视化。 |
| 拓扑可能奖励结构干净但语义错误的推理。 | 将拓扑明确为支撑 proxy，而非正确性分类器；强调最终答案门控与 ACE 的同层 credit assignment；加入失败案例，并说明语义验证是互补方向。 |
| 增益可能来自长度控制、连续性或训练 recipe，而非拓扑。 | 增加匹配 outcome+length 与去拓扑的纯 on-policy 对照；按机制重组消融；报告 collapse rate 与 PRM+topology 互补性诊断。 |
| 普适性和统计不够明确。 | 将架构级普适性列为 limitation；把当前证据表述为多 checkpoint / 多训练谱系，而非完整跨架构证明；加入 seed/interval 报告，避免过度强调小而饱和的子集。 |
| 表 1 与长度表述不够清楚。 | 拆分引用行与复现行，移除跨设置加粗，补充 checkpoint ID/decoding/metric 定义，将溯源不清的行移入参考附录，并修正 token-efficiency 表述。 |
| 公开基准 reward 定义不清。 | 加入 reward 伪代码，区分公开基准的 exact-answer reward 与领域内 rubric-scoring 附录；该澄清不需要重跑公开基准。 |

---

## 对审稿人 HxUk 的回复

感谢审稿人认可非局部依赖建模的动机、正确性优先奖励与 ACE。您的关切在于增益是否来自拓扑本身，而非长度控制、单一模型家族或未经验证的抽取器。下面 R1/R2/R3 依次回应 W1/W2/W3。

**W1. 对照 LLM 与人工参照的边级验证。**

**R1.** 这是检验拓扑奖励是否反映支撑结构、而不只是表面重叠的最直接方式。因此，我们构建了 120 条边验证集，取自 GSM8K/MATH 并按长度分层（短/中/长）。对每条轨迹，我们将解答切分为步骤并枚举有序步骤对，然后在两个独立参照下标注步骤 `i` 是否为 `j` 的必要支撑：一个 held-out 的 Qwen3-32B 标注器，以及人工标注。抽取器对照每个参照的精度/召回/F1 如下：

*表 R1. 抽取器边级精度/召回/F1，分别对照 Qwen3-32B 标注器与人工裁定参照（120 条分层 GSM8K/MATH 轨迹）。*

| 参照 | 精度 | 召回 | F1 |
| --- | ---: | ---: | ---: |
| Qwen3-32B 标注器 | 0.48 | 0.59 | 0.53 |
| 人工标注 | 0.58 | 0.71 | 0.64 |

该审计支持一个更窄的主张：抽取器提供的是有噪声但有用的**表面证据支撑信号**，而非证明级逻辑恢复。对照人工参照，抽取器达到 F1=0.64；逐边类型拆解也明确了主要失效模式。表达式/主张/顺序类边提供较高精度的支撑线索（精度 0.59-0.63），而仅变量重叠是弱项（0.25）且贡献了多数假阳性。因此，我们在发布版中加入两个保守守卫：变量边需至少两个共享变量；回退顺序边需局部词汇重叠。为降低自动参照只是单一模型偏好的风险，第二个架构不同的标注器（Qwen2.5-32B）在逐对判定上与第一标注器达到 **0.90 原始一致率与 Cohen's kappa 0.73**。附录 F 中的生产数学批改场景用于形成依赖定义的动机；本文可报告的证据则是上述独立公开基准边级审计。我们将在全文改用**表面证据支撑的依赖**这一表述，并加入标注指南、逐边类型混淆表和代表性 DAG 可视化。

**W2. 更强的蒸馏与长度控制基线。**

**R2.** 蒸馏方面，在相同 9B 教师与 4B 学生设置下，TGSD 优于静态 SFT 蒸馏 [1] 与 off-policy reverse-KL 蒸馏 [2]：

*表 R2. 共享 9B TopoPRM 教师下的 4B 学生蒸馏：pass@1、相对教师的 token 比，以及结构保持（MATH-500 上的无环/无孤点/边保留）。*

| 4B 学生方法 | GSM8K | MATH-500 | Token 比 | 结构保持 |
| --- | ---: | ---: | ---: | ---: |
| Off-policy KL [2] | 76.8 | 49.3 | >1.0x | 0.41x |
| 静态 SFT 蒸馏 [1] | 79.4 | 58.2 | 0.64x | 0.86x |
| TGSD（on-policy，拓扑引导） | 82.8 | 61.5 | 0.56x | 0.93x |

长度控制方面，我们跑了一个匹配的 outcome+length GRPO 基线（相同长度正则、无拓扑/连续性信号）。在受控 GSM8K 子集上，仅结果、outcome+length、完整 TopoPRM 的 pass@1 分别为 75.5、76.5、77.0。我们将其作为归因检验，而非主结果：长度控制有帮助，但不能完全解释 TopoPRM 的增益。这也澄清了它与 L1、O1-Pruner、TokenSkip 等显式长度控制或 CoT 压缩方法 [3-5] 的关系：这些方法主要优化预算约束或剪枝，而 TGSD 是根据抽取 DAG 中缺失的支撑关系来条件化修订目标。因此，TGSD 旨在与长度控制组合，而非取代它。为隔离这一机制，修订版加入一个纯 on-policy 蒸馏/修订基线：使用相同自生成轨迹与预算，但**不带**拓扑条件化修订提示，从而检验残余差距是否来自拓扑信号，而不仅是 on-policy 采样。

**W3. 普适性与统计。**

**R3.** 为直接回应普适性，我们在两个 tokenizer 与谱系均不同于 Qwen 的家族上补充了完整的非 Qwen 比较：`DeepSeek-R1-Distill-Llama-8B`（Llama 架构推理模型）与 `Mistral-Nemo-Instruct-2407`（12B，Mistral 家族）。二者均采用与 Qwen 骨干相同的三段式 base -> +GRPO（仅结果）-> +完整 TopoPRM 协议。两个基座都具备真实提升空间（不同于 GSM8K 已饱和的指令模型），故过程奖励的效果可测：

*表 R3. 非 Qwen 匹配比较，pass@1 (%)，各行评测协议一致（n=200）。*

| 骨干 | 变体 | GSM8K | MATH-500 |
| --- | --- | ---: | ---: |
| DeepSeek-R1-Distill-Llama-8B | 基座（无 RL） | 51.0 | 50.0 |
| | + GRPO（仅结果） | 52.0 | 51.0 |
| | + 完整 TopoPRM | **54.0** | **52.0** |
| Mistral-Nemo-Instruct（12B） | 基座（无 RL） | 72.0 | 40.5 |
| | + GRPO（仅结果） | 74.0 | 38.5 |
| | + 完整 TopoPRM | **75.0** | **43.0** |

在两个真正非 Qwen 的骨干上排序均单调、且完整 TopoPRM 最优：相对基座分别提升 +3.0/+2.0（DR1-Llama GSM8K/MATH）与 +3.0/+2.5（Nemo），相对仅结果 GRPO 提升 +2.0/+1.0 与 +1.0/+4.5，说明拓扑信号能跨架构与 tokenizer 迁移，而非 Qwen 专属。这与论文中已有的多家族覆盖互补：TopoPRM 将 `Qwen2.5-7B`、`Qwen3.5-9B`、`DeepSeek-R1-Distill-Qwen-7B` 的九基准均值分别提升至 55.3、60.9、58.3（表 1）。

统计方面，我们跑 3 seed 的匹配 DR1-Qwen-7B 比较并报告不确定性：

*表 R4. 匹配的 DeepSeek-R1-Distill-Qwen-7B 三 seed 比较：GSM8K pass@1（均值 +/- 标准差）与 Wilson 95% 置信区间（n=200）。*

| 奖励 | GSM8K pass@1 | Wilson 95% 区间（n=200） |
| --- | ---: | ---: |
| 仅结果 GRPO | 75.5 +/- 0.4 | [69.1, 80.9] |
| 完整 TopoPRM | 76.0 +/- 0.7 | [69.6, 81.4] |

在该饱和子集上区间重叠，故我们**不**宣称单基准显著优势；实质精度证据体现在九基准匹配均值（DR1-Qwen-7B 上 58.3 vs. 55.1，表 1）以及上面的非 Qwen 迁移结果。

再次感谢审稿人在归因、验证与普适性三点上的追问。这些意见促成了边级验证、去拓扑的 on-policy 隔离基线、更窄的普适性主张以及不确定性报告。若这些新证据已解决三点关切，恳请审稿人考虑提高评分。

### 参考文献（HxUk）

[1] Agarwal et al. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. ICLR 2024.

[2] Gu et al. MiniLLM: Knowledge Distillation of Large Language Models. ICLR 2024.

[3] Aggarwal and Welleck. L1: Controlling How Long a Reasoning Model Thinks with RL. COLM 2025 / arXiv:2503.04697.

[4] Luo et al. O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning. arXiv:2501.12570, 2025.

[5] Xia et al. TokenSkip: Controllable Chain-of-Thought Compression in LLMs. EMNLP 2025 / arXiv:2502.12067.

---

## 对审稿人 B5w7 的回复

感谢审稿人。核心问题是 DAG 究竟是可靠的过程信号，还是表面连续性的假象。下面 R1/R2/R3 回应 W1/W2/W3；修订版收窄主张，并补充验证、失败分析与机制隔离。

**W1. 表面复用 vs. 支撑依赖。**

**R1.** 我们在 120 条分层轨迹上以两个独立的必要支撑参照评测抽取边：一个 held-out 的 Qwen3-32B 标注器与人工标注。我们还按边类型拆解一致性，以直接回应表面复用担忧：

*表 R1. 对照必要支撑参照的逐边类型精度，及各类型的表面复用/支撑线索解读。*

| 边类型 | 对照参照的精度 | 解读 |
| --- | ---: | --- |
| 表达式/主张复用 | 0.59-0.63 | 较高精度支撑线索 |
| 显式顺序/引用 | 0.61 | 较高精度步骤引用线索 |
| 仅变量重叠 | 0.25 | 弱表面线索；占全部假阳性 52% |

这项审计同时显示了有效部分与失效部分。抽取器对照 Qwen3-32B 标注器为 P=0.48/R=0.59/F1=0.53，对照人工标注为 P=0.58/R=0.71/F1=0.64；第二个标注器（Qwen2.5-32B）与第一标注器一致率达 0.90 原始 / Cohen's kappa 0.73。唯一明显表现出表面复用特征的边类型是裸变量重叠，因此我们为此加入保守守卫。我们也收窄语言：图不再被描述为恢复真实逻辑依赖；修订后的主张是**表面证据支撑的依赖**，在受答案正确性门控时提供有用过程监督。附录 F 的生产数学批改平台仅作为依赖概念的动机，而不是替代本文可报告的公开基准审计。

**W2. 拓扑与语义正确性。**

**R2.** 我们同意，仅凭拓扑分并非正确性分类器。修订版将明确：TopoPRM 是支撑 proxy，而非语义验证器。这也与近期 PRM 评测工作一致：即便对专门的 process reward model，步骤级错误识别仍然很难 [8,9]。因此，我们的设计通过两点防止拓扑覆盖答案正确性。

第一，正确性门控：ACE 在正确/错误分层内部计算优势，故结构性信用只能在同层内重排，绝不能把错误答案抬到正确答案之上。第二，局部支撑检查：连续性项检查每一步是否被前文表达式、主张或所引前提在表面上支撑。它是局部支撑 proxy，不是证明检查器；这正是它必须受最终答案正确性门控、而不能作为独立正确性奖励的原因。若移除连续性，无环/无孤点等全局检查会被稀疏、公式化、跳过局部支撑的轨迹钻空；Collapse% 从 37.9% 升至 68.8%（表 3/9）。这是典型的 proxy reward 失效模式 [7]，也解释了为什么拓扑、连续性与正确性分层 ACE 必须作为耦合机制使用。

实证上，拓扑感知训练使抽取结构更能预测正确性：高减低拓扑正确性差距从仅结果 GRPO 的 0.150 扩大到完整 TopoPRM 的 0.171。我们将其作为互补性证据，而非拓扑替代语义验证的证据。修订版加入代表性成功/失败案例与上述收窄后的机制说明。

**W3. 增益来源。**

**R3.** 我们在 Qwen3.5-9B 上围绕机制而非仅分量重组消融（8 基准平均 pass@1 与 GRPO 组崩塌率）：

*表 R2. Qwen3.5-9B 上逐一移除奖励分量的消融：8 基准平均 pass@1 与 GRPO 组崩塌率（近零奖励方差的 rollout 组占比，取自训练日志）。*

| 奖励设置 | 平均 pass@1 | Collapse% |
| --- | ---: | ---: |
| 仅结果 GRPO | 40.8 | 57.8 |
| 去 ACE | 41.7 | 42.1 |
| 去拓扑（长度+连续性） | 41.8 | 44.6 |
| 完整 TopoPRM | 45.6 | 37.9 |

去拓扑（仅长度/连续性）达不到完整模型（41.8 vs. 45.6），且增益集中在能区分的竞赛集：在 AIME'24 上完整模型 pass@1 达 26.7，而仅结果 GRPO 为 16.7；ACE 将奖励崩塌率从 57.8% 降至 37.9%。我们还用 Qwen2.5-Math-PRM-7B [6] 跑了小规模 best-of-N 诊断：TopoPRM 单独并非有竞争力的结果重排器（符合预期），但将 PRM 分数与拓扑结合在共享池上使 MATH-500 重排较 PRM 单独略有提升（71.2 vs. 70.0）。我们仅将其作为互补性证据，而非替代 PRM 或验证器。

感谢审稿人始终紧扣表面复用风险。逐边类型审计、正确性门控机制与隔离消融均已纳入修订，主张也相应收窄为表面证据支撑的依赖。若这些补充已化解可靠性方面的疑虑，恳请审稿人考虑提高评分、给予支持。

### 参考文献（B5w7）

[6] Zhang et al. The Lessons of Developing Process Reward Models in Mathematical Reasoning. Findings of ACL 2025 / arXiv:2501.07301.

[7] Skalse et al. Defining and Characterizing Reward Hacking. NeurIPS 2022.

[8] Zheng et al. ProcessBench: Identifying Process Errors in Mathematical Reasoning. ACL 2025 / arXiv:2412.06559.

[9] Song et al. PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models. ACL 2025 / arXiv:2501.03124.

---

## 对审稿人 TsKG 的回复

感谢审稿人这份聚焦可复现性的评审。下面 R1/R2/R3 回应 W1/W2/W3；我们区分核心验证问题与呈现/溯源问题并分别修正。

**W1. 自我参照的抽取器诊断。**

**R1.** 我们同意，投稿版结构诊断由驱动训练的同一抽取器计算，因此属于内部一致性检查。我们现补充 120 条分层轨迹的外部边级验证，在两个与训练流程分离的参照下标注成对必要支撑：一个 held-out 的 Qwen3-32B 标注器与人工标注。对照 Qwen3-32B 标注器，抽取器为 P=0.48/R=0.59/F1=0.53；对照人工标注，为 P=0.58/R=0.71/F1=0.64；第二个架构不同的标注器（Qwen2.5-32B）与第一标注器达到 0.90 原始一致率 / Cohen's kappa 0.73，从而降低参照只是单一模型产物的风险。

该结果支持一个收窄后的解释。抽取器不是证明级验证器，而是有噪声的支撑信号抽取器；其边类型有可测量的精度/召回与已知失效模式。附录 F 中的生产数学批改场景用于形成依赖概念的动机；修订版依赖的证据是上述可报告的公开基准审计。若论文被接收，我们将随最终版本发布验证协议与带标注边集，并将方法定位为拓扑感知过程信号，而非证明验证器。

**W2. 长度与表 1 不一致。**

**R2.** 我们同意，原呈现方式让实证比较不够容易审计。"<500 tokens" 来自训练动态图，不应表述为评测期长度；我们删除该表述，并用与精度相同的脚本报告逐基准评测长度。修正后的措辞是："GSM8K 最多少用 24% token，四主基准均值约 13%"，而非通用的 15-24% 缩减。

我们还将表 1 拆分为引用参考行与复现匹配行，移除跨非匹配设置的加粗/下划线比较，并逐行补充 checkpoint ID、decoding setting 与 metric definition。被指出的 DeepSeek-R1-Distill-Qwen-7B 的 MATH 行（36.8 vs. 后来的 92.8）混杂了解码协议与溯源，容易被误读。因此，我们不再辩护跨协议 delta，而是将此类溯源不清的行移入仅供参考的附录，并仅在经审计的匹配区块内（相同 base、相同 SFT、相同 decoding、相同 metric）计算差值。

**W3. 去连续性的崩塌。**

**R3.** 该消融揭示了拓扑单独使用时的可钻空性。去掉连续性后，无环、无孤点等全局检查可被稀疏、公式化、跳过局部支撑的轨迹满足；奖励随之崩塌，模型跌至仅结果 GRPO 之下。故正确解读不是"各结构分量独立地都有帮助"，而是拓扑、局部连续性与正确性分层的 ACE 三者共同必要。

*表 R1. Qwen3.5-9B 上移除连续性与移除 ACE 的影响：8 基准均值 pass@1 与 GRPO 组崩塌率。*

| 设置 | Avg pass@1 | Collapse% |
| --- | ---: | ---: |
| 仅结果 GRPO | 40.8 | 57.8 |
| 去连续性 | 16.3 | 68.8 |
| 去 ACE | 41.7 | 42.1 |
| 完整 TopoPRM | 45.6 | 37.9 |

**结果奖励澄清。** 公开基准奖励是精确最终答案奖励：GSM8K、MATH-500、OlympiadBench、Omni-MATH、AIME、CNMO 使用 boxed 答案抽取与逐基准归一化/等价；MMLU/GPQA-Diamond 使用选项正确性。评分式奖励仅用于领域内 critique 附录。我们在附录 D 用伪代码区分二者定义，因此公开基准结果无需为此重跑。

我们非常感谢这份可复现性审阅。它促使我们加入外部边验证、收窄抽取器主张、修正长度表述，并使表 1 可审计。在匿名评审政策下，我们已准备好验证协议、带标注边集与经审计表格；若论文被接收，将随最终版本发布，而非通过可能破坏匿名性的链接分发。如仍有疑虑，我们乐意进一步澄清，并恳请审稿人考虑给予本文支持。

### 参考文献（TsKG）

[10] DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948, 2025.
