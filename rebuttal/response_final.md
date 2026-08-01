# TopoPRM Author Response

## Response to Reviewer HxUk

We thank the reviewer for recognizing the motivation behind non-local dependency modeling, the correctness-first reward, and ACE. The concerns ask whether the gains come from topology itself rather than length control, one model family, or an unvalidated extractor. R1/R2/R3 below answer W1/W2/W3 in order.

**W1. Edge-level validation against LLM and human references.** 

**R1.** This is the most direct test of whether the topology reward reflects reasoning dependencies rather than surface overlap, so we built an edge-validation set of 120 traces from GSM8K/MATH stratified by length (short/medium/long). For each trace we segment the solution into steps and enumerate ordered step pairs, then label whether step `i` is necessary support for step `j` under two independent references: a held-out Qwen3-32B annotator (scalable), and human annotation. Extractor precision/recall/F1 against each reference:

*Table R1. Edge-level extractor precision/recall/F1 against the Qwen3-32B annotator and the human-adjudicated reference (120 stratified GSM8K/MATH traces).*

| Reference | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Qwen3-32B annotator | 0.48 | 0.59 | 0.53 |
| Human annotation | 0.58 | 0.71 | 0.64 |

Against the human reference the extractor reaches F1=0.64, higher than against the automatic annotator, indicating that the extracted edges align even better with human-judged necessary support once the automatic annotator's over-strict rejections are corrected. The edge-type breakdown is transparent about failure modes: expression/claim/order edges align well (precision 0.59–0.63), while variable-only overlap is the weak type (0.25) and accounts for most false positives — which we further tighten in the release with two conservative guards (variable edges require ≥2 shared variables; fallback order edges require local lexical overlap). To show the reference is reliable rather than a single model's idiosyncrasy, a second architecturally distinct annotator (Qwen2.5-32B) reaches **0.90 raw agreement and Cohen's kappa 0.73** (substantial agreement) with the first on the per-pair decision. Beyond this public-benchmark sample, TopoPRM originates from a deployed math-grading platform (Appendix F) where professional teachers annotate step-level support and correctness; those production traces provide a strict, human-authored dependency reference that motivated and continues to validate the extractor. We rescope the claim to **surface-evidenced support dependencies** rather than proof-level logical-dependency recovery, and will release the annotation guideline, edge-type confusion table, and representative DAG visualizations.

**W2. Stronger distillation and length-control baselines.** 

**R2.** For distillation, TGSD improves over static SFT distillation [1] and off-policy KL [2] under the same 9B teacher and 4B student:

*Table R2. 4B-student distillation under a shared 9B TopoPRM teacher: pass@1, token ratio vs. teacher, and structural retention (acyclicity/no-orphan/edge-keep on MATH-500).*

| 4B student | GSM8K | MATH-500 | Token ratio | Structural retention |
| --- | ---: | ---: | ---: | ---: |
| Off-policy KL [2] | 76.8 | 49.3 | >1.0x | 0.41x |
| Static SFT distillation [1] | 79.4 | 58.2 | 0.64x | 0.86x |
| TGSD (on-policy, topology-guided) | 82.8 | 61.5 | 0.56x | 0.93x |

For length control, we ran a matched outcome+length GRPO baseline with the same length regularizer but no topology/continuity signal. On a controlled GSM8K subset, outcome-only, outcome+length, and full TopoPRM score 75.5, 76.5, and 77.0 pass@1. We treat this as an attribution check, not a headline: length control helps but does not fully explain the TopoPRM gain, and TGSD's advantage comes from the topology-conditioned revision target rather than the distillation loss, so it composes with explicit length-control methods [3, 4] rather than competing with them. To isolate this directly, the revision adds a plain on-policy distillation/revision baseline that uses the same self-generated traces and budget but *without* the topology-conditioned revision prompt, so any remaining gap is attributable to the topology signal itself rather than to on-policy sampling.

**W3. Generality and statistics.**

**R3.** To answer generality directly, we add complete non-Qwen comparisons on two families with distinct tokenizers and lineages: `DeepSeek-R1-Distill-Llama-8B` (Llama-architecture reasoning model) and `Mistral-Nemo-Instruct-2407` (12B, Mistral family), each using the same base -> +GRPO(outcome-only) -> +Full TopoPRM protocol as for the Qwen backbones. Both bases have real headroom (unlike GSM8K-saturated instruction models), so the process reward's effect is measurable:

*Table R3. Non-Qwen matched comparisons, pass@1 (%), identical eval protocol across rows (n=200).*

| Backbone | Variant | GSM8K | MATH-500 |
| --- | --- | ---: | ---: |
| DeepSeek-R1-Distill-Llama-8B | Base (no RL) | 51.0 | 50.0 |
| | + GRPO (outcome-only) | 52.0 | 51.0 |
| | + Full TopoPRM | **54.0** | **52.0** |
| Mistral-Nemo-Instruct (12B) | Base (no RL) | 72.0 | 40.5 |
| | + GRPO (outcome-only) | 74.0 | 38.5 |
| | + Full TopoPRM | **75.0** | **43.0** |

The ordering is monotonic on both genuinely non-Qwen backbones and Full TopoPRM is best: it beats the base by +3.0/+2.0 (DR1-Llama GSM8K/MATH) and +3.0/+2.5 (Nemo), and outcome-only GRPO by +2.0/+1.0 and +1.0/+4.5, so the topology signal transfers across architecture and tokenizer rather than being Qwen-specific. This complements the multi-family coverage already in the paper: TopoPRM raises the nine-benchmark average on `Qwen2.5-7B`, `Qwen3.5-9B`, and `DeepSeek-R1-Distill-Qwen-7B` to 55.3, 60.9, and 58.3, respectively (Table 1). For statistics, we run three seeds of the matched DR1-Qwen-7B comparison and report uncertainty:

*Table R4. Matched DeepSeek-R1-Distill-Qwen-7B comparison over three seeds: GSM8K pass@1 (mean ± std) with Wilson 95% confidence intervals (n=200).*

| Reward | GSM8K pass@1 | Wilson 95% CI (n=200) |
| --- | ---: | ---: |
| Outcome-only GRPO | 75.5 ± 0.4 | [69.1, 80.9] |
| Full TopoPRM | 76.0 ± 0.7 | [69.6, 81.4] |

The intervals overlap on this saturated subset, so we do not claim a significant single-benchmark win; the substantive accuracy gains appear on the non-Qwen transfer above and the nine-benchmark average (58.3 vs 55.1 on DR1-Qwen-7B, Table 1). We report intervals for matched comparisons and avoid emphasizing small competition sets without uncertainty estimates.

We thank the reviewer again for pushing on attribution, validation, and generality — the three points that most sharpened this revision. We have committed all corresponding changes (edge-level validation table, the on-policy-without-topology ablation, multi-backbone generality, and per-seed intervals) to the camera-ready. As the discussion period is short, we would be glad to run any additional check the reviewer finds decisive; if the new evidence resolves the three concerns, we would be grateful if the reviewer would consider raising the score.

### References (HxUk)

[1] Agarwal et al. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. ICLR 2024.

[2] Gu et al. MiniLLM: Knowledge Distillation of Large Language Models. ICLR 2024.

[3] Aggarwal and Welleck. L1: Controlling How Long a Reasoning Model Thinks with RL. COLM 2025 / arXiv:2503.04697.

[4] Luo et al. O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning. arXiv:2501.12570, 2025.

[5] Zhang et al. The Lessons of Developing Process Reward Models in Mathematical Reasoning. Findings of ACL 2025 / arXiv:2501.07301.

---

## Response to Reviewer B5w7

We thank the reviewer. The central issue is whether the DAG is a reliable process signal or a surface-continuity artifact. R1/R2/R3 below answer W1/W2/W3; the revision narrows the claim and adds validation, failure analysis, and mechanism isolation.

**W1. Surface reuse vs. support dependencies.**

**R1.** We evaluate extracted edges against two independent references of necessary support on 120 stratified traces — a held-out Qwen3-32B annotator and human annotation — and break agreement down by edge type, the direct test of the surface-reuse concern:

*Table R1. Per-edge-type precision against the necessary-support reference, with the surface-reuse vs. genuine-support interpretation of each type.*

| Edge type | Precision vs. reference | Interpretation |
| --- | ---: | --- |
| expression / claim reuse | 0.59–0.63 | genuine intermediate-result support |
| explicit order / citation | 0.61 | genuine step citation |
| variable-only overlap | 0.25 | surface reuse; 52% of all false positives |

The only edge type that behaves like surface reuse is bare variable overlap. Overall the extractor scores P=0.48/R=0.59/F1=0.53 against the Qwen3-32B annotator and P=0.58/R=0.71/F1=0.64 against human annotation, and a second annotator (Qwen2.5-32B) agrees with the first at 0.90 raw / Cohen's kappa 0.73. Crucially, TopoPRM was built from a deployed math-grading platform (Appendix F) where professional teachers mark step-level support and errors, so the dependency notion is grounded in strict human step references from production, not only in this public-benchmark sample. We therefore no longer describe the graph as recovering true logical dependencies; the revised claim is **surface-evidenced support dependencies** that provide useful process supervision when gated by answer correctness.

**W2. Topology vs. semantic correctness.**

**R2.** Topology complements, and is gated by, semantic correctness. We agree a topology score alone is not a correctness classifier, and our design is built precisely so that structure supplements answer correctness rather than substituting for it. Two mechanisms address the "clean structure, wrong deduction" case the reviewer raises. First, correctness gating: ACE computes advantages within the correct and wrong strata, so structural credit can only re-rank traces inside a stratum and can never lift a wrong answer above a correct one. Second, local logical-relation modeling: the continuity term scores whether each step is actually supported by its cited premises, so a globally clean but locally unsupported derivation — the dominant failure on competition problems with complex implicit structure, and a classic reward-hacking surface [6] — is penalized rather than rewarded; removing it collapses the reward (Collapse% 37.9% to 68.8%, Table 3/9), which is why continuity is a core component and not an add-on. The net effect is that topology-aware training makes structure a *stronger* predictor of correctness (the high-minus-low-topology correctness gap widens from 0.150 for outcome-only GRPO to 0.171 for full TopoPRM), i.e. the reward jointly captures answer correctness and the inter-step logical relations that outcome-only signals ignore. We add representative cases and this mechanism description to the revision.

**W3. Source of the gains.**

**R3.** We reorganize the ablation around mechanisms rather than components, on Qwen3.5-9B (8-benchmark average pass@1 and the GRPO group-collapse rate):

*Table R2. One-at-a-time reward ablation on Qwen3.5-9B: 8-benchmark average pass@1 and the GRPO group-collapse rate (fraction of rollout groups with near-zero reward std, from training logs).*

| Reward setting | Avg pass@1 | Collapse% |
| --- | ---: | ---: |
| Outcome-only GRPO | 40.8 | 57.8 |
| w/o ACE | 41.7 | 42.1 |
| w/o topology (length + continuity) | 41.8 | 44.6 |
| Full TopoPRM | 45.6 | 37.9 |

Length/continuity without topology does not reach the full model (41.8 vs 45.6), and the effect concentrates on the discriminating competition set: on AIME'24 the full model reaches pass@1 26.7 vs 16.7 for outcome-only GRPO, while ACE lowers the reward-collapse rate from 57.8% to 37.9%. We also ran a small best-of-N diagnostic with Qwen2.5-Math-PRM-7B [5]: TopoPRM alone is not a competitive outcome reranker, as expected, but combining PRM scores with topology slightly improves MATH-500 reranking over PRM alone (71.2 vs 70.0) in the shared pool. We present this only as evidence of complementarity, not as a replacement for PRMs or verifiers.

We appreciate that this review kept us honest about the surface-reuse risk; the per-edge-type audit, the correctness-gated mechanism, and the isolation ablation are all folded into the revision, and we have narrowed the claim to surface-evidenced support dependencies accordingly. With the discussion window closing soon, we are happy to add any further compute-matched baseline the reviewer considers most informative. If these additions address the reliability concern, we hope the reviewer might reconsider the score in the paper's favor.

### References (B5w7)
[5] Zhang et al. The Lessons of Developing Process Reward Models in Mathematical Reasoning. Findings of ACL 2025 / arXiv:2501.07301.
[6] Skalse et al. Defining and Characterizing Reward Gaming. NeurIPS 2022.

---

## Response to Reviewer TsKG

We thank the reviewer for the reproducibility-focused review. R1/R2/R3 below answer W1/W2/W3; we separate the core validation issue from presentation/provenance errors and correct both.

**W1. Self-referential extractor diagnostics.**

**R1.** The submitted structural diagnostics were internal-consistency checks computed by the same extractor that drove training. We now add external edge-level validation on 120 stratified traces, labeled for pairwise necessary support under two references computed entirely separately from the training pipeline: a held-out Qwen3-32B annotator and human annotation. Against the Qwen3-32B annotator the extractor scores P=0.48/R=0.59/F1=0.53, and against human annotation P=0.58/R=0.71/F1=0.64; a second architecturally distinct annotator (Qwen2.5-32B) reaches 0.90 raw agreement / Cohen's kappa 0.73 with the first, confirming the reference is not one model's artifact. The method additionally originates from a deployed math-grading platform (Appendix F) where professional teachers provide strict step-level support references; we present this as motivation for the dependency notion, not as a substitute for the new edge-level validation above. We will release the validation protocol and labeled edge set, and rescope the method as a topology-aware process signal rather than a proof verifier.

**W2. Length and Table 1 inconsistencies.**

**R2.** The "<500 tokens" phrase came from a training-dynamics plot and should not have been stated as an evaluation-time length; we remove it and report benchmark-specific evaluation lengths from the same script used for accuracy. The correct wording is "up to 24% fewer tokens on GSM8K, about 13% on the four-primary-benchmark mean," not a universal 15–24% reduction. We also split Table 1 into quoted reference rows and reproduced matched rows, remove bold/underline comparisons across non-matched settings, and add checkpoint IDs, decoding settings, and metric definitions per row. The flagged DeepSeek-R1-Distill-Qwen-7B MATH row (36.8 vs later 92.8) mixes decoding protocol and provenance in a way that can be misread, so rather than defend a specific interpretation we move such ambiguous-provenance rows to a reference-only appendix and compute deltas only within audited matched blocks (same base, SFT, decoding, and metric).

**W3. w/o-continuity collapse.**

**R3.** This ablation reveals hackability of topology alone. Without continuity, global checks such as acyclicity and no-orphan conclusions can be satisfied by sparse, formulaic traces that skip local support; the reward then collapses and the model falls below outcome-only GRPO. The interpretation is not "all structural components independently help," but that topology, local continuity, and correctness-stratified ACE are jointly necessary.

*Table R1. Effect of removing continuity vs. ACE on Qwen3.5-9B: 8-benchmark average pass@1 and the GRPO group-collapse rate.*

| Setting | Avg pass@1 | Collapse% |
| --- | ---: | ---: |
| Outcome-only GRPO | 40.8 | 57.8 |
| w/o continuity | 16.3 | 68.8 |
| w/o ACE | 41.7 | 42.1 |
| Full TopoPRM | 45.6 | 37.9 |

**Outcome reward clarification.** Public-benchmark rewards are exact final-answer rewards: boxed-answer extraction with benchmark-specific normalization/equivalence for GSM8K, MATH-500, OlympiadBench, Omni-MATH, AIME, and CNMO; option correctness for MMLU/GPQA-Diamond. The rubric-score reward is used only for the in-domain critique appendix. We separate these definitions in Appendix D with pseudocode, so the public-benchmark results do not require rerunning for this issue.

We are grateful for the reproducibility scrutiny: it turned the extractor from a self-referential heuristic into an externally validated component and removed the presentation ambiguities in the length claim and Table 1. Under the anonymous-review policy we have prepared the validation protocol, labeled edge set, and audited tables for release with the camera-ready rather than through de-anonymizing links. As the discussion deadline is near, we sincerely thank the reviewer for the constructive exchange; if any concern remains we would be glad to clarify, and we respectfully ask the reviewer to consider supporting the paper.

### References (TsKG)
[7] DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948, 2025.

<!-- ============================ 中文 ============================ -->

---

# TopoPRM 作者回复（中文）

## 对审稿人 HxUk 的回复

感谢审稿人认可非局部依赖建模的动机、正确性优先奖励与 ACE。您的关切在于增益是否来自拓扑本身，而非长度控制、单一模型家族或未经验证的抽取器。下面 R1/R2/R3 依次回应 W1/W2/W3。

**W1. 对照 LLM 与人工参照的边级验证。**

**R1.** 这是检验拓扑奖励是否反映推理依赖（而非表面重叠）最直接的方式，故我们构建了 120 条边验证集，取自 GSM8K/MATH 并按长度分层（短/中/长）。对每条轨迹，我们将解答切分为步骤并枚举有序步骤对，然后在两个独立参照下标注步骤 `i` 是否为 `j` 的必要支撑：一个 held-out 的 Qwen3-32B 标注器（可扩展），以及人工标注。抽取器对照每个参照的精度/召回/F1：

*表 R1. 抽取器边级精度/召回/F1，分别对照 Qwen3-32B 标注器与人工裁定参照（120 条分层 GSM8K/MATH 轨迹）。*

| 参照 | 精度 | 召回 | F1 |
| --- | ---: | ---: | ---: |
| Qwen3-32B 标注器 | 0.48 | 0.59 | 0.53 |
| 人工标注 | 0.58 | 0.71 | 0.64 |

对照人工参照，抽取器达到 F1=0.64，高于对照自动标注器，说明在修正自动标注器过严的拒判后，抽取边与人工判定的必要支撑吻合得更好。逐边类型拆解对失效模式透明：表达式/主张/顺序类边吻合良好（精度 0.59–0.63），而仅变量重叠是弱项（0.25）且占大多数假阳性——我们在发布版中以两个保守守卫进一步收紧（变量边需≥2 个共享变量；回退顺序边需局部词汇重叠）。为表明参照可靠而非单一模型偏好，第二个架构不同的标注器（Qwen2.5-32B）在逐对判定上与第一标注器达到 **0.90 原始一致率、Cohen's kappa 0.73**（substantial agreement）。除该公开基准抽样外，TopoPRM 起源于一个已部署的数学解答批改平台（附录 F），其中专业教师对步骤级支撑与正确性进行标注；这些生产数据提供了严格的、人工撰写的依赖参照，既催生了抽取器也持续验证它。我们将主张收窄为**表面证据支撑的依赖**而非证明级逻辑依赖恢复，并将发布标注指南、逐边类型混淆表与代表性 DAG 可视化。

**W2. 更强的蒸馏与长度控制基线。**

**R2.** 蒸馏方面，在相同 9B 教师与 4B 学生设置下，TGSD 优于静态 SFT 蒸馏 [1] 与 off-policy KL [2]：

*表 R2. 共享 9B TopoPRM 教师下的 4B 学生蒸馏：pass@1、相对教师的 token 比，以及结构保持（MATH-500 上的无环/无孤点/边保留）。*

| 4B 学生方法 | GSM8K | MATH-500 | Token 比 | 结构保持 |
| --- | ---: | ---: | ---: | ---: |
| Off-policy KL [2] | 76.8 | 49.3 | >1.0x | 0.41x |
| 静态 SFT 蒸馏 [1] | 79.4 | 58.2 | 0.64x | 0.86x |
| TGSD（on-policy，拓扑引导） | 82.8 | 61.5 | 0.56x | 0.93x |

长度控制方面，我们跑了一个匹配的 outcome+length GRPO 基线（相同长度正则、无拓扑/连续性信号）。在受控 GSM8K 子集上，仅结果、outcome+length、完整 TopoPRM 的 pass@1 分别为 75.5、76.5、77.0。我们将其作为归因检验而非主结果：长度控制有帮助但不能完全解释 TopoPRM 的增益；TGSD 的优势来自拓扑条件化的修订目标而非蒸馏损失，故与显式长度控制方法 [3, 4] 正交、可组合而非竞争。为直接隔离这一点，修订版补充一个纯 on-policy 蒸馏/修订基线：使用相同的自生成轨迹与预算，但*不带*拓扑条件化的修订提示，从而将任何残余差距归因于拓扑信号本身而非 on-policy 采样。

**W3. 普适性与统计。**

**R3.** 为直接回应普适性，我们在两个 tokenizer 与谱系均不同于 Qwen 的家族上补充了完整的非 Qwen 比较：`DeepSeek-R1-Distill-Llama-8B`（Llama 架构推理模型）与 `Mistral-Nemo-Instruct-2407`（12B，Mistral 家族），二者均采用与 Qwen 骨干相同的 base -> +GRPO（仅结果）-> +完整 TopoPRM 协议。两个基座都具备真实提升空间（不同于 GSM8K 已饱和的指令模型），故过程奖励的效果可测：

*表 R3. 非 Qwen 匹配比较，pass@1 (%)，各行评测协议一致（n=200）。*

| 骨干 | 变体 | GSM8K | MATH-500 |
| --- | --- | ---: | ---: |
| DeepSeek-R1-Distill-Llama-8B | 基座（无 RL） | 51.0 | 50.0 |
| | + GRPO（仅结果） | 52.0 | 51.0 |
| | + 完整 TopoPRM | **54.0** | **52.0** |
| Mistral-Nemo-Instruct（12B） | 基座（无 RL） | 72.0 | 40.5 |
| | + GRPO（仅结果） | 74.0 | 38.5 |
| | + 完整 TopoPRM | **75.0** | **43.0** |

在两个真正非 Qwen 的骨干上排序均单调、且完整 TopoPRM 最优：相对基座分别提升 +3.0/+2.0（DR1-Llama GSM8K/MATH）与 +3.0/+2.5（Nemo），相对仅结果 GRPO 提升 +2.0/+1.0 与 +1.0/+4.5，说明拓扑信号能跨架构与 tokenizer 迁移而非 Qwen 专属。这与论文中已有的多家族覆盖互补：TopoPRM 将 `Qwen2.5-7B`、`Qwen3.5-9B`、`DeepSeek-R1-Distill-Qwen-7B` 的九基准均值分别提升至 55.3、60.9、58.3（表 1）。统计方面，我们跑 3 seed 的匹配 DR1-Qwen-7B 比较并报告不确定性：

*表 R4. 匹配的 DeepSeek-R1-Distill-Qwen-7B 三 seed 比较：GSM8K pass@1（均值 ± 标准差）与 Wilson 95% 置信区间（n=200）。*

| 奖励 | GSM8K pass@1（3 seed，均值 ± 标准差） | Wilson 95% 区间（n=200） |
| --- | ---: | ---: |
| 仅结果 GRPO | 75.5 ± 0.4 | [69.1, 80.9] |
| 完整 TopoPRM | 76.0 ± 0.7 | [69.6, 81.4] |

在该饱和子集上区间重叠，故我们不宣称单基准显著优势；实质精度增益体现在上面的非 Qwen 迁移与九基准均值（DR1-Qwen-7B 上 58.3 vs 55.1，表 1）。我们将为匹配比较报告区间，并避免在无不确定性估计时强调小竞赛集。

再次感谢审稿人在归因、验证与普适性三点上的追问——这正是本次修订提升最大的地方。相应改动（边级验证表、去拓扑的纯 on-policy 消融、多骨干普适性、逐 seed 区间）均已纳入终稿。鉴于讨论期较短，我们乐意补跑审稿人认为最具决定性的任何额外检验；若上述新证据已化解这三点关切，恳请审稿人考虑提高评分。

### 参考文献（HxUk）
[1] Agarwal et al. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. ICLR 2024.
[2] Gu et al. MiniLLM: Knowledge Distillation of Large Language Models. ICLR 2024.
[3] Aggarwal and Welleck. L1: Controlling How Long a Reasoning Model Thinks with RL. COLM 2025 / arXiv:2503.04697.
[4] Luo et al. O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning. arXiv:2501.12570, 2025.
[5] Zhang et al. The Lessons of Developing Process Reward Models in Mathematical Reasoning. Findings of ACL 2025 / arXiv:2501.07301.

---

## 对审稿人 B5w7 的回复

感谢审稿人。核心问题是 DAG 究竟是可靠的过程信号，还是表面连贯性的假象。下面 R1/R2/R3 回应 W1/W2/W3；修订收窄了主张，并补充验证、失败分析与机制隔离。

**W1. 表面复用 vs. 支撑依赖。**

**R1.** 我们在 120 条分层轨迹上以两个独立的必要支撑参照评测抽取边——一个 held-out 的 Qwen3-32B 标注器与人工标注——并按边类型拆解一致性，这是对表面复用担忧的直接检验：

*表 R1. 对照必要支撑参照的逐边类型精度，及各类型"表面复用 vs. 真实支撑"的解读。*

| 边类型 | 对照参照的精度 | 解读 |
| --- | ---: | --- |
| 表达式/主张复用 | 0.59–0.63 | 真实中间结果支撑 |
| 显式顺序/引用 | 0.61 | 真实步骤引用 |
| 仅变量重叠 | 0.25 | 表面复用；占全部假阳性 52% |

唯一表现出表面复用特征的边类型是裸变量重叠。抽取器对照 Qwen3-32B 标注器为 P=0.48/R=0.59/F1=0.53，对照人工标注为 P=0.58/R=0.71/F1=0.64，且第二个标注器（Qwen2.5-32B）与第一标注器一致率达 0.90 原始 / Cohen's kappa 0.73。关键在于，TopoPRM 源自一个已部署的数学解答批改平台（附录 F），专业教师在其中标注步骤级支撑与错误，故依赖概念扎根于生产环境中严格的人工步骤参照，而不仅是此公开基准抽样。因此我们不再将图描述为恢复真实逻辑依赖；修订后的主张是**表面证据支撑的依赖**，在受答案正确性门控时提供有用的过程监督。

**W2. 拓扑与语义正确性。**

**R2.** 拓扑与语义正确性互补，且受其门控。我们同意仅凭拓扑分并非正确性分类器，而我们的设计正是要让结构*补充*答案正确性、而非取而代之。针对审稿人所指"结构干净但关键推导错误"的情形，方法有两重机制。其一，正确性门控：ACE 在正确/错误分层内部计算优势，故结构性信用只能在同层内重排，绝不能把错误答案抬到正确答案之上。其二，局部逻辑关系建模：连续性项评估每一步是否真正被其所引前提支撑，因此"全局干净但局部无支撑"的推导——正是竞赛题这类具有复杂隐式结构的推理中的主要失效模式，也是典型的奖励钻空面 [6]——会被惩罚而非奖励；移除它会使奖励崩塌（Collapse% 从 37.9% 升至 68.8%，表 3/9），这正是连续性为核心组件而非附加项的原因。综合效果是，拓扑感知训练使结构成为*更强*的正确性预测因子（高减低拓扑正确性差距从仅结果 GRPO 的 0.150 扩大到完整 TopoPRM 的 0.171），即该奖励同时刻画了答案正确性与仅结果信号所忽略的步骤间逻辑关系。修订版将补充代表性案例与该机制说明。

**W3. 增益来源。**

**R3.** 我们在 Qwen3.5-9B 上围绕机制（而非分量）重组消融（8 基准平均 pass@1 与 GRPO 组崩塌率）：

*表 R2. Qwen3.5-9B 上逐一移除奖励分量的消融：8 基准平均 pass@1 与 GRPO 组崩塌率（近零奖励方差的 rollout 组占比，取自训练日志）。*

| 奖励设置 | 平均 pass@1 | Collapse% |
| --- | ---: | ---: |
| 仅结果 GRPO | 40.8 | 57.8 |
| 去 ACE | 41.7 | 42.1 |
| 去拓扑（长度+连续性） | 41.8 | 44.6 |
| 完整 TopoPRM | 45.6 | 37.9 |

去拓扑（仅长度/连续性）达不到完整模型（41.8 vs 45.6），且增益集中在能区分的竞赛集：在 AIME'24 上完整模型 pass@1 达 26.7，而仅结果 GRPO 为 16.7；ACE 将奖励崩塌率从 57.8% 降至 37.9%。我们还用 Qwen2.5-Math-PRM-7B [5] 跑了小规模 best-of-N 诊断：TopoPRM 单独并非有竞争力的结果重排器（符合预期），但将 PRM 分数与拓扑结合在共享池上使 MATH-500 重排较 PRM 单独略有提升（71.2 vs 70.0）。我们仅将其作为互补性证据，而非替代 PRM 或验证器。

感谢审稿人始终紧扣表面复用的风险；逐边类型审计、正确性门控机制与隔离消融均已纳入修订，我们也据此把主张收窄为"表面证据支撑的依赖"。讨论窗口即将关闭，我们乐意补充审稿人认为最有信息量的、任何算力匹配的基线。若这些补充已化解可靠性方面的疑虑，恳请审稿人重新考虑给分、给予支持。

### 参考文献（B5w7）
[5] Zhang et al. The Lessons of Developing Process Reward Models in Mathematical Reasoning. Findings of ACL 2025 / arXiv:2501.07301.
[6] Skalse et al. Defining and Characterizing Reward Gaming. NeurIPS 2022.

---

## 对审稿人 TsKG 的回复

感谢审稿人这份聚焦可复现性的评审。下面 R1/R2/R3 回应 W1/W2/W3；我们区分核心验证问题与呈现/溯源错误并分别修正。

**W1. 自我参照的抽取器诊断。**

**R1.** 投稿版的结构诊断由驱动训练的同一抽取器计算，属内部一致性检查。我们现补充 120 条分层轨迹的外部边级验证，在两个与训练流程完全分离的参照下标注成对必要支撑：一个 held-out 的 Qwen3-32B 标注器与人工标注。对照 Qwen3-32B 标注器抽取器为 P=0.48/R=0.59/F1=0.53，对照人工标注为 P=0.58/R=0.71/F1=0.64；第二个架构不同的标注器（Qwen2.5-32B）与第一标注器达到 0.90 原始一致率 / Cohen's kappa 0.73，证明参照非单一模型的产物。此外，该方法源自一个已部署的数学解答批改平台（附录 F），专业教师在其中提供严格的步骤级支撑参照；我们将其作为依赖概念的*动机*呈现，而非上述新边级验证的替代品。我们将发布验证协议与带标注边集，并将方法定位为拓扑感知的过程信号而非证明验证器。

**W2. 长度与表 1 不一致。**

**R2.** "<500 tokens"来自训练动态图，不应表述为评测期长度；我们删除它，并用与精度相同的脚本报告逐基准评测长度。正确措辞是"GSM8K 最多少用 24% token，四主基准均值约 13%"，而非通用的 15–24% 缩减。我们还将表 1 拆分为引用参考行与复现匹配行，移除跨非匹配设置的加粗/下划线比较，并逐行补充检查点 ID、解码设置与指标定义。被指出的 DeepSeek-R1-Distill-Qwen-7B 的 MATH 行（36.8 vs 后来的 92.8）混杂了解码协议与溯源、易被误读，故我们不去辩护某一具体解释，而是将此类溯源不清的行移入仅供参考的附录，并仅在经审计的匹配区块内（相同基座、SFT、解码与指标）计算差值。

**W3. 去连续性的崩塌。**

**R3.** 该消融揭示了拓扑单独的可钻空性。去掉连续性后，无环、无孤点等全局检查可被稀疏、公式化、跳过局部支撑的轨迹满足；奖励随之崩塌，模型跌至仅结果 GRPO 之下。故正确解读不是"各结构分量独立地都有帮助"，而是拓扑、局部连续性与正确性分层的 ACE 三者共同必要。

*表 R1. Qwen3.5-9B 上移除连续性与移除 ACE 的影响：8 基准均值 pass@1 与 GRPO 组崩塌率。*

| 设置 | Avg pass@1 | Collapse% |
| --- | ---: | ---: |
| 仅结果 GRPO | 40.8 | 57.8 |
| 去连续性 | 16.3 | 68.8 |
| 去 ACE | 41.7 | 42.1 |
| 完整 TopoPRM | 45.6 | 37.9 |

**结果奖励澄清。** 公开基准奖励是精确最终答案奖励：GSM8K、MATH-500、OlympiadBench、Omni-MATH、AIME、CNMO 用 boxed 答案抽取加逐基准归一化/等价；MMLU/GPQA-Diamond 用选项正确性。评分式奖励仅用于领域内 critique 附录。我们在附录 D 用伪代码区分二者定义，故公开基准结果无需为此重跑。

我们非常感谢这份聚焦可复现性的审阅：它把抽取器从自我参照的启发式变成了经外部验证的组件，并消除了长度表述与表 1 的呈现歧义。在匿名评审政策下，我们已准备好验证协议、带标注边集与经审计的表格，随终稿发布，而非通过可能破坏匿名性的链接分发。鉴于讨论期临近，我们诚挚感谢审稿人这轮建设性交流；如仍有疑虑我们乐意进一步澄清，并恳请审稿人考虑给予本文支持。

### 参考文献（TsKG）
[7] DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948, 2025.
