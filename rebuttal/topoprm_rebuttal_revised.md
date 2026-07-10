# TopoPRM EMNLP/ARR Rebuttal Strategy and Drafts

Date: 2026-07-09  
Paper: `4012_Rewarding_the_Graph_Behin.pdf`  
Reviews: OpenReview PDF exported 2026-07-09  
Status: strategy + draft template; all `~` numbers are placeholders and must be replaced by real experiments.

---

## 0. Rebuttal Style Learned from Top-Tier Author Responses

### External sources consulted

1. [ICML 2026 Peer Review FAQ](https://icml.cc/Conferences/2026/PeerReviewFAQ): author responses are read by all reviewers, should correct factual errors and answer specific questions, and are expected to remain professional and concise under a strict character limit.
2. [Devi Parikh, How we write rebuttals](https://faculty.cc.gatech.edu/~parikh/citizenofcvpr/static/slides/rebuttals.pdf): rebuttals should be direct, organized by reviewer comments, positive first, then common/high-impact concerns, with concrete evidence rather than vague promises.
3. [Kargaran et al., Insights from the ICLR Peer Review and Rebuttal Process](https://arxiv.org/abs/2511.15462): evidence-backed clarifications and targeted responses are more useful than generic defenses; rebuttals can help borderline papers when they resolve concrete uncertainty.
4. [Huang et al., Rebuttal: The Art of Scientific Persuasion](https://arxiv.org/abs/2307.03371): empirical ICLR analysis shows rebuttals with effective interaction can shift reviewer scores, especially when authors address reviewer roots rather than only surface text.
5. [Jiu-Jitsu Argumentation for Writing Peer Review Rebuttals](https://aclanthology.org/2023.emnlp-main.894/): strong rebuttals often concede the reviewer’s concern at the right abstraction level, then redirect it to the paper’s actual claim/evidence.
6. [Ten simple rules for writing a response to reviewers](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013485): professional tone, clear point-by-point replies, and traceable revision promises reduce friction.

### Adapted rules for this paper

- Use **W/R format**: `W1. Reviewer concern` followed by `R1. Our response`.
- Start each reviewer response by reflecting their positive assessment because all reviewers gave Overall=3 / Findings-level acceptance.
- For shared concerns, answer once per reviewer but keep the framing reviewer-specific.
- For true draft errors, concede narrowly: “presentation/logging/table-provenance issue,” not “method invalid.”
- For single-reviewer misunderstandings, argue by clarifying what the paper actually does and promise clearer wording.
- Avoid overclaiming: TopoPRM is a topology-aware *process signal*, not a proof verifier.
- Put data/math early: edge validation, ACE sign guarantee, answer-reward clarification, token-stat audit.
- Use `~` placeholders only during drafting; all such values need replacement before submission.

---

## 1. Global Rebuttal Positioning

### Core acceptance message

All three reviewers agree that the idea is interesting and relevant: long CoT is not purely linear, topology-aware process supervision is a meaningful direction, and the experiments suggest accuracy/token-efficiency gains. Therefore the rebuttal should not sound defensive. The main task is to remove uncertainty about:

1. Whether the extracted DAG has external validity.
2. Whether topology can override semantic correctness.
3. Whether empirical claims are internally consistent and reproducible.
4. Whether the gains are only length control / training recipe / distillation.

### One-sentence central claim after rebuttal

> TopoPRM is not a proof-level verifier; it is a correctness-gated, topology-aware process signal that improves matched GRPO training and topology-guided distillation by rewarding recoverable dependency structure while preserving answer correctness as the primary signal.

### Recommended shared evidence package

| Evidence | Reviewer(s) answered | Why it matters | Placeholder result |
| --- | --- | --- | --- |
| Human/verifier edge validation | HxUk, B5w7, TsKG | Converts DAG extractor from self-referential to independently checked | P ~0.78, R ~0.71, F1 ~0.74, agreement ~0.69 |
| Structure-semantic gap analysis | B5w7, TsKG | Shows we understand when topology fails | high-topology wrong ~12% MATH, ~30% AIME/CNMO |
| Answer reward clarification | TsKG | Fixes severe misunderstanding about rubric reward | exact answer reward for public math, rubric only in in-domain appendix |
| Table/token audit | TsKG | Restores empirical trust | corrected mean tokens ~1.5k-1.7k, no “<500 tokens” claim |
| Length-aware GRPO baseline | HxUk, B5w7 | Separates topology from brevity penalty | outcome+length ~42.1 avg vs TopoPRM 45.6 |
| OPD/length KD baselines | HxUk | Strengthens TGSD comparison | OPD w/o topology ~80.6/58.9 vs TGSD 82.8/61.5 |
| Non-Qwen run | HxUk | Generality | Llama 8B +~2.5-3.5 avg, token -~8-13% |

---

## 2. Response to Reviewer HxUk

### Reviewer summary

Reviewer HxUk is positive and moderately confident. They recognize the value of non-local dependency modeling, the correctness-first reward, ACE, and the breadth of experiments. Their concerns are mostly “missing evidence,” not rejection of the method.

### Draft response

**Response to Reviewer HxUk**

Thank you for the constructive review. We appreciate your recognition that non-local dependencies provide a meaningful alternative to conventional linear PRMs, and that our correctness-first hierarchical reward and ACE mechanism are motivated by the need to avoid structurally coherent but incorrect solutions. We address the three main concerns below and will revise the paper to make these points clearer.

**W1. Dependency edges are automatically extracted; it would be useful to compare them with human-annotated graphs or report edge-level precision/recall/agreement.**

**R1. We agree that independent edge validation would make the topology signal more interpretable, and we will add it.** Our intended claim is not that TopoPRM recovers a formal proof graph, but that it extracts a *surface-evidenced dependency graph* useful for process supervision. The extractor uses expression reuse, canonicalized claim overlap, variable propagation, explicit step references, and conservative fallback sequential edges only when a derivation/conclusion step lacks other support. To test whether these edges correspond to human-recognizable support relations, we will add a blinded validation set of ~120 traces sampled from GSM8K, MATH-500, OlympiadBench, and Omni-MATH, stratified by correctness and trace length. Two math-trained annotators will mark support edges without seeing TopoPRM scores or model variants. We will report

\[
P=\frac{|A_E\cap A^\*|}{|A_E|},\quad
R=\frac{|A_E\cap A^\*|}{|A^\*|},\quad
F_1=\frac{2PR}{P+R},
\]

plus inter-annotator agreement and edge-type breakdown. Our current estimate is precision ~0.78, recall ~0.71, F1 ~0.74, and agreement ~0.69; expression/claim edges are expected to be more reliable (~0.82-0.85 precision) than variable-only or fallback edges (~0.62-0.66). We will add this as an independent diagnostic rather than another extractor-computed metric.

**中文思路：** 这里要“求同”：承认 reviewer 的问题是合理的，因为三位 reviewer 都在问 DAG 是否真实。不要说原文已经足够，而是说我们的 claim 不是 proof graph，而是 surface-evidenced support graph。这样既不放弃方法，又把评审的要求转化成“需要补独立验证”的合理要求。数学公式给 P/R/F1，让 AC 看到我们有可执行的验证方案。

**W2. Distillation is compared mainly with SFT distillation and off-policy KL; stronger on-policy distillation, reasoning compression, or length-control baselines would make the comparison more convincing.**

**R2. We agree that this is a useful strengthening, and we will add compute-matched baselines rather than claim TGSD wins only over weak distillation.** The current comparison was designed to isolate the difference between static teacher traces and topology-guided on-policy revisions. However, your comment points to the broader concern: whether TGSD improves because it preserves dependency structure, or merely because it shortens outputs. We will therefore add three baselines under the same teacher, student, data, and training budget: (i) on-policy KD without topology-conditioned revision prompts, following the motivation of on-policy distillation; (ii) length-controlled KD/GRPO with explicit budget pressure but no DAG signal; and (iii) SFT distillation using the same filtered teacher outputs. The planned 4B-student comparison is:

| Method | GSM8K | MATH-500 | Mean tokens | Structural retention |
| --- | ---: | ---: | ---: | ---: |
| SFT-distill | 79.4 | 58.2 | 438 | 0.86 |
| Off-policy KL | 76.8 | 49.3 | 4096 | 0.41 |
| On-policy KD w/o topology | ~80.6 | ~58.9 | ~421 | ~0.86 |
| Length-controlled KD | ~79.8 | ~57.6 | ~330 | ~0.72 |
| TGSD | 82.8 | 61.5 | 382 | 0.93 |

The key comparison will be TGSD vs on-policy KD without topology and vs length-controlled KD. If the final numbers follow this pattern, the evidence will show that TGSD is not only shorter, but retains the teacher’s dependency profile more faithfully.

**中文思路：** 不要把这个写成“我们之前 baseline 不够强所以错了”。应写成“当前 baseline 的目标是 isolate static vs topology-guided，但 reviewer 提醒了更强的 deeper intent：是不是只是 length control”。因此我们补 baseline 是为了回答更深层问题，而不是承认原实验无效。表格中 `~` 全是占位。

**W3. Most experiments are within the Qwen family; evaluation on Llama and multiple seeds/significance tests would support generality.**

**R3. We will clarify what the current evidence does and does not claim, and add a non-Qwen sanity run plus uncertainty estimates.** The current main experiments focus on Qwen-family checkpoints to keep tokenizer, training recipe, and evaluation protocol controlled; we did not intend to claim that the method is specific to Qwen. We already include Llama-family reference rows in Table 1, but agree that a trained non-Qwen TopoPRM run is a stronger generality test. We will add a matched Llama-3.1-8B-Instruct or DeepSeek-R1-Distill-Llama-8B run on the four primary benchmarks under the same SFT/GRPO budget. The expected pattern is an average gain of ~2.5-3.5 points over outcome-only GRPO with ~8-13% fewer tokens, but these values will be replaced by measured results.

For statistical reliability, we will add paired bootstrap confidence intervals for the main Qwen3.5-9B comparison and Wilson/McNemar intervals for small competition benchmarks such as AIME and CNMO. This is important because AIME/CNMO have small sample sizes, and we should not overstate single-run differences there. We will revise the claims to emphasize aggregate trends and matched comparisons rather than isolated small-set wins.

**中文思路：** HxUk 的语气是“could better support generality”，不是致命攻击。回答时要解释为什么主实验控制在 Qwen 是为了公平，而不是方法依赖 Qwen；同时补一个 Llama run 作为 sanity test。对于 small benchmark，承认统计不稳定但不是承认方法失败，而是说明我们会用合适的 CI 处理。

**Closing.** These additions directly address your requested evidence: independent graph validation, stronger distillation/length-control baselines, a non-Qwen run, and uncertainty estimates. We will also revise the limitation section to state that TopoPRM provides a structural process signal rather than proof-level semantic verification.

**中文思路：** 结尾重申“我们补的是 reviewer 要的证据”，同时把 claim 收窄。这会让 reviewer 感觉他们的意见被认真采纳，也让 AC 看到我们有清晰的修订路线。

---

## 3. Response to Reviewer B5w7

### Reviewer summary

Reviewer B5w7 is also positive but more focused on soundness. They worry that DAG quality is not validated, topology may reward formally clean but semantically wrong traces, and gains may come from recipe/length/distillation rather than topology-aware supervision.

### Draft response

**Response to Reviewer B5w7**

Thank you for the careful review and for clearly identifying the main uncertainty: whether the extracted dependency graph is a reliable process signal rather than a surface-continuity artifact. We agree that this distinction should be made sharper, and we will revise both the empirical evidence and the claim wording accordingly.

**W1. The extracted DAG may reflect expression reuse or surface continuity rather than true logical dependencies.**

**R1. We will clarify that TopoPRM extracts surface-evidenced support dependencies, and we will add independent validation to quantify how often these edges align with human-labeled reasoning support.** The method is intentionally training-free and does not claim proof-level dependency recovery. Its role is closer to an implicit PRM signal: it checks whether a trace exposes recoverable support structure that outcome-only rewards cannot see. To make this testable, we will add a human/verifier edge validation set of ~120 traces from four public benchmarks. Annotators will mark whether a source step is needed to justify a target step. We will report edge precision/recall/F1, agreement, and edge-type reliability. Placeholder results are P ~0.78, R ~0.71, F1 ~0.74, agreement ~0.69; expression/claim edges are more precise than variable-only and fallback edges. We will also add examples where the extractor succeeds and fails, especially variable reuse false positives.

This addition changes the evidence from internal consistency (“the trained model scores higher under the same extractor”) to external validity (“the extracted edges overlap with independently judged support relations”). We will also revise language such as “actual logical dependencies” to “surface-evidenced support dependencies” where needed.

**中文思路：** B5w7 的 W1 和其他 reviewer 是共同问题，必须认真回应。但措辞上不要承认“只是 surface artifact”，而是说“我们的对象本来就是 surface-evidenced support dependencies，并通过独立验证证明它不是纯 artifact”。最后承诺改 wording，避免过度 claim。

**W2. The relation between topology reward and semantic correctness is unclear; structurally clean traces can still be wrong.**

**R2. This is a real limitation, but the optimization design is specifically built so topology cannot override answer correctness.** The total reward contains structural diagnostics, but ACE applies correctness-stratified clipping. For each prompt group, let

\[
C=\{i:r_{out}^{(i)}=1\},\qquad W=\{i:r_{out}^{(i)}=0\}.
\]

For correct samples, the advantage is clipped to be non-negative; for wrong samples, it is clipped to be non-positive:

\[
\hat A_i\in[0,c_+]\ \text{if}\ i\in C,\qquad
\hat A_i\in[c_-,0]\ \text{if}\ i\in W.
\]

Thus topology can re-rank traces *within* the same correctness stratum, but it cannot promote an incorrect trace above a correct one. This is why we describe the reward as correctness-first rather than additive multi-reward optimization. We will make this mathematical point more explicit in Section 3.3 and Appendix B.

We also agree that this guarantee does not eliminate the structure-semantic gap. We will add a diagnostic table measuring high-topology wrong traces, e.g., \(\Pr(r_{out}=0\mid q_{topo}>0.8)\), and correct but low-topology traces, e.g., \(\Pr(r_{out}=1\mid q_{topo}<0.5)\). Estimated values are ~11-14% high-topology wrong on MATH-500 and ~28-33% on AIME/CNMO. This explains why TopoPRM improves average/token efficiency while not always beating outcome-only GRPO on the most adversarial competition-level benchmarks.

**中文思路：** 这里要“求同存异”：承认 reviewer 的本质担忧是真的——结构正确不等于语义正确；但马上用 ACE 数学性质说明这不是致命漏洞，因为 topology 不会跨 correctness stratum 翻转 advantage。这个数学解释是 rebuttal 的核心防线。

**W3. It is hard to isolate whether gains come from topology supervision, length control, continuity regularization, recipe, or distillation.**

**R3. We will reorganize the ablations around mechanism isolation and add a length-aware GRPO baseline.** The existing ablations already show that removing topology, continuity, or ACE changes both accuracy and collapse rate, but we agree that the explanation should be sharper. In particular, the “w/o continuity” result should not be framed as a benign complementarity result: it reveals that global topology without local traceability can be hacked. Continuity is the local guard that prevents a trace from satisfying acyclicity/no-orphan checks while skipping necessary step-to-step support.

We will add a matched baseline that uses outcome correctness plus the same length regularizer, but no topology/continuity signal. This directly tests whether the gain is merely length control. A planned table is:

| Variant | Mechanism tested | Avg pass@1 | Mean tokens | Collapse% |
| --- | --- | ---: | ---: | ---: |
| Outcome-only GRPO | sparse answer reward | 40.8 | 1823 | 57.8 |
| Outcome+length GRPO | explicit length control | ~42.1 | ~1605 | ~53.0 |
| + continuity only | local traceability | ~41.8 | 1943 | 44.6 |
| + topology only | global structure without local guard | 16.3 | 2047 | 68.8 |
| Full TopoPRM | topology + continuity + ACE | 45.6 | 1692 | 37.9 |

The revised interpretation will be: topology and continuity are not two independently sufficient rewards; they form a coupled process signal under correctness-first clipping. This is consistent with reward-hacking concerns in multi-signal RL and with prior PRM work showing that dense process signals need calibration against final-answer correctness.

**中文思路：** B5w7 说“full system 很难 isolate”，这是合理的。回答不要硬说已完全 isolate，而是说我们会按 mechanism 重新组织消融。特别要把 w/o continuity 的崩溃从“弱点”转成“说明 continuity 是 anti-hacking guard 的证据”。这就是 jiu-jitsu 式回应。

**Closing.** We will revise the claims to be more precise: TopoPRM is a correctness-gated structural process signal that improves matched GRPO accuracy and token efficiency, but it does not replace semantic verification. We believe this scoped claim is supported by the current matched results and will be strengthened by the new validation, gap analysis, and length-aware baseline.

**中文思路：** 结尾强调“scoped claim”，避免 reviewer 认为我们在逃避语义正确性问题。这个 reviewer 的 Soundness=3，所以重点是让他相信我们知道边界且补了证据。

---

## 4. Response to Reviewer TsKG

### Reviewer summary

Reviewer TsKG is the most dangerous review, not because score is lower, but because it questions empirical trust: unvalidated extractor, internal inconsistency, and brittleness of ablations. We should concede true presentation errors narrowly, but not concede that the method is invalid. The response must separate: (a) manuscript/table clarity mistakes; (b) genuine method limitation; (c) reviewer misunderstanding caused by unclear appendix wording.

### Draft response

**Response to Reviewer TsKG**

Thank you for the detailed review. We take the concerns about extractor validation and empirical consistency seriously. Several issues you flagged are presentation/provenance problems in the draft rather than intended methodological claims, and we will correct them explicitly. We respond point by point below.

**W1. The extractor is not validated against ground truth; structural-quality gains are self-referential.**

**R1. We agree that the current structural diagnostics alone are internal-consistency evidence, and we will add independent edge-level validation.** The extractor-computed metrics show whether training changes the model toward the structural signal, but they do not by themselves prove that extracted edges match human-perceived dependencies. We will add a blinded validation study on ~120 public-benchmark traces with human-labeled support edges. We will report precision, recall, F1, annotator agreement, and per-edge-source reliability. Placeholder results are precision ~0.78, recall ~0.71, F1 ~0.74, agreement ~0.69. We will also add failure cases, especially high-topology wrong answers and variable-overlap false positives.

We will revise the claim accordingly: TopoPRM is a topology-aware *process signal* based on recoverable support evidence, not a ground-truth proof verifier. This aligns the method with implicit/automated process supervision rather than formal proof checking.

**中文思路：** TsKG 的第一点是共同问题，而且他说得很重。这里应该承认“structural diagnostics are internal-consistency evidence”，因为这是事实；但马上转到补充 independent validation，避免让他把“self-referential”扩大为“method rests on nothing”。

**W2. The paper has internal inconsistencies in length reporting and main-table provenance.**

**R2. We found two presentation issues and will correct them; they do not change the matched TopoPRM-vs-GRPO comparison, but they made the draft confusing.** First, the sentence saying that TopoPRM “remains below 500 tokens” is incorrect in the public-benchmark evaluation scale and will be removed. The correct token statistics are those in Table 2/Table 3/Figure 4, approximately 1.5k-1.7k mean generated tokens depending on benchmark aggregation. The revised text will state “up to 24% fewer tokens” and will report benchmark-specific averages rather than the erroneous “<500 tokens” phrase.

Second, Table 1 mixed published reference rows and reproduced/matched training variants too tightly. We will split it into two blocks: (i) quoted open-source reference models, not used for bold/underline matched comparison; and (ii) our reproduced training variants from the same starting checkpoint and evaluation script. We will add checkpoint IDs, source labels, and exact evaluation protocol for every row. Any row whose provenance cannot be audited will be removed or replaced by rerun numbers. The main deltas will be computed only within matched backbone groups, e.g., outcome-only GRPO vs TopoPRM under the same checkpoint and training budget.

**中文思路：** 这里是真错/真危险。要承认但收窄：这是 presentation/provenance issue，不是 algorithm invalid。尤其 “<500 tokens” 明显冲突，必须承认并修。Table 1 的问题也要承认清楚，但强调 matched comparison 会保留并重新审计。

**W3. The appendix seems to imply rubric rewards were used on public math benchmarks; if so, results need rerunning with answer-correctness reward.**

**R3. This is a misunderstanding caused by unclear appendix organization: public math benchmarks use exact final-answer correctness rewards, while the rubric reward applies only to the in-domain teacher-critique appendix.** For GSM8K, MATH-500, OlympiadBench, Omni-MATH, AIME, and CNMO, \(r_{out}\) is computed from normalized final-answer matching, including boxed-answer extraction and benchmark-specific answer normalization. For MMLU/GPQA-Diamond, \(r_{out}\) is exact option correctness. The rubric-score reward described in Appendix D belongs to the in-domain critique validation set, where the target is teacher score prediction. We will rewrite Appendix D into two separated subsections:

\[
r_{out}^{public}(x,y)=\mathbf{1}[\mathrm{Normalize}(\hat a(y))=\mathrm{Normalize}(a^\*)],
\]

and

\[
r_{out}^{critique}(x,y)=\mathrm{RubricMatch}(\hat s(y),s^\*),
\]

so that the public benchmark reward is unambiguous. We will also include pseudocode and a manual answer-checker audit on ~200 samples, with expected agreement ~99%.

**中文思路：** 这是单个 reviewer 的理解，但原文 Appendix D 确实容易导致误解。不要说“we used wrong reward”；要说“public math 用的是 exact answer correctness，rubric 只用于 in-domain critique；我们会拆开 appendix”。这属于“argue + clarify”，不是完全承认错误。

**W4. Removing continuity drops below outcome-only GRPO; this looks like brittleness or hackability rather than complementarity.**

**R4. We agree that the topology-only ablation reveals a hackability failure mode, but this supports the design choice of coupling topology with continuity and ACE rather than invalidating the full method.** Global graph checks such as acyclicity and no-orphan conclusions can be satisfied by traces that are sparse, formulaically ordered, or superficially connected. The local continuity term prevents this by checking whether each step is supported by prior steps or the problem statement:

\[
q_{cont}(y)=
\begin{cases}
1, & \eta(y)=1,\\
0.8\eta(y), & \eta(y)<1,
\end{cases}
\quad
\eta(y)=\frac{1}{T}\sum_{i=1}^T \mathbf{1}[\mathrm{supported}(s_i,s_{<i},q)].
\]

Without continuity, a model can learn a high-level DAG shape while skipping local justifications, which explains the collapse rate increase. With full TopoPRM, ACE further ensures that structural rewards cannot flip wrong answers into positive-advantage samples:

\[
\hat A_i\in[0,c_+]\ \text{if}\ r_{out}^{(i)}=1,\qquad
\hat A_i\in[c_-,0]\ \text{if}\ r_{out}^{(i)}=0.
\]

We will add the direct measurement you suggested: high-structure wrong-answer traces before/after ACE and with/without continuity. Expected placeholder values are ~55-60% high-topology wrong traces in the topology-only condition, dropping to ~12-15% on MATH-style benchmarks and ~28-33% on AIME/CNMO under full TopoPRM+ACE. The revised wording will not claim topology alone is sufficient; it will state that topology, continuity, and correctness-first clipping are jointly necessary.

**中文思路：** 这是最需要“反转”的点。Reviewer 说 ablation 看起来是 brittleness；我们回应：是的，topology-only 会被 hack，但这正说明 continuity 是必要 guard。不要用“complementarity”硬圆，而是改成“joint necessity under correctness-first clipping”。

**W5. Some rows and model names are non-standard or implausible.**

**R5. We will audit and standardize model provenance.** The revised table will include exact HuggingFace/model checkpoint identifiers, whether a number is quoted or reproduced, pass@1/pass@5 setting, generation budget, and answer extractor. We will also remove bold/underline ranking from quoted reference rows to avoid implying a fully matched comparison. If any row cannot be reproduced under the stated protocol, we will remove it from the main table and move it to a clearly marked reference-only appendix.

**中文思路：** 这部分不要争辩，因为 table provenance 是 AC 很在意的信任问题。承诺 audit、checkpoint ID、source label、quoted/reproduced 分离，是最稳的修复。

**Closing.** We appreciate that your comments identified places where the draft over-compressed important experimental details. The revised version will separate public vs in-domain rewards, audit table provenance, correct the token-length statement, and add independent edge validation plus high-structure-wrong analysis. These changes narrow the claim but strengthen its evidential basis: TopoPRM is a correctness-gated topology-aware process signal, not a proof verifier or a topology-only reward.

**中文思路：** 结尾要把“被抓到错误”转成“我们会修订为更准确的 claim”。这对 TsKG 这种 reviewer 很重要：他最在意 reproducibility 和 empirical trust，所以不要空泛感谢，要列出具体修复项。

---

## 5. Revision Ledger

| ID | Source | Concern | Response claim | Manuscript action | Location | Status | Evidence | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1 | HxUk | Edge validity missing | Add human edge validation | Annotate ~120 traces, report P/R/F1/agreement | New appendix + main text sentence | planned | Placeholder P ~0.78/R ~0.71/F1 ~0.74 | Need annotation time |
| H2 | HxUk | Stronger distillation baselines | Add OPD and length KD baselines | Run 4B student baselines | Distillation section/Table 4 | planned | Placeholder table | Compute cost |
| H3 | HxUk | Qwen-only + no stats | Add Llama sanity run and CIs | Run 4-benchmark non-Qwen, bootstrap/Wilson intervals | Experiments appendix | planned | Placeholder +~2.5-3.5 | May be small |
| B1 | B5w7 | DAG may be surface artifact | Clarify surface-evidenced support graph | Revise claim wording and add validation | Intro/Method/Limitations | planned | Human validation | Wording must not weaken too much |
| B2 | B5w7 | Topology vs correctness | ACE prevents cross-stratum promotion | Add ACE sign-bound explanation | Method/App B | drafted | Formula in response | Need align with code |
| B3 | B5w7 | Gains may be length/training recipe | Add outcome+length GRPO and mechanism ablation | Run baseline, reorganize ablation | Table 3/Appendix | planned | Placeholder table | Need matched settings |
| T1 | TsKG | Extractor self-referential | Agree diagnostics internal, add independent validation | Same as H1 | Appendix | planned | Human validation | High priority |
| T2 | TsKG | Token inconsistency | Remove “<500 tokens”; audit token stats | Correct Sec 4.3/Fig 3 text | Section 4.3/Fig 3 | open | Table 2/3 logs | Must verify logs |
| T3 | TsKG | Table rows implausible | Split quoted vs reproduced; add checkpoint IDs | Audit Table 1 | Table 1 | open | Evaluation scripts/logs | Could reduce apparent gains |
| T4 | TsKG | Reward ambiguity | Public math exact answer, critique uses rubric | Rewrite reward appendix | Appendix D | drafted | Code pseudocode | Need code check |
| T5 | TsKG | w/o continuity collapse | Topology-only hackability; continuity is guard | Add failure analysis | Ablation section | planned | Placeholder high-structure wrong rates | Need logs |

---

## 6. Experiments to Run Next

### P0: must run before final rebuttal

1. **Audit existing code/logs/table provenance**
   - Locate reward computation for public math vs critique data.
   - Locate evaluation scripts for Table 1/2/3/4.
   - Check whether Table 1 rows are quoted, reproduced, or mixed.
   - Verify token accounting in Figure 3 vs Table 2/3/4.
   - Output: `rebuttal_audit.md` with exact file paths, command lines, and any inconsistent rows.

2. **Human/verifier edge validation**
   - Extract 120 traces from public benchmark eval logs.
   - Generate anonymized annotation JSON with segmented steps and candidate edges.
   - Use two annotators or one annotator + verifier-derived pseudo-label if deadline is tight.
   - Metrics: P/R/F1, agreement, type-specific precision, false-positive taxonomy.
   - Output: `edge_validation_results.json` and a markdown table.

3. **Structure-semantic gap analysis**
   - Compute high-topology wrong and low-topology correct rates.
   - Stratify by benchmark and by model variant: outcome-only, topology-only/w/o continuity, full TopoPRM.
   - Output qualitative cases: 2 high-topology wrong, 2 low-topology correct.

4. **Answer reward clarification audit**
   - Confirm public math uses exact answer correctness.
   - Add pseudocode for `r_out_public` and `r_out_critique`.
   - Manual check ~200 samples for answer extractor agreement.

### P1: strongly recommended

5. **Outcome+length GRPO baseline**
   - Same SFT checkpoint, same GRPO steps, same length regularizer, no topology/continuity.
   - Goal: show topology is not merely brevity pressure.

6. **On-policy KD without topology + length-controlled KD**
   - Same teacher/student/data as TGSD.
   - Goal: isolate topology-conditioned revision from ordinary OPD/compression.

7. **Non-Qwen sanity run**
   - Llama-3.1-8B-Instruct or DeepSeek-R1-Distill-Llama-8B.
   - If time limited, run GSM8K, MATH-500, OlympiadBench, Omni-MATH only.

### P2: if time allows

8. **Three seeds / confidence intervals**
   - Main Qwen3.5-9B comparison.
   - Bootstrap for larger sets; Wilson/McNemar for small AIME/CNMO.

---

## 7. Prompt for Cursor: Code Reading and Audit Phase

Copy the following prompt into Cursor when you connect to the server/project.

```text
You are helping finish an EMNLP/ACL ARR rebuttal for the project at /Knowin/foundation/weilinruan/TopoPRM/. Do not modify training logic yet. First perform a reproducibility and claim-audit pass.

Context:
- Paper: TopoPRM / TGSD, topology-aware process rewards for GRPO and reasoning distillation.
- Reviews question: DAG extractor validity, public-math outcome reward, token-length inconsistency, Table 1 provenance, w/o-continuity collapse, stronger baselines.

Tasks:
1. Read the repository structure and identify all files related to:
   - DAG extraction / edge construction / topology score / continuity score
   - reward aggregation and ACE/SCAE advantage calculation
   - public benchmark evaluation and answer extraction
   - in-domain critique/rubric reward
   - Table 1/2/3/4 generation scripts or logged outputs
   - TGSD distillation scripts
2. Produce a markdown audit file `rebuttal_audit.md` with:
   - exact file paths and functions/classes
   - which reward is used on each benchmark family
   - whether public math uses exact answer correctness or rubric score
   - where token lengths are computed and whether Figure 3/Table 2 use the same units
   - checkpoint/model names used for every Table 1 row
   - any row that appears quoted vs reproduced vs unclear
3. Do not run expensive training yet. You may run lightweight grep/python inspection and small dry-run scripts.
4. Flag contradictions instead of silently fixing them.
5. End with a prioritized list of experiments/scripts to run next.
```

---

## 8. Prompt for Cursor: P0 Experiment Phase

```text
Continue from `rebuttal_audit.md`. Now implement and run the P0 rebuttal experiments only. Keep outputs small, reproducible, and clearly logged.

P0 experiments:
1. Edge validation data export:
   - Sample ~120 public-benchmark traces from existing eval logs, stratified by benchmark, correctness, and trace length.
   - Segment steps and export candidate DAG edges with edge source/type.
   - Save to `rebuttal_outputs/edge_validation_annotation.jsonl`.
   - If human labels are available, compute edge precision/recall/F1/agreement. If labels are not available, create the annotation pack and a metric script with placeholder labels.
2. Structure-semantic gap:
   - Compute Pr(wrong | q_topo > 0.8), Pr(correct | q_topo < 0.5), and high-topology-wrong rates by benchmark/model variant.
   - Include outcome-only, w/o continuity/topology-only if logs exist, and full TopoPRM.
   - Save `rebuttal_outputs/semantic_gap_table.csv` and 4 qualitative cases.
3. Answer reward audit:
   - Confirm exact-answer reward for public math and option correctness for MMLU/GPQA.
   - Generate pseudocode or minimal extracted implementation snippets for rebuttal.
   - Manual/sample agreement check on ~200 examples if feasible.
4. Token/table audit:
   - Recompute mean generated tokens for Table 2/3/Figure 3 from logs.
   - Identify and correct any inconsistent aggregation.

Output:
- `rebuttal_outputs/p0_summary.md` with real numbers replacing all `~` placeholders where possible.
- All tables in markdown and CSV.
- A list of remaining missing values.

Do not invent results. If a log is missing, write MISSING and suggest the exact command to regenerate it.
```

---

## 9. Prompt for Cursor: P1 Baseline Phase

```text
Use the audit and P0 results to run P1 baselines if compute/time permit. Keep all settings matched to the paper unless explicitly stated.

Baselines:
1. Outcome+length GRPO:
   - Same SFT checkpoint, same data, same GRPO steps, same length reward, but remove topology and continuity reward.
   - Compare against outcome-only GRPO and full TopoPRM.
   - Metrics: pass@1/pass@5/maj@5, mean tokens, Acc/kTok, reward collapse%.
2. On-policy KD without topology:
   - Same teacher/student/data as TGSD, but remove topology-conditioned revision prompts/gates.
   - Metrics: GSM8K, MATH-500, mean tokens, structural retention.
3. Length-controlled KD:
   - Same teacher/student/data, explicit length objective or prompt budget, no topology signal.
   - Metrics same as above.
4. Non-Qwen sanity run if feasible:
   - Llama-3.1-8B-Instruct or DeepSeek-R1-Distill-Llama-8B on GSM8K/MATH/Olympiad/Omni.

Output:
- `rebuttal_outputs/p1_baselines.md`
- exact commands, config files, checkpoint IDs, seeds
- markdown tables ready to paste into rebuttal
- note if any baseline is incomplete or not compute-matched

Do not overwrite existing checkpoints. Do not claim significance without CIs or matched tests.
```

---

## 10. Prompt for Cursor: Final Rebuttal Polishing Phase

```text
We now have real P0/P1 results. Update `topoprm_rebuttal_revised.md` into a final submission-ready author response.

Rules:
1. Replace every `~` placeholder with real values or delete the claim.
2. Preserve W/R structure:
   - Response to Reviewer HxUk
   - W1 ... R1 ...
   - Response to Reviewer B5w7
   - W1 ... R1 ...
   - Response to Reviewer TsKG
   - W1 ... R1 ...
3. Keep a calm, professional tone.
4. For true draft errors, concede narrowly and state exact correction.
5. For misunderstandings, clarify original intent and promise clearer wording.
6. For shared concerns, avoid repetitive prose but ensure each reviewer sees their concern answered.
7. Make every promised change traceable to a result, table, appendix, or code artifact.
8. Remove Chinese commentary from the final author-response section, but keep it in a separate internal appendix if needed.
9. If there is a character limit, produce:
   - full version
   - 5000-character-per-reviewer version
   - ultra-compact global response version
10. Run a final consistency checklist:
   - no unsupported claims
   - no fake or placeholder numbers
   - no overclaiming proof-level verification
   - public vs in-domain reward separated
   - quoted vs reproduced rows separated
   - token numbers consistent
   - all reviewer questions answered

Output:
- `final_author_response.md`
- `final_author_response_compact.md`
- `revision_ledger_final.md`
```

---

## 11. Claims to Avoid in Final Response

- Avoid: “TopoPRM recovers true logical dependencies.”  
  Use: “TopoPRM extracts surface-evidenced support dependencies.”

- Avoid: “Topology alone improves reasoning.”  
  Use: “Topology, continuity, and correctness-first ACE are jointly necessary.”

- Avoid: “TopoPRM remains below 500 tokens.”  
  Use: corrected benchmark-specific token reduction.

- Avoid: “All baselines are directly comparable.”  
  Use: split quoted reference rows and matched reproduced rows.

- Avoid: “We will add verifier-guided training” unless actually run.  
  Use: limitation/future work if not run.

- Avoid: submitting any `~` placeholder value.

---

## 12. Final Internal Checklist

- [ ] All `~` numbers replaced by real results or removed.
- [ ] Table 1 source/provenance audited.
- [ ] Public math reward confirmed as exact answer correctness.
- [ ] In-domain critique reward separated.
- [ ] Token-length statement corrected.
- [ ] Edge validation completed or downgraded to planned limitation if labels unavailable.
- [ ] Structure-semantic gap table completed.
- [ ] w/o-continuity collapse explained as topology-only hackability.
- [ ] Reviewer-specific W/R format preserved.
- [ ] Chinese commentary removed from final submission version.
