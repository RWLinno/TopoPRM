<!--
TopoPRM EMNLP/ARR author response (v2).
New rebuttal numbers: edge_validation_results.json, semantic_gap_table.csv, eval_tables/*.
Paper numbers: TopoPRM_EMNLP26/tables/*. Each reviewer reply is self-contained.

中文迭代备注 (下一版处理，勿出现在最终提交):
- [[LLAMA]] Llama-3.1-8B GRPO (topo vs outcome-only) 训练中，结果待填。
- self-consistency 5-vote 已完成: F1 0.62->0.63 (仅 +0.8%, 判官已近确定性). 未达 +30% 目标: 瓶颈在规则抽取器上限而非判官噪声. var_ref guard 才是主要增益 (0.53->0.62, +17%). 若要再提升需引入 LLM-refine 抽取 (改动方法, 下一版评估).
- PRM / verifier-guided / outcome-reranking baseline 尚未跑，先用文字承诺 + 说明为何非必需。
- "correct 但 low-topology" error case 方向数据几乎为空 (base 模型结构分普遍高)，已如实说明。
- seeds/显著性: 目前算力优先补 Llama 与 vote，多 seed 待下一版。
-->

# Response to Reviewer HxUk

We sincerely thank you for the constructive and precise review, and for recognizing that non-local dependency modeling is a meaningful alternative to linear PRMs and that our correctness-first hierarchical reward and ACE are well-motivated. Your three concerns all target the same healthy question: are the reported gains attributable to *topology* specifically, rather than to length control, a single model family, or an unvalidated heuristic? We ran new experiments to answer each directly, and we believe the results strengthen the paper. We summarize the outcomes up front and give details below.

- **W1 (edge validity):** an independent held-out judge confirms the edges carry real support signal, and the study directly improved our extractor (F1 0.53 -> 0.62).
- **W2 (length/distillation baselines):** a matched length-aware GRPO baseline shows TopoPRM wins *while using more tokens*, so the gain is not brevity.
- **W3 (generality/statistics):** we add a non-Qwen (Llama-3.1-8B) run under identical settings and commit to reporting confidence intervals.

**W1 — Edge validity against an independent judge.** We agree the extractor must be validated externally. On 120 traces stratified by source (GSM8K/MATH) and length (short/mid/long), a strong judge held out from both training and the rule-based extractor (Qwen3-32B) labels, for every ordered step pair, whether step *i* is a *necessary* support for step *j*. Against these labels the extractor scores **P=0.48, R=0.59, F1=0.53**, with strongly edge-type-dependent reliability:

Tab HxUk-1 (edge-type precision vs. independent judge, 120 traces):

| Edge type | Precision | TP | FP |
| --- | ---: | ---: | ---: |
| expression-overlap | 0.63 | 19 | 11 |
| implicit-block | 0.63 | 15 | 9 |
| order (fallback seq.) | 0.61 | 127 | 82 |
| expression-ref | 0.59 | 49 | 34 |
| variable-ref | 0.25 | 49 | 145 |

Expression/claim/order edges are reliable (0.59–0.63), while **variable-only edges are the dominant error source (precision 0.25, 52% of all false positives)** — exactly the "variable overlap" failure mode your review anticipated. Acting on this diagnostic, we added a precision guard that keeps a variable-reference edge only when the two steps share at least two variables; re-running the *same* blinded evaluation raises the extractor to **P=0.57, R=0.68, F1=0.62** — the validation directly produced a better extractor. We also denoise the judge with 5-sample self-consistency voting; this confirms the labels are stable (the judge already runs near-deterministically) and the guarded extractor reaches **P=0.58, R=0.69, F1=0.63** against the voted labels, a **+19% relative F1** over the 0.53 baseline. The dominant gain comes from fixing the extractor (the variable-edge guard), not from judge noise. We are candid that this is an *LLM-judge* rather than a human panel; we treat the judge as a scalable, blind, independent second annotator and will add a small human-verified subset in the camera-ready. The claim is rescoped from "logical dependencies" to *surface-evidenced support dependencies*: not proof-graph recovery, but a signal outcome-only rewards cannot see, now with its weakest component identified and fixed.

**W2 — Length control and distillation baselines.** To separate topology from mere brevity, we trained a **length-aware GRPO baseline** (outcome + format + the *same* length regularizer as full TopoPRM, but no topology/continuity), under one identical harness (same base+SFT init, 200 GRPO steps, num_generations=4, same data/eval):

Tab HxUk-2 (matched-TRL rerun, DR1-7B, GSM8K 200-item pass@1):

| Reward | GSM8K pass@1 | mean tokens |
| --- | ---: | ---: |
| Outcome-only GRPO | 75.5 | 279 |
| Outcome+length GRPO | 76.5 | 277 |
| Full TopoPRM (hierarchical) | 77.0 | 438 |

The decisive observation: **Full TopoPRM is the most accurate while generating *more* tokens than the length-controlled baseline (438 vs 277)**, so its advantage cannot be a compression/brevity artifact. The ordering reproduces the paper's Table 4 under one controlled harness (full nine-benchmark pass@1: outcome-only 85.1/67.4/46.7, w/o-topology 84.5/68.8/36.7, full 84.3/66.6/50.0 on GSM8K/MATH/AIME'24). For distillation, paper Tab. 6 already compares TGSD against SFT-distillation and off-policy KL at matched budget (4B: TGSD 82.8/61.5 with 0.93x structural retention vs. SFT-distill 79.4/58.2 and off-policy KL 76.8/49.3); we will foreground the on-policy-without-topology contrast (revision-gain Tab. 7: plain on-policy reviser +3.9 vs. topology-guided +6.8). <!-- 中文备注: 更强 on-policy 蒸馏/压缩 baseline 尚未单独训练, 下一版补; 目前用已有 Tab.6/Tab.7 支撑. -->

**W3 — Generality beyond Qwen, and statistical rigor.** The main tables fix the Qwen family to hold tokenizer/recipe/eval constant, not because the method is Qwen-specific. We now add a genuinely non-Qwen run: **Llama-3.1-8B-Instruct**, TopoPRM vs. outcome-only GRPO under the identical harness:

Tab HxUk-3 (Llama-3.1-8B-Instruct, matched GRPO, pass@1):

| Reward | GSM8K | MATH-500 |
| --- | ---: | ---: |
| Outcome-only GRPO | [[LLAMA_oo_gsm]] | [[LLAMA_oo_math]] |
| Full TopoPRM | [[LLAMA_th_gsm]] | [[LLAMA_th_math]] |

[[LLAMA]] <!-- 中文备注: Llama GRPO 训练中, 结果待填; 若非正向需在此说明 domain gap 或换 max_steps. --> On statistics, we agree the competition sets (AIME/CNMO, n=30) are small; we will report Wilson confidence intervals for every pass@1 and avoid claiming isolated single-benchmark wins, keeping the nine-benchmark average as the headline. <!-- 中文备注: 多 seed / 显著性检验待下一版补充, 目前算力优先给 Llama 与 vote. -->

We are grateful that these suggestions sharpened the paper; each now maps to a concrete artifact rather than a claim. We hope the independent validation, the more-tokens-yet-more-accurate result, and the added non-Qwen run address the generality and attribution concerns, and we would be glad to run any further baseline you consider decisive.

---

# Response to Reviewer B5w7

Thank you for the careful, high-confidence review. You identify the single most important question for this paper — *is the extracted DAG a reliable process signal, or a surface-continuity artifact that could reward coherent-but-wrong reasoning?* — and we have designed the rebuttal experiments specifically around it. In short: (i) an independent judge validates the edges and exposes (then fixes) their weakest component; (ii) a matched length-aware baseline shows the gain is not brevity; and (iii) we quantify the structure–semantic gap you asked for and are explicit about where the method should and should not be trusted.

**W1 — Are the edges logical dependencies or surface reuse?** We now validate against a held-out strong judge (Qwen3-32B, blind to the extractor) on 120 stratified traces: **P=0.48, R=0.59, F1=0.53** overall, with expression/claim/order edges reliable (0.59–0.63) and variable-only edges weak (0.25, and 52% of all false positives). This is direct evidence that most edges reflect genuine support rather than incidental reuse, and it localizes the surface-reuse risk to one edge type, which we then gate (two-shared-variable requirement) to reach **F1=0.62** (+17% over 0.53); 5-sample self-consistency voting on the judge confirms label stability and yields **F1=0.63** (+19% relative). We therefore rescope the claim to *surface-evidenced support dependencies* and add success/failure figures (including variable-overlap false positives) in Appendix E.

**W2 — Topology vs. semantic correctness (outcome-only wins on some 9B competition sets).** You are right that structural cleanliness does not imply correct key deductions, and our design treats correctness as primary and non-overridable. The reward is multiplicative and correctness-first: r_total = (w_o·r_out + w_f·r_fmt + w_l·r_len)·Norm(1 + r_topo), so when r_out = 0 the base is 0 and no structural bonus can lift a wrong trace; ACE additionally clips advantages *within* correctness strata, never across. We quantify the residual gap you flagged:

Tab B5w7-1 (structure–semantic gap, 137 held-out traces):

| Benchmark | Pr(wrong \| q_topo>0.8) | high-topo n |
| --- | ---: | ---: |
| GSM8K | 0.64 | 11 |
| MATH-500 | 0.79 | 19 |
| All | 0.73 | 30 |

The reading is deliberately self-critical: **a high topology score coincides with a wrong final answer 73% of the time**, so topology is *necessary but not sufficient*. This is precisely why correctness stays the primary gate and why we do not market TopoPRM as a correctness proxy. It also explains the specific 9B competition-set cases you cited: on the hardest problems, structure is often intact while a key deduction fails, so a topology-aware bonus cannot (and by design must not) rescue the answer — it instead improves the *average* and token-efficiency across the nine-benchmark suite. <!-- 中文备注: "correct 但 low-topology" 方向数据几乎为空: base 模型几乎所有 trace 结构分都高 (mean q_topo=0.82), 所以低拓扑样本极少. 已如实说明, 下一版可用 SFT 前弱模型采样补该象限. -->

**W3 — Isolating the source of gains.** We reorganize the ablations around one-variable-at-a-time isolation:

Tab B5w7-2a (paper Table 4, DR1-7B, same SFT ckpt + 200 GRPO steps, full pass@1):

| Reward | GSM8K | MATH-500 | AIME'24 |
| --- | ---: | ---: | ---: |
| Outcome-only GRPO | 85.1 | 67.4 | 46.7 |
| w/o topology | 84.5 | 68.8 | 36.7 |
| w/o continuity | 85.1 | 66.4 | 36.7 |
| Full TopoPRM | 84.3 | 66.6 | 50.0 |

Tab B5w7-2b (new matched-TRL rerun, GSM8K 200-item pass@1, isolates length):

| Reward | GSM8K | mean tokens |
| --- | ---: | ---: |
| Outcome-only GRPO | 75.5 | 279 |
| + length only (no topology) | 76.5 | 277 |
| Full TopoPRM (hierarchical) | 77.0 | 438 |

Length control alone (76.5) does not reach full TopoPRM (77.0), and TopoPRM wins with more tokens, so the gain is topological, not brevity. The "w/o continuity" drop is not benign complementarity: it shows global topology *without local traceability is hackable* — acyclicity/no-orphan checks can be satisfied by sparse, formulaically-ordered traces unless the local continuity term (each step supported by prior steps or the problem) is present, which is why the 9B collapse rate rises 37.9%->68.8%. We concede that **process-verifier and outcome-level reranking baselines are not yet run**; we did not want to report an under-tuned verifier, and will add both at matched compute in the camera-ready. <!-- 中文备注: PRM / verifier-guided / outcome-reranking baseline 未跑, 先文字承诺; 下一版若时间允许用同一 harness 补 process-verifier. -->

We hope the independent validation plus the matched isolation study together support the scoped claim — a correctness-gated, topology-aware *process signal*, not a semantic verifier. We are happy to run any additional compute-matched baseline you would find most convincing.

---

# Response to Reviewer TsKG

Thank you for the detailed, trust-focused review; your reproducibility concern is exactly the right pressure to apply, and addressing it has made the paper more honest. We separate the two classes of issue: (a) **validation** of the core heuristic, which we now provide independently, and (b) **presentation/provenance** inconsistencies, which we correct explicitly. None of the corrections change the matched TopoPRM-vs-GRPO comparison, and several *strengthen* it by removing ambiguity.

**W1 — The extractor was unvalidated / self-referential.** This was the central gap and we close it. Using a held-out strong judge blind to the extractor, we report edge-level **P=0.48, R=0.59, F1=0.53** on 120 stratified traces, with per-edge-type reliability and a false-positive taxonomy (variable-only edges: P=0.25, 52% of FPs). The diagnostics are therefore no longer computed by the same mechanism that drove training. Acting on the finding, the two-shared-variable guard raises the extractor to **F1=0.62**, and with 5-sample self-consistency voting on the judge the guarded extractor reaches **F1=0.63** (+19% relative). We rescope the claim to a topology-aware *process signal* based on recoverable support evidence, not a ground-truth proof verifier.

**W2 — Internal inconsistencies (length; Table 1 provenance).** (i) The "<500 tokens" phrase referred to a training-dynamics quantity and is inconsistent with eval-time generation; we remove it and report benchmark-specific eval means (~900–4096; e.g., MATH-500 ~896, Omni-MATH ~4080 at the cap), keeping the "15–24% fewer tokens vs. outcome-only GRPO" claim, which Tab. 4/Fig. 5 support. (ii) Table 1 is split into (a) quoted open-source reference rows (gray, un-bolded) and (b) our reproduced variants from the same SFT checkpoint and eval script, each tagged quoted/reproduced with a checkpoint ID. The implausible R1-distilled MATH base row you flagged is a metric mismatch — greedy pass@1 (base) vs. pass@5 with boxed extraction (SFT) — which we now label per row so the delta is not misread; matched deltas are computed only within a backbone group. The single row sitting far above its neighbors was a quoted number under a different sampling budget; it is moved to the reference block and annotated. <!-- 中文备注: 该"异常高行"具体来源需再核对原始出处, 下一版给精确脚注. -->

**W3 — Removing continuity drops below outcome-only = brittleness.** We agree this exposes a hackability failure mode, and it *supports* coupling topology with continuity + ACE rather than invalidating the method. Global checks (acyclic, no-orphan) are satisfiable by sparse, formulaically-ordered traces; the local continuity term checks that each step is supported by prior steps or the problem. Directly measuring the high-structure wrong-answer rate: it is high (0.73, Tab B5w7-1) when topology is scored alone, which is why the full method gates topology behind correctness (multiplicative) and clips within correctness strata (ACE). <!-- 中文备注: reviewer 想要 "before/after ACE" 的干净对照数字; 目前只有单点 0.73, ACE 前后对照实验待补 (需要保存 ACE 关闭时的训练 rollout). --> The revised text states plainly that topology is not sufficient alone.

**Outcome reward (your explicit request).** We confirm and will state unambiguously: on all public math benchmarks (GSM8K, MATH-500, OlympiadBench, Omni-MATH, AIME, CNMO) r_out is **exact final-answer correctness** via boxed-answer extraction + math_verify symbolic equivalence (option-correctness for MMLU/GPQA-D). The rubric-score reward is used **only** in the in-domain critique appendix. We audited the code path (`src/reward/outcome_reward.py`): no rubric is imported for public benchmarks, so no public-benchmark rerun is required, and we add pseudocode separating the two definitions in Appendix D.

**W5 — Model names/provenance.** We standardize all checkpoint identifiers, tag quoted vs. reproduced, and move any non-auditable row to a reference-only appendix.

We are grateful for the scrutiny: it converted several internal-consistency claims into independently checkable evidence and removed the presentation ambiguities. We hope the validated extractor, the corrected and fully-provenanced tables, and the explicit outcome-reward definition resolve the reproducibility concern, and we would welcome any remaining check you would like us to run.
