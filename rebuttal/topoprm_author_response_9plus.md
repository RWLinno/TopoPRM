# TopoPRM Author Response - 9+ Polished Version

## Response to Reviewer HxUk

Thank you for the constructive review and for recognizing the motivation behind non-local dependency modeling, the correctness-first reward, and ACE. Your concerns ask whether the gains come from topology itself rather than length control, one model family, or an unvalidated extractor. We answer the three points in order.

**R1 - Edge-level validation against human-adjudicated dependency references.** We agree this is the most direct test of the topology signal. We built a 120-trace validation set from GSM8K/MATH, stratified by solution length. For each trace, we segment the solution into steps, enumerate ordered step pairs, and label whether step `i` is necessary support for step `j`. We use two references: (i) a held-out Qwen3-32B annotator for scalable annotation and diagnostics, and (ii) a human-adjudicated reference used as the final dependency graph for scoring.


| Score                | Precision | Recall | F1   |
| -------------------- | --------- | ------ | ---- |
| Qwen3-32B annotation | 0.48      | 0.59   | 0.53 |
| Human-adjudicated    | 0.58      | 0.71   | 0.64 |


The human-adjudicated reference gives **P=0.58/R=0.71/F1=0.64**, showing that extracted edges align with human-judged support relations beyond arbitrary text overlap. The edge-type audit also exposes the main failure mode: expression/claim/order edges are much more reliable (precision 0.59-0.63), while variable-only overlap is weak (0.25) and accounts for most false positives. We therefore add two conservative guards in the released extractor: variable edges require at least two shared variables, and fallback order edges require local lexical overlap. To quantify reference stability, a second architecture-distinct annotator (Qwen2.5-32B) agrees with Qwen3-32B at 0.90 raw agreement and Cohen's kappa 0.73; we will report this as annotator-stability evidence, while keeping the human-adjudicated labels as the final reference. The revision will include the annotation guideline, edge-type table, and representative DAG visualizations. We also narrow the claim to **surface-evidenced support dependencies**, not proof-level logical dependency recovery.

**R2 - Stronger distillation and length-control comparisons.** For distillation, we compare TGSD with matched 4B baselines under the same 9B teacher and data.


| 4B student              | GSM8K | MATH-500 | Token ratio | Structural retention |
| ----------------------- | ----- | -------- | ----------- | -------------------- |
| Off-policy KL [2]       | 76.8  | 49.3     | >1.0x       | 0.41x                |
| Static SFT distillation | 79.4  | 58.2     | 0.64x       | 0.86x                |
| TGSD                    | 82.8  | 61.5     | 0.56x       | 0.93x                |


This table addresses the static/off-policy baselines. To address the deeper concern--whether topology helps beyond generic compression--we additionally ran a matched outcome+length GRPO baseline with the same length regularizer but no topology/continuity signal. On a controlled GSM8K subset, outcome-only, outcome+length, and full TopoPRM obtain 75.5, 76.5, and 77.0 pass@1, respectively. We treat this as an attribution check rather than a headline: length helps, but does not fully account for the TopoPRM gain. In the revision, we will make the remaining comparison explicit by adding a plain on-policy distillation/revision ablation without topology-conditioned prompts; the key test is whether the same on-policy mechanism still benefits from the DAG-derived revision target.

**R3 - Generality and statistics.** You are right that most trainable runs use Qwen-family or Qwen-distilled backbones. We chose them to keep the RL recipe, tokenizer, and evaluation protocol controlled, but this alone does not prove cross-family generality. We therefore add a non-Qwen sanity check with Llama-3.1-8B-Instruct. On GSM8K, outcome-only GRPO and TopoPRM both reach 85.0 pass@1, while TopoPRM reduces mean generation length from 1008 to 619 tokens. Since this benchmark is saturated for this instruct model, we interpret the result as efficiency-transfer evidence, not a broad non-Qwen accuracy claim. For statistical reliability, we run three seeds of the matched DR1-7B comparison:


| Reward            | GSM8K pass@1 | Wilson 95% CI, n=200 |
| ----------------- | ------------ | -------------------- |
| Outcome-only GRPO | 75.5 +/- 0.4 | [69.1, 80.9]         |
| Full TopoPRM      | 76.0 +/- 0.7 | [69.6, 81.4]         |


The intervals overlap on this saturated subset, so we do not claim a significant single-benchmark win. The substantive accuracy gain appears in the nine-benchmark average on DR1-7B (58.3 vs. 55.1), and we will report uncertainty rather than overemphasize small competition sets.

### References

[1] Agarwal et al. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. ICLR 2024.  
[2] Gu et al. MiniLLM: Knowledge Distillation of Large Language Models. ICLR 2024.  
[3] Aggarwal and Welleck. L1: Controlling How Long a Reasoning Model Thinks with RL. arXiv:2503.04697, 2025.  
[4] Luo et al. O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning. arXiv:2501.12570, 2025.

---



## Response to Reviewer B5w7

Thank you for the careful review. The central question is whether the extracted DAG is a reliable process signal or a surface-continuity artifact. We address this by validating the edges, quantifying the structure-semantics gap, and isolating mechanisms.

**R1 - Surface reuse vs. support dependencies.** We evaluate extracted edges against two independent necessary-support references on 120 stratified traces: a held-out Qwen3-32B annotator and a human-adjudicated reference. The edge-type breakdown directly tests the surface-reuse concern.


| Edge type                 | Precision | Interpretation                    |
| ------------------------- | --------- | --------------------------------- |
| expression / claim reuse  | 0.59-0.63 | intermediate-result support       |
| explicit order / citation | 0.61      | step-level support cue            |
| variable-only overlap     | 0.25      | main surface-overlap failure mode |


Overall, the extractor reaches P=0.48/R=0.59/F1=0.53 against the Qwen3-32B pre-annotation and P=0.58/R=0.71/F1=0.64 against the human-adjudicated reference. The clearest surface-overlap failure is variable-only matching, so the released extractor adds the two guards described above. We also revise the wording: the graph should not be read as a proof graph or complete logical dependency graph, but as a **surface-evidenced support graph** that provides useful process supervision when gated by correctness.

**R2 - Topology is not semantic correctness.** We agree that a clean dependency structure does not guarantee a correct key deduction. This is why topology is never used as a standalone correctness classifier. ACE computes advantages within correct and wrong strata, so structural scores can re-rank completions inside a stratum but cannot create a topology-induced positive update that crosses the correctness boundary. We will add the requested error cases: high-topology wrong traces, where a locally supported chain contains one false key deduction, and low-topology correct traces, where the answer is correct despite skipped or compressed intermediate support. These cases make the limitation explicit: `q_topo` measures exposed support structure, not semantic validity. The observed high-vs-low-topology correctness gap increases from 0.150 under outcome-only GRPO to 0.171 under full TopoPRM, suggesting better coupling after training, but we present this only as diagnostic evidence, not as proof that topology verifies correctness.

**R3 - Source of the gains.** We reorganize the ablations around mechanisms on Qwen3.5-9B.


| Reward setting                    | GSM8K | MATH-500 | AIME'24 | Avg  | Collapse% |
| --------------------------------- | ----- | -------- | ------- | ---- | --------- |
| Outcome-only GRPO                 | 93.3  | 50.8     | 16.7    | 40.8 | 57.8      |
| w/o ACE                           | 93.8  | 50.8     | 20.0    | 41.7 | 42.1      |
| w/o topology, length + continuity | 93.0  | 51.0     | 20.0    | 41.8 | 44.6      |
| Full TopoPRM                      | 94.1  | 51.0     | 30.0    | 45.6 | 37.9      |


Length/continuity without topology does not reach the full model, and ACE reduces collapse. We also ran a small best-of-N diagnostic with Qwen2.5-Math-PRM-7B [5]. TopoPRM alone is not a competitive outcome reranker, as expected, but combining PRM scores with topology slightly improves MATH-500 reranking over PRM alone (71.2 vs. 70.0) in the same candidate pool. We present this as complementarity evidence, not as a replacement for PRMs or verifiers.

### References

[5] Zhang et al. The Lessons of Developing Process Reward Models in Mathematical Reasoning. arXiv:2501.07301, 2025.  
[6] Skalse et al. Defining and Characterizing Reward Gaming. NeurIPS 2022.

---



## Response to Reviewer TsKG

Thank you for the reproducibility-focused review. We agree that the submitted draft made several empirical claims harder to trust than necessary. We separate the core validation issue from presentation/provenance errors and correct both.

**R1 - Self-referential extractor diagnostics.** The submitted structural diagnostics were internal-consistency checks computed by the same extractor used during training. We now add external edge-level validation on 120 stratified traces. The references are computed separately from the training pipeline: a held-out Qwen3-32B annotator for pre-annotation, followed by human adjudication. The extractor scores P=0.48/R=0.59/F1=0.53 against the Qwen3-32B pre-annotation and P=0.58/R=0.71/F1=0.64 against the human-adjudicated reference; Qwen2.5-32B agrees with Qwen3-32B at 0.90 raw agreement and Cohen's kappa 0.73. We will release the validation protocol and labeled edge set, and rescope the method as a topology-aware process signal rather than a proof verifier. The in-domain teacher-critiquing data in Appendix F will be described more carefully: it provides human step-level critique evidence that motivated our failure taxonomy, but we will not present it as a substitute for the new edge-level validation.

**R2 - Length and Table 1 inconsistencies.** The "<500 tokens" phrase came from a training-dynamics plot and should not have been stated as an evaluation-time length; we remove it and report benchmark-specific evaluation lengths from the same script used for accuracy. The correct wording is "up to 24% fewer tokens on GSM8K, about 13% on the four-primary-benchmark mean," not a universal 15-24% reduction. We also split Table 1 into quoted reference rows and reproduced matched rows, remove bold/underline comparisons across non-matched settings, and add checkpoint IDs, decoding settings, and metric definitions per row. For the DeepSeek-R1-Distill-Qwen-7B MATH row, the current presentation mixes protocol/provenance in a way that can be misread; we will move ambiguous rows to a reference-only appendix and compute deltas only within audited matched blocks.

**R3 - w/o-continuity collapse.** This ablation reveals hackability of topology alone. Without continuity, global checks such as acyclicity and no-orphan conclusions can be satisfied by sparse, formulaic traces that skip local support; the reward then collapses and the model falls below outcome-only GRPO. The interpretation is not that all structural components independently help, but that topology, local continuity, and correctness-stratified ACE are jointly necessary.


| Setting           | Avg pass@1 | Collapse% |
| ----------------- | ---------- | --------- |
| Outcome-only GRPO | 40.8       | 57.8      |
| w/o continuity    | 16.3       | 68.8      |
| w/o ACE           | 41.7       | 42.1      |
| Full TopoPRM      | 45.6       | 37.9      |


This table shows the estimator-level effect on collapse and accuracy. We will also add the requested high-structure-wrong cases before/after the estimator to clarify the failure mode qualitatively.

**Outcome reward clarification.** Public-benchmark rewards are exact final-answer rewards: boxed-answer extraction with benchmark-specific normalization/equivalence for GSM8K, MATH-500, OlympiadBench, Omni-MATH, AIME, and CNMO; option correctness for MMLU/GPQA-Diamond. The rubric-score reward is used only for the in-domain critique appendix. We separate these definitions in Appendix D with pseudocode, so the public-benchmark results do not require rerunning for this issue.

### References

[7] DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948, 2025.