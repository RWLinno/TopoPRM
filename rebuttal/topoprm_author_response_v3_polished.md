# TopoPRM Author Response v3 Polished

## Response to Reviewer HxUk

Thank you for the constructive review and for recognizing the motivation behind non-local dependency modeling, the correctness-first reward, and ACE. Your concerns ask whether the gains come from topology itself, rather than from length control, one model family, or an unvalidated extractor. We address these points with new diagnostics and a narrower claim.

**W1. Human validation of DAG extraction.** We agree this is the most direct way to test whether the topology reward reflects reasoning dependencies rather than surface overlap. We therefore added a human-adjudicated edge-validation study on 120 stratified traces from GSM8K/MATH, covering short, medium, and long solutions. The procedure is: (i) segment each reasoning trace into steps; (ii) enumerate candidate ordered step pairs; (iii) ask a held-out Qwen3-32B annotator to propose whether step `i` is necessary support for step `j`; and, crucially, (iv) have human experts adjudicate the proposed support relation into the final reference edge set. Thus Qwen3-32B is used only to scale pre-annotation/diagnostics; the reported precision/recall/F1 are computed against the human-adjudicated reference DAGs.

| Extractor variant | Human-reference precision | Human-reference recall | Human-reference F1 | Main finding |
| --- | ---: | ---: | ---: | --- |
| Original extractor | 0.48 | 0.59 | 0.53 | variable-only and bare order edges dominate false positives |
| Guarded extractor | 0.58 | 0.71 | 0.64 | conservative variable/order guards improve all metrics |

We also report edge-type agreement to make the failure modes transparent: expression/claim/order edges are substantially more reliable, while variable-only edges have low precision and account for most false positives. Based on this audit, we add two conservative guards: variable edges require at least two shared variables, and fallback order edges require local lexical overlap. We will include the annotation guideline, human adjudication protocol, edge-type confusion table, and representative DAG visualizations in the revision. Importantly, we will scope the claim as **surface-evidenced support dependencies**, not proof-level logical dependency recovery.

**W2. Stronger distillation and length-control baselines.** The revised response separates two questions. First, for distillation, TGSD improves over static SFT distillation and off-policy KL under the same 9B teacher and 4B student setting.

| 4B student method | GSM8K | MATH-500 | Token ratio | Structural retention |
| --- | ---: | ---: | ---: | ---: |
| Off-policy KL | 76.8 | 49.3 | >1.0x | 0.41x |
| Static SFT distillation | 79.4 | 58.2 | 0.64x | 0.86x |
| TGSD | 82.8 | 61.5 | 0.56x | 0.93x |

Second, to test whether the reward gain is merely brevity, we ran a matched outcome+length GRPO baseline with the same length regularizer but no topology/continuity signal. On a controlled GSM8K subset, outcome-only, outcome+length, and full TopoPRM score 75.5, 76.5, and 77.0 pass@1, respectively. We will treat this as an attribution check rather than a headline result: length helps, but it does not fully explain the TopoPRM gain. We will also add a plain on-policy-without-topology distillation ablation if space permits.

**W3. Generality and statistics.** The main experiments use Qwen-family models to keep the tokenizer, SFT format, and evaluation protocol controlled, not because the method is Qwen-specific. We add a non-Qwen sanity check with Llama-3.1-8B-Instruct: on GSM8K, outcome-only GRPO and TopoPRM both reach 85.0 pass@1, while TopoPRM reduces mean generation length from 1008 to 619 tokens. This supports transfer of the efficiency signal, though we do not claim a broad non-Qwen accuracy gain from one saturated benchmark. We will report confidence intervals for matched comparisons and avoid emphasizing small-sample competition benchmarks without uncertainty estimates.

---

## Response to Reviewer B5w7

Thank you for the careful review. We agree that the central issue is whether the DAG is a reliable process signal or a surface-continuity artifact. The revision narrows the claim and adds validation, failure analysis, and mechanism isolation.

**W1. Surface reuse vs. support dependencies.** We now evaluate extracted edges against independently adjudicated support labels. The original extractor obtains P=0.48/R=0.59/F1=0.53; after adding guards for the two weakest edge types, it reaches P=0.58/R=0.71/F1=0.64. The edge-type analysis shows why this matters: variable-only overlap is indeed noisy, while expression/claim reuse and explicit/order support are substantially more reliable. We therefore no longer describe the graph as recovering “true logical dependencies.” The revised claim is that TopoPRM extracts **surface-evidenced support dependencies** that provide useful process supervision when gated by answer correctness.

**W2. Topology is not semantic correctness.** We agree and make this limitation explicit. A high topology score alone is not a correctness classifier: in a held-out diagnostic set, many high-topology traces still have wrong final answers. This is exactly why the method is correctness-first. ACE computes advantages within correct and wrong strata, so topology can re-rank traces inside a stratum but cannot turn a wrong answer into a positive-advantage update over correct answers. In the revision, we add high-topology/wrong-answer cases and state the expected failure mode: on competition problems, a clean dependency structure may still contain one invalid key deduction.

**W3. Source of the gains.** We reorganize the ablation around mechanisms rather than components.

| Reward setting | GSM8K | MATH-500 | AIME'24 | Avg | Collapse% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Outcome-only GRPO | 93.3 | 50.8 | 16.7 | 40.8 | 57.8 |
| w/o ACE | 93.8 | 50.8 | 20.0 | 41.7 | 42.1 |
| w/o topology | 93.0 | 51.0 | 20.0 | 41.8 | 44.6 |
| Full TopoPRM | 94.1 | 51.0 | 30.0 | 45.6 | 37.9 |

This shows that length/continuity without topology does not reach the full model, while ACE reduces collapse and stabilizes the process signal. We also ran a small best-of-N diagnostic with Qwen2.5-Math-PRM-7B. TopoPRM alone is not a competitive outcome reranker, as expected; however, combining PRM scores with topology slightly improves MATH-500 reranking over PRM alone in the shared candidate pool. We will present this only as evidence of complementarity, not as a replacement for PRMs or verifiers.

---

## Response to Reviewer TsKG

Thank you for the detailed reproducibility-focused review. We agree that the draft made some empirical claims harder to trust than necessary. We separate the core validation issue from presentation/provenance errors and correct both.

**W1. Self-referential extractor diagnostics.** The original structural diagnostics were internal-consistency checks. We now add an external edge-level validation: 120 stratified traces are labeled for pairwise necessary support by an independent annotator and manually adjudicated. The original extractor obtains P=0.48/R=0.59/F1=0.53; the guarded revision reaches P=0.58/R=0.71/F1=0.64. We will release the validation protocol and labeled edge set, and we rescope the method as a topology-aware process signal rather than a proof verifier.

**W2. Length and Table 1 inconsistencies.** We agree these must be fixed. The “<500 tokens” phrase came from a training-dynamics plot and should not have been stated as an evaluation-time length. We will remove it and report benchmark-specific evaluation lengths from the same script used for accuracy. The correct wording is “up to 24% fewer tokens, with about 13% reduction on the four-primary-benchmark mean,” not a universal 15-24% reduction. We will also split Table 1 into quoted reference rows and reproduced matched rows, remove bold/underline comparisons across non-matched settings, and add checkpoint IDs, decoding settings, and metric definitions. Rows whose provenance or metric is unclear will be moved to a reference-only appendix and not used for deltas.

**W3. w/o-continuity collapse.** We agree that this ablation reveals hackability of topology alone. Without continuity, global checks such as acyclicity and no-orphan conclusions can be satisfied by sparse, formulaic traces that skip local support; the reward then collapses and the model falls below outcome-only GRPO. The revised interpretation is therefore not “all structural components independently help,” but: topology, local continuity, and correctness-stratified ACE are jointly necessary.

| Setting | Avg pass@1 | Collapse% |
| --- | ---: | ---: |
| Outcome-only GRPO | 40.8 | 57.8 |
| w/o continuity | 16.3 | 68.8 |
| w/o ACE | 41.7 | 42.1 |
| Full TopoPRM | 45.6 | 37.9 |

**Outcome reward clarification.** Public benchmark rewards are exact final-answer rewards: boxed-answer extraction and benchmark-specific normalization/equivalence for GSM8K, MATH-500, OlympiadBench, Omni-MATH, AIME, and CNMO; option correctness for MMLU/GPQA-Diamond. The rubric-score reward is used only for the in-domain critique appendix. We will separate these definitions in Appendix D and include the corresponding pseudocode, so the public benchmark results do not require rerunning for this issue.

---

## References to keep in the response if space allows

- Rishabh Agarwal et al. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*. ICLR 2024.
- Yuxian Gu et al. *MiniLLM: Knowledge Distillation of Large Language Models*. ICLR 2024.
- Zhenru Zhang et al. *The Lessons of Developing Process Reward Models in Mathematical Reasoning*. Findings of ACL 2025.
- Joar Skalse et al. *Defining and Characterizing Reward Gaming*. NeurIPS 2022.
- Pranjal Aggarwal and Sean Welleck. *L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning*. COLM 2025 / arXiv:2503.04697.
- Haotian Luo et al. *O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning*. Findings of ACL 2026 / arXiv:2501.12570.
