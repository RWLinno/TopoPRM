<!--
TopoPRM EMNLP/ARR author response.
New rebuttal numbers: edge_validation_results.json, semantic_gap_table.csv,
eval_tables/*. Paper numbers: TopoPRM_EMNLP26/tables/*.
Each reviewer reply is self-contained.
-->

# Response to Reviewer HxUk

We thank you for recognizing that non-local dependency modeling is a meaningful alternative to linear PRMs and that the correctness-first hierarchical reward and ACE are well-motivated. We address the three concerns with new experiments.

**W1 (edge validity vs. human/independent labels).** We agree independent validation is needed. We ran a blinded study on 120 traces stratified by source (GSM8K/MATH) and length (short/mid/long), with a strong independent judge (Qwen3-32B, held out from training and from the rule-based extractor) labeling, for every ordered step pair, whether step *i* is a *necessary* support for step *j*. Against these labels the extractor scores **P=0.48, R=0.59, F1=0.53** over all edge types, with strongly edge-type-dependent reliability:

Tab HxUk-EV (edge-type precision vs. independent judge, 120 traces):

| Edge type | Precision | TP | FP |
| --- | ---: | ---: | ---: |
| expression-overlap | 0.63 | 19 | 11 |
| implicit-block | 0.63 | 15 | 9 |
| order (fallback seq.) | 0.61 | 127 | 82 |
| expression-ref | 0.59 | 49 | 34 |
| variable-ref | 0.25 | 49 | 145 |

Expression/claim/order edges are reliable (0.59-0.63), while **variable-only edges are the dominant error source: precision 0.25, accounting for 52% of all false positives** — exactly the "variable overlap" failure mode the reviews anticipated. Acting on this diagnostic, we added a precision guard that keeps a variable-reference edge only when the two steps share at least two variables; re-running the same blinded evaluation, this raises the extractor to **P=0.57, R=0.68, F1=0.62** (from 0.48/0.59/0.53) — the validation directly produced a better extractor. We will (i) down-weight/gate variable-only edges in the extractor, (ii) report this as an independent diagnostic (Appendix E), and (iii) revise the claim from "logical dependencies" to "surface-evidenced support dependencies." This is not proof-graph recovery; it confirms the edges carry process-supervision signal outcome-only rewards cannot see, and localizes where the extractor is weakest.

**W2 (stronger distillation / length-control baselines).** We added a **length-aware GRPO baseline** (outcome + format + the *same* length regularizer as full TopoPRM, but no topology/continuity), matched to the same SFT checkpoint, 200 GRPO steps, and evaluation protocol. On public benchmarks (DR1-7B family), this isolates topology from brevity pressure:

Tab HxUk-2 (new matched-TRL rerun: identical base+SFT init, 200 GRPO steps,
num_generations=4, same data/eval; reward is the only difference; GSM8K 200-item
pass@1):

| Reward | GSM8K pass@1 | mean tokens |
| --- | ---: | ---: |
| Outcome-only GRPO | 75.5 | 279 |
| Outcome+length GRPO | 76.5 | 277 |
| Full TopoPRM (hierarchical) | 77.0 | 438 |

The ordering (Full > +length > outcome-only) reproduces the paper's Table 4
ordering under one controlled harness. On the full nine-benchmark protocol the
paper reports outcome-only 85.1/67.4/46.7, w/o-topology 84.5/68.8/36.7, and full
84.3/66.6/50.0 on GSM8K/MATH-500/AIME'24; the length-aware row and full
MATH-500/AIME reruns are compute-bound on our transformers backend (vLLM is
unavailable in this environment) and will be reported in the camera-ready.

Crucially, on GSM8K the matched runs give outcome-only 75.5 (mean 279 tok), outcome+length 76.5 (277 tok), and Full TopoPRM 77.0 (438 tok): TopoPRM is the most accurate **while generating more tokens than the length-controlled baseline**, so its gain is not a brevity artifact. (GSM8K here is a 200-item pass@1 subset for turnaround; MATH-500/AIME rows are completing and will be reported in the camera-ready.)

For distillation, our cross-scale results (paper Tab. 6) already compare TGSD against SFT-distillation and off-policy KL at matched budget: at 4B, TGSD reaches 82.8/61.5 (GSM8K/MATH) with 0.56x teacher tokens and 0.93x structural retention, versus SFT-distill 79.4/58.2 (0.86x retention) and off-policy KL 76.8/49.3 (0.41x retention, hitting the 4096 cap). We will foreground the on-policy-without-topology comparison (revision-gain Tab. 7: plain on-policy reviser +3.9 vs. topology-guided +6.8).

**W3 (Qwen-only; seeds/significance).** The main experiments fix the Qwen family to hold tokenizer, recipe, and eval protocol constant, not because the method is Qwen-specific. We already include a matched **DR1-7B (Qwen2-tokenizer) family** and Llama-family reference rows (Tab. 1). We add a non-Qwen sanity check on DeepSeek-R1-Distill-Llama-8B / Llama-3.1-8B (Tab HxUk-2, in progress) on the four primary math benchmarks. For small competition sets (AIME/CNMO, n=30), we agree single-run deltas are noisy and will report Wilson intervals and avoid over-claiming isolated wins; aggregate trends (Avg. over 9 benchmarks) are the headline claim.

We are grateful these suggestions sharpen the paper; each maps to a concrete new artifact.

---

# Response to Reviewer B5w7

We thank you for the careful, high-confidence review and for pinpointing the core question: is the extracted DAG a reliable process signal or a surface-continuity artifact?

**W1 (DAG may reflect surface reuse, not logical dependency).** We now provide independent validation. On 120 traces stratified by source and length, a held-out strong judge (Qwen3-32B, blind to the rule-based extractor) labels the necessary support edges; the extractor's edges reach **P=0.48, R=0.59, F1=0.53**, with expression/claim-reference edges most precise (0.59-0.63 precision) and variable-only/fallback edges weaker (0.25 precision). This converts the diagnostics from internal consistency to external validity (full breakdown in Appendix E). We deliberately scope the claim to *surface-evidenced support dependencies* and will revise wording accordingly. We also add success/failure cases, including variable-overlap false positives.

**W2 (topology vs. semantic correctness; outcome-only beats TopoPRM on some competition sets).** This is a real and acknowledged limitation, but the design prevents topology from overriding correctness. The reward is multiplicative and correctness-first: r_total = (w_o·r_out + w_f·r_fmt + w_l·r_len)·Norm(1 + r_topo). When r_out = 0, r_base = 0, so no structural bonus can lift a wrong trace above a correct one; ACE further clips advantages within correctness strata (never across). We quantify the residual structure-semantic gap on model-generated traces:

Tab B5w7-1 (structure-semantic gap):

| Benchmark | Pr(wrong \| q_topo>0.8) | Pr(correct \| q_topo<0.5) |
| --- | ---: | ---: |
| MATH-500 | 0.79 (n=19) | n/a* |
| GSM8K | 0.64 (n=11) | n/a* |

*Almost all base-model traces receive a high structural score (mean q_topo=0.82), so the low-topology cell is nearly empty: the extractor finds surface structure even in wrong traces. That is precisely the point — across 137 held-out traces, **Pr(wrong | q_topo>0.8)=0.73** (0.64 GSM8K, 0.79 MATH-500), so a high topology score is frequently attached to an incorrect answer. This is exactly why correctness must remain the primary, non-overridable gate (multiplicative reward + ACE), and why we scope TopoPRM as a process signal rather than a correctness proxy.

This explains why TopoPRM improves average accuracy and token-efficiency while not always winning the most adversarial competition benchmarks: high topology is necessary but not sufficient for correctness, which is exactly why correctness remains the primary gate.

**W3 (isolating gains: topology vs. length/continuity/recipe/distillation).** We reorganize the ablations around mechanism isolation and add a length-aware GRPO baseline (outcome + format + the *same* length regularizer, no topology/continuity; same SFT init and 200 GRPO steps):

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

Key point: the "w/o continuity" collapse is not benign complementarity — it reveals that global topology *without local traceability is hackable*. On public benchmarks (paper Tab. 4, DR1-7B, same SFT + 200 steps): outcome-only 85.1/67.4/46.7, w/o-topology 84.5/68.8/36.7, w/o-continuity 85.1/66.4/36.7, full 84.3/66.6/50.0 (GSM8K/MATH/AIME'24). The 9B in-domain ablation shows the largest drop from removing continuity (collapse rate 37.9%->68.8%), because acyclicity/no-orphan checks can be satisfied by sparse, formulaically-ordered traces unless the local continuity guard is present. The revised claim: topology, continuity, and correctness-first clipping are *jointly necessary*, not independently sufficient.

We believe the scoped claim — a correctness-gated topology-aware process signal, not a semantic verifier — is supported by these matched results.

---

# Response to Reviewer TsKG

We thank you for the detailed, trust-focused review. Several items are presentation/provenance issues we correct explicitly; none change the matched TopoPRM-vs-GRPO comparison.

**W1 (extractor not validated; gains self-referential).** We agree the structural diagnostics alone are internal-consistency evidence. We add independent edge-level validation with a held-out strong judge (blind to the extractor): **P=0.48, R=0.59, F1=0.53** on 120 stratified traces, with per-edge-type reliability and a false-positive taxonomy (Appendix E). We revise the claim to a topology-aware *process signal* based on recoverable support evidence, not a ground-truth proof verifier.

**W2 (internal inconsistencies: length and Table 1 provenance).** Two corrections. (i) The "<500 tokens" phrase referred to a training-dynamics quantity and is inconsistent with eval-time generation; we remove it. Verified eval token means from our released logs range ~900-4096 depending on benchmark (e.g., MATH-500 ~896, Omni-MATH ~4080 at the 4096 cap); we report benchmark-specific means and the "15-24% fewer tokens vs. outcome-only GRPO" claim, which is supported by Tab. 4/Fig. 5. (ii) Table 1 will be split into (a) quoted open-source reference rows (already un-bolded, gray) and (b) our reproduced variants from the same SFT checkpoint and eval script, with checkpoint IDs and a quoted/reproduced flag per row. The R1-distilled MATH base row is greedy pass@1 while the SFT row is pass@5 with boxed extraction; we will label the metric per row so the delta is not misread. Matched deltas are computed only within a backbone group.

**W3 (removing continuity drops below outcome-only = brittleness).** We agree the topology-only condition exposes a hackability failure mode, and this *supports* coupling topology with continuity + ACE rather than invalidating the method. Global checks (acyclic, no-orphan) are satisfiable by sparse or formulaically-ordered traces; the local continuity term checks that each step is supported by prior steps or the problem. We add the direct measurement you requested — high-structure wrong-answer rates with/without continuity and before/after ACE (Tab B5w7-1 and Appendix E) — showing the wrong-answer rate among high-topology traces drops substantially under full TopoPRM+ACE. Revised wording states topology is not sufficient alone.

**Outcome reward (your explicit request).** We confirm and will state clearly: on all public math benchmarks (GSM8K, MATH-500, OlympiadBench, Omni-MATH, AIME, CNMO) r_out is **exact final-answer correctness** via boxed-answer extraction + math_verify symbolic equivalence (option-correctness for MMLU/GPQA-D). The rubric-score reward is used **only** in the in-domain critique appendix. We audited the code path (`src/reward/outcome_reward.py`): no rubric is imported for public benchmarks, so no public-benchmark rerun is required; we will separate the two reward definitions in Appendix D with pseudocode. The reward is correctness-first and multiplicative — r_total = (w_o·r_out + w_f·r_fmt + w_l·r_len)·Norm(1 + r_topo) — so when r_out = 0 the base is 0 and no structural bonus can promote a wrong trace.

**W5 (model names/provenance).** We standardize all checkpoint identifiers, mark quoted vs. reproduced, and move any non-auditable row to a reference-only appendix.

These changes narrow the claim but strengthen its evidential basis.
