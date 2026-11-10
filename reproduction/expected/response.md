# Response to Reviewer Lxds

We appreciate your assessment that recovering reasoning structure is a promising direction, particularly the connection between inspectable rewards and localized teacher feedback. The new matched comparisons show more compact reasoning under ACE and a non-AIME accuracy benefit from graph-guided distillation.

**W1. How graph information improves training.** TopoPRM uses the same support relations to assign structural credit during RL and locate teacher feedback during distillation. Please refer to Tables R1–R2 in our response to Reviewer ZYfo. Across seeds 42/123, ACE improves mean non-AIME accuracy over correctness clipping by 1.09 points with 9.0% shorter responses. Graph-guided distillation then improves this accuracy measure over local guidance by 2.05 points with 9.6% shorter responses. These comparisons connect the graph's two training roles to student performance under matched settings.

**W2. Accuracy and efficiency beyond AIME.** The distillation-matched comparison in Table 2 shows shorter reasoning across GSM8K, MATH-500 and OlympiadBench, while additional AIME solutions supply most of its macro-accuracy increment. The new R2 comparison extends the accuracy evidence beyond AIME: graph guidance reaches a three-task mean of 57.46%, versus 55.42% for local guidance and 55.72% for generic correction. R2 reports every benchmark alongside mean generated length, so the accuracy and efficiency contributions can be assessed together.

**W3/W5. Inspectable support relations beyond the scalar score.** The edge ledger in §2.1 records the supporting premise, conclusion, relation type and triggering text. For example, conclusion-support profiles $(1,0.3)$ and $(0.3,1)$ both score $10/13$, yet locate missing support at different conclusions. TGD retains this distinction when requesting feedback. The relation audit in §3.4 measures extraction fidelity against model references; Appendix D.2 establishes invariance to the surface changes removed by normalization. These evaluate the recovered structure and its normalization, respectively. We will make their connection to the inspectable ledger explicit, while distinguishing them from a user-rated interpretability measure or a dependency-changing intervention.

**W4. Teacher supervision on unchanged text.** Eq. (5) transfers the teacher's token distribution. At the same prefix, its two distributions are

$$
p_t=\pi_\theta(\cdot\mid x,\widetilde y_{<t}),\qquad
q_t=\pi_T(\cdot\mid x,y^\star,P_r,\widetilde y_{<t}).
$$

The teacher additionally sees the original trace and diagnosis, so $D_{\mathrm{KL}}(p_t\|q_t)$ can provide supervision even when $\widetilde y=y^\star$. This is consistent with learning from teacher feedback on student-generated sequences [1]. Table R4 in our response to Reviewer eXXt separates text changes, independently judged repairs and supervision tokens. The preservation-prompt follow-up there further improves revision fidelity. These measurements explain the supervision available to the student without equating every accepted response with a repair.

**W5. Transfer across backbones.** Appendix E shows response compression on the Llama tasks and a backbone- and decoder-dependent accuracy–length balance. We will discuss these results at the task and decoding-setting level. The new R1/R2 comparisons provide the corresponding controlled evidence for structural credit and feedback localization. Code access and experiment records are addressed under Q4 in our response to Reviewer eXXt.

**References**

[1] [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649), 2024.

# Response to Reviewer ZYfo

Thank you for highlighting the value of using the same dependency representation for reward assignment and localized teacher feedback. The graph records which premises support each conclusion, including dependencies across non-adjacent steps. Two traces can receive the same scalar score while lacking support at different conclusions; the graph preserves distinct targets for teacher feedback. ACE uses the recovered support structure to assign credit within correctness groups, and TGD directs the teacher to the identified gaps. In the matched TGD comparison, graph guidance improves non-AIME accuracy by 2.05 percentage points over local guidance with 9.6% shorter responses (Table R2).

**W1/Q1. Separating structure from credit assignment.** Table R1 uses Graph + ACE as the shared reference. With graph extraction fixed, correctness clipping and SCAE [2] test the credit rule; with ACE fixed, local support and rewiring test the structural source. Rewiring preserves node/edge counts and forward direction. All arms share a new 128-update SFT checkpoint, data, reward weights, completion cap and GRPO [1] budget (200 updates; seeds 42/123), and are evaluated before TGD.

**Table R1. Structure and credit controls.** Pass@1 (%) on GSM8K/MATH-500/OlympiadBench/AIME2024 (1,319/500/675/30 questions). Avg3 excludes AIME; Tokens is the four-task mean visible-token count. Bold marks the best Avg3 and Tokens per seed; prose uses the equal mean of both seeds.

| Structure | Credit | Seed | GSM | MATH | Olymp. | AIME | Avg3 ↑ | Tokens ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Graph | ACE | 42 | 84.15 | 57.80 | 19.85 | 6.67 | **53.94** | **445.1** |
| Graph | Clipping | 42 | 82.94 | 52.20 | 20.44 | 6.67 | 51.86 | 498.7 |
| Graph | SCAE | 42 | 82.71 | 56.40 | 21.33 | 3.33 | 53.48 | 489.2 |
| Local support | ACE | 42 | 84.00 | 55.80 | 21.04 | 3.33 | 53.61 | 461.7 |
| Rewired graph | ACE | 42 | 84.08 | 55.20 | 20.74 | 3.33 | 53.34 | 472.1 |
| Graph | ACE | 123 | 82.94 | 55.40 | 19.85 | 3.33 | 52.73 | 453.6 |
| Graph | Clipping | 123 | 84.46 | 53.00 | 20.44 | 3.33 | 52.63 | 489.1 |
| Graph | SCAE | 123 | 84.31 | 56.20 | 21.19 | 3.33 | **53.90** | **451.0** |
| Local support | ACE | 123 | 83.85 | 55.20 | 19.56 | 6.67 | 52.87 | 495.5 |
| Rewired graph | ACE | 123 | 82.94 | 58.00 | 18.37 | 3.33 | 53.10 | 452.0 |

**W1/Q1. Fixed graph source: the credit rule.** Averaged equally over seeds 42/123, ACE raises Avg3 over clipping by 1.09 points with 9.0% shorter responses. ACE and SCAE average 53.33%/53.69% Avg3 and 449.4/470.1 tokens, respectively: ACE produces the shortest mean responses among the three credit rules.

**W1/Q1. Fixed ACE: the structural source.** Graph, local and rewired inputs average 53.33%/53.24%/53.22% Avg3 and 449.4/478.6/462.1 tokens across the two seeds. The extracted graph combines the highest mean accuracy with the shortest mean responses in this comparison. Rewiring changes the applied loss coefficient on 44/800 and 43/800 fixed rollouts, directly showing that edge connections affect the RL signal. We will clarify this connection alongside Eq. (1) and Appendix B.2.

**W1/W2/Q2. Graph-guided teacher feedback.** We repeated TGD from one Stage-II checkpoint with the same teacher, question order, acceptance gate, 512 attempts and 64 updates per arm. Accepted examples and supervision tokens are reported in Table R4 of our response to Reviewer eXXt.

**Table R2. Matched TGD guidance.** Same benchmarks and metrics as Table R1.

| Guidance | GSM | MATH | Olymp. | AIME | Avg3 ↑ | Tokens ↓ |
|---|---:|---:|---:|---:|---:|---:|
| **Graph** | **85.75** | **62.20** | **24.44** | **6.67** | **57.46** | 467.2 |
| Local continuity | 85.22 | 59.40 | 21.63 | 3.33 | 55.42 | 516.8 |
| Generic correction | 85.60 | 59.20 | 22.37 | 3.33 | 55.72 | **451.9** |

Graph guidance improves the non-AIME mean over local guidance by **2.05 points with 9.6% shorter responses**. Against generic correction, it gains 1.74 points with 3.4% longer responses. Graph guidance has the highest Avg3 of the three sources and improves both accuracy and length over local guidance. Appendix E's backbone and decoding comparisons are discussed under W5 in our response to Reviewer Lxds.

**Q2. From diagnosis to teacher output.** On 512 fixed traces with a shared preservation prompt, graph/local instructions differ on 239 inputs; target steps differ on 23/479 jointly localized traces, and outputs differ on 8/512 inputs. This measures the teacher's response to a fixed solution, while online TGD also follows the student's evolving trajectories. Please refer to Table R4 in our response to Reviewer eXXt for repair and supervision counts.

**References**

[1] [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), 2024.
[2] Graph Reasoning Paradigm, 2026 (SCAE); see Reference [1] in our response to Reviewer eXXt.

# Response to Reviewer eXXt

Your review recognizes two strengths: the unchanged inference interface and the controlled TGD comparison. Thank you for asking us to make the credit estimator and the source of teacher supervision equally explicit.

**W1/Q2. Structural credit beyond SCAE and clipping.** ACE adds within-correctness structural credit to the standardized total reward, then bounds the coefficient by correctness. Let $c_i\in\{0,1\}$, $\Delta u_i=u_i-\bar u_{\{j:c_j=c_i\}}$, $z_i=(r_i-\bar r)/(s_r+10^{-4})$, and $h_i=\max(\Delta u_i,0)$ if correct, $\min(\Delta u_i,0)$ otherwise. Here $s_r$ is the sample SD. With unit bounds,

$$
A_i^{\rm ACE}=\Pi_{I(c_i)}(z_i+h_i),\quad
I(1)=[0,1],\ I(0)=[-1,0].
$$

For the same auxiliary signal, SCAE [1] uses $A_i^{\rm SCAE}=c_i-\bar c+h_i$, whereas correctness clipping uses $\Pi_{I(c_i)}(z_i)$. Two correct tied-return traces with $u=(0.8,0.2)$ receive ACE/SCAE credit $(0.3,0)$, versus $(0,0)$ from clipping. For $c=(1,0)$ and $r=(1,0.4)$, both strata are singletons: SCAE gives $(0.5,-0.5)$, while ACE gives approximately $(0.707,-0.707)$. Thus tied rewards expose the structural credit, while mixed correctness distinguishes the baselines. Projection bounds the final coefficient.

The saturation threshold in Appendix B.3 is $b_i(1+u_i)\geq1$, where $b_i=\max(r_{{\rm base},i},\epsilon)$. ACE retains the pre-cap $u_i$. Table R3 measures its effect on the training coefficients.

**Table R3. Same-rollout credit diagnostics.** Each seed has 400 groups/800 rollouts from its ACE run. Ties use tolerance $10^{-6}$.

| Diagnostic | Seed 42 | Seed 123 |
|---|---:|---:|
| Tied groups retaining nonzero ACE | 86/174 | 89/163 |
| Coefficients differing from clipping | 189/800 | 220/800 |
| Coefficients differing from SCAE | 349/800 | 341/800 |
| Coefficients differing from uncapped ACE | 294/800 | 268/800 |

Recomputed and applied coefficients agree within $1.20\times10^{-7}$. Please refer to Table R1 in our response to Reviewer ZYfo for downstream ACE/clipping/SCAE comparisons. Uncapped ACE removes the scalar-reward cap while retaining coefficient projection; this is an offline coefficient comparison.

**W1/Q1. Structural information in reward and feedback.** Table R1 in our response to Reviewer ZYfo isolates credit rules and structural sources before TGD. R2 uses a common Stage-II checkpoint to compare graph, local and generic guidance. Graph guidance gains 2.05 non-AIME points over local guidance with 9.6% fewer tokens. The fixed data, reward weights, caps and update budgets make these comparisons informative about the graph's two training roles.

**W2/Q3. Revision quality and supervision volume.** Table R4 reports a new independent breakdown of online TGD supervision. A blinded 27B model judged problem/original/revised triples without arm, gate or extractor labels. Repairs require an affirmative independent judgment.

**Table R4. Online TGD supervision.** Each arm has 512 attempts and 64 updates. Repairs use accepted changed texts; answer changes use all attempts.

| Metric | Graph | Local | Generic |
|---|---:|---:|---:|
| Accepted | 324 | 305 | 343 |
| Unchanged / changed, accepted | 238 / 86 | 179 / 126 | 327 / 16 |
| Answer corrected / regressed | 1 / 4 | 2 / 3 | 2 / 0 |
| Support repairs / accepted changed | 2 / 86 | 3 / 126 | 3 / 16 |
| Accepted supervision tokens | 38,085 | 33,478 | 45,670 |

In a subsequent fixed-input diagnostic, a preservation prompt reduced graph-guided accepted outputs flagged as introducing errors from 79 to 8 and answer regressions from 4 to 0. Graph/local/generic accepted 358/358/363 outputs (29/30/6 changed), totaling 49,316/49,371/50,828 output tokens, without student updates.

These measurements clarify why useful TGD supervision need not produce a large edit rate: the teacher can supply different token probabilities under the structural diagnosis even when its chosen text is unchanged (see our response to Reviewer Lxds). We will add the independent repair counts and actual supervision volume to Appendix C.3.

**W2/Q4. Reproduction artifacts and training cost.** The [anonymous code](https://anonymous.4open.science/r/TopoPRM-B47E/) is accessible and contains the ACE/TGD implementations, pinned dependencies and stage launchers. For the added runs, the manifest records 19,466 training examples (11,994 math, 7,472 GSM), answer validation, deduplication and normalized exact-question overlap filtering. Successful SFT/ten RL/three TGD runs used 0.23/21.39/19.85 GPU-hours, excluding evaluation and retries. R1 supplies two training seeds for the Stage-II comparison; the submitted Full/w/o ACE experiment evaluates the complete pipeline. We will add the executed ablation code, R1–R4 run manifests, per-item outputs and scoring commands to the artifact. Extraction and transfer are addressed under W3/W5 in our response to Reviewer Lxds.

**References**

[1] [Graph Reasoning Paradigm](https://arxiv.org/abs/2601.12995), 2026.
