# TopoPRM: Topology-Aware Process Rewards for Verifiable Mathematical Reasoning

## Title

**Chinese**: TopoPRM: 基于拓扑感知过程奖励的可验证数学推理框架

**English**: Topology-Aware Process Rewards for Verifiable Mathematical Reasoning

---

## Abstract

Reinforcement learning with verifiable rewards (RLVR) improves final-answer accuracy in mathematical reasoning, yet the optimisation signal is typically dominated by outcome correctness, leaving intermediate reasoning quality only weakly constrained. Models may therefore reach correct answers through unsupported deductions, circular dependencies, or unnecessarily verbose traces.

We propose **TopoPRM**, a verifiable process-reward framework that introduces topology-aware structural supervision without learned reward models or human annotations. For each reasoning trace, a deterministic parser constructs a directed acyclic graph (DAG) over inter-step dependencies. Two process signals are then computed: a **topological structure reward** that penalises cycles, orphan conclusions, and directional inconsistency; and a **continuity reward** that verifies each step's traceability to prior claims or problem givens. These signals are combined with outcome, format, and length rewards through **Stratified Clipping Advantage Estimation** (SCAE), an accuracy-first stratified shaping strategy for GRPO training that prevents process rewards from favouring structurally valid but factually incorrect traces.

On 7,595 private mathematical critique problems, TopoPRM improves critique accuracy by +13.4 points over outcome-only GRPO while reducing average response length by 11%. A reverse-KL distillation stage transfers these gains to a Qwen3-8B student that achieves 82.5% on GSM8K and 59.8% on MATH-500, competitive with models of comparable scale.

---

## 1. Background and Challenges

### 1.1 Limitations of Outcome-Only RLVR

RLVR has become the standard paradigm for improving LLM mathematical reasoning. The core idea: programmatic checks verify whether the model's answer matches ground truth, producing binary reward signals for GRPO policy optimization.

However, outcome-only supervision creates three systematic failure modes:

1. **Unsupported Deductions**: Models may skip critical reasoning steps, producing correct answers without logical support.
2. **Circular Dependencies**: Reasoning chains may contain A->B->C->A circular references -- correct answer but unsound structure.
3. **Verbose Traces**: Models may generate excessive exploration branches and repeated derivations, increasing inference cost without quality improvement.

### 1.2 Limitations of Existing Process Reward Models (PRM)

Existing process reward methods (Math-Shepherd, PRM800K) rely on human-annotated step-level correctness labels to train neural reward models:

- **High annotation cost**: Each reasoning step requires expert judgment.
- **Poor generalization**: Trained PRMs generalize poorly to out-of-distribution reasoning patterns.
- **Non-verifiable**: Neural reward models are black boxes with no reliability guarantees.

### 1.3 Core Challenge

How to provide effective process-level supervision for RLVR training without additional annotation costs or non-verifiable neural reward models?

---

## 2. Solution Overview

TopoPRM proposes a fully deterministic, verifiable process reward framework with three core components:

1. **Multi-source Reward Module**: DAG extractor (rule-based parsing -> DAG), topological structure reward (global graph validity), continuity reward (local step traceability), outcome/format/length rewards.

2. **SFT -> GRPO Training Pipeline**: SFT cold-start (structured output interface), GRPO + SCAE (accuracy-first stratified reward shaping), dynamic weight adjustment.

3. **Reverse-KL Distillation**: Teacher generates high-quality traces, quality filtering (R_total > tau_d), mode-seeking distillation to smaller student.

---

## 3. Detailed Method

### 3.1 DAG Extraction

For each reasoning trace, the deterministic parser performs four steps:

1. **Step Segmentation**: Split trace into ordered step sequence using textual markers and line-level boundary heuristics.
2. **Expression Mining**: Extract mathematical expressions, variable assignments, and propositional claims via regex templates.
3. **Type Classification**: Classify each step into 7 semantic types: definition, derivation, computation, conclusion, auxiliary, substitution, case analysis.
4. **Edge Construction**: Instantiate three edge types:
   - **Solid edges**: Sequential-order dependencies between consecutive steps
   - **Virtual edges**: Expression-reuse dependencies (main supervision signal)
   - **Double-barrier edges**: Structural connectors for derivation/conclusion steps without explicit evidence

The entire module is non-parametric.

### 3.2 Topological Structure Reward R_topo

Quantifies global validity of the extracted DAG:

R_topo(G) = lambda_b * I[|V|>0] + lambda_a * I[acyclic] + lambda_o * I[rho_orphan=0] + lambda_d * delta_virtual + lambda_s * sigma_solid + lambda_k * kappa_virtual

### 3.3 Continuity Reward R_cont

Evaluates local traceability across reasoning steps:

R_cont(c) = 1.0 if eta(c)=1, else 0.8*eta(c)

where eta(c) = (1/T) * sum_i I[supported(r_i, r_{<i}, q)]

### 3.4 SCAE (Stratified Clipping Advantage Estimation)

Prevents process rewards from favouring structurally valid but factually incorrect traces:

1. **Stratification**: Partition completions into correct stratum C and wrong stratum W based on outcome reward threshold tau=0.66
2. **Standardization**: Normalize composite rewards to z-scores within each stratum
3. **Asymmetric clipping**: Correct stratum clip to [0, 1.5], wrong stratum clip to [-1.5, 0]

### 3.5 Reward Aggregation Strategies

| Strategy | Formula | Properties |
|----------|---------|------------|
| Linear weighted | R = sum_m w_m R_m | Simple, direct |
| Clipped linear | R = clip(sum, 0, 1) | Prevents extremes |
| Hierarchical | R = R_base * (1+a*R_topo) * (1+b*R_cont) | Topo/cont as gain factors |
| Gated lexicographic | R = R_out + eps*R_process, eps<<1 | Mathematical guarantee: outcome rank preserved |

### 3.6 Topology-Verified Self-Distillation (TVSD)

Plain Reverse-KL distillation failed in our pilot: only 0.4% of 32B teacher
traces emitted a closed `<answer>` tag, so the student never learned to stop.
We replace it with **Topology-Verified Self-Distillation (TVSD)**, a compact
two-phase procedure that densifies a scalar outcome reward into per-token
supervision using our own deterministic topology / continuity diagnostics
(Eq. R_topo, R_cont) — with no external teacher or human step-level
annotation.

**Phase III-A: Topology-Conditioned Self-Refinement.** A single model
alternates between generator and reviser. For each problem x we sample y_init
on-policy, compute (r_out, r_topo, r_cont) on y_init, and build a
topology-aware revision prompt P_r via a 2x2 dispatch over
(correctness, structure):

| r_out | r_topo | P_r |
|-------|--------|------|
| 1 | >=0.5 | "Rephrase this correct critique more concisely." |
| 1 | <0.5  | "Correct answer but step k has no dependency; rewrite with explicit references." |
| 0 | >=0.5 | "Well-structured but final verdict wrong, reconsider." |
| 0 | <0.5  | "Wait, this is not correct, let me start over." |

where k is drawn from the orphan-conclusion set of the extracted DAG. The
refinement loss combines revision NLL and generation NLL
(L_revision + L_generation).

**Phase III-B: Topology-Gated On-Policy Distillation.** The refined model is
frozen as a topology-verified teacher. Student samples y on-policy; teacher
computes per-token distribution conditioned on (x, y, P_r, y_{<t}). Loss:

L_TVSD = E_{x, y ~ pi_theta(.|x)} sum_t KL( pi_theta(.|x, y_{<t}) || pi_ref(.|x, y, P_r, y_{<t}) )

**Why topology verification matters.** Standard on-policy distillation
densifies a scalar outcome through per-token targets at uniformly distributed
positions. Our topology-verified P_r steers the teacher-student divergence
toward tokens that are *structurally suspect* — orphan conclusions, acyclicity
violations, continuity breaks — positions that no binary-reward teacher can
identify. This converts our DAG diagnostics into dense per-token supervision
while staying entirely within the verifiable-reward paradigm.

**Student compression (<=4B).** Because Phase III-B student rollouts are
on-policy and format-closed, we can compress into much smaller students
without the stopping failure seen with plain Reverse-KL. Student is
initialised from Qwen3.5-4B / 2B / 0.8B, teacher stays the 9B refined model.

---

## 4. Experiment Plan

### 4.1 Benchmarks

| Category | Benchmark | Samples | Description |
|----------|-----------|---------|-------------|
| Private | Middle School Math Critique | 2,181 | Chinese, medium difficulty |
| Private | High School Math Critique | 5,414 | Chinese, higher difficulty |
| Public-Math | GSM8K | 1,319 | Elementary math |
| Public-Math | MATH-500 | 500 | Competition math |
| Public-Math | OlympiadBench | - | Olympiad level |
| Public-Math | Omni-MATH | - | Comprehensive math |
| Public-Math | AIME 2024 | 30 | AMC/AIME competition |
| Public-Math | CNMO 2024 | - | Chinese Math Olympiad |
| Public-Math | LiveCodeBench | - | Code + math |
| Public-General | MMLU | - | Multi-discipline knowledge |
| Public-General | GPQA-Diamond | - | Graduate-level QA |

### 4.2 Metrics

- **Private**: Score-Acc, Error-F1, Format%, Avg-Len
- **Public**: pass@1, pass@k, maj@k, prm@k, F1, #Tokens (k in {1,5})

### 4.3 Model Matrix (updated 2026-04-17 - TVSD pivot)

| Model | Params | Training | Status |
|-------|--------|----------|--------|
| Qwen3-32B | 32B | Base / SFT / GRPO variants / TopoPRM | Legacy checkpoints (deprioritized) |
| Qwen3.5-9B | 9B | Base / SFT / GRPO (outcome/no-topo/no-cont) / TopoPRM (hier, gated) | **v2 measured with chat template** |
| Qwen3.5-9B + TVSD | 9B | SRT Phase III-A + OPSD Phase III-B | To train |
| Qwen2.5-7B | 7B | Base / TopoPRM (hier) | Measured |
| Qwen3.5-4B + TVSD | 4B | TVSD (teacher = 9B SRT) | Primary student target |
| Qwen3.5-2B + TVSD | 2B | TVSD | Aggressive compression |
| Qwen3.5-0.8B + TVSD | 0.8B | TVSD | Extreme compression |

**Breakthrough from v2 evaluation protocol:** with chat-template prompting
and sft-style system message (matching training format), our SFT 9B variant
achieves 96-97% GSM8K (vs. 88% under raw-text prompt). This closes the
SFT/base gap and unlocks TopoPRM's advantage: best MATH-500 accuracy
(55.4%, +0.4 over base) at 3x shorter trace length.

**Distillation pivot.** We deprecate reverse-KL into Qwen3-8B because:
1. 32B teacher tokenizer mismatch + truncation produced only 0.4% closed
   `<answer>` traces, so students never learned to stop (every sample hit
   max_new_tokens = 2048).
2. 8B is not a useful deployment target.

Our new route is TVSD (Section 3.6): on-policy self-distillation
with topology-aware revision prompts, targeting Qwen3.5-{4B, 2B, 0.8B}.

### 4.4 Ablation Studies

1. **Reward component ablation**: TopoPRM (full) vs w/o Topology vs w/o Continuity vs Outcome Only
2. **Aggregation strategy ablation**: Linear vs Clipped vs Hierarchical vs Gated
3. **Cross-scale ablation**: Component contributions across 32B / 9B / 7B
4. **Distillation ablation**: Effect of quality threshold tau_d
5. **Accuracy-length trade-off**: Acc/kTok efficiency across configurations

---

## 5. Expected Contributions

1. **Verifiable process rewards**: First to use reasoning trace topological properties (acyclicity, connectivity, directional consistency) as deterministic process reward signals in RLVR, without human annotations or neural reward models.

2. **Accuracy-first reward shaping**: SCAE mechanism theoretically guarantees process rewards cannot favour structurally valid but factually incorrect traces.

3. **End-to-end training recipe**: Complete pipeline from SFT cold-start to GRPO training to reverse-KL distillation, all components as plugins compatible with standard GRPO frameworks.

4. **Comprehensive empirical validation**: Validated on private Chinese math critique and 9 public benchmarks, covering 0.5B-32B model scales.

---

## 6. Timeline

| Phase | Content | Status |
|-------|---------|--------|
| Phase 0 | Reward invariant checks | Done |
| Phase 1 | Archive old experiments + wandb cleanup | Done |
| Phase 2 | Data augmentation pipeline | Done |
| Phase 3 | Main + student model training | Awaiting GPU |
| Phase 4 | Unified 9-benchmark evaluation | In Progress |
| Phase 5 | Result sync + paper table backfill | Awaiting Phase 4 |
| Phase 6 | Paper polish + figure drawing | In Progress |
| Phase 7 | Camera-ready submission | Awaiting all |
