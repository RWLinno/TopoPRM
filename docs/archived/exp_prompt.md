# TASK: Plan and Execute Experiments for TopoPRM (ACL Submission Rescue)

## Context
I have an ACL paper titled "TopoPRM: Verifiable Mathematical Reasoner via Topology-Aware 
Process Rewards and On-Policy Distillation". The experimental results are scattered and 
contain self-contradictions. I need you to (1) audit the current state, (2) plan a focused 
set of experiments, and (3) execute them.

## Repository Layout (fill in / verify)
- Training: `train/` uses HuggingFace TRL + LoRA on 8×A100-80GB
- Models: DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, Qwen2.5-7B, plus distilled 4B/2B/0.8B students
- Reward code: `reward/` contains DAG extractor, qtopo, qcont, hierarchical aggregation
- Eval: `eval/` contains transformers-backend unified protocol

## Core Narrative Pivot (IMPORTANT)
The paper should NOT claim SOTA accuracy. The claim should be:
"TopoPRM provides deterministic, auditable process rewards without human labels or 
 trained verifiers, achieving comparable re-ranking performance to trained PRMs 
 (Skywork/Math-Shepherd/Qwen2.5-Math-PRM) at 1-2 orders of magnitude lower cost, 
 while additionally improving token efficiency and structural auditability."

## Known Issues to Fix
1. Abstract claims "84.5% GSM8K with 7B" but Table 1 shows Qwen2.5-7B + TopoPRM = 83.8 
   GSM8K / 38.8 MATH-500 (WORSE than base 84.2/55.2). Resolve this inconsistency.
2. Many cells in tables contain "~" approximate values. Replace with real measurements 
   or remove.
3. 9B family: SFT-v2 (96.6 GSM8K) > TopoPRM variants. Investigate: is RL actually helping?
4. Qwen2.5-7B TopoPRM regression: either debug or move to "limitation/failure analysis".

## Your Plan Should Produce Exactly 3 Main Tables

### Table 1 (MAIN CLAIM): PRM Comparison for Best-of-N Re-ranking
Compare against: Skywork-PRM-1.5B/7B, Math-Shepherd-PRM-7B, RLHFlow-PRM-Mistral-8B, 
RLHFlow-PRM-Deepseek-8B, EurusPRM-Stage1/2, Qwen2.5-Math-PRM-7B/72B, Qwen2.5-Math-7B-PRM800K.
Benchmarks: GSM8K, MATH-500, OlympiadBench, Omni-MATH.
Metrics: pass@1, pass@5, maj@5, prm@5, #Tokens, F1.
Setting: k=5 candidates per problem, identical generator (Qwen2.5-Math-7B-Instruct or 
similar), PRM used only as re-ranker.
GOAL: TopoPRM ≥ median of trained PRMs despite NO training cost.

### Table 2 (ABLATION): Component Necessity on PUBLIC benchmarks (not in-domain)
Rows: full TopoPRM, w/o topology, w/o continuity, w/o stratified clipping, outcome-only GRPO, SFT baseline.
Columns: GSM8K, MATH-500, OlympiadBench, AIME'24 (optional).
Metric: pass@1 + token count.
GOAL: monotonic degradation when removing components.

### Table 3 (EFFICIENCY + STRUCTURAL AUDITING)
Two sub-panels:
(a) Acc/kToken + mean response length across configurations.
(b) DAG structural quality: Acyclic%, Orphan%, Direction-consistency, on generated traces.
GOAL: TopoPRM produces shorter + more structurally valid traces.

## Experiments to Execute (in priority order)

### Phase 1: Data Audit (no GPU needed, 1 hour)
1. Scan all log files and extract REAL numbers for every "~xx" cell in current tables.
2. Produce `audit.json` with: {table_id, row, column, current_value, true_value_or_missing}.
3. Flag any row where TopoPRM underperforms SFT baseline by >2 points → candidates for removal or debugging.

### Phase 2: Fill PRM Re-ranking Table (highest ROI, ~8 GPU-hours)
1. Generate k=5 candidates per problem for GSM8K (1319), MATH-500 (500), OlympiadBench 
   (500 subset), Omni-MATH (500 subset) using a fixed generator.
2. Score each candidate with:
   - TopoPRM (our qtopo * qcont-based score)
   - Each baseline PRM (call them as HuggingFace models)
3. Compute pass@1, prm@5 (best-of-5 by PRM), maj@5, token counts.
4. Produce `table1_prm_comparison.csv`.

### Phase 3: Clean Ablation (6 GPU-hours)
Run on SAME SFT checkpoint, SAME 200 GRPO steps:
- full TopoPRM
- w/o qtopo (set α=0 in Eq.1, keep qcont)
- w/o qcont (α=1)
- w/o stratified clipping (linear aggregation)
- outcome-only (wo=1, rest=0)
Evaluate on GSM8K, MATH-500, OlympiadBench.
Produce `table2_ablation.csv`.

### Phase 4: Efficiency Table (2 GPU-hours)
From Phase 3 outputs, compute Acc/kToken and DAG quality on generated traces.
Use the DAG extractor in reward/ to score all outputs.
Produce `table3_efficiency.csv`.

### Phase 5: Sanity Checks (2 GPU-hours)
1. Reward-hacking probe: count samples where qtopo ≥ 0.9 but rout = 0. 
   Verify stratified clipping keeps their reward ≤ 0.
2. Reproducibility: re-run one configuration with seed ∈ {0, 1, 2}, report std.

## Deliverables Cursor Must Produce
1. `audit_report.md`: summary of current-state issues with line-pointers into existing tables.
2. `experiment_plan.md`: concrete command list with estimated GPU-hours, ordered by priority.
3. `run_phase_X.sh` scripts for each phase.
4. `results/` directory with CSV output per table.
5. `paper_tables.tex`: three clean LaTeX tables ready to paste into the paper.
6. `known_failures.md`: list of configurations that regressed (e.g., Qwen2.5-7B), with 
   a suggested "Limitations" paragraph.

## Constraints
- Total compute budget: 20 GPU-hours on 8×A100-80GB.
- Evaluation protocol MUST be identical across all rows (same chat template, same answer 
  extractor, same max_new_tokens). Log the evaluation command in each CSV row.
- No "~" approximate numbers in the final tables. If a cell cannot be measured, use "—".
- If any experiment reveals TopoPRM truly underperforms, document it honestly in 
  known_failures.md rather than hiding.

## Start by:
1. Reading the current paper draft in `paper/main.tex`@weilinruan/TopoPRM 
2. Reading the current tables and cross-referencing with logs in `logs/`
3. Producing `audit_report.md` and `experiment_plan.md` BEFORE running any GPU jobs
4. Asking me for approval of the plan before Phase 2