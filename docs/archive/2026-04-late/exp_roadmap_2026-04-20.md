# TopoPRM Experiment Roadmap (2026-04-20)

Follow-up experiments, captured as a concrete plan but **not executed** in
the 2026-04-20 sprint. Each item lists the motivation, concrete scripts to
write / extend, and the acceptance signal that would let us move it into
the main paper.

## Context

After the 2026-04-20 sprint we have:
- Unified v3 benchmark re-run started on GPU 1/2/7 with corrected
  `max_new_tokens` (long-CoT = 2560, medium = 1536, short = 512 with
  MMLU subset).
- A best-of (pass@1, maj@5, pass@5) view for the main paper table
  (`docs/rft_bestof_ours.csv`).
- A concrete checkpoint triage report
  (`docs/checkpoints_triage_2026-04-20.md`) and reward collapse
  diagnosis (`docs/reward_collapse_diagnosis_2026-04-20.md`).
- A rewritten distillation narrative: token-efficiency + structural
  retention, **not** accuracy parity with the teacher.

The roadmap below is what we would run **next** to tighten the paper.

## R1. Continue TopoPRM GRPO from `checkpoint-79`

**Motivation.** 79 steps is warm-up; `frac_reward_zero_std`$\approx\!0.36$
and `reward_std` narrowing to ~0.09 suggest the policy can still move if
we give it more budget plus a variance floor.

**What to implement.**
1. Warm-start from `output/grpo_hierarchical_qwen35_9b_mcl4096/v2-20260407-162048/checkpoint-79` and run **300--500 additional steps**.
2. Add `reward_std_floor` regulariser to the GRPO trainer: when a group's
   `reward_std < 0.02`, skip the group or add Gaussian noise to scoring.
3. Anneal the gated-variant threshold: start at `tau=0.3` in the first
   200 steps, then linearly climb to `tau=0.5` by step 500.
4. Maintain a separate schedule for the gated-variant (which is
   effectively SFT+noise at `checkpoint-79`; see
   `docs/reward_collapse_diagnosis_2026-04-20.md`).

**Files to add / touch.**
- `configs/grpo_continue_topoprm_hier_9b.yaml`
- `configs/grpo_continue_topoprm_gated_9b.yaml`
- `src/reward/*.py` (add `reward_std_floor`; keep behaviour opt-in)
- `scripts/run_grpo_continue.sh`

**Acceptance signal.**
- `reward_mean` on held-out prompts increases by $\geq$0.05 between the
  continuation start and end.
- AIME~2024 pass@1 for `topoprm_hier_9b_cont` beats
  `topoprm_hier_9b_v3` by $\geq$5~pp.

## R2. Finish TVSD end-to-end pipeline (4B / 2B / 0.8B students)

**Motivation.** The paper narrative (`sections/3_method.tex`,
`sections/4_experiments.tex`) now commits to TVSD as the compression
route, but we only have the SFT-distill 4B baseline measured. The TVSD
Phase III-B rollout + KL trainer needs to actually run and produce
checkpoints.

**What to implement.**
1. Close out `scripts/rollout_srt.py` (Phase III-A) and run to produce
   refined verifier model $\pi_\mathrm{ref}$ on top of the 9B SFT
   checkpoint.
2. Close out `src/distill/opsd_trainer.py` (Phase III-B). Required
   fixes we already know of: topology-aware $P_r$ injection into teacher
   forward pass, correct KL masking over student tokens only.
3. Train three TVSD students end-to-end (Qwen3.5-4B / 2B / 0.8B).

**Files to add / touch.**
- `scripts/rollout_srt.py` (finish)
- `src/distill/opsd_trainer.py` (finish)
- `configs/tvsd_student_{4b,2b,0p8b}.yaml` (already stubs)
- `scripts/run_tvsd_students.sh`

**Acceptance signal.** For the 4B TVSD student:
- GSM8K pass@1 within 3~pp of 9B teacher
- avg_tokens $\leq$60\% of teacher
- Structural retention: Acyclic\%/No-Orphan\%/Edge-Keep\% all within 5~pp
  of teacher.

## R3. Structural-retention evaluator

**Motivation.** Our compression claim rests on structural retention
(`Acyclic%, No-Orphan%, Edge-Keep%` vs teacher). We have these signals
during training but not at inference time on public benchmarks.

**What to implement.** Add `scripts/eval_dag_structure.py` that:
- Takes a model + an eval benchmark,
- Generates N samples,
- Runs the deterministic DAG extractor `src/reward/topo_reward.py`,
- Reports Acyclic\%, No-Orphan\%, Edge-Keep\%, plus mean DAG depth and
  nodes-per-trace.

Output is a JSON per model / benchmark, consumable by
`scripts/fill_rft_csv.py` and by a new `tables/structural_metrics.tex`
companion column.

**Acceptance signal.**
- Numbers for teacher and at least one TVSD student are written to
  `tables/structural_metrics.tex` automatically.

## R4. Pareto plot: accuracy vs tokens

**Motivation.** The compression claim is easier to communicate with a
scatter plot than with tables. Each model becomes one dot; x=avg_tokens,
y=pass@1; we draw a Pareto frontier.

**What to implement.** `scripts/make_pareto_plot.py` reads
`output/eval/*_metrics.json`, groups by model, and emits a PDF with one
panel per benchmark (GSM8K / MATH-500 / Olympiad / Omni-MATH).

**Acceptance signal.**
- `topoprm_paper/figures/pareto_accuracy_tokens.pdf` referenced in
  Section~\ref{sec:rq4} (Experiments).

## R5. Secondary: honest AIME/CNMO recovery attempts

**Motivation.** AIME/CNMO is where the current TopoPRM hurts; rather
than hide it we commit to a concrete recovery path in
`sections/6_appendix.tex:Limitations`.

**What to implement.**
- After R1 completes, re-eval on AIME 2024/2025 and CNMO 2024 with
  `max_new_tokens=2560`, `num_samples=8`, `temperature=0.9`.
- Report `pass@1/pass@5/maj@5` in a new appendix table.

**Acceptance signal.**
- The continued-training checkpoint at least matches SFT on AIME pass@1;
  we can then replace the current stalled numbers in the main table.

## Scheduling note

- R1 and R3 are independent and can run in parallel.
- R2 depends on R3 (structural eval is needed to verify TVSD retention).
- R4 can be produced at any point once at least two models are in the
  best-of CSV.
- R5 requires R1 to finish.

## Owner checklist

- [ ] R1: continue TopoPRM GRPO (hier + gated) with reward_std_floor
- [ ] R2: finish TVSD Phase III-A / III-B end-to-end, train 4B/2B/0.8B
- [ ] R3: `scripts/eval_dag_structure.py` + `tables/structural_metrics.tex`
- [ ] R4: `scripts/make_pareto_plot.py` + `figures/pareto_accuracy_tokens.pdf`
- [ ] R5: AIME/CNMO honest recovery eval + appendix table
