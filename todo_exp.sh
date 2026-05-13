#!/usr/bin/env bash
# ============================================================================
# TopoPRM Submission Experiment Runbook
# ----------------------------------------------------------------------------
# This runbook produces all numbers required by the paper, in the exact order
# they appear in the manuscript. Each phase is independent and uses nohup so
# you can launch and observe via the companion log files.
#
# Conventions:
#   - All training/eval outputs live under ./output and ./logs (per phase).
#   - Public benchmarks: GSM8K, MATH-500, Olympiad, Omni-MATH, AIME 2024,
#     AIME 2025, CNMO 2024, MMLU, GPQA-Diamond.
#   - Backbone families: Qwen3.5-9B (primary), Qwen2.5-7B, DeepSeek-R1-Distill-Qwen-7B.
#   - Distillation students: 4B, 2B, 0.8B.
#
# Read each phase header before executing. Uncomment the line you want to run.
# ============================================================================

set -u
mkdir -p logs output/{sft,grpo,distill,eval,analysis,dag_metrics,revision_gain,case_study}

# =============================================================================
# ★ Core 6-variant matrix (DR1-7B stack, used as the headline comparison)
# =============================================================================
# Each variant is independent. Launch what you need; all commands use nohup so
# the terminal can be closed. Logs land under logs/<variant>.log .
#
#   Variant                     Entry                                  Config / adapter
#   -------------------------   ------------------------------------   ---------------------------------
#   V1 Base (no training)       scripts/bench_transformers.py          DR1-Distill-Qwen-7B
#   V2 + SFT                    scripts/run_sft_config.sh sft_dr1_7b   configs/sft_deepseek_r1_7b.yaml
#   V3 + DAPO (low priority)    scripts/run_swift_rlhf.sh dapo_dr1_7b  configs/dapo_dr1_7b.yaml
#   V4 + GRPO (outcome)         scripts/run_swift_rlhf.sh grpo_outcome_only_qwen25_7b
#                               (or python3 scripts/train_grpo_ablation.py --reward outcome_only)
#   V5 + OPD                    scripts/run_swift_rlhf.sh opd_dr1_7b_to_qwen3_4b gkd   (teacher ≠ student)
#      + OPSD                   scripts/run_swift_rlhf.sh opsd_dr1_7b gkd              (teacher == student)
#   V6 + TopoPRM (full)         python3 scripts/train_grpo_ablation.py --reward hierarchical
#          ablations            --reward outcome_only | no_topo | no_continuity
#
# Training-time accuracy logging (any of V3-V6 run through train_grpo_ablation.py)
# now produces `[eval] step=X acc=Y n=Z mean_new_tokens=W` lines every
# --eval_every steps. tutorials/training_curve.py auto-picks that up and
# overlays pass@1 on the middle subplot.
#
# Evaluation switch: add --save_solutions to scripts/bench_transformers.py if
# you want tutorials/render_dag_cases.py --from-rollout to visualize the DAGs.

# V1: Base DR1-7B (no adapter) --------------------------------------------------
# nohup python3 scripts/bench_transformers.py \
#     --model /Knowin/foundation/weilinruan/hf_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#     --label base_dr1_7b --benchmarks gsm8k math500 aime2024 aime2025 \
#     --save_solutions \
#     > logs/v1_base_dr1_7b_eval.log 2>&1 &

# V2: SFT cold-start -----------------------------------------------------------
# nohup bash scripts/run_sft_config.sh sft_deepseek_r1_7b > logs/v2_sft_dr1_7b.log 2>&1 &

# V3: DAPO (low priority) ------------------------------------------------------
# nohup bash scripts/run_swift_rlhf.sh dapo_dr1_7b grpo > logs/v3_dapo_dr1_7b.log 2>&1 &

# V4: Outcome-only GRPO (no topology signals) ----------------------------------
# nohup env CUDA_VISIBLE_DEVICES=4 python3 scripts/train_grpo_ablation.py \
#     --reward outcome_only --output_dir output/grpo_outcome_only_dr1_7b \
#     --eval_every 20 --eval_size 32 \
#     > logs/v4_grpo_outcome_only.log 2>&1 &

# V5a: OPD — teacher=DR1-7B (±TopoPRM), student=Qwen3-4B -----------------------
# TEACHER_ADAPTER=output/grpo_topoprm_deepseek_r1_7b/final \
# nohup bash scripts/run_swift_rlhf.sh opd_dr1_7b_to_qwen3_4b gkd \
#     > logs/v5a_opd_dr1_to_qwen3_4b.log 2>&1 &

# V5b: OPSD — self-distillation (teacher = student backbone) --------------------
# nohup bash scripts/run_swift_rlhf.sh opsd_dr1_7b gkd > logs/v5b_opsd_dr1_7b.log 2>&1 &
# (or the legacy native trainer: bash scripts/run_tvsd_opd_minimal.sh)

# V6: TopoPRM full + ablations -------------------------------------------------
# nohup env CUDA_VISIBLE_DEVICES=4 python3 scripts/train_grpo_ablation.py \
#     --reward hierarchical --output_dir output/grpo_topoprm_deepseek_r1_7b \
#     --eval_every 20 --eval_size 32 \
#     > logs/v6_topoprm_full.log 2>&1 &
# nohup env CUDA_VISIBLE_DEVICES=5 python3 scripts/train_grpo_ablation.py \
#     --reward no_topo       --output_dir output/grpo_no_topo_dr1_7b       \
#     --eval_every 20 --eval_size 32  > logs/v6_topoprm_no_topo.log 2>&1 &
# nohup env CUDA_VISIBLE_DEVICES=6 python3 scripts/train_grpo_ablation.py \
#     --reward no_continuity --output_dir output/grpo_no_continuity_dr1_7b \
#     --eval_every 20 --eval_size 32  > logs/v6_topoprm_no_cont.log 2>&1 &

# After the six variants finish, regenerate the training-curve figure:
#   python3 tutorials/training_curve.py
# Then evaluate every adapter under V2..V6 via scripts/run_eval.sh and invoke
#   python3 -m src.eval.sync_paper_tables
# to push numbers into topoprm_paper/tables/.

# =============================================================================
# Legacy phased runbook (kept for full-paper reproduction, not required for the
# headline 6-variant comparison above)
# =============================================================================

# 0.1 Build DAG-annotated SFT/GRPO data (NuminaMath-CoT + MetaMathQA + DAG ann.)
# nohup bash scripts/run_data_pipeline.sh > logs/p0_data_pipeline.log 2>&1 &

# 0.2 Download and unify public benchmarks under one transformers protocol
# nohup bash scripts/download_benchmarks.sh > logs/p0_download_benchmarks.log 2>&1 &

# 0.3 Reward invariants sanity check (fail fast if rewards are mis-wired)
# python3 scripts/check_reward_invariants.py 2>&1 | tee logs/p0_reward_invariants.log


# =============================================================================
# Phase 1: SFT cold-start (Stage I)
# =============================================================================
# 1.1 Primary SFT (Qwen3.5-9B base)
# nohup bash scripts/run_sft_config.sh sft_qwen35_9b           > logs/p1_sft_qwen35_9b.log 2>&1 &

# 1.2 Secondary SFT for transferability (Qwen2.5-7B base)
# nohup bash scripts/run_sft_config.sh sft_qwen25_7b           > logs/p1_sft_qwen25_7b.log 2>&1 &

# 1.3 Student SFT cold-starts (used as starting point for distillation)
# nohup bash scripts/run_sft_config.sh sft_student_4b          > logs/p1_sft_student_4b.log 2>&1 &
# nohup bash scripts/run_sft_config.sh sft_student_2b          > logs/p1_sft_student_2b.log 2>&1 &
# nohup bash scripts/run_sft_config.sh sft_student_0p5b        > logs/p1_sft_student_0p5b.log 2>&1 &


# =============================================================================
# Phase 2: GRPO with hierarchical TopoPRM reward (Stage II)
# =============================================================================
# 2.1 Main run: hierarchical multiplicative aggregation (Eq. 1 in the paper)
# nohup bash scripts/run_grpo.sh grpo_hierarchical_qwen35_9b   > logs/p2_grpo_hier_9b.log 2>&1 &

# 2.2 Reward-component ablation (Sec. 4.2 + appendix tables)
# nohup bash scripts/run_grpo.sh grpo_outcome_only_qwen35_9b   > logs/p2_grpo_outcome_only.log 2>&1 &
# nohup bash scripts/run_grpo.sh grpo_no_topo_qwen35_9b        > logs/p2_grpo_no_topo.log 2>&1 &
# nohup bash scripts/run_grpo.sh grpo_no_continuity_qwen35_9b  > logs/p2_grpo_no_cont.log 2>&1 &

# 2.3 Aggregation strategy ablation (Table: aggregation)
# nohup bash scripts/run_grpo.sh grpo_linear_qwen35_9b         > logs/p2_grpo_linear.log 2>&1 &
# nohup bash scripts/run_grpo.sh grpo_clipped_qwen35_9b        > logs/p2_grpo_clipped.log 2>&1 &
# nohup bash scripts/run_grpo.sh grpo_gated_qwen35_9b          > logs/p2_grpo_gated.log 2>&1 &

# 2.4 Cross-scale transferability (Sec. 4.4 / appendix cross-scale table)
# nohup bash scripts/run_grpo.sh grpo_hierarchical_qwen25_7b   > logs/p2_grpo_hier_7b.log 2>&1 &

# 2.5 DAG data scaling (appendix RQ4): vary DAG-supervised subset size
# for FRAC in 0.0 0.25 0.5 1.0; do
#   DAG_FRAC=$FRAC nohup bash scripts/run_grpo.sh grpo_dagscale_qwen35_9b \
#     > logs/p2_grpo_dagscale_${FRAC}.log 2>&1 &
# done


# =============================================================================
# Phase 3: Topology-guided self-distillation (Stage III)
# =============================================================================
# 3.1 Phase III-A: topology-conditioned self-refinement rollout
# nohup bash scripts/run_srt_rollout.sh srt_qwen35_9b          > logs/p3a_srt_rollout.log 2>&1 &

# 3.2 Phase III-A: build SRT training data with topology dispatch P_r
# nohup bash scripts/run_build_srt_data.sh                     > logs/p3a_srt_build.log 2>&1 &

# 3.3 Phase III-B: TVSD on-policy distillation to compact students
# nohup bash scripts/run_distill.sh tvsd_student_4b            > logs/p3b_tvsd_4b.log 2>&1 &
# nohup bash scripts/run_distill.sh tvsd_student_2b            > logs/p3b_tvsd_2b.log 2>&1 &
# nohup bash scripts/run_distill.sh tvsd_student_0p5b          > logs/p3b_tvsd_0p5b.log 2>&1 &

# 3.4 Reverse-KL anti-example (kept as diagnostic row in compression table)
# nohup bash scripts/run_distill.sh distill_rkl_8b             > logs/p3b_rkl_8b.log 2>&1 &


# =============================================================================
# Phase 4: Public-benchmark accuracy (main result, Table 1 in the paper)
# =============================================================================
# Each line evaluates one model on the nine public benchmarks under the
# unified transformers protocol with chat template. Outputs land in
# output/eval/<label>/.
#
# 4.1 Reference open-source models
# nohup bash scripts/run_public_benchmarks.sh Qwen/Qwen2.5-7B-Instruct       "" qwen25_7b_instruct > logs/p4_qwen25_7b_instruct.log 2>&1 &
# nohup bash scripts/run_public_benchmarks.sh meta-llama/Llama-3.1-8B-Instruct "" llama31_8b      > logs/p4_llama31_8b.log 2>&1 &
# nohup bash scripts/run_public_benchmarks.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-7B "" dr1_distill_7b > logs/p4_dr1_distill_7b.log 2>&1 &

# 4.2 Our 9B family
# nohup bash scripts/run_public_benchmarks.sh /path/to/Qwen3.5-9B "$(ls -dt output/sft/sft_qwen35_9b/*/checkpoint-* | head -1)" sft_9b           > logs/p4_sft_9b.log 2>&1 &
# nohup bash scripts/run_public_benchmarks.sh /path/to/Qwen3.5-9B "$(ls -dt output/grpo/grpo_outcome_only_qwen35_9b/*/checkpoint-* | head -1)" grpo_outcome_9b > logs/p4_grpo_outcome_9b.log 2>&1 &
# nohup bash scripts/run_public_benchmarks.sh /path/to/Qwen3.5-9B "$(ls -dt output/grpo/grpo_no_topo_qwen35_9b/*/checkpoint-* | head -1)" grpo_no_topo_9b > logs/p4_grpo_no_topo_9b.log 2>&1 &
# nohup bash scripts/run_public_benchmarks.sh /path/to/Qwen3.5-9B "$(ls -dt output/grpo/grpo_no_continuity_qwen35_9b/*/checkpoint-* | head -1)" grpo_no_cont_9b > logs/p4_grpo_no_cont_9b.log 2>&1 &
# nohup bash scripts/run_public_benchmarks.sh /path/to/Qwen3.5-9B "$(ls -dt output/grpo/grpo_hierarchical_qwen35_9b/*/checkpoint-* | head -1)" topoprm_hier_9b > logs/p4_topoprm_hier_9b.log 2>&1 &
# nohup bash scripts/run_public_benchmarks.sh /path/to/Qwen3.5-9B "$(ls -dt output/grpo/grpo_gated_qwen35_9b/*/checkpoint-* | head -1)" topoprm_gated_9b > logs/p4_topoprm_gated_9b.log 2>&1 &

# 4.3 Our 7B family (Qwen2.5-7B base)
# nohup bash scripts/run_public_benchmarks.sh /path/to/Qwen2.5-7B "" base_qwen25_7b              > logs/p4_base_qwen25_7b.log 2>&1 &
# nohup bash scripts/run_public_benchmarks.sh /path/to/Qwen2.5-7B "$(ls -dt output/grpo/grpo_hierarchical_qwen25_7b/*/checkpoint-* | head -1)" topoprm_hier_7b > logs/p4_topoprm_hier_7b.log 2>&1 &

# 4.4 Distilled students
# nohup bash scripts/run_public_benchmarks.sh /path/to/Qwen3-4B "$(ls -dt output/distill/tvsd_student_4b/*/checkpoint-* | head -1)" student_tvsd_4b > logs/p4_student_tvsd_4b.log 2>&1 &
# nohup bash scripts/run_public_benchmarks.sh /path/to/Qwen3-2B "$(ls -dt output/distill/tvsd_student_2b/*/checkpoint-* | head -1)" student_tvsd_2b > logs/p4_student_tvsd_2b.log 2>&1 &
# nohup bash scripts/run_public_benchmarks.sh /path/to/Qwen3-0.5B "$(ls -dt output/distill/tvsd_student_0p5b/*/checkpoint-* | head -1)" student_tvsd_0p5b > logs/p4_student_tvsd_0p5b.log 2>&1 &
# nohup bash scripts/run_public_benchmarks.sh /path/to/Qwen3-8B "$(ls -dt output/distill/distill_rkl_8b/*/checkpoint-* | head -1)" student_rkl_8b > logs/p4_student_rkl_8b.log 2>&1 &


# =============================================================================
# Phase 5: In-domain validation (appendix)
# =============================================================================
# 5.1 Critique-style accuracy on the in-domain validation split
# nohup bash scripts/run_eval_indomain.sh   > logs/p5_eval_indomain.log 2>&1 &


# =============================================================================
# Phase 6: Structural diagnostics (DAG metrics)
# =============================================================================
# 6.1 Run DAG extractor and metrics over completed eval JSONLs
# nohup bash scripts/run_dag_metrics.sh     > logs/p6_dag_metrics.log 2>&1 &
# 6.2 (paper-only) The "critical dependency chain" view used in 3_method/6_appendix
#     is derived offline by src/dag/compress.py (compress_dag, compress_dag_by_layers)
#     and visualized in src/gui/dag_reward_viewer.py; no extra training command.


# =============================================================================
# Phase 7: Revision-gain probe (Appendix RQ5)
# =============================================================================
# 7.1 First-attempt vs revised diagnostics (TVSD vs plain reviser baseline)
# nohup bash scripts/run_revision_gain.sh   > logs/p7_revision_gain.log 2>&1 &


# =============================================================================
# Phase 8: Case study (Appendix)
# =============================================================================
# 8.1 Same-prompt teacher/student trace pairs aligned by prompt hash
# nohup bash scripts/run_case_study.sh      > logs/p8_case_study.log 2>&1 &


# =============================================================================
# Phase 9: Aggregate and sync paper tables
# =============================================================================
# 9.1 Collect raw eval outputs into a single experiment summary JSON
# python3 -m src.eval.collect_experiment_results \
#     --eval_dir output/eval --output_dir output/analysis 2>&1 | tee logs/p9_collect.log

# 9.2 Sync paper tables (overwrite numeric cells in topoprm_paper/tables/*)
# python3 -m src.eval.sync_paper_tables \
#     --summary output/analysis/experiment_summary.json \
#     --paper_dir topoprm_paper \
#     --eval_dir output/eval 2>&1 | tee logs/p9_sync.log

# 9.3 Optional: regenerate combined CSV for camera-ready review
# python3 -m src.eval.export_paper_tables \
#     --eval_dir output/eval \
#     --output output/analysis/paper_table_summary.csv 2>&1 | tee logs/p9_export.log


# =============================================================================
# Notes for reviewers / camera-ready
# =============================================================================
# - The unified protocol is intentionally simple (one script, one chat template).
# - The DAG extractor and reward components are deterministic; per-batch
#   diagnostics are logged under output/grpo/<run>/diagnostics/*.jsonl.
# - The reverse-KL distillation row is kept only as an anti-example.
# - To reproduce 8-page main numbers, only Phase 1.1 + 2.1 + 4.* are required.
