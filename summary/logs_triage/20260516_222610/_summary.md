# Log triage summary (2026-05-16T22:26:11)

- Root: `/mnt/users/rwl/topoprm/logs`
- ACTIVE_MIN: `30` min
- INCLUDE_SERVER_B: `False`
- PURGE_ZERO: `False`
- CONFIRM: `True`

| category | count | total size (KB) |
|---|---:|---:|
| VALID | 39 | 450.7 |
| PARTIAL_VALID | 2 | 64.4 |
| TRAIN_LOG | 10 | 614.7 |
| ALL_ZERO | 0 | 0.0 |
| SKIPPED | 0 | 0.0 |
| CRASHED | 0 | 0.0 |
| EMPTY | 0 | 0.0 |
| SHELL_PRINTF | 0 | 0.0 |
| ACTIVE | 0 | 0.0 |

## VALID logs (keep in place; mine these for paper)

- `phase2_dr1_7b_chat_math.log`  (05-07 10:11, 16 KB, 4 pass@1 / 4 saved)
- `phase2_dr1_7b_chat_mmlu.log`  (05-06 20:05, 48 KB, 1 pass@1 / 1 saved)
- `phase2_dr1_7b_math500.log`  (05-06 12:38, 13 KB, 1 pass@1 / 1 saved)
- `phase2_dr1_7b_mmlu.log`  (05-05 19:54, 45 KB, 1 pass@1 / 1 saved)
- `phase2_eval_baseline.log`  (05-02 09:07, 22 KB, 1 pass@1 / 1 saved)
- `phase2_eval_cnmo.log`  (05-01 12:55, 4 KB, 1 pass@1 / 1 saved)
- `phase2_eval_aime.log`  (05-01 03:52, 4 KB, 1 pass@1 / 1 saved)
- `eval_topoprm_hier_9b_v3_v3b_mnt512_232045.log`  (04-27 18:57, 3 KB, 1 pass@1 / 1 saved)
- `eval_outcome_only_9b_v3_v3b_mnt512_145614.log`  (04-27 07:48, 4 KB, 1 pass@1 / 1 saved)
- `eval_topoprm_gated_9b_v3_v3b_mnt512_023219.log`  (04-27 03:35, 3 KB, 1 pass@1 / 1 saved)
- `eval_sft_9b_v3_v3b_mnt512_012416.log`  (04-26 23:20, 3 KB, 1 pass@1 / 1 saved)
- `eval_student_4b_sft_distill_v3_v3b_mnt512_101620.log`  (04-26 05:44, 3 KB, 1 pass@1 / 1 saved)
- `eval_topoprm_hier_qwen25_7b_v3_v3b_mnt512_020810.log`  (04-26 02:32, 3 KB, 1 pass@1 / 1 saved)
- `eval_student_4b_sft_distill_v3_v3b_mnt512_014620.log`  (04-25 14:56, 3 KB, 1 pass@1 / 1 saved)
- `eval_topoprm_hier_9b_continue_v3_v3b_mnt1536_191645.log`  (04-25 09:58, 9 KB, 2 pass@1 / 2 saved)
- `eval_topoprm_hier_9b_v3_v3b_mnt1536_132804.log`  (04-25 03:49, 8 KB, 2 pass@1 / 2 saved)
- `eval_outcome_only_9b_v3_v3b_mnt1536_031414.log`  (04-24 18:16, 9 KB, 2 pass@1 / 2 saved)
- `eval_sft_9b_v3_v3b_mnt512_000621.log`  (04-24 13:28, 4 KB, 1 pass@1 / 1 saved)
- `eval_topoprm_hier_9b_continue_v3_v3b_mnt2560_175417.log`  (04-24 05:54, 6 KB, 5 pass@1 / 5 saved)
- `eval_sft_9b_v3_v3b_mnt512_193453.log`  (04-24 05:50, 4 KB, 1 pass@1 / 1 saved)
- `eval_topoprm_gated_9b_v3_v3b_mnt1536_135743.log`  (04-24 02:57, 8 KB, 2 pass@1 / 2 saved)
- `eval_sft_9b_v3_v3b_mnt1536_135718.log`  (04-24 00:06, 8 KB, 2 pass@1 / 2 saved)
- `eval_topoprm_gated_9b_v3_v3b_mnt2560_140153.log`  (04-23 01:40, 6 KB, 5 pass@1 / 5 saved)
- `eval_student_4b_sft_distill_v3_v3b_mnt1536_095850.log`  (04-23 01:26, 11 KB, 2 pass@1 / 2 saved)
- `eval_topoprm_hier_9b_v3_v3b_mnt2560_135053.log`  (04-23 01:20, 7 KB, 5 pass@1 / 5 saved)
- `eval_no_topo_9b_v3_v3b_mnt1536_095603.log`  (04-22 21:28, 12 KB, 2 pass@1 / 2 saved)
- `eval_topoprm_gated_9b_v3_mnt512_215145.log`  (04-22 14:01, 21 KB, 1 pass@1 / 1 saved)
- `eval_topoprm_hier_9b_v3_mnt512_215145.log`  (04-22 13:50, 21 KB, 1 pass@1 / 1 saved)
- `eval_student_4b_sft_distill_v3_v3b_mnt2560_225142.log`  (04-22 09:58, 5 KB, 5 pass@1 / 5 saved)
- `eval_no_topo_9b_v3_v3b_mnt2560_224906.log`  (04-22 09:55, 6 KB, 5 pass@1 / 5 saved)
- `eval_outcome_only_9b_v3_mnt2560_200153.log`  (04-21 19:00, 10 KB, 5 pass@1 / 5 saved)
- `eval_sft_9b_v3_mnt2560_182700.log`  (04-21 16:05, 10 KB, 5 pass@1 / 5 saved)
- `eval_topoprm_hier_qwen25_7b_v3_mnt512_032439.log`  (04-21 10:21, 23 KB, 1 pass@1 / 1 saved)
- `eval_topoprm_hier_qwen25_7b_v3_mnt1536_215412.log`  (04-21 03:24, 23 KB, 2 pass@1 / 2 saved)
- `eval_topoprm_hier_qwen25_7b_v3_mnt2560_182700.log`  (04-20 21:54, 10 KB, 5 pass@1 / 5 saved)
- `eval_topoprm_gated_9b_v2_ext_k5_gpu2_resume.log`  (04-20 18:22, 10 KB, 2 pass@1 / 2 saved)
- `eval_student_4b_sft_distill_ext_k5_gpu1_resume.log`  (04-20 18:21, 9 KB, 4 pass@1 / 4 saved)
- `eval_sft_9b_v2_ext_k5_gpu1_resume.log`  (04-20 18:21, 10 KB, 2 pass@1 / 2 saved)
- `eval_topoprm_hier_9b_v2_ext_k5_gpu7_resume.log`  (04-20 18:19, 7 KB, 2 pass@1 / 2 saved)

## PARTIAL_VALID logs (killed mid-run, but a long enough running-acc tail to read off an estimate)

- `eval_sft_9b_v3_mnt1536_160516.log`  (04-21 21:44, 12 KB) - killed mid-run; last running acc=93.9% at 572/1319
- `eval_topoprm_hier_qwen25_7b_ext_k5_gpu7_resume.log`  (04-20 18:22, 52 KB) - killed mid-run; last running acc=65.4% at 4088/14042

## TRAIN_LOG (training/pipeline runs, kept because they store W&B URLs / training trace)

- `phase3_sft.log`  (05-07 20:51, 351 KB) - non-eval log, 351 KB
- `phase2_download.log`  (05-01 00:37, 0 KB) - non-eval log, 0 KB
- `phase1_dag_build.log`  (05-01 00:35, 2 KB) - non-eval log, 2 KB
- `v3b_gpu0_b.log`  (04-27 17:16, 1 KB) - non-eval log, 1 KB
- `v3b_gpu4_d.log`  (04-26 02:07, 0 KB) - non-eval log, 0 KB
- `v3b_gpu1_b.log`  (04-25 01:25, 1 KB) - non-eval log, 1 KB
- `v3b_gpu3.log`  (04-25 01:24, 0 KB) - non-eval log, 0 KB
- `grpo_7b_alpha09.log`  (04-24 19:15, 1 KB) - non-eval log, 1 KB
- `grpo_hier_continue.log`  (04-23 17:50, 254 KB) - non-eval log, 254 KB, contains traceback
- `rerun_unified_v3_top.log`  (04-20 18:27, 0 KB) - non-eval log, 0 KB

## ALL_ZERO logs (every pass@1 = 0.0%; verify before deleting)

