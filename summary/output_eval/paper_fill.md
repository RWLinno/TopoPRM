# Paper-fill checklist

Generated 2026-05-17T00:57:55

For each paper row we list the matching `label`s found in `output/eval/` and the pass@1 (in %) per benchmark. Use this as the source list; copy numbers into the LaTeX tables.


## table: `tables/public_results_unified.tex`

### Qwen3.5-9B (base)
- `base_9b`
    gsm8k=90.98  math500=55.00  olympiadbench=--  omni_math=--
    aime2024=--  aime2025=--  cnmo2024=--  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `base_9b_v3`
    gsm8k=55.19  math500=19.80  olympiadbench=11.19  omni_math=16.41
    aime2024=--  aime2025=2.22  cnmo2024=10.00  mmlu=28.60
    gpqa_diamond=--  high=--  middle=--
- `ref_qwen35_9b_base_B`
    gsm8k=69.45  math500=66.60  olympiadbench=55.22  omni_math=61.45
    aime2024=16.67  aime2025=21.11  cnmo2024=16.67  mmlu=--
    gpqa_diamond=--  high=--  middle=--

### Qwen3.5-9B + SFT
- `sft_9b`
    gsm8k=87.95  math500=53.00  olympiadbench=--  omni_math=--
    aime2024=--  aime2025=--  cnmo2024=--  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `sft_9b_v2`
    gsm8k=96.60  math500=--  olympiadbench=--  omni_math=--
    aime2024=--  aime2025=--  cnmo2024=--  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `sft_9b_v2_ext`
    gsm8k=--  math500=50.80  olympiadbench=--  omni_math=--
    aime2024=16.67  aime2025=13.33  cnmo2024=--  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `sft_9b_v2_ext_k5_gpu1`
    gsm8k=--  math500=--  olympiadbench=32.09  omni_math=40.46
    aime2024=23.33  aime2025=12.22  cnmo2024=23.33  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `sft_9b_v3`
    gsm8k=94.09  math500=50.80  olympiadbench=32.84  omni_math=42.37
    aime2024=30.00  aime2025=15.56  cnmo2024=30.00  mmlu=68.20
    gpqa_diamond=20.71  high=--  middle=--
- `sft_qwen35_9b_B`
    gsm8k=91.96  math500=64.80  olympiadbench=44.03  omni_math=53.82
    aime2024=26.67  aime2025=13.33  cnmo2024=26.67  mmlu=78.40
    gpqa_diamond=46.97  high=--  middle=--
- `sft_qwen35_9b_light200`
    gsm8k=--  math500=--  olympiadbench=--  omni_math=--
    aime2024=--  aime2025=--  cnmo2024=--  mmlu=--
    gpqa_diamond=--  high=acc:30.50/fmt:94.00/tok:2191  middle=acc:40.50/fmt:98.00/tok:1602

### Qwen3.5-9B + GRPO (outcome-only)
- `grpo_outcome_only_qwen35_9b_B`
    gsm8k=66.57  math500=41.20  olympiadbench=26.12  omni_math=--
    aime2024=3.33  aime2025=4.44  cnmo2024=3.33  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `grpo_outcome_only_qwen35_9b_light200`
    gsm8k=--  math500=--  olympiadbench=--  omni_math=--
    aime2024=--  aime2025=--  cnmo2024=--  mmlu=--
    gpqa_diamond=--  high=acc:25.00/fmt:82.00/tok:571  middle=acc:31.50/fmt:84.00/tok:493
- `grpo_outcome_only_qwen35_9b_mcl4096_light200`
    gsm8k=--  math500=--  olympiadbench=--  omni_math=--
    aime2024=--  aime2025=--  cnmo2024=--  mmlu=--
    gpqa_diamond=--  high=acc:25.00/fmt:83.50/tok:569  middle=acc:33.00/fmt:86.50/tok:528
- `outcome_only_9b`
    gsm8k=88.32  math500=54.40  olympiadbench=--  omni_math=--
    aime2024=--  aime2025=--  cnmo2024=--  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `outcome_only_9b_v3`
    gsm8k=93.33  math500=50.80  olympiadbench=31.34  omni_math=41.60
    aime2024=16.67  aime2025=13.33  cnmo2024=16.67  mmlu=63.00
    gpqa_diamond=--  high=--  middle=--

### Qwen3.5-9B + GRPO (w/o topology)
- `grpo_no_topo_9b_own_sft_light200`
    gsm8k=--  math500=--  olympiadbench=--  omni_math=--
    aime2024=--  aime2025=--  cnmo2024=--  mmlu=--
    gpqa_diamond=--  high=acc:28.50/fmt:89.50/tok:2073  middle=acc:37.00/fmt:99.50/tok:1577
- `grpo_no_topo_qwen35_9b_light200`
    gsm8k=--  math500=--  olympiadbench=--  omni_math=--
    aime2024=--  aime2025=--  cnmo2024=--  mmlu=--
    gpqa_diamond=--  high=acc:28.00/fmt:82.00/tok:552  middle=acc:34.00/fmt:85.00/tok:494
- `grpo_no_topo_qwen35_9b_mcl4096_light200`
    gsm8k=--  math500=--  olympiadbench=--  omni_math=--
    aime2024=--  aime2025=--  cnmo2024=--  mmlu=--
    gpqa_diamond=--  high=acc:28.00/fmt:86.50/tok:578  middle=acc:33.00/fmt:86.50/tok:503
- `no_topo_9b`
    gsm8k=88.70  math500=54.20  olympiadbench=--  omni_math=--
    aime2024=--  aime2025=--  cnmo2024=--  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `no_topo_9b_v3`
    gsm8k=92.95  math500=51.00  olympiadbench=29.85  omni_math=42.37
    aime2024=20.00  aime2025=12.22  cnmo2024=20.00  mmlu=65.80
    gpqa_diamond=--  high=--  middle=--

### Qwen3.5-9B + GRPO (w/o continuity)
- `grpo_no_continuity_qwen35_9b_light200`
    gsm8k=--  math500=--  olympiadbench=--  omni_math=--
    aime2024=--  aime2025=--  cnmo2024=--  mmlu=--
    gpqa_diamond=--  high=acc:29.00/fmt:82.00/tok:561  middle=acc:29.50/fmt:83.50/tok:479
- `no_continuity_9b`
    gsm8k=90.90  math500=55.00  olympiadbench=--  omni_math=--
    aime2024=--  aime2025=--  cnmo2024=--  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `no_continuity_9b_v3`
    gsm8k=51.02  math500=21.60  olympiadbench=8.21  omni_math=14.89
    aime2024=6.67  aime2025=1.11  cnmo2024=6.67  mmlu=20.20
    gpqa_diamond=--  high=--  middle=--

### Qwen3.5-9B + TopoPRM (hierarchical)
- `topoprm_hier_9b`
    gsm8k=87.72  math500=53.40  olympiadbench=--  omni_math=--
    aime2024=--  aime2025=--  cnmo2024=--  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `topoprm_hier_9b_continue_v3`
    gsm8k=93.48  math500=48.80  olympiadbench=33.58  omni_math=40.08
    aime2024=20.00  aime2025=10.00  cnmo2024=20.00  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `topoprm_hier_9b_v2`
    gsm8k=94.40  math500=--  olympiadbench=--  omni_math=--
    aime2024=--  aime2025=--  cnmo2024=--  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `topoprm_hier_9b_v2_ext_k5_gpu7`
    gsm8k=--  math500=--  olympiadbench=32.84  omni_math=41.60
    aime2024=13.33  aime2025=7.78  cnmo2024=13.33  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `topoprm_hier_9b_v2_full`
    gsm8k=93.48  math500=49.80  olympiadbench=--  omni_math=--
    aime2024=20.00  aime2025=13.33  cnmo2024=20.00  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `topoprm_hier_9b_v3`
    gsm8k=93.48  math500=49.80  olympiadbench=32.84  omni_math=42.75
    aime2024=26.67  aime2025=12.22  cnmo2024=26.67  mmlu=61.80
    gpqa_diamond=15.66  high=--  middle=--

### Qwen3.5-9B + TopoPRM (gated)
- `topoprm_gated_9b`
    gsm8k=87.79  math500=55.40  olympiadbench=--  omni_math=--
    aime2024=--  aime2025=--  cnmo2024=--  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `topoprm_gated_9b_v2`
    gsm8k=93.78  math500=50.80  olympiadbench=--  omni_math=--
    aime2024=--  aime2025=--  cnmo2024=--  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `topoprm_gated_9b_v2_ext`
    gsm8k=--  math500=--  olympiadbench=31.34  omni_math=39.69
    aime2024=10.00  aime2025=10.00  cnmo2024=10.00  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `topoprm_gated_9b_v2_ext_k5_gpu2`
    gsm8k=--  math500=--  olympiadbench=32.09  omni_math=38.17
    aime2024=10.00  aime2025=7.78  cnmo2024=10.00  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `topoprm_gated_9b_v3`
    gsm8k=93.78  math500=50.80  olympiadbench=32.84  omni_math=41.98
    aime2024=20.00  aime2025=13.33  cnmo2024=20.00  mmlu=61.07
    gpqa_diamond=21.21  high=--  middle=--

### DR1-7B (base)
- `baseline_dr1_7b`
    gsm8k=89.16  math500=56.00  olympiadbench=--  omni_math=--
    aime2024=--  aime2025=--  cnmo2024=--  mmlu=49.99
    gpqa_diamond=--  high=--  middle=--
- `baseline_dr1_7b_aime`
    gsm8k=--  math500=--  olympiadbench=--  omni_math=--
    aime2024=20.00  aime2025=--  cnmo2024=--  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `baseline_dr1_7b_chat`
    gsm8k=83.47  math500=55.60  olympiadbench=--  omni_math=--
    aime2024=23.33  aime2025=--  cnmo2024=23.33  mmlu=42.45
    gpqa_diamond=--  high=--  middle=--
- `baseline_dr1_7b_cnmo`
    gsm8k=--  math500=--  olympiadbench=--  omni_math=--
    aime2024=--  aime2025=--  cnmo2024=20.00  mmlu=--
    gpqa_diamond=--  high=--  middle=--

### DR1-7B + SFT  -- *(no matching label)*

### DR1-7B + GRPO (outcome-only)  -- *(no matching label)*

### DR1-7B + TopoPRM (hierarchical)  -- *(no matching label)*

### Student (4B, SFT distill)
- `base_4b_v3`
    gsm8k=40.49  math500=18.80  olympiadbench=9.70  omni_math=14.12
    aime2024=--  aime2025=--  cnmo2024=--  mmlu=29.80
    gpqa_diamond=14.14  high=--  middle=--
- `student_4b_sft_distill_ext_k5_gpu1`
    gsm8k=--  math500=--  olympiadbench=14.18  omni_math=19.85
    aime2024=--  aime2025=1.11  cnmo2024=--  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `student_4b_sft_distill_v3`
    gsm8k=90.52  math500=38.20  olympiadbench=21.64  omni_math=34.35
    aime2024=6.67  aime2025=6.67  cnmo2024=6.67  mmlu=43.00
    gpqa_diamond=--  high=--  middle=--

### Qwen2.5-7B-Instruct (measured)
- `ref_qwen25_7b_instruct_B`
    gsm8k=91.28  math500=65.40  olympiadbench=42.54  omni_math=54.20
    aime2024=16.67  aime2025=12.22  cnmo2024=16.67  mmlu=70.07
    gpqa_diamond=28.28  high=--  middle=--

### Qwen2.5-Math-7B-Instruct (measured)
- `ref_qwen25_math_7b_instruct_B`
    gsm8k=95.45  math500=73.80  olympiadbench=58.21  omni_math=62.98
    aime2024=13.33  aime2025=10.00  cnmo2024=13.33  mmlu=54.80
    gpqa_diamond=28.28  high=--  middle=--

### Llama-3.1-8B-Instruct (measured)  -- *(no matching label)*

### DeepSeek-R1-Distill-Llama-8B (measured)  -- *(no matching label)*

### DeepSeek-R1-Distill-Qwen-7B (measured)  -- *(no matching label)*

### Qwen3.5-2B (measured)
- `ref_qwen35_2b_B`
    gsm8k=77.18  math500=60.80  olympiadbench=32.84  omni_math=48.47
    aime2024=13.33  aime2025=6.67  cnmo2024=13.33  mmlu=63.67
    gpqa_diamond=37.88  high=--  middle=--

### Qwen3.5-4B (measured)
- `ref_qwen35_4b_B`
    gsm8k=57.70  math500=38.60  olympiadbench=19.40  omni_math=--
    aime2024=3.33  aime2025=2.22  cnmo2024=3.33  mmlu=--
    gpqa_diamond=--  high=--  middle=--

### Qwen3.5-9B (measured)
- `ref_qwen35_9b_B`
    gsm8k=66.57  math500=41.20  olympiadbench=26.12  omni_math=--
    aime2024=3.33  aime2025=4.44  cnmo2024=3.33  mmlu=--
    gpqa_diamond=--  high=--  middle=--
- `ref_qwen35_9b_base_B`
    gsm8k=69.45  math500=66.60  olympiadbench=55.22  omni_math=61.45
    aime2024=16.67  aime2025=21.11  cnmo2024=16.67  mmlu=--
    gpqa_diamond=--  high=--  middle=--

### Qwen3-8B (measured)  -- *(no matching label)*


## table: `tables/private_results.tex`

### Qwen3-32B (zero-shot)
- `baseline_qwen3_32b_light200_mnt1024_gpu5`
    high=acc:7.00/fmt:43.50/tok:1240  middle=acc:16.00/fmt:64.50/tok:1200

### Qwen3-32B + SFT
- `grpo_hier_9b_own_sft_light200`
    high=acc:24.50/fmt:89.50/tok:2345  middle=acc:37.50/fmt:97.50/tok:1716
- `grpo_no_topo_9b_own_sft_light200`
    high=acc:28.50/fmt:89.50/tok:2073  middle=acc:37.00/fmt:99.50/tok:1577
- `grpo_outcome_9b_own_sft_light200`
    high=acc:22.50/fmt:87.50/tok:1776  middle=acc:37.00/fmt:96.50/tok:1447
- `sft_light200`
    high=acc:26.50/fmt:45.75/tok:--  middle=acc:39.00/fmt:48.75/tok:--
- `sft_private_boost_light200_mnt1024`
    high=acc:19.50/fmt:65.00/tok:1233  middle=acc:29.00/fmt:78.50/tok:1164

### Qwen3-32B + Outcome Only
- `grpo_outcome_light200`
    high=acc:13.00/fmt:85.50/tok:--  middle=acc:19.50/fmt:89.00/tok:--
- `grpo_outcome_light200_mnt1024_gpu7`
    high=acc:13.00/fmt:85.50/tok:475  middle=acc:19.50/fmt:89.00/tok:410

### Qwen3-32B + w/o Topology
- `grpo_no_topo_light200_mnt1024_gpu6`
    high=acc:16.50/fmt:86.00/tok:460  middle=acc:23.00/fmt:91.00/tok:397

### Qwen3-32B + w/o Continuity
- `grpo_no_continuity_light200_mnt1024_gpu7`
    high=acc:20.00/fmt:85.00/tok:436  middle=acc:27.50/fmt:90.50/tok:379

### Qwen3-32B + Clipped Linear
- `grpo_clipped_light200_mnt1024_gpu5`
    high=acc:21.50/fmt:88.50/tok:458  middle=acc:29.00/fmt:93.00/tok:382

### Qwen3-32B + TopoPRM (full)
- `grpo_main_full`
    high=acc:22.40/fmt:87.10/tok:430  middle=acc:37.23/fmt:94.64/tok:364
- `grpo_main_light200`
    high=acc:25.50/fmt:89.00/tok:--  middle=acc:33.00/fmt:93.00/tok:--
- `grpo_main_light200_mnt1024_gpu6`
    high=acc:25.50/fmt:89.00/tok:436  middle=acc:33.00/fmt:93.00/tok:362

### Qwen3.5-9B + SFT (legacy)
- `sft_qwen35_9b_light200`
    high=acc:30.50/fmt:94.00/tok:2191  middle=acc:40.50/fmt:98.00/tok:1602

### Qwen3.5-9B + Outcome Only (legacy)
- `grpo_outcome_only_qwen35_9b_light200`
    high=acc:25.00/fmt:82.00/tok:571  middle=acc:31.50/fmt:84.00/tok:493

### Qwen3.5-9B + w/o Topology (legacy)
- `grpo_no_topo_qwen35_9b_light200`
    high=acc:28.00/fmt:82.00/tok:552  middle=acc:34.00/fmt:85.00/tok:494

### Qwen3.5-9B + w/o Continuity (legacy)
- `grpo_no_continuity_qwen35_9b_light200`
    high=acc:29.00/fmt:82.00/tok:561  middle=acc:29.50/fmt:83.50/tok:479

### Qwen3.5-9B + TopoPRM (full, legacy)
- `grpo_hierarchical_qwen35_9b_light200`
    high=acc:27.00/fmt:83.50/tok:546  middle=acc:30.50/fmt:87.50/tok:486

### Qwen3.5-9B + TopoPRM (ng4, legacy)
- `grpo_hierarchical_qwen35_9b_ng4_light200`
    high=acc:29.00/fmt:81.50/tok:544  middle=acc:33.00/fmt:86.50/tok:479

### Qwen3.5-9B (mcl4096) + Outcome
- `grpo_outcome_only_qwen35_9b_mcl4096_light200`
    high=acc:25.00/fmt:83.50/tok:569  middle=acc:33.00/fmt:86.50/tok:528

### Qwen3.5-9B (mcl4096) + w/o Topology
- `grpo_no_topo_qwen35_9b_mcl4096_light200`
    high=acc:28.00/fmt:86.50/tok:578  middle=acc:33.00/fmt:86.50/tok:503

### Qwen3.5-9B (mcl4096) + TopoPRM (hier)
- `grpo_hierarchical_qwen35_9b_mcl4096_light200`
    high=acc:25.50/fmt:86.00/tok:564  middle=acc:29.50/fmt:85.00/tok:499

### Qwen3.5-9B (mcl4096) + TopoPRM (gated)
- `grpo_gated_qwen35_9b_mcl4096_light200`
    high=acc:25.00/fmt:83.50/tok:569  middle=acc:33.00/fmt:86.50/tok:528

### Qwen3.5-9B (mcl4096, own-SFT) + Outcome
- `grpo_outcome_9b_own_sft_light200`
    high=acc:22.50/fmt:87.50/tok:1776  middle=acc:37.00/fmt:96.50/tok:1447

### Qwen3.5-9B (mcl4096, own-SFT) + w/o Topology
- `grpo_no_topo_9b_own_sft_light200`
    high=acc:28.50/fmt:89.50/tok:2073  middle=acc:37.00/fmt:99.50/tok:1577

### Qwen3.5-9B (mcl4096, own-SFT) + TopoPRM (hier)
- `grpo_hier_9b_own_sft_light200`
    high=acc:24.50/fmt:89.50/tok:2345  middle=acc:37.50/fmt:97.50/tok:1716

### Qwen2.5-7B + SFT (boost)
- `sft_qwen25_7b_boost_light200`
    high=acc:9.00/fmt:31.00/tok:1415  middle=acc:22.50/fmt:45.50/tok:1185

### Qwen2.5-7B + Outcome Only
- `grpo_outcome_only_qwen25_7b_light200`
    high=acc:8.00/fmt:16.50/tok:756  middle=acc:22.00/fmt:19.50/tok:705

### Qwen2.5-7B + w/o Topology
- `grpo_no_topo_qwen25_7b_light200`
    high=acc:5.00/fmt:35.50/tok:797  middle=acc:21.00/fmt:26.00/tok:725

### Qwen2.5-7B + w/o Continuity
- `grpo_no_continuity_qwen25_7b_light200`
    high=acc:8.00/fmt:52.00/tok:974  middle=acc:12.00/fmt:47.50/tok:890

### Qwen2.5-7B + TopoPRM (hier)
- `grpo_hierarchical_qwen25_7b_light200`
    high=acc:4.50/fmt:23.00/tok:986  middle=acc:14.00/fmt:38.00/tok:970

### Distill: Qwen3-8B + RKL
- `distill_rkl_8b`
    high=--  middle=--
- `distill_rkl_8b_compact_light200`
    high=acc:58.00/fmt:97.00/tok:613  middle=acc:60.00/fmt:94.00/tok:532


## unmatched labels (review and re-tag if useful)

- `base_qwen25_7b` (2 benches: gsm8k, math500)
- `grpo_hierarchical_qwen35_9b_mem70_light200` (2 benches: high, middle)
- `topoprm_hier_7b` (2 benches: gsm8k, math500)
- `topoprm_hier_qwen25_7b_ext_k5_gpu7` (5 benches: aime2024, aime2025, cnmo2024, olympiadbench, omni_math)
- `topoprm_hier_qwen25_7b_v3` (9 benches: aime2024, aime2025, cnmo2024, gpqa_diamond, gsm8k, math500, mmlu, olympiadbench, omni_math)
