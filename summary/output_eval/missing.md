# missing (label, bench) cells per label

Each label is auto-tagged as **public** or **private** based on which benchmark family it has data for, so we only flag missing cells within that family.


## public labels (target: 9 public benches)

- `base_4b_v3`: missing aime2024, aime2025, cnmo2024
- `base_9b`: missing olympiadbench, omni_math, aime2024, aime2025, cnmo2024, mmlu, gpqa_diamond
- `base_9b_v3`: missing aime2024, gpqa_diamond
- `base_qwen25_7b`: missing olympiadbench, omni_math, aime2024, aime2025, cnmo2024, mmlu, gpqa_diamond
- `baseline_dr1_7b`: missing olympiadbench, omni_math, aime2024, aime2025, cnmo2024, gpqa_diamond
- `baseline_dr1_7b_aime`: missing gsm8k, math500, olympiadbench, omni_math, aime2025, cnmo2024, mmlu, gpqa_diamond
- `baseline_dr1_7b_chat`: missing olympiadbench, omni_math, aime2025, gpqa_diamond
- `baseline_dr1_7b_cnmo`: missing gsm8k, math500, olympiadbench, omni_math, aime2024, aime2025, mmlu, gpqa_diamond
- `distill_rkl_8b`: missing olympiadbench, omni_math, aime2024, aime2025, cnmo2024, mmlu, gpqa_diamond
- `grpo_outcome_only_qwen35_9b_B`: missing omni_math, mmlu, gpqa_diamond
- `no_continuity_9b`: missing olympiadbench, omni_math, aime2024, aime2025, cnmo2024, mmlu, gpqa_diamond
- `no_continuity_9b_v3`: missing gpqa_diamond
- `no_topo_9b`: missing olympiadbench, omni_math, aime2024, aime2025, cnmo2024, mmlu, gpqa_diamond
- `no_topo_9b_v3`: missing gpqa_diamond
- `outcome_only_9b`: missing olympiadbench, omni_math, aime2024, aime2025, cnmo2024, mmlu, gpqa_diamond
- `outcome_only_9b_v3`: missing gpqa_diamond
- `ref_qwen25_7b_instruct_B`: **complete**
- `ref_qwen25_math_7b_instruct_B`: **complete**
- `ref_qwen35_2b_B`: **complete**
- `ref_qwen35_4b_B`: missing omni_math, mmlu, gpqa_diamond
- `ref_qwen35_9b_B`: missing omni_math, mmlu, gpqa_diamond
- `ref_qwen35_9b_base_B`: missing mmlu, gpqa_diamond
- `sft_9b`: missing olympiadbench, omni_math, aime2024, aime2025, cnmo2024, mmlu, gpqa_diamond
- `sft_9b_v2`: missing math500, olympiadbench, omni_math, aime2024, aime2025, cnmo2024, mmlu, gpqa_diamond
- `sft_9b_v2_ext`: missing gsm8k, olympiadbench, omni_math, cnmo2024, mmlu, gpqa_diamond
- `sft_9b_v2_ext_k5_gpu1`: missing gsm8k, math500, mmlu, gpqa_diamond
- `sft_9b_v3`: **complete**
- `sft_qwen35_9b_B`: **complete**
- `student_4b_sft_distill_ext_k5_gpu1`: missing gsm8k, math500, aime2024, cnmo2024, mmlu, gpqa_diamond
- `student_4b_sft_distill_v3`: missing gpqa_diamond
- `topoprm_gated_9b`: missing olympiadbench, omni_math, aime2024, aime2025, cnmo2024, mmlu, gpqa_diamond
- `topoprm_gated_9b_v2`: missing olympiadbench, omni_math, aime2024, aime2025, cnmo2024, mmlu, gpqa_diamond
- `topoprm_gated_9b_v2_ext`: missing gsm8k, math500, mmlu, gpqa_diamond
- `topoprm_gated_9b_v2_ext_k5_gpu2`: missing gsm8k, math500, mmlu, gpqa_diamond
- `topoprm_gated_9b_v3`: **complete**
- `topoprm_hier_7b`: missing olympiadbench, omni_math, aime2024, aime2025, cnmo2024, mmlu, gpqa_diamond
- `topoprm_hier_9b`: missing olympiadbench, omni_math, aime2024, aime2025, cnmo2024, mmlu, gpqa_diamond
- `topoprm_hier_9b_continue_v3`: missing mmlu, gpqa_diamond
- `topoprm_hier_9b_v2`: missing math500, olympiadbench, omni_math, aime2024, aime2025, cnmo2024, mmlu, gpqa_diamond
- `topoprm_hier_9b_v2_ext_k5_gpu7`: missing gsm8k, math500, mmlu, gpqa_diamond
- `topoprm_hier_9b_v2_full`: missing olympiadbench, omni_math, mmlu, gpqa_diamond
- `topoprm_hier_9b_v3`: **complete**
- `topoprm_hier_qwen25_7b_ext_k5_gpu7`: missing gsm8k, math500, mmlu, gpqa_diamond
- `topoprm_hier_qwen25_7b_v3`: **complete**

## private labels (target: high + middle)

- `baseline_qwen3_32b_light200_mnt1024_gpu5`: **complete**
- `distill_rkl_8b_compact_light200`: **complete**
- `grpo_clipped_light200_mnt1024_gpu5`: **complete**
- `grpo_gated_qwen35_9b_mcl4096_light200`: **complete**
- `grpo_hier_9b_own_sft_light200`: **complete**
- `grpo_hierarchical_qwen25_7b_light200`: **complete**
- `grpo_hierarchical_qwen35_9b_light200`: **complete**
- `grpo_hierarchical_qwen35_9b_mcl4096_light200`: **complete**
- `grpo_hierarchical_qwen35_9b_mem70_light200`: **complete**
- `grpo_hierarchical_qwen35_9b_ng4_light200`: **complete**
- `grpo_main_full`: **complete**
- `grpo_main_light200`: **complete**
- `grpo_main_light200_mnt1024_gpu6`: **complete**
- `grpo_no_continuity_light200_mnt1024_gpu7`: **complete**
- `grpo_no_continuity_qwen25_7b_light200`: **complete**
- `grpo_no_continuity_qwen35_9b_light200`: **complete**
- `grpo_no_topo_9b_own_sft_light200`: **complete**
- `grpo_no_topo_light200_mnt1024_gpu6`: **complete**
- `grpo_no_topo_qwen25_7b_light200`: **complete**
- `grpo_no_topo_qwen35_9b_light200`: **complete**
- `grpo_no_topo_qwen35_9b_mcl4096_light200`: **complete**
- `grpo_outcome_9b_own_sft_light200`: **complete**
- `grpo_outcome_light200`: **complete**
- `grpo_outcome_light200_mnt1024_gpu7`: **complete**
- `grpo_outcome_only_qwen25_7b_light200`: **complete**
- `grpo_outcome_only_qwen35_9b_light200`: **complete**
- `grpo_outcome_only_qwen35_9b_mcl4096_light200`: **complete**
- `sft_light200`: **complete**
- `sft_private_boost_light200_mnt1024`: **complete**
- `sft_qwen25_7b_boost_light200`: **complete**
- `sft_qwen35_9b_light200`: **complete**
