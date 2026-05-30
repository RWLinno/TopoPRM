# TopoPRM 可视化呈现

## 1. DAG 提取可视化

TopoPRM 的核心是从推理链中提取依赖 DAG。以下展示了 9 个 benchmark 上的 DAG 提取案例。

### 1.1 GSM8K（小学数学）

GSM8K 的推理链通常较短（3-5 步），DAG 结构清晰：

![GSM8K Case 1](../assets/dag_cases/gsm8k/showcase_gsm8k_0_Gretchen_has_110_coins_There_ar.png)
![GSM8K Case 2](../assets/dag_cases/gsm8k/showcase_gsm8k_1_Melanie_is_a_door_to_door_salesw.png)
![GSM8K Case 3](../assets/dag_cases/gsm8k/showcase_gsm8k_2_Darrell_and_Allen_s_ages_are_in.png)

**观察：** GSM8K 的 DAG 通常是线性链或简单分支，节点间依赖关系明确。

### 1.2 MATH-500（高中数学）

MATH 题目的推理链更长，出现分支合并：

![MATH Case 1](../assets/dag_cases/math500/showcase_math500_0_Define_p_sum__k_1_infty.png)
![MATH Case 2](../assets/dag_cases/math500/showcase_math500_1_Let_p_x_be_a_polynomial_of_de.png)
![MATH Case 3](../assets/dag_cases/math500/showcase_math500_2_The_set_of_points_x_y_z_that.png)

**观察：** MATH 题目的 DAG 出现了多分支合并模式——多个中间结论汇聚到最终答案。

### 1.3 AIME 2024（竞赛数学）

AIME 的推理链最长，DAG 结构最复杂：

![AIME Case 1](../assets/dag_cases/aime2024/showcase_aime2024_0_There_exist_real_numbers_x_and.png)
![AIME Case 2](../assets/dag_cases/aime2024/showcase_aime2024_1_Every_morning_Aya_goes_for_a_9.png)
![AIME Case 3](../assets/dag_cases/aime2024/showcase_aime2024_2_Jen_enters_a_lottery_by_picking.png)

**观察：** AIME 的 DAG 呈现深层嵌套结构，存在大量跨步依赖（非相邻步骤间的引用）。

### 1.4 AIME 2025

![AIME25 Case 1](../assets/dag_cases/aime2025/showcase_aime2025_0_Find_the_sum_of_all_integer_base.png)
![AIME25 Case 2](../assets/dag_cases/aime2025/showcase_aime2025_1_The_9_members_of_a_baseball_team.png)
![AIME25 Case 3](../assets/dag_cases/aime2025/showcase_aime2025_2_On_triangle_ABC_points_A_D.png)

### 1.5 CNMO 2024（中国数学奥林匹克）

![CNMO Case 1](../assets/dag_cases/cnmo2024/showcase_cnmo2024_0_What_is_the_product_of_all_real.png)
![CNMO Case 2](../assets/dag_cases/cnmo2024/showcase_cnmo2024_1_How_many_ways_are_there_to_split.png)
![CNMO Case 3](../assets/dag_cases/cnmo2024/showcase_cnmo2024_2__frac_m_n_is_the_Irreducible.png)

### 1.6 OlympiadBench

![Olympiad Case 1](../assets/dag_cases/olympiadbench/showcase_olympiadbench_0_What_is_the_smallest_value_of_x.png)
![Olympiad Case 2](../assets/dag_cases/olympiadbench/showcase_olympiadbench_1_Find_the_sum_of_all_integers_tha.png)
![Olympiad Case 3](../assets/dag_cases/olympiadbench/showcase_olympiadbench_2_Evaluate_i_5_i_25_i_45.png)

### 1.7 Omni-MATH

![Omni Case 1](../assets/dag_cases/omni_math/showcase_omni_math_0_Find_x_such_that_lceil_x_rc.png)
![Omni Case 2](../assets/dag_cases/omni_math/showcase_omni_math_1_What_is_the_smallest_value_of_x.png)
![Omni Case 3](../assets/dag_cases/omni_math/showcase_omni_math_2_Find_the_sum_of_all_integers_tha.png)

### 1.8 GPQA-Diamond（研究生级）

![GPQA Case 1](../assets/dag_cases/gpqa_diamond/showcase_gpqa_diamond_0_A_spin_half_particle_is_in_a_lin.png)
![GPQA Case 2](../assets/dag_cases/gpqa_diamond/showcase_gpqa_diamond_1_trans_cinnamaldehyde_was_treated.png)
![GPQA Case 3](../assets/dag_cases/gpqa_diamond/showcase_gpqa_diamond_2_Two_quantum_states_with_energies.png)

### 1.9 MMLU

![MMLU Case 1](../assets/dag_cases/mmlu/showcase_mmlu_0_Find_the_degree_for_the_given_fi.png)
![MMLU Case 2](../assets/dag_cases/mmlu/showcase_mmlu_1_Find_all_zeros_in_the_indicated.png)
![MMLU Case 3](../assets/dag_cases/mmlu/showcase_mmlu_2_Let_p_1_2_5_4_2_3_in_S.png)

## 2. 训练动态曲线

训练过程中的 reward、completion length、token efficiency 变化：

**关键观察：**
- TopoPRM (Full) 从训练开始就保持较短的 completion length（<500 tokens）
- Outcome-only GRPO 的 completion length 从 3700 逐渐压缩到 1200
- w/o-continuity 变体出现 reward collapse（reward std 趋近 0）

## 3. Token 效率对比

各 reward 配置的 token 效率对比：

| 配置 | 平均 Tokens | Acc/kTok | 特点 |
|------|-------------|----------|------|
| TopoPRM (Full) | 1,568 | 48.1 | 最短、最高效 |
| w/o Topology | 1,812 | 23.1 | 中等 |
| w/o ACE | 1,756 | 24.2 | 中等 |
| Outcome-only | 1,737 | 37.9 | 较长 |
| w/o Continuity | 2,046 | 14.0 | 最长、最低效 |

## 4. TGSD 框架可视化

TGSD（Topology-Guided Self-Distillation）的完整流程：

```
Student rollout → DAG extraction → Topology diagnostics →
Revision prompt → Teacher supervision → RKL optimization
```

详细的交互式可视化见 `docs/tgsd_framework.html`。

## 5. DAG Showcase（交互式）

完整的 DAG 提取案例展示见 `docs/dag_showcase.html`，包含：
- 原始推理 trace
- 提取的 DAG 结构（节点 + 边）
- 各项结构指标（valid_dag, acyclic, no_orphan, direction, step_alignment）
- r_topo 计算过程

## 6. 结构对比

### 6.1 好的 DAG vs 差的 DAG

**好的 DAG 特征：**
- 无环（acyclic=1.0）
- 无孤儿结论（no_orphan=1.0）
- 方向一致（direction>0.9）
- 步骤对齐（step_alignment>0.9）

**差的 DAG 特征：**
- 存在孤儿结论（结论没有前置依赖支撑）
- 方向不一致（后面的步骤引用了前面未出现的表达式）
- 步骤数与 DAG 节点数不匹配

### 6.2 TopoPRM vs Outcome-only 的输出对比

TopoPRM 训练后的模型倾向于：
- 更短的推理链（减少冗余分支）
- 更明确的步骤间引用（减少孤儿结论）
- 更线性的依赖结构（减少不必要的分支合并）
