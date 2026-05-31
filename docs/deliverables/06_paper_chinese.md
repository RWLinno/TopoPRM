# 学习奖励然后蒸馏数学推理结构：基于拓扑感知过程信号的方法

> **TopoPRM: Learning to Reward then Distill Mathematical Reasoning Structure with Topology-Aware Process Signals**
> EMNLP 2026 (ARR May cycle)

---

## 摘要

大语言模型中的长思维链推理既非非错即对，也非纯粹线形推导。仅以结果为奖励会忽略不合理的中间步骤，而现有过程奖励模型依赖逐步标注，无法捕捉推理链内部的隐式条件依赖。

我们提出 **TopoPRM**，一种拓扑感知的隐式过程奖励模型：它从每条推理轨迹中提取依赖有向无环图（DAG），将条件依赖与顺序连贯融入多源奖励信号。在 TopoPRM 之上，我们设计后训练框架 **TGSD**（Topology-Guided Self-Distillation），在有监督热启动、以正确性优先的层次化奖励驱动的 GRPO 优化以及拓扑引导的同策略蒸馏中加以应用。

在多个公开基准上，本文的多阶段后训练流水线在 7B 与 9B 规模上均超越标准 GRPO，同时将生成 token 数压缩最多 24%，在基础数学和通用推理任务上均有一致提升。

---

## 1. 引言

![TopoPRM 动机图](TopoPRM_EMNLP26%20(1)/figures/fig1_topoprm_motivation.pdf)
*图 1：TopoPRM 的动机。自由形式的推理轨迹首先被转化为带有类型化节点和边的隐式依赖 DAG，然后通过拓扑感知诊断驱动训练各阶段。*

**背景与问题。** 可验证奖励的强化学习（RLVR）已大幅提升了大语言模型的长思维链推理能力，成为训练数学和符号推理的核心机制。从稀疏的最终答案奖励到提供中间步骤级监督的过程奖励模型（PRM），现有方法统一将推理轨迹视为**线性步骤序列**。然而实际上，长推理几乎从不是线性的：在数学推导中，后续结论常依赖若干非相邻前提，中间等式被多条后续推导复用，子目标在汇合到最终答案前先发生分支。

**现有方法的不足。** 近期推理模型让推理的宏观结构更清晰可靠，却仍停留在"CoT 即结果"的视角：
- **RLVR 一侧**：DeepSeek-R1、DAPO、Tulu 3 证明仅凭结果奖励即可扩展，但无法区分"有支持的推导"与"经无依据中间断言到达同一答案的轨迹"
- **PRM 一侧**：Math-Shepherd、OmegaPRM、GenPRM 等提供更密集的步骤级信号，但都将推理轨迹视为线性序列，逐步赋予局部信用，未建模跨非相邻步骤的依赖结构

**我们的方法。** 我们提出 TopoPRM，用**隐式拓扑依赖结构**而非表面顺序来监督推理。核心思路：
1. 从任意自由形式推理轨迹中提取 DAG 形式的拓扑结构
2. 混合 DAG 打分器给出双重诊断——全局拓扑质量与局部连贯质量
3. TGSD 框架共用同一套拓扑诊断，通过三阶段后训练得到更强且更高效的推理模型

**贡献：**
1. **推理拓扑作为过程监督**：将长 CoT 形式化为隐式依赖 DAG，定义拓扑感知过程信号
2. **统一的奖励与后训练框架**：TopoPRM + TGSD，三阶段共用单一 DAG 打分器
3. **准确率与效率的实证提升**：在 7B/9B 规模上超越标准 GRPO，响应长度压缩最多 24%

---

## 2. 相关工作

### 2.1 可验证奖励的强化学习 (RLVR)

RLVR 用程序化正确性检查替代偏好模型。近期工作通过动态采样和裁剪（DAPO）、延长训练（ProRL）、更细的信用分配（VinePPO）、负样本的非对称处理、熵正则化和简化基线（REINFORCE++）来改进 GRPO 式优化。然而，仅结果信号在结构上是盲目的：具有非常不同依赖支持的轨迹可能获得相同奖励。

*TopoPRM 从隐式推理 DAG 中注入确定性的拓扑和连续性诊断，直接针对结构信用分配，而非通过学习型奖励代理。*

### 2.2 推理的过程奖励模型 (PRM)

PRM 对中间步骤打分以解决结果奖励的稀疏性，最初依赖密集的人工标注。后续工作通过 Monte Carlo 或树搜索 rollout 降低标注成本，生成式 PRM 使验证器推理显式化，隐式 PRM 仅从结果反馈推导步骤级信号。共同局限是：打分仍是步骤局部的，将每步仅视为以其直接前驱为条件。

*TopoPRM 不训练神经验证器，不声称证明级保证，而是引入确定性的全局结构监督——覆盖依赖拓扑和局部连续性。*

### 2.3 在线策略蒸馏与推理压缩

知识蒸馏将能力从大 teacher 迁移到小 student，在线策略变体通过监督 student 生成的轨迹来消除 train-test 不匹配。推理导向的 OPD 包括 OPSDC、ExOPD、BOND、KDRL 等。大多数方法仍优化 token 级匹配目标，缺乏显式结构约束。

*TopoPRM 同时驱动拓扑感知的奖励优化和在线策略自蒸馏，使压缩以结构充分性为条件，而非仅模仿 token。*

---

## 3. 方法

![TopoPRM 框架总览](TopoPRM_EMNLP26%20(1)/figures/fig2_topoprm_framework.pdf)
*图 2：TopoPRM 与后训练框架 TGSD 的总览。DAG 打分器将推理轨迹提升为结构化表示。层次化乘法奖励和 ACE 约束层内信用，拓扑引导的在线策略蒸馏将长 CoT teacher 压缩为紧凑 student。*

TGSD 是一个拓扑感知的后训练框架，其三个阶段共享一个 **DAG 打分器** $\mathcal{E}$：
- **Stage I**：有监督冷启动，强制可解析格式
- **Stage II**：GRPO + 层次化奖励（融合 TopoPRM 与 ACE）
- **Stage III**：拓扑引导自蒸馏（Stage-II teacher 提供 token 级 reverse-KL 监督）

### 3.1 问题设定

设 $x=(q,c)$ 为输入（数学问题 $q$ + 可选上下文 $c$），$\pi_\theta$ 为生成推理轨迹 $y=(s_1,\dots,s_T)$ 的策略。定义 $\mathcal{G}_y=(\mathcal{V},\mathcal{A})$ 为 $y$ 上的有向依赖图，$\mathcal{E}:y\mapsto\mathcal{G}_y$ 为 DAG 打分器。

**目标：** 从 $\mathcal{G}_y$ 中恢复隐式依赖结构，作为 SFT、RL 和蒸馏的统一监督，无需人工步骤标注或学习型验证器。

### 3.2 拓扑感知数据处理

**Trace-to-DAG 提取：** DAG 打分器 $\mathcal{E}$ 分三步进行：
1. **步骤分割与解析**：用编号标记、数学话语标记和行边界切分推理步骤
2. **隐式特征提取**：每步 $s_i$ 归约为三个归一化特征集——数学表达式 $E_i$、规范化声明键 $C_i$、变量引用 $V_i$
3. **DAG 构建**：对每对有序步骤 $(s_i,s_j)$（$i<j$），当 $E_i\cap E_j$、$C_i\cap C_j$ 或 $V_i\cap V_j$ 非空时插入隐式边 $i\to j$

**拓扑与连续性诊断：** 在结果图 $\mathcal{G}_y$ 上计算两个归一化分数：

$$q_{\mathrm{topo}}(\mathcal{G}_y) = \mathbb{1}[|\mathcal{V}|>0] + \mathbb{1}[\mathcal{G}_y\text{ acyclic}] + \mathbb{1}[\rho_{\mathrm{orph}}(\mathcal{G}_y)=0] + \delta(\mathcal{G}_y) + \kappa(\mathcal{G}_y,\mathcal{G}^{\star})$$

$$q_{\mathrm{cont}}(y) = \gamma\,\eta(y) + (1-\gamma)\,\mathbb{1}[\eta(y)=1]$$

$$r_{\mathrm{topo}}(y) = \alpha\,q_{\mathrm{topo}}(\mathcal{G}_y) + (1-\alpha)\,q_{\mathrm{cont}}(y)$$

其中 $\rho_{\mathrm{orph}}$、$\delta$、$\kappa$ 分别度量孤儿结论节点比例、边的前提-结论方向一致性、与参考 DAG 的对齐度；$\eta(y)$ 是步骤可追溯比例。

### 3.3 层次化奖励聚合

加权求和奖励在高过程分补偿错误最终答案时会产生误排序。我们采用**乘法聚合**，将 $r_{\mathrm{topo}}$ 作为结果锚定基础上的结构调制器：

$$r_{\mathrm{total}}(x,y) = \bigl(w_o\,r_{\mathrm{out}} + w_f\,r_{\mathrm{fmt}} + w_l\,r_{\mathrm{len}}\bigr) \cdot \mathrm{Norm}_{\sigma}\!\bigl(1 + r_{\mathrm{topo}}(y)\bigr)$$

**正确性优先保证**：$r_{\mathrm{out}}=0$ 的轨迹不可能被任何过程 bonus 提升；结果分相同的轨迹通过依赖结构区分。

### 3.4 拓扑引导后训练

**Stage I: 有监督热启动。** 在结构化数学推理数据上进行 SFT，教会模型遵循所需响应格式、产生逐步解答、暴露可被 $\mathcal{E}$ 解析的中间推理。

**Stage II: GRPO + TopoPRM 奖励。** 从 SFT checkpoint 出发，用 GRPO 在 TopoPRM 奖励下优化策略。结果奖励选择正确最终答案，拓扑和连续性奖励鼓励模型使中间依赖显式化、避免无支持结论、减少结构冗余推理。

**非对称裁剪优势估计器 (ACE)：** 乘法结构奖励引入潜在的 hacking 风险。ACE 将正确性分层信用分配与标准 GRPO z-score 裁剪融合为单一估计器：

$$\widehat{A}_i = \begin{cases} \mathrm{clip}\bigl(z_i^{\mathcal{C}} + \max(0,\Delta r_{\mathrm{aux}}^{(i)}),\, 0,\, \overline{c}\bigr), & i\in\mathcal{C} \\ \mathrm{clip}\bigl(z_i^{\mathcal{W}} + \min(0,\Delta r_{\mathrm{aux}}^{(i)}),\, \underline{c},\, 0\bigr), & i\in\mathcal{W} \end{cases}$$

非对称 $\max/\min$ 确保结构信号只能在正确性层**内部**重排轨迹，永远不能翻转其符号。

**Stage III: 拓扑引导自蒸馏。** Stage-II 优化后的模型作为 teacher $\pi_T$，较小的预训练 checkpoint 初始化 student $\pi_\theta$。对每个 prompt $x$：
1. Student 先 rollout 自己的响应 $y^{\star}\sim\pi_\theta(\cdot|x)$
2. 同一提取器 $\mathcal{E}$ 产生 $\mathcal{G}_{y^{\star}}$ 及诊断分数
3. 调度器 $\psi$ 发出修订指令 $P_r$，针对 $\mathcal{G}_{y^{\star}}$ 中的孤儿结论位置
4. Teacher 在 $P_r$ 下监督 student

蒸馏目标：

$$\mathcal{L}_{\mathrm{TGSD}}(\theta) = \mathbb{E}_{x,y\sim\pi_\theta}\sum_{t=1}^{|y|} \mathcal{D}_{\mathrm{RKL}}\bigl(p_t \| q_t\bigr)$$

**关键依赖链作为压缩目标：** 通过步骤类型收缩、层聚合、传递稀疏化三个操作，将 DAG 压缩为"关键依赖链"——类似项目调度中的关键路径——student 被训练恢复这条最小链，而非匹配 teacher 的 token 频率。

---

## 4. 实验

### 4.1 设置

在 9 个 benchmark 上进行广泛实验：GSM8K、MATH-500、OlympiadBench、Omni-MATH、AIME 2024、AIME 2025、CNMO 2024、MMLU、GPQA-Diamond。

基座模型：Qwen3.5-9B、Qwen2.5-7B、DeepSeek-R1-Distill-Qwen-7B。

训练配置：
- SFT: 3 epochs, ~19.5K 样本, LoRA rank=64
- GRPO: 300 steps, 4 generations/prompt, lr=5e-6, β=0.04
- 硬件: 8×A100-80GB

### 4.2 主要结果 (RQ1)

TopoPRM 在三个基座模型上均达到最高平均准确率（55.3%、58.3%、60.9%）。

在 Qwen2.5-7B 上，TopoPRM 在 9 个 benchmark 中的 7 个超过 GRPO baseline，MMLU 提升 +20.2 点（82.7 vs 62.5），表明拓扑监督可迁移到通用推理。

**更短的链，更高的效率：** TopoPRM (Full) 在所有四个主要 benchmark 上达到最高 Acc/kTok（GSM8K: 118.2, MATH-500: 35.1, OlympiadBench: 18.6, Omni-MATH: 26.6），生成 token 比 outcome-only GRPO 少 15-24%。

### 4.3 消融实验 (RQ2)

| 配置 | Avg Acc | Collapse% | Tokens | Acc/kTok |
|------|---------|-----------|--------|----------|
| TopoPRM (Full) | 57.2 | 37.9% | 1,692 | 26.9 |
| w/o Continuity | 27.9 | 68.8% | 2,046 | 14.0 |
| w/o Topology | 53.4 | 42.1% | 1,812 | 23.1 |
| w/o ACE | 53.4 | 45.3% | 1,756 | 24.2 |
| Outcome-only | 52.5 | 57.8% | 1,737 | 37.9 |

去除连续性导致最大退化（Δ=-29.3），Collapse% 从 37.9% 升至 68.8%，表明局部步骤可追溯性提供了 GRPO 的主要梯度信号。

### 4.4 效率分析 (RQ3)

![Token 效率对比](TopoPRM_EMNLP26%20(1)/figures/Fig5.Efficiency_Barplot.pdf)
*图 3：各 reward 配置的 token 效率（Qwen3.5-9B，4-benchmark 平均）。*

TopoPRM (Full) 达到 48.1 Acc/kTok（1,568 tokens），outcome-only GRPO 为 37.9（1,737 tokens），w/o-continuity 崩溃至 14.0（2,046 tokens）。

**训练成本：** 完整 TopoPRM 后训练在 8×A100-80GB 上约 30 GPU-hours（SFT ~3h, GRPO ~15h, TGSD ~12h）。DAG 提取并行运行，几乎不增加 wall-clock 开销。

### 4.5 跨尺度蒸馏 (RQ4)

| Student | GSM8K | MATH-500 | Token Ratio | Structural Retention |
|---------|-------|----------|-------------|---------------------|
| Teacher (9B) | 84.3 | 66.6 | 1.0× | 100% |
| 4B (TGSD) | 82.8 | 61.5 | 0.56× | 93% |
| 2B (TGSD) | 77.5 | 55.2 | 0.48× | 87% |
| 0.8B (TGSD) | 71.2 | 48.8 | 0.42× | 81% |

4B student 保留 teacher 98.2% 的 GSM8K 准确率和 92.3% 的 MATH-500 准确率，同时将 token 使用降至 0.56×，结构保留率 93%。

---

## 5. 结论

我们提出了 TopoPRM——一种拓扑感知的过程奖励模型，以及其后训练框架 TGSD。该方法将长推理轨迹重铸为隐式依赖 DAG，复用单一套结构诊断同时用于奖励塑形和推理压缩。混合图提取器产生隐式过程奖励，正确性优先的层次化聚合通过 ACE 约束结构信用，TGSD 以关键依赖链为压缩目标进行蒸馏。

这些组件解决了仅结果监督无法区分的失败模式，同时不需要人工步骤标注、学习型验证器或图格式解码。

---

## 局限性

TopoPRM 提供的是结构过程信号而非证明级验证器。局部看似合理但语义错误的步骤仍可能获得正面结构信用。在竞赛级 benchmark（AIME/CNMO）上，outcome-only GRPO 在 9B 规模可超过 TopoPRM，表明结构奖励以牺牲单一推导的尖锐准确性换取更广泛的泛化。

未来方向：
1. 训练轻量依赖模型收紧混合提取器
2. 将拓扑感知奖励与生成式过程验证器耦合
3. 扩展到代码、科学和多模态推理领域
