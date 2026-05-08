OPD
On-Policy Distillation (在线策略蒸馏) 结合强化学习 on-policy特征与蒸馏的强监督，是让模型更稳定、高效训练的方法。在RL中可以on-policy学自己生成的样本，鲁棒切不易遗忘、生成更稳定；但是效率低，一个序列只能给一个奖励，每个token没有监督。而SFT每个token都有监督，训练高效。
OPD的目标是：同时拥有RL的on-policy优势和SFT的高效监督。
标准做法就是：1. 学生模型自己rollout生成完整序列；2. 用教师模型在同一条序列上计算每个位置的logit分布；3. 让学生模型的分布区模仿教师分布，完成蒸馏优化。
OPD原文核心公式：反向KL散度 
(cite https://arxiv.org/abs/2604.13016)
A standard formulation minimizes the sequence-level reverse KL over student-generated trajectories:
$$L_(OP\mathcal{L}_{OPD}(\theta)=\mathbb{E}_{x \sim \mathcal{D}_x} [D_{KL}(\pi_\theta(\cdot | x)||\pi_T(\cdot | x))]$$
Using the autoregressive factorization, this sequence-level objective admits the exact token-level
decomposition:
$$\mathcal{L}_{OPD}(\theta)=\mathbb{E}_{x\sim \mathcal{D}_x, \hat{y} \sim \pi_\theta(\cdot | x)} [\sum_{t=1}^T D_{KL}(p_t||q_t)]$$
让学生模型在自己生成的序列中，逐token逼近教师的概率分布，梯度会对每个token和全词表做监督，比普通的RL信息更密集、训练更高效。

OPD后训练工作
MiMo-V2-Flash Technical Report: https://arxiv.org/abs/2601.02780
[图片]
Qwen3 Technical Report: https://arxiv.org/pdf/2505.09388
4.5 Strong-to-Weak Distillation: On-policy Distillation: In this phase, the student model generates on-policy sequences for fine-tuning. Specifically, prompts are sampled, and the student model produces responses in
either /think or /no think mode. The student model is then fine-tuned by aligning its logits with those of a teacher model (Qwen3-32B or Qwen3-235B-A22B) to minimize the KL divergence.
[图片]
GLM-5:from Vibe Coding to Agentic Engineering https://www.alphaxiv.org/abs/2602.15763
[图片]
Qwen3使用OPD高效训练轻量模型; GLM-5用其修复多阶段RL后的能力遗忘； 小米MiMo-V2通过多教师OPD整合数学、代码、搜索等专家能力；DeepSeek-V4从架构到Infra全栈重构。

OPD Reference
DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence.
A Survey of On-Policy Distillation for Large Language Models
Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe
Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes.
Scaling Reasoning Efficiently via Relaxed On-Policy Distillation / REOPOLD
Distillation through Adaptive Target Reformulation / Veto
Entropy-Aware On-Policy Distillation of Language Models
Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models / OPSD
On-Policy Context Distillation for Language Models / OPCD
SODA: Semi On-Policy Black-Box Distillation for Large Language Models
PACED: Distillation and On-Policy Self-Distillation at the Frontier of Student Competence
Self-distillation分支：不依赖外部teacher的OPD变体
Self-Distillation Enables Continual Learning / SDFT
Reinforcement Learning via Self-Distillation / SDPO
CRISP: Compressed Reasoning via Iterative Self-Policy Distillation
Self-Distilled RLVR / RLSD
OPD从文本推理扩展到多模态/机器人
Video-OPD: Efficient Post-Training of Multimodal Large Language Models for Temporal Video Grounding via On-Policy Distillation
X-OPD: Cross-Modal On-Policy Distillation for Capability Alignment in Speech LLMs
On-Policy Distillation of Language Models for Autonomous Vehicle Motion Planning
