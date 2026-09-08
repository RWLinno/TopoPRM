# Title
Learning to Reward then Distill Mathematical Reasoning Structure with Topology-Aware Process Signals
 
# Abstract
大语言模型中的长思维链推理既非非错即对，也非纯粹线形推导。仅以结果为奖励会忽略不合理的中间步骤，而现有过程奖励模型依赖逐步标注，无法捕捉推理链内部的隐式条件依赖。我们提出 TopoPRM，一种拓扑感知的隐式过程奖励模型：它从每条推理轨迹中提取依赖有向无环图，将条件依赖与顺序连贯融入多源奖励信号。在 TopoPRM 之上，我们设计后训练框架 TGSD，在有监督热启动、以正确性优先的层次化奖励驱动的 GRPO 优化以及拓扑引导的同策略蒸馏中加以应用。在多个公开基准上，本文的多阶段后训练流水线在 7B 与 9B 规模上均超越标准 GRPO，同时将平均响应长度压缩 11.2%。

### 动机、挑战和对应的解决方案
1. 数学推理中经常存在跨步依赖、分支合并、公式复用和中间结论复用。=> 长 CoT 推理的结构不应是顺序线性的,推理序列存在隐式依赖关系。 => DAG Scorer把推理轨迹表示为存在顺序边和条件依赖边的图，从而编码为奖励信号。
2. 传统 PRM 成本高，不适合轻量在线 reward => TopoPRM而是用 deterministic DAG scorer 从推理图中计算结构指标，作为 implicit process reward。
3. teacher 的输出可能很长、存在冗余分支，或者包含一些不必要的中间步骤。如果 student 直接模仿完整 token 序列，推理成本高,学到可能是表面线性文本，而不是关键依赖结构。=> 提出 TGSD，用拓扑诊断指导 on-policy self-distillation. 先让 student 自己生成推理轨迹，然后用同一个 DAG extractor 和 scorer 分析 student 输出的结构问题，再让 teacher 根据这些拓扑诊断进行 revision 或 compression，最后用 teacher revision 监督 student。TGSD 的优势是:a.student 从自己的错误状态中学习，train-test mismatch 更小; b. teacher revision 被 topology diagnostics 条件化，更关注结构缺陷；c.蒸馏目标不是盲目模仿完整长 CoT，而是保留关键 dependency path；d.可以减少冗余分支，提高 token efficiency。

### 提出概念和整体方法介绍
TopoPRM: 我们的方法名，提出拓扑结构敏感的Implicit Process Reward Model，由DAG打分器计算拓扑奖励和格式、长度、答案准确度等多源奖励信号分层聚合而来.
TGSD: 整个后训练蒸馏优化框架, Topology-Guided Self-Distillation
SCAE: Stratified Clipping Advantage Estimatio, 沿用先前工作的优势估计方法 https://arxiv.org/abs/2601.12995
GoT: Graph-of-Thoughts,把推理以图的形式呈现出来

从推理链中的隐式条件依赖这一问题上出发，我们对数据进行了DAG提取和采样时的打分(DAG Scorer)，这里提供了拓扑结构敏感的隐式奖励信号r_{topo}(q_topo和q_cont门控得到)，会跟着格式分、长度奖惩等一块进行分层多源奖励聚合，构成我们最终的PRM。我们的后训练框架包含了Stage1. SFT冷启动(这是常规操作，好处有比方说更快掌握正确格式等等你可以结合调研补充)；Stage2. 用我们提的奖励函数进行GRPO，充分提升模型的推理能力和泛化能力，这其中也用到了一种叫做SCAE的优势估计技术，至此我们已经得到很好的效果了；然后Stage3则需要用到Topology-Guided On-Policy Distillation技术(也算一种novelty，但本质上是用OPD再训一遍)做一遍优化，OPD会让我们stage2训好的Teacher去做token-level的监督，Student自己rollout完整序列，让学生模型模仿教师分布，同时RKL完成蒸馏优化，比普通RFT信息更密集、训练更高效。理论上，这一过程会让带有graph信息的reasoning-trace变得极度高效，通过正确的提示引导就可以让一个DAG信息在拓扑结构上进行层次合并、缩点缩环，从而形成精准推理链。

### 方法Outlin
1. Problem Setup
2. Topology-aware Process Reward Model
   - Deterministic Trace-to-DAG Scorer: 包含DAG转化编码和条件依赖&连贯性打分deterministic
   - Hierarchical Reward Aggregation: 仍以final-answer correctness为核心，融入层次化辅助奖励结构，采用SCAE优势估计约束
3. Topology-Guided Post-Training
   - Stage I. Supervised Warm-Start: SFT做冷启动
   - Stage II. GRPO with TopoPRM Rewards: 使用提出的多源奖励模型做RFT
   - Stage III. Topology-Guided Self Distillation: 自蒸馏提高token利用率?
4. Distillation and Optimization: RLK优化学生模型

### Reference
https://github.com/opendilab/awesome-RLVR
https://github.com/RyanLiu112/Awesome-Process-Reward-Models
https://github.com/chrisliu298/awesome-on-policy-distillation
可以从上述仓库入手找到值得参考的文章，或者自行调研

### 实验信息
conda环境/screen窗口指定: topoprm
训练框架: ms-swift (本地路径../ms-swift)
火山云代理加速：export ALL_PROXY=http://accelerator-cname-hnpmnhnmdul3rmxrwhgend.c.vegalb.com:80
benchmark: (GSM8K/MATH-500/Olympiad/Omni-MATH/AIME'24/AIME'25	CNMO'24/MMLU/GPQA-D)
metrics：#accuracy (error/correct/F1/pass@1(唯一必需)/pass@k/maj@k/prm@k) #Tokens (Acc/kTok, Tok/s)
baseline越多越好，目前我的base model为deepseek-r1-7b和qwen35-9b，变体为SFT、GRPO、（可选的有DAPO和DPO）
github/huggingface账号: rwlinno 1264532114@qq.com
仓库：https://github.com/RWLinno/TopoPRM/
github token:<redacted>
hf token: <redacted>
wandb API KEY: <redacted>_EBhzyR2Xgjj7Hjgn15I1AzfbFXbqHiuMEegl7tyC3Ef1mVXQn

### 实验内容
1. 主表1全benchmark的pass@k accuracy,对比我们方法以及各种variants
2. 主表2部分benchmark的各项指标，也包含tokens/time效率
3. 消融包含:1.w/o cont; 2.w/o topo; 3.outcome-only; 4. w/o scae;
SFT (Stage I) ~3.0 GPU-hours
GRPO (Stage II, 300 steps) ~25 GPU-hours
TGSD (Stage III, 200 steps) ~15 GPU-hours

### TBD
20260523: 要加baseline、figure要修、测评结果还要优化
20260709: rebuttal