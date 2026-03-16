# *TopoPRM: Efficient and Verifiable* Mathematical Reasoning via Topological Process Reward Model

业务做的是教育批注，数学解题批改这一块。主要想研究RLVR (**基于可验证奖励的强化学习**)和奖励模型 (Reward Models?Reasoning Models?)。我们能拿到的数据是单模态的，只有解答过程和对错的标签。后面也有转到多模态<图片-语料-标注>的想法。

![f81f73bcae3ed4e063e05746b7cb94a8.png](attachment:c1117d8c-eb12-414f-908b-1d44e7013507:f81f73bcae3ed4e063e05746b7cb94a8.png)

（可以分析下现在的思维链和错因有什么问题，日志里的llm_result字段就是模型输出的全部内容）

Mentor：训练一个针对思维链和错因的RM，用作清洗训练数据/参与人工标注数据飞轮/接入RL作为奖励信号

**我的想法**：Challenges / Motivation → 解决方案如下：

C1. 高质量推理链（Rollouts）消耗大量资源 ⇒ Latent Reasoning？或者推理压缩，能参考这篇文章：https://arxiv.org/pdf/2603.05433

C2. 现有RM往往将推理步骤视为独立的局部决策，或者仅依赖答案提供稀疏监督，而数学解题推导过程中具有拓扑性质(线性或DAG) ⇒ 提出一个新Topological Process Reward Model，其奖励信号应建立在连续正确的前缀过程，而不是独立分布在每一步推理上

- **目标A：做一个更好的数学过程奖励模型 PRM**
- **目标B：把“解题步骤”建成 DAG / 拓扑结构**
- **目标C：用 RLVR / RFT 训练出“推理更强且更简洁”的批改模型**

方法最好是基于VeRL去改，我们已经能够考虑解决已经有了一个可以参考这篇文章：https://openreview.net/pdf?id=MQV4TJyqnb

其他讨论点：

- Optimization：通过把PRM过程定义成拓扑结构的验证器而非打分器，Reward里至少要结合VerifiableAnswer + TopoPRMScore两者；
- Reward Hacking：传统 PPO/GRPO 在长推理链下极易发散，模型通过重复正确但不相关的推论来“骗取”PRM分数，我们希望推理过程是推理链压缩到更精简高效的；

### Motivation

在复杂数学题批改场景中，现有后训练方法通常只依赖最终答案奖励，或对中间步骤进行局部独立打分，缺乏对推理结构和步骤连续性的建模。这会导致模型容易生成局部看似合理、但整体依赖关系错误或冗余的长推理链。

### Core Idea

我们提出一种简单的 process-level RLVR 方法：在经典数学推理 reward 基础上，加入两个可验证过程信号：

1. **Topological Reward**：判断整条推理是否构成合法、无环、方向一致的依赖结构；
2. **Continuity Reward**：判断每一步是否可以由前面的步骤正确推出。

这两个信号均可通过规则、符号计算或程序验证得到，因此属于 verifiable rewards。它们与原有 reward 联合用于 RFT，使模型不仅学会“答案正确”，还学会“推理结构合理、过程连续”。

### Compression

为缓解长链推理中的 reward hacking，我们进一步对训练得到的 teacher 推理图进行压缩，将冗余链段和重复改写缩并为更小的 support graph，再蒸馏到 student 模型。由此得到的 student 在保持正确性的同时生成更简洁、更稳定的批改链。

### Expected Contributions

1. 将 RLVR 从 outcome-level reward 扩展到 process-level verifiable reward；
2. 提出 topology + continuity 两个简单有效的结构奖励；
3. 在数学批改任务上提升推理质量、错误定位能力和链式推理紧凑性。

# Abstract

复杂数学推理不仅要求模型输出正确答案，还要求其生成结构清晰、前后连贯且尽可能紧凑的推导过程。然而，现有后训练方法大多依赖结果级奖励或局部步骤打分，缺乏对推理链整体结构的显式建模，因而容易产生依赖关系错误、步骤衔接不连续以及冗余冗长的推理过程。这一问题在复杂数学题批改与长链推理场景中尤为突出，也限制了强化学习方法的训练稳定性与推理效率。

为此，我们提出 **TopoPRM**，一个结合拓扑思想的**强化微调（RFT）与推理压缩**的统一框架，用于学习结构感知的数学推理过程。TopoPRM 不再仅将推理视为若干局部步骤的线性序列，而是从推理依赖结构出发建模过程质量。具体而言，我们设计了两个新的**可验证奖励信号**：其一是**拓扑结构奖励**，用于衡量整条推理过程是否满足无环、依赖一致的有效结构；其二是**前后连续奖励**，用于判断每一步是否能够由此前上下文正确推出。这两个奖励均由显式规则或程序自动验证得到，属于一种过程级的 **RLVR（Reinforcement Learning with Verifiable Rewards）** 方法。在此基础上，我们使用**反向KL散度**进一步对 RFT 学到的长推理链进行**结构压缩**，并将压缩后的紧凑推理模式蒸馏到学生模型中，缓解长链推理中的冗余展开与 reward hacking 问题。

实验结果表明，TopoPRM 在复杂数学推理任务上能够显著提升推理过程的结构一致性与步骤连续性，并在保持或提升任务性能的同时生成更短、更紧凑的推理链。结果说明，从推理拓扑与过程连续性的角度设计可验证奖励，是提升长链数学推理质量与压缩效率的一条有效路径。

We study reinforcement fine-tuning for complex mathematical critique, where models are expected not only to produce correct judgments, but also to provide coherent and concise reasoning about intermediate solution steps. Existing approaches mainly rely on outcome-level rewards or step-wise process supervision, while largely ignoring the structural dependency of mathematical reasoning. We propose a simple process-level RLVR framework that augments conventional reasoning rewards with two verifiable signals: **topological reward**, which evaluates whether a reasoning trajectory forms a valid DAG-like dependency structure, and **continuity reward**, which checks whether each step can be correctly derived from its preceding context. Since both signals are obtained through explicit verification procedures rather than subjective preference modeling, our method remains within the RLVR paradigm. We further introduce an answer-preserving compression stage that contracts redundant reasoning chains into compact support graphs and distills them into a student model, mitigating reward hacking from unnecessarily long chains. We expect this framework to improve mathematical critique quality, step-level consistency, and reasoning compactness in educational annotation settings.

# Related Work

### **基于可验证奖励的强化学习 (**RLVR)

### 多智能体强化学习 (MARL)

# Method

### CPT训练集

### SFT-then-RFT

### RLVR：

# Experiments

### Codebase选择：

https://github.com/OpenRLHF/OpenRLHF：比较适合做 SFT / RM / PPO 类强化学习，DPO / KTO / ORPO 这类偏好优化

https://github.com/modelscope/ms-swift：快速训ckpts，少写很多底层训练 plumbing，可以RM / GRPO类训练为主。

https://github.com/verl-project/verl

基座模型的选择：Qwen2.5-Math系列/Deepseek-Math-v2或许对这类问题更好？否则直接选最新的Qwen3.5/Intern去做。(关于API/Token怎么报销的问题？)

### Baselines

### 新数据集

主表：各种baseline + math model在中文解题上的表现

### 测评集（公开）

外文：MATH-500, MATH, GSM8K, AIME2025, MMLU, AMC23, Minerva 和 Olympiad

中文：

### 未整理文献

REINFORCEMENT LEARNING WITH VERIFIABLE RE-WARDS IMPLICITLY INCENTIVIZES CORRECT REA-SONING IN BASE LLMS

Heterogeneous Agent Collaborative Reinforcement Learning

[ICLR 2026] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling

On-Policy Self-Distillation for Reasoning Compression

| Unified Framework for Policy Divergence in GRPO | arXiv:2602.05494 | 2026-02-05 | Qingyuan Wu et al. | GRPO 策略发散度量的统一框架，提出 KL3 估计器（方差减少的蒙特卡罗 KL 散度估计器）作为关键策略发散约束 |
| --- | --- | --- | --- | --- |
| Proof-RM: Math Proof Reward Model | arXiv:2602.02377 | 2026-02-02 | Haotong Yang et al. | 可扩展且可泛化的数学证明奖励模型，针对数学证明领域设计 |
| Grad2Reward: Sparse to Dense Rewards | arXiv:2602.01791 | 2026-02-02 | Zheng Zhang et al. | 从稀疏判断到密集奖励的转换，改进开放式 LLM 推理 |
| Harder Is Better: Difficulty-Aware GRPO | arXiv:2601.20614 | 2026-01-28 | Yanqi Dai et al. | 难度感知 GRPO + 多方面问题重构，根据问题难度动态调整训练策略 |
| Length-Unbiased Sequence Policy Optimization | arXiv:2602.05261 | 2026-02-04 | Fanfan Liu et al. | 长度无偏序列策略优化，揭示和控制 RLVR 中响应长度变化 |
| P2S: Probabilistic Process Supervision | arXiv:2601.20649 | 2026-01-28 | Wenlin Zhong et al. | 概率过程监督，用于通用领域推理问答 |
| Thickening-to-Thinning: Reward Shaping | arXiv:2602.04265 | 2026-02-04 | Wenze Lin et al. | 受人类学习动态启发的奖励塑造，从厚到薄的学习过程 |

[Paper Reading (Prompts)](https://www.notion.so/007f4b78d6b44888850b66b276ee395a?pvs=21)

https://github.com/OpenRLHF/OpenRLHF

https://zzx-peter.github.io/hacrl/

### 项目初始化prompt
```
# TopoPRM 项目初始化 Prompt

请你帮我创建一个完整的研究项目 `TopoRM`，用于基于 ms-swift 框架(不要对基本框架进行任何修改)对 Qwen3-14B 进行 SFT + GRPO(RFT) 训练，最终目标是训练一个中文数学批改模型，核心创新点是引入拓扑结构的过程奖励。

## 项目背景

- 基座模型：Qwen/Qwen3-14B（通过 ms-swift 训练）
- 训练阶段：SFT → GRPO(RFT with custom reward) → 蒸馏
- 业务场景：中文数学题解题过程批改（错因分析、步骤批改、评分）
- 核心创新：把数学推理步骤建模为 DAG 依赖图，设计 topological reward + continuity reward 用于 RLVR

## 项目文件结构

请严格按照以下结构创建下面目录，每个文件都要有完整可运行的代码，可以参考结构（不要占位符）：

```
topoprm/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── sft_qwen3_14b.yaml          # ms-swift SFT 配置
│   ├── grpo_qwen3_14b.yaml         # ms-swift GRPO 配置
│   └── eval_config.yaml            # 评估配置
├── scripts/
│   ├── download_models.sh           # 下载模型
│   ├── download_benchmarks.sh       # 下载数学benchmark
│   ├── run_sft.sh                   # 启动SFT训练
│   ├── run_grpo.sh                  # 启动GRPO训练
│   ├── run_eval.sh                  # 启动评估
│   └── run_data_pipeline.sh         # 一键跑数据处理全流程
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── parse_raw.py             # 解析原始业务JSON数据
│   │   ├── build_dag.py             # 从解析后数据构建 DAG
│   │   ├── prepare_sft.py           # 转为ms-swift SFT格式
│   │   ├── prepare_grpo.py          # 转为ms-swift GRPO格式
│   │   ├── clean.py                 # 数据清洗脚本
│   │   └── merge_datasets.py        # 混合中英文数据
│   ├── dag/
│   │   ├── __init__.py
│   │   ├── node.py                  # Node 数据结构定义
│   │   ├── graph.py                 # DAG 构建、验证、可视化
│   │   └── compress.py              # DAG 压缩（去冗余）
│   ├── reward/
│   │   ├── __init__.py
│   │   ├── outcome_reward.py        # 答案正确性奖励
│   │   ├── topo_reward.py           # 拓扑结构合法性奖励
│   │   ├── continuity_reward.py     # 步骤连续性奖励
│   │   ├── format_reward.py         # 输出格式合规奖励
│   │   └── composite_reward.py      # 组合奖励函数（给ms-swift用）
│   └── eval/
│       ├── __init__.py
│       ├── benchmark_runner.py      # 跑benchmark的评测脚本
│       └── critique_eval.py         # 批改质量的专项评估
├── data/
│   ├── raw/                         # 放原始业务数据
│   │   └── .gitkeep
│   ├── processed/                   # 清洗后的数据
│   │   └── .gitkeep
│   ├── dag/                         # DAG JSON 文件
│   │   └── .gitkeep
│   ├── sft_ready/                   # SFT 训练数据
│   │   └── .gitkeep
│   ├── grpo_ready/                  # GRPO 训练数据
│   │   └── .gitkeep
│   └── benchmarks/                  # 下载的benchmark
│       └── .gitkeep
├── docs/
│   ├── reward_design.md             # 自定义Reward详细设计文档
│   ├── dag_schema.md                # DAG结构规范文档
│   ├── ms_swift_custom_reward.md    # 如何在ms-swift中接入自定义reward
│   └── training_pipeline.md         # 训练流水线文档
└── tests/
    ├── test_parse_raw.py
    ├── test_build_dag.py
    ├── test_rewards.py
    └── test_graph.py
```

## 详细要求

---

### 1. 数据解析与 DAG 构建

#### 1.1 `src/data/parse_raw.py`

原始数据格式如下（这是一个API返回的嵌套JSON，最外层有 header 和 payload，核心内容在 `payload.output.content` 中，content 本身是一个 JSON 字符串需要二次 json.loads 解析）：

```json
{
  "header": {"code": 0, "message": "success", "sid": "xxx", "status": 2},
  "payload": {
    "output": {
      "status": "finish",
      "content": "{\"data\":{\"answerAreas\":[],\"id\":\"0_1_1\",\"stepCorrectInfo\":{\"correctInfo\":{\"rejectInfo\":{\"rejectReason\":\"\",\"rejectType\":0},\"scoreInfo\":{\"procedureScore\":0.0,\"score\":-1.0}},\"extendInfos\":[{\"key\":\"StepCorrProcess\",\"value\":{\"corret_method\":\"ApplicationByLLM\",\"errorReason\":\"success\",\"llm_analysis\":\"略\",\"llm_prompt\":\"...(包含题目、标准答案、学生作答的完整prompt)...\",\"llm_result\":\"...(模型输出的完整批改思维链和JSON结果)...\",\"llm_ret\":0,\"llm_stdanswer\":\"...(标准答案全文)...\",\"llm_stem\":\"...(题干)...\",\"llm_sysprompt\":\"你是一名资深、严谨、和蔼的高中数学老师\",\"llm_user\":\"...(学生作答)...\",\"prompt_name\":\"StepCorrProcess\"}}],\"subCorrectInfos\":[...]},\"subTopics\":[],\"topicId\":\"xxx\"}}"
    }
  }
}
```

请实现以下功能：
- `parse_single_record(raw_json: dict) -> dict`：从嵌套JSON中提取出干净的训练形式：`stem`(题干), `standard_answer`(标准答案), `student_answer`(学生作答), `llm_result`(模型批改结果), `score`(得分), `sub_correct_infos`(各小题批改详情)
- `parse_dataset(input_dir: str, output_path: str)`：批量处理 data/raw/ 下所有JSON文件，输出到 data/processed_train/parsed.jsonl
	- 处理DAG，把对应数据的标准答案的推理过程转化为json构建弱监督依赖图，里面可以以每个推理步骤为节点，存储"text/exprs/claims/type/verdict"等信息。对依赖关系建边，同时上下步骤之间也有弱顺序边，输出到 data/processed_dag/parsed.jsonl
- 清洗不合规数据，比如存在各种异常：content 为空、JSON 解析失败、字段缺失等
- 标准答案和学生作答中包含大量 LaTeX（双反斜杠），解析时要保留原始 LaTeX 不要损坏

#### 1.2 `src/dag/node.py` （可选）
DAG 的最小单元 Node的参考结构如下，使用 dataclass：

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

class StepType(Enum):
    DEFINITION = "definition"          # 引用定义/已知条件
    DERIVATION = "derivation"          # 推导步骤
    COMPUTATION = "computation"        # 计算步骤
    CONCLUSION = "conclusion"          # 结论
    AUXILIARY = "auxiliary"            # 辅助构造（如作辅助线）
    SUBSTITUTION = "substitution"     # 代入/替换
    CASE_ANALYSIS = "case_analysis"   # 分类讨论

class LocalVerdict(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNVERIFIABLE = "unverifiable"

@dataclass
class Node:
    step_id: int
    raw_text: str                                    # 原始文本
    normalized_text: str                             # 归一化文本（去掉冗余修饰）
    exprs: List[str] = field(default_factory=list)   # 提取出的数学表达式（LaTeX）
    claims: List[str] = field(default_factory=list)  # 原子命题（如 "OC=2OA"）
    step_type: StepType = StepType.DERIVATION
    local_verdict: LocalVerdict = LocalVerdict.UNVERIFIABLE
    sub_question_id: Optional[int] = None            # 属于哪个小题
```

#### 1.3 `src/dag/graph.py` （可选）
使用 networkx 构建和管理推理 DAG：
```python
import networkx as nx

class ReasoningDAG:
    """管理一道题的完整推理依赖图"""
    
    def __init__(self, problem_id: str):
        self.problem_id = problem_id
        self.graph = nx.DiGraph()
    
    def add_node(self, node: Node): ...
    def add_sequential_edges(self): ...          # 相邻步骤间的弱顺序边（weight=0.5）
    def add_dependency_edge(self, src_id, tgt_id, dep_type: str): ...  # 逻辑依赖边（weight=1.0）
    def validate_dag(self) -> dict: ...          # 验证：是否无环、是否连通、是否有孤立节点
    def get_topological_order(self) -> List[int]: ...
    def get_dependency_depth(self) -> int: ...   # 最长依赖链长度
    def to_json(self) -> dict: ...               # 序列化为JSON
    def from_json(cls, data: dict) -> 'ReasoningDAG': ...
    def visualize(self, output_path: str): ...   # 用 matplotlib 可视化保存为图片
```

关键设计：
- 边有两种：`sequential`（相邻步骤自动添加，弱边）和 `dependency`（逻辑依赖，强边）
- `dependency` 边由 LLM 辅助标注（见 build_dag.py），也可以通过规则匹配（如果 step B 引用了 step A 中出现的表达式/结论）
- `validate_dag` 返回一个 dict，包含 `is_acyclic`, `is_connected`, `isolated_nodes`, `max_depth` 等字段

#### 1.4 `src/data/build_dag.py`
从解析后的标准答案构建 DAG：
```python
def extract_steps_from_answer(standard_answer: str) -> List[dict]:
    """
    将标准答案按行拆分为步骤列表。
    标准答案格式如：
    第1行：解：(1) 证明：连接AC，交BD于点O，连接OE，
    第2行：∵ CD∥AB，∴ ∠OBA=∠ODC, ∠OAC=∠OCD，
    ...
    返回 [{"step_id": 0, "raw_text": "...", "sub_question_id": 1}, ...]
    """

def extract_expressions(text: str) -> List[str]:
    """从一行文本中提取所有 LaTeX 数学表达式"""
    # 匹配 $...$ 和 \(...\) 和行内公式

def extract_claims(text: str) -> List[str]:
    """从一行文本中提取原子命题，如等式、不等式、平行、垂直关系"""
    # 例如从 "∴ OC=2OA" 提取 "OC=2OA"

def classify_step_type(text: str) -> StepType:
    """基于关键词规则分类步骤类型"""
    # "∵" → DEFINITION, "∴" → DERIVATION, "解得" → COMPUTATION, "故" → CONCLUSION, "连接/作" → AUXILIARY

def build_dependency_edges_by_rules(nodes: List[Node]) -> List[tuple]:
    """
    基于规则匹配构建依赖边：
    如果 node_j 的 claims 中引用了 node_i 的 claims/exprs 中的符号，则添加 (i, j) 依赖边。
    使用简单的字符串匹配 + 正则表达式。
    """

def build_dependency_edges_by_llm(nodes: List[Node], llm_client=None) -> List[tuple]:
    """
    可选：用 LLM 判断步骤间是否存在逻辑依赖。
    对每个步骤 j，将其与前面所有步骤一起发给 LLM，问 "步骤j 逻辑上依赖哪些前置步骤？"
    返回边列表。
    注意：这是一个辅助标注工具，不用于 reward 的实时计算。
    """
```

#### 1.5 `src/data/prepare_sft.py`

将处理好的数据转为 ms-swift 的 SFT 数据格式。ms-swift 支持的格式是 jsonl，每行一个样本：

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

需要实现：
- `format_critique_prompt(stem, standard_answer, student_answer) -> str`：构造 user prompt
- `format_critique_response(llm_result_json) -> str`：构造 assistant response（从原始模型输出中提取，或者用标准答案重新构造理想输出）
- `prepare_sft_dataset(input_path, output_path, max_samples=None)`
- 同时支持生成"解题"格式的 SFT 数据（不只是批改，还有纯数学问答）

#### 1.6 `src/data/clean.py`
数据清洗，过滤掉以下情况：
- 学生完全未作答但模型生成了冗长分析的样本（如示例中学生只写了"解:(1)"）
- 模型输出 JSON 格式不合法的样本
- 标准答案缺失或题干缺失的样本
- 模型输出中错因与标准答案步骤重复度 >80% 的样本（说明模型只是复读标准答案）
- 统计清洗前后的数据分布（总数、各学科、各题型、各分数段）

#### 1.7 `src/data/merge_datasets.py`
混合中英文数据：
- 支持从 HuggingFace 加载公开数学数据集（NuminaMath, MetaMathQA 等）
- 将公开数据集转为统一的 messages 格式
- 按比例混合（默认 70% 中文业务数据 + 30% 英文公开数据）
- 支持设置随机种子和采样策略
---

### 2. Reward 设计
#### 2.1 `src/reward/outcome_reward.py`

```python
def outcome_reward(model_output: str, ground_truth: dict) -> float:
    """
    答案正确性奖励。
    从 model_output 中解析出 JSON，提取 "学生得分" 和 "结论批改" 字段，
    与 ground_truth 中的标注对比。
    
    奖励设计：
    - 得分完全一致：+1.0
    - 得分误差在 ±1 分内：+0.5
    - 结论批改完全一致：+0.5（额外奖励）
    - 其他：0.0
    """
```

#### 2.2 `src/reward/format_reward.py`

```python
def format_reward(model_output: str) -> float:
    """
    格式合规奖励。
    检查模型输出是否包含 <think>...</think> 和 <answer>...</answer> 标签，
    <answer> 内是否为合法 JSON（双引号、字段完整）。
    
    奖励设计：
    - 完全合规：+1.0
    - 有 answer 标签但 JSON 不合法：+0.3
    - 缺少标签：0.0
    """
```

#### 2.3 `src/reward/topo_reward.py`（核心创新）
```python
def topo_reward(model_output: str, reference_dag: Optional[ReasoningDAG] = None) -> float:
    """
    拓扑结构合法性奖励（弱监督版本）
    不要求模型输出与 reference_dag 完全一致，而是检查模型推理链的结构合法性：
    
    1. 从模型 <think> 部分提取推理步骤
    2. 构建推理步骤的弱依赖图（用 build_dependency_edges_by_rules）
    3. 验证结构合法性：
       - is_acyclic: 推理链是否无环 → 如果有环扣 0.3
       - no_orphan_conclusions: 结论步骤是否都有前置依赖 → 有孤立结论扣 0.2
       - consistent_direction: 推理方向是否一致（不出现条件→结论→条件的回退） → 回退扣 0.1
    4. 如果提供了 reference_dag，额外检查关键依赖是否被覆盖（overlap ratio）
    
    奖励设计（总分 1.0）：
    - 基础分 0.4（只要能提取出步骤就给）
    - 无环 +0.2
    - 无孤立结论 +0.2  
    - 方向一致 +0.1
    - 与 reference_dag 关键依赖覆盖率 * 0.1（可选）
    """
```

#### 2.4 `src/reward/continuity_reward.py`

```python
def continuity_reward(model_output: str) -> float:
    """
    步骤连续性奖励。
    
    检查模型推理链中每一步是否可以由前面的上下文支持：
    
    1. 提取推理步骤序列
    2. 对每一步检查：
       a. 当前步骤引用的数学表达式/数值是否在前面出现过
       b. 当前步骤引用的中间结论是否在前面被推导过
       c. 是否出现"跳步"（引用了完全没有前文依据的结论）
    3. 用规则 + 正则实现，不需要LLM
    
    奖励设计：
    - continuity_score = 连续步骤数 / 总步骤数
    - 如果全部连续：+1.0
    - 否则：continuity_score * 0.8
    
    注意：对于引用题目已知条件的步骤，视为连续（因为已知条件是全局可用的）
    """
```

#### 2.5 `src/reward/composite_reward.py`

```python
def composite_reward_fn(model_output: str, ground_truth: dict, reference_dag: Optional[ReasoningDAG] = None) -> float:
    """
    组合奖励函数，用于 ms-swift GRPO 训练。
    
    R_total = w1 * R_outcome + w2 * R_format + w3 * R_topo + w4 * R_continuity + w5 * R_length_penalty
    
    默认权重：
    - w1 = 0.4 (答案正确性，最重要)
    - w2 = 0.15 (格式合规)
    - w3 = 0.2 (拓扑结构)
    - w4 = 0.15 (步骤连续性)
    - w5 = 0.1 (长度惩罚，鼓励简洁)
    
    R_length_penalty: 如果输出 token 数超过阈值（如 2000），按比例扣分
    """

# 同时提供 ms-swift 兼容的接口
def get_reward_func(reward_type: str = "composite"):
    """
    返回一个符合 ms-swift reward_func 接口的函数。
    ms-swift GRPO 的 reward_func 签名是：
    def reward_func(completions, **kwargs) -> list[float]
    
    其中 completions 是模型生成的文本列表。
    kwargs 中会包含 prompt、ground_truth 等信息（取决于数据集格式）。
    """
```

---

### 3. 配置文件

#### 3.1 `configs/sft_qwen3_14b.yaml`

写一个 ms-swift 的 SFT 配置文件，关键参数：
- model: Qwen/Qwen3-14B
- dataset: data/sft_ready/train.jsonl
- train_type: lora (rank=64, alpha=128, target_modules=all-linear)
- batch_size: 2, gradient_accumulation_steps: 8
- learning_rate: 1e-4
- num_train_epochs: 3
- max_length: 8192
- eval_steps: 100
- save_strategy: steps, save_steps: 200
- 加上 deepspeed zero3 配置
- 加上 wandb 日志

#### 3.2 `configs/grpo_qwen3_14b.yaml`

写一个 ms-swift 的 GRPO 配置文件，关键参数：
- model: 使用 SFT checkpoint 的路径（占位，README中说明需要替换）
- rlhf_type: grpo
- reward_funcs: 指向 src/reward/composite_reward.py 中的函数
- num_generations: 8 (每个 prompt 采样 8 个 rollout)
- max_new_tokens: 4096
- temperature: 0.7
- 其余参数参考 ms-swift GRPO 默认值

---

### 4. 脚本

#### 4.1 `scripts/download_models.sh`

```bash
# 下载 Qwen3-14B 到 models/ 目录
# 使用 modelscope 或 huggingface-cli
```

#### 4.2 `scripts/download_benchmarks.sh`

下载以下 benchmark 到 data/benchmarks/：
- MATH (Hendrycks)
- GSM8K
- CMATH（中文数学）
- GaoKao-Bench（高考数学）
- C-Eval（数学子集）

每个 benchmark 下载后做简单的格式统一（统一为 jsonl，包含 question, answer, subject 字段）

#### 4.3 `scripts/run_data_pipeline.sh`

一键执行完整数据处理流程：
```bash
# Step 1: 解析原始数据
python -m src.data.parse_raw --input_dir data/raw --output_path data/processed/parsed.jsonl

# Step 2: 清洗数据
python -m src.data.clean --input_path data/processed/parsed.jsonl --output_path data/processed/cleaned.jsonl

# Step 3: 构建 DAG
python -m src.data.build_dag --input_path data/processed/cleaned.jsonl --output_dir data/dag --use_llm false

# Step 4: 准备 SFT 数据
python -m src.data.prepare_sft --input_path data/processed/cleaned.jsonl --output_path data/sft_ready/train.jsonl

# Step 5: 混合英文数据
python -m src.data.merge_datasets --zh_path data/sft_ready/train.jsonl --output_path data/sft_ready/train_mixed.jsonl --en_ratio 0.3

# Step 6: 准备 GRPO 数据
python -m src.data.prepare_grpo --input_path data/processed/cleaned.jsonl --dag_dir data/dag --output_path data/grpo_ready/train.jsonl
```

---

### 5. 文档

#### 5.1 `docs/reward_design.md`

详细文档，包含：
- 奖励设计动机：为什么需要 process-level reward，现有方法的不足
- 每个 reward component 的数学定义、伪代码、设计理由
- 权重选择的 ablation 建议
- 与 ms-swift 的集成方式（如何注册自定义 reward_func）
- 已知限制和未来改进方向
- 画出 reward 计算的流程图（用 mermaid 语法）

#### 5.2 `docs/ms_swift_custom_reward.md`

具体说明如何在 ms-swift 中接入自定义 reward function：
- ms-swift GRPO 的 reward_func 接口规范
- 代码示例：如何传入 ground_truth 数据
- 调试技巧：如何单独测试 reward function
- 常见坑：reward 数值范围、NaN 处理、timeout 处理

#### 5.3 `docs/dag_schema.md`

DAG JSON 的完整 schema 定义，包含示例。

#### 5.4 `docs/training_pipeline.md`

整个训练流水线的分步指南：
- Phase 0: 环境配置
- Phase 1: 数据准备
- Phase 2: SFT 训练
- Phase 3: SFT 评估（在 benchmark 上测）
- Phase 4: GRPO 训练（先 outcome-only，再加 topo reward）
- Phase 5: GRPO 评估
- Phase 6: 蒸馏（预留接口，详细实现 TODO）

---

### 6. README.md

写一个精简的 README，包含：
- 项目一句话介绍
- Quick Start（3步跑通：安装→数据准备→训练）
- 项目结构概览（用 tree 展示，每个目录一句话说明）
- 训练命令速查表
- 评估命令
- 自定义 Reward 快速上手（指向 docs/）
- Citation placeholder

风格要求：简洁、不啰嗦、面向已有 PyTorch/ms-swift 经验的研究者。

---

### 7. Tests

写基本的单元测试：
- `test_parse_raw.py`：用一个 mock 的原始 JSON 测试解析流程
- `test_build_dag.py`：用一个简单的三步推理测试 DAG 构建
- `test_rewards.py`：测试每个 reward function 的边界情况
- `test_graph.py`：测试 DAG 验证（有环图、孤立节点、正常图）

---

### 8. requirements.txt

```
ms-swift>=3.0
torch>=2.1
transformers>=4.40
networkx>=3.0
matplotlib>=3.7
sympy>=1.12
datasets>=2.19
modelscope>=1.14
tqdm
jsonlines
regex
pytest
wandb
```

---

### 代码规范

- 所有 Python 文件使用 type hints
- 所有函数都有 docstring
- 使用 argparse 处理 CLI 参数
- 日志使用 Python logging 模块，不用 print
- 代码中不要有 TODO 或 placeholder，所有函数都要有完整实现（即使是简化版本）
- 中文注释和英文代码混合可以，但 docstring 统一用英文

请开始生成所有文件。一个文件一个文件地输出，每个文件都给出完整路径和完整代码。
```