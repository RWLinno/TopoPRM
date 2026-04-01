# DAG 生成流水线技术文档

> 对应代码：`src/data/build_dag.py` · `src/dag/graph.py` · `src/dag/node.py` · `src/reward/topo_reward.py`

---

## 1. 整体流程概览

```
原始推理文本（<think> 块）
        │
        ▼
① 步骤切分   extract_steps_from_answer
        │  每行 → 一个步骤节点，同时识别子题编号
        ▼
② 内容抽取   extract_expressions + extract_claims
        │  每个节点 → 数学表达式列表 + 命题关系列表
        ▼
③ 步骤类型分类   classify_step_type
        │  keyword 匹配 → StepType 枚举（7 种）
        ▼
④ 构建 ReasoningDAG   build_dag_from_answer
        │  add_sequential_edges()           顺序边（虚边，weight=0.5）
        │  build_dependency_edges_by_rules() 依赖边（实边，weight=1.0）
        ▼
⑤ 拓扑奖励评分   TopoReward
        │  无环性 / 孤儿结论 / 方向一致性 / 参考覆盖率
        ▼
  R_topo ∈ [0, 1]
```

---

## 2. 步骤切分（Step Segmentation）

**入口**：`extract_steps_from_answer(standard_answer)`

**切分粒度**：以换行符为边界，每个非空行对应一个步骤节点（`step_id = 行号`）。

**子题识别**：遍历每行时用 4 个正则模式检测子题起始：

| 模式 | 示例 |
|------|------|
| `【小题N】` | 【小题1】 |
| `(N)` 行首 | (1)、(2) |
| `（N）` 行首（全角） | （1）、（2） |
| `第N小题 / 第N题 / 第N问` | 第1小题、第2问 |

匹配到子题标记后，后续所有步骤的 `sub_question_id` 继承该编号，直到遇到下一个子题标记。

**输出**：每个步骤字典包含 `{step_id, raw_text, sub_question_id}`。

---

## 3. 内容抽取

### 3.1 数学表达式（`extract_expressions`）

按优先级依次匹配三类模式，长度 ≥ 3 且不重复才保留：

| 优先级 | 正则类型 | 示例 |
|-------|---------|------|
| 1（最高）| LaTeX 行内公式 `$...$` 或 `\(...\)` | `$x^2 + 1$` |
| 2 | 通用等式/不等式（含希腊字母、Unicode 运算符） | `a + b = c`、`x ≤ 3` |
| 3 | 中文变量赋值 `设/令 x = ...` | `设 k=2` |

### 3.2 数学命题（`extract_claims`）

匹配四类关系断言：

| 模式 | 语义 | 示例 |
|------|------|------|
| `变量 关系符 变量/数` | 代数关系 | `AB ≥ 2` |
| `两大写字母 ∥/⊥/≅/∽ 两大写字母` | 线段几何关系 | `AB ∥ CD` |
| `∠字母 = 数°` | 角度关系 | `∠ABC = 60°` |
| `∵/∴ ... 到标点` | 因果命题 | `∴ x = 1` |

---

## 4. 步骤类型分类（Step Typing）

**入口**：`classify_step_type(text) → StepType`

7 种类型按关键词优先顺序匹配（先匹配先返回）：

| StepType | 触发关键词（中文/符号） | 语义 |
|----------|----------------------|------|
| DEFINITION | ∵、已知、由题意、根据题意、题目给出 | 题设条件引入 |
| DERIVATION | ∴、推得、所以、因此、由此可得、则 | 逻辑推导 |
| COMPUTATION | 解得、计算、化简、整理得 | 代数计算 |
| CONCLUSION | 故、综上、综上所述、答、因此答案 | 最终结论 |
| AUXILIARY | 连接、作、过点、延长 | 辅助构造（几何） |
| SUBSTITUTION | 代入、令、将…代入、把…代入 | 变量替换 |
| CASE_ANALYSIS | 分类讨论、当…时、分两种情况、情况一、情况二 | 分情况讨论 |

**兜底规则**：含 `=` 或 `＝` 且不含 `∵/∴` → `COMPUTATION`；其余 → `UNKNOWN`。

---

## 5. DAG 构建（`build_dag_from_answer`）

### 5.1 节点加入

每个步骤创建 `Node` 对象（含 `exprs`、`claims`、`step_type`）并加入 `ReasoningDAG.graph`（networkx `DiGraph`）。

### 5.2 顺序边（Sequential Edges）

`add_sequential_edges()` 在相邻步骤 `(i, i+1)` 之间添加**虚顺序边**：

```
edge_type = "sequential",  weight = 0.5
```

仅表示文本顺序，不代表逻辑依赖，**不参与**拓扑奖励评分。

### 5.3 依赖边（Dependency Edges）

`build_dependency_edges_by_rules(nodes)` 分两遍扫描：

**第一遍——建立溯源表**

按 `step_id` 升序遍历，以**首次出现原则**记录每个表达式/命题的来源步骤：

```
expr_origin[expr]   = 首次出现该表达式的 step_id
claim_origin[claim] = 首次出现该命题的 step_id
```

**第二遍——建边**

对每个节点 `j`，遍历其表达式和命题：

- 若 `expr` 首次来源 `i < j` → 添加依赖边 `(i → j)`，类型 `expr_ref`
- 若 `claim` 首次来源 `i < j` → 添加依赖边 `(i → j)`，类型 `claim_ref`
- 每个来源步骤只建一条边（`seen_sources` 去重）

**兜底隐式边**：若节点 `j > 0`、未找到任何显式依赖来源、且类型为 `DERIVATION` 或 `CONCLUSION`，则添加隐式边 `(j-1 → j)`，类型 `implicit`。

所有依赖边属性：

```
edge_type = "dependency",  weight = 1.0
dep_type  = "expr_ref" | "claim_ref" | "implicit"
```

---

## 6. ReasoningDAG 核心方法速查

| 方法 | 说明 |
|------|------|
| `is_valid_dag()` | networkx `is_directed_acyclic_graph()` 判断无环 |
| `direction_consistency()` | 依赖边中 `u < v`（正向）的比例；全正向 → 1.0 |
| `orphan_nodes()` | 与所有依赖边均无关的游离节点 |
| `get_dependency_depth()` | 仅考虑依赖边时的最长路径长度 |
| `validate_dag()` | 返回 `{is_acyclic, is_connected, isolated_nodes, max_depth, has_orphan_conclusions}` |
| `root_nodes()` | 图入度为 0 的节点 |
| `leaf_nodes()` | 图出度为 0 的节点 |

**孤儿结论节点**的精确定义：`StepType == CONCLUSION` 且所有入边均不是 `edge_type="dependency"` 的结论节点，即仅靠顺序边连接的结论节点。

---

## 7. 拓扑奖励评分（`TopoReward`）

从 `<think>` 块调用 `build_dag_from_answer` 建图后，按以下公式评分（最终 clip 至 [0, 1]）：

| 项目 | 条件 | 贡献 |
|------|------|------|
| 基础分 | `len(V) > 0`（能提取到步骤） | +0.4 |
| 无环 | `is_valid_dag() == True` | +0.2 |
| 无孤儿结论 | `orphan_conclusion_ratio == 0` | +0.2 |
| 方向一致性 | `direction_consistency()` ∈ [0,1] | ×0.1 |
| 参考覆盖率 | `key_dep_coverage(dag, ref_dag)` ∈ [0,1]（无参考 DAG 时为 0） | ×0.1 |

**孤儿结论比率** = 无 `dependency` 类型入边的结论节点数 / 全部结论节点数。

**参考覆盖率** = `|E_dep ∩ E_ref_dep| / |E_ref_dep|`，即预测 DAG 与参考 DAG 依赖边集合的交集占比。

---

## 8. R_topo 与 R_cont 的分工

|  | 拓扑奖励 `R_topo` | 连续性奖励 `R_cont` |
|---|---|---|
| 粒度 | **全局**（整个 DAG 图结构） | **局部**（逐步检查） |
| 评估对象 | 有向图的拓扑性质 | 每步内容是否可追溯 |
| 循环检测 | ✓ | ✗ |
| 孤儿结论检测 | ✓（仅结论节点） | 间接 |
| 所需数据结构 | `ReasoningDAG` 图 | 仅步骤文本的表达式/命题集合 |

---

## 9. 已知局限

| 局限 | 说明 |
|------|------|
| 切分粒度固定 | 以换行为边界，无法处理跨行单步骤 |
| 字符串精确匹配 | 变量重命名（如 `x` → `t`）会导致依赖边缺失 |
| 隐式边覆盖有限 | 仅 `DERIVATION` / `CONCLUSION` 类型步骤才触发兜底边 |
| LLM 检测占位符 | `build_dependency_edges_by_llm` 目前仍回退到规则检测 |
| 方向一致性代理 | 以 `step_id` 大小代理前提→结论方向，对乱序步骤可能误判 |
