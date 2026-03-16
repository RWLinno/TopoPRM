# DAG JSON Schema

This document specifies the JSON schema for TopoPRM's reasoning DAG (Directed Acyclic Graph) representation, as implemented by `ReasoningDAG` in `src/dag/graph.py`.

---

## Top-Level Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `problem_id` | `string` | yes | Unique identifier for the problem / record. |
| `nodes` | `array[Node]` | yes | Ordered list of reasoning step nodes. |
| `edges` | `array[Edge]` | yes | List of directed edges between nodes. |

```json
{
  "problem_id": "record_42",
  "nodes": [ ... ],
  "edges": [ ... ]
}
```

---

## Node Schema

Each node represents a single reasoning step.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `step_id` | `int` | yes | — | Zero-based index of this step within the answer. |
| `raw_text` | `string` | yes | — | Original text of the reasoning step. |
| `normalized_text` | `string` | no | `""` | Cleaned / normalised version of the text. |
| `exprs` | `array[string]` | no | `[]` | Mathematical expressions extracted from the step (e.g., `"AC² = 16"`). |
| `claims` | `array[string]` | no | `[]` | Mathematical claims / relations extracted (e.g., `"AB∥CD"`, `"∠ABC = 90°"`). |
| `step_type` | `enum` | no | `"derivation"` | Classification of the step (see below). |
| `local_verdict` | `enum` | no | `"unverifiable"` | Local correctness verdict (see below). |
| `sub_question_id` | `int \| null` | no | `null` | Sub-question number if the problem has multiple parts. |

### `step_type` Enum

| Value | Description |
|---|---|
| `"definition"` | Restates givens or known conditions (已知, given, 由题意). |
| `"derivation"` | Logical inference from prior steps (∴, 推得, 所以, 因此). |
| `"computation"` | Arithmetic or algebraic calculation (解得, 计算, 化简). |
| `"conclusion"` | Final answer or summary statement (故, 综上, 答). |
| `"auxiliary"` | Construction step in geometry (连接, 作, 过点, 延长). |
| `"substitution"` | Variable substitution (代入, 令, 将…代入). |
| `"case_analysis"` | Case split (分类讨论, 当…时, 情况一/二). |
| `"unknown"` | Could not be classified. |

### `local_verdict` Enum

| Value | Description |
|---|---|
| `"correct"` | The step is verified as correct. |
| `"incorrect"` | The step contains an error. |
| `"unverifiable"` | Correctness cannot be determined automatically. |

---

## Edge Schema

Each edge represents a relationship between two reasoning steps.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `source` | `int` | yes | — | `step_id` of the source node. |
| `target` | `int` | yes | — | `step_id` of the target node. Must differ from `source`. |
| `edge_type` | `enum` | no | `"sequential"` | Either `"sequential"` or `"dependency"`. |
| `dep_type` | `string` | no | `""` | Dependency sub-type (only meaningful when `edge_type` is `"dependency"`). |
| `weight` | `float` | no | `0.5` | Edge weight. Sequential edges default to 0.5; dependency edges default to 1.0. |

### `edge_type` Values

| Value | Weight (default) | Description |
|---|---|---|
| `"sequential"` | 0.5 | Adjacent steps in the original text order. Added automatically between consecutive `step_id` values. |
| `"dependency"` | 1.0 | Logical dependency — the target step references an expression or claim that originated in the source step. |

### `dep_type` Values (for dependency edges)

| Value | Description |
|---|---|
| `"expr_ref"` | Target step reuses a mathematical expression from the source. |
| `"claim_ref"` | Target step reuses a mathematical claim from the source. |
| `"implicit"` | No explicit overlap detected, but an implicit dependency is inferred (e.g., a derivation step immediately follows another with no other predecessor). |
| `"logical"` | Generic logical dependency (used when added manually). |

---

## Validation Rules

A well-formed DAG must satisfy:

### 1. Acyclicity

The graph must be a DAG — no directed cycles. Verified by `ReasoningDAG.is_valid_dag()` which calls `networkx.is_directed_acyclic_graph`.

### 2. Connectivity

The undirected projection of the graph should be connected — every node is reachable from every other node (ignoring edge direction). Checked by `validate_dag()`.

### 3. No Orphan Conclusions

Every node with `step_type == "conclusion"` must have at least one incoming **dependency** edge. A conclusion that has only sequential predecessors (or no predecessors at all) is considered "orphaned" — it lacks grounding in prior reasoning steps.

### 4. Additional Checks (from `validate_dag()`)

| Check | Description |
|---|---|
| `isolated_nodes` | Nodes with no edges at all (in-degree = 0 and out-degree = 0). |
| `max_depth` | Longest path length in the DAG (only computed when acyclic). |
| `num_nodes` / `num_edges` | Basic size statistics. |

---

## Complete Example

A 4-step geometry proof: *"In triangle ABC, AB = 5, BC = 3. Find AC given that angle B = 90°."*

```json
{
  "problem_id": "geom_right_triangle_001",
  "nodes": [
    {
      "step_id": 0,
      "raw_text": "已知三角形ABC中，∠B=90°，AB=5，BC=3。",
      "normalized_text": "已知三角形ABC中，∠B=90°，AB=5，BC=3。",
      "exprs": ["AB=5", "BC=3"],
      "claims": ["∠ABC = 90°"],
      "step_type": "definition",
      "local_verdict": "correct",
      "sub_question_id": null
    },
    {
      "step_id": 1,
      "raw_text": "由勾股定理，AC² = AB² + BC²。",
      "normalized_text": "由勾股定理，AC² = AB² + BC²。",
      "exprs": ["AC² = AB² + BC²"],
      "claims": [],
      "step_type": "derivation",
      "local_verdict": "correct",
      "sub_question_id": null
    },
    {
      "step_id": 2,
      "raw_text": "代入数值，AC² = 25 + 9 = 34。",
      "normalized_text": "代入数值，AC² = 25 + 9 = 34。",
      "exprs": ["AC² = 25 + 9 = 34"],
      "claims": [],
      "step_type": "computation",
      "local_verdict": "correct",
      "sub_question_id": null
    },
    {
      "step_id": 3,
      "raw_text": "故 AC = √34。",
      "normalized_text": "故 AC = √34。",
      "exprs": ["AC = √34"],
      "claims": [],
      "step_type": "conclusion",
      "local_verdict": "correct",
      "sub_question_id": null
    }
  ],
  "edges": [
    {
      "source": 0,
      "target": 1,
      "edge_type": "sequential",
      "dep_type": "",
      "weight": 0.5
    },
    {
      "source": 1,
      "target": 2,
      "edge_type": "sequential",
      "dep_type": "",
      "weight": 0.5
    },
    {
      "source": 2,
      "target": 3,
      "edge_type": "sequential",
      "dep_type": "",
      "weight": 0.5
    },
    {
      "source": 0,
      "target": 2,
      "edge_type": "dependency",
      "dep_type": "expr_ref",
      "weight": 1.0
    },
    {
      "source": 1,
      "target": 2,
      "edge_type": "dependency",
      "dep_type": "expr_ref",
      "weight": 1.0
    },
    {
      "source": 2,
      "target": 3,
      "edge_type": "dependency",
      "dep_type": "expr_ref",
      "weight": 1.0
    }
  ]
}
```

### Visual Representation

```
  [0: definition]  ──seq──▶  [1: derivation]  ──seq──▶  [2: computation]  ──seq──▶  [3: conclusion]
        │                          │                          ▲                          ▲
        │                          │                          │                          │
        └─────────── dep(expr_ref) ┘─────── dep(expr_ref) ───┘──── dep(expr_ref) ──────┘
```

- **Sequential edges** (dashed in visualisations): 0→1, 1→2, 2→3
- **Dependency edges** (solid): 0→2 (AB=5, BC=3 reused), 1→2 (AC² formula reused), 2→3 (AC² = 34 reused)
- The conclusion node (3) has a dependency predecessor (2), so it is **not** orphaned.
- The graph is acyclic and connected.

---

## Programmatic Usage

### Building a DAG from text

```python
from src.data.build_dag import build_dag_from_answer

answer_text = """已知三角形ABC中，∠B=90°，AB=5，BC=3。
由勾股定理，AC² = AB² + BC²。
代入数值，AC² = 25 + 9 = 34。
故 AC = √34。"""

dag = build_dag_from_answer(answer_text, problem_id="geom_001")
print(dag.to_json())
```

### Loading from JSON

```python
from src.dag.graph import ReasoningDAG

dag = ReasoningDAG.from_json(json_string)
print(dag.validate_dag())
```

### Validation

```python
result = dag.validate_dag()
# {
#   "is_acyclic": True,
#   "is_connected": True,
#   "isolated_nodes": [],
#   "max_depth": 3,
#   "num_nodes": 4,
#   "num_edges": 6,
#   "has_orphan_conclusions": False,
# }
```
