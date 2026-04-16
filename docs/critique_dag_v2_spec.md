# Critique-DAG v2 数据规范

> **原则**: 增量扩展，不破坏 v1 兼容。所有新字段为 optional，旧管线遇到新字段时跳过即可。

## 1. 与 v1 的关系

v1 schema（`data/processed/cleaned.jsonl`）保持原样，字段不变：
`stem`, `standard_answer`, `student_answer`, `llm_result`, `score`,
`procedure_score`, `sub_correct_infos`, `topic_id`, `topic_type`,
`source_style`, `source_file`, `category`, `record_id`

v2 在同一 JSONL 行内新增以下 **optional** 字段。

## 2. v2 新增字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `v2_dag` | `object \| null` | 原生 DAG（优先于规则构图）；schema 见下方 |
| `error_tags` | `list[str]` | 错误类型标签，如 `["algebra_sign", "domain_miss"]` |
| `trace_quality` | `str` | `"correct"` / `"partial"` / `"wrong"` |
| `trace_confidence` | `float` | 质量置信度 `[0,1]` |
| `difficulty` | `str` | `"easy"` / `"medium"` / `"hard"` / `"competition"` |
| `sub_questions` | `list[object]` | 子题拆分（见下） |
| `multi_trace_id` | `str \| null` | 同题多轨迹组标识 |
| `trace_variant` | `str \| null` | `"correct_short"` / `"correct_long"` / `"local_wrong"` / `"global_wrong"` / `"tool_verified"` |

### 2.1 `v2_dag` 结构

```json
{
  "nodes": [
    {
      "id": "n0",
      "type": "decompose | derive | check | conclude | definition | auxiliary",
      "text": "...",
      "sub_question_id": null,
      "local_verdict": "correct | incorrect | unverifiable",
      "verify": {
        "method": "substitution | equivalence | domain_check | theorem_condition | none",
        "evidence": "..."
      }
    }
  ],
  "edges": [
    {
      "from": "n0",
      "to": "n1",
      "rel": "depends_on | validated_by | supports | contradicts",
      "evidence_span": "..."
    }
  ]
}
```

### 2.2 `sub_questions` 结构

```json
[
  {"id": 1, "text": "求 x 的值", "answer": "x=3"},
  {"id": 2, "text": "求 y 的取值范围", "answer": "y>0"}
]
```

## 3. 与现有管线的兼容契约

- `build_dag.py` 改造后：检查 `rec.get("v2_dag")`，有值则直接构造 `ReasoningDAG`；无值走现有规则构图（fallback）。
- `prepare_grpo.py` 改造后：`reference_dag` 优先取 `v2_dag` 对应的图；fallback 到 `data/dag/<record_id>.json`。
- `prepare_sft.py`：不需要改动（SFT 只消费 `messages`）。
- `topo_reward.py` / `continuity_reward.py` / `composite_reward.py`：**不做任何改动**，它们只依赖 `build_dag_from_answer` 的输出和 `reference_dag` 字段。
- `clean.py` 改造后：新增 optional 的 DAG 可构建性检查（仅当 `v2_dag` 存在时）。

## 4. 错误类型标签 taxonomy

按数学推理领域常见错误分类：

| tag | 说明 |
|-----|------|
| `algebra_sign` | 代数符号/正负号错误 |
| `domain_miss` | 定义域/值域遗漏 |
| `logic_jump` | 逻辑跳步 |
| `boundary_miss` | 边界条件忽略 |
| `definition_misuse` | 定义/定理误用 |
| `calculation_error` | 计算错误 |
| `substitution_error` | 代入错误 |
| `case_analysis_miss` | 分类讨论不完整 |
| `unit_conversion` | 单位换算错误 |
| `graph_construction` | 几何作图/辅助线错误 |

## 5. 数据生成优先策略

1. **高错题区间优先**：每个难度区间按模型错误率采样，不均匀。
2. **同题多轨迹**：每题 3-5 条（correct_short, correct_long, local_wrong, global_wrong, tool_verified），由 `multi_trace_id` + `trace_variant` 标记。
3. **错误类型覆盖**：按 taxonomy 统计，保证每类至少 50 条。
4. **hard negatives**：同 DAG 结构微调关键节点/边，与正确轨迹配对。

## 6. 质量阈值

- 每条 `v2_dag` 必须：有向无环、至少一个 `check` 节点、`conclude` 可追溯到前置节点。
- `llm_result` 必须含完整 `<think>` 和 `<answer>` 块。
- 文本节点信息密度：表达式或 claim >= 1 的节点占比 >= 60%。
