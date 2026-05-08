# Critique-DAG Data Specification

This spec extends the current processed JSONL schema in a backward-compatible way.
All new fields are optional. Legacy pipelines can ignore them safely.

## 1) Compatibility with existing records

Existing fields remain unchanged:
`stem`, `standard_answer`, `student_answer`, `llm_result`, `score`,
`procedure_score`, `sub_correct_infos`, `topic_id`, `topic_type`,
`source_style`, `source_file`, `category`, `record_id`.

## 2) Optional extension fields

| Field | Type | Purpose |
|---|---|---|
| `native_dag` | `object | null` | Native DAG annotation, preferred over rule-based build |
| `v2_dag` | `object | null` | Backward alias for `native_dag` |
| `error_tags` | `list[str]` | Structured error labels |
| `trace_quality` | `str` | `correct` / `partial` / `wrong` |
| `trace_confidence` | `float` | Confidence in `[0,1]` |
| `difficulty` | `str` | `easy` / `medium` / `hard` / `competition` |
| `sub_questions` | `list[object]` | Optional sub-question split |
| `multi_trace_id` | `str | null` | Multi-trace group id per original problem |
| `trace_variant` | `str | null` | `correct_short`, `correct_long`, `local_wrong`, `global_wrong`, `tool_verified` |

### 2.1 `native_dag` schema

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

## 3) Pipeline contract

- `src/data/build_dag.py`:
  - Use `native_dag` first.
  - If missing, try `v2_dag`.
  - If still missing, use rule-based fallback from `standard_answer`.
- `src/data/prepare_grpo.py`:
  - Keep `reference_dag` behavior unchanged.
  - Pass through metadata fields (`error_tags`, `trace_quality`, etc.).
- Reward modules are unchanged and remain source-of-truth:
  - `src/reward/topo_reward.py`
  - `src/reward/continuity_reward.py`
  - `src/reward/composite_reward.py`

## 4) Error tag taxonomy

Recommended labels:
`algebra_sign`, `domain_miss`, `logic_jump`, `boundary_miss`,
`definition_misuse`, `calculation_error`, `substitution_error`,
`case_analysis_miss`, `unit_conversion`, `graph_construction`.

## 5) Data quality constraints

- DAG must be acyclic.
- At least one `check` node per trace.
- Conclusion node must be traceable from prior nodes.
- `llm_result` must include both `<think>` and `<answer>` blocks.
- Minimum information density: at least 60% nodes contain expression/claim evidence.
