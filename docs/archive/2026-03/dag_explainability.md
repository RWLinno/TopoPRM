# DAG Explainability and Compression Evaluation

## 1) Aggregate metrics

Run:

```bash
python3 -m src.eval.dag_metrics \
  --dag_dir data/dag \
  --output output/eval/dag_metrics.json
```

Key metrics:

- `acyclic_rate`
- `connected_rate`
- `orphan_conclusion_ratio`
- `direction_consistency`
- `dependency_depth`
- Compression ratios (`node_ratio`, `edge_ratio`, `depth_ratio`, `dependency_edge_keep_ratio`)

## 2) Human-annotation check (5-10 samples)

Prepare `data/annotations/dag_gold_edges.json` with format:

```json
{
  "record_001": {"gold_dependency_edges": [[0,2],[2,3]]},
  "record_002": {"gold_dependency_edges": [[0,1],[1,4]]}
}
```

Then run:

```bash
python3 -m src.eval.dag_metrics \
  --dag_dir data/dag \
  --annotation data/annotations/dag_gold_edges.json \
  --output output/eval/dag_metrics_annotated.json
```

## 3) Paper reporting suggestion

- Structural quality table: report `acyclic_rate`, `orphan_conclusion_ratio`,
  `direction_consistency`, `dependency_depth`.
- Compression table: report `node_ratio`, `edge_ratio`, `dependency_edge_keep_ratio`.
- Case study: include one raw DAG and one compressed DAG for interpretability.
