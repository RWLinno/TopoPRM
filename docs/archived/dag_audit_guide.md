# DAG Audit Guide

## 1) Build trace pool

```bash
python scripts/build_trace_pool.py \
  --label dag_audit_dr1_7b \
  --sample-size 50 \
  --max-items 50
```

Output files:

- `output/dag_audit/dag_audit_dr1_7b_<bench>_traces.jsonl`

## 2) Run diagnostics + pass/fail report

Rule-only (default):

```bash
python scripts/dag_quality_audit.py \
  --label dag_audit_dr1_7b \
  --report
```

With offline LLM refinement:

```bash
python scripts/dag_quality_audit.py \
  --label dag_audit_dr1_7b \
  --use-llm \
  --report
```

Output files:

- `output/dag_audit/<bench>_diagnostics.json`
- `output/dag_audit/<bench>_failures.jsonl`
- `output/dag_audit/report.md`

## 3) Visual case study

Interactive Streamlit GUI:

```bash
bash scripts/run_dag_gui.sh 8765 output/dag_audit/dag_audit_dr1_7b_gsm8k_traces.jsonl
```

Recommended default port: `8765`.

Static exports (per benchmark):

```bash
python tutorials/render_dag_cases.py \
  --from-rollout output/dag_audit/dag_audit_dr1_7b_gsm8k_traces.jsonl \
  --bench gsm8k \
  --n 5
```

Files are written to:

- `TopoPRM_EMNLP26/figures/dag_cases/<bench>/case_*.pdf`
- `TopoPRM_EMNLP26/figures/dag_cases/<bench>/case_*.png`
