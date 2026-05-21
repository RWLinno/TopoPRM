# merge_manifest (server_B)

This manifest tells server_A's `results/method_v2/*` how to join the
server_B baseline outputs.

## Source files (server_B)
- `results/baseline/leaderboard_baseline.csv` (one row per `(label, benchmark)`)
- `results/baseline/metrics_full_baseline.json` (full metrics nested)
- `results/baseline/eval_trace_baseline.md` (chronological run trace)
- `results/baseline/missing_cells_report.md`

## Raw evidence
- All raw `*_metrics.json` and `*_details.jsonl` live under `output/eval`.
- Server_B labels carry the `_B` suffix (e.g. `sft_qwen35_9b_B`).

## Suggested join
Join key: `(label, benchmark)`. The server_A leaderboard
(`results/method_v2/leaderboard_method_v2.csv`) follows the same schema, so
a `csv.DictReader` over both files can be concatenated as-is.

When normalising labels for the paper table:
- strip the `_B` suffix when reporting headline numbers; keep the suffix in
  the audit trail to track which server produced each cell.
