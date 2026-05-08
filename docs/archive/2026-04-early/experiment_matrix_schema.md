# Experiment Matrix Schema

This project uses `scripts/experiment_matrix.jsonl` as the single source of truth
for orchestrated experiments.

Each line is one JSON object with the following fields:

- `id` (string, required): globally unique task id.
- `stage` (string, required): logical stage, e.g. `t0_core`, `t1_qwen25`, `t2_efficiency`, `finalize`.
- `gpu` (integer, required): assigned single GPU id for the worker queue.
- `deps` (array[string], required): upstream task ids that must be `success`.
- `retries` (integer, optional): retry attempts after first failure, default `1`.
- `expected_outputs` (array[string], optional): artifacts used for idempotent skip logic.
- `tags` (array[string], optional): lightweight labels for filtering/reporting.
- `command` (string, required): shell command executed by worker.

Execution semantics:

1. Four workers are launched by orchestrator (GPU0/1/2/3 by default).
2. Tasks are filtered by `gpu` and run in matrix order.
3. Within a worker, tasks are strictly serial.
4. A task with all `expected_outputs` already present is marked `success_cached`.
5. If any dependency reaches terminal failure, dependent task is marked `skipped_dep_failed`.
6. Every lifecycle event is appended to `output/analysis/experiment_registry.jsonl`.

Registry event schema (`experiment_registry.jsonl`):

- `ts`: ISO-like timestamp.
- `worker`: worker label like `gpu0`.
- `task_id`: matrix task id.
- `status`: `running` / `success` / `success_cached` / `failed_retry` / `failed_final` / `skipped_dep_failed`.
- `attempt`: attempt index (1-based).
- `exit_code`: process exit code if available.
- `duration_sec`: elapsed seconds for finished attempts.
- `gpu`: bound GPU id.
- `stage`: copied from matrix.
- `command`: executed command.
- `artifacts`: expected outputs copied from matrix.

