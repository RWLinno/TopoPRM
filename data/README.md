# Data

Research datasets, training logs, checkpoints, and per-item experimental records are planned for release after acceptance. This branch contains implementation and configuration files only.

The training entrypoint accepts JSONL with `messages` (or `question`), `solution`, and an optional `reference_dag`. Public benchmarks can be obtained with `scripts/download_benchmarks.py`. Dataset versions, subset sizes, and limitations are specified in the paper.
