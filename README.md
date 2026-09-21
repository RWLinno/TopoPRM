# TopoPRM

**Rewarding the Graph Behind the Chain: Topology-Aware Process Supervision for RL and Reasoning Distillation**

TopoPRM extracts a typed forward support DAG from ordinary reasoning text and reuses its evidence for reinforcement learning and targeted teacher revision. The deployed policy emits ordinary text, with no graph decoder or auxiliary verifier.

## Method

- **Stage I — SFT:** a shared supervised warm start makes reasoning traces segmentable.
- **Stage II — GRPO + ACE:** an outcome-anchored hierarchical reward combines recovered conclusion support and local continuity. ACE constrains the final advantage coefficients after group standardization: correct completions receive non-negative coefficients and incorrect completions non-positive coefficients.
- **Stage III — TGD:** a frozen teacher revises traces collected from the Stage-II policy snapshot. Accepted revisions form a fixed corpus; teacher and target receive the same saved prefixes for token-level reverse KL.

Forward extraction restricts edges to `i < j`, making direction and acyclicity construction guards. On a valid nonempty graph, the reference-free topology score is `(0.50 + 0.15 * no_orphan) / 0.65`. This is a recovered-support gate, not proof that every necessary premise was recovered. Reference-edge F1 is an offline diagnostic and is excluded from policy rewards.

```text
base   = 0.70 * outcome + 0.15 * format + 0.15 * length
u      = 0.60 * scale(topology) + 0.40 * scale(continuity)
reward = clip(max(base, 0.05) * (1 + u), 0, 1)
```

`scale` is min–max rescaling within each reward call; constant components map to 0.5. ACE uses prompt-group sample standard deviation plus `1e-4`, centers `u` within correctness strata, then applies one-sided adjustments and coefficient bounds `[-1, 1]` without recentering. The pinned TRL 0.28.0 trainer consumes these coefficients directly.

## Run

Install `requirements.txt`. The paper launchers specify their defaults directly and do not load stored training configurations or infer checkpoints from logs. Supply your data and checkpoints explicitly.

```bash
pip install -r requirements.txt

SFT_MODEL=/path/to/Qwen3.5-9B SFT_DATA=/path/to/sft.jsonl \
  bash scripts/run_sft.sh

bash scripts/run_grpo.sh full \
  --model /path/to/Qwen3.5-9B \
  --sft_adapter /path/to/stage1_adapter \
  --data /path/to/rl.jsonl --output_dir output/topoprm_stage2

STUDENT_MODEL=/path/to/merged_stage1_model \
STUDENT_ADAPTER=/path/to/stage2_adapter \
TEACHER_MODEL=/path/to/teacher_base \
TEACHER_ADAPTER=/path/to/fixed_teacher_adapter \
PROMPT_DATA=/path/to/rl.jsonl \
  bash scripts/run_topology_distill.sh
```

The Stage-II adapter is trained on the base **after merging the SFT adapter**; Stage III therefore requires that merged Stage-I model plus the Stage-II adapter. Teacher and target must share the token-to-ID vocabulary. Stage III initializes the target from its supplied adapter, freezes the teacher, and applies reverse KL over non-padding positions in saved system/problem/revised-response sequences. Its default objective has no cross-entropy term. `DISTILL_PHASE=rollout|data|train|all` selects the phase; `DISTILL_VARIANT=topology|generic|length|static` selects the revision procedure.

Stage I defaults: LoRA 64/128, learning rate `5e-5`, cosine schedule, 3% warmup, two epochs, microbatch 2, accumulation 8, sequence cap 4096. Stage II defaults: LoRA 64/128, learning rate `5e-6`, cosine schedule, 5% warmup, one epoch, microbatch 1, accumulation 16, two completions, temperature 0.8, top-p 0.95, completion cap 4096. Complete prompts exceeding 4096 tokens are excluded. Set `NPROC_PER_NODE` for distributed training. `TOPOPRM_DRY_RUN=1` or `DISTILL_DRY_RUN=1` prints a launch summary without starting training.

The GRPO launcher also accepts `without_ace`, `outcome_only`, `outcome_length`, `no_topology`, and `no_continuity`. Full and w/o ACE include a subsequent TGD stage in the paper; source-removal rows stop after Stage II. No-continuity uses a 1024-token completion cap. These are pipeline comparisons, not isolated causal estimates of topology or continuity. A generic revision option does not imply that the reported no-topology checkpoint underwent distillation.

For evaluation, merge the adapters in training order and run `scripts/bench_transformers.py --help` for the benchmark, sampling, and output options. Pass@1 and pass@5 require their corresponding sample counts and must be kept separate.

## Availability

The full source release, trained weights, training logs, research datasets, and per-item evaluation records will be made public after acceptance. The submission project page is maintained separately on `comingsoon`.
