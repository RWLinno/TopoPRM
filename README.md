# TopoPRM

**Rewarding the Graph Behind the Chain: Topology-Aware Process Supervision for RL and Reasoning Distillation**

<p align="center">
  <img src="docs/assets/topoprm_overview.png" width="80%" alt="TopoPRM framework overview">
</p>

Long chain-of-thought reasoning is not linear. A later step often draws on
several non-adjacent premises, an intermediate result gets reused downstream,
and subgoals branch before merging. Outcome rewards and step-local process
rewards both flatten a trace into a sequence and cannot see those failures.

TopoPRM recovers an **order-agnostic directed support graph** from a free-form
trace, then projects it onto a dependency DAG. The edges that projection
discards are the point: backward-edge mass measures direction inconsistency and
cycle-edge mass measures circular support. Both survive as separate reward
signals instead of being averaged into one topology score. The policy never has
to emit graph tags.

## Where things live

| What | Path |
|---|---|
| Code | `src/`, `scripts/`, `configs/` |
| Paper (the only LaTeX version) | `paper/`, root file `iclr2027_conference.tex` |
| Run records backing the tables | `output/`, `results/` |
| Archived weights and datasets | `/knowin-oss/weilinruan/topoprm_archive` |
| History and handoff notes | `docs/` |

Model weights are not in the tree. They were moved to object storage with the
directory layout preserved, so a path like `output/<run>/checkpoint-N/` maps to
`/knowin-oss/weilinruan/topoprm_archive/checkpoints/output/<run>/checkpoint-N/`.
Training logs, per-example outputs, and metric JSON stayed in the repository:
weights can be retrained from this code, run records cannot.

## Method

Three stages share one graph encoder `E_phi` and one acyclic projection `Pi`.

**Stage I — Supervised warm start.** SFT on structured mathematical solutions so
rollouts are parsable. Topology diagnostics are meaningless on a trace with no
recoverable steps.

**Stage II — GRPO on a topology-conditioned reward.** For a rollout `y`, the
frozen encoder builds the raw support graph `H_y`, and `G_y = Pi(H_y)` keeps the
edges consistent with text order. Two diagnostics come off the raw graph, each
shrunk toward `0.5` by edge coverage `c_H` so a trace with no recovered edges
reads as uncertain rather than perfect:

```
q_dir  = 1/2 + c_H * (d_H - 1/2)        d_H = forward-edge mass
q_acyc = 1/2 + c_H * ((1 - u_H) - 1/2)  u_H = cycle-edge mass
```

Five sources — outcome, format, `q_dir`, `q_acyc`, continuity — are combined by a
monotone 2-additive Choquet integral with non-negative Möbius coefficients:

```
r_total = sum_i m_i z_i + sum_{(i,j) in I} m_ij * min(z_i, z_j)
I = {(o,d), (o,a), (d,c), (a,c), (d,a)}
```

Because `min(u,v) = (u+v)/2 - |u-v|/2`, this is exactly its Shapley-matched
additive counterpart minus a disagreement penalty. Two rollouts with equal
coordinate sums tie under any additive aggregator; the Choquet reward prefers the
balanced one. Under GRPO's within-group normalization a tie yields no gradient,
so the difference is mechanical, not cosmetic.

**Stage III — Topology-Guided Distillation (TGD).** The Stage-II model teaches a
smaller student. The student rolls out first, the encoder localizes a defect
(backward edge, cyclic component, orphan conclusion, continuity break), and the
teacher revises against that specific defect. A revision is kept only if it is
correct, well-formed, within budget, and does not lower either raw-graph
diagnostic. Compression cannot be bought by deleting necessary support.

### Why direction and acyclicity stay separate

Every directed cycle contains a backward edge under text order, but a backward
edge need not close a cycle — the relation is asymmetric, so neither coordinate
determines the other. Empirically the two are strongly anticorrelated on reversed
controls (`r = -0.907`), and averaging them collapses cross-trace standard
deviation from `0.110` to `0.023`. Keeping them apart preserves information that
premature averaging destroys.

## Quick start

```bash
pip install -r requirements.txt   # torch>=2.1, transformers, peft, ms-swift, math-verify, networkx
```

Evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/bench_transformers.py \
  --model /path/to/Qwen3.5-9B \
  --adapter /knowin-oss/weilinruan/topoprm_archive/checkpoints/output/<run>/final \
  --label topoprm_9b \
  --benchmarks math500 aime2024 olympiadbench \
  --use_chat_template \
  --num_samples_per_item 5 --k_values 1 5 \
  --max_new_tokens 4096
```

Training:

```bash
export TOPO_LENGTH_UNIT=tokens
export TOPO_DAG_SENTENCE_FALLBACK=1
swift rlhf configs/grpo_9b_from_ckpt79.yaml
```

Tests: `pytest tests/ -v`

## Evaluation protocol

Two protocols appear in the paper and must not be mixed on one axis.

- **EMNLP protocol** — 5 samples per item, mostly a 2,560-token cap. The
  completed result tables use it, marked teal in the manuscript.
- **ICLR canonical protocol** — one seed-0 sample per item, 4,096-token cap,
  single shared verifier. Reproduction is in progress; pending cells are red
  placeholders and carry no claim.

Correctness is read only from the last non-empty `\boxed{}` or an explicitly
marked terminal answer, so a truncated derivation earns nothing. Visible length
is obtained by retokenizing without special tokens. Every retained row maps to a
checkpoint, a manifest, and per-example outputs.

## Layout

```
TopoPRM/
├── src/
│   ├── reward/          # Choquet aggregation, outcome/format/continuity sources
│   ├── dag/             # Raw support graph, acyclic projection, diagnostics
│   ├── data/            # Segmentation, edge extraction, DAG cache
│   ├── distill/         # TGD: defect dispatch, target acceptance
│   └── eval/            # Verifier and metric utilities
├── paper/               # ICLR 2027 submission (single source of truth)
│   ├── iclr2027_conference.tex
│   ├── sections/  tabs/  figs/
├── scripts/             # Training launchers, benchmark runners
├── configs/             # Training YAML
├── data/grpo_ready/     # query + solution + reference_dag
├── output/  results/    # Run records and metrics (weights live in OSS)
├── rebuttal/            # Edge validation, PRM best-of-N, matched baselines
├── tests/  tutorials/   # Unit tests, figure and analysis tools
└── docs/                # Handoff notes, archived drafts, consolidation record
```

## Status

The manuscript is under revision for ICLR 2027 after an EMNLP 2026 borderline
decision. Three reviewer concerns drive the current work: the edge extractor
needed independent validation rather than self-referential metrics, gains needed
attribution to topology rather than to length or continuity shaping, and every
table row needed traceable provenance.

Consequently some cells are deliberately empty. A red placeholder marks a run
that has not finished, never a zero result, and no claim in the paper rests on
one. `docs/archived/iclr27_workspace_consolidation.md` records how the scratch
workspaces were folded into this repository.

Topology is a training-time structural signal, not a substitute for semantic
verification: on a shared best-of-8 pool, topology-only selection underperforms a
dedicated PRM, and their product does not beat the PRM alone. That negative
result is kept in the appendix.

## Citation

```bibtex
@article{ruan2026topoprm,
  title={Rewarding the Graph Behind the Chain: Topology-Aware Process
         Supervision for RL and Reasoning Distillation},
  author={Ruan, Weilin},
  year={2026}
}
```

## License

MIT
