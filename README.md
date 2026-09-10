# TopoPRM

> Anonymous code release for ICLR 2027 double-blind review.
> Paper, appendix, and review correspondence are intentionally **not** part of this branch.

## What this is

TopoPRM is a **topology-conditioned reward and distillation stack** for training
reasoning models with GRPO. Instead of scoring a rollout only by whether its
final answer is right, TopoPRM reads the *support structure* of the free-text
trace and turns two structural defects — inconsistent dependency direction and
circular support — into separate, independently graded reward sources.

Three properties are worth stating up front, because they are what distinguish
this from a "add a graph score to the reward" recipe:

1. **The policy never emits a graph.** Structure is recovered from ordinary
   free-text reasoning by an extractor. No output-format constraint is imposed
   on the model, so nothing about the trained policy's interface changes.
2. **Direction and acyclicity stay separate.** They are not averaged into one
   topology scalar. Averaging them provably destroys most of the usable signal
   (see [Why two sources](#why-direction-and-acyclicity-must-stay-separate)).
3. **Aggregation is a 2-additive Choquet integral**, not a product and not a
   stratified clip. Its interaction terms make it prefer *balanced* rollouts over
   lopsided ones with the same coordinate sum — a difference that is invisible
   to any additive reward under GRPO's group normalization.

## Method

### From text to a support graph, and what projection throws away

Given a reasoning trace `y`, the extractor segments it into steps and recovers an
**order-agnostic directed support graph** `H_y`: an edge `u -> v` means step `v`
draws support from step `u`, regardless of where the two appear in the text. `H_y`
may contain edges that run backward relative to textual order, and it may contain
directed cycles.

The DAG used downstream is the projection

```
G_y = Pi(H_y)
```

which drops whatever prevents `H_y` from being acyclic. **The discarded edges are
the signal, not noise.** Two independent quantities are read off the pre-projection
graph, both as *weight mass fractions* rather than edge counts:

| Quantity | Read from | Measures |
| --- | --- | --- |
| `d_H` | forward-edge weight mass / total mass | dependency direction consistency |
| `u_H` | cycle-edge weight mass / total mass | circular support |

Cycle mass is computed over strongly connected components, so an edge counts as
cyclic exactly when it lies inside a non-trivial SCC (or is a self-loop).

### Coverage shrinkage

A trace with almost no recovered dependencies would otherwise get an extreme
score from a nearly empty graph. Both quantities are therefore shrunk toward the
uninformative value `1/2` by the dependency coverage `c_H` (the fraction of steps
incident to at least one edge):

```
q_dir  = 1/2 + c_H * (d_H       - 1/2)
q_acyc = 1/2 + c_H * ((1 - u_H) - 1/2)
```

At `c_H = 1` the scores are the raw measurements; at `c_H = 0` both collapse to
`1/2` and contribute no gradient. Implemented in
[`src/dag/graph.py`](src/dag/graph.py) (`analyze_topology_projection`).

### 2-additive Choquet aggregation

Normalized sources `z_i` are aggregated with non-negative Möbius coefficients
that sum to one:

```
r_total = sum_i m_i * z_i  +  sum_(i,j) m_ij * min(z_i, z_j)
```

The interaction set pairs outcome with each structural source, each structural
source with continuity, and — critically — the two structural sources with each
other:

```
I = { (o,d), (o,a), (d,c), (a,c), (d,a) }
```

with `o` = outcome, `d` = direction, `a` = acyclicity, `c` = continuity.
Non-negative coefficients summing to one make the aggregate **monotone and
bounded**, so the reward cannot be gamed by pushing one coordinate alone.

### Why the interaction terms are not cosmetic

Because `min(u, v) = (u + v)/2 - |u - v|/2`, every interaction term splits into an
additive half and a penalty on disagreement. So the Choquet reward is **exactly**
its Shapley-matched additive counterpart minus a disagreement penalty:

```
r_choquet = r_matched_additive - (1/2) * sum_(i,j) m_ij * |z_i - z_j|
```

This is the mechanism, not a reweighting. Two rollouts with the same coordinate
*sum* but different spread are tied under **any** additive aggregation. GRPO
normalizes advantages within a group, so a tie yields zero gradient — an additive
reward literally cannot express a preference between them. The Choquet form breaks
the tie in favour of the balanced rollout. Both the matched-additive and
equal-weight additive controls are implemented
([`src/reward/composite_reward.py`](src/reward/composite_reward.py):
`topo_matched_additive`, `topo_equal_additive`) precisely so this comparison is
measurable rather than asserted.

## Why direction and acyclicity must stay separate

The two quantities are related **asymmetrically**, not equivalently:

- Every directed cycle must contain at least one backward edge.
- A backward edge need not lie on any cycle.

So low `q_acyc` implies some backward mass, but backward mass says nothing about
whether a cycle exists. Neither determines the other, and collapsing them to a
mean discards the part where they disagree.

This is measurable, and the measurement is why the separation is a design
requirement rather than a preference. On order-reversed controls the two
coordinates are strongly **anti**-correlated (`r = -0.907`); averaging them
collapses cross-trace standard deviation from `0.110` to `0.023`, i.e. roughly
four-fifths of the discriminative range is cancelled.

The diagnostic that produces these statistics — correlation, per-source spread,
spread of the average, and the fraction of sources whose two coordinates move in
opposite directions, each with a bootstrap 95% interval — lives in
[`src/eval/source_separation.py`](src/eval/source_separation.py) and is pinned by
[`tests/test_edge_validation.py`](tests/test_edge_validation.py).

## Three training stages

| Stage | Name | What happens |
| --- | --- | --- |
| **I** | SFT warm start | Establishes a usable reasoning format so that structure extraction is meaningful before any RL signal is applied. |
| **II** | GRPO on topology-conditioned Choquet reward | The policy is optimized against `r_total` over outcome, format, `q_dir`, `q_acyc`, and continuity. |
| **III** | TGD — Topology-Guided Distillation | A teacher issues **targeted revisions of the student's specific structural defects**, rather than replacing the trace with its own reasoning. Supervision is conditioned on which structural source the student failed. |

Stage III is the reason the pipeline is not just "GRPO with an extra reward":
the structural diagnosis produced in Stage II is what selects and shapes the
distillation targets in Stage III.

## Repository layout

```
.
├── src/                          Library code. Import root is `src.*`.
│   ├── capacity_profiles.py      Named Choquet capacity profiles (`balanced`,
│   │                             `structure_forward`). A canonical run must name
│   │                             one explicitly — there is no silent default.
│   │
│   ├── dag/                      Reasoning-graph representation and projection.
│   │   ├── graph.py              `ReasoningDAG` + `analyze_topology_projection`:
│   │   │                         coverage, forward mass, SCC-based cycle mass,
│   │   │                         and the shrunk `q_dir` / `q_acyc` scores.
│   │   ├── node.py               Step node: text span, index, typed role.
│   │   ├── edge_encoder.py       Typed dependency edges with support weights.
│   │   └── compress.py           Chain contraction for long traces.
│   │
│   ├── data/                     Corpus construction and trace -> graph parsing.
│   │   ├── parse_raw.py          Segment raw generations into ordered steps.
│   │   ├── build_dag.py          Recover `H_y` from a trace; optional LLM edge
│   │   │                         refinement (off during online GRPO by design).
│   │   ├── prepare_sft.py        Stage I corpus.
│   │   ├── prepare_grpo.py       Stage II prompt set, eval-overlap filtered.
│   │   ├── generate_distill_data.py  Stage III teacher-revision pairs.
│   │   ├── merge_datasets.py     Mix sources under a language/ratio budget.
│   │   └── clean.py              Dedup and malformed-record removal.
│   │
│   ├── reward/                   All reward sources and aggregators (ORM plugins).
│   │   ├── reward_config.py      Every knob, resolved from `TOPO_*` env vars.
│   │   ├── outcome_reward.py     Answer correctness via `math_verify`.
│   │   ├── format_reward.py      Progressive format-compliance rubric.
│   │   ├── topo_reward.py        Structural verification -> `q_dir`, `q_acyc`.
│   │   ├── continuity_reward.py  Step-to-step local coherence.
│   │   ├── composite_reward.py   Aggregators, incl. `topo_independent_choquet`
│   │   │                         (canonical) and the matched additive / equal /
│   │   │                         multiplicative controls used for attribution.
│   │   ├── ablation_rewards.py   Source-removal variants for the ablation grid.
│   │   ├── gat_topo_reward.py    Learned-GAT topology baseline.
│   │   ├── sarl_reward.py        Embedding-based small-world topology baseline.
│   │   ├── topo_position_encoding.py  Positional features for graph scoring.
│   │   └── utils.py              Completion normalization helpers.
│   │
│   ├── distill/                  Stage III (TGD).
│   │   ├── teacher_trace_filter.py   Keep only traces that fix a real defect.
│   │   ├── build_srt_data.py         Assemble revision training records.
│   │   ├── student_train.py          Student fine-tuning loop.
│   │   ├── opsd_trainer.py           On-policy self-distillation variant.
│   │   ├── reverse_kl_loss.py        Reverse-KL objective.
│   │   └── chain_compression_metrics.py  Trace-length/structure summary.
│   │
│   ├── eval/                     Scoring, benchmarking, reporting.
│   │   ├── unified_benchmark.py       Canonical single-verifier harness.
│   │   ├── benchmark_runner.py        Generation + scoring driver.
│   │   ├── math_scoring.py            Shared answer-extraction/verification.
│   │   ├── dag_metrics.py             Structural metrics over recovered graphs.
│   │   ├── source_separation.py       Direction/acyclicity independence stats
│   │   │                              with bootstrap intervals.
│   │   ├── structural_from_jsonl.py   Structural metrics from saved eval jsonl.
│   │   ├── critique_eval.py           Critique-style grading.
│   │   ├── distill_analysis.py        Teacher -> student compression effects.
│   │   ├── collect_experiment_results.py / export_paper_tables.py /
│   │   │   sync_paper_tables.py       Aggregate runs into tables (output paths
│   │   │                              are CLI arguments).
│   │   └── data_visualization_analysis.py  Dataset summary figures.
│   │
│   ├── prm/                      Process-reward-model interface used as a
│   │   └── model.py              semantic baseline and for best-of-n selection.
│   ├── training/accuracy_callback.py  Mid-training accuracy probe.
│   └── gui/dag_reward_viewer.py       Local viewer for graph + reward inspection.
│
├── scripts/                      Runnable entrypoints (thin over `src/`).
│   ├── run_sft.sh / run_sft_config.sh        Stage I.
│   ├── run_grpo.sh / train_grpo.py           Stage II.
│   ├── run_iclr27_stage2.sh                  Canonical Stage II launcher; maps a
│   │                                         variant name to a reward function
│   │                                         and refuses unsafe combinations.
│   ├── run_topology_distill.sh               Stage III (TGD).
│   ├── run_unified_eval.sh                   Canonical evaluation entrypoint.
│   ├── unified_eval_orchestrator.py          Multi-model / multi-benchmark sweep.
│   ├── train_grpo_ablation.py                Ablation-grid driver.
│   ├── check_reward_invariants.py            Asserts monotonicity/bound
│   │                                         invariants of the aggregators.
│   ├── dag_quality_audit.py                  Extraction-quality audit.
│   ├── preprocess_dag_cache.py               Materialize cached DAGs offline.
│   ├── evaluate_aggregation_pool.py          Aggregator comparison on a shared
│   │                                         best-of-n pool.
│   └── dist/                                 Cluster launchers (local, Slurm,
│                                             Ray, PyTorch-DDP) + worker script.
│
├── configs/                      84 run configs. Naming is `<stage>_<variant>_
│   │                             <model>.yaml`; `.env` siblings carry the `TOPO_*`
│   │                             overrides for that run.
│   ├── grpo_topoprm_iclr27.yaml  Canonical Stage II config.
│   ├── ablation_template.yaml    Start here for a new ablation.
│   └── dist/                     Cluster task manifests.
│
├── tests/                        193 tests: graph construction, edge validation,
│   ├── fixtures/dag_audit/       reward invariants, parsing, distillation filter.
│   └── ...                       Per-benchmark trace fixtures for extraction.
│
├── tutorials/                    Reproducible analysis / figure scripts.
│   ├── dag_coverage_audit.py     Coverage distribution across benchmarks.
│   ├── render_dag_cases.py       Render recovered graphs for qualitative cases.
│   └── training_curve.py         Training-curve figure from run logs.
│
├── data/                         Benchmark prompt sets (`data/benchmarks/`) and
│                                 minimal distillation samples. Generated
│                                 training corpora are gitignored.
├── results/                      Metric JSON + per-item JSONL for completed runs
│                                 (evidence for reported numbers; logs excluded).
├── requirements.txt              Python dependencies.
└── setup.py                      Editable install (`pip install -e .`).
```

## Quick start

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest tests/ -q          # expect 193 passed
```

### Path configuration

No absolute paths are hard-coded. Scripts and configs resolve four variables,
so point them at your own storage before launching anything:

```bash
export MODEL_ROOT=/path/to/pretrained_models   # local HF model snapshots
export EXP_ROOT=/path/to/experiments           # checkpoints, eval artifacts
export PYTHON_ENV_BIN=/path/to/venv/bin        # interpreter used by launchers
export REPO_ROOT="$PWD"
```

Configs referencing `${MODEL_ROOT}/...` are expanded at launch; Python defaults go
through `os.path.expandvars`, so an unset variable surfaces as a visibly wrong
path rather than silently loading the wrong weights.

### Train

```bash
# Stage I — SFT warm start
bash scripts/run_sft_config.sh sft_qwen35_9b

# Stage II — GRPO on the topology-conditioned Choquet reward
export TOPO_CHOQUET_CAPACITY_PROFILE=balanced   # required; no default
bash scripts/run_iclr27_stage2.sh full

# Stage III — Topology-Guided Distillation
bash scripts/run_topology_distill.sh
```

`run_iclr27_stage2.sh <variant>` is also how every ablation is launched; the
variant name selects the reward function and clears any conflicting `TOPO_*`
overrides so runs cannot silently mix conditions.

### Evaluate

```bash
# Canonical protocol: single sample, seed 0, 4096-token cap, unified verifier
bash scripts/run_unified_eval.sh "${MODEL_ROOT}/<base-model>" "<adapter-or-empty>" <run_tag>

# Multi-model / multi-benchmark sweep
python scripts/unified_eval_orchestrator.py --model "${MODEL_ROOT}/<base-model>"

# Verify aggregator invariants (monotone, bounded) without a GPU
python scripts/check_reward_invariants.py
```

## Reward sources and aggregators

Sources (each normalized to `[0, 1]`):

| Source | Module | Signal |
| --- | --- | --- |
| outcome | `outcome_reward.py` | final-answer correctness |
| format | `format_reward.py` | format compliance, progressive rubric |
| direction (`q_dir`) | `topo_reward.py` | forward-mass fraction, coverage-shrunk |
| acyclicity (`q_acyc`) | `topo_reward.py` | 1 − cycle mass, coverage-shrunk |
| continuity | `continuity_reward.py` | local step-to-step coherence |

Registered aggregators (all resolvable by name via `GRPO_REWARD_FUNC`):

| Name | Role |
| --- | --- |
| `topo_independent_choquet` | **canonical** — 2-additive Choquet, direction and acyclicity as separate sources |
| `topo_independent_matched_additive` | Shapley-matched additive control: isolates the interaction terms |
| `topo_independent_equal_additive` | equal-weight additive control |
| `topo_independent_matched_multiplicative` | product-form control |
| `topo_choquet` | Choquet over a single collapsed topology source (legacy comparison) |
| `matched_laser_d`, `topo_independent_hero`, `sarl_structure`, `gat_topo_reward` | external shaping / topology baselines |
| `ablation_*` | source-removal variants |

Because `topo_choquet` collapses direction and acyclicity into one source, it is
retained only as the comparison point that motivates the split — it is not the
canonical configuration.

## Evaluation protocols — do not mix them

Two protocols exist in this repository's history. Numbers from one are **not
comparable** with numbers from the other, and mixing them silently is the easiest
way to produce a wrong table.

| | Earlier protocol | **Canonical protocol** |
| --- | --- | --- |
| samples per item | 5 | 1 |
| seed | varies | fixed, seed 0 |
| generation cap | 2560 tokens (mostly) | 4096 tokens |
| verifier | per-benchmark variants | single unified verifier |
| reported metrics | `pass@k`, `maj@k` | single-sample accuracy |

Anything reported as canonical must come from
[`scripts/run_unified_eval.sh`](scripts/run_unified_eval.sh) /
[`src/eval/unified_benchmark.py`](src/eval/unified_benchmark.py). Files under
`results/` carry their own `num_samples_per_item`, `k_values`, and generation-cap
fields — read those before comparing two runs rather than assuming a protocol
from the directory name.

## Ablation grid

The reported ablation dimensions are:

| Condition | What it removes |
| --- | --- |
| Base | no RL, warm-started policy only |
| Outcome Only | correctness reward alone |
| w/o Topology | drops both structural sources |
| w/o Continuity | drops the continuity source |
| w/o ACE | drops the adaptive-capacity component of aggregation |
| TopoPRM (Full) | canonical configuration |

Launch each with `bash scripts/run_iclr27_stage2.sh <variant>`; source-removal
variants live in [`src/reward/ablation_rewards.py`](src/reward/ablation_rewards.py)
and finer-grained switches (`no_direction`, `no_acyclicity`,
`no_dir_acyc_interaction`) exist for attributing the effect to a specific source
or to the interaction term itself.

## Limitations

**Topology is a training-time structural signal. It does not replace semantic
verification.** This is a measured limitation, not a hedge:

- On a shared best-of-8 candidate pool, selecting by topology alone is **worse**
  than selecting with a dedicated process reward model.
- Multiplying the topology score into the PRM score is **not better** than the PRM
  alone.

So the contribution is about *shaping the learning signal during RL*, not about
building a better inference-time verifier. A topology score should not be used as
a standalone correctness proxy.

Further caveats:

- Structure quality is bounded by the extractor. Traces whose steps resist
  segmentation yield low coverage `c_H`, and coverage shrinkage then correctly
  pushes both structural scores toward the uninformative `1/2` — they stop
  contributing rather than contributing noise, but they also stop helping.
- Evidence is concentrated on mathematical and competition-style reasoning, where
  answers are automatically verifiable. Transfer to open-ended domains is untested.
- Optional LLM-based edge refinement is deliberately **off** during online GRPO;
  enabling it there would add a model forward pass per completion.

## Reproducing what is here

`results/` contains metric JSON and per-item JSONL for completed runs so the
reported numbers can be traced to concrete artifacts. Raw run logs are excluded
from this branch: they embed machine-specific absolute paths and would compromise
anonymity. Model weights are not distributed here.

## Citation

```bibtex
@misc{topoprm_anonymous,
  title  = {TopoPRM: Topology-Conditioned Process Rewards for Reasoning},
  author = {Anonymous Authors},
  year   = {2027},
  note   = {Under review at ICLR 2027}
}
```

## Anonymity notice

This branch is an anonymized, code-only snapshot prepared for double-blind review:
it carries no commit history, no author metadata, no paper or review
correspondence, and no machine-specific paths. Please do not attempt to
deanonymize it.

