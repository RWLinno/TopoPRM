# TopoPRM

**Rewarding the Graph Behind the Chain**

*Topology-Aware Process Supervision for RL and Reasoning Distillation*

TopoPRM uses the implicit dependencies between reasoning steps to guide both **structural credit assignment** and **online teacher revision**. A frozen extractor maps ordinary reasoning text to a typed, forward dependency graph. The trained policy generates ordinary text.

```text
Stage I                  Stage II                         Stage III
Supervised warm start ──► Hierarchical rewards + ACE ──► Online TGD
                                 ▲                         ▲
                         Typed dependency graph ───────────┘
```

| Component | Role | Implementation |
| :--- | :--- | :--- |
| **TopoPRM** | Combine answer correctness, conclusion support, and local continuity | `src/reward/composite_reward.py` |
| **ACE** | Assign one-sided structural credit within correctness strata | `scripts/train_grpo_ablation.py` |
| **TGD** | Diagnose current student traces, guide a fixed teacher's revisions, and distill accepted responses | `src/distill/student_train.py` |

## Method contract

Edges satisfy `i < j`. Direction and acyclicity are construction invariants. On a nonempty valid graph, the reference-free topology score is `(0.50 + 0.15 * no_orphan) / 0.65`. Reference-edge F1 is used for offline analysis and does not enter policy rewards.

```text
base   = 0.70 × outcome + 0.15 × format + 0.15 × length
u      = 0.60 × scale(topology) + 0.40 × scale(continuity)
reward = clip(max(base, 0.05) × (1 + u), 0, 1)
```

`scale` is min–max normalization within each complete prompt group, gathered across devices. Constant channels map to 0.5. ACE standardizes scalar rewards using the group sample standard deviation plus `1e-4`, centers `u` within correctness strata, and applies one-sided adjustments with bounds `[-1, 1]`. Correct completions receive nonnegative coefficients and incorrect completions nonpositive coefficients. The coefficients enter the policy loss without recentering.

TGD samples from the **current student** at each update. The frozen teacher receives the problem, complete student trace, and localized revision instruction. Acceptance requires answer correctness, valid formatting, the response budget, and nondecreasing topology and continuity scores. Reverse KL is summed over the revised response tokens and averaged over all revision attempts, including zero contributions from rejected revisions. Teacher and student use the teacher-revised response prefixes. Only the teacher also receives the original trace and revision request.

## Run the three stages

Use Python with the dependencies in `requirements.txt`. The implementation pins Transformers 5.5.4 and TRL 0.28.0. Supply checkpoints and data explicitly.

```bash
pip install -r requirements.txt

# I. Supervised initialization
SFT_MODEL=/path/to/Qwen3.5-9B SFT_DATA=/path/to/sft.jsonl \
  bash scripts/run_sft.sh

# II. Hierarchical process rewards and ACE
bash scripts/run_grpo.sh full \
  --model /path/to/Qwen3.5-9B --sft_adapter /path/to/stage1_adapter \
  --data /path/to/rl.jsonl --output_dir output/stage2

# III. Online topology-guided distillation
STUDENT_MODEL=/path/to/stage2/final \
TEACHER_MODEL=/path/to/Qwen3.5-9B \
TEACHER_ADAPTER=/path/to/stage1_adapter \
PROMPT_DATA=/path/to/revision_prompts.jsonl \
DISTILL_OUTPUT_DIR=output/stage3 \
  bash scripts/run_topology_distill.sh
```

The Stage-II and Stage-III `final/` exports are fully merged checkpoints. For adapter inputs, supply the corresponding base and adapter separately. A Stage-II adapter requires the already merged Stage-I base. Loading checks all adapter tensor names and shapes. TGD additionally checks architecture settings, parameter shapes, tokenization rules, vocabulary, and chat serialization.

| Default | Stage I | Stage II | Stage III |
| :--- | :---: | :---: | :---: |
| Learning rate | `5e-5` | `5e-6` | `2e-5` |
| LoRA rank / alpha | 64 / 128 | 64 / 128 | 64 / 128 |
| Response or sequence cap | 4,096 sequence | 4,096 response | 2,048 response |
| Accumulation | 8 | 16 | 8 attempts |

`NPROC_PER_NODE` controls distributed Stage-I/II training. Stage-III device placement uses `STUDENT_DEVICE` and `TEACHER_DEVICE`. Set `TOPOPRM_ENV_BIN` to a virtual environment's `bin` directory if needed. `TOPOPRM_DRY_RUN=1` and `DISTILL_DRY_RUN=1` print commands without starting training. Launchers read arguments and environment variables, not stored training configurations.

### Supplementary mechanism comparison

For the seed-42 suite, add `--max_steps 200 --num_generations 2 --generation_batch_size 16 --seed 42` to Stage II. For each Stage-III arm, use the same resulting student, fixed Stage-I teacher, and the same ordered 512-prompt file:

```bash
# Add these variables to the Stage-III command above.
DISTILL_SEED=42 MAX_PROMPTS=512 TOKEN_BUDGET=2048 \
DISTILL_MAX_LENGTH=8192 DISTILL_ACCUMULATION=8 \
REVISION_STRATEGY=topology REVISION_SELECTION=topology \
  bash scripts/run_topology_distill.sh
```

Repeat with `REVISION_STRATEGY=generic` and a fresh output directory to change only the revision instruction. Keep `REVISION_SELECTION=topology` for the shared acceptance gate. The loop records accepted and rejected attempts in the local output directory.

The historical ablation launchers also support `without_ace`, `outcome_only`, `outcome_length`, `no_topology`, and `no_continuity`. Full and w/o ACE include TGD. Historical source-removal rows stop after Stage II, and no-continuity uses a shorter cap. The supplementary `--structural_ablation` option instead neutralizes a structural channel at 0.5 while preserving the reward mixture. These are distinct protocols.

## Evaluation and repository scope

Run `python scripts/bench_transformers.py --help` for benchmark and decoding options. Set `--num_samples_per_item` and `--k_values` explicitly. Evaluation exports the number of items, samples per item, and correctness counts. Generation failures stop evaluation instead of being counted as wrong answers.

This anonymous branch contains the implementation and compact launchers. Training configurations, logs, datasets, and model weights are excluded from Git. Model artifacts are maintained separately from this anonymous source snapshot.
