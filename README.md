# TopoPRM

Anonymous implementation for **Rewarding the Graph Behind the Chain: Topology-Aware Process Supervision for Reasoning**.

TopoPRM extracts a typed forward support graph from ordinary reasoning text. Recovered conclusion support and local continuity shape a hierarchical reward during GRPO. The deployed model generates text without a graph decoder or auxiliary verifier.

## Paper configuration

The retained shared-stage comparison uses the forward rule extractor and `TopoHierarchicalReward`. Candidate edges satisfy `i < j`; direction and acyclicity are construction invariants. The varying evidence is recovered conclusion support, local continuity, and an optional reference-edge term when a reference graph is supplied. Numeric step indices alone do not establish semantic alignment across different solutions.

```text
base = 0.70 * outcome + 0.15 * format + 0.15 * length
reward = clip(max(base, 0.05) * (1 + 0.60 * scale(topology)
                                 + 0.40 * scale(continuity)), 0, 1)
```

`scale` is min-max rescaling over a reward-call batch; constant inputs map to 0.5. GRPO subsequently normalizes rewards within prompt groups. Length shaping uses decoded characters: score 1 through 2,000 characters, then linear decay to 0 at 4,000.

The shared-stage comparisons stop after GRPO and do not use ACE or TGD. The raw-graph/Choquet classes and distillation utilities remain available as separate experimental implementations. Their presence does not make them the source of the retained main-table results. `src/distill/student_train.py` matches teacher and student distributions on a fixed dataset; it is not an on-policy rollout procedure and does not imply equal model sizes.

## Install and run

```bash
pip install -r requirements.txt

python scripts/train_grpo_ablation.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --sft_adapter /path/to/shared_sft_adapter \
  --data /path/to/training.jsonl \
  --reward topo_hierarchical \
  --max_steps 200 --num_generations 4 --max_completion_len 2048 \
  --output_dir output/hierarchical
```

Use `--reward outcome_only` or `--reward outcome_length` with the same SFT adapter, data, and budget for the displayed controls. The latter includes format as well as length (weights 0.85/0.05/0.10 for outcome/format/length), so it is not an isolated topology ablation. The entrypoint pins the forward reward mode to prevent an inherited raw-graph environment from changing the method. It does not select a checkpoint by benchmark performance.

For evaluation, supply a standalone model with the shared SFT and GRPO adapters already merged in that order:

```bash
python scripts/bench_transformers.py \
  --model /path/to/merged_model --label hierarchical \
  --benchmarks gsm8k --max_items 200 \
  --num_samples_per_item 1 --k_values 1 \
  --max_new_tokens 4096 --batch_size 8 \
  --use_chat_template --sft_style --save_solutions \
  --output_dir output/eval
```

The current evaluator saves complete responses when requested, verifies a terminal answer, and counts visible tokens per response. The paper's legacy table instead reaggregates saved correctness flags from an earlier harness. Its extracted-answer records cannot support a full terminal-answer recheck or corrected visible-token efficiency claims. A new run is not an exact reproduction of those historical decisions.

## Evidence and availability

The shared 200-update GSM8K comparison records 151/200, 153/200, and 154/200 correct for outcome-only, the length-aware control, and hierarchical training. Five stochastic draws do not preserve that ordering. Cross-backbone accuracy is mixed; Llama MATH accuracy decreases. Semantic-PRM fusion does not improve on the PRM in the rescored shared pools. The paper does not claim SOTA, isolated topology gains, or validated distillation/compression effects.

Weights, training logs, research datasets, and per-item records are planned for release after acceptance. The code and configuration snapshot is available for inspection; the private runs cannot be independently reproduced from this branch alone during review.

## Layout

- `src/`: extraction, graph diagnostics, rewards, distillation utilities, evaluation.
- `scripts/`: training and evaluation entrypoints.
- `configs/`: explicit experiment configurations and checkpoint identifiers.
- `data/README.md`: input format and availability.

Manuscript drafts and review notes are maintained outside this anonymous branch.
