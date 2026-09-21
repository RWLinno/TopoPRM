# TopoPRM

**Rewarding the Graph Behind the Chain: Topology-Aware Process Supervision for RL and Reasoning Distillation**

TopoPRM uses the dependency structure behind a reasoning trace to guide post-training. A deterministic extractor recovers a typed support graph from ordinary text; the same evidence informs reinforcement learning and targeted teacher revision.

- **Structural supervision:** recovered conclusion support and local continuity complement final-answer rewards.
- **Correctness-constrained credit:** hierarchical reward shaping and asymmetric credit estimation guide policy updates.
- **Topology-guided distillation:** graph diagnostics localize teacher revisions, which supply filtered reasoning targets.

The resulting policy generates ordinary reasoning text without a graph decoder or an auxiliary verifier at inference.

## Release plan

The full implementation, training and evaluation scripts, model weights, training logs, research datasets, and per-item evaluation records will be released after acceptance.

This page will be updated with the release and usage instructions when the materials are available.
