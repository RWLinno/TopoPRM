# TopoPRM

**Rewarding the Graph Behind the Chain**

*Topology-Aware Process Supervision for RL and Reasoning Distillation*

TopoPRM uses the implicit dependencies between reasoning steps as a shared representation for process rewards and online distillation. A frozen extractor recovers a typed forward graph from ordinary reasoning text.

| Component | Contribution |
| :--- | :--- |
| **Hierarchical process rewards** | Combine answer correctness, conclusion support, and local continuity |
| **Asymmetric credit estimation (ACE)** | Assign structural credit under correctness constraints |
| **Topology-guided distillation (TGD)** | Guide a fixed teacher to revise current student traces and distill accepted responses |

The deployed policy generates ordinary reasoning text. Direction and acyclicity are enforced by graph construction.

## Availability

The anonymous implementation snapshot and compact launchers are maintained on the `iclr_anonymous` branch. Model artifacts are distributed separately. Training logs, research datasets, and per-item evaluation records are not included on this project page.
