# Work Summary (2026-05-14)

## Branch and Scope

- Source working branch: `rethink`
- Target sync branch: `exp_May14`
- Focus: TopoPRM/TGSD v2 experiment readiness, reward-path hardening, and prompt-driven execution split

## Completed Code and Config Work

- Added/updated v2 experiment entrypoints and configs for GRPO and TG-OPD:
  - `scripts/run_grpo_topoprm_v2.sh`
  - `scripts/run_tg_opd.sh`
  - `configs/grpo_topoprm_v2.yaml`
  - `configs/grpo_topoprm_v2.env`
- Extended reward and data-path robustness (env-gated where applicable):
  - `src/reward/composite_reward.py`
  - `src/reward/reward_config.py`
  - `src/reward/continuity_reward.py`
  - `src/data/build_dag.py`
- Updated training/eval helpers and orchestration scripts:
  - `scripts/check_reward_invariants.py`
  - `scripts/bench_transformers.py`
  - `todo_exp_ours.sh`

## Paper/Docs and Prompt Assets

- Migrated/updated paper workspace into `TopoPRM_EMNLP26/` and aligned related docs.
- Added focused execution prompts for split-window experiment operation:
  - `docs/prompt_server_a_method_v2.md` (method line only)
  - `docs/prompt_server_b_baseline_v2.md` (baseline/supplementary only)
- Added diagnosis/context docs for method-performance debugging and rollback-safe iteration:
  - `docs/method_diagnosis_2026-05-14.md`

## Operational Constraints Captured

- Unified environment assumptions:
  - conda env: `topoprm`
  - framework: `ms-swift`
  - proxy: `ALL_PROXY=...vegalb.com:80`
- Required benchmark suite:
  - GSM8K, MATH-500, Olympiad, Omni-MATH, AIME'24, AIME'25, CNMO'24, MMLU, GPQA-D
- Required metrics:
  - `error`, `correct`, `F1`, `pass@1`, `pass@k`, `maj@k`, `prm@k`, `#Tokens`

## Next Actions

1. Run server A method pipeline and enforce ">= SFT/GRPO or redo with analysis".
2. Run server B baseline pipeline with independent scheduling via `todo_baseline.sh`.
3. Merge both result trees into unified table refresh for paper-facing artifacts.
