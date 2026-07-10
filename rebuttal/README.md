# TopoPRM Rebuttal Experiments

Supplementary experiments and the author response for the EMNLP/ARR rebuttal.
All heavy jobs run on GPUs 4-7; GPUs 0-3 are left for other users.

## Layout

```
rebuttal/
├── response.md                 # author response (3 reviewers, W/R format)
├── rebuttal.sh                 # master launcher (nohup run_xxx.sh > run_xxx.log 2>&1 &)
├── configs/
│   └── eval_rebuttal.yaml       # unified eval protocol + run matrix
├── scripts/
│   ├── env.sh                   # shared env (paths, HF/wandb tokens, proxy)
│   ├── edge_validation.py       # sample/annotate/score DAG edges vs independent judge
│   ├── semantic_gap.py          # generate/score structure-semantic gap
│   ├── train_grpo_rebuttal.py   # TRL GRPO trainer (no vLLM; broken in this env)
│   ├── to_swift_grpo.py         # data -> swift/TRL GRPO format
│   ├── measure_varref_guard.py  # precision effect of the var_ref guard
│   ├── run_edge_validation.sh
│   ├── run_semantic_gap.sh
│   ├── run_outcome_length_baseline.sh
│   ├── run_topoprm_repro.sh
│   ├── run_nonqwen_eval.sh
│   └── run_dag_visualization.sh
└── outputs/
    ├── p0_audit.md                     # code/claim audit
    ├── edge_validation_results.json    # P/R/F1 + per-edge-type + var_ref guard effect
    ├── semantic_gap_table.csv/.json    # Pr(wrong|high q_topo) etc.
    ├── dag_cases/                      # rendered DAG success/failure figures
    ├── eval_tables/                    # per-run metric JSONs
    └── logs/                           # per-experiment run logs
```

## Key results

| Experiment | Result | Answers |
| --- | --- | --- |
| Edge validation (120 traces, Qwen3-32B judge) | P=0.48 R=0.59 F1=0.53; var_ref weakest (P=0.25, 52% of FPs) | HxUk W1, B5w7 W1, TsKG W1 |
| var_ref precision guard (>=2 shared vars) | F1 0.53 -> 0.62 (P 0.48->0.57, R 0.59->0.68) | HxUk W1, TsKG W1 |
| Structure-semantic gap (137 traces) | Pr(wrong \| q_topo>0.8)=0.73 (0.64 GSM8K, 0.79 MATH) | B5w7 W2, TsKG comment |
| Outcome+length GRPO baseline (DR1-7B) | see eval_tables/outcome_length_dr1_7b_* | HxUk W2, B5w7 W3 |
| TopoPRM reproducibility (released DR1-7B) | see eval_tables/topoprm_dr1_7b_repro_* | TsKG W2/W5 |
| Code audit | public math = exact answer reward (math_verify); "<500 tokens" corrected | TsKG W2/comment |

## Environment notes

- Python: `/Knowin/foundation/weilinruan/env/topoprm/bin/python` (swift 4.2, torch 2.5.1+cu121, trl 0.28, transformers 5.5.4).
- vLLM is broken in this env (0.20.2 dist-info shadowed by a 0.6.0 stub; reinstall needs torch 2.11). GRPO therefore trains via TRL (`use_vllm=False`) and eval uses the transformers backend.
- Independent edge judge: local Qwen3-32B (the provided external APIs were unreachable). Still independent of the rule-based extractor.

## Reproduce

```bash
source rebuttal/scripts/env.sh
bash rebuttal/rebuttal.sh p0      # edge validation + semantic gap + dag viz
bash rebuttal/rebuttal.sh p1      # outcome+length baseline (train + eval)
bash rebuttal/rebuttal.sh evals   # reproducibility + non-Qwen evals
```
