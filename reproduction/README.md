# TopoPRM supplementary experiments

This anonymous release candidate contains the executed scientific code, frozen training and evaluation inputs, raw outputs, exact run configurations and a CPU reconstruction of Tables R1–R4. The 13 evaluated models belong to the new controlled experiment suite; the shared SFT is a fourteenth checkpoint. Historical manuscript scores and previously released Full models are separate experiments.

## Reproduce the tables

```bash
python reproduce_tables.py
python -m pip install -r requirements-scoring-lock.txt
python rescore_raw.py
```

The first command verifies file hashes and reconstructs every table cell from per-item records. The second evaluation script rescores 32,812 saved answers, recomputes 3,200 rollout coefficients and reconstructs 1,536 TGD attempts from the original revision and model-judge outputs. It makes no model calls. Generated verification outputs are written under `recomputed/`.

Accuracy and length are means over trajectories within each benchmark. Avg3 averages GSM8K, MATH-500 and OlympiadBench equally; Tokens averages all four benchmarks equally. The R1 prose then averages seeds 42/123 equally. The table retains both seeds. Model-judge repair labels are not human annotations.

## Reproduce training and model evaluation

`provenance/executed_jobs.json` records the exact successful commands. `run_recorded.py` maps their paths to a chosen workspace without changing scientific arguments; by default it only prints the command. Training requires the isolated Topo dependencies: Transformers 5.5.4, TRL 0.28.0, PEFT 0.18.1, datasets 5.0.0 and math-verify 0.8.0. The preserved generic code requirements differ from this executed environment. In a dedicated virtualenv, run `python prepare_runtime.py --apply` for the recorded TRL import-availability adjustment; it refuses to modify a shared installation. Do not include the optional vLLM backend in this Transformers-based execution environment.

1. Obtain the official Qwen3.5-9B source files whose exact hashes are in `provenance/base_source_identity.json`. Run `python project_base.py --source /path/to/source --output /path/to/work/models/base_text`. The CPU extraction verifies source hashes, removes visual tensors and preserves every text tensor.
2. Run `python run_recorded.py topo_shared_sft --workspace /path/to/work` to inspect the common 128-update SFT command. Add `--execute` when the required training environment and GPU are provisioned.
3. Run the ten `t1_{ace,clip,scae,local,random}_{42,123}` jobs from the same SFT adapter. Each uses 200 updates. Run their matching `eval_t1_*` jobs with the frozen benchmark files included under `code/data/benchmarks/`.
4. Run `t2_topology`, `t2_continuity`, and `t2_generic` from `t1_ace_42/final`, with the shared-SFT teacher. Each uses 512 attempts and 64 updates; these jobs require two visible GPUs. Then run the matching `eval_t2_*` jobs.

The runner leaves device selection to the invoking environment. Run one selected job at a time, or provision a scheduler externally. It never submits cloud jobs. The declared run and input identities are in `provenance/runs/` and `data/`. New evaluation samples may differ due to hardware-dependent generation; the archived raw outputs reproduce the reported tables exactly.

## Checkpoint release

`checkpoints.json` identifies all fourteen merged checkpoints by SHA256. Their weights are not embedded in this small code/data archive. They remain pending release; no existing checkpoint is presented as an interchangeable substitute. When provided, place the chosen model directory under `/path/to/work/results/<run>/final` to execute its recorded evaluation command. The shared SFT adapter is needed to rerun Stage-II training from the original base.

Private paths, host names and personal repository references have been removed from this copy. Raw dataset rows, model answers, rollout records and per-item decisions are unchanged. Scientific source hashes and the release file manifest document the retained identities.
