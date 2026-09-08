# ICLR 2027 workspace consolidation (2026-09-08)

Five scratch workspaces for the ICLR 2027 submission were folded into the main
repository. This note records what they held and where it went, so the removals
stay auditable.

## Single source of truth after consolidation

| What | Path |
|---|---|
| Engineering repo | `/Knowin/foundation/weilinruan/TopoPRM` |
| Paper (only LaTeX version) | `/Knowin/foundation/weilinruan/TopoPRM/paper` |
| Paper root file | `paper/iclr2027_conference.tex` |
| Archived weights / datasets | `/knowin-oss/weilinruan/topoprm_archive` |

Nothing else is a live paper workspace. Any other copy is historical.

## Removed workspaces

| Workspace | Held | Disposition |
|---|---|---|
| `demo/weilinruan/TopoPRM_ICLR27` | 5.2 GB pilot checkpoint + draft PDFs | Checkpoint to OSS `demo_canonical/`; PDFs and figure sources to `docs/archived/iclr27_draft_history/`; directory deleted |
| `demo/weilinruan/TopoPRM_ICLR27_build` | LaTeX build products, page PNGs | Final PDF kept as `iclr2027_build_snapshot.pdf`; rest deleted (regenerable) |
| `demo/weilinruan/topoprm_iclr27_env` | 234 MB Python venv | Deleted; recreate from `requirements.txt` |
| `demo/weilinruan/TopoPRM_ICLR27_smoke` | empty | Deleted |
| `foundation/weilinruan/TopoPRM_ICLR27` | Standalone ICLR git repo | Improvements merged into `paper/`; pushed to its own remote as branch `0908` before removal |

## Retained material

`docs/archived/iclr27_draft_history/` holds the draft trail:

- `iclr2027_draft_20260826.pdf`, `..._20260827.pdf`, `..._20260827_corrected.pdf`
- `topoprm_fig_iclr27_20260827.pptx` and its corrected variant (editable figure sources)
- `iclr2027_build_snapshot.pdf`

## Storage move

Model weights were moved out of the working tree, not deleted:

- 240 GB of `.safetensors` / `.bin` / `.pt` / `.pth` from `output/` → `/knowin-oss/weilinruan/topoprm_archive/checkpoints/output/`, original directory layout preserved
- 5.2 GB pilot run `grpo_full_qwen35_9b_old_parser_length_collapse_pilot_20260824` → `.../demo_canonical/`

Evidence files stayed in the repository on purpose: roughly 1.1 GB of JSON, JSONL,
CSV, and training logs under `output/` back the result tables and cannot be
regenerated without rerunning training. Weights can be retrained from the code;
run records cannot be recovered.

## Superseded section files

`paper/sections/` still contains `*_old.tex` and `*_v0904.tex` variants. They are
kept for traceability but are **not** `\input` by `iclr2027_conference.tex`. The
live section set is `0_abstract`, `1_intro`, `2_related_new`, `3_method`,
`4_experiments`, `5_conclusion`, `6_appendix`.

`tabs/private_results.tex` is referenced only from the unused
`6_appendix_old.tex`. It stays out of the compiled document; clearing its
authorization and privacy status is a prerequisite before it is ever restored.

## Terminology

Retired names were removed from every compiled section: `TGSD` → `TopoPRM`,
`DAG Scorer` → frozen edge encoder, and the distillation stage is `TGD`
(Topology-Guided Distillation). Historical documents under `docs/` keep their
original wording and were left untouched.
