#!/bin/bash
set -euo pipefail

###############################################################################
# Auto post-experiment pipeline
# 1) Wait until serial GRPO runner finishes
# 2) Run unified evaluation for all completed checkpoints
# 3) Dump latest metric summary CSV/MD for paper filling
#
# Usage:
#   bash scripts/auto_post_exp.sh
###############################################################################

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

PATTERN='for exp in grpo_no_topo grpo_no_continuity grpo_clipped grpo_confgate grpo_mulgate grpo_scae'
LOG="output/auto_post_exp_$(date +%Y%m%d_%H%M%S).log"
mkdir -p output/analysis

{
  echo "[auto_post_exp] started at $(date '+%F %T')"
  echo "[auto_post_exp] waiting for serial GRPO loop to finish..."

  while pgrep -f "$PATTERN" >/dev/null 2>&1; do
    echo "[auto_post_exp] training still running... $(date '+%F %T')"
    sleep 120
  done

  echo "[auto_post_exp] serial GRPO finished. start evaluation..."
  bash scripts/run_eval_all.sh

  echo "[auto_post_exp] building metric summaries..."
  python3 - << 'PY'
import glob,json,os,csv,datetime
rows=[]
for fp in sorted(glob.glob('output/eval/*_metrics.json')):
    try:
        d=json.load(open(fp))
    except Exception:
        continue
    rows.append({
        'file': os.path.basename(fp),
        'num_samples': d.get('num_samples'),
        'format_compliance': d.get('format_compliance'),
        'score_accuracy': d.get('score_accuracy'),
        'error_identification_f1': d.get('error_identification_f1'),
        'step_coverage': d.get('step_coverage'),
    })

csv_path='output/eval/paper_table_summary_latest.csv'
if rows:
    with open(csv_path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

md=['# Post-Experiment Summary', '', f'- generated_at: {datetime.datetime.now().isoformat(timespec="seconds")}', '']
for r in rows:
    md.append(f"- `{r['file']}`: n={r['num_samples']}, format={r['format_compliance']}, score_acc={r['score_accuracy']}, f1={r['error_identification_f1']}, step_cov={r['step_coverage']}")
open('output/analysis/post_exp_summary.md','w').write('\n'.join(md)+'\n')
print('wrote', csv_path)
print('wrote output/analysis/post_exp_summary.md')
PY

  echo "[auto_post_exp] done at $(date '+%F %T')"
} | tee "$LOG"
