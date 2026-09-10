#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="${PYTHON_ENV_BIN}:$PATH"
mkdir -p output/analysis
LOG="output/auto_post_exp_v2_$(date +%Y%m%d_%H%M%S).log"

{
  echo "[auto_post_v2] started at $(date '+%F %T')"
  while pgrep -f "bash scripts/run_grpo.sh grpo_" >/dev/null 2>&1; do
    echo "[auto_post_v2] GRPO still running... $(date '+%F %T')"
    sleep 120
  done
  echo "[auto_post_v2] GRPO done, start eval_all"
  bash scripts/run_eval_all.sh || echo "[auto_post_v2] run_eval_all failed"

  python3 - << 'PY'
import glob,json,os,csv,datetime
rows=[]
for fp in sorted(glob.glob('output/eval/*_metrics.json')):
    try:d=json.load(open(fp))
    except:continue
    rows.append({
      'file':os.path.basename(fp),
      'num_samples':d.get('num_samples'),
      'format_compliance':d.get('format_compliance'),
      'score_accuracy':d.get('score_accuracy'),
      'error_identification_f1':d.get('error_identification_f1'),
      'step_coverage':d.get('step_coverage'),
    })
if rows:
    with open('output/eval/paper_table_summary_latest.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
md=['# Post-Experiment Summary',f'- generated_at: {datetime.datetime.now().isoformat(timespec="seconds")}','']
for r in rows:
    md.append(f"- `{r['file']}`: n={r['num_samples']}, format={r['format_compliance']}, score_acc={r['score_accuracy']}, f1={r['error_identification_f1']}, step_cov={r['step_coverage']}")
open('output/analysis/post_exp_summary.md','w').write('\n'.join(md)+'\n')
print('summary artifacts updated')
PY
  echo "[auto_post_v2] done at $(date '+%F %T')"
} | tee "$LOG"
