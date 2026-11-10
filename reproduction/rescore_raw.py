"""Rescore raw predictions and offline coefficients; no inference or training."""
from pathlib import Path
import json,subprocess,sys
P=Path(__file__).resolve().parent
def run(*args):subprocess.run([sys.executable,*map(str,args)],check=True,stdout=subprocess.DEVNULL)
arms=[f't1_{arm}_{s}' for s in [42,123] for arm in ['ace','clip','scae','local','random']]+['t2_topology','t2_continuity','t2_generic']
for name in arms:
    dest=P/'recomputed/scored'/name
    run(P/'scoring/score_topo.py','eval','--folder',P/'results/raw_eval','--label',name,'--output',dest)
    old=(P/f'results/scored/{name}/per_item.jsonl').read_text();new=(dest/'per_item.jsonl').read_text()
    assert old==new,('per_item_scoring_differs',name)
    print('Verified',name,flush=True)
for seed in [42,123]:
    for arm in ['ace','random']:
        name=f't1_{arm}_{seed}';dest=P/'recomputed/mechanism'/name
        run(P/'scoring/topo_mechanism.py','--rollouts',P/f'results/training/{name}/rollouts_rank0.jsonl','--output',dest)
        actual=json.loads((dest/'metrics.json').read_text());expected=json.loads((P/f'results/training/{name}/mechanism/metrics.json').read_text())
        for k in ['tied_return_groups','tied_groups_with_nonzero_ace','coefficient_comparisons','graph_intervention']: assert actual[k]==expected[k],(name,k)
dest=P/'recomputed/t2_summary'
run(P/'scoring/score_topo.py','revisions','--folder',P/'results/training','--output',dest,
    '--judge',P/'results/semantic_judge','--mapping',P/'data/semantic_judge_mapping.jsonl',
    '--judge-inputs',P/'data/semantic_judge_inputs.jsonl')
assert (dest/'per_attempt.jsonl').read_text()==(P/'results/t2_summary/per_attempt.jsonl').read_text(),'R4 raw reconstruction differs'
result={'raw_answer_scores_verified':13*2524,'mechanism_rollouts_verified':4*800,'TGD_attempts_reconstructed':1536,'new_model_calls':0,'status':'pass'}
(P/'recomputed/raw_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
