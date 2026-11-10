"""Verify the bundle and reconstruct R1-R4 from per-item records (stdlib only)."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,math,re
P=Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
for name,r in load(P/'file_manifest.json')['files'].items():
    assert hashlib.sha256((P/name).read_bytes()).hexdigest()==r['sha256'],name
B=['gsm8k','math500','olympiadbench','aime2024'];N=[1319,500,675,30]
response=(P/'expected/response.md').read_text()
def table(n):
    lines=response.split('**Table '+n+'.',1)[1].splitlines()
    start=next(i for i,l in enumerate(lines) if l.startswith('|'))
    result=[]
    for line in lines[start+2:]:
        if not line.startswith('|'):break
        result.append([x.strip().replace('**','') for x in line.strip('|').split('|')])
    return result
out={};means={}
for group in ['R1','R2']:
    result=[]
    for row in table(group):
        if group=='R1':
            arm={('Graph','ACE'):'ace',('Graph','Clipping'):'clip',('Graph','SCAE'):'scae',('Local support','ACE'):'local',('Rewired graph','ACE'):'random'}[tuple(row[:2])]
            name=f't1_{arm}_{row[2]}';prefix=row[:3]
        else:
            name='t2_'+{'Graph':'topology','Local continuity':'continuity','Generic correction':'generic'}[row[0]];prefix=row[:1]
        rr=rows(P/f'results/scored/{name}/per_item.jsonl')
        assert len(rr)==2524 and len({x['uid'] for x in rr})==2524
        acc=[];length=[]
        for b,n in zip(B,N):
            x=[r for r in rr if r['benchmark']==b];assert len(x)==n
            acc.append(100*sum(r['credit'] for r in x)/n)
            length.append(sum(r['visible_tokens'] for r in x)/n)
        avg3=sum(acc[:3])/3;tokens=sum(length)/4
        expected=prefix+[f'{x:.2f}' for x in acc+[avg3]]+[f'{tokens:.1f}']
        assert row==expected,(name,row,expected)
        result.append(expected)
        means[name]={'avg3':avg3,'tokens':tokens}
    out[group]=result
out['R3']=[]
for row_i,row in enumerate(table('R3')):
    values=[]
    for seed in [42,123]:
        rr=rows(P/f'results/training/t1_ace_{seed}/mechanism/coefficients.jsonl');assert len(rr)==800
        groups=defaultdict(list)
        for r in rr:groups[r['group']].append(r)
        assert len(groups)==400 and all(len(v)==2 for v in groups.values())
        if row_i==0:
            tied=[v for v in groups.values() if abs(v[0]['reward']-v[1]['reward'])<=1e-6]
            values.append(f"{sum(any(abs(r['ace'])>1e-6 for r in v) for v in tied)}/{len(tied)}")
        else:
            key=['clip_only','scae','ace_without_saturation'][row_i-1]
            values.append(f"{sum(abs(r[key]-r['ace'])>1e-6 for r in rr)}/800")
    assert row[1:]==values,(row,values)
    out['R3'].append([row[0]]+values)
rr=rows(P/'results/t2_summary/per_attempt.jsonl');cols=[]
for arm in ['topology','continuity','generic']:
    x=[r for r in rr if r['benchmark']==arm];assert len(x)==512
    n=sum(r['accepted'] for r in x);c=sum(r['accepted_changed'] for r in x)
    cols.append([str(n),f'{n-c} / {c}',f"{sum(r['answer_corrected'] for r in x)} / {sum(r['answer_damaged'] for r in x)}",f"{sum(r['accepted_semantic_repair'] for r in x)} / {c}",f"{sum(r['supervised_tokens'] for r in x):,}"])
assert [r[1:] for r in table('R4')]==[list(x) for x in zip(*cols)]
out['R4']=table('R4')
equal_seed={arm:{key:sum(means[f't1_{arm}_{seed}'][key] for seed in [42,123])/2 for key in ['avg3','tokens']} for arm in ['ace','clip','scae','local','random']}
expected=load(P/'expected/mean_summary.json')['T1']
assert all(abs(v-expected[arm][k])<1e-9 for arm,m in equal_seed.items() for k,v in m.items())
q=P/'recomputed';q.mkdir(exist_ok=True)
(q/'tables.json').write_text(json.dumps({'tables':out,'equal_seed_means':equal_seed},indent=2)+'\n')
md=[]
for name,rr in out.items():
    md.extend(['## '+name,'','| '+' | '.join(['Field '+str(i+1) for i in range(len(rr[0]))])+' |','|'+'|'.join(['---']*len(rr[0]))+'|'])
    md.extend('| '+' | '.join(r)+' |' for r in rr);md.append('')
(q/'tables.md').write_text('\n'.join(md)+'\n')
print(json.dumps({'file_hashes':'pass','R1_R4_numeric_match':True,'benchmark_predictions':13*2524,'online_TGD_attempts':1536,'aggregation':'per-item means, equal task weights, equal training-seed weights for R1 prose','tables':str(q/'tables.json')}))
