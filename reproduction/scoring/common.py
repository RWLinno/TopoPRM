"""CPU-only, evidence-bound summaries and paired bootstrap (protocol v1)."""
from pathlib import Path
import hashlib,json,datetime
import numpy as np

SEED=20261107
REPEATS=10000
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return [json.loads(l) for l in Path(p).open() if l.strip()]
def save(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def write(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in rows))
def completed_predictions(folder,inputs):
 folder=Path(folder);p=folder/'predictions.jsonl';receipt=json.loads((folder/'complete.json').read_text());protocol=json.loads((folder/'protocol.json').read_text())
 assert receipt['predictions_sha256']==sha(p) and receipt['protocol_sha256']==sha(folder/'protocol.json')
 assert protocol['input_sha256']==sha(inputs)
 rows=read(inputs);pred=read(p);assert len(pred)==receipt['rows']==len(rows)
 assert len({str(x['uid']) for x in rows})==len(rows)
 assert [str(x['uid']) for x in rows]==[str(x['uid']) for x in pred]
 for r,q in zip(rows,pred):assert q['prompt_sha256']==hashlib.sha256(r['prompt'].encode()).hexdigest()
 return rows,pred,{'input':str(inputs),'input_sha256':sha(inputs),'predictions':str(p),'predictions_sha256':sha(p),'protocol_sha256':sha(folder/'protocol.json')}
def summaries(rows,group='benchmark'):
 out={}
 for k in sorted({x[group] for x in rows}):
  rr=[x for x in rows if x[group]==k];keys=sorted(set.intersection(*[set(x) for x in rr]))
  means={f:float(np.mean([x[f] for x in rr])) for f in keys if all(isinstance(x[f],(int,float,bool)) for x in rr)}
  out[k]={'n':len(rr),'means':means}
 return out
def paired(left,right,metric='credit',group='benchmark',cluster=None):
 """Right minus left; fixed UID pairing; cluster resampling shares draws across forms."""
 a={r['uid']:r for r in left};b={r['uid']:r for r in right};assert len(a)==len(left) and len(b)==len(right) and set(a)==set(b),'UID mismatch'
 keys=sorted(a);groups=sorted({a[k][group] for k in keys});rng=np.random.default_rng(SEED);draws=[];result={}
 if cluster:
  clusters=sorted({a[k][cluster] for k in keys});ci={s:i for i,s in enumerate(clusters)};gi={s:i for i,s in enumerate(groups)}
  sums=np.zeros((len(clusters),len(groups)));counts=sums.copy()
  for k in keys:
   assert a[k][cluster]==b[k][cluster] and a[k][group]==b[k][group]
   i,j=ci[a[k][cluster]],gi[a[k][group]];sums[i,j]+=b[k][metric]-a[k][metric];counts[i,j]+=1
  boot=[]
  for start in range(0,REPEATS,128):
   ids=rng.integers(len(clusters),size=(min(128,REPEATS-start),len(clusters)));den=counts[ids].sum(1);assert np.all(den>0),'empty form in cluster bootstrap'
   boot.append(sums[ids].sum(1)/den)
  matrix=np.concatenate(boot);draws=[matrix[:,j] for j in range(len(groups))]
 else:
  for g in groups:
   kk=[k for k in keys if a[k][group]==g];assert all(b[k][group]==g for k in kk)
   delta=np.array([b[k][metric]-a[k][metric] for k in kk]);chunks=[]
   for start in range(0,REPEATS,128):chunks.append(delta[rng.integers(len(kk),size=(min(128,REPEATS-start),len(kk)))].mean(1))
   draws.append(np.concatenate(chunks))
 for g,boot in zip(groups,draws):
  kk=[k for k in keys if a[k][group]==g];lv=np.array([a[k][metric] for k in kk],dtype=float);rv=np.array([b[k][metric] for k in kk],dtype=float)
  result[g]={'n':len(kk),'left':float(lv.mean()),'right':float(rv.mean()),'delta':float((rv-lv).mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'improved':int((rv>lv).sum()),'worsened':int((rv<lv).sum()),'equal':int((rv==lv).sum())}
 macro=np.mean(draws,axis=0)
 return {'metric':metric,'direction':'right_minus_left','seed':SEED,'bootstrap_repeats':REPEATS,'cluster':cluster,'groups':result,'macro_delta':float(np.mean([v['delta'] for v in result.values()])),'macro_ci95':np.quantile(macro,[.025,.975]).tolist()}
