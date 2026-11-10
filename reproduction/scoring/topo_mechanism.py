"""T3: recompute all coefficient interventions on the same saved online rollouts."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from common import read,save,write,sha

def coefficients(c,u,z):
 delta=np.zeros_like(u)
 for mask in [c==1,c==0]:
  mean=(u*mask).sum(1,keepdims=True)/np.maximum(mask.sum(1,keepdims=True),1)
  delta+=np.where(mask,u-mean,0)
 one=np.where(c==1,np.maximum(delta,0),np.minimum(delta,0))
 clip=lambda v:np.where(c==1,np.clip(v,0,1),np.clip(v,-1,0))
 return {'ace':clip(z+one),'clip_only':clip(z),'scae':c-c.mean(1,keepdims=True)+one}
def reward(comp):
 structural=comp[:,:,2:4];span=np.ptp(structural,axis=1,keepdims=True)
 scaled=np.where(span>0,(structural-structural.min(1,keepdims=True))/np.maximum(span,1e-8),.5)
 u=.6*scaled[:,:,0]+.4*scaled[:,:,1];base=.7*comp[:,:,0]+.15*comp[:,:,1]+.15*comp[:,:,4]
 raw=np.maximum(base,.05)*(1+u);return np.clip(raw,0,1),u,raw
def standardized(r):return (r-r.mean(1,keepdims=True))/(r.std(1,ddof=1,keepdims=True)+1e-4)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--rollouts',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 rows=read(a.rollouts);assert rows and len(rows)%2==0
 for i in range(0,len(rows),2):
  assert rows[i]['prompt']==rows[i+1]['prompt'] and rows[i]['step']==rows[i+1]['step'],'Group order mismatch'
 comp=np.array([[r['components'][k] for k in ['outcome','format','topology','continuity','length']] for r in rows],dtype=np.float32).reshape(-1,2,5)
 c=comp[:,:,0];r,u,raw=reward(comp);z=standardized(r);values=coefficients(c,u,z);values['ace_without_saturation']=coefficients(c,u,standardized(raw))['ace']
 actual=np.array([v['advantage'] for v in rows]).reshape(-1,2);mode={v['advantage_mode'] for v in rows};assert len(mode)==1;mode=mode.pop();assert mode in values
 errors={'reward':float(np.max(np.abs(r.ravel()-[v['reward'] for v in rows]))),'auxiliary':float(np.max(np.abs(u.ravel()-[v['auxiliary'] for v in rows]))),'standardized_return':float(np.max(np.abs(z.ravel()-[v['standardized_return'] for v in rows]))),'actual_coefficient':float(np.max(np.abs(values[mode]-actual)))}
 assert max(errors.values())<2e-4,('Recompute mismatch',errors)
 tol=1e-6;ties=np.ptp(r,axis=1)<=tol;saturated=raw>1+tol
 original=comp.copy();original[:,:,2]=np.array([x['graph_diagnostic']['original_topology'] for x in rows]).reshape(-1,2)
 original_r,original_u,_=reward(original);original_a=coefficients(c,original_u,standardized(original_r))[mode]
 result={'scope':'offline signal comparison on one online policy cohort, not training-effect estimate','rollouts':len(rows),'groups':len(c),'input':str(a.rollouts),'input_sha256':sha(a.rollouts),'script_sha256':sha(__file__),'mode_used_for_online_loss':mode,'recompute_max_error':errors,'saturated_rollouts':int(saturated.sum()),'tied_return_groups':int(ties.sum()),'tied_groups_with_nonzero_ace':int((ties & (np.abs(values['ace'])>tol).any(1)).sum()),'all_correct_groups':int((c==1).all(1).sum()),'all_wrong_groups':int((c==0).all(1).sum()),'mixed_groups':int(((c==1).any(1)&(c==0).any(1)).sum()),'coefficient_comparisons':{},'graph_intervention':{'rollouts_with_changed_edges':sum(x['graph_diagnostic']['changed_edges']>0 for x in rows),'rollouts_with_changed_raw_topology':sum(x['graph_diagnostic']['topology_changed']>0 for x in rows),'rollouts_with_changed_total_reward':int((np.abs(r-original_r)>tol).sum()),'rollouts_with_changed_coefficient':int((np.abs(values[mode]-original_a)>tol).sum())}}
 for name,v in values.items():
  diff=np.abs(v-values['ace']);result['coefficient_comparisons'][name]={'differs_from_ace_rollouts':int((diff>tol).sum()),'mean_abs_difference':float(diff.mean()),'nonzero_rollouts':int((np.abs(v)>tol).sum()),'correct_negative':int(((c==1)&(v < -tol)).sum()),'wrong_positive':int(((c==0)&(v > tol)).sum())}
 details=[]
 for i,x in enumerate(rows):
  group,idx=divmod(i,2);details.append({'row':i,'step':x['step'],'group':group,'correct':float(c[group,idx]),'auxiliary':float(u[group,idx]),'reward':float(r[group,idx]),'uncapped_reward':float(raw[group,idx]),'actual_advantage':x['advantage'],**{k:float(v[group,idx]) for k,v in values.items()},'original_graph_reward':float(original_r[group,idx]),'original_graph_coefficient':float(original_a[group,idx])})
 a.output.mkdir(parents=True,exist_ok=True);write(a.output/'coefficients.jsonl',details);save(a.output/'metrics.json',result);print(json.dumps(result))
if __name__=='__main__':main()
