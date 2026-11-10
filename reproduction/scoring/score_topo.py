"""Normalize complete Topo benchmark records, and decompose every TGD attempt."""
from pathlib import Path
import argparse,hashlib,json,re
from common import read,save,write,sha,summaries,completed_predictions

EXPECTED={'gsm8k':1319,'math500':500,'olympiadbench':675,'aime2024':30}
def normalize(s):return re.sub(r'\s+',' ',s).strip()
def evaluate(folder,label,out):
 from topo_final_answer import score as final_score,VERSION,SOURCE
 manifest=Path(__file__).resolve().parents[1]/'data/benchmark_manifest.json';frozen=json.loads(manifest.read_text())['benchmarks']
 rows=[];bindings={str(manifest):sha(manifest),str(SOURCE):sha(SOURCE),str(Path(__file__).with_name('topo_final_answer.py')):sha(Path(__file__).with_name('topo_final_answer.py'))};audit={}
 for bench,n in EXPECTED.items():
  p=folder/(label+'_'+bench+'_details.jsonl');mp=folder/(label+'_'+bench+'_metrics.json');rr=read(p);metrics=json.loads(mp.read_text());assert len(rr)==n and len({x['item_id'] for x in rr})==n
  assert all(x['num_samples']==1 for x in rr)
  identity=frozen[bench];assert identity['n']==n and [x['item_id'] for x in rr]==identity['ordered_item_ids'],('evaluation_cohort_changed',bench)
  payload='\n'.join(f"{bench}\t{x['question']}\t{x['gold']}" for x in rr)
  assert hashlib.sha256(payload.encode()).hexdigest()==metrics['dataset_fingerprint']==identity['dataset_fingerprint'],('evaluation_questions_or_gold_changed',bench)
  changed=[];categories={}
  for x in rr:
   verified,category=final_score(x['response'],x['gold']);categories[category]=categories.get(category,0)+1
   if verified!=x['correct_pass1']:changed.append({'item_id':x['item_id'],'recorded_correct':bool(x['correct_pass1']),'verified_correct':verified,'answer_source':category})
   r={'uid':bench+':'+x['item_id'],'benchmark':bench,'credit':float(verified),'recorded_credit':float(x['correct_pass1']),'answer_source':category,'visible_tokens':x['gen_tokens_pass1']}
   if 'finish_reason_pass1' in x:r.update(generated_tokens=x['generated_tokens_pass1'],truncated=x['finish_reason_pass1']=='length')
   rows.append(r)
  correct=sum(x['correct_pass1'] for x in rr);actual=correct/n
  assert metrics['correct']==correct and metrics['error']==n-correct,('recorded_counts_disagree',bench)
  assert round(actual,4)==metrics['pass@1'],('recorded_rounded_pass1_disagrees',bench)
  bindings[str(p)]=sha(p);bindings[str(mp)]=sha(mp)
  audit[bench]={'n':n,'recorded_correct':correct,'corrected_correct':sum(r['credit'] for r in rows if r['benchmark']==bench),'changed_items':changed,'answer_sources':categories}
 summary=summaries(rows);save(out/'metrics.json',{'label':label,'answer_contract':VERSION,'bindings':bindings,'scoring_sha256':sha(__file__),'benchmarks':summary,'macro_accuracy':sum(v['means']['credit'] for v in summary.values())/4,'macro_accuracy_without_aime':sum(v['means']['credit'] for k,v in summary.items() if k!='aime2024')/3,'aime_correct':sum(x['credit'] for x in rows if x['benchmark']=='aime2024'),'truncation_available':all('truncated' in x for x in rows)})
 save(out/'answer_contract_audit.json',{'version':VERSION,'source_predictions_unchanged':True,'benchmarks':audit})
 write(out/'per_item.jsonl',rows)
def revisions(folder,out,judge=None,mapping=None,judge_inputs=None):
 inputs={};judged={};bindings={};maprows=[]
 if judge:
  assert mapping and judge_inputs
  _,pp,binding=completed_predictions(judge,judge_inputs);bindings['judge']=binding;maprows=read(mapping);bindings['mapping_sha256']=sha(mapping)
  for p in pp:
   text=p['response'];start=text.find('{');end=text.rfind('}')
   try:
    q=json.loads(text[start:end+1]);assert all(type(q.get(k)) is bool for k in ['original_has_local_gap','revised_resolves_that_gap','introduces_error','uncertain']);assert isinstance(q.get('evidence'),str)
    judged[p['uid']]={'valid':True,'repair':q['original_has_local_gap'] and q['revised_resolves_that_gap'] and not q['introduces_error'] and not q['uncertain'],'uncertain':q['uncertain'],'evidence':q['evidence']}
   except (ValueError,AssertionError):judged[p['uid']]={'valid':False,'repair':False,'uncertain':True,'evidence':''}
 rows=[]
 for arm in ['topology','continuity','generic']:
  p=folder/('t2_'+arm)/'revisions.jsonl';summary=json.loads((p.parent/'summary.json').read_text());rr=read(p);assert len(rr)==summary['attempted']==512
  assert sum(x['keep'] for x in rr)==summary['accepted'];bindings[str(p)]=sha(p)
  lookup={m['attempt']:m for m in maprows if m['arm']==arm}
  for x in rr:
   original=x.get('original',{});revised=x.get('revised',{});changed=bool(x.get('y_revised')) and normalize(x.get('y_init',''))!=normalize(x['y_revised'])
   r={'uid':arm+':'+str(x['attempt']),'benchmark':arm,'record_id':x['record_id'],'accepted':x['keep'],'changed':changed,'accepted_changed':x['keep'] and changed,'supervised_tokens':x.get('revised_tokens',0) if x['keep'] else 0,'answer_corrected':original.get('r_out')==0 and revised.get('r_out')==1,'answer_damaged':original.get('r_out')==1 and revised.get('r_out')==0,'topology_improved':revised.get('r_topo',0)>original.get('r_topo',0),'continuity_improved':revised.get('r_cont',0)>original.get('r_cont',0),'rejection_reason':x['rejection_reason']}
   if judge:
    m=lookup[x['attempt']];assert m['changed']==changed and m['keep']==x['keep'];q=judged[m['uid']] if changed else {'valid':False,'repair':False,'uncertain':False,'evidence':''}
    r.update(semantic_repair=q['repair'],accepted_semantic_repair=x['keep'] and q['repair'],judge_uncertain=q['uncertain'],judge_parse_failure=changed and not q['valid'],semantic_evidence=q['evidence'])
   rows.append(r)
  inputs[arm]=summary
 result={'bindings':bindings,'training_summaries':inputs,'all_attempts':summaries(rows),'accepted_only':summaries([r for r in rows if r['accepted']]),'semantic_judgment':'independent blinded model, not human' if judge else 'not yet available','scoring_sha256':sha(__file__)}
 write(out/'per_attempt.jsonl',rows);save(out/'metrics.json',result)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('kind',choices=['eval','revisions']);ap.add_argument('--folder',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--label');ap.add_argument('--judge',type=Path);ap.add_argument('--mapping',type=Path);ap.add_argument('--judge-inputs',type=Path);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 if a.kind=='eval':assert a.label;evaluate(a.folder,a.label,a.output)
 else:revisions(a.folder,a.output,a.judge,a.mapping,a.judge_inputs)
 print('SCORED',a.kind,str(a.output))
if __name__=='__main__':main()
