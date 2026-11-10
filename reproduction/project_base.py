"""Losslessly project the Qwen3.5 multimodal checkpoint onto its text backbone."""
from pathlib import Path
import json,hashlib,shutil
from safetensors.torch import load_file,save_file
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
src=args.source;out=args.output;out.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
 return h.hexdigest()
identity=json.loads((Path(__file__).resolve().parent/'provenance/base_source_identity.json').read_text())
for f in identity['files']:
 assert sha(src/f['name'])==f['sha256'],f['name']
if (out/'projection_receipt.json').exists():
 print('Text backbone already sealed');raise SystemExit
c=json.loads((src/'config.json').read_text());tc=c['text_config'];tc.update(architectures=['Qwen3_5ForCausalLM'],dtype='bfloat16')
(out/'config.json').write_text(json.dumps(tc,indent=2)+'\n')
for name in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','chat_template.jinja','generation_config.json']:
 if (src/name).exists():shutil.copy2(src/name,out/name)
index=json.loads((src/'model.safetensors.index.json').read_text());mapping={};total=0;receipt=[]
for i,name in enumerate(sorted(set(index['weight_map'].values()))):
 dest=f'model-{i+1:05d}.safetensors'; tensors={}
 for key,tensor in load_file(src/name,device='cpu').items():
  if key.startswith('model.language_model.'):
   new=key.replace('model.language_model.','model.',1)
  elif key=='lm_head.weight':new=key
  else:continue
  tensors[new]=tensor;mapping[new]=dest;total+=tensor.numel()*tensor.element_size()
 assert tensors
 save_file(tensors,out/dest,metadata={'format':'pt'})
 receipt.append({'input':name,'sha256':sha(src/name),'output':dest,'output_sha256':sha(out/dest),'tensors':len(tensors)})
 print(dest,len(tensors),flush=True)
(out/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':total},'weight_map':mapping},indent=2)+'\n')
(out/'projection_receipt.json').write_text(json.dumps({'source':str(src),'source_config_sha256':sha(src/'config.json'),'operation':'Remove visual modules; rename model.language_model to model; preserve every text tensor exactly; no training','tensors':len(mapping),'files':receipt},indent=2)+'\n')
print('TEXT_READY',out,flush=True)
