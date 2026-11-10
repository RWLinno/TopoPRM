"""Print a frozen command, or execute one explicitly selected job in a workspace."""
from pathlib import Path
import argparse,json,os,shlex,subprocess,sys
P=Path(__file__).resolve().parent
def command(job,workspace):
    records=json.loads((P/'provenance/executed_jobs.json').read_text())
    if job not in records:raise ValueError('Unknown recorded job: '+job)
    result=[]
    for item in records[job]['command']:
        if item.startswith('${WORKSPACE}/rebuttal/runtime/') and item.endswith('/bin/python'):
            item=sys.executable
        elif item.startswith('${WORKSPACE}/TopoPRM/rebuttal/data/'):
            item=str(P/'data'/item.split('/TopoPRM/rebuttal/data/',1)[1])
        elif item.startswith('${WORKSPACE}/TopoPRM/rebuttal/results/'):
            item=str(workspace/'results'/item.split('/TopoPRM/rebuttal/results/',1)[1])
        elif item=='${WORKSPACE}/cache/Qwen3.5-9B-text':
            item=str(workspace/'models/base_text')
        else:item=item.replace('${WORKSPACE}',str(workspace))
        assert '${' not in item,('Unresolved path',item)
        result.append(item)
    return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('job');ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    cmd=command(args.job,args.workspace.expanduser().resolve())
    print('Working directory:',P/'code');print(shlex.join(cmd),flush=True)
    if args.execute:
        for flag in ['--model','--student_model','--teacher_model','--sft_adapter','--data','--prompts']:
            if flag in cmd:
                path=Path(cmd[cmd.index(flag)+1]);assert path.exists(),f'Missing {flag}: {path}'
        env=os.environ.copy()
        for key in ['RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','NPROC_PER_NODE','PYTHONPATH']:
            env.pop(key,None)
        env.update(json.loads((P/'provenance/executed_jobs.json').read_text())[args.job]['environment'])
        env.update(TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',WANDB_MODE='disabled',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_HF='1',PYTHONUNBUFFERED='1',DS_IGNORE_CUDA_DETECTION='1')
        subprocess.run(cmd,cwd=P/'code',env=env,check=True)
if __name__=='__main__':main()
