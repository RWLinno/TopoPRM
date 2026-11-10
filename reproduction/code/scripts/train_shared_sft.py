"""New common short SFT initialization; never represented as the historical Stage I."""
from pathlib import Path
import sys,os,json,argparse
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 assert os.environ.get('REBUTTAL_DLC')=='1'
 import torch
 from datasets import Dataset
 from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
 from trl import SFTTrainer,SFTConfig
 from peft import LoraConfig,TaskType
 set_seed(42)
 out=Path(a.output);assert not (out/'final').exists()
 tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True);tokenizer.pad_token_id=tokenizer.eos_token_id
 rows=[json.loads(l) for l in open(a.data) if l.strip()]
 kept=[r for r in rows if len(tokenizer.apply_chat_template(r['prompt']+r['completion'],tokenize=True,return_dict=False))<=4096]
 assert kept
 model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True)
 config=SFTConfig(output_dir=a.output,max_steps=128,per_device_train_batch_size=1,gradient_accumulation_steps=8,learning_rate=5e-5,lr_scheduler_type='cosine',warmup_ratio=.03,max_length=4096,packing=False,completion_only_loss=True,bf16=True,gradient_checkpointing=True,seed=42,data_seed=42,report_to='none',logging_steps=8,save_steps=64,save_total_limit=2)
 trainer=SFTTrainer(model=model,args=config,train_dataset=Dataset.from_list(kept),processing_class=tokenizer,peft_config=LoraConfig(task_type=TaskType.CAUSAL_LM,r=64,lora_alpha=128,target_modules='all-linear',lora_dropout=.05))
 out.mkdir(parents=True,exist_ok=True);(out/'run_arguments.json').write_text(json.dumps({'new_controlled_initialization':True,'rows':len(kept),'overlength_excluded':len(rows)-len(kept),'model':a.model,'steps':128,'seed':42,'effective_batch':8,'data':a.data},indent=2)+'\n')
 trainer.train();trainer.save_model(str(out/'adapter'));tokenizer.save_pretrained(out/'adapter')
 from src.training import save_merged_checkpoint
 save_merged_checkpoint(trainer.accelerator.unwrap_model(trainer.model),tokenizer,out/'final')
if __name__=='__main__':main()
