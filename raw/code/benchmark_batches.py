"""Controlled batch throughput check with real per-forward telemetry."""
import os
os.environ.setdefault('UNSLOTH_COMPILE_DISABLE','1')
os.environ.setdefault('UNSLOTH_DECISION_COMPILE','0')
import argparse
from collections import Counter
from datetime import datetime
import json
import random
import time
from pathlib import Path
from model import DecisionEngine
from data import ROOT, DATA, load_arrays, display_image, decision_row
from train import dump
from telemetry import traced_trainer_class

def run(batch_size, steps):
    engine=DecisionEngine('unsloth/Qwen3.5-0.8B',train=True)
    import torch
    from unsloth import DecisionTrainer
    from transformers import TrainingArguments, TrainerCallback
    arrays=load_arrays(); splits=json.loads((DATA/'splits.json').read_text())
    ids=splits['train'][:50]; rng=random.Random(3407)
    rows=[]; metadata={}
    for i in ids:
        image=display_image(arrays['train_images'][i]); target=int(arrays['train_labels'][i])
        for current in range(10):
            row=decision_row(image,current,target,(i+current)%4); rows.append(row)
            metadata[id(row)]={'image_id':i,'source':'train','current':current,
                'state':row['state'],'gold':row['gold']['move'],'target_digit_metadata':target}
    groups={a:[r for r in rows if r['gold']['move']==a] for a in ['-1','0','+1']}
    size=max(map(len,groups.values()))
    balanced=[row for group in groups.values() for row in group+rng.choices(group,k=size-len(group))]
    rng.shuffle(balanced)
    items=engine.build(balanced)
    folder=ROOT/'runs'/(datetime.now().strftime('%Y%m%d-%H%M%S')+f'-batch{batch_size}')
    folder.mkdir(parents=True)
    config={'model':'unsloth/Qwen3.5-0.8B','kind':'batch_benchmark','input_language':'en','steps':steps,
        'batch_size':batch_size,'grad_accum':8//batch_size,'train_images':50,
        'train_image_ids':ids,'balanced_training_rows':len(items),
        'order':'identical seeded rows; sequential sampler; effective batch 8',
        'trace_batches':True,'compile_disabled':'1'}
    dump(folder/'config.json',config)
    durations=[]
    class Progress(TrainerCallback):
        def on_step_begin(self,args,state,control,**kwargs):
            torch.cuda.synchronize(); self.started=time.perf_counter()
            dump(folder/'live_status.json',{'phase':'training','step':state.global_step,'max_steps':steps})
        def on_step_end(self,args,state,control,**kwargs):
            torch.cuda.synchronize(); durations.append(time.perf_counter()-self.started)
        def on_log(self,args,state,control,**kwargs):
            dump(folder/'training_history.json',state.log_history)
    class Trainer(traced_trainer_class(DecisionTrainer)):
        def _get_train_sampler(self,train_dataset=None):
            return torch.utils.data.SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)
    trainer=Trainer(model=engine.model,processing_class=engine.processor,train_dataset=items,
        callbacks=[Progress()],args=TrainingArguments(output_dir=str(folder/'checkpoints'),
            per_device_train_batch_size=batch_size,gradient_accumulation_steps=8//batch_size,
            max_steps=steps,learning_rate=2e-4,bf16=True,fp16=False,logging_steps=1,
            save_strategy='no',eval_strategy='no',report_to='none',dataloader_num_workers=0,seed=3407))
    trainer.enable_trace(folder,[metadata[id(row)] for row in balanced])
    torch.cuda.reset_peak_memory_stats()
    try:
        result=trainer.train()
        steady=durations[2:]
        benchmark={'batch_size':batch_size,'grad_accum':8//batch_size,'effective_batch':8,
            'optimizer_steps':steps,'warmup_steps_excluded':2,'step_seconds':durations,
            'steady_seconds_per_update':sum(steady)/len(steady),
            'steady_examples_per_second':8*len(steady)/sum(steady),
            'training_metrics':result.metrics,'telemetry_enabled':True}
        dump(folder/'benchmark.json',benchmark)
        dump(folder/'gpu_memory.json',{'allocated_bytes':torch.cuda.max_memory_allocated(),
            'reserved_bytes':torch.cuda.max_memory_reserved()})
        dump(folder/'live_status.json',{'phase':'complete','step':steps})
        print(json.dumps({'run':folder.name,**benchmark},indent=2))
    except Exception as error:
        dump(folder/'live_status.json',{'phase':'failed','error':str(error)})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch-size',type=int,choices=[1,2,4,8],required=True)
    p.add_argument('--steps',type=int,default=12);a=p.parse_args()
    if a.steps<3:p.error('At least 3 steps required')
    run(a.batch_size,a.steps)
