"""Measured batch sweep and exact tiny-set memorization, with live evidence."""
import os
os.environ.setdefault('UNSLOTH_COMPILE_DISABLE','1')
os.environ.setdefault('UNSLOTH_DECISION_COMPILE','0')
import json, time, gc
from datetime import datetime
from model import DecisionEngine
from data import ROOT, DATA, load_arrays, display_image, decision_row
from train import dump, evaluate
from telemetry import traced_trainer_class

def run():
    engine=DecisionEngine('unsloth/Qwen3.5-0.8B',train=True)
    import torch
    from unsloth import DecisionTrainer
    from transformers import TrainingArguments, TrainerCallback
    arrays=load_arrays(); splits=json.loads((DATA/'splits.json').read_text())
    ids=[next(i for i in splits['train'] if int(arrays['train_labels'][i])==digit) for digit in (2,5,8)]
    held=[next(i for i in splits['validation'] if int(arrays['train_labels'][i])==digit) for digit in (2,5,8)]
    rows=[]; meta=[]
    for i in ids:
        target=int(arrays['train_labels'][i])
        for current in range(10):
            row=decision_row(display_image(arrays['train_images'][i]),current,target,0)
            assert row['state'].isascii() and json.dumps(row['questions'],ensure_ascii=False).isascii()
            rows.append(row); meta.append(dict(image_id=i,source='train',current=current,state=row['state'],gold=row['gold']['move'],target_digit_metadata=target))
    items=engine.build(rows)
    folder=ROOT/'runs'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-memorize')
    folder.mkdir(parents=True)
    config=dict(kind='memorization_diagnostic',input_language='en',steps=300,batch_size=30,grad_accum=1,
                train_images=3,train_image_ids=ids,validation_image_ids=held,balanced_training_rows=30,
                learning_rate=2e-4,head_learning_rate=1e-3,trace_batches=True,
                note='30 exact training pairs. Train-set memorization is not generalization. Full batch; no duplicated rows just to consume memory.')
    dump(folder/'config.json',config)
    def status(phase,step=0,**kw):dump(folder/'live_status.json',dict(phase=phase,step=step,max_steps=300,**kw))
    status('preparing')
    trainer=DecisionTrainer(model=engine.model,processing_class=engine.processor,train_dataset=items,
        args=TrainingArguments(output_dir=str(folder/'probe'),per_device_train_batch_size=30,bf16=True,report_to='none'))
    # Audit actual collated image tensors and token differences before fitting.
    a=trainer.data_collator([items[0],items[1],items[10]])
    audit={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in a.items() if isinstance(v,torch.Tensor)}
    audit['token_difference_state']=bool((a['input_ids'][0]!=a['input_ids'][1]).any())
    audit['row_count']=len(items)
    dump(folder/'input_audit.json',audit)
    del a
    # Probe backward memory and throughput without optimizer updates (no contamination).
    prior=os.environ.get('BATCH_SWEEP_SOURCE')
    probes=json.loads((ROOT/'runs'/prior/'batch_sweep.json').read_text()) if prior else []
    if prior:dump(folder/'batch_sweep.json',probes)
    engine.api.for_training(engine.model)
    for batch in (() if prior else (8,16,32,64,128)):
        status('preparing',batch_probe=batch)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(); durations=[]
        try:
            for repeat in range(3):
                inputs=trainer._prepare_inputs(trainer.data_collator([items[i%len(items)] for i in range(batch)]))
                torch.cuda.synchronize(); start=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=trainer.compute_loss(engine.model,inputs)
                loss.backward(); torch.cuda.synchronize();durations.append(time.perf_counter()-start)
                engine.model.zero_grad(set_to_none=True);del inputs,loss
            peak=torch.cuda.max_memory_allocated()
            probes.append(dict(batch_size=batch,examples_per_second=batch*2/sum(durations[1:]),peak_bytes=peak,seconds=durations))
            print('BATCH_PROBE '+json.dumps(probes[-1]),flush=True)
            dump(folder/'batch_sweep.json',probes)
            if peak>torch.cuda.get_device_properties(0).total_memory*.82:break
        except torch.OutOfMemoryError:
            probes.append(dict(batch_size=batch,error='CUDA OOM'));dump(folder/'batch_sweep.json',probes)
            engine.model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();break
    del trainer;gc.collect();torch.cuda.empty_cache()
    validation=[];memorization=[]
    def measure(step):
        status('validation',step);engine.api.for_inference(engine.model)
        train=evaluate(engine,arrays,ids,source='train');val=evaluate(engine,arrays,held,source='train')
        memorization.append(dict(step=step,**train));validation.append(dict(step=step,**val))
        dump(folder/'memorization_history.json',memorization);dump(folder/'validation_history.json',validation)
        print('MEASURE '+json.dumps(dict(step=step,train_accuracy=train['action_accuracy'],train_success=train['episode_success_rate'],val_accuracy=val['action_accuracy'])),flush=True)
        engine.api.for_training(engine.model);return train
    measure(0)
    class Progress(TrainerCallback):
        def on_log(self,args,state,control,**kw):
            dump(folder/'training_history.json',state.log_history);status('training',state.global_step)
        def on_step_end(self,args,state,control,**kw):
            dump(folder/'gpu_memory.json',dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved()))
            if state.global_step%50==0:
                m=measure(state.global_step)
                if m['action_accuracy']==1 and m['episode_success_rate']==1:control.should_training_stop=True
        def on_pre_optimizer_step(self,args,state,control,**kw):
            if state.global_step==0:
                grads={}
                for group in ('encoder','head'):
                    p=[v for n,v in engine.model.named_parameters() if n.startswith(group+'.') and v.grad is not None]
                    grads[group]=dict(tensors=len(p),norm=sum(float(v.grad.float().norm())**2 for v in p)**.5)
                dump(folder/'gradient_audit.json',grads)
    trainer=traced_trainer_class(DecisionTrainer)(model=engine.model,processing_class=engine.processor,
        train_dataset=items,head_learning_rate=1e-3,callbacks=[Progress()],
        args=TrainingArguments(output_dir=str(folder/'checkpoints'),per_device_train_batch_size=30,
            gradient_accumulation_steps=1,max_steps=300,learning_rate=2e-4,lr_scheduler_type='constant',
            weight_decay=0,bf16=True,logging_steps=1,save_strategy='no',report_to='none',seed=3407))
    trainer.enable_trace(folder,meta);torch.cuda.reset_peak_memory_stats();status('training')
    trainer.train()
    engine.save(folder/'adapter')
    status('complete',trainer.state.global_step)
    dump(folder/'diagnosis.json',dict(memorized=memorization[-1]['action_accuracy']==1,
        train=memorization[-1],held_out=validation[-1],batch_recommendation=max((p for p in probes if 'error' not in p),key=lambda p:p['examples_per_second']),
        note='Head LR explicitly 1e-3; previous runs default head LR was 1e-4 independently of encoder LR.'))
    print('DIAGNOSTIC_COMPLETE '+str(folder),flush=True)

if __name__=='__main__':run()
