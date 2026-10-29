"""Warm inference latency, excluding model loading, on fixed test samples."""
import os
os.environ.setdefault('UNSLOTH_COMPILE_DISABLE','1');os.environ.setdefault('UNSLOTH_DECISION_COMPILE','0')
import time,json
from pathlib import Path
from data import ROOT,load_arrays,display_image,decision_row
from core import state_text
from model import DecisionEngine
from train import dump

run=ROOT/'runs'/'20261030-010019';engine=DecisionEngine(str(run/'adapter'))
import torch
import numpy as np
from unsloth.models.decision import _decision_logits
arrays=load_arrays();images=[display_image(x) for x in arrays['test_images'][:100]]
def measure(fn):
    torch.cuda.synchronize();start=time.perf_counter();fn();torch.cuda.synchronize();return 1000*(time.perf_counter()-start)
def stats(ms,count):
    return dict(repeats=len(ms),decisions_per_call=count,mean_ms=float(np.mean(ms)),p50_ms=float(np.percentile(ms,50)),p95_ms=float(np.percentile(ms,95)),min_ms=min(ms),max_ms=max(ms),decisions_per_second=1000*count/float(np.mean(ms)))
for i in range(5):engine.predict(images[i],state_text(i,0))
single=[measure(lambda i=i:engine.predict(images[i],state_text(i%10,0))) for i in range(100)]
results=dict(single_predict=stats(single,1),notes=['Model loading and disk IO excluded. CUDA synchronized. English state; calibrated saved model.','Single predict includes processor/tokenization and public API output conversion; images preloaded and resized.','Batch logits includes collation/transfers/model forward; encoding excluded. Batch pipeline includes image resize, encoding and batched logits.','Five warmup calls before each mode. Repeated fixed batch for batch timings; not application concurrency or cold-start latency.'])
dump(run/'full_test'/'latency.json',results)
for b in (1,8,32,64):
    rows=[decision_row(images[i],i%10,int(arrays['test_labels'][i]),0) for i in range(b)];items=engine.build(rows)
    call=lambda:_decision_logits(engine.model,engine.processor,items,batch_size=b)
    for _ in range(5):call()
    torch.cuda.reset_peak_memory_stats();ms=[measure(call) for _ in range(20)]
    results[f'batch_{b}']=dict(**stats(ms,b),peak_allocated_bytes=torch.cuda.max_memory_allocated())
    dump(run/'full_test'/'latency.json',results)
    print('BATCH',b,results[f'batch_{b}'],flush=True)
def pipeline():
    rows=[decision_row(display_image(arrays['test_images'][i]),i%10,int(arrays['test_labels'][i]),0) for i in range(32)]
    _decision_logits(engine.model,engine.processor,engine.build(rows),batch_size=32)
for _ in range(5):pipeline()
results['batch32_pipeline']=stats([measure(pipeline) for _ in range(20)],32)
results['complete']=True;dump(run/'full_test'/'latency.json',results)
print('LATENCY_COMPLETE',flush=True)
