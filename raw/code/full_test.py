"""Resumable, batched evaluation of all official MNIST test images; no training."""
import os
os.environ.setdefault('UNSLOTH_COMPILE_DISABLE','1')
os.environ.setdefault('UNSLOTH_DECISION_COMPILE','0')
import argparse, json, time, math
from collections import Counter
from pathlib import Path
from data import ROOT,load_arrays,display_image,decision_row
from core import ACTIONS,state_text,rollout_summary,oracle_action
from train import dump

def aggregate(records):
    cells=[[dict(total=0,errors=0,episode_failures=0,predictions={a:0 for a in ACTIONS}) for s in range(10)] for t in range(10)]
    confusion=[[0]*3 for _ in ACTIONS];reasons=Counter();nll=0;wrong=0;fail=0
    for r in records:
        for s,pred in enumerate(r['predictions']):
            gold=oracle_action(r['target'],s);c=cells[r['target']][s];c['total']+=1
            c['errors']+=pred!=gold;c['predictions'][pred]+=1;wrong+=pred!=gold
            confusion[ACTIONS.index(gold)][ACTIONS.index(pred)]+=1
            nll-=math.log(max(r['probabilities'][s][gold],1e-12))
            episode=r['episodes'][s];bad=not episode['success'];fail+=bad;c['episode_failures']+=bad
            if bad:reasons[episode['reason']]+=1
    n=len(records)*10
    return dict(image_count=len(records),decision_count=n,action_accuracy=1-wrong/n if n else None,
        episode_success_rate=1-fail/n if n else None,errors=wrong,failed_episodes=fail,
        nll=nll/n if n else None,cells=cells,confusion=confusion,action_order=ACTIONS,failure_reasons=dict(reasons))

def run(name,batch_size,limit=1000):
    folder=ROOT/'runs'/name/'full_test';folder.mkdir(exist_ok=True)
    checkpoint=ROOT/'runs'/name/'adapter'
    from model import DecisionEngine
    engine=DecisionEngine(str(checkpoint))
    from unsloth.models.decision import _decision_logits
    from unsloth.models._decision_common import _served_temperatures
    arrays=load_arrays();count=min(limit,len(arrays['test_labels']));records=[]
    path=folder/'images.jsonl'
    if path.exists():
        raw=path.read_bytes();last=raw.rfind(b'\n')+1
        if last<len(raw):
            with path.open('r+b') as f:f.truncate(last)
        records=[json.loads(line) for line in raw[:last].splitlines()]
    assert [r['image_id'] for r in records]==list(range(len(records)))
    dump(folder/'config.json',dict(checkpoint=str(checkpoint),test_images=count,decisions=count*10,
        batch_size=batch_size,input_language='en',state_template='The current number is {current}.',
        note='Reloaded saved calibrated model. All official test IDs, no fitting or model selection. Previously inspected 100 test images are included.'))
    def predict(items):
        logits,flat=_decision_logits(engine.model,engine.processor,items,batch_size=batch_size)
        scales=_served_temperatures(engine.model.decision_config,logits,flat)
        return [dict(zip(item['record'].questions[0].option_ids,(z/scale).softmax(-1).tolist())) for item,z,scale in zip(items,logits,scales)]
    started=time.perf_counter();initial=len(records)
    # Compare the batched path against the public single-item predict API.
    check_rows=[decision_row(display_image(arrays['test_images'][i]),s,int(arrays['test_labels'][i]),0) for i in range(3) for s in range(10)]
    probs=predict(engine.build(check_rows));deltas=[];matches=0
    for row,p in zip(check_rows,probs):
        single=engine.predict(row['images'][0],row['state']);matches+=single['action']==max(p,key=p.get)
        deltas.append(max(abs(single['probabilities'][a]-p[a]) for a in ACTIONS))
    dump(folder/'batch_parity.json',dict(cases=30,action_matches=matches,max_probability_difference=max(deltas)))
    if matches!=30:raise RuntimeError('Batched/single prediction mismatch in parity check')
    for start in range(len(records),count,100):
        ids=list(range(start,min(start+100,count)));rows=[]
        for i in ids:
            image=display_image(arrays['test_images'][i])
            rows.extend(decision_row(image,s,int(arrays['test_labels'][i]),0) for s in range(10))
        probabilities=predict(engine.build(rows))
        with path.open('a',encoding='utf-8') as f:
            for j,i in enumerate(ids):
                p=probabilities[j*10:j*10+10];pred=[max(x,key=x.get) for x in p];target=int(arrays['test_labels'][i])
                r=dict(image_id=i,target=target,predictions=pred,probabilities=p,
                    episodes=[rollout_summary(target,s,pred.__getitem__) for s in range(10)])
                records.append(r);f.write(json.dumps(r)+'\n')
        elapsed=time.perf_counter()-started
        summary=aggregate(records);summary.update(phase='complete' if len(records)==count else 'running',
            total_images=count,elapsed_seconds=elapsed,images_per_second=(len(records)-initial)/elapsed)
        dump(folder/'summary.json',summary)
        print(json.dumps({k:summary[k] for k in ['phase','image_count','action_accuracy','episode_success_rate','images_per_second']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--limit',type=int,default=1000);a=p.parse_args()
    try:run(a.run,a.batch_size,a.limit)
    except Exception as e:
        dump(ROOT/'runs'/a.run/'full_test'/'error.json',dict(error=str(e)));raise
