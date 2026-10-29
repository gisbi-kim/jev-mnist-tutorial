"""Read-only full test summary and filtered failure examples."""
from flask import Blueprint,jsonify,request
from training_web import run_path,read,events
from core import oracle_action,state_text
full_test_web=Blueprint('full_test_web',__name__)

@full_test_web.get('/api/training/full-test/<name>')
def summary(name):
    p=run_path(name)/'full_test'
    return jsonify(summary=read(p/'summary.json'),config=read(p/'config.json'),error=read(p/'error.json'),parity=read(p/'batch_parity.json'),analysis=read(p/'analysis.json'),latency=read(p/'latency.json'))

@full_test_web.get('/api/training/full-test/<name>/errors')
def errors(name):
    records=events(run_path(name)/'full_test'/'images.jsonl')
    target=request.args.get('target',type=int);current=request.args.get('current',type=int)
    mode=request.args.get('mode','action');offset=max(0,request.args.get('offset',0,type=int));found=[]
    for r in records:
        if target is not None and r['target']!=target:continue
        for s,pred in enumerate(r['predictions']):
            if current is not None and current!=s:continue
            gold=oracle_action(r['target'],s);episode=r['episodes'][s]
            if (pred==gold if mode=='action' else episode['success']):continue
            found.append(dict(image_id=r['image_id'],source='test',target=r['target'],current=s,
                state=state_text(s,0),gold=gold,prediction=pred,probabilities=r['probabilities'][s],episode=episode))
    return jsonify(total=len(found),offset=offset,rows=found[offset:offset+12])
