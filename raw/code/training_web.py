"""Read-only training observatory routes. No training or inference launched by HTTP."""
import json
import re
from functools import lru_cache
from io import BytesIO
from flask import Blueprint, jsonify, request, send_file, send_from_directory
from data import ROOT, load_arrays, display_image
from core import state_text, legacy_state_text, oracle_action, questions

training_web = Blueprint('training_web', __name__)

def read(path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding='utf-8'))

def run_path(name):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Invalid run')
    path = ROOT / 'runs' / name
    if not path.is_dir():
        raise ValueError('Run not found')
    return path

@lru_cache(maxsize=4)
def jsonl_cached(path, size, stamp):
    rows = []
    with open(path, encoding='utf-8') as stream:
        for line in stream:
            if not line.endswith('\n'):
                break  # A writer may still be appending the last record.
            rows.append(json.loads(line))
    return rows

def events(path):
    if not path.is_file():
        return []
    stat = path.stat()
    return jsonl_cached(str(path), stat.st_size, stat.st_mtime_ns)

@lru_cache(maxsize=1)
def arrays():
    return load_arrays()

@training_web.get('/training')
def page():
    return send_from_directory(ROOT / 'web', 'training.html')

@training_web.get('/api/training/runs')
def runs():
    result = []
    for path in sorted((ROOT/'runs').glob('*'), reverse=True):
        config = read(path/'config.json')
        if config:
            result.append({'id':path.name,'steps':config.get('steps'),
                'batch_size':config.get('batch_size',1),
                'has_trace':(path/'batches.jsonl').is_file(),
                'kind':config.get('kind','training')})
    return jsonify(runs=result)

@training_web.get('/api/training/run/<name>')
def run(name):
    path = run_path(name)
    trace = events(path/'batches.jsonl')
    logs = read(path/'training_history.json')
    if logs is None:
        checkpoints = sorted((path/'checkpoints').glob('checkpoint-*'),
                             key=lambda p:int(p.name.split('-')[-1]))
        logs = read(checkpoints[-1]/'trainer_state.json',{}).get('log_history',[]) if checkpoints else []
    state = read(path/'live_status.json')
    if not state:
        state = read(ROOT/'status.json',{})
        if state.get('run') != name:
            state = {'phase':'complete' if (path/'report.json').exists() else 'incomplete'}
    curve = [{k:e[k] for k in ['microbatch','optimizer_step','loss','batch_size','examples_seen','gpu_allocated_mib']} for e in trace]
    return jsonify(run=name,config=read(path/'config.json',{}),status=state,
        curve=curve,logs=logs,validation=read(path/'validation_history.json',[]),
        report=read(path/'report.json'),benchmark=read(path/'benchmark.json'),
        memorization=read(path/'memorization_history.json',[]),
        batch_sweep=read(path/'batch_sweep.json',[]),diagnosis=read(path/'diagnosis.json'),
        gpu=read(path/'gpu_memory.json'),batch_count=len(trace),
        trace_note='실제 학습 forward에서 기록한 배치입니다. 확률은 보정 전 softmax입니다.' if trace else '이 실행은 배치별 입력·예측을 기록하지 않았습니다. 아래 그래프는 저장된 5-step 평균 loss입니다.')

@training_web.get('/api/training/batch/<name>/<int:index>')
def batch(name,index):
    path = run_path(name)
    trace = events(path/'batches.jsonl')
    if not 1 <= index <= len(trace):
        raise ValueError('Batch not recorded')
    result = json.loads(json.dumps(trace[index-1]))
    schema = read(path/'input_schema.json', {})
    for sample in result['samples']:
        if 'model_input' not in sample:
            sample['model_input'] = {'image':{'source':sample['source'],
                'image_id':sample['image_id'],'preprocessing':'MNIST 28x28 → RGB 112x112 (nearest)'},
                'state':sample['state'],'questions':schema.get('questions')}
            sample['input_provenance'] = schema.get('provenance','questions_not_recorded')
    return jsonify(result)

@training_web.get('/api/training/image/<source>/<int:index>')
def image(source,index):
    if source not in ('train','test') or not 0 <= index < len(arrays()[source+'_images']):
        raise ValueError('Invalid image')
    stream=BytesIO()
    display_image(arrays()[source+'_images'][index]).save(stream,format='PNG')
    stream.seek(0)
    return send_file(stream,mimetype='image/png',max_age=3600)

@training_web.get('/api/training/data/<name>')
def examples(name):
    config=read(run_path(name)/'config.json',{})
    ids=config.get('train_image_ids',[])
    offset=max(0,int(request.args.get('offset',0)))
    limit=min(40,max(1,int(request.args.get('limit',8))))
    rows=[]
    for number in range(offset,min(len(ids)*10,offset+limit)):
        image_id=ids[number//10]; current=number%10
        target=int(arrays()['train_labels'][image_id])
        rows.append({'image_id':image_id,'source':'train','current':current,
            'target_digit_metadata':target,'state':(state_text if config.get('input_language')=='en' else legacy_state_text)(current,0 if config.get('kind')=='memorization_diagnostic' else (image_id+current)%4),
            'gold':oracle_action(target,current)})
    return jsonify(rows=rows,total=len(ids)*10,questions=questions(),
        note='학습 원본 행을 재구성한 예시입니다. 실제 배치 순서는 배치 기록에서 확인하세요. target_digit_metadata는 화면 설명용이며 모델 입력에 들어가지 않습니다.')
