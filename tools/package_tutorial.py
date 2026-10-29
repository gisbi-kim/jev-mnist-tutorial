"""Freeze experiment evidence and export backend-free tutorial data. No inference."""
from pathlib import Path
import json, shutil, hashlib, urllib.request, datetime
import numpy as np
from PIL import Image

SRC=Path(r'C:\Users\USER\Downloads\mnist-decision-lab')
OUT=Path(r'C:\Users\USER\Documents\Codex\2026-10-29\e\outputs')
ROOT=OUT/'jev-mnist-tutorial'
SITE=ROOT/'site'
def read(p, default=None):
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default
def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def js(p,lhs,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(lhs+'='+json.dumps(obj,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
def copytree(a,b):
    shutil.copytree(a,b,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))

ROOT.mkdir(parents=True,exist_ok=True)
raw=ROOT/'raw'
raw.mkdir(exist_ok=True)
for f in SRC.iterdir():
    if f.is_file():
        dest=raw/('code' if f.suffix in ('.py','.ps1','.md') or f.name.startswith('requirements') else 'session_logs')/f.name
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
for name in ['web','tests']: copytree(SRC/name,raw/'code'/name)
for name in ['runs','data']: copytree(SRC/name,raw/name)
for f in OUT.iterdir():
    if f.name==ROOT.name:continue
    if f.is_file():
        (raw/'prior_reports').mkdir(exist_ok=True);shutil.copy2(f,raw/'prior_reports'/f.name)
    elif f.name=='test-report-images':copytree(f,raw/'prior_reports'/f.name)
pkg=SRC/'.venv/Lib/site-packages/unsloth'
for name in ['decision.py','clef.py','_decision_common.py','_decision_fast.py','decision_from_lm.py']:
    if (pkg/'models'/name).exists():
        d=raw/'upstream/unsloth/models'/name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(pkg/'models'/name,d)
if (pkg/'_vendor/clef').exists(): copytree(pkg/'_vendor/clef',raw/'upstream/unsloth/_vendor/clef')
for dist in (SRC/'.venv/Lib/site-packages').glob('unsloth-*.dist-info'):
    if (dist/'licenses').exists():copytree(dist/'licenses',raw/'upstream/unsloth/licenses')
    for name in ['METADATA','direct_url.json']:
        if (dist/name).exists():shutil.copy2(dist/name,raw/'upstream/unsloth'/name)

catalog=[];image_refs=set();snapshots=[]
for run in sorted((raw/'runs').iterdir()):
    if not run.is_dir():continue
    name=run.name
    with urllib.request.urlopen('http://127.0.0.1:7864/api/training/run/'+name) as response:
        snapshot=json.load(response)
    write(raw/'dashboard_api'/f'{name}.json',snapshot)
    trace=[]
    if (run/'batches.jsonl').exists():
        trace=[json.loads(s) for s in (run/'batches.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    schema=read(run/'input_schema.json',{})
    for i,b in enumerate(trace,1):
        for s in b['samples']:
            image_refs.add((s['source'],s['image_id']))
            if 'model_input' not in s:
                s['model_input']={'image':{'source':s['source'],'image_id':s['image_id']},'state':s['state'],'questions':schema.get('questions')}
                s['input_provenance']=schema.get('provenance','questions_not_recorded')
        js(SITE/'data/batches'/name/f'{i}.js','window.loadedBatches['+json.dumps(name+'/'+str(i))+']',b)
    snapshot['files']=[{'path':p.relative_to(run).as_posix(),'bytes':p.stat().st_size} for p in run.rglob('*') if p.is_file()]
    snapshot['metrics']=read(run/'train_metrics.json',{})
    snapshot['raw_batch_count']=len(trace)
    js(SITE/'data/runs'/f'{name}.js','window.loadedRuns['+json.dumps(name)+']',snapshot)
    catalog.append({'id':name,'config':snapshot['config'],'status':snapshot['status'],'batch_count':len(trace),'sample_count':sum(len(b['samples']) for b in trace)})

main=raw/'runs/20261030-010019'
test=[json.loads(s) for s in (main/'full_test/images.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
for r in test:image_refs.add(('test',r['image_id']))
full={k:read(main/'full_test'/f'{k}.json') for k in ['summary','analysis','latency','config','parity']}
full['images']=test
js(SITE/'data/test.js','window.testEvidence',full)
catalog_data={'runs':catalog,'snapshot_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'main':'20261030-010019','source':str(SRC),'recorded_batches':sum(r['batch_count'] for r in catalog),'recorded_samples':sum(r['sample_count'] for r in catalog)}
js(SITE/'data/catalog.js','window.catalog',catalog_data)
write(ROOT/'catalog.json',catalog_data)
arrays=np.load(raw/'data/mnist.npz')
for source,index in sorted(image_refs):
    p=SITE/'images'/source/f'{index}.png';p.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(arrays[source+'_images'][index]).convert('RGB').resize((112,112),Image.Resampling.NEAREST).save(p)
write(ROOT/'snapshot.json',{'created_utc':catalog_data['snapshot_at'],'source':str(SRC),'runs':len(catalog),'batches':catalog_data['recorded_batches'],'samples':catalog_data['recorded_samples'],'test_images':len(test),'exported_images':len(image_refs),'excluded':['.venv environment binaries','.cache downloaded backbone and HF caches','.tmp','unsloth_compiled_cache','__pycache__'],'note':'All run directories including retained checkpoints copied. Upstream package source subset and licenses copied; dependencies are pinned in per-run lock files. Original README and absolute paths retained as historical evidence.'})
print(json.dumps(read(ROOT/'snapshot.json'),ensure_ascii=False,indent=2))
