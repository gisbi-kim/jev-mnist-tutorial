from pathlib import Path
import json, hashlib, re, collections, math, sys, struct
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def jsdata(p):return json.loads(p.read_text(encoding='utf-8').split('=',1)[1].strip().removesuffix(';'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(4*1024*1024):h.update(b)
    return h.hexdigest()
sys.dont_write_bytecode=True
errors=[];batches=samples=0
for run in (ROOT/'raw/runs').iterdir():
    records=[]
    if (run/'batches.jsonl').exists():records=[json.loads(x) for x in (run/'batches.jsonl').read_text(encoding='utf-8').splitlines()]
    export=jsdata(ROOT/'site/data/runs'/f'{run.name}.js')
    assert len(records)==export['batch_count']
    for i,b in enumerate(records,1):
        e=jsdata(ROOT/'site/data/batches'/run.name/f'{i}.js')
        for k,v in b.items():
            if k!='samples':assert e[k]==v,(run.name,i,k)
        assert b['batch_size']==len(b['samples'])
        for source,derived in zip(b['samples'],e['samples']):
            for k,v in source.items():assert derived[k]==v,(run.name,i,k)
            assert (ROOT/'site/images'/source['source']/f"{source['image_id']}.png").exists()
            assert math.isclose(sum(source['probabilities'].values()),1,abs_tol=1e-5)
            if run.name=='20261030-010019':assert not re.search('[가-힣]',source['model_input']['state']+json.dumps(source['model_input']['questions'],ensure_ascii=False))
        batches+=1;samples+=len(b['samples'])
assert (batches,samples)==(770,20892)
test=jsdata(ROOT/'site/data/test.js');records=test['images'];assert len(records)==1000 and [r['image_id'] for r in records]==list(range(1000))
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'raw/code'));from core import oracle_action,rollout_summary
wrong=failed=0;reasons=collections.Counter();cells=[[{'total':0,'errors':0,'episode_failures':0} for _ in range(10)] for _ in range(10)]
for r in records:
    assert (ROOT/'site/images/test'/f"{r['image_id']}.png").exists()
    for s in range(10):
        e=rollout_summary(r['target'],s,r['predictions'].__getitem__)
        assert e==r['episodes'][s],(r['image_id'],s)
        error=r['predictions'][s]!=oracle_action(r['target'],s)
        wrong+=error;failed+=not e['success']
        if not e['success']:reasons[e['reason']]+=1
        c=cells[r['target']][s];c['total']+=1;c['errors']+=error;c['episode_failures']+=not e['success']
        assert math.isclose(sum(r['probabilities'][s].values()),1,abs_tol=1e-5)
assert wrong==230 and failed==638 and reasons=={'wrong_stop':434,'cycle':203,'out_of_range':1}
for t in range(10):
    for s in range(10):
        for k,v in cells[t][s].items():assert test['summary']['cells'][t][s][k]==v
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=set()
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:
            assert a['id'] not in self.ids,('duplicate id',a['id']);self.ids.add(a['id'])
        for k in ['src','href']:
            if k in a:self.links.append(a[k])
parser=Links();parser.feed((ROOT/'site/index.html').read_text(encoding='utf-8'))
for link in parser.links:
    u=urlsplit(link)
    if u.scheme:continue
    if u.path:
        p=(ROOT/'site'/unquote(u.path)).resolve()
        if p.name not in ('manifest.json',) and not p.exists():errors.append('missing link '+link)
    elif u.fragment and u.fragment not in parser.ids:errors.append('missing anchor '+link)
assert not errors,errors
params={}
for name in ['adapter_model.safetensors','joint_head.safetensors']:
    with (ROOT/'raw/runs/20261030-010019/adapter'/name).open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
    tensors=[v for k,v in header.items() if k!='__metadata__']
    params[name]={'tensors':len(tensors),'parameters':sum(math.prod(v['shape']) for v in tensors),'dtypes':dict(collections.Counter(v['dtype'] for v in tensors))}
result={'passed':True,'recorded_batches':batches,'sample_exposures':samples,'test_images':len(records),'test_decisions':10000,'recomputed_action_errors':wrong,'recomputed_episode_failures':failed,'recomputed_failure_reasons':dict(reasons),'heatmap_cells_verified':100,'training_exports_equal_raw':True,'english_main_inputs_verified':True,'static_links_verified':len(parser.links),'saved_parameter_inventory':params}
(ROOT/'docs/data-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
