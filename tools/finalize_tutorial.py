"""Finalize copied evidence and create checksums; never edits original experiment."""
from pathlib import Path
import shutil,json,hashlib,datetime
ROOT=Path(r'C:\Users\USER\Documents\Codex\2026-10-29\e\outputs\jev-mnist-tutorial')
SRC=Path(r'C:\Users\USER\Downloads\mnist-decision-lab')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
tools=ROOT/'tools';tools.mkdir(exist_ok=True)
for n in ['package_tutorial.py','verify_tutorial.py','finalize_tutorial.py']:shutil.copy2(Path(__file__).parent/n,tools/n)
cache=ROOT/'raw/code/__pycache__'
if cache.is_dir():
 for f in cache.glob('*.pyc'):f.unlink()
 if not list(cache.iterdir()):cache.rmdir()
# The API snapshot requests append access logs; take the final session-log snapshot now.
for f in (ROOT/'raw/session_logs').iterdir():
 if (SRC/f.name).is_file():shutil.copy2(SRC/f.name,f)
verified=[]
for folder in ['runs','data']:
 for source in sorted((SRC/folder).rglob('*')):
  if not source.is_file() or '__pycache__' in source.parts:continue
  dest=ROOT/'raw'/source.relative_to(SRC)
  assert dest.exists() and sha(source)==sha(dest),str(source)
  verified.append(source.relative_to(SRC).as_posix())
for dest in (ROOT/'raw/code').rglob('*'):
 if dest.is_file():
  source=SRC/dest.relative_to(ROOT/'raw/code')
  assert source.exists() and sha(source)==sha(dest),str(dest)
  verified.append(source.relative_to(SRC).as_posix())
proof={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_files_verified_sha256':len(verified),'scope':'All runs, all data, all final code files; original source files remain unchanged. Session logs refreshed after read-only API snapshot requests.','paths':verified}
(ROOT/'docs/source-copy-verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf8')
files=[]
for p in sorted(ROOT.rglob('*')):
 if not p.is_file() or p.name in ('manifest.json','SHA256SUMS.txt') and p.parent==ROOT:continue
 files.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'file_count':len(files),'total_bytes':sum(x['bytes'] for x in files),'note':'Manifest and SHA256SUMS exclude themselves. SHA256SUMS additionally hashes manifest.json.'}
(ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
(ROOT/'SHA256SUMS.txt').write_text(''.join(f"{f['sha256']}  {f['path']}\n" for f in files)+sha(ROOT/'manifest.json')+'  manifest.json\n',encoding='utf8')
print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))
print('Source equality:',len(verified),'files')
