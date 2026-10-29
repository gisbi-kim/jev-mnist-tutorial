"""Regenerate integrity records for tracked/public source files before committing."""
from pathlib import Path
import datetime, hashlib, json

ROOT = Path(__file__).resolve().parents[1]
IGNORE = {'.git', '_site', '__pycache__', '.venv', '.cache', 'workspaces'}
files = []
for p in sorted(ROOT.rglob('*')):
    if not p.is_file():
        continue
    q = p.relative_to(ROOT)
    if set(q.parts) & IGNORE or q.as_posix() in {'manifest.json','SHA256SUMS.txt'}:
        continue
    if q.parent.as_posix() == 'raw/data' and p.suffix == '.jsonl':
        continue
    h = hashlib.sha256()
    with p.open('rb') as stream:
        while block := stream.read(4*1024*1024):
            h.update(block)
    files.append({'path': q.as_posix(), 'bytes': p.stat().st_size, 'sha256': h.hexdigest()})
manifest = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'scope': 'GitHub publication: final model only, historical logs preserved', 'files': files, 'file_count': len(files), 'total_bytes': sum(f['bytes'] for f in files), 'note': 'Manifest and SHA256SUMS exclude themselves. SHA256SUMS additionally hashes manifest.json.'}
(ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
digest=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()
(ROOT/'SHA256SUMS.txt').write_text(''.join(f"{f['sha256']}  {f['path']}\n" for f in files)+digest+'  manifest.json\n',encoding='utf8')
print(f"{len(files)} files, {manifest['total_bytes']:,} bytes")
