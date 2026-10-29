"""Validate distributed files, compressed data and every dynamic run-file link."""
from pathlib import Path
import gzip, hashlib, json, re

ROOT = Path(__file__).resolve().parents[1]

def digest(stream):
    h = hashlib.sha256()
    while block := stream.read(4 * 1024 * 1024):
        h.update(block)
    return h.hexdigest()

def main():
    inventory = json.loads((ROOT / 'docs/publication-inventory.json').read_text(encoding='utf8'))
    for item in inventory['losslessly_compressed']:
        with gzip.open(ROOT / item['path'], 'rb') as stream:
            assert digest(stream) == item['original_sha256'], item['path']
    dynamic_links = 0
    for p in (ROOT / 'site/data/runs').glob('*.js'):
        run = json.loads(p.read_text(encoding='utf8').split('=', 1)[1].strip().removesuffix(';'))
        for entry in run['files']:
            path = ROOT / 'raw/runs' / run['run'] / entry['path']
            assert path.is_file() and path.stat().st_size == entry['bytes'], path
            dynamic_links += 1
    weights = list((ROOT / 'raw/runs').rglob('*.safetensors'))
    assert len(weights) == 2 and all('20261030-010019/adapter/' in p.as_posix() for p in weights)
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf8'))
    for item in manifest['files']:
        p = ROOT / item['path']
        assert p.stat().st_size == item['bytes'], p
        assert p.stat().st_size < 100 * 1024**2, p
        with p.open('rb') as stream:
            assert digest(stream) == item['sha256'], p
    credentials = re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9_-]{30,}|AIza[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
    for item in manifest['files']:
        p = ROOT / item['path']
        if p.suffix.lower() in {'.json','.jsonl','.md','.txt','.log','.py','.js','.html','.css','.yml','.ps1','.cmd','.jinja'}:
            assert not credentials.search(p.read_text(encoding='utf8')), f'Credential-like content: {p}'
    print(json.dumps({'passed': True, 'manifest_files': len(manifest['files']), 'bytes': manifest['total_bytes'], 'dynamic_file_links': dynamic_links, 'model_weight_files': len(weights), 'gzip_restoration_verified': 3}, indent=2))

if __name__ == '__main__':
    main()
