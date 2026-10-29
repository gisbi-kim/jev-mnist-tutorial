"""Losslessly restore archived JSONL datasets; Python standard library only."""
from pathlib import Path
import gzip, hashlib, json, shutil

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()

def restore():
    entries = json.loads((ROOT / 'docs/publication-inventory.json').read_text(encoding='utf8'))['losslessly_compressed']
    for entry in entries:
        target = ROOT / entry['original_path']
        if target.exists():
            if digest(target) != entry['original_sha256']:
                raise RuntimeError(f'Refusing to overwrite different data: {target}')
        else:
            temp = target.with_suffix('.restoring')
            with gzip.open(ROOT / entry['path'], 'rb') as source, temp.open('xb') as dest:
                shutil.copyfileobj(source, dest)
            if digest(temp) != entry['original_sha256']:
                raise RuntimeError(f'Checksum mismatch: {temp}')
            temp.rename(target)
        print(f'Verified: {entry["original_path"]}')

if __name__ == '__main__':
    restore()
