"""Create a new inference/training directory without modifying archived results."""
from pathlib import Path
import argparse, shutil

ROOT = Path(__file__).resolve().parents[1]
FINAL = '20261030-010019'

def prepare(destination):
    destination = destination.resolve()
    if destination.exists():
        raise SystemExit('Choose a new directory; existing directories are never overwritten.')
    destination.mkdir(parents=True)
    shutil.copytree(ROOT / 'raw/code', destination, dirs_exist_ok=True)
    shutil.copytree(ROOT / 'raw/data', destination / 'data', ignore=shutil.ignore_patterns('*.jsonl', '*.jsonl.gz'))
    model_root = destination / 'runs' / FINAL
    shutil.copytree(ROOT / 'raw/runs' / FINAL / 'adapter', model_root / 'adapter')
    for filename in ['config.json', 'requirements.lock.txt', 'calibration.json']:
        source = ROOT / 'raw/runs' / FINAL / filename
        if source.exists():
            shutil.copy2(source, model_root / filename)
    shutil.copy2(ROOT / 'tools/predict_saved.py', destination / 'predict_saved.py')
    print(f'Workspace ready: {destination}')
    print('No old full_test directory copied: a new evaluation will not reuse archived predictions.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    prepare(parser.parse_args().destination)
