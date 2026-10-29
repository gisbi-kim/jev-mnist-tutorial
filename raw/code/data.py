"""Download verified original MNIST; split IMAGES before expanding decision rows."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import struct
import urllib.request
import numpy as np
from PIL import Image
from core import oracle_action, questions, state_text

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
FILES = {
    'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
    'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432',
    't10k-images-idx3-ubyte.gz': '9fb629c4189551a2d022fa330f9573f3',
    't10k-labels-idx1-ubyte.gz': 'ec29112dd5afa0611ce80d1b7f02629c',
}


def download_file(name: str, checksum: str) -> Path:
    target = DATA / 'raw' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and hashlib.md5(target.read_bytes()).hexdigest() == checksum:
        return target
    url = 'https://ossci-datasets.s3.amazonaws.com/mnist/' + name
    print('Downloading', url, flush=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    if hashlib.md5(content).hexdigest() != checksum:
        raise RuntimeError(f'MNIST checksum mismatch: {name}')
    temporary = target.with_suffix('.partial')
    temporary.write_bytes(content)
    temporary.replace(target)
    return target


def read_idx(path: Path) -> np.ndarray:
    raw = gzip.decompress(path.read_bytes())
    magic, count = struct.unpack('>II', raw[:8])
    if magic == 2051:
        height, width = struct.unpack('>II', raw[8:16])
        return np.frombuffer(raw[16:], dtype=np.uint8).reshape(count, height, width).copy()
    if magic == 2049:
        return np.frombuffer(raw[8:], dtype=np.uint8).reshape(count).copy()
    raise ValueError('Invalid IDX magic')


def split_ids(labels: np.ndarray, seed: int = 3407) -> dict:
    rng = random.Random(seed)
    train, validation = [], []
    for digit in range(10):
        ids = np.flatnonzero(labels == digit).tolist()
        rng.shuffle(ids)
        validation.extend(ids[:500])
        train.extend(ids[500:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return {'train': train, 'validation': validation}


def load_arrays():
    with np.load(DATA / 'mnist.npz', allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def display_image(pixels: np.ndarray) -> Image.Image:
    return Image.fromarray(pixels.astype(np.uint8)).convert('RGB').resize((112, 112), Image.Resampling.NEAREST)


def decision_row(image, current: int, target: int, variant: int) -> dict:
    return {'state': state_text(current, variant), 'images': [image],
            'questions': questions(), 'gold': {'move': oracle_action(target, current)}}


def prepare(seed: int = 3407) -> dict:
    paths = {name: download_file(name, checksum) for name, checksum in FILES.items()}
    arrays = {'train_images': read_idx(paths['train-images-idx3-ubyte.gz']),
              'train_labels': read_idx(paths['train-labels-idx1-ubyte.gz']),
              'test_images': read_idx(paths['t10k-images-idx3-ubyte.gz']),
              'test_labels': read_idx(paths['t10k-labels-idx1-ubyte.gz'])}
    assert arrays['train_images'].shape == (60000, 28, 28)
    assert arrays['test_images'].shape == (10000, 28, 28)
    np.savez_compressed(DATA / 'mnist.npz', **arrays)
    splits = {**split_ids(arrays['train_labels'], seed), 'test': list(range(10000))}
    (DATA / 'splits.json').write_text(json.dumps(splits), encoding='utf-8')
    # Manifest rows reference raw image IDs; target stays metadata/gold only.
    for split, ids in splits.items():
        prefix = 'test' if split == 'test' else 'train'
        with (DATA / f'{split}.jsonl').open('w', encoding='utf-8') as stream:
            for image_id in ids:
                target = int(arrays[f'{prefix}_labels'][image_id])
                for current in range(10):
                    variant = (image_id + current) % 4
                    stream.write(json.dumps({'image_id': image_id, 'source': prefix,
                        'current': current, 'state': state_text(current, variant),
                        'questions': questions(), 'gold': {'move': oracle_action(target, current)}},
                        ensure_ascii=False) + '\n')
    summary = {'source': 'Original MNIST via PyTorch OSSCI mirror', 'seed': seed,
        'images': {k: len(v) for k, v in splits.items()},
        'decision_rows': {k: len(v)*10 for k, v in splits.items()},
        'split_rule': 'Image-disjoint: 500 per digit held out from original train; official test untouched.',
        'note': 'Language variants increase rows, not independent handwriting diversity.'}
    (DATA / 'manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=3407)
    print(json.dumps(prepare(parser.parse_args().seed), indent=2))
