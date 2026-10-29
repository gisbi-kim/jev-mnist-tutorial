"""Copy into the workspace via prepare_workspace.py, then run on a CUDA machine."""
import os
os.environ.setdefault('UNSLOTH_COMPILE_DISABLE', '1')
os.environ.setdefault('UNSLOTH_DECISION_COMPILE', '0')
import argparse, json
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-id', type=int, default=0)
    parser.add_argument('--current', type=int, choices=range(10), default=3)
    args = parser.parse_args()
    if not 0 <= args.image_id < 10000:
        parser.error('--image-id must be 0..9999')
    from data import load_arrays, display_image
    from core import state_text
    from model import DecisionEngine
    checkpoint = Path(__file__).resolve().parent / 'runs/20261030-010019/adapter'
    engine = DecisionEngine(str(checkpoint), train=False)
    image = display_image(load_arrays()['test_images'][args.image_id])
    print(json.dumps(engine.predict(image, state_text(args.current, 0)), indent=2))
