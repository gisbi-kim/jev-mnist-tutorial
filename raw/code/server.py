"""Loopback-only experiment UI. Inference fails explicitly until a model is trained."""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import secrets
import threading
import time
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps
from core import apply_action, parse_start, state_text
from data import DATA, ROOT, display_image, load_arrays

app = Flask(__name__, static_folder='web', static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 6 * 1024 * 1024
from training_web import training_web
app.register_blueprint(training_web)
from full_test_web import full_test_web
app.register_blueprint(full_test_web)
sessions = {}
session_lock = threading.Lock()
gpu_lock = threading.Lock()
engine = None
engine_path = None


def read_json(path, fallback=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else fallback


def load_engine():
    global engine, engine_path
    latest = read_json(ROOT / 'latest.json')
    if not latest or not Path(latest['adapter']).is_dir():
        raise RuntimeError('학습된 모델이 없어. 다운로드·학습을 완료한 뒤 다시 시도해줘.')
    if engine is None or engine_path != latest['adapter']:
        from model import DecisionEngine
        engine = DecisionEngine(latest['adapter'])
        engine_path = latest['adapter']
    return engine


def normalize_upload(raw: bytes):
    with Image.open(BytesIO(raw)) as source:
        if source.width * source.height > 16_000_000:
            raise ValueError('이미지는 1600만 픽셀 이하로 올려줘.')
        source = ImageOps.exif_transpose(source)
        rgba = source.convert('RGBA')
        background = Image.new('RGBA', rgba.size, 'white')
        background.alpha_composite(rgba)
        gray = background.convert('L')
    # Match MNIST: bright ink on black; detect background using corners.
    corners = [gray.getpixel(p) for p in [(0,0),(gray.width-1,0),(0,gray.height-1),(gray.width-1,gray.height-1)]]
    if sum(corners)/4 > 127:
        gray = ImageOps.invert(gray)
    mask = gray.point(lambda p: 255 if p > 35 else 0)
    box = mask.getbbox()
    if not box:
        raise ValueError('숫자가 비어 있어. 숫자 하나를 그리거나 업로드해줘.')
    crop = gray.crop(box)
    crop.thumbnail((20,20), Image.Resampling.LANCZOS)
    image = Image.new('L', (28,28))
    image.paste(crop, ((28-crop.width)//2, (28-crop.height)//2))
    return image.convert('RGB').resize((112,112), Image.Resampling.NEAREST)


def image_uri(image):
    stream = BytesIO()
    image.save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()


@app.get('/')
def home():
    return send_from_directory(ROOT / 'web', 'index.html')


@app.get('/api/status')
def status():
    latest = read_json(ROOT / 'latest.json')
    ready = bool(latest and Path(latest['adapter']).is_dir())
    return jsonify({'model_ready': ready, 'data_ready': (DATA / 'mnist.npz').is_file(),
        'status': read_json(ROOT / 'status.json', {'phase':'not_started'}),
        'results': read_json(ROOT / 'results.json'), 'model': 'Qwen3.5-0.8B + Clef decision head'})


@app.get('/api/sample')
def sample():
    if not (DATA / 'mnist.npz').is_file():
        return jsonify(error='MNIST가 아직 다운로드되지 않았어.'), 503
    arrays = load_arrays()
    i = secrets.randbelow(len(arrays['test_labels']))
    return jsonify(image=image_uri(display_image(arrays['test_images'][i])), sample_id=i)


@app.post('/api/start')
def start():
    payload = request.get_json()
    instruction = payload.get('instruction', '')
    current = parse_start(instruction)
    uri = payload.get('image', '')
    if not uri.startswith('data:image/') or ',' not in uri:
        raise ValueError('이미지를 먼저 선택해줘.')
    sample_id = payload.get('sample_id')
    if sample_id is not None:
        if type(sample_id) is not int or not 0 <= sample_id < 10000:
            raise ValueError('Invalid sample ID')
        arrays = load_arrays()
        image = display_image(arrays['test_images'][sample_id])
    else:
        raw = base64.b64decode(uri.split(',',1)[1], validate=True)
        image = normalize_upload(raw)
    token = secrets.token_urlsafe(24)
    with session_lock:
        # Expire idle sessions and bound memory. No files uploaded to external services.
        for key in list(sessions):
            if time.time() - sessions[key]['updated'] > 3600:
                del sessions[key]
        if len(sessions) >= 100:
            raise ValueError('열려 있는 실험이 너무 많아. 서버를 재시작해줘.')
        sessions[token] = {'image': image, 'current': current, 'instruction': instruction,
            'trace': [], 'stopped': False, 'visited': set(), 'updated': time.time()}
    return jsonify(session=token, current=current, image=image_uri(image))


@app.post('/api/step')
def step():
    payload = request.get_json()
    with gpu_lock:
        session = sessions.get(payload.get('session'))
        if session is None:
            raise ValueError('실험을 먼저 시작해줘.')
        if session['stopped']:
            raise ValueError('종료된 실험이야. 새로 시작해줘.')
        current = session['current']
        instruction = session['instruction'] if not session['trace'] else state_text(current, 1)
        predictor = load_engine()
        then = time.perf_counter()
        prediction = predictor.predict(session['image'], instruction)
        elapsed = (time.perf_counter()-then)*1000
        after, stopped = apply_action(current, prediction['action'])
        session['visited'].add(current)
        reason = 'model_stop' if stopped else None
        if not stopped and not 0 <= after <= 9:
            reason, stopped = 'out_of_range', True
        if not stopped and after in session['visited']:
            reason, stopped = 'cycle', True
        if not stopped and len(session['trace']) >= 19:
            reason, stopped = 'turn_limit', True
        row = {'turn':len(session['trace'])+1, 'before':current, 'after':after,
            **prediction, 'latency_ms':elapsed, 'stopped':stopped, 'reason':reason}
        session['trace'].append(row)
        session.update(current=after, stopped=stopped, updated=time.time())
        return jsonify(row)


@app.errorhandler(ValueError)
def bad_input(error):
    return jsonify(error=str(error)), 400


@app.errorhandler(RuntimeError)
def unavailable(error):
    return jsonify(error=str(error)), 503


@app.errorhandler(Exception)
def error(error):
    from werkzeug.exceptions import HTTPException
    if isinstance(error, HTTPException):
        return jsonify(error=error.description), error.code
    app.logger.exception('Request failed')
    return jsonify(error=f'{type(error).__name__}: {error}'), 500


@app.before_request
def local_only():
    # Avoid drive-by web requests to the local inference server.
    if request.host.split(':')[0] not in ('127.0.0.1', 'localhost'):
        return jsonify(error='Local access only'), 403
    origin = request.headers.get('Origin')
    if origin and origin != request.host_url.rstrip('/'):
        return jsonify(error='Cross-origin requests are not accepted'), 403


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7864)
    app.run(host='127.0.0.1', port=parser.parse_args().port, debug=False, threaded=True)
