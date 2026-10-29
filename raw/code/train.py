"""Train and measure a real decision head. Network + CUDA dependencies required."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import time
from core import ACTIONS, oracle_action, rollout_summary, state_text
from data import DATA, ROOT, decision_row, display_image, load_arrays


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    temporary.replace(path)


def evaluate(engine, arrays, ids, variant=0, ablation=None, source='test'):
    """Compute all states once per image, then simulate deterministic rollouts."""
    confusion = [[0]*3 for _ in ACTIONS]
    outcomes, latencies, nll, brier, confidence, correct = [], [], [], [], [], []
    examples = []
    for index, image_id in enumerate(ids):
        target = int(arrays[f'{source}_labels'][image_id])
        image_index = ids[(index + 1) % len(ids)] if ablation == 'shuffled' else image_id
        pixels = arrays[f'{source}_images'][image_index]
        if ablation == 'blank':
            pixels = pixels * 0
        image, cache = display_image(pixels), {}
        for current in range(10):
            then = time.perf_counter()
            prediction = engine.predict(image, state_text(current, variant))
            latencies.append(time.perf_counter() - then)
            action, probabilities = prediction['action'], prediction['probabilities']
            expected = oracle_action(target, current)
            cache[current] = action
            confusion[ACTIONS.index(expected)][ACTIONS.index(action)] += 1
            import math
            nll.append(-math.log(max(probabilities[expected], 1e-12)))
            brier.append(sum((probabilities[a] - int(a == expected))**2 for a in ACTIONS))
            confidence.append(max(probabilities.values()))
            correct.append(action == expected)
        for start in range(10):
            outcome = rollout_summary(target, start, cache.__getitem__)
            outcomes.append(outcome)
            if len(examples) < 20 or (not outcome['success'] and len(examples) < 40):
                examples.append({'test_image_id': image_id, 'target': target, 'start': start, **outcome})
        if (index + 1) % 10 == 0:
            print(f'Evaluation {index+1}/{len(ids)} images', flush=True)
    count = len(correct)
    recalls = [confusion[i][i] / sum(confusion[i]) if sum(confusion[i]) else None for i in range(3)]
    ece = 0.
    for b in range(10):
        selected = [i for i,p in enumerate(confidence) if b/10 <= p <= 1 and (p < (b+1)/10 or b == 9)]
        if selected:
            ece += len(selected)/count * abs(sum(correct[i] for i in selected)/len(selected) - sum(confidence[i] for i in selected)/len(selected))
    successes = [o for o in outcomes if o['success']]
    return {'image_count': len(ids), 'decision_count': count, 'action_accuracy': sum(correct)/count,
        'macro_recall': sum(r for r in recalls if r is not None)/sum(r is not None for r in recalls),
        'per_action_recall': dict(zip(ACTIONS, recalls)), 'confusion_true_rows_pred_columns': confusion,
        'action_order': ACTIONS, 'episode_success_rate': len(successes)/len(outcomes),
        'wrong_stop_rate': sum(o['reason']=='wrong_stop' for o in outcomes)/len(outcomes),
        'failure_reasons': dict(Counter(o['reason'] for o in outcomes if not o['success'])),
        'mean_extra_moves_success_only': sum(o['extra_moves'] for o in successes)/len(successes) if successes else None,
        'mean_latency_ms': 1000*sum(latencies)/len(latencies), 'nll': sum(nll)/count,
        'brier': sum(brier)/count, 'ece_10_bins': ece, 'examples': examples,
        'variant': variant, 'ablation': ablation, 'source': source,
        'note': 'Deterministic inference; per-image state predictions cached for exact rollouts. No test labels passed to the model.'}


def run(args):
    # Unsloth must import before transformers/torch.
    from model import DecisionEngine
    engine = DecisionEngine(args.model, train=True)
    import torch
    from unsloth import DecisionTrainer
    from transformers import TrainingArguments, TrainerCallback
    arrays = load_arrays()
    splits = json.loads((DATA / 'splits.json').read_text(encoding='utf-8'))
    train_ids = splits['train'][:args.train_images]
    val_ids = splits['validation'][:args.val_images]
    calibration_ids = splits['validation'][args.val_images:args.val_images + 50]
    if len(train_ids) != args.train_images or len(val_ids) != args.val_images or len(calibration_ids) != 50:
        raise ValueError('Requested subset exceeds available split')
    assert not set(train_ids) & set(val_ids)
    rng = random.Random(3407)
    rows = []
    row_metadata = {}
    for image_id in train_ids:
        image = display_image(arrays['train_images'][image_id])
        target = int(arrays['train_labels'][image_id])
        for current in range(10):
            row = decision_row(image, current, target, (image_id+current) % 4)
            rows.append(row)
            row_metadata[id(row)] = {'image_id':image_id,'source':'train','current':current,
                'state':row['state'],'gold':row['gold']['move'],'target_digit_metadata':target}
    # Balance operation classes without introducing test/validation images.
    groups = {a: [r for r in rows if r['gold']['move']==a] for a in ACTIONS}
    if any(not group for group in groups.values()):
        raise ValueError('Training subset must contain all action classes')
    size = max(map(len, groups.values()))
    balanced = [row for a in ACTIONS for row in
                (groups[a] + rng.choices(groups[a], k=size-len(groups[a])))]
    rng.shuffle(balanced)
    val_rows = [decision_row(display_image(arrays['train_images'][i]), s,
        int(arrays['train_labels'][i]), (i+s) % 4) for i in calibration_ids for s in range(10)]
    items, val_items = engine.build(balanced), engine.build(val_rows)
    run_dir = ROOT / 'runs' / datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir.mkdir(parents=True)
    def status(phase, **extra):
        value = {'phase': phase, 'run': run_dir.name,
            'updated_at': datetime.now(timezone.utc).isoformat(), **extra}
        dump(ROOT / 'status.json', value)
        dump(run_dir / 'live_status.json', value)
    dump(run_dir / 'config.json', {**vars(args), 'input_language':'en', 'train_image_ids': train_ids,
        'validation_image_ids': val_ids, 'calibration_image_ids': calibration_ids,
        'balanced_training_rows': len(balanced),
        'compile_disabled': __import__('os').environ.get('UNSLOTH_COMPILE_DISABLE'),
        'versions': {p: importlib.metadata.version(p) for p in ['unsloth','torch','transformers','peft']}})
    test_ids = random.Random(3407).sample(splits['test'], args.test_images)
    status('baseline_evaluation')
    engine.api.for_inference(engine.model)
    before = evaluate(engine, arrays, test_ids[:min(20, len(test_ids))])
    before['label'] = 'Untrained decision head on pretrained VLM (not a zero-shot VLM benchmark)'
    dump(run_dir / 'before.json', before)
    validation_history = []
    training_peak = {'allocated_bytes': 0, 'reserved_bytes': 0}
    def record_memory():
        training_peak['allocated_bytes'] = max(training_peak['allocated_bytes'], torch.cuda.max_memory_allocated())
        training_peak['reserved_bytes'] = max(training_peak['reserved_bytes'], torch.cuda.max_memory_reserved())
    def validate(step, ids):
        status('validation', step=step, max_steps=args.steps)
        engine.api.for_inference(engine.model)
        metrics = evaluate(engine, arrays, ids, source='train')
        validation_history.append({'step': step, **metrics})
        dump(run_dir / 'validation_history.json', validation_history)
        print('VALIDATION ' + json.dumps({k: validation_history[-1][k] for k in
              ['step','image_count','action_accuracy','macro_recall','episode_success_rate','per_action_recall']}), flush=True)
        engine.api.for_training(engine.model)
        return metrics
    validate(0, val_ids[:min(30, len(val_ids))])
    class Progress(TrainerCallback):
        def on_log(self, arguments, state, control, logs=None, **kwargs):
            status('training', step=state.global_step, max_steps=state.max_steps, logs=logs)
            dump(run_dir / 'training_history.json', state.log_history)
        def on_step_end(self, arguments, state, control, **kwargs):
            record_memory()
            if state.global_step % args.validation_steps == 0:
                validate(state.global_step, val_ids[:min(30, len(val_ids))])
                dump(run_dir / 'gpu_memory.json', training_peak)
                torch.cuda.reset_peak_memory_stats()
                status('training', step=state.global_step, max_steps=state.max_steps)
    engine.api.for_training(engine.model)
    if args.trace_batches:
        from telemetry import traced_trainer_class
        Trainer = traced_trainer_class(DecisionTrainer)
    else:
        Trainer = DecisionTrainer
    trainer = Trainer(model=engine.model, processing_class=engine.processor, head_learning_rate=args.head_lr,
        train_dataset=items, eval_dataset=val_items, callbacks=[Progress()],
        args=TrainingArguments(output_dir=str(run_dir / 'checkpoints'),
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.grad_accum, max_steps=args.steps, learning_rate=args.learning_rate,
            warmup_steps=min(30, args.steps//10), lr_scheduler_type='cosine',
            weight_decay=.01, bf16=True, fp16=False, logging_steps=5,
            eval_strategy='no', save_strategy='steps', save_steps=100,
            save_total_limit=2, report_to='none', seed=3407, dataloader_num_workers=0))
    if args.trace_batches:
        trainer.enable_trace(run_dir, [row_metadata[id(row)] for row in balanced])
    status('training', step=0, max_steps=args.steps)
    torch.cuda.reset_peak_memory_stats()
    stats = trainer.train()
    record_memory()
    dump(run_dir / 'gpu_memory.json', training_peak)
    dump(run_dir / 'train_metrics.json', stats.metrics)
    validation = validate(args.steps, val_ids)
    status('calibrating')
    engine.api.for_inference(engine.model)
    calibration = engine.api.calibrate(engine.model, engine.processor, val_items, batch_size=1)
    engine.save(run_dir / 'adapter')
    dump(run_dir / 'calibration.json', calibration)
    status('test_evaluation')
    after = evaluate(engine, arrays, test_ids)
    dump(run_dir / 'after.json', after)
    # Same-image subset for a fair before/after comparison.
    matched_after = evaluate(engine, arrays, test_ids[:min(20, len(test_ids))])
    unseen = evaluate(engine, arrays, test_ids[:min(20, len(test_ids))], variant=4)
    shuffled = evaluate(engine, arrays, test_ids[:min(20, len(test_ids))], ablation='shuffled')
    blank = evaluate(engine, arrays, test_ids[:min(20, len(test_ids))], ablation='blank')
    report = {'run': run_dir.name, 'model': args.model, 'trained': True, 'before': before,
        'after': after, 'matched_after': matched_after, 'unseen_language': unseen,
        'shuffled_image': shuffled, 'blank_image': blank, 'calibration': calibration,
        'validation': validation, 'validation_history': validation_history,
        'gpu_training_peak': training_peak}
    dump(run_dir / 'report.json', report)
    dump(ROOT / 'latest.json', {'adapter': str(run_dir / 'adapter'), 'report': str(run_dir / 'report.json')})
    dump(ROOT / 'results.json', report)
    (run_dir / 'requirements.lock.txt').write_text(subprocess.check_output(
        [__import__('sys').executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    status('complete', report=str(run_dir / 'report.json'))
    print(json.dumps({k: after[k] for k in ['action_accuracy','episode_success_rate','wrong_stop_rate']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='unsloth/Qwen3.5-0.8B')
    parser.add_argument('--train-images', type=int, default=1200)
    parser.add_argument('--val-images', type=int, default=100)
    parser.add_argument('--test-images', type=int, default=100)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--validation-steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--trace-batches', action='store_true')
    args = parser.parse_args()
    if min(args.train_images, args.val_images, args.test_images, args.steps,
           args.batch_size, args.grad_accum, args.validation_steps) <= 0:
        parser.error('Counts and steps must be positive')
    try:
        run(args)
    except Exception as error:
        dump(ROOT / 'status.json', {'phase': 'failed', 'error': str(error),
            'updated_at': datetime.now(timezone.utc).isoformat()})
        raise
