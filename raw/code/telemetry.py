"""Record the actual training forward pass, without additional model inference."""
import json
import time


class TraceCollator:
    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, items):
        batch = self.base(items)
        batch['_trace_rows'] = [int(item['row']) for item in items]
        # This experiment has one choice question per record.
        batch['_trace_options'] = [list(item['record'].questions[0].option_ids) for item in items]
        batch['_trace_inputs'] = [json.loads(json.dumps({
            'state': item['source']['state'], 'questions': item['source']['questions']
        }, ensure_ascii=False)) for item in items]
        return batch


def traced_trainer_class(base):
    class TracedTrainer(base):
        def enable_trace(self, run_dir, metadata):
            self.trace_path = run_dir / 'batches.jsonl'
            self.trace_metadata = metadata
            self.microbatch_number = 0
            self.examples_seen = 0
            self.data_collator = TraceCollator(self.data_collator)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch
            row_ids = inputs.pop('_trace_rows', None)
            options = inputs.pop('_trace_options', None)
            model_inputs = inputs.pop('_trace_inputs', None)
            target = inputs['target'].detach().clone()
            mask = inputs['marker_mask'].detach().clone()
            started = time.perf_counter()
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True,
                                                 num_items_in_batch=num_items_in_batch)
            if model.training and row_ids is not None:
                logits = outputs['logits'].detach().float().masked_fill(~mask, -1e4)
                probabilities = logits.softmax(-1).cpu().tolist()
                losses = (-(target * logits.log_softmax(-1)).sum(-1)).cpu().tolist()
                gold_indices = target.argmax(-1).cpu().tolist()
                self.microbatch_number += 1
                self.examples_seen += len(row_ids)
                samples = []
                for index, row_id in enumerate(row_ids):
                    keys = options[index]
                    probs = dict(zip(keys, probabilities[index][:len(keys)]))
                    gold = keys[gold_indices[index]]
                    meta = self.trace_metadata[row_id]
                    if gold != meta['gold']:
                        raise RuntimeError('Telemetry detected target/metadata misalignment')
                    samples.append({**meta, 'training_row': row_id, 'gold': gold,
                        'model_input': {**model_inputs[index], 'image': {
                            'source':meta['source'], 'image_id':meta['image_id'],
                            'preprocessing':'MNIST 28x28 → RGB 112x112 (nearest)'}},
                        'input_provenance':'captured_from_encoded_training_source',
                        'prediction': max(probs, key=probs.get), 'probabilities': probs,
                        'cross_entropy': losses[index]})
                event = {'microbatch': self.microbatch_number,
                    'optimizer_step': self.state.global_step + 1,
                    'examples_seen': self.examples_seen, 'batch_size': len(row_ids),
                    'loss': float(loss.detach().cpu()), 'samples': samples,
                    'time': time.time(), 'forward_ms': 1000*(time.perf_counter()-started),
                    'gpu_allocated_mib': torch.cuda.memory_allocated()/2**20,
                    'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20}
                with self.trace_path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            return (loss, outputs) if return_outputs else loss
    return TracedTrainer
