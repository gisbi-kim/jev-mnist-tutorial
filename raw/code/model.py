"""Actual multimodal Unsloth decision model. No oracle fallback."""
import inspect
import math
import os
from pathlib import Path
from core import ACTIONS, questions


class DecisionEngine:
    def __init__(self, model_name: str, train: bool = False):
        os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parent / '.cache' / 'huggingface'))
        from unsloth import FastDecisionModel
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA PyTorch is required for this experiment.')
        if 'images' not in inspect.signature(FastDecisionModel.predict).parameters:
            raise RuntimeError('Installed Unsloth lacks multimodal decision predict(images=...). Update Unsloth.')
        self.api = FastDecisionModel
        self.model, self.processor = FastDecisionModel.from_pretrained(
            model_name=model_name, max_seq_length=1024, load_in_4bit=True,
            dtype=torch.bfloat16, random_state=3407,
            use_gradient_checkpointing='unsloth')
        if not hasattr(self.processor, 'image_processor'):
            raise RuntimeError('A multimodal processor is required; refusing a text-only model.')
        if train:
            self.model = FastDecisionModel.get_peft_model(self.model, r=16, lora_alpha=16,
                lora_dropout=0, random_state=3407, use_gradient_checkpointing='unsloth')
        else:
            FastDecisionModel.for_inference(self.model)

    def predict(self, image, instruction: str) -> dict:
        result = self.api.predict(self.model, self.processor, instruction, questions(), images=[image])['move']
        probabilities = {a: float(result['probabilities'][a]) for a in ACTIONS}
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()):
            raise RuntimeError('Model returned invalid probabilities')
        # Unsloth rounds each returned probability to four decimal places.
        # Allow that rounding budget plus floating-point representation error.
        if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0,
                            abs_tol=len(ACTIONS) * 0.5e-4 + 1e-12):
            raise RuntimeError('Model probabilities do not sum to one')
        action = str(result['answer'])
        if action not in ACTIONS:
            raise RuntimeError('Model returned an unknown action')
        return {'action': action, 'probabilities': probabilities}

    def build(self, rows):
        items, report = self.api.build_dataset(rows, self.processor, self.model)
        if not items or report['skipped'] or report.get('truncated'):
            raise RuntimeError(f'Dataset encoding must be lossless: {report}')
        return items

    def save(self, folder: Path):
        self.model.save_pretrained(str(folder))
