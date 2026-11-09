"""Portable inference for the saved JEV Clef model on Apple Silicon or CPU."""
import json, math, os
from pathlib import Path
import torch
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoProcessor
from clef_reference import JointSchemaHead, ClefModel, encode_record, collate_records
from core import ACTIONS, questions

class DecisionEngine:
    def __init__(self, model_name, train=False):
        if train: raise RuntimeError('Portable inference does not support training.')
        path = Path(model_name)
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.dtype = torch.float16 if self.device.type == 'mps' else torch.float32
        self.config = json.loads((path/'unsloth_decision_config.json').read_text())
        base = AutoModelForImageTextToText.from_pretrained(self.config['base_model'], dtype=self.dtype, attn_implementation='eager')
        self.encoder = PeftModel.from_pretrained(base, str(path), is_trainable=False)
        saved = load_file(str(path/'adapter_model.safetensors'))
        actual = get_peft_model_state_dict(self.encoder)
        if set(saved) != set(actual) or any(not torch.equal(saved[k].to(actual[k].dtype), actual[k].cpu()) for k in saved):
            raise RuntimeError('Loaded adapter differs from saved weights')
        head = JointSchemaHead(**json.loads((path/'joint_head_config.json').read_text()))
        head.load_state_dict(load_file(str(path/'joint_head.safetensors')), strict=True)
        self.model = ClefModel(self.encoder, head).to(self.device, dtype=self.dtype).eval()
        self.processor = AutoProcessor.from_pretrained(str(path))
        self.temperature = float(self.config.get('temperature', [1.0])[0])

    @torch.inference_mode()
    def predict(self, image, instruction):
        record = encode_record(self.processor.tokenizer, {'state': instruction, 'images': [image.convert('RGB')], 'questions': questions()}, max_length=16384, processor=self.processor)
        if len(record.input_ids) > int(self.config.get('max_len', 1024)): raise ValueError('Input exceeds model context length')
        batch = collate_records([record], self.processor.tokenizer.pad_token_id, self.device)
        for key, value in (batch.get('media') or {}).items():
            if torch.is_tensor(value) and value.is_floating_point(): batch['media'][key] = value.to(self.dtype)
        raw = self.model(batch)[0][0].float()
        mapping = dict(zip(record.questions[0].option_ids, raw))
        probs = (torch.stack([mapping[a] for a in ACTIONS])/self.temperature).softmax(-1).cpu().tolist()
        if not all(math.isfinite(p) for p in probs): raise RuntimeError('Invalid probabilities')
        return {'action': ACTIONS[max(range(len(ACTIONS)), key=lambda i: probs[i])], 'probabilities': dict(zip(ACTIONS, probs))}
