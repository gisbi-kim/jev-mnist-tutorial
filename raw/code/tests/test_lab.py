import base64
import gzip
from io import BytesIO
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import ACTIONS, apply_action, oracle_action, parse_start, rollout_summary
from data import decision_row, read_idx, split_ids
import server


class TaskTests(unittest.TestCase):
    def test_new_instructions_are_english(self):
        from core import state_text, questions
        for current in range(10):
            for variant in range(6):
                self.assertTrue(state_text(current,variant).isascii())
        self.assertTrue(json.dumps(questions(),ensure_ascii=False).isascii())

    def test_all_100_rollouts(self):
        for target in range(10):
            for start in range(10):
                outcome = rollout_summary(target, start, lambda s: oracle_action(target,s))
                self.assertTrue(outcome['success'])
                self.assertEqual(len(outcome['trace']), abs(target-start)+1)
                self.assertEqual(outcome['extra_moves'], 0)

    def test_failures_are_not_successes(self):
        self.assertEqual(rollout_summary(8,2,lambda s:'0')['reason'], 'wrong_stop')
        self.assertEqual(rollout_summary(8,2,lambda s:'+1' if s==2 else '-1')['reason'], 'cycle')
        self.assertEqual(rollout_summary(8,0,lambda s:'-1')['reason'], 'out_of_range')
        self.assertEqual(rollout_summary(8,0,lambda s:'+1', max_turns=2)['reason'], 'turn_limit')

    def test_parser(self):
        for text, expected in [('2에서 시작해',2),('삼에서 시작해',3),('start from seven',7),('0',0),('nine',9)]:
            self.assertEqual(parse_start(text),expected)
        for bad in ['2에서 5로 가','2.5','.5','100','-1','아무 숫자','one two']:
            with self.assertRaises(ValueError): parse_start(bad)

    def test_no_target_in_model_inputs(self):
        image=Image.new('RGB',(28,28))
        first=decision_row(image,2,3,0)
        second=decision_row(image,2,7,0)
        self.assertEqual(first['state'],second['state'])
        self.assertEqual(first['questions'],second['questions'])
        self.assertEqual(set(first),{'state','images','questions','gold'})
        with self.assertRaises(ValueError): apply_action(3,'+2')

    def test_image_split_not_row_split(self):
        labels=np.repeat(np.arange(10),600)
        splits=split_ids(labels)
        self.assertFalse(set(splits['train']) & set(splits['validation']))
        self.assertEqual(len(splits['validation']),5000)
        self.assertEqual(set(splits['train']) | set(splits['validation']), set(range(6000)))
        self.assertEqual(splits,split_ids(labels))

    def test_idx_reader(self):
        # Synthetic binary fixture tests the parser; it is NOT MNIST training data.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            file=Path(tmp)/'fixture.gz'
            file.write_bytes(gzip.compress(struct.pack('>IIII',2051,1,2,2)+bytes([0,1,2,3])))
            self.assertEqual(read_idx(file).tolist(),[[[0,1],[2,3]]])
            file.write_bytes(gzip.compress(struct.pack('>II',2049,2)+bytes([3,8])))
            self.assertEqual(read_idx(file).tolist(),[3,8])


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client=server.app.test_client()
        self.headers={'Host':'localhost'}
        server.sessions.clear()
        image=Image.new('RGB',(28,28),'black')
        ImageDraw.Draw(image).line([(8,5),(20,5),(14,22)],fill='white',width=3)
        buf=BytesIO();image.save(buf,format='PNG')
        self.uri='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()

    def test_status_has_no_fabricated_metrics(self):
        with patch.object(server,'read_json',return_value=None):
            response=self.client.get('/api/status')
        self.assertFalse(response.json['model_ready'])
        self.assertIsNone(response.json['results'])

    def test_empty_image_rejected(self):
        buf=BytesIO();Image.new('RGB',(28,28),'white').save(buf,format='PNG')
        with self.assertRaises(ValueError): server.normalize_upload(buf.getvalue())

    def test_single_operation_and_stop(self):
        class Stub:
            def predict(self,image,instruction):
                self.last=(image,instruction)
                return {'action':'+1','probabilities':{'-1':.1,'0':.2,'+1':.7}}
        stub=Stub()
        start=self.client.post('/api/start',json={'image':self.uri,'instruction':'2에서 시작해'})
        self.assertEqual(start.status_code,200)
        token=start.json['session']
        with patch.object(server,'load_engine',return_value=stub):
            response=self.client.post('/api/step',json={'session':token})
        self.assertEqual(response.json['before'],2)
        self.assertEqual(response.json['after'],3)
        self.assertEqual(stub.last[1],'2에서 시작해')
        self.assertNotIn('target',stub.last[1])
        self.assertFalse(response.json['stopped'])

    def test_missing_model_does_not_use_oracle(self):
        start=self.client.post('/api/start',json={'image':self.uri,'instruction':'2에서 시작해'})
        with patch.object(server,'load_engine',side_effect=RuntimeError('No checkpoint')):
            response=self.client.post('/api/step',json={'session':start.json['session']})
        self.assertEqual(response.status_code,503)
        self.assertEqual(server.sessions[start.json['session']]['current'],2)

    def test_origin_and_host_protection(self):
        self.assertEqual(self.client.get('/api/status',headers={'Host':'evil.example'}).status_code,403)
        self.assertEqual(self.client.post('/api/start',json={},headers={'Origin':'https://evil.example'}).status_code,403)


if __name__=='__main__': unittest.main()
