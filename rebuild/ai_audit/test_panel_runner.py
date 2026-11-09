"""Offline tests for scheduling/Responses additions; no provider requests."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel_runner as panel


class PanelTests(unittest.TestCase):
    @unittest.skipUnless((panel.HERE / 'items.json').exists(),
                         'Requires local frozen audit items, not included in the repository')
    def test_all_formal_request_hashes_unchanged_for_tokenhub(self):
        panel.verify_request_parity()

    @unittest.skipIf(os.name == 'nt', 'Production locking path is Linux; run this check on the server')
    def test_shared_rolling_rate_limit_across_jobs_and_threads(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(panel.base, 'OUT', Path(tmp)):
            a = panel.SharedLimiter('tokenhub', 'qwen', 'offline')
            b = panel.SharedLimiter('tokenhub', 'deepseek', 'offline')
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = [executor.submit((a if i % 2 else b).acquire) for i in range(9)]
                starts = sorted(f.result() for f in futures)
            self.assertEqual(len(starts), 9)
            for start in starts:
                self.assertLessEqual(sum(start <= t < start + 1 for t in starts), 3)
            self.assertGreaterEqual(min(y - x for x, y in zip(starts, starts[1:])), .339)
            rows = [json.loads(line) for line in a.events.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows), 9)
            self.assertEqual({r['model_alias'] for r in rows}, {'qwen', 'deepseek'})

    def test_cooldown_is_shared(self):
        now = [1000.0]
        def sleep(seconds):
            now[0] += seconds
        with tempfile.TemporaryDirectory() as tmp, patch.object(panel.base, 'OUT', Path(tmp)):
            a = panel.SharedLimiter('tokenhub', 'qwen', 'offline', lambda: now[0], sleep)
            b = panel.SharedLimiter('tokenhub', 'deepseek', 'offline', lambda: now[0], sleep)
            a.cooldown(15)
            self.assertGreaterEqual(b.acquire(), 1015)

    def terra(self):
        obj = object.__new__(panel.PanelJudge)
        extension = panel.base.read_json(panel.HERE / 'terra_extension.json')
        obj.alias = 'terra'
        obj.spec = extension['model']
        obj.plan = panel.base.read_json(panel.HERE / 'protocol.json')
        obj.plan['request'] = extension['request']
        obj.rubric = (panel.HERE / 'rubric.txt').read_text(encoding='utf-8')
        obj.identity = 'offline'
        obj.operational_identity = 'offline-schedule'
        obj.secret = 'TEST_ONLY_CREDENTIAL'
        obj.stop = threading.Event()
        obj.pace = lambda: None
        return obj

    def test_terra_medium_response_and_privacy(self):
        obj = self.terra()
        item, _ = panel.base.synthetic_cases()[0]
        verdict = {'item_id': item['item_id'], 'label': 'supported',
                   'evidence_quotes': ['three hikers were rescued near Pine Lake'],
                   'issue_span': '', 'rationale': 'The source states this.'}
        class Response:
            status_code = 200
            def json(self):
                return {'model': 'gpt-6.1-terra', 'status': 'completed', 'reasoning': {'effort': 'medium'},
                        'usage': {'output_tokens_details': {'reasoning_tokens': 42}},
                        'output': [{'type': 'reasoning', 'content': 'HIDDEN_TEXT_MUST_NOT_APPEAR'},
                                   {'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(verdict)}]}]}
        with patch.object(panel.base.requests, 'post', return_value=Response()) as post:
            record = obj.request(item, 1)
        self.assertEqual(record['status'], 'valid')
        self.assertEqual(record['reasoning_observation']['reasoning_tokens'], 42)
        self.assertEqual(record['reasoning_observation']['returned_effort'], 'medium')
        self.assertEqual(obj.spec['concurrency'], 10)
        body = post.call_args.kwargs['json']
        self.assertEqual(body['reasoning'], {'effort': 'medium'})
        self.assertEqual(json.loads(body['input']), item)
        self.assertEqual(body['instructions'], obj.rubric)
        self.assertNotIn('temperature', body)
        self.assertNotIn(obj.secret, json.dumps(record))
        self.assertNotIn('HIDDEN_TEXT_MUST_NOT_APPEAR', json.dumps(record))
        self.assertEqual(record['operational_identity'], 'offline-schedule')

    def test_terra_wrong_effort_stops(self):
        obj = self.terra()
        item, _ = panel.base.synthetic_cases()[0]
        class Response:
            status_code = 200
            def json(self):
                return {'model': 'gpt-6.1-terra', 'status': 'completed', 'reasoning': {'effort': 'low'}, 'output': []}
        with patch.object(panel.base.requests, 'post', return_value=Response()):
            record = obj.request(item, 1)
        self.assertEqual(record['status'], 'reasoning_effort_not_medium')
        self.assertFalse(record['retriable'])
        self.assertTrue(obj.stop.is_set())


if __name__ == '__main__':
    unittest.main()
