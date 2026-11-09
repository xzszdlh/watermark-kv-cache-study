"""Offline regression checks for the server-pair adapter; never call an API."""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge


class ServerPairTests(unittest.TestCase):
    def make_judge(self, alias):
        obj = object.__new__(judge.Judge)
        obj.plan = judge.read_json(judge.HERE / 'protocol.json')
        obj.spec = obj.plan['models'][alias]
        obj.alias = alias
        obj.rubric = (judge.HERE / 'rubric.txt').read_text(encoding='utf-8')
        obj.identity = 'offline-test'
        obj.secret = 'synthetic-test-secret'
        obj.stop = threading.Event()
        obj.rate_lock = threading.Lock()
        obj.next_start = 0
        return obj

    def test_only_authorized_server_models_and_controls(self):
        item, _ = judge.synthetic_cases()[0]
        plan = judge.read_json(judge.HERE / 'protocol.json')
        self.assertEqual(set(plan['models']), {'qwen', 'deepseek'})
        for alias in plan['models']:
            obj = self.make_judge(alias)
            body = obj.body(item)
            self.assertEqual(obj.spec['execution_location'], 'server')
            self.assertEqual(obj.spec['endpoint'], 'https://example.invalid/v1/chat/completions')
            self.assertEqual(json.loads(body['messages'][1]['content']), item)
            self.assertEqual(body['temperature'], 0)
            self.assertEqual(body['max_tokens'], 8192)
            self.assertTrue(body['chat_template_kwargs']['enable_thinking'])
            if alias == 'qwen':
                self.assertEqual(body['reasoning_effort'], 'medium')
            else:
                self.assertNotIn('reasoning_effort', body)
                self.assertEqual(body['chat_template_kwargs']['reasoning_effort'], 50)
                self.assertEqual(body['thinking'], {'type': 'enabled'})

    def test_reasoning_metadata_without_hidden_text_or_secret(self):
        item, _ = judge.synthetic_cases()[0]
        verdict = {'item_id': item['item_id'], 'label': 'supported',
                   'evidence_quotes': ['three hikers were rescued near Pine Lake'],
                   'issue_span': '', 'rationale': 'The source states this.'}
        for alias in ('qwen', 'deepseek'):
            obj = self.make_judge(alias)
            payload = {'model': obj.spec['requested_model'],
                       'usage': {'completion_tokens': 100, 'completion_tokens_details': {'reasoning_tokens': 42}},
                       'choices': [{'finish_reason': 'stop', 'message': {
                           'content': json.dumps(verdict), 'reasoning': 'HIDDEN_REASONING_NOT_TO_SAVE'}}]}
            class Response:
                status_code = 200
                def json(self):
                    return payload
            with patch.object(judge.requests, 'post', return_value=Response()) as post:
                record = obj.request(item, 1)
            self.assertEqual(record['status'], 'valid')
            self.assertEqual(record['reasoning_observation']['reasoning_tokens'], 42)
            self.assertNotIn('HIDDEN_REASONING_NOT_TO_SAVE', json.dumps(record))
            self.assertNotIn(obj.secret, json.dumps(record))
            self.assertEqual(record['request_sha256'], judge.digest(post.call_args.kwargs['json']))

    def test_semantic_disagreement_not_retried_or_hidden(self):
        obj = self.make_judge('qwen')
        with tempfile.TemporaryDirectory() as folder:
            obj.directory = Path(folder)
            def record(item):
                return {'status': 'valid', 'returned_model_matches_request': True,
                        'verdict': {'label': 'contradicted' if item['item_id'] == 'P3' else dict(
                            (v['item_id'], label) for v, label in judge.synthetic_cases())[item['item_id']]},
                        'evidence_checks': {'quote_evidence_gate': True},
                        'reasoning_observation': {'reasoning_tokens': 10},
                        'requested_reasoning_effort': 'medium'}
            with patch.object(obj, 'preflight_record', side_effect=record):
                obj.preflight()
            checks = judge.read_json(obj.directory / 'preflight/checks.json')
            self.assertTrue(checks['passed'])
            self.assertEqual(checks['semantic_matches'], 3)
            self.assertFalse(checks['checks'][2]['semantic_match'])

    def test_returned_model_mismatch_stops_without_substitution(self):
        obj = self.make_judge('deepseek')
        item, _ = judge.synthetic_cases()[0]
        class Response:
            status_code = 200
            def json(self):
                return {'model': 'different-model', 'choices': [{'message': {'content': '{}'}}]}
        with patch.object(judge.requests, 'post', return_value=Response()):
            record = obj.request(item, 1)
        self.assertEqual(record['status'], 'returned_model_mismatch')
        self.assertTrue(obj.stop.is_set())
        self.assertFalse(record['retriable'])


if __name__ == '__main__':
    unittest.main()
