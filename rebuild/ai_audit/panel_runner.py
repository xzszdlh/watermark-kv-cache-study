"""Operational three-judge scheduler. The frozen server-pair runner stays unchanged.
Secrets are process-only. Rate events contain no request text, headers or keys.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import platform
import threading
import time

import judge as base

HERE = base.HERE
ROOT = base.ROOT
LOG = base.LOG


def scheduling_plan():
    return base.read_json(HERE / 'scheduling_amendment.json')


def freeze_operation():
    paths = [HERE / name for name in ('panel_runner.py', 'scheduling_amendment.json',
                                    'terra_extension.json', 'terra_capability_checks.json',
                                    'judge.py', 'protocol.json', 'rubric.txt', 'items.json')]
    payload = {'plan': scheduling_plan(),
               'sources': {p.name: base.digest(p.read_bytes()) for p in paths}}
    identity = base.digest(payload)
    folder = base.OUT / 'scheduling'
    path = folder / 'manifest.json'
    if path.exists():
        assert base.read_json(path)['identity'] == identity, 'Operational identity changed'
    else:
        base.atomic_json(path, {'identity': identity, 'frozen_at': base.now(), **payload})
        for p in paths:
            dest = folder / 'source' / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
    return identity


class SharedLimiter:
    """One process-safe start queue per provider, including retries and cooldowns."""
    def __init__(self, provider, alias, operational_identity, clock=time.time, sleeper=time.sleep):
        self.provider = provider
        self.alias = alias
        self.operational_identity = operational_identity
        self.folder = base.OUT / 'scheduling'
        self.path = self.folder / (provider + '_rate_state.json')
        self.lock = self.folder / (provider + '_rate.lock')
        self.events = self.folder / (provider + '_rate_events.jsonl')
        self.thread_lock = threading.Lock()
        self.clock = clock
        self.sleeper = sleeper
        cfg = scheduling_plan()['tokenhub_rate_limit']
        self.spacing = cfg['minimum_start_spacing_seconds'] if provider == 'tokenhub' else 0
        self.window = cfg['window_seconds']
        self.capacity = cfg['maximum_starts_per_window'] if provider == 'tokenhub' else None

    def _load(self):
        return base.read_json(self.path) if self.path.exists() else {'starts': [], 'not_before': 0}

    def _event(self, event):
        self.events.parent.mkdir(parents=True, exist_ok=True)
        with self.events.open('a', encoding='utf-8') as file:
            file.write(json.dumps({'model_alias': self.alias,
                                   'operational_identity': self.operational_identity,
                                   **event}, ensure_ascii=False) + '\n')

    def acquire(self):
        while True:
            try:
                with self.thread_lock, base.exclusive_lock(self.lock):
                    current = self.clock()
                    state = self._load()
                    starts = [v for v in state.get('starts', []) if v > current - self.window]
                    eligible = max(current, state.get('not_before', 0))
                    if self.capacity:
                        if starts:
                            eligible = max(eligible, starts[-1] + self.spacing)
                        if len(starts) >= self.capacity:
                            eligible = max(eligible, starts[-self.capacity] + self.window + 0.005)
                    delay = eligible - current
                    if delay <= 0:
                        starts.append(current)
                        state['starts'] = starts
                        state['updated_at'] = current
                        base.atomic_json(self.path, state)
                        self._event({'event': 'request_start_admission', 'unix_time': current,
                                     'pid': os.getpid()})
                        return current
            except BlockingIOError:
                delay = 0.02
            self.sleeper(max(0.001, min(delay, 1)))

    def cooldown(self, seconds=15):
        while True:
            try:
                with self.thread_lock, base.exclusive_lock(self.lock):
                    current = self.clock()
                    state = self._load()
                    state['not_before'] = max(state.get('not_before', 0), current + seconds)
                    base.atomic_json(self.path, state)
                    self._event({'event': 'cooldown', 'unix_time': current, 'seconds': seconds})
                    return
            except BlockingIOError:
                self.sleeper(0.02)


def freeze_terra():
    extension = base.read_json(HERE / 'terra_extension.json')
    bundle = base.public_data()
    assert extension['dataset_id'] == bundle['dataset_id']
    inherited = deepcopy(base.read_json(HERE / 'protocol.json'))
    spec = deepcopy(extension['model'])
    inherited['models'] = {'terra': spec}
    inherited['request'] = extension['request']
    inherited['analysis'] = extension['panel']
    inherited['authorization'] = extension['authorization']
    payload = {'model_alias': 'terra', 'requested_model': spec['requested_model'],
               'dataset_id': bundle['dataset_id'], 'evaluation_type': inherited['evaluation_type'],
               'extension': extension, 'resolved_plan': inherited,
               'sources': {name: base.digest((HERE / name).read_bytes()) for name in (
                   'panel_runner.py', 'judge.py', 'protocol.json', 'terra_extension.json',
                   'scheduling_amendment.json', 'rubric.txt', 'items.json', 'terra_capability_checks.json')}}
    identity = base.digest(payload)
    directory = base.OUT / 'terra'
    path = directory / 'manifest.json'
    if path.exists():
        assert base.read_json(path)['identity'] == identity, 'Terra identity changed'
    else:
        base.atomic_json(path, {'identity': identity, 'frozen_at': base.now(), **payload})
        for name in payload['sources']:
            p = directory / 'source' / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((HERE / name).read_bytes())
    return inherited, spec, bundle, identity, directory


class PanelJudge(base.Judge):
    def __init__(self, alias):
        self.operational_identity = freeze_operation()
        if alias != 'terra':
            super().__init__(alias)
        else:
            self.plan, self.spec, self.bundle, self.identity, self.directory = freeze_terra()
            self.alias = alias
            self.rubric = (HERE / 'rubric.txt').read_text(encoding='utf-8')
            if platform.system() != 'Linux':
                raise RuntimeError('API evaluation is authorized from the server only')
            self.secret = os.environ.get(self.spec['api_key_env'])
            if not self.secret:
                raise RuntimeError('Required credential environment variable is absent')
            self.stop = threading.Event()
            self.rate_lock = threading.Lock()
            self.next_start = 0
        # Only the executor settings change. The scientific manifest remains immutable.
        self.spec = deepcopy(self.spec)
        self.spec['concurrency'] = scheduling_plan()['workers'][alias]
        self.spec['minimum_request_interval_seconds'] = 0
        self.limiter = SharedLimiter('cheaprouter' if alias == 'terra' else 'tokenhub',
                                     alias, self.operational_identity)

    def pace(self):
        self.limiter.acquire()

    def body(self, item):
        if self.alias != 'terra':
            return super().body(item)
        return {'model': self.spec['requested_model'], 'instructions': self.rubric,
                'input': json.dumps(item, ensure_ascii=False), **self.plan['request']}

    def request(self, item, number):
        if self.alias == 'terra':
            record = self.responses_request(item, number)
        else:
            record = super().request(item, number)
        record['operational_identity'] = self.operational_identity
        if record.get('http_status') in (429, 503):
            self.limiter.cooldown(15)
        return record

    def responses_request(self, item, number):
        body = self.body(item)
        self.pace()
        began = time.monotonic()
        limits = self.plan['http']
        record = {'identity': self.identity, 'item_id': item['item_id'], 'attempt': number,
                  'at': base.now(), 'requested_model': self.spec['requested_model'],
                  'annotation_source': 'AI', 'request_sha256': base.digest(body),
                  'requested_controls': {k: v for k, v in body.items() if k not in ('instructions', 'input')},
                  'api_dialect': 'responses', 'requested_effort_tier': 'medium',
                  'requested_reasoning_effort': 'medium'}
        try:
            response = base.requests.post(self.spec['endpoint'],
                headers={'Authorization': 'Bearer ' + self.secret, 'Content-Type': 'application/json'},
                json=body, timeout=(limits['connect_timeout_seconds'], limits['read_timeout_seconds']),
                verify=True, allow_redirects=False)
            record['http_status'] = response.status_code
            try:
                data = response.json()
            except ValueError:
                data = None
            if response.status_code != 200:
                record.update(status='http_error', retriable=response.status_code in limits['retriable_status_codes'])
                if isinstance(data, dict) and isinstance(data.get('error'), dict):
                    error = data['error']
                    record['error_type'] = str(error.get('type', ''))[:100]
                    record['error_code'] = str(error.get('code', ''))[:100]
                    record['error_message'] = str(error.get('message', ''))[:600].replace(self.secret, '[REDACTED]')
                if response.status_code in limits['fatal_status_codes']:
                    self.stop.set()
            elif not isinstance(data, dict):
                record.update(status='non_json_http_response', retriable=True)
            else:
                record['response_metadata'] = {k: data.get(k) for k in (
                    'id', 'model', 'created_at', 'usage', 'system_fingerprint', 'service_tier') if k in data}
                record['returned_model_matches_request'] = data.get('model') == self.spec['requested_model']
                output = data.get('output') or []
                texts = [part.get('text', '') for obj in output if obj.get('type') == 'message'
                         for part in obj.get('content', []) if part.get('type') == 'output_text']
                refusals = [part.get('refusal') for obj in output if obj.get('type') == 'message'
                            for part in obj.get('content', []) if part.get('type') == 'refusal']
                record['content'] = '\n'.join(texts)
                record['refusal'] = refusals or None
                record['response_status'] = data.get('status')
                record['finish_reason'] = 'stop' if data.get('status') == 'completed' else 'incomplete'
                record['incomplete_details'] = data.get('incomplete_details')
                reasoning = data.get('reasoning') or {}
                usage = data.get('usage') or {}
                record['reasoning_observation'] = {
                    'returned_effort': reasoning.get('effort'), 'returned_mode': reasoning.get('mode'),
                    'reasoning_tokens': (usage.get('output_tokens_details') or {}).get('reasoning_tokens'),
                    'reasoning_output_items': sum(obj.get('type') == 'reasoning' for obj in output),
                    'hidden_reasoning_saved': False}
                if not record['returned_model_matches_request']:
                    record.update(status='returned_model_mismatch', retriable=False)
                    self.stop.set()
                elif reasoning.get('effort') != 'medium':
                    record.update(status='reasoning_effort_not_medium', retriable=False)
                    self.stop.set()
                elif data.get('status') != 'completed':
                    record.update(status='truncated', retriable=True)
                elif record['refusal']:
                    record.update(status='refusal', retriable=False)
                else:
                    try:
                        verdict, checks = base.parse_verdict(record['content'], item)
                        record.update(status='valid', verdict=verdict, evidence_checks=checks, retriable=False)
                    except ValueError as exc:
                        record.update(status='invalid_output', validation_error=str(exc), retriable=True)
        except base.requests.RequestException as exc:
            record.update(status='transport_error', error_type=type(exc).__name__, retriable=True)
        record['elapsed_seconds'] = round(time.monotonic() - began, 3)
        return record

    def preflight(self):
        if self.alias != 'terra':
            return super().preflight()
        capability = base.read_json(HERE / 'terra_capability_checks.json')
        demonstrated = any(r.get('http_status') == 200 and r.get('returned_model') == self.spec['requested_model']
                           and (r.get('returned_reasoning') or {}).get('effort') == 'medium'
                           and ((r.get('usage') or {}).get('output_tokens_details') or {}).get('reasoning_tokens', 0) > 0
                           for r in capability['checks'])
        assert demonstrated, 'No verified positive Terra medium-reasoning capability signal'
        results = []
        with base.exclusive_lock(self.directory / '.preflight.lock'):
            for item, expected in base.synthetic_cases():
                record = self.one(item, 'preflight')
                observed = record.get('reasoning_observation') or {}
                actual = record.get('verdict', {}).get('label')
                passed = (record.get('status') == 'valid' and record['evidence_checks']['quote_evidence_gate']
                          and record.get('returned_model_matches_request') and observed.get('returned_effort') == 'medium')
                results.append({'item_id': item['item_id'], 'expected': expected, 'actual': actual,
                                'status': record['status'], 'semantic_match': actual == expected,
                                'reasoning_observation': observed, 'passed': bool(passed)})
                LOG.info('PREFLIGHT model=terra item=%s engineering_pass=%s semantic_match=%s',
                         item['item_id'], passed, actual == expected)
            checks = {'identity': self.identity, 'purpose': 'synthetic_engineering_check_not_study_evidence',
                      'passed': all(row['passed'] for row in results),
                      'semantic_agreement_is_not_engineering_gate': True,
                      'semantic_matches': sum(row['semantic_match'] for row in results),
                      'positive_reasoning_capability_verified': demonstrated, 'checks': results}
            base.atomic_json(self.directory / 'preflight/checks.json', checks)
            if not checks['passed']:
                raise RuntimeError('Terra preflight failed; no formal launch')

    def run(self):
        with base.exclusive_lock(self.directory / '.panel_wrapper.lock'):
            old_execution = self.directory / 'formal/execution.json'
            if old_execution.exists():
                raise RuntimeError('Archive the previous execution metadata before a controlled resume')
            base.atomic_json(self.directory / 'operational_execution.json', {
                'identity': self.identity, 'operational_identity': self.operational_identity,
                'started_at': base.now(), 'pid': os.getpid(), 'workers': self.spec['concurrency'],
                'provider_rate_limit': 'shared rolling 3/s across both TokenHub jobs' if self.alias != 'terra' else 'concurrency=10',
                'scientific_request_unchanged': self.alias != 'terra'})
            super().run()


def verify_request_parity():
    plan = base.read_json(HERE / 'protocol.json')
    items = base.public_data()['items']
    for alias in ('qwen', 'deepseek'):
        old = object.__new__(base.Judge)
        old.plan = plan
        old.spec = plan['models'][alias]
        old.rubric = (HERE / 'rubric.txt').read_text(encoding='utf-8')
        new = object.__new__(PanelJudge)
        new.plan = plan
        new.alias = alias
        new.spec = deepcopy(old.spec)
        new.spec['concurrency'] = scheduling_plan()['workers'][alias]
        new.spec['minimum_request_interval_seconds'] = 0
        new.rubric = old.rubric
        for item in items:
            assert base.digest(new.body(item)) == base.digest(old.body(item))
        print('UNCHANGED_REQUEST_HASHES', alias, len(items))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'verify-parity', 'preflight-run', 'run'])
    parser.add_argument('--model', choices=['qwen', 'deepseek', 'terra'])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
    if args.action == 'verify-parity':
        verify_request_parity()
    elif args.action == 'prepare':
        verify_request_parity()
        print('OPERATIONAL_IDENTITY', freeze_operation())
        print('TERRA_IDENTITY', freeze_terra()[3])
        for alias in ('qwen', 'deepseek'):
            print('PRESERVED_IDENTITY', alias, base.freeze(alias)[3])
    else:
        if not args.model:
            parser.error('--model required')
        obj = PanelJudge(args.model)
        if args.action == 'preflight-run':
            obj.preflight()
        obj.run()


if __name__ == '__main__':
    main()
