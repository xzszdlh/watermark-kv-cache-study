"""Isolated, explicitly authorized Terra model replacement; no base source edits."""
import argparse
from copy import deepcopy
import json
import logging
import os
import platform
import threading

import panel_runner as panel

base = panel.base
HERE = base.HERE
ROOT = base.ROOT


def freeze():
    replacement = base.read_json(HERE / 'terra56_replacement.json')
    extension = base.read_json(HERE / replacement['base_extension'])
    bundle = base.public_data()
    assert replacement['dataset_id'] == bundle['dataset_id']
    plan = deepcopy(base.read_json(HERE / 'protocol.json'))
    spec = deepcopy(replacement['model'])
    plan['models'] = {'terra56': spec}
    plan['request'] = replacement['request']
    plan['authorization'] = replacement['authorization']
    plan['analysis'] = deepcopy(extension['panel'])
    plan['analysis']['active_models'] = replacement['active_panel']
    sources = ['terra56_runner.py', 'terra56_replacement.json', 'terra56_capability_checks.json',
               'panel_runner.py', 'judge.py', 'protocol.json', 'terra_extension.json',
               'scheduling_amendment.json', 'rubric.txt', 'items.json']
    payload = {'model_alias': 'terra56', 'requested_model': spec['requested_model'],
               'dataset_id': bundle['dataset_id'], 'evaluation_type': plan['evaluation_type'],
               'replacement': replacement, 'resolved_plan': plan,
               'sources': {name: base.digest((HERE / name).read_bytes()) for name in sources}}
    identity = base.digest(payload)
    directory = base.OUT / 'terra56'
    path = directory / 'manifest.json'
    if path.exists():
        assert base.read_json(path)['identity'] == identity, 'Terra56 identity changed'
    else:
        base.atomic_json(path, {'identity': identity, 'frozen_at': base.now(), **payload})
        for name in sources:
            dest = directory / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((HERE / name).read_bytes())
    return plan, spec, bundle, identity, directory


class Terra56Judge(panel.PanelJudge):
    def __init__(self):
        self.plan, self.spec, self.bundle, self.identity, self.directory = freeze()
        if platform.system() != 'Linux':
            raise RuntimeError('API evaluation is authorized from the server only')
        self.alias = 'terra56'
        self.operational_identity = self.identity
        self.rubric = (HERE / 'rubric.txt').read_text(encoding='utf-8')
        self.secret = os.environ.get(self.spec['api_key_env'])
        if not self.secret:
            raise RuntimeError('Required credential environment variable is absent')
        self.stop = threading.Event()
        self.rate_lock = threading.Lock()
        self.next_start = 0
        self.limiter = panel.SharedLimiter('cheaprouter', self.alias, self.operational_identity)

    def body(self, item):
        return {'model': self.spec['requested_model'], 'instructions': self.rubric,
                'input': json.dumps(item, ensure_ascii=False), **self.plan['request']}

    def request(self, item, number):
        record = self.responses_request(item, number)
        record['operational_identity'] = self.operational_identity
        if record.get('http_status') in (429, 503):
            self.limiter.cooldown(15)
        return record

    def preflight(self):
        replacement = base.read_json(HERE / 'terra56_replacement.json')
        capability = base.read_json(HERE / replacement['capability_check'])
        demonstrated = any(
            r.get('http_status') == 200 and r.get('returned_model') == self.spec['requested_model']
            and (r.get('returned_reasoning') or {}).get('effort') == 'medium'
            and ((r.get('usage') or {}).get('output_tokens_details') or {}).get('reasoning_tokens', 0) > 0
            for r in capability['checks'])
        assert demonstrated, 'No positive Terra56 medium-reasoning capability signal'
        results = []
        with base.exclusive_lock(self.directory / '.preflight.lock'):
            for item, expected in base.synthetic_cases():
                record = self.one(item, 'preflight')
                observed = record.get('reasoning_observation') or {}
                actual = record.get('verdict', {}).get('label')
                passed = (record.get('status') == 'valid' and record['evidence_checks']['quote_evidence_gate']
                          and record.get('returned_model_matches_request')
                          and observed.get('returned_effort') == 'medium')
                results.append({'item_id': item['item_id'], 'expected': expected, 'actual': actual,
                                'status': record['status'], 'semantic_match': actual == expected,
                                'reasoning_observation': observed, 'passed': bool(passed)})
                base.LOG.info('PREFLIGHT model=terra56 item=%s engineering_pass=%s semantic_match=%s',
                              item['item_id'], passed, actual == expected)
            checks = {'identity': self.identity, 'purpose': 'synthetic_engineering_check_not_study_evidence',
                      'passed': all(r['passed'] for r in results),
                      'semantic_agreement_is_not_engineering_gate': True,
                      'semantic_matches': sum(r['semantic_match'] for r in results),
                      'positive_reasoning_capability_verified': demonstrated, 'checks': results}
            base.atomic_json(self.directory / 'preflight/checks.json', checks)
            if not checks['passed']:
                raise RuntimeError('Terra56 preflight failed; formal run not launched')

    def run(self):
        with base.exclusive_lock(self.directory / '.panel_wrapper.lock'):
            if (self.directory / 'formal/execution.json').exists():
                raise RuntimeError('Archive previous execution metadata before controlled resume')
            base.atomic_json(self.directory / 'operational_execution.json', {
                'identity': self.identity, 'operational_identity': self.operational_identity,
                'started_at': base.now(), 'pid': os.getpid(), 'workers': self.spec['concurrency'],
                'provider_rate_limit': 'concurrency=10', 'replaces_model': 'gpt-6.1-terra',
                'requested_model': self.spec['requested_model'], 'requested_reasoning_effort': 'medium'})
            base.Judge.run(self)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'preflight-run', 'run'])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
    if args.action == 'prepare':
        print('TERRA56_IDENTITY', freeze()[3])
    else:
        judge = Terra56Judge()
        if args.action == 'preflight-run':
            judge.preflight()
        judge.run()


if __name__ == '__main__':
    main()
