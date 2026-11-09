"""Independent API judges on a frozen blinded sample. No human labels are written.
Credentials are read from the process environment and never serialized.
"""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT),str(ROOT/'postprocess')]
import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import json
import logging
import os
import platform
import re
import threading
import time
import requests
from common import read_json,atomic_json,digest
from validate_human_audit import load_public

LOG=logging.getLogger('ai_audit')
OUT=ROOT/'runs/ai-audit'
LABELS={'supported','contradicted','not_in_source','unclear'}
FIELDS={'item_id','label','evidence_quotes','issue_span','rationale'}


@contextmanager
def exclusive_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'a+b') as f:
        if f.seek(0,2)==0:f.write(b'0');f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            if os.name=='nt':f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f,fcntl.LOCK_UN)


def now():return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def public_data():
    src=HERE/'items.json'
    if src.exists():bundle=read_json(src)
    else:
        bundle=load_public(ROOT/'human_audit/开始人工核验.html')
        atomic_json(src,bundle)
    plan=read_json(HERE/'protocol.json')
    assert len(bundle['items'])==plan['items']==260
    assert bundle['dataset_id']==plan['dataset_id']==digest(bundle['items'])
    for row in bundle['items']:
        assert set(row)=={'item_id','source_visible_to_model','summary_sentence','summary_context'}
    assert len({r['item_id'] for r in bundle['items']})==260
    return bundle


def freeze(alias):
    bundle=public_data();plan=read_json(HERE/'protocol.json');spec=plan['models'][alias]
    m={'evaluation_type':plan['evaluation_type'],'dataset_id':bundle['dataset_id'],
       'plan':plan,'model_alias':alias,'requested_model':spec['requested_model'],
       'sources':{p.name:digest(p.read_bytes()) for p in [HERE/'judge.py',HERE/'rubric.txt',HERE/'protocol.json',HERE/'items.json']}}
    identity=digest(m);directory=OUT/alias;directory.mkdir(parents=True,exist_ok=True)
    path=directory/'manifest.json'
    if path.exists():assert read_json(path)['identity']==identity,'Resume identity differs'
    else:
        atomic_json(path,{'identity':identity,'frozen_at':now(),**m})
        for name in m['sources']:
            dest=directory/'source'/name;dest.parent.mkdir(exist_ok=True)
            dest.write_bytes((HERE/name).read_bytes())
    return plan,spec,bundle,identity,directory


def parse_verdict(content,item):
    if not isinstance(content,str):raise ValueError('non_text_content')
    text=content.strip()
    if text.startswith('```'):
        match=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',text,re.S)
        if not match:raise ValueError('malformed_code_fence')
        text=match.group(1)
    try:value=json.loads(text)
    except (json.JSONDecodeError,TypeError):raise ValueError('invalid_json') from None
    if not isinstance(value,dict) or set(value)!=FIELDS:raise ValueError('schema_fields')
    if value['item_id']!=item['item_id']:raise ValueError('wrong_item_id')
    if value['label'] not in LABELS:raise ValueError('label_outside_rubric')
    if not isinstance(value['evidence_quotes'],list) or any(not isinstance(q,str) or not q.strip() for q in value['evidence_quotes']):
        raise ValueError('invalid_quote_list')
    if not isinstance(value['issue_span'],str) or not isinstance(value['rationale'],str) or not value['rationale'].strip():
        raise ValueError('invalid_explanation_fields')
    normalize=lambda text:re.sub(r'\s+',' ',text).strip()
    source=normalize(item['source_visible_to_model'])
    checks=[normalize(q) in source for q in value['evidence_quotes']]
    required=value['label'] in ('supported','contradicted')
    quote_gate=all(checks) and (bool(checks) if required else True)
    span=value['issue_span'];span_found=not span or normalize(span) in normalize(item['summary_sentence'])
    return value,{'quote_matches':checks,'quote_evidence_gate':quote_gate,'issue_span_found_in_target':span_found,
                  'semantic_entailment_not_verified_by_string_check':True}


def synthetic_cases():
    source='On Monday, three hikers were rescued near Pine Lake. The local fire department led the rescue. No one was injured.'
    return [({'item_id':'P'+str(i),'source_visible_to_model':source,'summary_sentence':sentence,'summary_context':sentence},expected)
        for i,(sentence,expected) in enumerate([
            ('Three hikers were rescued near Pine Lake on Monday.','supported'),
            ('Five hikers were rescued near Pine Lake on Monday.','contradicted'),
            ('A military helicopter rescued the hikers.','not_in_source'),
            ('They because blue however the.','unclear')],1)]


class Judge:
    def __init__(self,alias):
        self.plan,self.spec,self.bundle,self.identity,self.directory=freeze(alias)
        self.alias=alias;self.rubric=(HERE/'rubric.txt').read_text(encoding='utf-8')
        if self.spec['execution_location']!='server' or platform.system()!='Linux':
            raise RuntimeError('This protocol authorizes API judging from the Linux server only')
        self.secret=os.environ.get(self.spec['api_key_env'])
        if not self.secret:raise RuntimeError('Required credential environment variable is absent: '+self.spec['api_key_env'])
        self.stop=threading.Event()
        self.rate_lock=threading.Lock();self.next_start=0.0

    def body(self,item):
        settings=self.plan['request']
        assert self.spec['api_dialect']=='chat_completions'
        return {'model':self.spec['requested_model'],'messages':[{'role':'system','content':self.rubric},
                {'role':'user','content':json.dumps(item,ensure_ascii=False)}],
                **settings,**self.spec['request_overrides']}

    def pace(self):
        with self.rate_lock:
            delay=max(0,self.next_start-time.monotonic())
            if delay:time.sleep(delay)
            self.next_start=time.monotonic()+self.spec['minimum_request_interval_seconds']

    def request(self,item,number):
        body=self.body(item);self.pace();began=time.monotonic();limits=self.plan['http']
        record={'identity':self.identity,'item_id':item['item_id'],'attempt':number,'at':now(),
                'requested_model':self.spec['requested_model'],'request_sha256':digest(body),'annotation_source':'AI',
                'requested_controls':{k:v for k,v in body.items() if k not in ('messages','instructions','input')},
                'api_dialect':self.spec['api_dialect'],
                'requested_effort_tier':self.plan['requested_effort_tier'],
                'requested_reasoning_effort':self.spec['expected_returned_effort']}
        try:
            response=requests.post(self.spec['endpoint'],headers={'Authorization':'Bearer '+self.secret,
                 'Content-Type':'application/json'},json=body,
                 timeout=(limits['connect_timeout_seconds'],limits['read_timeout_seconds']),
                 verify=True,allow_redirects=False)
            record['http_status']=response.status_code
            try:data=response.json()
            except ValueError:data=None
            if response.status_code!=200:
                record['status']='http_error'
                record['retriable']=response.status_code in limits['retriable_status_codes']
                if isinstance(data,dict):
                    error=data.get('error',{})
                    if isinstance(error,dict):
                        record['error_type']=str(error.get('type',''))[:100]
                        record['error_code']=str(error.get('code',''))[:100]
                        record['error_message']=str(error.get('message',''))[:600].replace(self.secret,'[REDACTED]')
                if response.status_code in limits['fatal_status_codes']:self.stop.set()
            elif not isinstance(data,dict):record.update(status='non_json_http_response',retriable=True)
            else:
                record['response_metadata']={k:data.get(k) for k in ('id','model','created','usage','system_fingerprint','service_tier') if k in data}
                record['returned_model_matches_request']=data.get('model')==self.spec['requested_model']
                usage=data.get('usage') or {}
                choices=data.get('choices') or []
                if choices and isinstance(choices,list) and isinstance(choices[0],dict):
                    choice=choices[0];message=choice.get('message') or {}
                    record['finish_reason']=choice.get('finish_reason');record['content']=message.get('content')
                    record['refusal']=message.get('refusal')
                    record['reasoning_observation']={
                        'returned_effort':data.get('reasoning_effort'),
                        'reasoning_tokens':(usage.get('completion_tokens_details') or {}).get('reasoning_tokens'),
                        'reasoning_field_characters':sum(len(message.get(k) or '') for k in ('reasoning','reasoning_content') if isinstance(message.get(k),str)),
                        'hidden_reasoning_saved':False}
                else:record.update(status='missing_choices',retriable=True)
                # Keep final answer and reasoning metadata, never hidden reasoning text.
                if 'status' not in record:
                    effort=record.get('reasoning_observation',{}).get('returned_effort')
                    if not record['returned_model_matches_request']:
                        record.update(status='returned_model_mismatch',retriable=False);self.stop.set()
                    elif effort is not None and effort!=self.spec['expected_returned_effort']:
                        record.update(status='reasoning_effort_mismatch',retriable=False);self.stop.set()
                    elif record.get('finish_reason') in ('length','incomplete'):record.update(status='truncated',retriable=True)
                    elif record.get('refusal'):record.update(status='refusal',retriable=False)
                    else:
                        try:
                            verdict,checks=parse_verdict(record.get('content'),item)
                            record.update(status='valid',verdict=verdict,evidence_checks=checks,retriable=False)
                        except ValueError as exc:record.update(status='invalid_output',validation_error=str(exc),retriable=True)
        except requests.RequestException as exc:
            record.update(status='transport_error',error_type=type(exc).__name__,retriable=True)
        record['elapsed_seconds']=round(time.monotonic()-began,3)
        return record

    def one(self,item,phase):
        folder=self.directory/phase;dest=folder/'records'/(item['item_id']+'.json')
        if dest.exists():
            old=read_json(dest);assert old['identity']==self.identity
            return old
        attempts=folder/'attempts'/item['item_id'];attempts.mkdir(parents=True,exist_ok=True)
        maximum=self.plan['http']['max_attempts_per_item'];existing=sorted(attempts.glob('attempt*.json'))
        last=None
        for p in existing:
            value=read_json(p);assert value['identity']==self.identity;last=value
            if value['status']=='valid':
                atomic_json(dest,value);return value
        if last is not None and not last.get('retriable',False):
            atomic_json(dest,last);return last
        for number in range(len(existing)+1,maximum+1):
            if self.stop.is_set():return {'item_id':item['item_id'],'status':'cancelled_due_to_fatal_error'}
            record=self.request(item,number)
            atomic_json(attempts/f'attempt{number:02}.json',record);last=record
            if record['status']=='valid' or not record.get('retriable'):
                atomic_json(dest,record);return record
            if number<maximum:time.sleep(self.plan['http']['retry_backoff_seconds'][number-1])
        if last is not None:
            last={**last,'attempts_exhausted':True};atomic_json(dest,last);return last
        raise RuntimeError('No attempt produced')

    def preflight_record(self,item):
        reuse=self.spec.get('reuse_preflight')
        dest=self.directory/'preflight/records'/(item['item_id']+'.json')
        if not reuse or dest.exists():return self.one(item,'preflight')
        source=ROOT/reuse/'records'/(item['item_id']+'.json')
        original=read_json(source)
        assert original['request_sha256']==digest(self.body(item)), 'Cannot reuse a changed synthetic request'
        assert original['status']=='valid' and original['requested_model']==self.spec['requested_model']
        verdict,checks=parse_verdict(original['content'],item)
        assert verdict==original['verdict'] and checks==original['evidence_checks']
        record={**original,'identity':self.identity,
                'preflight_reuse':{'path':source.relative_to(ROOT).as_posix(),'sha256':digest(source.read_bytes()),
                                  'original_identity':original['identity'],'no_new_API_request':True},
                'requested_effort_tier':self.plan['requested_effort_tier']}
        atomic_json(dest,record)
        return record

    def preflight(self):
        results=[]
        with exclusive_lock(self.directory/'.preflight.lock'):
            for item,expected in synthetic_cases():
                r=self.preflight_record(item)
                observation=r.get('reasoning_observation') or {}
                thinking=(observation.get('reasoning_tokens') or 0)>0 or (observation.get('reasoning_field_characters') or 0)>0
                passed=(r.get('status')=='valid' and r['evidence_checks']['quote_evidence_gate']
                        and r.get('returned_model_matches_request') and thinking)
                actual=r.get('verdict',{}).get('label')
                results.append({'item_id':item['item_id'],'expected':expected,'status':r['status'],
                   'actual':actual,'semantic_match':actual==expected,'reasoning_observation':observation,
                   'requested_effort':r.get('requested_reasoning_effort'),'passed':bool(passed),
                   'reused':bool(r.get('preflight_reuse'))})
                LOG.info('PREFLIGHT model=%s item=%s engineering_pass=%s semantic_match=%s status=%s',
                         self.alias,item['item_id'],passed,actual==expected,r['status'])
            checks={'identity':self.identity,'purpose':'synthetic_engineering_check_not_study_evidence',
                    'semantic_agreement_is_not_engineering_gate':True,
                    'semantic_matches':sum(r['semantic_match'] for r in results),
                    'passed':all(r['passed'] for r in results),'checks':results}
            atomic_json(self.directory/'preflight/checks.json',checks)
            if not checks['passed']:raise RuntimeError('Synthetic engineering preflight failed; formal judging not launched')

    def run(self):
        gate=read_json(self.directory/'preflight/checks.json')
        assert gate['passed'] and gate['identity']==self.identity
        items=self.bundle['items'];folder=self.directory/'formal';folder.mkdir(exist_ok=True)
        with exclusive_lock(self.directory/'.formal.lock'):
            atomic_json(folder/'execution.json',{'identity':self.identity,'pid':os.getpid(),'execution_location':self.spec['execution_location'],
                'platform':platform.platform(),'python':sys.version,'requests_version':requests.__version__,'started_at':now(),
                'credentials_persisted':False,'endpoint':self.spec['endpoint'],'workers':self.spec['concurrency'],
                'requested_effort_tier':self.plan['requested_effort_tier'],
                'request_settings':self.plan['request'],'request_overrides':self.spec['request_overrides']})
            completed=[];last_log=0
            with ThreadPoolExecutor(max_workers=self.spec['concurrency']) as pool:
                pending={pool.submit(self.one,item,'formal'):item['item_id'] for item in items}
                while pending:
                    done,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future);completed.append(future.result())
                    if done or time.monotonic()-last_log>=15:
                        valid=sum(r.get('status')=='valid' for r in completed)
                        failures=len(completed)-valid
                        status={'stage':'running','identity':self.identity,'at':now(),'total':len(items),'returned':len(completed),
                           'valid':valid,'failed':failures,'pending':len(pending),'pid':os.getpid(),'annotation_source':'AI'}
                        atomic_json(folder/'status.json',status)
                        LOG.info('PROGRESS model=%s valid=%d/%d failed=%d pending=%d',self.alias,valid,len(items),failures,len(pending));last_log=time.monotonic()
            records=sorted(completed,key=lambda r:r['item_id']);valid=[r for r in records if r.get('status')=='valid']
            usage={};attempt_count=0
            for p in (folder/'attempts').glob('*/*.json'):
                attempt_count+=1;u=read_json(p).get('response_metadata',{}).get('usage') or {}
                for key,value in u.items():
                    if isinstance(value,(int,float)):usage[key]=usage.get(key,0)+value
            summary={'identity':self.identity,'model_alias':self.alias,'requested_model':self.spec['requested_model'],
               'evaluation_type':self.plan['evaluation_type'],'dataset_id':self.bundle['dataset_id'],'total':len(items),
               'valid':len(valid),'failed':len(items)-len(valid),'http_attempts':attempt_count,'reported_usage_all_attempts':usage,
               'quote_gate_failures':sum(not r['evidence_checks']['quote_evidence_gate'] for r in valid),
               'returned_model_mismatches':sum(not r.get('returned_model_matches_request') for r in valid),
               'requested_effort_tier':self.plan['requested_effort_tier'],
               'requested_reasoning_effort':self.spec['expected_returned_effort'],
               'reasoning_observation_counts':{
                   'tokens_reported':sum(r.get('reasoning_observation',{}).get('reasoning_tokens') is not None for r in valid),
                   'positive_reasoning_tokens':sum((r.get('reasoning_observation',{}).get('reasoning_tokens') or 0)>0 for r in valid),
                   'zero_reasoning_tokens':sum(r.get('reasoning_observation',{}).get('reasoning_tokens')==0 for r in valid),
                   'missing_reasoning_tokens':sum(r.get('reasoning_observation',{}).get('reasoning_tokens') is None for r in valid),
                   'requested_effort_echoed':sum(r.get('reasoning_observation',{}).get('returned_effort')==self.spec['expected_returned_effort'] for r in valid)},
               'reported_reasoning_tokens_final_answers':sum(r.get('reasoning_observation',{}).get('reasoning_tokens') or 0 for r in valid),
               'completed_at':now(),'records':records}
            atomic_json(folder/'results.json',summary)
            atomic_json(folder/'status.json',{'stage':'complete' if len(valid)==len(items) else 'complete_with_missing',
                'identity':self.identity,'at':now(),'total':len(items),'valid':len(valid),'failed':len(items)-len(valid),'annotation_source':'AI'})
            LOG.info('JUDGE_FINISHED model=%s valid=%d/%d quote_flags=%d',self.alias,len(valid),len(items),summary['quote_gate_failures'])


def selftest():
    item={'item_id':'X','source_visible_to_model':'Three people arrived.','summary_sentence':'Three people arrived.'}
    verdict={'item_id':'X','label':'supported','evidence_quotes':['Three people arrived.'],'issue_span':'','rationale':'The source states this.'}
    _,checks=parse_verdict(json.dumps(verdict),item);assert checks['quote_evidence_gate']
    verdict['evidence_quotes']=['Five people arrived.'];_,checks=parse_verdict(json.dumps(verdict),item);assert not checks['quote_evidence_gate']
    verdict['label']='unknown'
    try:parse_verdict(json.dumps(verdict),item)
    except ValueError:pass
    else:raise AssertionError('Invalid label accepted')
    print('JUDGE_SELFTEST_PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','preflight','run','preflight-run','selftest'])
    parser.add_argument('--model',choices=list(read_json(HERE/'protocol.json')['models']));args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True)
    if args.action=='selftest':selftest()
    elif args.action=='prepare':
        aliases=[args.model] if args.model else list(read_json(HERE/'protocol.json')['models'])
        for alias in aliases:freeze(alias);print('JUDGE_PROTOCOL_FROZEN',alias)
    else:
        if not args.model:parser.error('--model required')
        judge=Judge(args.model)
        if args.action in ('preflight','preflight-run'):judge.preflight()
        if args.action in ('run','preflight-run'):judge.run()
