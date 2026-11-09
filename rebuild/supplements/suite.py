"""Frozen post-observation KV sensitivity suite; disjoint condition/seed workers."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT/'kvstudy'), str(ROOT/'execution')]
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import fcntl
import gc
import logging
import math
import os
import signal
import subprocess
import time
import traceback
from common import read_json, atomic_json, digest, run_manifest, lock_manifest, stage_status, LOG
from kv_shard_worker import verify_sources, link_checked, publish
from cache_adapter import selection_policy, random_positions

BASE = ROOT/'runs/kv-supplement'
PLAN = HERE/'protocol.json'
ENV = {**os.environ, 'HF_HUB_OFFLINE':'1', 'HF_DATASETS_OFFLINE':'1',
       'TOKENIZERS_PARALLELISM':'false', 'PYTHONUNBUFFERED':'1', 'OMP_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4'}
STOP = False


def configurations(plan):
    base = read_json(ROOT/'runs/kv-cache/full/formal/manifest.json')['protocol']
    for group in plan['groups']:
        for case in group['cases']:
            name = group['name']+'-'+case['name']
            c = deepcopy(base)
            c['experiment_id'] = 'kv-supplement-'+name
            c['study_role'] = plan['study_role']
            c['cache_condition'] = name
            c['generation'].update(group['generation'])
            c['generation'].update(seeds=plan['seeds'], batch_size=plan['batch_size'])
            c['watermark']['deltas'] = plan['deltas']
            c['resources'] = deepcopy(plan['resource_policy'])
            c['cache']['conditions'] = [name]
            c['cache']['cross_ratios'] = {name:case['cross']}
            c['cache']['self_ratios'] = {name:case['self']}
            c['cross_selection'] = case.get('cross_selection','evenly_spaced')
            c['source_policy_seed'] = plan['selection_seed']
            c['cache']['cross_policy'] = plan['random_source_policy'] if c['cross_selection']=='random_endpoints' else base['cache']['cross_policy']
            c['supplement_group'] = group['name']
            c['supplement_reference'] = group['reference']
            yield name, c


def runtime(cap):
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(cap)


def prepare():
    plan=read_json(PLAN); BASE.mkdir(parents=True,exist_ok=True)
    frozen=BASE/'frozen_plan.json'
    payload={'plan_sha256':digest(PLAN.read_bytes()),'plan':plan}
    if frozen.exists():
        if read_json(frozen)['plan_sha256']!=payload['plan_sha256']: raise RuntimeError('Frozen plan changed')
    else: atomic_json(frozen,{**payload,'frozen_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    original=read_json(ROOT/'runs/kv-cache/full/formal/manifest.json')
    verify_sources(original)
    names=[]
    for name,config in configurations(plan):
        out=BASE/name/'formal'; m=run_manifest(config,'formal')
        for p in sorted((ROOT/'kvstudy').glob('*.py')): m['sources'][p.relative_to(ROOT).as_posix()]=digest(p.read_bytes())
        for p in [HERE/'suite.py',HERE/'cache_adapter.py',ROOT/'execution/kv_shard_worker.py']:
            m['sources'][p.relative_to(ROOT).as_posix()]=digest(p.read_bytes())
        m['gpu']='A100 placement recorded per shard'; m['CUDA_VISIBLE_DEVICES']='dynamic placement'
        m['plan_sha256']=payload['plan_sha256']
        identity=lock_manifest(out,m)
        link_checked(ROOT/'runs/formal/samples.json',out/'samples.json')
        names.append(name)
        LOG.info('FROZEN %s identity=%s',name,identity)
    assert len(names)==plan['new_configurations']
    atomic_json(BASE/'index.json',{'conditions':names,'new_records':plan['new_generation_records'],'seeds':plan['seeds']})


def preflight():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from kvgeneration import decode_cache_batch
    from evaluation import evaluate_all
    from kvcache import BartCache
    runtime(.16)
    plan=read_json(PLAN); configs=dict(configurations(plan))
    out=BASE/'preflight'; out.mkdir(parents=True,exist_ok=True)
    samples=read_json(ROOT/'runs/kv-cache/full/smoke/samples.json')[:2]
    cfg=next(iter(configs.values()))
    tok=AutoTokenizer.from_pretrained(ROOT/cfg['model']['directory'],local_files_only=True)
    model=AutoModelForSeq2SeqLM.from_pretrained(ROOT/cfg['model']['directory'],local_files_only=True,
           torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
    checks=[]
    paths=['retention-both25','retention-both75','random50-both50','nucleus256-full','topk50-full']
    reference_cfg=read_json(ROOT/'runs/kv-cache/full/formal/manifest.json')['protocol']
    # Exact full-cache path check using an existing fixed 32-document batch.
    reference=read_json(ROOT/'runs/kv-cache/full/formal/generation/seed42_delta0_batch00000.json')
    lookup={s['id']:s for s in read_json(ROOT/'runs/formal/samples.json')}
    batch=[lookup[i] for i in reference['sample_ids']]
    rows,_=decode_cache_batch(model,tok,batch,reference_cfg,42,0.)
    assert [r['raw_ids'] for r in rows]==[r['raw_ids'] for r in reference['records']]
    checks.append({'test':'unchanged_full_path','exact_tokens_documents':32})
    pending=[]
    for name in paths:
        config=configs[name]
        # End-to-end unique new paths. Long-limit case keeps 256, all others 32.
        c=deepcopy(config)
        if name!='nucleus256-full': c['generation']['max_new_tokens']=32
        for delta in (0.,2.):
            with selection_policy(c['cross_selection'],c['source_policy_seed']):
                rows,audit=decode_cache_batch(model,tok,samples,c,42,delta)
            case=c['cache_condition']; cr=c['cache']['cross_ratios'][case]; sr=c['cache']['self_ratios'][case]
            assert all(len(p)==math.ceil(n*cr) for p,n in zip(audit['cross_retained_source_positions'],audit['cross_valid_source_lengths']))
            for step in audit['cache_by_decoder_step']:
                assert step['self_cache_after_eviction']==math.ceil(step['positions_seen']*sr)
            directory=out/name/f'delta{delta:g}'
            identity=digest({'config':c,'purpose':'engineering_only'})
            atomic_json(directory/'generation/batch.json',{'identity':identity,'sample_ids':[s['id'] for s in samples],'records':rows})
            pending.append((c,directory,identity))
            checks.append({'test':'new_path_generation','name':name,'delta':delta,'cache_storage_and_positions':True})
    # Random selection must be reproducible and nested ownership must restore.
    for ratio in (.25,.5,.75,1.):
        p=random_positions(list(range(41)),ratio,plan['selection_seed'])
        assert p==random_positions(list(range(41)),ratio,plan['selection_seed'])
        assert len(p)==math.ceil(41*ratio) and p==sorted(set(p))
    # Exercise learned absolute positions beyond the old 128-step ceiling.
    inputs=tok.pad({'input_ids':[samples[0]['source_input_ids']]},padding=True,return_tensors='pt').to(model.device)
    with torch.inference_mode(),BartCache(model,inputs,.5,.5) as cache:
        for step in range(160):
            token=model.config.decoder_start_token_id if step==0 else (tok.bos_token_id if step==1 else 100)
            logits,a=cache.forward(torch.tensor([[token]],device=model.device),step)
            assert torch.isfinite(logits).all() and a['decoder_absolute_position']==step
        assert a['self_cache_after_eviction']==80
    checks.append({'test':'long_absolute_positions','steps':160,'retained_self_positions':80})
    del model; gc.collect(); torch.cuda.empty_cache()
    for c,directory,identity in pending:
        evaluate_all(c,samples,tok,directory,identity)
        ev=read_json(directory/'evaluation/batch.json')
        assert len(ev['records'])==2
        assert all(s['alignment_valid'] and s['nli'] is not None for r in ev['records'] for s in r['sentences'])
    atomic_json(out/'checks.json',{'passed':True,'plan_sha256':digest(PLAN.read_bytes()),'checks':checks,'purpose':'engineering_only'})
    LOG.info('SUPPLEMENT_PREFLIGHT_PASS tests=%d',len(checks))


def worker(name,seed):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from kvgeneration import generate_condition
    from evaluation import evaluate_all
    m=read_json(BASE/name/'formal/manifest.json'); verify_sources(m)
    identity=m['identity']; c=deepcopy(m['protocol']); canonical=BASE/name/'formal'
    if seed not in c['generation']['seeds']: raise ValueError('Seed not in protocol')
    view=BASE/'shards'/name/f'seed{seed}'; view.mkdir(parents=True,exist_ok=True)
    with open(view/'.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime(c['resources']['allocator_fraction']); c['generation']['seeds']=[seed]
        atomic_json(view/'execution.json',{'identity':identity,'pid':os.getpid(),'seed':seed,
                    'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'time':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
        for folder in ('generation','evaluation'):
            for p in (canonical/folder).glob(f'seed{seed}_*.json'): link_checked(p,view/folder/p.name)
        samples=read_json(canonical/'samples.json'); tok=AutoTokenizer.from_pretrained(ROOT/c['model']['directory'],local_files_only=True)
        expected=len(c['watermark']['deltas'])*math.ceil(len(samples)/c['generation']['batch_size'])
        try:
            if len(list((view/'generation').glob('*.json')))<expected:
                model=AutoModelForSeq2SeqLM.from_pretrained(ROOT/c['model']['directory'],local_files_only=True,
                       torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
                with selection_policy(c['cross_selection'],c['source_policy_seed']):
                    generate_condition(model,tok,c,samples,view,identity)
                del model; gc.collect(); torch.cuda.empty_cache()
            publish(view,canonical,seed,identity)
            evaluate_all(c,samples,tok,view,identity)
            counts=publish(view,canonical,seed,identity)
            if counts!={'generation':expected,'evaluation':expected}: raise RuntimeError('Incomplete shard')
            stage_status(view,'complete',identity=identity,records=len(samples)*len(c['watermark']['deltas']))
        except Exception:
            stage_status(view,'failed',traceback=traceback.format_exc()); raise
        finally: publish(view,canonical,seed,identity)


def finish_one(name):
    import torch
    from analysis import analyze
    torch.set_num_threads(1)
    out=BASE/name/'formal'; m=read_json(out/'manifest.json'); verify_sources(m)
    if (out/'analysis.json').exists() and (out/'status.json').exists() and read_json(out/'status.json')['stage']=='complete':
        return name
    result=analyze(m['protocol'],out,m['identity'])
    result['interpretation_limits']=[s for s in result['interpretation_limits'] if s!='No KV compression or causal claim is tested.']
    result['interpretation_limits'] += ['Post-observation sensitivity on the same documents, not independent confirmation.',
             'Physical BART cache ablation; not a HalluKV replication or an equal-total-memory comparison.']
    result['associations']=[{**a,'primary':False,'posthoc_reference_cell':a['primary']} for a in result['associations']]
    result['cache_protocol']=m['protocol']['cache']
    atomic_json(out/'analysis.json',result)
    stage_status(out,'complete',identity=m['identity'],records=result['record_count'],human_validation_pending=True)
    return name


def counts(name,seed):
    return {f:len({p.name for root in (BASE/name/'formal',BASE/'shards'/name/f'seed{seed}')
                   for p in (root/f).glob(f'seed{seed}_*.json')}) for f in ('generation','evaluation')}


def stop_handler(signum,frame):
    global STOP
    STOP=True


def fleet():
    import kv_fleet as scheduler
    plan=read_json(PLAN)
    check=read_json(BASE/'preflight/checks.json')
    assert check['passed'] and check['plan_sha256']==digest(PLAN.read_bytes())
    rp=plan['resource_policy']; active={}; failed={}; finished=set()
    expected=44; names=read_json(BASE/'index.json')['conditions']
    pending=[(n,s) for n in names for s in plan['seeds']]
    total=len(pending)*expected
    for key in pending[:]:
        if counts(*key)=={'generation':expected,'evaluation':expected}:
            finished.add(key);pending.remove(key)
    scheduler.CHILDREN=active
    with open(BASE/'.fleet.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,stop_handler)
        atomic_json(BASE/'fleet.pid.json',{'pid':os.getpid()})
        last=0
        try:
            while pending or active:
                if STOP: raise InterruptedError('stop requested')
                for key,j in list(active.items()):
                    code=j['proc'].poll()
                    if code is not None:
                        j['log'].close(); del active[key]
                        if code==0 and counts(*key)=={'generation':expected,'evaluation':expected}:
                            finished.add(key); LOG.info('DONE %s',key)
                        else: failed[str(key)]=code; LOG.error('FAILED %s code=%d',key,code)
                infos=scheduler.gpu_inventory()
                while pending:
                    choices=[]
                    for gpu,i in infos.items():
                        n=sum(j['gpu']==gpu for j in active.values())
                        if n+i['other_study_processes']>=rp['maximum_workers'][gpu]: continue
                        reserve=rp['allocator_fraction']*i['total']+rp['context_allowance_MiB_per_worker']
                        available=i['free']+i['fleet_used']-n*reserve-rp['free_reserve_MiB']
                        if available<reserve or i['free']<reserve+rp['free_reserve_MiB']: continue
                        choices.append((n+i['other_study_processes'],i['util'],-available,gpu))
                    if not choices: break
                    gpu=min(choices)[-1];key=pending.pop(0); name,seed=key
                    log=open(ROOT/'logs'/f'supplement-{name}-s{seed}.log','a',encoding='utf-8')
                    p=subprocess.Popen([sys.executable,'-u',str(HERE/'suite.py'),'worker','--condition',name,'--seed',str(seed)],
                       cwd=ROOT,env={**ENV,'CUDA_VISIBLE_DEVICES':gpu},stdout=log,stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL,start_new_session=True)
                    active[key]={'proc':p,'gpu':gpu,'log':log}
                    LOG.info('START %s seed=%s GPU=%s PID=%s',name,seed,gpu,p.pid)
                if time.monotonic()-last>20:
                    progress={f:sum(counts(n,s)[f] for n in names for s in plan['seeds']) for f in ('generation','evaluation')}
                    stage_status(BASE,'running',active=len(active),queued=len(pending),finished=len(finished),failed=failed,
                        expected_batches_each_phase=total,**progress,
                        jobs=[{'name':n,'seed':s,'gpu':j['gpu'],'pid':j['proc'].pid} for (n,s),j in active.items()])
                    LOG.info('PROGRESS %s / %d active=%d queued=%d failed=%s',progress,total,len(active),len(pending),failed)
                    last=time.monotonic()
                time.sleep(3)
            if failed: raise RuntimeError(f'Failed shards: {failed}; no automatic blind retry')
            stage_status(BASE,'analysis',completed_shards=len(finished))
            # GPU children are gone; statistics use four CPU processes.
            with ProcessPoolExecutor(max_workers=4) as pool:
                for name in pool.map(finish_one,names): LOG.info('ANALYSIS_COMPLETE %s',name)
            summaries={n:read_json(BASE/n/'formal/analysis.json') for n in names}
            atomic_json(BASE/'comparison.json',{'status':'automatic_complete_human_validation_pending',
                        'plan_sha256':digest(PLAN.read_bytes()),'new_records':sum(r['record_count'] for r in summaries.values()),'runs':summaries})
            stage_status(BASE,'complete',new_records=plan['new_generation_records'],human_validation_pending=True)
            LOG.info('SUPPLEMENT_COMPLETE records=%d',plan['new_generation_records'])
        except Exception:
            stage_status(BASE,'failed',traceback=traceback.format_exc()); raise
        finally:
            for j in active.values():
                if j['proc'].poll() is None:
                    os.killpg(j['proc'].pid,signal.SIGTERM)
                    try: j['proc'].wait(timeout=20)
                    except subprocess.TimeoutExpired: os.killpg(j['proc'].pid,signal.SIGKILL); j['proc'].wait()
                j['log'].close()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['prepare','preflight','worker','fleet'])
    p.add_argument('--condition'); p.add_argument('--seed',type=int); args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True)
    if args.action=='worker': worker(args.condition,args.seed)
    else: globals()[args.action]()
