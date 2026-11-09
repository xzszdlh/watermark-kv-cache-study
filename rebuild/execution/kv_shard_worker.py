"""Execute one KV condition/seed shard using unchanged frozen scientific functions.
This module changes task placement only. Old manifests/checkpoints are not rewritten.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'kvstudy'))
import argparse
from copy import deepcopy
import fcntl
import gc
import hashlib
import json
import logging
import math
import os
import time
import torch
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from common import LOG,read_json,atomic_json,stage_status,digest,load_protocol
from kvgeneration import generate_condition,decode_cache_batch
from evaluation import evaluate_all

BASE=ROOT/'runs/kv-cache'


def verify_sources(manifest):
    for name,expected in manifest['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Frozen scientific dependency changed: {name}')


def link_checked(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if os.path.samefile(source,destination):return
        if hashlib.sha256(source.read_bytes()).digest()!=hashlib.sha256(destination.read_bytes()).digest():
            raise RuntimeError(f'Conflicting checkpoint (not overwritten): {destination}')
        return
    os.link(source,destination)


def publish(view,canonical,seed,identity):
    counts={}
    for folder in ('generation','evaluation'):
        files=sorted((view/folder).glob(f'seed{seed}_*.json'))
        for src in files:
            data=read_json(src)
            if data['identity']!=identity:raise RuntimeError('Wrong checkpoint identity')
            if any(r['seed']!=seed for r in data['records']):raise RuntimeError('Shard contains another seed')
            link_checked(src,canonical/folder/src.name)
        counts[folder]=len(files)
    return counts


def configure_runtime(memory_fraction):
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(memory_fraction)


def execution_metadata(manifest,args):
    import subprocess
    p=subprocess.run(['nvidia-smi',f'--id={os.environ["CUDA_VISIBLE_DEVICES"]}',
                      '--query-gpu=uuid,name','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15)
    return {'canonical_scientific_identity':manifest['identity'],
            'subset':{'cache_condition':args.condition,'seed':args.seed},
            'runtime_physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'gpu':p.stdout.strip(),
            'runtime_pid':os.getpid(),'allocator_memory_fraction':args.memory_fraction,
            'original_manifest_physical_gpu':manifest['CUDA_VISIBLE_DEVICES'],
            'placement_override_authorized':True,
            'scientific_protocol_changed':False,'batch_size_changed':False,
            'execution_sources':{p.name:digest(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')},
            'time':time.strftime('%Y-%m-%dT%H:%M:%S%z')}


def run_shard(args):
    canonical=BASE/args.condition/'formal';manifest=read_json(canonical/'manifest.json')
    verify_sources(manifest);identity=manifest['identity'];config=deepcopy(manifest['protocol'])
    if config['cache_condition']!=args.condition or args.seed not in config['generation']['seeds']:
        raise RuntimeError('Shard selector outside frozen protocol')
    view=BASE/'shards'/args.condition/f'seed{args.seed}'
    view.mkdir(parents=True,exist_ok=True)
    with open(view/'.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # The only algorithm-control change is selecting a disjoint subset of the
        # EXISTING seed list. Every document RNG/decoding parameter stays identical.
        config['generation']['seeds']=[args.seed]
        configure_runtime(args.memory_fraction)
        metadata=execution_metadata(manifest,args)
        atomic_json(view/'execution.json',metadata)
        for folder in ('generation','evaluation'):
            for path in (canonical/folder).glob(f'seed{args.seed}_*.json'):
                link_checked(path,view/folder/path.name)
        samples=read_json(canonical/'samples.json')
        tokenizer=AutoTokenizer.from_pretrained(ROOT/config['model']['directory'],local_files_only=True,use_fast=True)
        expected=len(config['watermark']['deltas'])*math.ceil(len(samples)/config['generation']['batch_size'])
        retained=len(list((view/'generation').glob('*.json')))
        LOG.info('SHARD_START condition=%s seed=%d GPU=%s inherited_batches=%d expected=%d',args.condition,args.seed,os.environ['CUDA_VISIBLE_DEVICES'],retained,expected)
        stage_status(view,'initializing',pid=os.getpid(),expected_batches=expected)
        try:
            if retained<expected:
                model=AutoModelForSeq2SeqLM.from_pretrained(ROOT/config['model']['directory'],local_files_only=True,
                    torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
                generate_condition(model,tokenizer,config,samples,view,identity,smoke=False)
                del model;gc.collect();torch.cuda.empty_cache()
            counts=publish(view,canonical,args.seed,identity)
            if counts['generation']!=expected:raise RuntimeError('Missing shard generation batches')
            evaluate_all(config,samples,tokenizer,view,identity)
            counts=publish(view,canonical,args.seed,identity)
            if counts['evaluation']!=expected:raise RuntimeError('Missing shard evaluation batches')
            stage_status(view,'complete',pid=os.getpid(),seed=args.seed,cache_condition=args.condition,
                         record_count=len(samples)*len(config['watermark']['deltas']),batches=counts)
            LOG.info('SHARD_COMPLETE condition=%s seed=%d records=%d',args.condition,args.seed,len(samples)*len(config['watermark']['deltas']))
        except Exception:
            import traceback
            stage_status(view,'failed',traceback=traceback.format_exc())
            raise
        finally:
            # Publish completed atomic files even on interruption/error; ignore .tmp.
            publish(view,canonical,args.seed,identity)


def approx_equal(left,right):
    if isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:approx_equal(left[key],right[key])
    elif isinstance(left,list):
        assert len(left)==len(right)
        for a,b in zip(left,right):approx_equal(a,b)
    elif isinstance(left,float):
        assert math.isclose(left,right,rel_tol=1e-5,abs_tol=2e-5),(left,right)
    else:assert left==right,(left,right)


def preflight(args):
    """One reference-generation and one evaluation check for the new scheduling path."""
    out=BASE/'execution/preflight';out.mkdir(parents=True,exist_ok=True)
    m=read_json(BASE/'full/formal/manifest.json');verify_sources(m);config=m['protocol']
    configure_runtime(args.memory_fraction)
    tok=AutoTokenizer.from_pretrained(ROOT/config['model']['directory'],local_files_only=True)
    reference=read_json(BASE/'full/formal/generation/seed42_delta0_batch00000.json')
    lookup={s['id']:s for s in read_json(BASE/'full/formal/samples.json')}
    samples=[lookup[i] for i in reference['sample_ids']]
    model=AutoModelForSeq2SeqLM.from_pretrained(ROOT/config['model']['directory'],local_files_only=True,
                     torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
    generated,audit=decode_cache_batch(model,tok,samples,config,42,0.0)
    assert [r['raw_ids'] for r in generated]==[r['raw_ids'] for r in reference['records']], 'Cross-GPU reference tokens changed'
    del model;gc.collect();torch.cuda.empty_cache()
    # Evaluate an existing smoke batch in an isolated output view; no formal overwrite.
    old=BASE/'full/smoke';sm=read_json(old/'manifest.json');samples=read_json(old/'samples.json')
    name='seed42_delta0_batch00000.json'
    link_checked(old/'generation'/name,out/'generation'/name)
    evaluate_all(sm['protocol'],samples,tok,out,sm['identity'])
    approx_equal(read_json(out/'evaluation'/name)['records'],read_json(old/'evaluation'/name)['records'])
    result={'passed':True,'exact_generated_token_matches':len(generated),'evaluated_reference_batch_matches':True,
            'nli_numeric_atol':2e-5,'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],
            'reference_generation_identity':m['identity'],'purpose':'engineering_check_not_research_evidence'}
    atomic_json(out/'checks.json',result);LOG.info('SHARD_PREFLIGHT_PASS %s',result)


def finalize():
    # Reuse the original statistical pipeline after every canonical checkpoint is
    # present. GPU 2 visibility preserves the original manifest identity; no new
    # generation/evaluation should be necessary at this stage.
    from runner import run_phase
    config=load_protocol(ROOT/'kvstudy/protocol.json')
    for case in config['cache']['conditions']:
        out=BASE/case/'formal';m=read_json(out/'manifest.json');verify_sources(m)
        expected=len(config['generation']['seeds'])*len(config['watermark']['deltas'])*math.ceil(700/config['generation']['batch_size'])
        for folder in ('generation','evaluation'):
            assert len(list((out/folder).glob('*.json')))==expected,(case,folder)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    tok=AutoTokenizer.from_pretrained(ROOT/config['model']['directory'],local_files_only=True)
    run_phase(config,'formal',tok)
    stage_status(BASE,'complete',record_count=16800,human_validation_pending=True,execution='multi_gpu_shards')
    LOG.info('FLEET_FINALIZE_COMPLETE records=16800')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--condition',choices=['full','cross-half','self-half','both-half'])
    parser.add_argument('--seed',type=int);parser.add_argument('--memory-fraction',type=float,default=.16)
    parser.add_argument('--preflight',action='store_true');parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True)
    if args.preflight:preflight(args)
    elif args.finalize:finalize()
    else:
        if args.condition is None or args.seed is None:parser.error('condition and seed required')
        run_shard(args)
