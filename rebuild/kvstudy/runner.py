"""Single bounded GPU-2 cache-study worker, isolated from ongoing sensitivity jobs."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import fcntl
import gc
import logging
import os
import subprocess
import time
import traceback
from copy import deepcopy
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from common import ROOT,LOG,read_json,load_protocol,run_manifest,lock_manifest,make_samples,atomic_json,stage_status,digest
from evaluation import evaluate_all
from analysis import analyze
from kvgeneration import generate_condition
from test_cache import real_checkpoint_check
from cache_analysis import compare_cache_conditions


def wait_capacity(config):
    fraction=config['resources']['memory_fraction'];index=config['resources']['physical_gpu']
    while True:
        p=subprocess.run(['nvidia-smi',f'--id={index}','--query-gpu=memory.total,memory.free','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,check=True,timeout=15)
        total,free=map(int,p.stdout.strip().split(','))
        needed=total*fraction+config['resources']['admission_extra_reserve_MiB']
        if free>=needed:
            LOG.info('KV_ADMISSION GPU=%s free=%dMiB allocator_cap=%.0fMiB reserve=%dMiB',index,free,total*fraction,config['resources']['admission_extra_reserve_MiB'])
            return
        LOG.info('KV_WAIT_CAPACITY free=%dMiB need=%.0fMiB; other jobs untouched',free,needed)
        time.sleep(30)


def freeze(config,condition,mode):
    c=deepcopy(config);c['cache_condition']=condition
    c['experiment_id']='kv-cache-'+condition
    out=ROOT/'runs/kv-cache'/condition/mode
    manifest=run_manifest(c,mode)
    for path in sorted(HERE.glob('*.py')):
        manifest['sources'][path.relative_to(ROOT).as_posix()]=digest(path.read_bytes())
    identity=lock_manifest(out,manifest)
    return c,out,identity


def run_phase(config,mode,tokenizer):
    root=ROOT/'runs/kv-cache';cases={}
    for condition in config['cache']['conditions']:
        c,out,identity=freeze(config,condition,mode)
        samples=make_samples(c,out,tokenizer,smoke=mode=='smoke')
        cases[condition]=(c,out,identity,samples)
    generation_needed=False
    for c,out,identity,samples in cases.values():
        nseeds=1 if mode=='smoke' else len(c['generation']['seeds'])
        expected=nseeds*len(c['watermark']['deltas'])*((len(samples)+c['generation']['batch_size']-1)//c['generation']['batch_size'])
        generation_needed |= len(list((out/'generation').glob('*.json')))<expected
    if generation_needed:
        model=AutoModelForSeq2SeqLM.from_pretrained(ROOT/config['model']['directory'],local_files_only=True,
                                                  torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
        if mode=='smoke':
            sample=next(iter(cases.values()))[-1]
            checks=real_checkpoint_check(model,tokenizer,sample,config,root)
            LOG.info('KV_REAL_CHECKPOINT_CHECKS_PASS count=%d',len(checks))
        for condition,(c,out,identity,samples) in cases.items():
            stage_status(root,'generation',mode=mode,condition=condition,pid=os.getpid())
            generate_condition(model,tokenizer,c,samples,out,identity,smoke=mode=='smoke')
        del model;gc.collect();torch.cuda.empty_cache()
    for condition,(c,out,identity,samples) in cases.items():
        stage_status(root,'evaluation',mode=mode,condition=condition,pid=os.getpid())
        evaluate_all(c,samples,tokenizer,out,identity)
        result=analyze(c,out,identity,smoke=mode=='smoke')
        # The shared analyzer was originally written for no-cache-intervention runs.
        # Scope its metadata to this actual experiment without changing any values.
        result['cache_condition']=condition
        result['cache_protocol']=c['cache']
        result['interpretation_limits']=[v for v in result['interpretation_limits']
                                         if v!='No KV compression or causal claim is tested.']
        result['interpretation_limits'].extend([
            'Physical BART cross/self KV compaction is tested; no HalluKV or decoder-only head mechanism is reproduced.',
            'Cache compartments have different total sizes; equal retention ratios are not equal total-memory budgets.',
            'NLI association is not causal hallucination detection.'])
        atomic_json(out/'analysis.json',result)
        if mode=='smoke' and any(r['token_alignment_failures'] or r['NLI_unscored_sentences'] for r in result['table']):
            raise RuntimeError('Cache smoke has unresolved evaluation gaps')
        stage_status(out,'complete',mode=mode,identity=identity,record_count=result['record_count'],human_validation_pending=mode=='formal')
        LOG.info('KV_CONDITION_COMPLETE condition=%s mode=%s records=%d',condition,mode,result['record_count'])
    compare_cache_conditions(config,root,mode)


def main():
    os.chdir(ROOT);(ROOT/'logs').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(ROOT/'logs/kv-cache.log',encoding='utf-8'),logging.StreamHandler()],force=True)
    root=ROOT/'runs/kv-cache';root.mkdir(parents=True,exist_ok=True)
    with open(HERE/'.worker.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=load_protocol(HERE/'protocol.json')
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=config['resources']['physical_gpu']:
            raise RuntimeError('Must launch with CUDA_VISIBLE_DEVICES=2')
        atomic_json(root/'worker.pid.json',{'pid':os.getpid()})
        stage_status(root,'tests',pid=os.getpid())
        try:
            tests=subprocess.run([sys.executable,'-u',str(HERE/'test_cache.py')],capture_output=True,text=True,encoding='utf-8')
            LOG.info('KV_UNIT_TEST_OUTPUT\n%s\n%s',tests.stdout,tests.stderr)
            if tests.returncode:raise RuntimeError('KV unit tests failed')
            wait_capacity(config)
            torch.set_num_threads(config['generation']['cpu_threads']);torch.set_num_interop_threads(1)
            if not torch.cuda.is_available():raise RuntimeError('CUDA not available')
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.cuda.set_per_process_memory_fraction(config['resources']['memory_fraction'])
            tokenizer=AutoTokenizer.from_pretrained(ROOT/config['model']['directory'],local_files_only=True,use_fast=True)
            LOG.info('KV_STUDY_START config_hash=%s pid=%d GPU=2 conditions=%s',digest(config),os.getpid(),config['cache']['conditions'])
            run_phase(config,'smoke',tokenizer)
            LOG.info('KV_SMOKE_COMPLETE; now start full fixed protocol')
            run_phase(config,'formal',tokenizer)
            stage_status(root,'complete',record_count=16800,human_validation_pending=True)
            LOG.info('KV_STUDY_COMPLETE records=16800 human_validation_pending=True')
        except Exception:
            stage_status(root,'failed',traceback=traceback.format_exc())
            LOG.exception('KV_STUDY_STOPPED: prior checkpoints retained; no blind restart')
            raise


if __name__=='__main__':main()
