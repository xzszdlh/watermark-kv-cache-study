#!/usr/bin/env python3
"""Bounded multi-GPU sensitivity jobs. Never terminate or reconfigure other workloads.
Each job runs its own smoke gate before formal generation/evaluation. Child output
is unbuffered, separately logged, and resumes per-batch on an explicit restart.
"""
import concurrent.futures
import fcntl
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from common import ROOT,load_protocol,read_json,atomic_json,digest

LOG=logging.getLogger('parallel')
NAMES=('nucleus-128','topk-128','nucleus-256')
ENV={**os.environ,'HF_ENDPOINT':'https://hf-mirror.com','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
     'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
children={};states={};mutex=threading.Lock();stop=threading.Event()


def gpu_info(index):
    result=subprocess.run(['nvidia-smi',f'--id={index}',
                           '--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu',
                           '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    i,uuid,total,used,free,util=[s.strip() for s in result.stdout.strip().split(',')]
    return {'index':i,'uuid':uuid,'total_MiB':int(total),'used_MiB':int(used),'free_MiB':int(free),'util_percent':int(util)}


def state(name,**kwargs):
    with mutex:
        states.setdefault(name,{}).update(kwargs)
        atomic_json(ROOT/'runs/parallel_status.json',{'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                                    'pid':os.getpid(),'jobs':states})


def wait_capacity(name,cfg):
    index=cfg['resources']['physical_gpu'];fraction=cfg['resources']['memory_fraction']
    while not stop.is_set():
        info=gpu_info(index)
        # Keep 4 GiB beyond the task's allocator cap for service growth/context overhead.
        required=info['total_MiB']*fraction+4096
        if info['free_MiB']>=required:
            LOG.info('CAPACITY_OK name=%s GPU=%s free=%dMiB allocator_cap=%.0fMiB additional_reserve=4096MiB',name,index,info['free_MiB'],info['total_MiB']*fraction)
            return info
        state(name,stage='waiting_for_capacity',gpu=info)
        LOG.info('WAIT_CAPACITY name=%s GPU=%s free=%dMiB required=%.0fMiB; not touching other processes',name,index,info['free_MiB'],required)
        stop.wait(30)
    raise InterruptedError('Supervisor stopping')


def run_process(name,cmd,env,logpath,stage):
    with open(logpath,'a',encoding='utf-8',buffering=1) as log:
        log.write(f'\n=== {time.strftime("%Y-%m-%dT%H:%M:%S%z")} stage={stage} cmd={cmd!r} ===\n');log.flush()
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,
                           stderr=subprocess.STDOUT,start_new_session=True)
        with mutex:children[name]=p
        atomic_json(ROOT/'logs'/f'{name}.pid.json',{'pid':p.pid,'stage':stage,'command':cmd})
        state(name,stage=stage,pid=p.pid,log=str(logpath))
        LOG.info('START name=%s stage=%s pid=%d log=%s',name,stage,p.pid,logpath)
        try:
            while p.poll() is None:
                if stop.wait(2):
                    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid,signal.SIGKILL);p.wait()
                    raise InterruptedError('Interrupted this supervisor-owned child only')
            code=p.returncode
            if code:raise RuntimeError(f'{name}/{stage} exit={code}; inspect {logpath}')
        finally:
            with mutex:children.pop(name,None)
        LOG.info('EXIT_OK name=%s stage=%s pid=%d',name,stage,p.pid)
        state(name,stage=stage+'_finished',pid=None)


def run_job(name):
    try:
        cfg=load_protocol(ROOT/'protocols'/f'{name}.json')
        index=cfg['resources']['physical_gpu'];env={**ENV,'CUDA_VISIBLE_DEVICES':index}
        state(name,stage='queued',physical_gpu=index,protocol=cfg)
        for mode in ('smoke','formal'):
            info=wait_capacity(name,cfg)
            state(name,admission_gpu=info)
            cmd=[sys.executable,'-u','study.py','--mode',mode,'--protocol',f'protocols/{name}.json','--run-name',name]
            run_process(name,cmd,env,ROOT/'logs'/f'{name}.log',mode)
        status=read_json(ROOT/'runs'/name/'formal/status.json')
        if status['stage']!='complete':raise RuntimeError('Exit without completed status')
        state(name,stage='complete',record_count=status['record_count'],human_validation_pending=True)
    except Exception as exc:
        LOG.exception('JOB_FAILED name=%s (other independent jobs continue)',name)
        state(name,stage='failed',error=str(exc))
        return False
    return True


def monitor():
    while not stop.wait(30):
        with mutex:active=list(children.items())
        for name,child in active:
            if child.poll() is not None:continue
            phase=states.get(name,{}).get('stage')
            p=ROOT/'runs'/name/str(phase)/'status.json'
            status=read_json(p) if p.exists() else {'stage':'initializing'}
            LOG.info('PROGRESS name=%s pid=%d status=%s',name,child.pid,status)
        for index in ('0','2','3'):
            try:LOG.info('GPU %s',gpu_info(index))
            except Exception as exc:LOG.warning('GPU monitor unavailable: %s',exc)


def stop_signal(signum,frame):
    LOG.warning('Received signal %d; stop only registered child process groups',signum)
    stop.set()


def main():
    os.chdir(ROOT);(ROOT/'logs').mkdir(exist_ok=True);(ROOT/'runs').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(ROOT/'logs/parallel.log',encoding='utf-8')],force=True)
    with open(ROOT/'.parallel.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(ROOT/'parallel.pid.json',{'pid':os.getpid()})
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop_signal)
        for name in NAMES:
            cfg=load_protocol(ROOT/'protocols'/f'{name}.json')
            LOG.info('FROZEN name=%s protocol_hash=%s GPU=%s',name,digest(cfg),cfg['resources']['physical_gpu'])
        testenv={**ENV,'CUDA_VISIBLE_DEVICES':'0'}
        run_process('correctness',[sys.executable,'-u','test_correctness.py'],testenv,ROOT/'logs/sensitivity-tests.log','tests')
        threading.Thread(target=monitor,daemon=True).start()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            outcomes=list(pool.map(run_job,NAMES))
        stop.set()
        LOG.info('PARALLEL_FINISHED statuses=%s',dict(zip(NAMES,outcomes)))
        if all(outcomes):
            # Summarization uses saved aggregates; it does not select a best setting.
            summary={name:read_json(ROOT/'runs'/name/'formal/analysis.json') for name in NAMES}
            summary['unrestricted-128']=read_json(ROOT/'runs/formal/analysis.json')
            atomic_json(ROOT/'runs/sensitivity_comparison.json',{
                'status':'all_automatic_runs_complete_human_validation_pending',
                'warning':'Supplementary settings were fixed after the initial run, not an independent confirmatory test set.',
                'runs':summary})
        else:raise SystemExit(1)


if __name__=='__main__':main()
