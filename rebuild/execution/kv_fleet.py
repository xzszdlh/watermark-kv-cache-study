"""Multi-GPU KV shard dispatcher. Preserves checkpoints and frozen science code.
Uses idle capacity, explicit per-process reservations, disjoint seed file names,
and exclusive ownership. Does not add or change scientific experiment settings.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'kvstudy'))
import argparse
import fcntl
import hashlib
import logging
import os
import signal
import subprocess
import time
from common import read_json,atomic_json,stage_status,digest
from kv_shard_worker import verify_sources

BASE=ROOT/'runs/kv-cache';EXEC=BASE/'execution';LOG=logging.getLogger('fleet')
CASES=('full','cross-half','self-half','both-half');SEEDS=(42,43,44)
CAP=.16;CONTEXT_MIB=1024;RESERVE_MIB=6144
ENV={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','PYTHONUNBUFFERED':'1',
     'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
STOP=False;CHILDREN={}


def signal_stop(signum,frame):
    global STOP
    LOG.warning('Signal %d; stopping registered children only',signum);STOP=True


def gpu_inventory():
    p=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu',
                      '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    infos={}
    for line in p.stdout.strip().splitlines():
        i,uuid,total,used,free,util=[s.strip() for s in line.split(',')]
        infos[i]={'uuid':uuid,'total':int(total),'used':int(used),'free':int(free),'util':int(util),
                  'fleet_used':0,'other_study_processes':0}
    p=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],
                     capture_output=True,text=True,check=True,timeout=15)
    byuuid={v['uuid']:v for v in infos.values()};ours={j['proc'].pid for j in CHILDREN.values()}
    for line in p.stdout.strip().splitlines():
        if not line.strip():continue
        uuid,pid,mem=[s.strip() for s in line.split(',')];pid=int(pid)
        if uuid not in byuuid:continue
        info=byuuid[uuid]
        if pid in ours:
            try:info['fleet_used']+=int(mem)
            except ValueError:pass
        else:
            try:
                proc=Path(f'/proc/{pid}')
                cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ')
                if (proc/'cwd').resolve()==ROOT and b'study.py' in cmd:info['other_study_processes']+=1
            except (OSError,PermissionError):pass
    return infos


def counts(case,seed):
    canonical=BASE/case/'formal';view=BASE/'shards'/case/f'seed{seed}'
    result={}
    for folder in ('generation','evaluation'):
        names={p.name for root in (canonical,view) for p in (root/folder).glob(f'seed{seed}_*.json')}
        result[folder]=len(names)
    return result


def stop_serial(pid):
    proc=Path(f'/proc/{pid}')
    if proc.exists():
        cmd=(proc/'cmdline').read_bytes().split(b'\0')
        if b'kvstudy/runner.py' not in cmd or (proc/'cwd').resolve()!=ROOT:
            raise RuntimeError('Refusing to stop a PID that is not the verified old KV worker')
        LOG.info('TAKEOVER stopping only prior KV worker PID=%d; keeping all committed checkpoints',pid)
        os.kill(pid,signal.SIGTERM)
        deadline=time.monotonic()+20
        while proc.exists() and time.monotonic()<deadline:time.sleep(.2)
        if proc.exists():raise RuntimeError('Old KV worker did not exit; no concurrent writer will be started')


def preserve_before_index():
    path=EXEC/'pre_takeover_checkpoints.json'
    if path.exists():return read_json(path)
    files={}
    for case in CASES:
        manifest=read_json(BASE/case/'formal/manifest.json');verify_sources(manifest)
        for folder in ('generation','evaluation'):
            for p in (BASE/case/'formal'/folder).glob('*.json'):
                files[p.relative_to(BASE).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    atomic_json(path,files);LOG.info('PRESERVED %d pre-existing checkpoint hashes',len(files))
    return files


def admission(infos):
    candidates=[]
    for gpu,info in infos.items():
        active=sum(j['gpu']==gpu for j in CHILDREN.values())
        limit=1 if gpu=='3' else 3
        # Count the ongoing independent sensitivity job toward the compute slot cap.
        if active+info['other_study_processes']>=limit:continue
        reserved_per_process=CAP*info['total']+CONTEXT_MIB
        # Add back known fleet allocations, then reserve each existing fleet slot
        # in FULL. This avoids launching many children before CUDA allocation appears.
        unreserved=info['free']+info['fleet_used']-active*reserved_per_process-RESERVE_MIB
        if unreserved<reserved_per_process:continue
        if info['free']<reserved_per_process+RESERVE_MIB:continue
        candidates.append((active+info['other_study_processes'],info['util'],-unreserved,gpu))
    return min(candidates)[-1] if candidates else None


def start_shard(key,gpu):
    case,seed=key;name=f'{case}-seed{seed}'
    logpath=ROOT/'logs'/f'kv-{name}.log';log=open(logpath,'a',encoding='utf-8',buffering=1)
    cmd=[sys.executable,'-u',str(Path(__file__).parent/'kv_shard_worker.py'),
         '--condition',case,'--seed',str(seed),'--memory-fraction',str(CAP)]
    log.write(f'\n=== {time.strftime("%Y-%m-%dT%H:%M:%S%z")} GPU={gpu} shard={name} ===\n');log.flush()
    p=subprocess.Popen(cmd,cwd=ROOT,env={**ENV,'CUDA_VISIBLE_DEVICES':gpu},stdin=subprocess.DEVNULL,
                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    CHILDREN[key]={'proc':p,'gpu':gpu,'log':log,'log_path':str(logpath),'started':time.monotonic()}
    LOG.info('START shard=%s GPU=%s PID=%d inherited=%s',name,gpu,p.pid,counts(case,seed))


def progress(pending,finished,failed):
    jobs={};totals={'generation':0,'evaluation':0}
    for case in CASES:
        for seed in SEEDS:
            key=(case,seed);name=f'{case}-seed{seed}';c=counts(case,seed)
            for f in totals:totals[f]+=c[f]
            p=BASE/'shards'/case/f'seed{seed}'/'status.json'
            status=read_json(p) if p.exists() else None
            running=CHILDREN.get(key)
            jobs[name]={'counts':c,'expected_batches_each_phase':44,'worker_status':status,
                        'state':'running' if running else 'failed' if key in failed else 'complete' if key in finished else 'queued',
                        'pid':running['proc'].pid if running else None,'gpu':running['gpu'] if running else None}
    atomic_json(EXEC/'fleet_status.json',{'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'pid':os.getpid(),
                'active':len(CHILDREN),'queued':len(pending),'failed':list(map(str,failed)),
                'generation_batches':totals['generation'],'evaluation_batches':totals['evaluation'],
                'expected_batches_each_phase':528,'jobs':jobs})
    stage_status(BASE,'parallel_shards',active_workers=len(CHILDREN),pending_shards=len(pending),
                 generation_batches=totals['generation'],evaluation_batches=totals['evaluation'],
                 expected_batches_each_phase=528)
    LOG.info('FLEET_PROGRESS active=%d queued=%d completed_shards=%d gen=%d/528 eval=%d/528 failed=%d',
             len(CHILDREN),len(pending),len(finished),totals['generation'],totals['evaluation'],len(failed))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--takeover-pid',type=int)
    args=parser.parse_args();os.chdir(ROOT);EXEC.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
       handlers=[logging.FileHandler(ROOT/'logs/kv-fleet.log',encoding='utf-8'),logging.StreamHandler()],force=True)
    with open(EXEC/'.fleet.lock','w') as ownlock:
        fcntl.flock(ownlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check=read_json(EXEC/'preflight/checks.json')
        if not check.get('passed'):raise RuntimeError('Placement preflight has not passed')
        if args.takeover_pid:stop_serial(args.takeover_pid)
        with open(ROOT/'kvstudy/.worker.lock','a') as seriallock:
            fcntl.flock(seriallock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            before=preserve_before_index()
            atomic_json(EXEC/'fleet.pid.json',{'pid':os.getpid()})
            source_hashes={p.name:digest(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')}
            for p in Path(__file__).parent.glob('*.py'):
                dst=EXEC/'source'/p.name;dst.parent.mkdir(parents=True,exist_ok=True)
                if dst.exists() and digest(dst.read_bytes())!=source_hashes[p.name]:raise RuntimeError('Execution source changed; explicit migration required')
                dst.write_bytes(p.read_bytes())
            atomic_json(EXEC/'placement_plan.json',{'scientific_settings_unchanged':True,
                'same_batch_size':32,'shards':12,'allocator_cap_fraction':CAP,'context_allowance_MiB':CONTEXT_MIB,
                'free_memory_reserve_MiB':RESERVE_MIB,'max_total_project_workers_per_gpu':{'0':3,'1':3,'2':3,'3':1},
                'original_manifests_retained':True,'per_shard_actual_GPU_in':'shards/<condition>/seed<seed>/execution.json',
                'execution_sources':source_hashes,'completed_checkpoint_preservation_count':len(before)})
            for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal_stop)
            pending=[];finished=set();failed={}
            for case in CASES:
                for seed in SEEDS:
                    if counts(case,seed)=={'generation':44,'evaluation':44}:finished.add((case,seed))
                    else:pending.append((case,seed))
            # Prioritize the largest missing workloads rather than finishing one
            # complete condition before starting another.
            pending.sort(key=lambda key:sum(counts(*key).values()))
            last=0
            try:
                while pending or CHILDREN:
                    if STOP:raise InterruptedError('Fleet stop requested')
                    for key,job in list(CHILDREN.items()):
                        code=job['proc'].poll()
                        if code is not None:
                            job['log'].close();del CHILDREN[key]
                            if code==0 and counts(*key)=={'generation':44,'evaluation':44}:
                                finished.add(key);LOG.info('DONE shard=%s',key)
                            else:failed[key]=code;LOG.error('FAILED shard=%s exit=%s log=%s',key,code,job['log_path'])
                    infos=gpu_inventory()
                    while pending:
                        gpu=admission(infos)
                        if gpu is None:break
                        start_shard(pending.pop(0),gpu)
                        # reservations use CHILDREN, so no need to wait for CUDA to appear
                    if time.monotonic()-last>=20:
                        progress(pending,finished,failed)
                        LOG.info('GPU_SNAPSHOT %s',infos);last=time.monotonic()
                    time.sleep(3)
                progress(pending,finished,failed)
                if failed:raise RuntimeError(f'{len(failed)} shards failed; completed results retained')
                for relative,expected in before.items():
                    if hashlib.sha256((BASE/relative).read_bytes()).hexdigest()!=expected:
                        raise RuntimeError(f'Pre-existing checkpoint changed: {relative}')
                LOG.info('ALL_SHARDS_COMPLETE; all %d pre-existing checkpoints preserved byte-for-byte',len(before))
                stage_status(BASE,'analysis',execution='multi_gpu_shards')
                with open(ROOT/'logs/kv-finalize.log','a',encoding='utf-8') as log:
                    proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).parent/'kv_shard_worker.py'),'--finalize'],
                        cwd=ROOT,env={**ENV,'CUDA_VISIBLE_DEVICES':'2'},stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    while proc.poll() is None:
                        if STOP:
                            os.killpg(proc.pid,signal.SIGTERM);proc.wait();raise InterruptedError()
                        time.sleep(3)
                    if proc.returncode:raise RuntimeError('Final analysis failed; see logs/kv-finalize.log')
                LOG.info('FLEET_COMPLETE; automatic results complete, human validation still pending')
            finally:
                for job in CHILDREN.values():
                    if job['proc'].poll() is None:
                        os.killpg(job['proc'].pid,signal.SIGTERM)
                        try:job['proc'].wait(timeout=20)
                        except subprocess.TimeoutExpired:os.killpg(job['proc'].pid,signal.SIGKILL);job['proc'].wait()
                    job['log'].close()


if __name__=='__main__':main()
