"""Single-job supervisor. Real-time merged logging, phase heartbeats and safe resume.
Download retry/resume is in prepare.py. Computation failures stop for diagnosis;
restart this supervisor to continue already committed batches, not from scratch.
"""
import fcntl
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parent
LOG=logging.getLogger('supervisor')
child=None
stop=threading.Event()


def heartbeat():
    while not stop.wait(30):
        if child is not None:
            LOG.info('HEARTBEAT child_pid=%d process_alive=%s (not a success assertion)',child.pid,child.poll() is None)


def terminate(signum,frame):
    stop.set()
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
    raise SystemExit(128+signum)


def run(cmd,env):
    global child
    LOG.info('START command=%s',cmd)
    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,text=True,encoding='utf-8',bufsize=1,start_new_session=True)
    (ROOT/'child.pid').write_text(str(child.pid),encoding='utf-8')
    for line in child.stdout:
        LOG.info('%s',line.rstrip())
    code=child.wait();LOG.info('EXIT code=%d',code)
    child=None
    if code:raise RuntimeError(f'Phase failed with exit {code}; check preceding traceback. Checkpoints preserved.')


def wait_downloads():
    c=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    started=time.monotonic()
    while True:
        ready=all((ROOT/c[k]['directory']/'verified_files.json').exists() for k in ('model','nli'))
        if ready:return
        state=[]
        for k in ('model','nli'):
            p=ROOT/c[k]['directory']
            size=sum(f.stat().st_size for f in p.glob('*') if f.is_file()) if p.exists() else 0
            state.append(f'{k}={size/2**20:.1f}MiB verified={(p/"verified_files.json").exists()}')
        LOG.info('WAIT_DOWNLOAD %s; detailed byte/speed/retry log: logs/download.log',' '.join(state))
        if time.monotonic()-started>12*3600:
            raise TimeoutError('Downloads not ready after 12 hours; inspect logs/download.log')
        time.sleep(30)


def main():
    os.chdir(ROOT);(ROOT/'logs').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(ROOT/'logs/run.log',encoding='utf-8'),logging.StreamHandler()],force=True)
    with open(ROOT/'.supervisor.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (ROOT/'supervisor.pid').write_text(str(os.getpid()),encoding='utf-8')
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,terminate)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'2','HF_ENDPOINT':'https://hf-mirror.com',
             'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
             'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
        LOG.info('SUPERVISOR PID=%d Python=%s GPU physical index=2',os.getpid(),sys.executable)
        threading.Thread(target=heartbeat,daemon=True).start()
        try:
            run([sys.executable,'-u','test_correctness.py'],env)
            wait_downloads()
            run([sys.executable,'-u','study.py','--mode','smoke'],env)
            # Verify resume path: second invocation should skip every committed batch.
            run([sys.executable,'-u','study.py','--mode','smoke'],env)
            run([sys.executable,'-u','study.py','--mode','formal'],env)
            LOG.info('PIPELINE_COMPLETE automatic results in runs/formal; human audit still pending')
        except Exception:
            LOG.exception('PIPELINE_STOPPED: do not label incomplete/failed work as successful')
            raise
        finally:stop.set()


if __name__=='__main__':main()
