"""Run generation -> NLI evaluation -> analysis, resuming completed atomic batches."""
import argparse
import fcntl
import json
import os
import sys
import traceback
import re
import torch
from transformers import AutoTokenizer
from common import ROOT,LOG,setup_logging,read_json,load_protocol,run_manifest,lock_manifest,make_samples,stage_status
from generation import generate_all
from evaluation import evaluate_all
from analysis import analyze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','formal'],required=True)
    parser.add_argument('--protocol',default='protocol.json')
    parser.add_argument('--run-name',default=None)
    args=parser.parse_args();setup_logging()
    config=load_protocol(ROOT/args.protocol)
    if args.run_name is not None and not re.fullmatch(r'[a-z0-9][a-z0-9-]*',args.run_name):
        raise ValueError('Run name must be a simple lowercase identifier')
    if config.get('experiment_id') and config['experiment_id']!=args.run_name:
        raise ValueError('Protocol experiment_id must match the explicit run name')
    out=ROOT/'runs'/args.run_name/args.mode if args.run_name else ROOT/'runs'/args.mode
    out.mkdir(parents=True,exist_ok=True)
    with open(out/'.lock','w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This run already has an active writer')
        torch.set_num_threads(config['generation']['cpu_threads']);torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        # Per-process cap is explicit for concurrently scheduled supplementary runs.
        resources=config.get('resources',{})
        memory_fraction=resources.get('memory_fraction',.70)
        if resources and os.environ.get('CUDA_VISIBLE_DEVICES')!=resources['physical_gpu']:
            raise ValueError('Physical GPU assignment does not match the frozen resource plan')
        if not 0<memory_fraction<=.70:raise ValueError('Unsafe GPU memory cap')
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        identity=lock_manifest(out,run_manifest(config,args.mode))
        LOG.info('RUN name=%s mode=%s identity=%s PID=%d CUDA_VISIBLE_DEVICES=%s memory_fraction=%.2f',args.run_name,args.mode,identity,os.getpid(),os.environ.get('CUDA_VISIBLE_DEVICES'),memory_fraction)
        stage_status(out,'initializing',identity=identity,pid=os.getpid())
        try:
            tok=AutoTokenizer.from_pretrained(ROOT/config['model']['directory'],local_files_only=True,use_fast=True)
            samples=make_samples(config,out,tok,smoke=args.mode=='smoke')
            generate_all(config,samples,tok,out,identity,smoke=args.mode=='smoke')
            evaluate_all(config,samples,tok,out,identity)
            result=analyze(config,out,identity,smoke=args.mode=='smoke')
            if args.mode=='smoke':
                # Engineering gates only: no data-dependent tuning of the hypothesis.
                for row in result['table']:
                    if row['token_alignment_failures'] or row['NLI_unscored_sentences']:
                        raise RuntimeError('Smoke contains unaligned/unscored sentences; fix before formal run')
            stage_status(out,'complete',identity=identity,record_count=result['record_count'],human_validation_pending=args.mode=='formal')
            LOG.info('COMPLETE mode=%s records=%d',args.mode,result['record_count'])
        except Exception:
            stage_status(out,'failed',identity=identity,traceback=traceback.format_exc())
            LOG.exception('FAILED (completed checkpoints retained; deterministic errors are not blindly retried)')
            raise


if __name__=='__main__':main()
