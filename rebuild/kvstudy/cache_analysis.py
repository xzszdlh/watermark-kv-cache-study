"""Paired cache contrasts on fixed documents; not a reproduction of a named method."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from collections import defaultdict
import numpy as np
from common import read_json,atomic_json,LOG
from analysis import ci_mean


def compact_records(directory):
    rows={}
    for p in (directory/'generation').glob('*.json'):
        for r in read_json(p)['records']:
            if r['role']!='test':continue
            last=r['cache_at_sequence_end']
            rows[(r['id'],r['seed'],r['delta'])]={
                'z':r['watermark_text']['z'],'length':r['n_content_tokens'],'capped':r['hit_token_limit'],
                'cache_ratio_at_own_final_step':last['KV_bytes_batch']/last['full_KV_bytes_batch_at_same_step'],
                'retained_source_fraction':len(r['cross_retained_source_positions'])/r['source_valid_tokens']}
    for p in (directory/'evaluation').glob('*.json'):
        for r in read_json(p)['records']:
            key=(r['id'],r['seed'],r['delta'])
            if key not in rows:continue
            vals=[s['nli']['source_support_proxy'] for s in r['sentences'] if s['nli'] is not None]
            rows[key].update(rougeL=r['rouge']['rougeL'],support=float(np.mean(vals)) if vals else None)
    return rows


def compare_cache_conditions(config,root,mode):
    conditions=config['cache']['conditions'];seeds=config['generation']['seeds'][:1] if mode=='smoke' else config['generation']['seeds']
    data={c:compact_records(root/c/mode) for c in conditions}
    for c in conditions:assert data[c].keys()==data['full'].keys(),'Condition pairing mismatch'
    full_summary=read_json(root/'full'/mode/'analysis.json')
    thresholds={int(s):t for s,t in full_summary['thresholds'].items()}
    contrasts=[];rng=np.random.default_rng(20261104);reps=50 if mode=='smoke' else 1000
    for seed in seeds:
        for delta in config['watermark']['deltas']:
            keys=[k for k in data['full'] if k[1]==seed and k[2]==delta]
            for condition in conditions:
                if condition=='full':continue
                differences={metric:[] for metric in ('z','rougeL','support','length')}
                for k in keys:
                    for metric in differences:
                        a=data['full'][k].get(metric);b=data[condition][k].get(metric)
                        if a is not None and b is not None:differences[metric].append(b-a)
                rows=[data[condition][k] for k in keys]
                base=[data['full'][k] for k in keys]
                common_threshold=thresholds[seed]
                contrasts.append({'seed':seed,'delta':delta,'condition':condition,'reference':'full','n_documents':len(keys),
                                  'paired_changes':{m:{'mean':float(np.mean(v)) if v else None,'document_bootstrap_95ci':ci_mean(v,reps,rng),'n':len(v)} for m,v in differences.items()},
                                  'full_calibrated_common_threshold':common_threshold,
                                  'condition_detected_with_full_threshold':float(np.mean([r['z']>common_threshold for r in rows if r['z'] is not None])),
                                  'full_detected_with_same_threshold':float(np.mean([r['z']>common_threshold for r in base if r['z'] is not None])),
                                  'condition_fraction_hitting_length_cap':float(np.mean([r['capped'] for r in rows])),
                                  'mean_retained_source_fraction':float(np.mean([r['retained_source_fraction'] for r in rows])),
                                  'mean_cache_bytes_fraction_at_own_final_step':float(np.mean([r['cache_ratio_at_own_final_step'] for r in rows]))})
    output={'mode':mode,'study':'BART compartment-wise cache intervention',
            'status':'smoke_only' if mode=='smoke' else 'automatic_complete_human_validation_pending',
            'contrasts':contrasts,
            'limits':['50% compartment retention does not imply equal total-memory budgets.',
                      'Full source is encoded before cross-cache eviction; retained states may encode discarded positions.',
                      'Evaluation uses full source visible to encoder, not retained cache positions.',
                      'NLI remains an automatic support proxy, not verified hallucination labels.',
                      'Shared GPU runtimes are operational timings, not controlled efficiency benchmarks.',
                      'No decoder-only retrieval-head drift or HalluKV replication is claimed.']}
    atomic_json(root/f'cache_comparison_{mode}.json',output)
    LOG.info('CACHE_COMPARISON_READY mode=%s contrasts=%d',mode,len(contrasts))
