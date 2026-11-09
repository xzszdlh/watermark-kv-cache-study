"""Paired document-cluster statistics; seeds are repeated measures, not new documents.
Read-only inputs. All outputs are post-observation analyses, not preregistered tests.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import argparse
import hashlib
import json
import math
import numpy as np
from common import atomic_json,read_json,digest

METRICS=('support_proxy','rougeL','displayed_text_z','content_tokens',
         'detected_at_common_full_threshold','length_cap','mean_base_entropy')
CONTRASTS={'cross-full':{'cross':1,'full':-1}, 'self-full':{'self':1,'full':-1},
           'both-full':{'both':1,'full':-1},'cross-self':{'cross':1,'self':-1},
           'both-cross-self+full':{'both':1,'cross':-1,'self':-1,'full':1}}


def load_records(directory):
    manifest=read_json(directory/'manifest.json'); identity=manifest['identity']
    generated={};evaluated={};hashes={}
    for folder,target in (('generation',generated),('evaluation',evaluated)):
        for path in sorted((directory/folder).glob('*.json')):
            raw=path.read_bytes();data=json.loads(raw)
            hashes[f'{folder}/{path.name}']=hashlib.sha256(raw).hexdigest()
            if data['identity']!=identity:raise ValueError(f'Wrong checkpoint identity {path}')
            for r in data['records']:
                key=(r['id'],int(r['seed']),float(r['delta']))
                if key in target:raise ValueError(f'Duplicate measurement {key}')
                if folder=='generation':
                    trace=[t['base_entropy'] for t in r['trace'] if not t['special'] and not t['forced']]
                    end=r.get('cache_at_sequence_end')
                    cache_fraction=end['KV_bytes_batch']/end['full_KV_bytes_batch_at_same_step'] if end else None
                    target[key]={'role':r['role'],'displayed_text_z':r['watermark_text']['z'],
                                 'content_tokens':r['n_content_tokens'],'length_cap':float(r['hit_token_limit']),
                                 'mean_base_entropy':float(np.mean(trace)) if trace else None,
                                 'cache_fraction_at_own_final_step':cache_fraction}
                else:
                    values=[s['nli']['source_support_proxy'] for s in r['sentences'] if s['nli'] is not None]
                    target[key]={'rougeL':r['rouge']['rougeL'],'support_proxy':float(np.mean(values)) if values else None,
                                 'unscored_sentences':sum(s['nli'] is None for s in r['sentences'])}
    assert generated.keys()==evaluated.keys(),f'Generation/evaluation pairing mismatch {directory}'
    c=manifest['protocol'];expected=700*len(c['generation']['seeds'])*len(c['watermark']['deltas'])
    assert len(generated)==expected,(directory,len(generated),expected)
    for key,row in generated.items():row.update(evaluated[key])
    tests={k:r for k,r in generated.items() if r['role']=='test'}
    thresholds={}
    for seed in c['generation']['seeds']:
        values=sorted(r['displayed_text_z'] for k,r in generated.items() if r['role']=='calibration' and k[1]==seed and k[2]==0.)
        assert len(values)==200
        thresholds[seed]=values[math.ceil((1-c['evaluation']['fpr_target'])*len(values))-1]
    source={'directory':directory.relative_to(ROOT).as_posix(),'identity':identity,
            'checkpoint_index_sha256':digest(hashes),'checkpoint_count':len(hashes),'records':len(generated),
            'n_independent_test_documents':len({k[0] for k in tests}),'protocol':c}
    return tests,thresholds,source


def summarize_vector(values,indices,family=False):
    values=np.asarray(values,dtype=float)
    if not np.isfinite(values).all():raise ValueError('Missing/nonfinite outcome; report explicitly, do not silently drop')
    boot=values[indices].mean(axis=1)
    out={'mean':float(values.mean()),'document_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),
         'n_documents':len(values)}
    if family:
        tail=.05/(2*16)
        out['bonferroni_16_nominal_familywise_95ci']=np.quantile(boot,[tail,1-tail]).tolist()
    return out


def compare_four(directories,output,reps=20000):
    loaded={name:load_records(path) for name,path in directories.items()}
    data={n:v[0] for n,v in loaded.items()};thresholds=loaded['full'][1]
    keys=set(data['full'])
    assert all(set(v)==keys for v in data.values()),'Document/seed/delta mismatch'
    seeds=sorted({k[1] for k in keys});deltas=sorted({k[2] for k in keys});docs=sorted({k[0] for k in keys})
    assert len(docs)==500 and seeds==[42,43,44] and deltas==[0.,2.]
    for cond,rows in data.items():
        for key,row in rows.items():row['detected_at_common_full_threshold']=float(row['displayed_text_z']>thresholds[key[1]])
    rng=np.random.default_rng(202611042)
    indices=rng.integers(0,len(docs),(reps,len(docs)),dtype=np.int32)
    all_contrasts=[];absolute=[]
    for delta in deltas:
        tensors={name:np.array([[[rows[(doc,seed,delta)][metric] for metric in METRICS] for seed in seeds] for doc in docs],dtype=float)
                 for name,rows in data.items()}
        assert all(np.isfinite(v).all() for v in tensors.values())
        for name,tensor in tensors.items():
            absolute.append({'condition':name,'delta':delta,'mean':dict(zip(METRICS,tensor.mean(axis=(0,1)).tolist())),
               'mean_cache_fraction_at_own_final_step':float(np.mean([data[name][(doc,s,delta)]['cache_fraction_at_own_final_step'] for doc in docs for s in seeds]))})
        for contrast,coefficients in CONTRASTS.items():
            change=sum(weight*tensors[name] for name,weight in coefficients.items())
            averaged=change.mean(axis=1)
            result={'contrast':contrast,'delta':delta,'coefficients':coefficients,'metrics':{},'seed_means':{}}
            for mi,metric in enumerate(METRICS):
                family=contrast in ('cross-self','both-cross-self+full') and mi<4
                result['metrics'][metric]=summarize_vector(averaged[:,mi],indices,family)
                result['seed_means'][metric]={str(seed):float(change[:,si,mi].mean()) for si,seed in enumerate(seeds)}
            all_contrasts.append(result)
            print('PAIRED',contrast,'delta',delta,'support',result['metrics']['support_proxy'],flush=True)
    result={'status':'complete','role':'post_observation_paired_reanalysis','inputs':{n:v[2] for n,v in loaded.items()},
            'analysis_source_sha256':digest(Path(__file__).read_bytes()),'thresholds_from_full_validation':thresholds,
            'bootstrap_replicates':reps,'bootstrap_seed':202611042,'n_independent_test_documents':500,
            'seed_aggregation':'mean the three matched differences within each document before document resampling',
            'intervals':'percentile bootstrap; primary posthoc family of 16 comparisons additionally receives Bonferroni nominal familywise intervals',
            'null_interpretation':'Intervals spanning zero do not prove equivalence, no effect or exact additivity.',
            'units':{'support_proxy':'NLI probability-scale score, not human factuality rate','rougeL':'0 to 1; multiply by 100 for points',
                     'displayed_text_z':'z-score','content_tokens':'BART content tokens','detected_at_common_full_threshold':'proportion',
                     'length_cap':'proportion','mean_base_entropy':'nats, unwatermarked current-prefix logits'},
            'absolute_means':absolute,'contrasts':all_contrasts}
    atomic_json(output,result)
    # Compact metric view without copying entire raw records or source text.
    md=['# KV 配对差异与交互分析','',
        '同一文档内先对三个种子的配对差值取平均，再对 500 篇文档做 20,000 次 bootstrap。下表为点估计与逐项 95% 区间；JSON 另含 16 项比较的 Bonferroni 校正区间。',
        '交互项 = both − cross − self + full。区间跨零不代表已经证明可加性或等效。NLI 是自动支持代理分数。','',
        '| δ | 对比 | NLI 支持分数差 | ROUGE-L 点数差（×100） | 文本 z 差 | 长度差 |',
        '|---:|---|---:|---:|---:|---:|']
    for r in all_contrasts:
        cells=[]
        for metric,scale in [('support_proxy',1),('rougeL',100),('displayed_text_z',1),('content_tokens',1)]:
            v=r['metrics'][metric];lo,hi=v['document_bootstrap_95ci']
            cells.append(f"{scale*v['mean']:+.4f} [{scale*lo:+.4f}, {scale*hi:+.4f}]")
        md.append('| '+str(int(r['delta']))+' | '+r['contrast']+' | '+' | '.join(cells)+' |')
    output.with_suffix('.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    return result


def selftest():
    rng=np.random.default_rng(7);idx=rng.integers(0,10,(100,10))
    a=np.arange(10,dtype=float)
    full=np.stack([a,a+1,a+2],axis=1)
    cross=full-2;self_=full-3;both=full-5
    assert np.allclose(both-cross-self_+full,0)
    assert np.allclose((cross-self_).mean(axis=1),1)
    assert summarize_vector((cross-self_).mean(axis=1),idx)['mean']==1
    assert summarize_vector(np.zeros(10),idx)['document_bootstrap_95ci']==[0.,0.]
    print('PAIRED_SELFTEST_PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true')
    p.add_argument('--group',choices=['original','nucleus256','topk50'],default='original')
    args=p.parse_args()
    if args.selftest:selftest()
    else:
        if args.group=='original':
            dirs={short:ROOT/'runs/kv-cache'/case/'formal' for short,case in [('full','full'),('cross','cross-half'),('self','self-half'),('both','both-half')]}
            out=ROOT/'runs/kv-cache/paired_analysis.json'
        else:
            dirs={short:ROOT/'runs/kv-supplement'/f'{args.group}-{case}'/'formal' for short,case in [('full','full'),('cross','cross50'),('self','self50'),('both','both50')]}
            out=ROOT/'runs/kv-supplement'/f'{args.group}_paired_analysis.json'
        compare_four(dirs,out)
