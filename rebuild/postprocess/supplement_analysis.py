"""Complete paired comparisons for the fixed supplementary suite.
No GPU use, no modification of generation/evaluation checkpoints or human labels.
"""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT),str(HERE)]
import argparse
import fcntl
import math
import os
import time
import traceback
import numpy as np
from common import read_json,atomic_json,digest,stage_status
from paired_analysis import load_records,METRICS,CONTRASTS
BASE=ROOT/'runs/kv-supplement'
PLAN=HERE/'supplement_analysis_plan.json'
GROUPS={
 'original50':{'full':'original/full','cross':'original/cross-half','self':'original/self-half','both':'original/both-half'},
 'retention75':{'full':'original/full','cross':'retention-cross75','self':'retention-self75','both':'retention-both75'},
 'retention25':{'full':'original/full','cross':'retention-cross25','self':'retention-self25','both':'retention-both25'},
 'nucleus256':{'full':'nucleus256-full','cross':'nucleus256-cross50','self':'nucleus256-self50','both':'nucleus256-both50'},
 'topk50':{'full':'topk50-full','cross':'topk50-cross50','self':'topk50-self50','both':'topk50-both50'},
 'random50':{'full':'original/full','cross':'random50-cross50','self':'original/self-half','both':'random50-both50'}}


def outdir(name):
    return ROOT/'runs/kv-cache'/name.split('/')[1]/'formal' if name.startswith('original/') else BASE/name/'formal'


def freeze():
    BASE.mkdir(parents=True,exist_ok=True)
    payload={'plan':read_json(PLAN),'plan_sha256':digest(PLAN.read_bytes()),
             'sources':{p.name:digest(p.read_bytes()) for p in [Path(__file__),HERE/'paired_analysis.py']}}
    dest=BASE/'postprocess_manifest.json'
    if dest.exists():
        old=read_json(dest)
        assert all(old[k]==v for k,v in payload.items()),'Analysis specification/source changed after freeze'
    else:atomic_json(dest,{**payload,'fixed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    return payload


def intervals(values,indices,family_size=None):
    values=np.asarray(values,dtype=float)
    if not np.isfinite(values).all():raise ValueError('Nonfinite outcome; no silent dropping')
    boot=np.empty(len(indices))
    # Keep memory bounded when several CPU jobs run alongside GPU generation.
    for start in range(0,len(indices),1000):boot[start:start+1000]=values[indices[start:start+1000]].mean(axis=1)
    out={'mean':float(values.mean()),'document_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist()}
    if family_size:
        tail=.05/(2*family_size)
        out[f'bonferroni_{family_size}_nominal_familywise_95ci']=np.quantile(boot,[tail,1-tail]).tolist()
    return out


def execute():
    manifest=freeze();plan=manifest['plan']
    names=sorted({v for group in GROUPS.values() for v in group.values()})
    loaded={}
    for name in names:
        directory=outdir(name)
        print('LOAD',name,flush=True)
        loaded[name]=load_records(directory)
        summary=read_json(directory/'analysis.json')
        assert summary['identity']==loaded[name][2]['identity']
    keys=set(loaded['original/full'][0]);docs=sorted({k[0] for k in keys});seeds=sorted({k[1] for k in keys})
    assert len(docs)==500 and seeds==[42,43,44]
    assert all(set(v[0])==keys for v in loaded.values())
    # Pair original immutable source/doc selections and scientifically fixed input settings.
    base_c=loaded['original/full'][2]['protocol']
    sample_hash=digest((outdir('original/full')/'samples.json').read_bytes())
    for name,(rows,thresholds,provenance) in loaded.items():
        c=provenance['protocol']
        for section in ('model','nli','dataset','evaluation'):assert c[section]==base_c[section],(name,section)
        assert c['generation']['batch_size']==32 and c['generation']['max_source_tokens']==1024
        assert digest((outdir(name)/'samples.json').read_bytes())==sample_hash,(name,'sample selection')
    rng=np.random.default_rng(plan['bootstrap_seed'])
    indices=rng.integers(0,len(docs),(plan['bootstrap_replicates'],len(docs)),dtype=np.int32)
    groups={};effect_tensors={};primary_intervals=0
    for group,mapping in GROUPS.items():
        print('GROUP',group,flush=True)
        thresholds=loaded[mapping['full']][1]
        tensors={};absolute=[];contrasts=[]
        for delta in (0.,2.):
            tensors[delta]={}
            for label,name in mapping.items():
                rows=loaded[name][0]
                vals=[]
                for doc in docs:
                    seed_values=[]
                    for seed in seeds:
                        row={**rows[(doc,seed,delta)]}
                        row['detected_at_common_full_threshold']=float(row['displayed_text_z']>thresholds[seed])
                        seed_values.append([row[m] for m in METRICS])
                    vals.append(seed_values)
                tensor=np.array(vals,dtype=float);assert np.isfinite(tensor).all()
                tensors[delta][label]=tensor
                summary=read_json(outdir(name)/'analysis.json')
                select=[r for r in summary['table'] if r['delta']==delta]
                absolute.append({'condition':label,'run_name':name,'delta':delta,
                   'mean':dict(zip(METRICS,tensor.mean(axis=(0,1)).tolist())),
                   'per_seed_common_threshold_detection':{str(s):float(tensor[:,i,4].mean()) for i,s in enumerate(seeds)},
                   'mean_cache_fraction_at_own_final_step':float(np.mean([rows[(d,s,delta)]['cache_fraction_at_own_final_step'] for d in docs for s in seeds])),
                   'NLI_unscored_sentences':sum(r['NLI_unscored_sentences'] for r in select),
                   'token_alignment_failures':sum(r['token_alignment_failures'] for r in select)})
            for contrast,coefficients in CONTRASTS.items():
                difference=sum(w*tensors[delta][k] for k,w in coefficients.items())
                means=difference.mean(axis=1);result={'contrast':contrast,'delta':delta,'metrics':{},'seed_means':{}}
                for mi,metric in enumerate(METRICS):
                    family=(group!='original50' and contrast in ('cross-self','both-cross-self+full') and mi<4)
                    result['metrics'][metric]=intervals(means[:,mi],indices,plan['global_family_size'] if family else None)
                    primary_intervals+=int(family)
                    result['seed_means'][metric]={str(s):float(difference[:,i,mi].mean()) for i,s in enumerate(seeds)}
                contrasts.append(result)
                effect_tensors[(group,delta,contrast)]=difference
        groups[group]={'thresholds_from_full_validation':thresholds,'absolute_means':absolute,'contrasts':contrasts}
    assert primary_intervals==80,primary_intervals
    extras=[]
    for group,comparisons in [('nucleus256',('cross-full','self-full','both-full')),
                              ('topk50',('cross-full','self-full','both-full')),
                              ('random50',('cross-full','both-full'))]:
        for delta in (0.,2.):
            for contrast in comparisons:
                difference=effect_tensors[(group,delta,contrast)]-effect_tensors[('original50',delta,contrast)]
                extras.append({'group':group,'reference':'original50','cache_effect':contrast,'delta':delta,
                    'meaning':'paired change in condition-minus-full effect; no global multiplicity claim',
                    'metrics':{m:intervals(difference[:,:,i].mean(axis=1),indices) for i,m in enumerate(METRICS)}})
    result={'status':'automatic_complete_human_validation_pending','role':plan['role'],
            'analysis_plan_sha256':manifest['plan_sha256'],'input_sources':{n:v[2] for n,v in loaded.items()},
            'sample_sha256':sample_hash,'n_independent_test_documents':len(docs),'seeds':seeds,
            'bootstrap_replicates':plan['bootstrap_replicates'],'global_family_intervals':primary_intervals,
            'groups':groups,'additional_exploratory_difference_in_differences':extras,'limits':plan['limits']}
    atomic_json(BASE/'supplement_analysis.json',result)
    lines=['# 补充实验汇总','',
      '固定的全部补充条件均纳入汇总；同一文档的三个种子先取配对差值均值，再对 500 篇文档进行 50,000 次 bootstrap。NLI 是自动支持代理分数。',
      'JSON 保留逐种子数值、全部区间及声明的 80 项比较族的 Bonferroni 校正区间。下表显示逐项 95% 区间。','']
    titles={'original50':'原始 50% 缓存条件','retention75':'75% 保留率','retention25':'25% 保留率',
            'nucleus256':'长度上限 256','topk50':'top-k=50 采样','random50':'随机源位置裁剪'}
    for name,group in groups.items():
        lines+=['## '+titles[name],'','| δ | 条件 | NLI 分数 | ROUGE-L ×100 | 水印检测比例（共同阈值） | 平均长度 | KV 字节比例 |',
               '|---:|---|---:|---:|---:|---:|---:|']
        for row in group['absolute_means']:
            m=row['mean'];lines.append(f"| {int(row['delta'])} | {row['condition']} | {m['support_proxy']:.4f} | {100*m['rougeL']:.2f} | {100*m['detected_at_common_full_threshold']:.2f}% | {m['content_tokens']:.2f} | {row['mean_cache_fraction_at_own_final_step']:.4f} |")
        lines+=['','| δ | 对比 | NLI 差 [95% CI] | 文本 z 差 [95% CI] |','|---:|---|---:|---:|']
        for row in group['contrasts']:
            cells=[]
            for metric in ('support_proxy','displayed_text_z'):
                v=row['metrics'][metric];lo,hi=v['document_bootstrap_95ci'];cells.append(f"{v['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            lines.append('| '+str(int(row['delta']))+' | '+row['contrast']+' | '+' | '.join(cells)+' |')
        lines+=['']
    lines+=['## 解释范围','',
       'δ=0 的检测比例是误报率，δ=2 是检出率。阈值只使用该组完整缓存的 200 篇验证文档标定。',
       'KV 字节比例是各序列结束步相对同一步完整缓存的比例；它不是整卡显存节省率，也不能用来声称共享 GPU 上的加速。',
       '交互项为 both − cross − self + full。区间跨零不证明精确可加、无效应或等效。全部补充结果来自原来的文档，不构成独立验证集。','']
    (BASE/'supplement_analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    stage_status(BASE/'postprocess','complete',input_conditions=len(loaded),new_generated_records=67200,human_validation_pending=True)
    print('SUPPLEMENT_ANALYSIS_COMPLETE',flush=True)


def main(wait):
    freeze();out=BASE/'postprocess';out.mkdir(exist_ok=True)
    with open(out/'.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if wait:
                start=time.monotonic()
                while True:
                    status=read_json(BASE/'status.json')
                    if status['stage']=='complete':break
                    if status['stage']=='failed':raise RuntimeError('Upstream supplement failed; inspect logs before resuming')
                    if time.monotonic()-start>8*3600:raise TimeoutError('Upstream completion wait exceeded eight hours')
                    stage_status(out,'waiting_for_supplement',pid=os.getpid(),upstream_stage=status['stage'])
                    print('WAIT',status['stage'],'generation',status.get('generation'),'evaluation',status.get('evaluation'),flush=True)
                    time.sleep(30)
            else:assert read_json(BASE/'status.json')['stage']=='complete'
            stage_status(out,'analysis',pid=os.getpid())
            execute()
        except Exception:
            stage_status(out,'failed',traceback=traceback.format_exc());raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args()
    if args.freeze_only:freeze();print('ANALYSIS_PLAN_FROZEN')
    else:main(args.wait)
