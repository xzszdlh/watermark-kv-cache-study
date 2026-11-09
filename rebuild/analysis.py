"""Prespecified analysis; held-out calibration, document-cluster uncertainty.
No threshold search on test data. No claims that NLI is ground-truth factuality.
"""
import csv
import math
import random
from collections import defaultdict
import numpy as np
from scipy import stats
from common import read_json,atomic_json,LOG,digest


def finite(value):
    return value is not None and math.isfinite(value)


def calibrated_threshold(z, alpha):
    # Strict > threshold; select an observed order statistic. Calibration FPR <= alpha.
    values=np.sort(np.asarray(z,dtype=float))
    if not len(values):raise ValueError('No calibration negatives')
    k=max(0,min(len(values)-1,math.ceil((1-alpha)*len(values))-1))
    return float(values[k])


def auc(pos,neg):
    if not pos or not neg:return None
    ranks=stats.rankdata(np.array(list(pos)+list(neg)))
    n,m=len(pos),len(neg)
    return float((ranks[:n].sum()-n*(n+1)/2)/(n*m))


def ci_mean(values,replicates,rng):
    a=np.array(values,dtype=float)
    if not len(a):return None
    means=np.mean(a[rng.integers(0,len(a),(replicates,len(a)))],axis=1)
    return [float(x) for x in np.quantile(means,[.025,.975])]


def wilson_interval(successes,n):
    """Binomial interval remains informative at 0/n and n/n (bootstrap can degenerate)."""
    if n==0:return None
    z=stats.norm.ppf(.975);p=successes/n;denom=1+z*z/n
    centre=(p+z*z/(2*n))/denom
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0.0,float(centre-half)),min(1.0,float(centre+half))]


def correlation(x,y):
    if len(x)<3 or np.std(x)<1e-9 or np.std(y)<1e-9:return None
    return float(np.corrcoef(x,y)[0,1])


def fit_assoc(rows):
    if len(rows)<10:return None
    # Intercept + watermark + entropy + log length + relative position + copy + repetition
    a=np.asarray([r['features'] for r in rows],dtype=float)
    y=np.asarray([r['support'] for r in rows],dtype=float)
    x=a[:,0];cov=a[:,1:];cov=np.column_stack([np.ones(len(a)),cov])
    xr=x-cov@np.linalg.lstsq(cov,x,rcond=None)[0]
    yr=y-cov@np.linalg.lstsq(cov,y,rcond=None)[0]
    return {'pearson_r':correlation(x,y),'spearman_r':float(stats.spearmanr(x,y).statistic) if np.std(x)>1e-9 and np.std(y)>1e-9 else None,
            'adjusted_partial_r':correlation(xr,yr)}


def association(rows,replicates=1000):
    observed=fit_assoc(rows)
    if observed is None:return {'status':'insufficient_eligible_sentences','n_sentences':len(rows)}
    by_doc=defaultdict(list)
    for r in rows:by_doc[r['id']].append(r)
    keys=sorted(by_doc);rng=np.random.default_rng(20261104)
    boots=[]
    for _ in range(replicates):
        sampled=[r for i in rng.integers(0,len(keys),len(keys)) for r in by_doc[keys[i]]]
        result=fit_assoc(sampled)
        if result and finite(result['adjusted_partial_r']):boots.append(result['adjusted_partial_r'])
    return {'status':'exploratory_NLI_proxy_association','n_sentences':len(rows),'n_documents':len(keys),
            **observed,'adjusted_partial_r_document_bootstrap_95ci':
            [float(v) for v in np.quantile(boots,[.025,.975])] if boots else None,
            'covariates':['base_entropy','log_unique_bigrams','relative_sentence_position','source_trigram_overlap','repeated_bigram_fraction']}


def analyze(config,out,identity,smoke=False):
    generated={};evaluated={}
    for path in sorted((out/'generation').glob('*.json')):
        data=read_json(path);assert data['identity']==identity
        for r in data['records']:
            key=(r['id'],r['seed'],r['delta']);assert key not in generated
            generated[key]=r
    for path in sorted((out/'evaluation').glob('*.json')):
        data=read_json(path);assert data['identity']==identity
        for r in data['records']:
            key=(r['id'],r['seed'],r['delta']);assert key not in evaluated
            evaluated[key]=r
    assert set(generated)==set(evaluated),'Missing evaluated records'
    samples=read_json(out/'samples.json');docs={s['id']:s for s in samples}
    seeds=config['generation']['seeds'][:1] if smoke else config['generation']['seeds']
    deltas=config['watermark']['deltas']
    expected=len(samples)*len(seeds)*len(deltas)
    assert len(generated)==expected, f'Incomplete run {len(generated)}/{expected}'
    table=[]; associations=[]; thresholds={};rng=np.random.default_rng(20261104)
    reps=50 if smoke else config['evaluation']['bootstrap_replicates']
    for seed in seeds:
        calibration=[r['watermark_text']['z'] for r in generated.values() if r['seed']==seed and r['delta']==0 and r['role']=='calibration' and finite(r['watermark_text']['z'])]
        threshold=calibrated_threshold(calibration,config['evaluation']['fpr_target']);thresholds[seed]=threshold
        baseline={r['id']:r for r in generated.values() if r['seed']==seed and r['delta']==0 and r['role']=='test'}
        negatives=[r['watermark_text']['z'] for r in baseline.values() if finite(r['watermark_text']['z'])]
        for delta in deltas:
            LOG.info('ANALYSIS_START seed=%d delta=%g (document bootstrap; no threshold fitting on test)',seed,delta)
            rows=[r for r in generated.values() if r['seed']==seed and r['delta']==delta and r['role']=='test']
            z=[r['watermark_text']['z'] for r in rows if finite(r['watermark_text']['z'])]
            detection=[int(v>threshold) for v in z]
            rouge_delta=[];quality=defaultdict(list);sentences=[];text_sentences=[];nli=[];missing=0;alignment_missing=0
            for r in rows:
                ev=evaluated[(r['id'],seed,delta)]
                for metric,value in ev['rouge'].items():quality[metric].append(value)
                base=evaluated[(r['id'],seed,0.0)]
                rouge_delta.append(ev['rouge']['rougeL']-base['rouge']['rougeL'])
                vals=[]
                for u in ev['sentences']:
                    alignment_missing+=not u['alignment_valid']
                    if u['nli'] is None:
                        missing+=1;continue
                    support=u['nli']['source_support_proxy'];vals.append(support)
                    for metric,target in [('watermark',sentences),('watermark_text',text_sentences)]:
                        wm=u[metric]
                        if wm and wm['n']>=config['evaluation']['min_segment_unique_bigrams'] and finite(wm['z']) and finite(u['mean_base_entropy']):
                            target.append({'id':r['id'],'support':support,
                                           'features':[wm['z'],u['mean_base_entropy'],math.log(wm['n']),u['relative_position'],
                                                       u['copy_trigram_fraction'],u['repeat_fraction']]})
                if vals:nli.append(float(np.mean(vals)))
            table.append({'seed':seed,'delta':delta,'n_test_documents':len(rows),
                          'detection_input':'displayed_text_retokenized',
                          'retokenization_changed_fraction':float(np.mean([r['retokenization_changed'] for r in rows])),
                          'internal_mean_z':float(np.mean([r['watermark']['z'] for r in rows if finite(r['watermark']['z'])])),
                          'threshold_calibrated_on_validation_baseline':threshold,
                          'calibration_negatives':len(calibration),
                          'calibration_empirical_FPR':float(np.mean(np.array(calibration)>threshold)),
                          'heldout_detection_rate':float(np.mean(detection)) if detection else None,
                          'rate_meaning':'FPR' if delta==0 else 'TPR',
                          'heldout_detection_rate_wilson_95ci':wilson_interval(sum(detection),len(detection)),
                          'heldout_detection_rate_doc_bootstrap_95ci':ci_mean(detection,reps,rng),
                          'heldout_z_gt_4_rate':float(np.mean(np.array(z)>4)) if z else None,
                          'mean_z':float(np.mean(z)) if z else None,
                          'AUROC_vs_unwatermarked':auc(z,negatives) if delta else None,
                          'wrong_key_z_gt_4_rate':float(np.mean([r['wrong_key_text']['z']>4 for r in rows if finite(r['wrong_key_text']['z'])])),
                          'mean_content_tokens':float(np.mean([r['n_content_tokens'] for r in rows])),
                          'token_limit_fraction':float(np.mean([r['hit_token_limit'] for r in rows])),
                          'mean_ROUGE':{k:float(np.mean(v)) for k,v in quality.items()},
                          'paired_ROUGE_L_change_vs_delta0':float(np.mean(rouge_delta)),
                          'paired_ROUGE_L_change_doc_bootstrap_95ci':ci_mean(rouge_delta,reps,rng),
                          'mean_NLI_support_proxy':float(np.mean(nli)) if nli else None,
                          'NLI_unscored_sentences':missing,'token_alignment_failures':alignment_missing})
            associations.append({'seed':seed,'delta':delta,'primary':seed==42 and delta==2.0,
                                 'signal':'internal_sampled_token_watermark',**association(sentences,reps)})
            associations.append({'seed':seed,'delta':delta,'primary':False,
                                 'signal':'displayed_text_watermark',**association(text_sentences,reps)})
            LOG.info('ANALYSIS_PROGRESS seed=%d delta=%g docs=%d eligible_sentences=%d',seed,delta,len(rows),len(sentences))
    result={'identity':identity,'experiment_id':config.get('experiment_id','unrestricted-128'),
            'study_role':config.get('study_role','initial_fixed_protocol'),
            'decoding':config['generation'],
            'status':'smoke_only' if smoke else 'automatic_experiment_complete_human_validation_pending',
            'record_count':len(generated),'n_independent_test_documents':sum(s['role']=='test' for s in samples),
            'thresholds':thresholds,'table':table,'associations':associations,
            'interpretation_limits':['NLI is an automatic source-support proxy, not verified hallucination labels.',
                                     'Normal-approximation p-values are diagnostic; report empirical FPR.',
                                     'Seeds are repeated measurements of the same documents, not extra independent samples.',
                                     'No KV compression or causal claim is tested.']}
    atomic_json(out/'analysis.json',result)
    if not smoke:make_human_audit(config,out,generated,evaluated,docs)
    LOG.info('ANALYSIS_READY records=%d file=%s',len(generated),out/'analysis.json')
    return result


def make_human_audit(config,out,generated,evaluated,docs):
    # Fixed random sample, not cherry-picked by z-score or NLI score.
    candidates=[]
    for key,r in evaluated.items():
        if r['role']=='test' and r['seed']==42 and r['delta'] in (0.0,2.0):
            for u in r['sentences']:
                candidates.append((key,u))
    random.Random(20261104).shuffle(candidates)
    chosen=candidates[:config['evaluation']['human_audit_n']]
    rows=[];answers=[]
    for i,(key,u) in enumerate(chosen):
        audit_id=f'A{i+1:03}'
        rows.append({'audit_id':audit_id,'source_visible_to_model':docs[key[0]]['visible_source'],
                     'summary_sentence':u['text'],'label_supported_contradicted_or_unclear':'','evidence_quote':'','notes':''})
        answers.append({'audit_id':audit_id,'document_id':key[0],'seed':key[1],'delta':key[2],
                        'sentence_index':u['sentence_index'],'watermark':u['watermark'],'nli':u['nli']})
    if rows:
        with open(out/'human_audit_blinded.csv','w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        atomic_json(out/'human_audit_key_do_not_show_annotator.json',answers)
