"""Analyze completed human labels against a private frozen key. Never creates labels."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import argparse
import csv
from collections import Counter,defaultdict
import json
import numpy as np
from scipy.stats import rankdata
from common import atomic_json,read_json,digest
from validate_human_audit import validate

PRIVATE=ROOT/'runs/human-audit-private'
PLAN=Path(__file__).with_name('human_audit_analysis_plan.json')


def freeze():
    content={'plan':read_json(PLAN),'sha256':digest(PLAN.read_bytes()),
             'analysis_source_sha256':digest(Path(__file__).read_bytes())}
    dest=PRIVATE/'analysis_specification.json'
    if dest.exists():assert read_json(dest)==content,'Audit analysis specification changed after freeze'
    else:atomic_json(dest,content)
    return content['plan']


def auc(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    n=int(y.sum());m=len(y)-n
    if not n or not m:return None
    return float((rankdata(p)[y==1].sum()-n*(n+1)/2)/(n*m))


def cluster_interval(rows,function,replicates,seed):
    by_doc=defaultdict(list)
    for row in rows:by_doc[row['document_id']].append(row)
    docs=sorted(by_doc)
    if not docs:return None
    rng=np.random.default_rng(seed);values=[]
    for _ in range(replicates):
        selected=[row for i in rng.integers(0,len(docs),len(docs)) for row in by_doc[docs[i]]]
        value=function(selected)
        if value is not None and np.isfinite(value):values.append(value)
    return {'document_bootstrap_95ci':np.quantile(values,[.025,.975]).tolist() if values else None,
            'valid_replicates':len(values),'n_documents':len(docs)}


def describe(rows,plan,bootstrap=True):
    judged=[r for r in rows if r['label']!='unclear']
    proxy=[r for r in judged if r['nli_score'] is not None]
    counts=Counter(r['label'] for r in rows)
    out={'n_items':len(rows),'n_documents':len({r['document_id'] for r in rows}),
         'label_counts':dict(counts),'n_binary_judged':len(judged),'n_NLI_available':len(proxy),
         'supported_fraction_among_judged':sum(r['label']=='supported' for r in judged)/len(judged) if judged else None,
         'supported_fraction_including_unclear_denominator':counts['supported']/len(rows) if rows else None,
         'mean_NLI_by_human_label':{label:float(np.mean([r['nli_score'] for r in rows if r['label']==label and r['nli_score'] is not None]))
                                   for label in counts if any(r['label']==label and r['nli_score'] is not None for r in rows)}}
    if proxy:
        y=[r['label']=='supported' for r in proxy];p=[r['nli_score'] for r in proxy]
        prediction=[v>=plan['nli_threshold'] for v in p]
        out['NLI_AUROC']=auc(y,p)
        out['Brier_score_descriptive']=float(np.mean((np.array(p)-np.array(y,dtype=float))**2))
        out['confusion_at_fixed_0.5']={name:sum(bool(a)==truth and bool(b)==pred for a,b in zip(y,prediction))
           for name,truth,pred in [('TP',True,True),('FN',True,False),('FP',False,True),('TN',False,False)]}
    if bootstrap and judged:
        out['supported_fraction_uncertainty']=cluster_interval(judged,
             lambda rr:sum(r['label']=='supported' for r in rr)/len(rr),**plan['bootstrap'])
    return out


def main(export):
    plan=freeze()
    # Remove the descriptor field before passing bootstrap keyword arguments.
    plan={**plan,'bootstrap':{k:v for k,v in plan['bootstrap'].items() if k!='unit'}}
    validation=validate(export,ROOT/'human_audit/开始人工核验.html')
    out=ROOT/'runs/human-audit-results';out.mkdir(exist_ok=True)
    atomic_json(out/'validation.json',validation)
    if validation['errors'] or validation['complete_items']!=260:
        print('HUMAN_AUDIT_INCOMPLETE_OR_INVALID',validation['complete_items'],'/260; scientific results not produced')
        return
    if export.suffix.lower()=='.json':
        data=read_json(export);rows=data.get('items') or list(data['answers'].values())
    else:
        with export.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    key=read_json(PRIVATE/'annotation_key_do_not_show_annotator.json')
    assert validation['dataset_id']==key['dataset_id']
    lookup={r['item_id']:r for r in key['rows']};joined=[]
    for row in rows:
        meta=lookup[row['item_id']];nli=meta.get('nli')
        joined.append({**row,**meta,'nli_score':nli['source_support_proxy'] if nli else None})
    result={'status':'human_export_analyzed','dataset_id':key['dataset_id'],
            'label_file_sha256':validation['file_sha256'],'analysis_plan_sha256':digest(PLAN.read_bytes()),
            'annotators':sorted({r['annotator'] for r in joined}),
            'limits':plan['limits'],'validation_warnings':validation['warnings'],'strata':{},'cache_cells':[],'cache_pairs':[]}
    for origin in ('initial_fixed_audit','paired_cache_audit'):
        selected=[r for r in joined if r['origin']==origin]
        result['strata'][origin]=describe(selected,plan)
    kv=[r for r in joined if r['origin']=='paired_cache_audit']
    for delta in (0.,2.):
        for case in ('full','cross-half','self-half','both-half'):
            result['cache_cells'].append({'condition':case,'delta':delta,
                  **describe([r for r in kv if r['condition']==case and r['delta']==delta],plan,bootstrap=False)})
        for left,right in [('cross-half','full'),('self-half','full'),('both-half','full'),('cross-half','self-half')]:
            a={r['document_id']:r for r in kv if r['condition']==left and r['delta']==delta}
            b={r['document_id']:r for r in kv if r['condition']==right and r['delta']==delta}
            assert a.keys()==b.keys() and len(a)==20
            differences=[{'document_id':doc,'value':float(a[doc]['label']=='supported')-float(b[doc]['label']=='supported')}
                        for doc in sorted(a) if a[doc]['label']!='unclear' and b[doc]['label']!='unclear']
            result['cache_pairs'].append({'left':left,'right':right,'delta':delta,'n_eligible_document_pairs':len(differences),
               'n_excluded_unclear_pairs':20-len(differences),
               'mean_supported_indicator_change':float(np.mean([r['value'] for r in differences])) if differences else None,
               'uncertainty':cluster_interval(differences,lambda rr:float(np.mean([r['value'] for r in rr])),**plan['bootstrap'])})
    # Error inventory for author review; human labels are never overwritten.
    errors=[{'item_id':r['item_id'],'label':r['label'],'nli_score':r['nli_score'],
             'evidence_quote':r.get('evidence_quote',''),'notes':r.get('notes','')}
            for r in joined if r['label']!='unclear' and r['nli_score'] is not None and
            (r['nli_score']>=plan['nli_threshold'])!=(r['label']=='supported')]
    atomic_json(out/'analysis.json',result);atomic_json(out/'NLI_disagreements.json',errors)
    print('HUMAN_AUDIT_ANALYZED',out,'NLI_disagreements',len(errors))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',type=Path);parser.add_argument('--freeze-only',action='store_true');parser.add_argument('--selftest',action='store_true')
    args=parser.parse_args()
    if args.freeze_only:freeze();print('HUMAN_ANALYSIS_SPECIFICATION_FROZEN')
    elif args.selftest:
        assert auc([1,0],[.9,.1])==1 and auc([1,0],[.1,.9])==0 and auc([1,0],[.5,.5])==.5
        assert auc([1,1],[.9,.8]) is None
        x=cluster_interval([{'document_id':'a','value':1},{'document_id':'b','value':1}],lambda rr:float(np.mean([r['value'] for r in rr])),100,7)
        assert x['document_bootstrap_95ci']==[1.,1.]
        print('HUMAN_ANALYSIS_SELFTEST_PASS; synthetic checks were in memory, no human labels created')
    elif args.export:main(args.export)
    else:parser.error('--export or --freeze-only required')
