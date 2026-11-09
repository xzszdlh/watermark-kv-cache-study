"""Freeze blinded annotation materials; never fill a human label automatically."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import csv
import io
import json
import random
from common import read_json,atomic_json,digest

OUT=ROOT/'human_audit'
PRIVATE=ROOT/'runs/human-audit-private'
CASES=('full','cross-half','self-half','both-half')


def load_seed_records(directory,folder):
    result={};manifest=read_json(directory/'manifest.json')
    for p in sorted((directory/folder).glob('seed42_*.json')):
        d=read_json(p);assert d['identity']==manifest['identity']
        for r in d['records']:
            if r['role']=='test' and r['delta'] in (0.,2.):
                key=(r['id'],r['delta'])
                assert key not in result
                result[key]=r
    return result


def create():
    plan=read_json(ROOT/'supplements/protocol.json')['human_audit']
    existing_sheet=OUT/'人工核验_空白表.csv'
    if existing_sheet.exists():
        with existing_sheet.open(encoding='utf-8-sig',newline='') as f:
            if any((r.get('label') or '').strip() or (r.get('evidence_quote') or '').strip()
                   or (r.get('notes') or '').strip() for r in csv.DictReader(f)):
                raise RuntimeError('Human work exists in the public sheet; refusing to overwrite it')
    samples=read_json(ROOT/'runs/formal/samples.json');lookup={s['id']:s for s in samples}
    original_csv=ROOT/plan['original_input']
    with original_csv.open(encoding='utf-8-sig',newline='') as f:original=list(csv.DictReader(f))
    assert len(original)==plan['original_rows']
    if any(r['label_supported_contradicted_or_unclear'].strip() for r in original):
        raise RuntimeError('Existing human labels found; import them explicitly instead of regenerating blank sheets')
    original_key={r['audit_id']:r for r in read_json(ROOT/'runs/formal/human_audit_key_do_not_show_annotator.json')}
    original_gen=load_seed_records(ROOT/'runs/formal','generation')
    records=[]
    for row in original:
        key=original_key[row['audit_id']]
        records.append({'source_visible_to_model':row['source_visible_to_model'],
                        'summary_sentence':row['summary_sentence'],
                        'summary_context':original_gen[(key['document_id'],key['delta'])]['summary'],
                        '_key':{'origin':'initial_fixed_audit','original_audit_id':row['audit_id'],**key}})
    rng=random.Random(plan['selection_seed'])
    ids=sorted(s['id'] for s in samples if s['role']=='test')
    chosen=sorted(rng.sample(ids,plan['kv_documents']))
    source_identities={}
    for case in CASES:
        directory=ROOT/'runs/kv-cache'/case/'formal'
        gen=load_seed_records(directory,'generation');ev=load_seed_records(directory,'evaluation')
        source_identities[case]=read_json(directory/'manifest.json')['identity']
        for doc in chosen:
            for delta in plan['kv_deltas']:
                key=(doc,delta);sentences=ev[key]['sentences']
                if not sentences:raise ValueError('Empty summary; no silent replacement of audit items')
                selected=rng.choice(sentences)
                records.append({'source_visible_to_model':lookup[doc]['visible_source'],
                                'summary_sentence':selected['text'],'summary_context':gen[key]['summary'],
                                '_key':{'origin':'paired_cache_audit','document_id':doc,'condition':case,
                                        'delta':delta,'seed':plan['kv_seed'],'sentence_index':selected['sentence_index'],
                                        'n_sentences_in_summary':len(sentences),'nli':selected['nli'],
                                        'watermark':selected['watermark'],'watermark_text':selected['watermark_text']}})
    assert len(records)==260
    random.Random(plan['display_shuffle_seed']).shuffle(records)
    public=[];private=[]
    for index,row in enumerate(records,1):
        item_id=f'H{index:03d}'
        public.append({'item_id':item_id,**{k:v for k,v in row.items() if k!='_key'}})
        private.append({'item_id':item_id,**row['_key']})
    fingerprint=digest(public)
    OUT.mkdir(exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    private_path=PRIVATE/'annotation_key_do_not_show_annotator.json'
    bundle={'dataset_id':fingerprint,'original_csv_sha256':digest(original_csv.read_bytes()),'kv_source_identities':source_identities,
            'plan':plan,'selected_kv_document_ids':chosen,'rows':private}
    if private_path.exists() and read_json(private_path)!=bundle:raise RuntimeError('Audit selection changed after freeze')
    atomic_json(private_path,bundle)
    template=(Path(__file__).parent/'human_audit_template.html').read_text(encoding='utf-8')
    data=json.dumps({'dataset_id':fingerprint,'items':public},ensure_ascii=False).replace('<','\\u003c')
    html=template.replace('__AUDIT_DATA__',data)
    assert '__AUDIT_DATA__' not in html
    (OUT/'开始人工核验.html').write_text(html,encoding='utf-8')
    columns=['item_id','source_visible_to_model','summary_sentence','summary_context','label','evidence_quote','notes','annotator','assistance']
    path=OUT/'人工核验_空白表.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        w.writerows({**r,'label':'','evidence_quote':'','notes':'','annotator':'','assistance':''} for r in public)
    # Public manifest contains no conditions/scores or source-document IDs.
    atomic_json(OUT/'audit_manifest.json',{'dataset_id':fingerprint,'items':260,'labels':['supported','contradicted','not_in_source','unclear'],
                'human_labels_filled':0,'blinded':True,'model_visible_source_only':True})
    # A distinct rubric keeps unsupported facts separate from difficult-to-judge text.
    print('HUMAN_AUDIT_READY',OUT,'rows',len(public),'dataset',fingerprint)


if __name__=='__main__':create()
