"""Validate a human CSV/HTML export without guessing or filling any labels.
Can run locally without the blinded key. Never prints condition/NLI/watermark data.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from collections import Counter

ROOT=Path(__file__).resolve().parent.parent
LABELS={'supported','contradicted','not_in_source','unclear'}
ASSISTANCE={'none','dictionary','translation','other',''}


def load_public(html):
    text=html.read_text(encoding='utf-8')
    match=re.search(r'<script id="audit-data" type="application/json">(.*?)</script>',text,re.S)
    if not match:raise ValueError('Blinded public data not found')
    return json.loads(match.group(1))


def normalize(value):
    return re.sub(r'\s+',' ',value).strip()


def validate(path,html):
    public=load_public(html);items={r['item_id']:r for r in public['items']}
    errors=[];warnings=[];complete=[];counts=Counter();assistance=Counter()
    if path.suffix.lower()=='.json':
        bundle=json.loads(path.read_text(encoding='utf-8-sig'))
        if bundle.get('dataset_id')!=public['dataset_id']:raise ValueError('Different annotation dataset')
        rows=bundle.get('items')
        if rows is None:rows=list(bundle.get('answers',{}).values())
        for row in rows:
            aid=row.get('item_id');answer=bundle.get('answers',{}).get(aid)
            if answer:
                for field in ('label','evidence_quote','notes','annotator','assistance'):
                    if field in row and row[field]!=answer.get(field,''):
                        errors.append(f'{aid}: items/answers disagree for {field}')
    elif path.suffix.lower()=='.csv':
        with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    else:raise ValueError('Expected JSON or CSV export')
    seen=set()
    for row in rows:
        aid=row.get('item_id')
        if aid in seen:errors.append(f'Duplicate item_id {aid}');continue
        seen.add(aid)
        if aid not in items:errors.append(f'Unknown item_id {aid}');continue
        source=items[aid]
        for field in ('source_visible_to_model','summary_sentence','summary_context'):
            if field in row and row[field]!=source[field]:errors.append(f'{aid}: original {field} changed')
        label=(row.get('label') or '').strip()
        if not label:
            if any((row.get(k) or '').strip() for k in ('evidence_quote','notes')):
                warnings.append(f'{aid}: notes or evidence entered but no label')
            continue
        if label not in LABELS:errors.append(f'{aid}: unrecognized label {label}');continue
        counts[label]+=1
        name=(row.get('annotator') or '').strip();evidence=(row.get('evidence_quote') or '').strip();notes=(row.get('notes') or '').strip()
        help_=(row.get('assistance') or '').strip();assistance[help_]+=1
        valid=True
        if not name:warnings.append(f'{aid}: missing annotator');valid=False
        if label in ('supported','contradicted') and not evidence:warnings.append(f'{aid}: evidence quote required');valid=False
        if label in ('not_in_source','unclear') and not notes:warnings.append(f'{aid}: reason required');valid=False
        if help_ not in ASSISTANCE:errors.append(f'{aid}: unknown assistance value {help_}');valid=False
        if help_=='other' and not notes:warnings.append(f'{aid}: describe other assistance');valid=False
        if evidence:
            # This is a quote-presence check, never a semantic label judgement.
            lines=[normalize(line) for line in evidence.splitlines() if normalize(line)]
            if lines and any(line not in normalize(source['source_visible_to_model']) for line in lines):
                warnings.append(f'{aid}: evidence is not an exact contiguous source quote; check ellipses, paraphrase or typo')
        if valid:complete.append(aid)
    missing=sorted(set(items)-seen)
    if missing:warnings.append(f'{len(missing)} item rows absent from export (partial work is allowed)')
    return {'file':path.name,'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'dataset_id':public['dataset_id'],
            'expected_items':len(items),'rows_seen':len(rows),'labeled_items':sum(counts.values()),
            'complete_items':len(complete),'unlabeled_items':len(items)-sum(counts.values()),
            'label_counts':dict(counts),'assistance_counts':dict(assistance),
            'errors':errors,'warnings':warnings,
            'semantic_labels_changed':False,
            'note':'Structure/quote presence only. Does not establish human authorship or validate semantic correctness.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('export',type=Path);p.add_argument('--html',type=Path,default=ROOT/'human_audit/开始人工核验.html')
    p.add_argument('--out',type=Path);args=p.parse_args()
    report=validate(args.export,args.html)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if report['errors']:raise SystemExit(1)
