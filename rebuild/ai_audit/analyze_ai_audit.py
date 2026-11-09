"""Analyze frozen 3-model AI audit outputs against the private key on the server.
Never writes to human_audit files or labels AI outputs as human verification.
"""
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT)]
from common import atomic_json, digest, read_json

MODELS = ('terra56', 'qwen', 'deepseek')
LABELS = ('supported', 'contradicted', 'not_in_source', 'unclear')


def auroc(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n = int(y.sum())
    m = len(y) - n
    if not n or not m:
        return None
    return float((rankdata(p)[y == 1].sum() - n * (n + 1) / 2) / (n * m))


def cohen_kappa(a_labels, b_labels, categories=LABELS):
    assert len(a_labels) == len(b_labels)
    n = len(a_labels)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a_labels, b_labels)) / n
    ca = Counter(a_labels)
    cb = Counter(b_labels)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in categories)
    if pe >= 1.0:
        return 1.0
    return float((po - pe) / (1.0 - pe))


def cluster_ci(rows, fn, replicates=5000, seed=202611051):
    by_doc = defaultdict(list)
    for r in rows:
        by_doc[r['document_id']].append(r)
    docs = sorted(by_doc)
    if not docs:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(replicates):
        picked = [r for idx in rng.integers(0, len(docs), len(docs)) for r in by_doc[docs[idx]]]
        v = fn(picked)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return np.quantile(vals, [0.025, 0.975]).tolist() if vals else None


def summarize_model_stratum(records_by_id, key_rows):
    valid_rows = []
    missing = []
    quote_fail = 0
    for meta in key_rows:
        iid = meta['item_id']
        rec = records_by_id[iid]
        if rec.get('status') != 'valid':
            missing.append({'item_id': iid, 'status': rec.get('status'), 'http_status': rec.get('http_status')})
            continue
        lbl = rec['verdict']['label']
        q_ok = bool(rec['evidence_checks']['quote_evidence_gate'])
        if not q_ok:
            quote_fail += 1
        nli_val = meta['nli']['source_support_proxy'] if meta.get('nli') else None
        valid_rows.append({
            **meta,
            'label': lbl,
            'quote_ok': q_ok,
            'nli_score': nli_val,
        })
    counts = Counter(r['label'] for r in valid_rows)
    clear = [r for r in valid_rows if r['label'] != 'unclear']
    proxy = [r for r in clear if r['nli_score'] is not None]
    out = {
        'total_items': len(key_rows),
        'valid_n': len(valid_rows),
        'missing_n': len(missing),
        'missing_items': missing,
        'quote_gate_failures': quote_fail,
        'label_counts': {k: counts.get(k, 0) for k in LABELS},
        'clear_n': len(clear),
        'clear_support_rate': (sum(r['label'] == 'supported' for r in clear) / len(clear)) if clear else None,
        'clear_support_ci95': cluster_ci(clear, lambda rr: sum(x['label'] == 'supported' for x in rr) / len(rr)) if clear else None,
        'all_denominator_support_rate': (counts.get('supported', 0) / len(valid_rows)) if valid_rows else None,
    }
    if proxy:
        y = [1 if r['label'] == 'supported' else 0 for r in proxy]
        p = [r['nli_score'] for r in proxy]
        agree = sum((s >= 0.5) == (yi == 1) for yi, s in zip(y, p))
        out['nli_binary_n'] = len(proxy)
        out['nli_agree_at_0.5'] = agree / len(proxy)
        out['nli_agree_count'] = agree
        out['nli_auroc'] = auroc(y, p)
        sup_scores = [s for yi, s in zip(y, p) if yi == 1]
        nonsup_scores = [s for yi, s in zip(y, p) if yi == 0]
        out['mean_nli_supported'] = float(np.mean(sup_scores)) if sup_scores else None
        out['mean_nli_nonsupported'] = float(np.mean(nonsup_scores)) if nonsup_scores else None
    return out, {r['item_id']: r for r in valid_rows}


def main():
    key = read_json(ROOT / 'runs/human-audit-private/annotation_key_do_not_show_annotator.json')
    key_rows = key['rows']
    model_results = {}
    records_map = {}
    for alias in MODELS:
        res = read_json(ROOT / 'runs/ai-audit' / alias / 'formal/results.json')
        assert res['dataset_id'] == key['dataset_id']
        model_results[alias] = {k: v for k, v in res.items() if k != 'records'}
        records_map[alias] = {r['item_id']: r for r in res['records']}

    report = {
        'evaluation_type': 'blinded_three_LLM_judge_audit_not_human_ground_truth',
        'dataset_id': key['dataset_id'],
        'models': model_results,
        'strata': {},
    }

    for stratum in ('initial_fixed_audit', 'paired_cache_audit'):
        sub_key = [r for r in key_rows if r['origin'] == stratum]
        s_out = {'n_items': len(sub_key), 'per_model': {}, 'pairwise': {}, 'three_way': {}, 'secondary_majority': {}}
        valid_maps = {}
        for alias in MODELS:
            summary, vmap = summarize_model_stratum(records_map[alias], sub_key)
            s_out['per_model'][alias] = summary
            valid_maps[alias] = vmap

        for a, b in combinations(MODELS, 2):
            shared_ids = sorted(set(valid_maps[a]) & set(valid_maps[b]))
            la = [valid_maps[a][i]['label'] for i in shared_ids]
            lb = [valid_maps[b][i]['label'] for i in shared_ids]
            clear_ids = [i for i in shared_ids if valid_maps[a][i]['label'] != 'unclear' and valid_maps[b][i]['label'] != 'unclear']
            ba = ['supported' if valid_maps[a][i]['label'] == 'supported' else 'nonsupported' for i in clear_ids]
            bb = ['supported' if valid_maps[b][i]['label'] == 'supported' else 'nonsupported' for i in clear_ids]
            s_out['pairwise'][f'{a}_vs_{b}'] = {
                'shared_valid_n': len(shared_ids),
                'exact_4way_agreement': sum(x == y for x, y in zip(la, lb)) / len(shared_ids),
                'cohen_kappa_4way': cohen_kappa(la, lb, LABELS),
                'shared_clear_n': len(clear_ids),
                'binary_clear_agreement': sum(x == y for x, y in zip(ba, bb)) / len(clear_ids) if clear_ids else None,
                'cohen_kappa_binary_clear': cohen_kappa(ba, bb, ('supported', 'nonsupported')) if clear_ids else None,
            }

        all3_ids = sorted(set(valid_maps['terra56']) & set(valid_maps['qwen']) & set(valid_maps['deepseek']))
        unanimous = 0
        maj_counts = Counter()
        maj_rows = []
        for iid in all3_ids:
            votes = [valid_maps[m][iid]['label'] for m in MODELS]
            c = Counter(votes)
            top_lbl, top_cnt = c.most_common(1)[0]
            if top_cnt == 3:
                unanimous += 1
            if top_cnt >= 2:
                maj_counts[top_lbl] += 1
                meta = valid_maps['terra56'][iid]
                maj_rows.append({**meta, 'label': top_lbl})
            else:
                maj_counts['no_consensus'] += 1

        clear_maj = [r for r in maj_rows if r['label'] != 'unclear' and r['nli_score'] is not None]
        y_m = [1 if r['label'] == 'supported' else 0 for r in clear_maj]
        p_m = [r['nli_score'] for r in clear_maj]
        s_out['three_way'] = {
            'all_three_valid_n': len(all3_ids),
            'unanimous_4way_count': unanimous,
            'unanimous_4way_rate': unanimous / len(all3_ids) if all3_ids else None,
            'majority_label_counts': dict(maj_counts),
            'majority_clear_n': len(clear_maj),
            'majority_clear_support_rate': (sum(y_m) / len(y_m)) if y_m else None,
            'majority_nli_agree_at_0.5': (sum((s >= 0.5) == (yi == 1) for yi, s in zip(y_m, p_m)) / len(y_m)) if y_m else None,
            'majority_nli_auroc': auroc(y_m, p_m) if y_m else None,
        }

        if stratum == 'paired_cache_audit':
            cells = {}
            for delta in (0.0, 2.0):
                for cond in ('full', 'cross-half', 'self-half', 'both-half'):
                    cell_key = f'{cond}_d{int(delta)}'
                    cells[cell_key] = {}
                    for alias in MODELS:
                        rows = [r for r in valid_maps[alias].values() if r['condition'] == cond and r['delta'] == delta]
                        cnt = Counter(r['label'] for r in rows)
                        clr = [r for r in rows if r['label'] != 'unclear']
                        cells[cell_key][alias] = {
                            'valid_n': len(rows),
                            'counts': {k: cnt.get(k, 0) for k in LABELS},
                            'support_all': cnt.get('supported', 0) / len(rows) if rows else None,
                            'support_clear': sum(r['label'] == 'supported' for r in clr) / len(clr) if clr else None,
                        }
                    m_rows = [r for r in maj_rows if r['condition'] == cond and r['delta'] == delta]
                    mcnt = Counter(r['label'] for r in m_rows)
                    mclr = [r for r in m_rows if r['label'] != 'unclear']
                    cells[cell_key]['majority_2of3'] = {
                        'valid_consensus_n': len(m_rows),
                        'counts': {k: mcnt.get(k, 0) for k in LABELS},
                        'support_all': mcnt.get('supported', 0) / len(m_rows) if m_rows else None,
                        'support_clear': sum(r['label'] == 'supported' for r in mclr) / len(mclr) if mclr else None,
                    }
            s_out['cache_cells'] = cells

        report['strata'][stratum] = s_out

    out_path = ROOT / 'runs/ai-audit/panel_analysis.json'
    atomic_json(out_path, report)
    print('PANEL_ANALYSIS_SAVED', out_path)


if __name__ == '__main__':
    main()
