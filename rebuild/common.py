"""Shared I/O. Atomic batch checkpoints, explicit run identity, UTF-8 logging."""
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger('study')


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_protocol(path):
    """Resolve explicit, task-local JSON overrides before any measurements."""
    from copy import deepcopy
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Protocol must be within the experiment workspace')
    overrides=read_json(path)
    def merge(base,update):
        result=deepcopy(base)
        for key,value in update.items():
            if isinstance(value,dict) and isinstance(result.get(key),dict):
                result[key]=merge(result[key],value)
            else:result[key]=deepcopy(value)
        return result
    if 'base' not in overrides:return overrides
    base=load_protocol(path.parent/overrides.pop('base'))
    return merge(base,overrides)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with open(temp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def digest(data):
    if not isinstance(data, bytes):
        data = json.dumps(data, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def run_manifest(config, mode):
    import torch, transformers, datasets, sys
    sources = {p.name: digest(p.read_bytes()) for p in sorted(ROOT.glob('*.py'))}
    sources['reference/watermark_processor.py'] = digest((ROOT/'reference/watermark_processor.py').read_bytes())
    assets = {k: read_json(ROOT/config[k]['directory']/'verified_files.json') for k in ('model', 'nli')}
    return {'protocol': config, 'mode': mode, 'sources': sources, 'asset_checksums': assets,
            'versions': {'python':sys.version, 'torch':torch.__version__, 'transformers':transformers.__version__, 'datasets':datasets.__version__},
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}


def lock_manifest(out, manifest):
    out.mkdir(parents=True, exist_ok=True)
    p = out/'manifest.json'
    identity = digest(manifest)
    if p.exists():
        previous = read_json(p)
        if previous['identity'] != identity:
            raise RuntimeError('Resume identity differs: do not mix changed code/config with existing measurements')
    else:
        # Retain the executed source so future extensions cannot erase provenance.
        for relative,expected in manifest.get('sources',{}).items():
            src=ROOT/relative
            if digest(src.read_bytes())!=expected:raise RuntimeError('Source changed while freezing run')
            dst=out/'source'/relative;dst.parent.mkdir(parents=True,exist_ok=True)
            dst.write_bytes(src.read_bytes())
        atomic_json(out/'resolved_protocol.json',manifest.get('protocol',{}))
        atomic_json(p, {'identity':identity, **manifest})
    return identity


def stage_status(out, stage, **kwargs):
    atomic_json(out/'status.json', {'stage':stage,'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'), **kwargs})


def make_samples(config, out, tokenizer, smoke=False):
    from datasets import Dataset, load_dataset
    dest = out/'samples.json'
    if dest.exists():
        return read_json(dest)
    if config.get('experiment_id') and not smoke:
        # Paired sensitivity analysis: use exactly the already-selected documents,
        # not a new draw from the test set. Preserve the complete visible inputs.
        original=ROOT/'runs/formal'
        old=read_json(original/'manifest.json')['protocol']
        if old['dataset']!=config['dataset'] or old['model']!=config['model'] or old['generation']['max_source_tokens']!=config['generation']['max_source_tokens']:
            raise ValueError('Supplement is not compatible with original sample selection')
        samples=read_json(original/'samples.json')
        assert len(samples)==config['dataset']['calibration_n']+config['dataset']['test_n']
        atomic_json(dest,samples)
        atomic_json(out/'dataset_provenance.json',{
            'inherited_from':str(original/'samples.json'),
            'sample_manifest_sha256':digest((original/'samples.json').read_bytes()),
            'split_counts':{role:sum(s['role']==role for s in samples) for role in ('calibration','test')}})
        LOG.info('DATA_REUSED exact original selection documents=%d hash=%s',len(samples),digest((original/'samples.json').read_bytes()))
        return samples
    spec=config['dataset']; samples=[]; provenance={}
    for public_split, role, n in [('validation','calibration',spec['calibration_n']),('test','test',spec['test_n'])]:
        cache = Path.home()/'.cache/huggingface/datasets/abisee___cnn_dailymail/3.0.0/0.0.0'/spec['revision']/f'cnn_dailymail-{public_split}.arrow'
        if cache.exists():
            LOG.info('DATA local complete split=%s path=%s',public_split,cache)
            ds=Dataset.from_file(str(cache))
            import prepare
            provenance[role]={'path':str(cache),'sha256':prepare.sha256(cache),'rows':len(ds)}
        else:
            LOG.info('DATA loading pinned %s split=%s',spec['repo'],public_split)
            ds=load_dataset(spec['repo'],spec['config'],revision=spec['revision'],split=public_split)
            provenance[role]={'fingerprint':ds._fingerprint,'rows':len(ds)}
        expected={'validation':13368,'test':11490}[public_split]
        if len(ds)!=expected:
            raise ValueError(f'Unexpected dataset split size: {len(ds)} != {expected}')
        indices=list(range(len(ds))); random.Random(spec['selection_seed']).shuffle(indices)
        if smoke:
            n=4
        for idx in indices[:n]:
            s=ds[idx]
            ids=tokenizer(s['article'],truncation=True,max_length=config['generation']['max_source_tokens'])['input_ids']
            visible=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            samples.append({'id':s['id'],'dataset_index':idx,'dataset_split':public_split,'role':role,
                            'source':s['article'],'reference':s['highlights'],
                            'source_input_ids':ids,'visible_source':visible,'source_sha256':digest(s['article'])})
    assert len({s['id'] for s in samples})==len(samples), 'Duplicate document across splits'
    atomic_json(dest,samples); atomic_json(out/'dataset_provenance.json',provenance)
    LOG.info('DATA_READY documents=%d (full splits cached; experiment uses fixed disjoint samples)',len(samples))
    return samples
