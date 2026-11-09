"""Pinned, checksum-verified downloads with byte progress, retries and Range resume.
No credentials, SSL bypass, external code execution, or global environment changes.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger('download')
ENDPOINT = 'https://hf-mirror.com'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(url, dst, size=None, digest=None, tries=20):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    label = f'{dst.parent.name}/{dst.name}'
    if dst.exists() and (size is None or dst.stat().st_size == size):
        if not digest or sha256(dst) == digest:
            LOG.info('CACHED %s bytes=%s', label, dst.stat().st_size)
            return
    partial = dst.with_name(dst.name + '.part')
    for attempt in range(1, tries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            if size and offset == size:
                if digest and sha256(partial) != digest:
                    partial.unlink()
                    raise ValueError('Partial checksum mismatch; discarded only this download partial')
                os.replace(partial, dst)
                LOG.info('VERIFIED %s bytes=%s', label, size)
                return
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            LOG.info('REQUEST %s attempt=%d/%d resume_bytes=%d', label, attempt, tries, offset)
            with requests.get(url, headers=headers, stream=True, timeout=(20, 40)) as r:
                r.raise_for_status()
                if offset and r.status_code == 206:
                    if not r.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Server returned wrong byte range')
                    mode = 'ab'
                else:
                    mode, offset = 'wb', 0
                total = size or (int(r.headers.get('Content-Length', 0)) + offset)
                t0 = last = time.monotonic()
                initial = current = offset
                with open(partial, mode) as f:
                    for block in r.iter_content(chunk_size=1024 * 1024):
                        if not block:
                            continue
                        f.write(block)
                        current += len(block)
                        now = time.monotonic()
                        if now - last >= 5:
                            f.flush()
                            speed = (current - initial) / max(now - t0, 1e-6)
                            eta = (total-current)/max(speed, 1) if total else None
                            LOG.info('PROGRESS %s %.1f%% %.1f/%.1f MiB %.2f MiB/s ETA=%ss',
                                     label, 100*current/total if total else 0,
                                     current/2**20, total/2**20, speed/2**20,
                                     round(eta) if eta is not None else '?')
                            last = now
                    f.flush()
                    os.fsync(f.fileno())
            if size and partial.stat().st_size != size:
                raise OSError('Incomplete body; will resume')
            if digest and sha256(partial) != digest:
                partial.unlink()
                raise ValueError('SHA256 mismatch; redownload required')
            os.replace(partial, dst)
            LOG.info('VERIFIED %s bytes=%d sha256=%s', label, dst.stat().st_size, sha256(dst))
            return
        except (requests.RequestException, OSError, ValueError) as exc:
            if attempt == tries:
                raise
            delay = min(300, 5 * 2 ** min(attempt-1, 6))
            LOG.warning('RETRY %s error=%s delay=%ss partial_bytes=%d', label, str(exc)[:250], delay,
                        partial.stat().st_size if partial.exists() else 0)
            time.sleep(delay)


def prepare_models():
    from concurrent.futures import ThreadPoolExecutor
    config = json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    def fetch_model(key):
        spec = config[key]
        dest = ROOT/spec['directory']
        meta_path = dest/'download_metadata.json'
        download(f'{ENDPOINT}/api/models/{spec["repo"]}/revision/{spec["revision"]}?blobs=true', meta_path)
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['sha'] != spec['revision']:
            raise ValueError('Model revision mismatch')
        siblings = {x['rfilename']: x for x in meta['siblings']}
        manifest = {}
        for filename in spec['files']:
            entry = siblings[filename]
            lfs = entry.get('lfs') or {}
            size = entry.get('size') or lfs.get('size')
            digest = lfs.get('sha256')
            download(f'{ENDPOINT}/{spec["repo"]}/resolve/{spec["revision"]}/{filename}', dest/filename, size, digest)
            manifest[filename] = {'sha256': sha256(dest/filename), 'bytes': (dest/filename).stat().st_size}
        manifest_path = dest/'verified_files.json'
        temporary = dest/'verified_files.json.tmp'
        temporary.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        os.replace(temporary, manifest_path)
        LOG.info('MODEL_READY %s revision=%s', spec['repo'], spec['revision'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch_model, ('model', 'nli')))
    LOG.info('ALL_DOWNLOADS_COMPLETE')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
    prepare_models()
