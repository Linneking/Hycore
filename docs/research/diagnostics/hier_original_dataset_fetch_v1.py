"""Download and verify public HIER image datasets into a fresh private directory.

Network downloads/extraction use CPU only. No credentials, training, model imports,
or existing result changes. The complete image archives must never enter Git.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import threading
import time
import urllib.request
import zipfile

DATASETS = {
    'cub': {'file': 'CUB_200_2011.tgz', 'urls': [
        'https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz'],
        'bytes': 1150585339, 'prefix_md5': 'd2acaa99439dff0483c7bbac1bfe2a92',
        'checksum_source': 'https://raw.githubusercontent.com/fastai/fastai/master/fastai/data/download_checks.py',
        'jpeg_count': 11788},
    'cars': {'file': 'car_ims.tgz', 'urls': [
        'https://hf-mirror.com/datasets/XiN0919/FGVC/resolve/921e8dce38536cd5982739d85d5e9235d2cb2a76/car_ims.tgz'],
        'bytes': 1956628579, 'sha256': '4d96e3e3e892c6b7e4a71959325c7de4804037d27ba5aaf1c756100fd94a5342',
        'jpeg_count': 16185},
    'sop': {'file': 'Stanford_Online_Products.zip', 'urls': [
        'ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip'],
        'bytes': 3083860082, 'sha256': 'b04fa78210e69fab050ce73939550e1cd0f96890bdd98ec9fee732c3b756ef48',
        'jpeg_count': 120053},
    'cars_annotations': {'file': 'cars_annos_original.mat', 'urls': [
        'https://hf-mirror.com/datasets/XiN0919/FGVC/resolve/921e8dce38536cd5982739d85d5e9235d2cb2a76/cars_annos.mat'],
        'bytes': 394471},
}
LOCK = threading.Lock()


def log(value):
    with LOCK:
        print(json.dumps(value, ensure_ascii=False), flush=True)


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError('Archive contains unsafe path: ' + name)
    return path


def extract_archive(source, destination):
    """Copy regular files only; reject escaping paths, links, and special files."""
    destination.mkdir(exist_ok=False)
    if source.name.endswith('.zip'):
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                safe_name(member.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError('Archive symlink is unsupported.')
            for member in archive.infolist():
                target = destination.joinpath(*safe_name(member.filename).parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as src, target.open('xb') as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
    else:
        with tarfile.open(source, 'r:gz') as archive:
            members = archive.getmembers()
            for member in members:
                safe_name(member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError('Archive link or special file is unsupported: ' + member.name)
            for member in members:
                target = destination.joinpath(*safe_name(member.name).parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as src, target.open('xb') as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)


def download_one(name, spec, root, extract):
    start = time.time()
    target = root / 'archives' / spec['file']
    url = spec['urls'][0]
    digest = hashlib.sha256()
    md5 = hashlib.md5()
    prefix_md5 = hashlib.md5()
    received = 0
    last = start
    req = urllib.request.Request(url, headers={'User-Agent': 'HyCoRe-Research-Dataset-Reader/1.0'})
    with urllib.request.urlopen(req, timeout=60) as stream, target.with_suffix(target.suffix+'.part').open('xb') as out:
        remote_bytes = stream.headers.get('Content-Length')
        log({'dataset': name, 'phase': 'download_started', 'expected_bytes': spec.get('bytes'), 'remote_bytes': remote_bytes})
        while True:
            data = stream.read(4 << 20)
            if not data:
                break
            if received < 2**20:
                prefix_md5.update(data[:2**20-received])
            out.write(data); digest.update(data); md5.update(data); received += len(data)
            if time.time() - last >= 30:
                log({'dataset': name, 'phase': 'downloading', 'bytes': received,
                     'MiB_per_second': round(received / (time.time()-start) / 2**20, 2)})
                last = time.time()
    if 'bytes' in spec and received != spec['bytes']:
        raise ValueError(name+' size mismatch')
    if 'sha256' in spec and digest.hexdigest() != spec['sha256']:
        raise ValueError(name+' SHA256 mismatch')
    if 'md5' in spec and md5.hexdigest() != spec['md5']:
        raise ValueError(name+' official MD5 mismatch')
    if 'prefix_md5' in spec and prefix_md5.hexdigest() != spec['prefix_md5']:
        raise ValueError(name+' published first-MiB MD5 mismatch')
    target.with_suffix(target.suffix+'.part').rename(target)
    result = {'name': name, 'url': url, 'bytes': received, 'sha256': digest.hexdigest(), 'md5': md5.hexdigest(),
              'archive': str(target), 'checksum_verified': bool({'sha256', 'md5', 'prefix_md5'} & spec.keys()),
              'verification': 'full SHA256' if 'sha256' in spec else 'published byte count and first-MiB MD5' if 'prefix_md5' in spec else 'recorded local SHA256; published byte count only',
              'seconds': time.time()-start}
    log({'dataset': name, 'phase': 'download_verified', 'bytes': received})
    if extract and 'jpeg_count' in spec:
        dest = root / 'extracted' / name
        extract_archive(target, dest)
        files = list(dest.rglob('*'))
        jpeg = sum(p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg') for p in files)
        if jpeg != spec['jpeg_count']:
            raise ValueError(f'{name} image count mismatch {jpeg}!={spec["jpeg_count"]}')
        result.update(extracted=str(dest), jpeg_count=jpeg,
                      extracted_file_bytes=sum(p.stat().st_size for p in files if p.is_file()))
        log({'dataset': name, 'phase': 'extraction_verified', 'images': jpeg,
             'extracted_bytes': result['extracted_file_bytes']})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=list(DATASETS))
    ap.add_argument('--no-extract', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'archives').mkdir()
    (args.output/'extracted').mkdir()
    manifest = {'started_utc': datetime.now(timezone.utc).isoformat(), 'gpu_used': False,
                'authorization': 'User explicitly requested downloading the three public image datasets.',
                'results': {}, 'errors': {}}
    with ThreadPoolExecutor(max_workers=3) as executor:
        future = {executor.submit(download_one, n, DATASETS[n], args.output, not args.no_extract): n for n in args.datasets}
        for done in as_completed(future):
            name = future[done]
            try:
                manifest['results'][name] = done.result()
            except Exception as exc:
                manifest['errors'][name] = {'type': type(exc).__name__, 'message': str(exc)}
                log({'dataset': name, 'phase': 'failed', 'error': str(exc)})
            (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    log({'phase': 'finished', 'completed': list(manifest['results']), 'errors': manifest['errors']})
    if manifest['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
