"""Fetch pinned public assets with resumed downloads and publisher LFS hashes."""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    a = p.parse_args()
    assets = json.loads(a.manifest.read_text())
    jobs = []
    for repo in assets:
        folder = a.root / ('models' if repo['kind'] == 'model' else 'data') / repo['id'].split('/')[-1]
        prefix = 'datasets/' if repo['kind'] == 'dataset' else ''
        for f in repo['files']:
            name = f['rfilename']
            if name.startswith('assets/') or name.endswith('.md'):
                continue
            jobs.append((folder / name, f'https://hf-mirror.com/{prefix}{repo["id"]}/resolve/{repo["revision"]}/{name}', f.get('lfs', {}).get('sha256'), f.get('size'), repo['id'], repo['revision']))
    jobs += [(a.root/'data/ESC-50.zip', 'https://codeload.github.com/karolpiczak/ESC-50/zip/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6', None, None, 'karolpiczak/ESC-50', '33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6'),
             (a.root/'data/dev_fan.zip', 'https://zenodo.org/records/19336329/files/dev_fan.zip?download=1', None, None, 'zenodo/19336329', 'v1')]
    def get(job):
        path, url, expected, size, repo, revision = job
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or (size and path.stat().st_size != size):
            log = path.with_name(path.name + '.download.log')
            with log.open('a') as out:
                result = subprocess.run(['aria2c', '-c', '-x', '4', '-s', '4', '-k', '4M', '--file-allocation=none', '--max-tries=5', '--retry-wait=5', '--timeout=60', '--console-log-level=warn', '--summary-interval=0', '-d', str(path.parent), '-o', path.name, url], stdout=out, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'download failed {path}; log={log}')
        digest = sha(path)
        if expected and digest != expected:
            raise ValueError('publisher LFS hash mismatch: ' + str(path))
        row = {'path': str(path.relative_to(a.root)), 'source_repo': repo, 'revision': revision, 'source_url': url, 'bytes': path.stat().st_size, 'sha256': digest, 'publisher_sha256': expected}
        print('VERIFIED', row['path'], row['bytes'], flush=True)
        return row
    start = time.monotonic()
    receipt = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(get, jobs):
            receipt.append(row)
            (a.root/'asset_receipt.json').write_text(json.dumps(receipt, indent=2))
    print('asset_download_seconds', time.monotonic()-start, flush=True)


if __name__ == '__main__':
    main()
