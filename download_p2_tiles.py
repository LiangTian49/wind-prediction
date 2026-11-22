"""Slurm-only download of a pinned, ROI-limited remote-sensing asset manifest."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from time import sleep

import requests

from temporal_folds import project_path

CONFIG = 'configs/rs_hpc_v3_p2_download.json'
CACHE = 'data/rs_static/raw_v1'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, content):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(content, indent=2)+'\n')
    temporary.replace(path)


def fetch_asset(asset, directory):
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / asset['filename']
    receipt = final.with_suffix('.receipt.json')
    if final.exists():
        saved = json.loads(receipt.read_text())
        if (saved['url'] != asset['url'] or saved['etag'] != asset['etag']
                or final.stat().st_size != asset['bytes'] or sha256(final) != saved['sha256']):
            raise ValueError(f'Cached asset differs: {final}')
        return saved
    partial = final.with_suffix('.part')
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > asset['bytes']:
            raise ValueError(f'Partial file exceeds pinned length: {partial}')
        if offset == asset['bytes']:
            break
        headers = {'Accept-Encoding': 'identity', 'If-Match': asset['etag']}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            with requests.get(asset['url'], headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if response.headers.get('ETag') != asset['etag']:
                    raise ValueError('Remote ETag changed; preserve cache and review a new manifest version.')
                if response.status_code == 206:
                    expected = f'bytes {offset}-{asset["bytes"]-1}/{asset["bytes"]}'
                    if response.headers.get('Content-Range') != expected:
                        raise ValueError('Server returned an inconsistent byte range.')
                    mode = 'ab' if offset else 'wb'
                elif response.status_code == 200:
                    # A server that ignores Range supplies a complete new temporary file.
                    mode = 'wb'
                else:
                    raise ValueError(f'Unexpected HTTP status: {response.status_code}')
                with partial.open(mode) as stream:
                    for chunk in response.iter_content(chunk_size=1024**2):
                        if chunk:
                            stream.write(chunk)
            if partial.stat().st_size != asset['bytes']:
                raise requests.ConnectionError('Download ended before pinned content length.')
            break
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == 2:
                raise
            sleep(2)
    if not partial.exists() or partial.stat().st_size != asset['bytes']:
        raise ValueError('Incomplete download.')
    with partial.open('rb') as stream:
        if stream.read(4) not in [b'II*\x00', b'MM\x00*', b'II+\x00', b'MM\x00+']:
            raise ValueError('Downloaded object is not a TIFF.')
    saved = dict(asset, sha256=sha256(partial), downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
                 validation='Pinned ETag, exact byte length, TIFF signature, local SHA256.')
    # Write the receipt first so an interruption after publication remains resumable.
    write_json(receipt, saved)
    partial.rename(final)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    config_path = project_path(CONFIG)
    config = json.loads(config_path.read_text())
    assets = config['assets']
    if args.list:
        print(f'{len(assets)} tiles; {sum(a["bytes"] for a in assets)} bytes; 0 model fits')
        for asset in assets:
            print(asset['product'], asset['filename'])
        return
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        parser.error('Submit slurm/run_p2_download.sh.')
    root = project_path(CACHE)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cache_manifest = root / 'source_manifest.json'
        if cache_manifest.exists() and sha256(cache_manifest) != sha256(config_path):
            raise ValueError('Cache manifest differs. Use a separately versioned cache for changed inputs.')
        if not cache_manifest.exists():
            cache_manifest.write_bytes(config_path.read_bytes())
        receipts = []
        for i, asset in enumerate(assets):
            receipt = fetch_asset(asset, root / asset['product'])
            receipts.append(receipt)
            print(f'[P2 download] {i+1}/{len(assets)} {asset["filename"]}', flush=True)
        report = dict(status='downloaded_waiting_geospatial_QA', assets=len(receipts),
                      bytes=sum(r['bytes'] for r in receipts), manifest_sha256=sha256(config_path),
                      cache=str(root), files=receipts,
                      rasterio_available=importlib.util.find_spec('rasterio') is not None,
                      pyproj_available=importlib.util.find_spec('pyproj') is not None,
                      next='Inspect CRS, NoData, vertical metadata and grid footprints before feature extraction.')
        write_json(root / 'download_complete.json', report)
        destination = project_path(f'result/rs_hpc_v3/P2/download_{job}.json')
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, report)
    print(f'[P2 download] Completed download checks: {destination}')


if __name__ == '__main__':
    main()
