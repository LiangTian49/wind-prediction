"""Monthly, Slurm-only CERRA analysis downloads with header QA and resumable task IDs."""
import argparse
import calendar
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
import uuid

import pandas as pd

from confirmation_protocol import GRID_KEYS
from direction_pipeline import sha256
from temporal_folds import project_path
from cerra_download_remote import resume_or_submit

DATASET = 'reanalysis-cerra-single-levels'
YEARS = (2025, 2023)
TASKS = [(year, month) for year in YEARS for month in range(1, 13)]
API_URL = 'https://cds.climate.copernicus.eu/api'


def monthly_request(year, month):
    if year not in YEARS or month not in range(1, 13):
        raise ValueError('Only the declared 2025/2023 calendar months are supported.')
    return dict(variable=['10m_wind_direction', '10m_wind_speed'],
        level_type='surface_or_atmosphere', data_type=['reanalysis'], product_type='analysis',
        year=[str(year)], month=[f'{month:02d}'],
        day=[f'{day:02d}' for day in range(1, calendar.monthrange(year, month)[1]+1)],
        time=[f'{hour:02d}:00' for hour in range(0, 24, 3)], data_format='grib')


def validate_header(header, reference):
    if any(header[k] != reference[k] for k in GRID_KEYS):
        raise ValueError('Grid changed; full native CERRA grid required.')
    for key, value in dict(dataType='an', typeOfLevel='heightAboveGround', level=10,
                           stepType='instant', stepRange='0').items():
        if str(header[key]) != str(value):
            raise ValueError(f'Unexpected analysis product: {key}')
    if header['paramId'] == 207:
        if header['units'] != reference['units']:
            raise ValueError('Wrong wind-speed units.')
    elif header['paramId'] == 260260:
        if 'degree' not in header['units'].lower():
            raise ValueError('Wrong wind-direction units.')
    else:
        raise ValueError('Unrequested variable in monthly file.')


def audit_month(path, year, month):
    import eccodes
    reference = json.loads(project_path('result/rs_hpc_v3/P0/grib_provenance.json').read_text())['sampled_headers'][0]
    expected = pd.date_range(f'{year}-{month:02d}-01', periods=calendar.monthrange(year, month)[1]*8, freq='3h')
    times = {207: [], 260260: []}
    with path.open('rb') as stream:
        while True:
            message = eccodes.codes_grib_new_from_file(stream, headers_only=True)
            if message is None:
                break
            try:
                header = {k: eccodes.codes_get(message, k) for k in GRID_KEYS+
                          ['paramId', 'dataType', 'typeOfLevel', 'level', 'stepType', 'stepRange', 'units']}
                validate_header(header, reference)
                time = pd.to_datetime(str(eccodes.codes_get(message, 'validityDate'))+
                    str(eccodes.codes_get(message, 'validityTime')).zfill(4), format='%Y%m%d%H%M')
                times[header['paramId']].append(time)
            finally:
                eccodes.codes_release(message)
    for parameter, dates in times.items():
        if not pd.DatetimeIndex(dates).sort_values().equals(expected):
            raise ValueError(f'Missing/duplicate/wrong-month times for parameter {parameter}.')
    return dict(status='headers_passed', records_per_variable=len(expected), parameters=[207, 260260],
                grid='matches frozen 2024 native grid', data_type='an', step_hours=0,
                numeric_values_decoded=False)


def download(index):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Downloads and full-file QA require manual sbatch.')
    year, month = TASKS[index]
    request = monthly_request(year, month)
    directory = project_path(f'data/CERRA/raw_{year}')
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/f'cerra_{year}_{month:02d}.grib'
    receipt = target.with_suffix('.download.json')
    with target.with_suffix('.download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            if not receipt.exists():
                raise ValueError('Existing GRIB has no receipt: preserve and review it; never silently overwrite.')
            saved = json.loads(receipt.read_text())
            if saved['status'] != 'completed' or saved['request'] != request or saved['sha256'] != sha256(target):
                raise ValueError('Existing monthly file/request/checksum changed.')
            audit_month(target, year, month)
            print(f'[CDS] Verified existing month: {year}-{month:02d}; no new request.')
            return
        if receipt.exists():
            raise ValueError('Receipt exists but GRIB is missing; preserve provenance and investigate.')
        key = os.environ.get('CDSAPI_KEY', '').strip()
        if not key:
            raise ValueError('Set CDSAPI_KEY in the submitting shell; no credentials read outside the project.')
        try:
            import cdsapi
        except ImportError as exc:
            raise RuntimeError('wind-env lacks cdsapi; first submit slurm/run_cerra_download_setup.sh.') from exc
        # Explicit credentials and a project-local RC path prevent home-directory reads.
        os.environ['CDSAPI_RC'] = str(project_path('environment/cdsapi_unused.rc'))
        client = cdsapi.Client(url=API_URL, key=key, verify=True, quiet=True,
                               debug=False, progress=False, timeout=120, retry_max=5, sleep_max=60,
                               wait_until_complete=False)
        partial = target.with_suffix('.grib.part')
        if partial.exists():
            partial.rename(directory/f'{partial.name}.preserved_{uuid.uuid4().hex}')
        attempt = directory/f'cerra_{year}_{month:02d}.attempt_{os.environ["SLURM_JOB_ID"]}_{uuid.uuid4().hex}.json'
        attempt.write_text(json.dumps(dict(status='request_started', dataset=DATASET, request=request,
            slurm_job_id=os.environ['SLURM_JOB_ID'], started_at_utc=datetime.now(timezone.utc).isoformat()), indent=2)+'\n')
        print(f'[CDS] Requesting {year}-{month:02d}, analysis, speed+direction, full native grid.', flush=True)
        remote = resume_or_submit(client, DATASET, request, target.with_suffix('.remote.json'))
        remote.download(str(partial))
        audit = audit_month(partial, year, month)
        digest = sha256(partial)
        saved = dict(status='completed', dataset=DATASET, request=request, year=year, month=month,
            file=str(target.relative_to(project_path('.'))), bytes=partial.stat().st_size, sha256=digest,
            headers=audit, completed_at_utc=datetime.now(timezone.utc).isoformat(),
            slurm_job_id=os.environ['SLURM_JOB_ID'], cdsapi_version=importlib.metadata.version('cdsapi'),
            remote_request_id=remote.request_id,
            downloader_sha256=sha256(project_path(__file__)),
            scope='Download/header QA only; no numeric extraction, errors inspected or confirmation freeze.')
        partial.rename(target)
        with receipt.open('x') as stream:
            stream.write(json.dumps(saved, indent=2)+'\n')
        print(f'[CDS] Completed and header-checked: {target.name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--task', type=int, choices=range(24))
    mode.add_argument('--list', action='store_true')
    mode.add_argument('--request', type=int, choices=range(24), help='Print request only; no network/authentication')
    args = parser.parse_args()
    if args.list:
        for i, (year, month) in enumerate(TASKS):
            print(i, year, f'{month:02d}')
    elif args.request is not None:
        print(json.dumps(monthly_request(*TASKS[args.request]), indent=2))
    else:
        download(args.task)


if __name__ == '__main__':
    main()
