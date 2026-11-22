"""Slurm-only GRIB header inventory; no weather-value decoding or forecast evaluation."""
import argparse
from collections import Counter
import json
import os

import pandas as pd

from direction_pipeline import output_transaction, sha256, code_hashes
from temporal_folds import project_path


def summarize_year(times, year):
    expected = pd.date_range(f'{year}-01-01', f'{year+1}-01-01', freq='3h', inclusive='left')
    dates = pd.DatetimeIndex([t for t in times if t.year == year])
    duplicates = int(len(dates)-dates.nunique())
    missing = expected.difference(dates)
    extra = dates.difference(expected)
    return dict(year=year, expected_steps=len(expected), selected_steps=len(dates), duplicate_steps=duplicates,
                missing_steps=len(missing), off_schedule_steps=len(extra),
                missing_examples=[str(v) for v in missing[:16]],
                headers_complete=len(dates)==len(expected) and not duplicates and not len(missing) and not len(extra))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grib', nargs='+', type=project_path)
    args = parser.parse_args()
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        parser.error('Submit slurm/run_newyear_audit.sh; do not scan GRIB archives on login nodes.')
    paths = args.grib or [project_path(p) for p in sorted(project_path('data/CERRA').rglob('*.grib'))]
    if not paths or len(set(paths)) != len(paths) or any(not p.is_file() for p in paths):
        raise ValueError('Require distinct existing project GRIB files.')
    provenance = json.loads(project_path('result/rs_hpc_v3/P0/grib_provenance.json').read_text())
    expected_header = provenance['sampled_headers'][0]
    keys = ['gridType','Nx','Ny','DxInMetres','DyInMetres','LoVInDegrees','LaDInDegrees',
            'Latin1InDegrees','Latin2InDegrees','latitudeOfFirstGridPointInDegrees',
            'longitudeOfFirstGridPointInDegrees','shapeOfTheEarth','scanningMode','radius']
    before = {str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths}
    fingerprint = dict(input_stat=before, reference_sha256=sha256(project_path('result/rs_hpc_v3/P0/grib_provenance.json')),
                       code=code_hashes(['audit_newyear_headers.py','direction_pipeline.py']))
    output = project_path(f'result/newyear_audit/headers_{job}')
    with output_transaction(output, fingerprint) as work:
        if work is None:
            return
        import eccodes
        times, files, incompatible = [], [], []
        for path in paths:
            counts = Counter()
            with path.open('rb') as stream:
                while True:
                    message = eccodes.codes_grib_new_from_file(stream, headers_only=True)
                    if message is None:
                        break
                    try:
                        counts['messages'] += 1
                        if int(eccodes.codes_get(message,'paramId')) != 207:
                            continue
                        counts['wind_messages'] += 1
                        header = {k:eccodes.codes_get(message,k) for k in keys}
                        matched = all(header[k] == expected_header[k] for k in keys)
                        product = (eccodes.codes_get(message,'dataType') == 'an'
                                   and eccodes.codes_get(message,'typeOfLevel') == 'heightAboveGround'
                                   and int(eccodes.codes_get(message,'level')) == 10
                                   and str(eccodes.codes_get(message,'stepRange')) == '0'
                                   and eccodes.codes_get(message,'stepType') == 'instant'
                                   and eccodes.codes_get(message,'units') == expected_header['units'])
                        if not product or not matched:
                            counts['incompatible_wind_messages'] += 1
                            if len(incompatible)<16:
                                incompatible.append(dict(file=str(path.relative_to(project_path('.'))),header=header,
                                                         product_matches=product,grid_matches=matched))
                            continue
                        date = str(eccodes.codes_get(message,'validityDate'))
                        clock = str(eccodes.codes_get(message,'validityTime')).zfill(4)
                        times.append(pd.to_datetime(date+clock,format='%Y%m%d%H%M'))
                        counts['matching_wind_messages'] += 1
                    finally:
                        eccodes.codes_release(message)
            files.append(dict(path=str(path.relative_to(project_path('.'))),**counts))
        after = {str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths}
        if before != after:
            raise ValueError('Source GRIB files changed during inventory.')
        years = sorted(set([2023,2024,2025]+[t.year for t in times]))
        report = dict(status='header_inventory_only',files=files,years=[summarize_year(times,y) for y in years],
            incompatible_wind_messages=sum(f.get('incompatible_wind_messages',0) for f in files),
            incompatible_examples=incompatible,
            limitations='No values decoded, no target statistics or errors inspected, no extraction READY or protocol freeze. '
                        'Duplicate timestamps across overlapping files require an explicit reviewed source manifest; no silent deduplication.')
        (work/'inventory.json').write_text(json.dumps(report,indent=2))
    print(f'[New-year headers] Completed: {output}')


if __name__ == '__main__':
    main()
