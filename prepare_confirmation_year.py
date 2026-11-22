"""Initialize, extract/QA, then freeze an independent-year protocol without scoring it."""
import argparse
from datetime import datetime, timezone
import json
import os

import numpy as np
import pandas as pd

from confirmation_protocol import (DEFAULT_REQUEST, GRID_KEYS, GROUPS, SCIENCE_CODE,
    make_folds, read_json, source_stats, validate_product, validate_request, versions, write_new)
from direction_pipeline import code_hashes, output_transaction, sha256, verify_artifacts
from temporal_folds import expected_fold_counts, FoldSpec, project_path


def initialize(args):
    template_path = 'configs/rs_hpc_v3_p3b_v1.json'
    template = read_json(template_path)
    gribs = [str(project_path(p)) for p in args.grib]
    if len(set(gribs)) != len(gribs) or any(not project_path(p).is_file() for p in gribs):
        raise ValueError('Require distinct existing project GRIB sources.')
    version = args.version
    metadata = ['data/CERRA/spatial_2024_full/centers.csv',
                'data/CERRA/spatial_2024_full/grid_mapping.csv',
                'result/rs_hpc_v3/P0/grib_provenance.json', template['direction_selection'],
                'data/rs_static/features_v1/READY.json', 'data/rs_static/features_v1/completion.json',
                'data/rs_static/features_v1/feature_metadata.json',
                'data/rs_static/features_v1/static_features.csv',
                'data/rs_static/features_v1/point_grid_mapping.csv']
    request = dict(status='planned_not_frozen', version=version, year=args.year,
        year_basis=args.year_basis, mode='refit_new_year_only', grib_files=gribs,
        groups=GROUPS, seeds=[42, 43, 44], folds=make_folds(args.year), training_fits=45,
        training_template=template_path, training_template_sha256=sha256(project_path(template_path)),
        data_dir=f'data/CERRA/{version}_{args.year}', run_dir=f'runs/{version}_{args.year}',
        result_dir=f'result/{version}_{args.year}', fold_manifest=f'configs/{version}/folds.json',
        bound_metadata=code_hashes(metadata),
        primary=['RS_S1', 'RS_S0', 'RMSE'], secondary=['RS_S1', 'S_COORD'],
        date_assignment='forecast origin UTC date', block_days=3, resamples=2000,
        bootstrap_seed=20261119, window_aggregate='equal window mean of within-seed gains',
        missing_policy='fail; no imputation or point/date/seed deletion',
        scope='Known-grid retrospective CERRA 10m forecasts; no operational or unseen-location claim')
    validate_request(request)
    path = project_path(f'configs/{version}/request.json')
    if path.exists() or project_path(request['fold_manifest']).exists():
        raise FileExistsError('Version already initialized; retain it and choose another --version.')
    write_new(path, request)
    write_new(request['fold_manifest'], dict(data_year=args.year, train_days=60,
        validation_days=28, test_days=28, folds=request['folds']))
    print(f'Initialized {path}; NOT frozen, no GRIB values read. Audit the listed files next.')


def selected_header(message, eccodes):
    keys = GRID_KEYS + ['paramId', 'dataType', 'typeOfLevel', 'level', 'stepType', 'stepRange', 'units']
    return {key: eccodes.codes_get(message, key) for key in keys}


def extract(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Extraction and numeric QA require manual sbatch.')
    import eccodes
    request = read_json(args.request)
    template = validate_request(request)
    audit_dir = project_path(args.audit)
    audit_marker = verify_artifacts(audit_dir)
    audit = read_json(audit_dir/'inventory.json')
    paths = request['grib_files']
    if audit_marker['fingerprint']['input_stat'] != source_stats(paths):
        raise ValueError('Header inventory does not bind exactly these source files/stats.')
    year_report = [r for r in audit['years'] if r['year'] == request['year']]
    if len(year_report) != 1 or not year_report[0]['headers_complete']:
        raise ValueError('Selected year is missing, duplicate or incomplete in the header audit.')
    if audit['incompatible_wind_messages']:
        raise ValueError('Select files containing only compatible analysis wind products, then re-audit.')
    before = source_stats(paths)
    raw_hashes = {p: sha256(project_path(p)) for p in paths}
    fingerprint = dict(request_sha256=sha256(project_path(args.request)), raw_sha256=raw_hashes,
                       audit_completion_sha256=sha256(audit_dir/'completion.json'),
                       code=code_hashes(['prepare_confirmation_year.py', 'confirmation_protocol.py']))
    destination = project_path(request['data_dir'])
    with output_transaction(destination, fingerprint) as work:
        if work is None:
            print(f'Existing extraction verified: {destination}')
            return
        centers = pd.read_csv(project_path('data/CERRA/spatial_2024_full/centers.csv'))
        mapping = pd.read_csv(project_path('data/CERRA/spatial_2024_full/grid_mapping.csv'))
        if len(centers) != 100 or centers.point_id.duplicated().any() or len(
                centers[['center_y', 'center_x']].drop_duplicates()) != 98:
            raise ValueError('Changed fixed target mapping.')
        reference = read_json('result/rs_hpc_v3/P0/grib_provenance.json')['sampled_headers'][0]
        cells = mapping[['grid_y', 'grid_x', 'grid_lat', 'grid_lon']].drop_duplicates()
        if cells.duplicated(['grid_y', 'grid_x']).any():
            raise ValueError('Inconsistent native cell coordinates.')
        yy, xx = cells.grid_y.to_numpy(dtype=int), cells.grid_x.to_numpy(dtype=int)
        if (yy < 0).any() or (xx < 0).any() or (yy >= reference['Ny']).any() or (xx >= reference['Nx']).any():
            raise ValueError('Neighborhood crosses native grid boundary.')
        flat = yy*reference['Nx']+xx
        lookup = {tuple(v): i for i, v in enumerate(cells[['grid_y', 'grid_x']].to_numpy())}
        wind_features = template['dynamic_groups']['A01']['features'][:-2]
        if template['dynamic_groups']['A01']['features'][-2:] != ['hour_sin', 'hour_cos']:
            raise ValueError('Unexpected feature order.')
        selectors = []
        for point in centers.point_id:
            selected = mapping[(mapping.point_id == point) & ((mapping.kind == 'far') |
                              ((mapping.dy == 0) & (mapping.dx == 0)))].set_index('feature', verify_integrity=True)
            selected = selected.loc[wind_features]
            selectors.append([lookup[(row.grid_y, row.grid_x)] for row in selected.itertuples()])
        dates = pd.date_range(f'{request["year"]}-01-01', f'{request["year"]+1}-01-01',
                              freq='3h', inclusive='left')
        positions = {time: i for i, time in enumerate(dates)}
        values = np.empty((len(dates), len(cells)), dtype=np.float32)
        seen = set()
        checks = []
        checked_coordinates = False
        for filename in paths:
            with project_path(filename).open('rb') as stream:
                while True:
                    message = eccodes.codes_grib_new_from_file(stream)
                    if message is None:
                        break
                    try:
                        if int(eccodes.codes_get(message, 'paramId')) != 207:
                            continue
                        validate_product(selected_header(message, eccodes), reference)
                        time = pd.to_datetime(str(eccodes.codes_get(message, 'validityDate'))+
                            str(eccodes.codes_get(message, 'validityTime')).zfill(4), format='%Y%m%d%H%M')
                        if time.year != request['year']:
                            continue
                        if time not in positions or time in seen:
                            raise ValueError('Duplicate or off-schedule selected analysis.')
                        if not checked_coordinates:
                            lat = eccodes.codes_get_array(message, 'latitudes')[flat]
                            lon = (eccodes.codes_get_array(message, 'longitudes')[flat]+180) % 360-180
                            np.testing.assert_allclose(lat, cells.grid_lat, atol=1e-7, rtol=0)
                            np.testing.assert_allclose(lon, cells.grid_lon, atol=1e-7, rtol=0)
                            checked_coordinates = True
                        decoded = eccodes.codes_get_values(message)
                        if len(decoded) != reference['Ny']*reference['Nx']:
                            raise ValueError('Wrong field shape.')
                        sampled = decoded[flat]
                        if not np.isfinite(sampled).all() or (sampled < 0).any() or (sampled > 200).any():
                            raise ValueError('Missing/nonfinite/negative/implausible wind values.')
                        index = positions[time]
                        values[index] = sampled
                        # Independent point-access API versus row-major full-field indexing.
                        if index in (0, len(dates)//2, len(dates)-1):
                            direct = eccodes.codes_get_double_elements(message, 'values', flat[:32].tolist())
                            np.testing.assert_allclose(direct, values[index, :32], rtol=1e-6, atol=1e-5)
                            checks.append(dict(datetime=str(time), elements=32,
                                max_abs_difference=float(np.max(np.abs(direct-values[index, :32])))))
                        seen.add(time)
                        if len(seen) % 100 == 0:
                            print(f'[Confirmation extraction] {len(seen)}/{len(dates)}', flush=True)
                    finally:
                        eccodes.codes_release(message)
        if len(seen) != len(dates) or len(checks) != 3:
            raise ValueError('Incomplete yearly extraction or independent point checks.')
        panel_values = values[:, np.asarray(selectors)]
        frames = []
        for i, point in enumerate(centers.point_id):
            frame = pd.DataFrame(panel_values[:, i, :], columns=wind_features)
            frame.insert(0, 'point_id', point)
            frame.insert(0, 'datetime', dates)
            frames.append(frame)
        panel = pd.concat(frames, ignore_index=True)
        panel.to_csv(work/'A01.csv', index=False)
        reread = pd.read_csv(work/'A01.csv', parse_dates=['datetime'])
        if reread.duplicated(['point_id', 'datetime']).any() or len(reread) != len(dates)*100:
            raise ValueError('CSV serialization changed coverage.')
        np.testing.assert_allclose(reread[wind_features], panel[wind_features], rtol=1e-6, atol=1e-5)
        centers.to_csv(work/'centers.csv', index=False)
        mapping.to_csv(work/'grid_mapping.csv', index=False)
        coverage = [dict(fold_id=f['fold_id'], **expected_fold_counts(FoldSpec.from_dict(f), 56, 1, 100))
                    for f in request['folds']]
        if source_stats(paths) != before:
            raise ValueError('GRIB source changed while extracting.')
        ready = dict(status='passed', year=request['year'], units='m/s',
            time_steps=len(dates), points=100, unique_grids=98, shape=[100, len(dates), 9],
            feature_order=template['dynamic_groups']['A01']['features'],
            A01_sha256=sha256(work/'A01.csv'), request_sha256=sha256(project_path(args.request)),
            raw_sha256=raw_hashes, raw_source_stats=before, coordinate_check='all mapped cells',
            independent_point_checks=checks, fold_counts=coverage,
            scope='A01-only fixed native mapping; no cross-year equality of weather values; no scores inspected')
        (work/'READY.json').write_text(json.dumps(ready, indent=2)+'\n')
        pd.DataFrame(coverage).to_csv(work/'fold_coverage.csv', index=False)
    print(f'[Confirmation extraction] READY: {destination}')


def freeze(args):
    if not args.attest_unseen:
        raise ValueError('--attest-unseen must truthfully confirm no selected-year model errors were inspected.')
    request_path = project_path(args.request)
    request = read_json(request_path)
    template = validate_request(request)
    data = project_path(request['data_dir'])
    marker = read_json(data/'completion.json')
    ready = read_json(data/'READY.json')
    if sha256(data/'READY.json') != marker['artifacts']['READY.json'] or (
            ready['status'] != 'passed' or ready['year'] != request['year'] or
            ready['request_sha256'] != sha256(request_path)):
        raise ValueError('Missing or incompatible numeric QA.')
    if marker['fingerprint']['request_sha256'] != sha256(request_path) or marker['fingerprint']['code'] != code_hashes(
            ['prepare_confirmation_year.py', 'confirmation_protocol.py']):
        raise ValueError('Extraction code/request changed since numeric QA.')
    if source_stats(request['grib_files']) != ready['raw_source_stats']:
        raise ValueError('Raw source stats changed since numeric QA.')
    actual_folds = read_json(request['fold_manifest'])
    if actual_folds['folds'] != request['folds']:
        raise ValueError('Fold file disagrees with request.')
    # Static products are frozen descriptions, not dynamically available RS observations.
    from train_p3 import load_static_bundle
    load_static_bundle(template)
    if request['year'] <= 2022:
        raise ValueError('WorldCover release availability needs a separate protocol for earlier years.')
    protocol = dict(status='frozen', frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        request_sha256=sha256(request_path), fold_manifest_sha256=sha256(project_path(request['fold_manifest'])),
        extraction_completion_sha256=sha256(data/'completion.json'),
        extracted_stats={name: source_stats([data/name])[str(data/name)] for name in marker['artifacts']},
        raw_source_stats=ready['raw_source_stats'], raw_sha256=ready['raw_sha256'],
        code_sha256=code_hashes(SCIENCE_CODE), versions=versions(),
        unseen_attestation='Operator affirms selected year has not been used for model-error analysis or selection.',
        static_time_scope='WorldCover 2021 v200 released 2022; same frozen DSM-derived static bundle. '
                          'Retrospective descriptors; no operational RS availability or unchanged land-cover claim.',
        primary=request['primary'], failure_policy='Keep all groups/seeds/windows, including negative results. '
            'Infrastructure-only retry with identical protocol; scientific edits require a new version and disclose test reuse.')
    if project_path(request['run_dir']).exists():
        raise ValueError('Run directory already exists: do not freeze retrospectively.')
    write_new(request_path.parent/'FROZEN.json', protocol)
    print(f'FROZEN: {request_path.parent}/FROZEN.json; 45 new fits, no training executed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    init = sub.add_parser('init')
    init.add_argument('--year', type=int, choices=[2023, 2025], default=2025)
    init.add_argument('--grib', nargs='+', required=True)
    init.add_argument('--version', default='confirmation_v1')
    init.add_argument('--year-basis', required=True, help='Data-availability reason only, not model performance.')
    for name in ['extract', 'freeze']:
        command = sub.add_parser(name)
        command.add_argument('--request', default=DEFAULT_REQUEST)
        if name == 'extract':
            command.add_argument('--audit', required=True)
        else:
            command.add_argument('--attest-unseen', action='store_true')
    args = parser.parse_args()
    {'init': initialize, 'extract': extract, 'freeze': freeze}[args.mode](args)


if __name__ == '__main__':
    main()
