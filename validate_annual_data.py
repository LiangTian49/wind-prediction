"""Slurm-only annual spatial data readiness checks; no models or test scoring."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from extract_cerra_spatial import check_center_mapping, check_reference
from temporal_folds import STEP, expected_fold_counts, load_manifest, project_path, select_fold_panel


ROOT = Path(__file__).resolve().parent


def validate_annual(spatial_dir, manifest_path, report_dir):
    manifest, folds, digest = load_manifest(manifest_path)
    metadata = json.loads((spatial_dir / 'metadata.json').read_text())
    year = int(manifest['data_year'])
    if metadata.get('status') != 'complete' or metadata.get('year') != year:
        raise ValueError('Annual extraction is incomplete or has the wrong year.')
    if metadata.get('units') not in {'m s**-1', 'm s-1', 'm/s', 'm s^-1'}:
        raise ValueError('Wind units are not recognized as m/s.')
    if set(metadata.get('months', [])) != set(range(1, 13)) or not metadata.get('require_complete_months'):
        raise ValueError('Annual extraction must require all twelve complete calendar months.')
    if metadata.get('reference_check', {}).get('matched_rows', 0) <= 0:
        raise ValueError('Annual data lack a passed summer reference check.')
    centers = pd.read_csv(spatial_dir / 'centers.csv')
    check_center_mapping(centers, project_path(manifest['point_mapping']))
    center = pd.read_csv(spatial_dir / 'center.csv', parse_dates=['datetime'])
    if center[['point_id', 'datetime', 'ws_10']].isna().any().any():
        raise ValueError('Center data contain missing values.')
    if center.duplicated(['point_id', 'datetime']).any():
        raise ValueError('Duplicate center point/time records.')
    if set(center.point_id) != set(centers.point_id):
        raise ValueError('Center table point IDs do not match mapping.')
    if not np.isfinite(center.ws_10).all() or (center.ws_10 < 0).any():
        raise ValueError('Invalid center wind speeds.')
    expected = pd.date_range(f'{year}-01-01', f'{year+1}-01-01', freq='3h', inclusive='left')
    for point_id, group in center.groupby('point_id'):
        if not np.array_equal(group.datetime.sort_values().to_numpy(), expected.to_numpy()):
            raise ValueError(f'Incomplete annual time axis at point {point_id}.')
    reference_check = check_reference(center, project_path(metadata['reference_csv']), 'reference_subset')
    key_index = pd.MultiIndex.from_frame(center[['point_id', 'datetime']])
    center_values = center.ws_10.to_numpy()
    files = {}
    feature_sets = metadata['feature_sets']
    for experiment in manifest['groups']:
        path = spatial_dir / ('center.csv' if experiment == 'T01' else f'{experiment}.csv')
        features = feature_sets[experiment]
        wind_columns = [feature for feature in features if feature not in {'hour_sin', 'hour_cos'}]
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if set(header) != {'datetime', 'point_id', *wind_columns}:
            raise ValueError(f'Unexpected columns in {path}.')
        seen = np.zeros(len(center), dtype=bool)
        for chunk in pd.read_csv(path, chunksize=20000, parse_dates=['datetime']):
            values = chunk[wind_columns].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f'Missing/non-finite/negative wind values in {path}.')
            index = pd.MultiIndex.from_frame(chunk[['point_id', 'datetime']])
            positions = key_index.get_indexer(index)
            if (positions < 0).any() or index.has_duplicates or seen[positions].any():
                raise ValueError(f'Duplicate or unexpected point/time records in {path}.')
            if not np.allclose(chunk.ws_10.to_numpy(), center_values[positions], rtol=1e-6, atol=1e-5):
                raise ValueError(f'Center targets differ across feature sets: {path}.')
            seen[positions] = True
        if not seen.all():
            raise ValueError(f'Missing point/time rows in {path}.')
        files[experiment] = {'path': str(path), 'rows': int(seen.sum()),
                             'features': features, 'input_channels': len(features)}
    with np.load(spatial_dir / 'neighborhoods.npz', allow_pickle=False) as tensor:
        side = 2 * int(metadata['radius']) + 1
        values = tensor['ws_10']
        if values.shape != (len(centers), len(expected), side, side) or not np.isfinite(values).all():
            raise ValueError('Invalid neighborhood tensor shape/values.')
        if not np.array_equal(tensor['datetime'], expected.to_numpy()):
            raise ValueError('Tensor timestamps differ from CSV timestamps.')
        if not np.array_equal(tensor['point_id'], centers.point_id.to_numpy()):
            raise ValueError('Tensor point order differs from center mapping.')
        for dimension in ('dy', 'dx'):
            if not np.array_equal(tensor[dimension], np.arange(-int(metadata['radius']), int(metadata['radius']) + 1)):
                raise ValueError(f'Unexpected tensor {dimension} ordering.')
        expected_center = center.pivot(index='point_id', columns='datetime', values='ws_10').reindex(
            index=centers.point_id.to_numpy(), columns=expected).to_numpy()
        radius = int(metadata['radius'])
        if not np.allclose(values[:, :, radius, radius], expected_center, rtol=1e-6, atol=1e-5):
            raise ValueError('Tensor center slice differs from CSV targets.')
    coverage = []
    for fold in folds.values():
        panel = select_fold_panel(center[['point_id', 'datetime']], fold)
        n_points = panel.point_id.nunique()
        for horizon in manifest['forecast_steps']:
            counts = expected_fold_counts(fold, manifest['input_len'], horizon, n_points)
            coverage.append({'fold_id': fold.fold_id, 'fold_role': fold.role,
                             'lead_hours': horizon * 3, 'points': n_points, **fold.as_dict(), **counts,
                             'first_validation_origin': fold.validation_start,
                             'first_validation_target': str(pd.Timestamp(fold.validation_start) + horizon * STEP),
                             'last_validation_target': str(pd.Timestamp(fold.test_start) - STEP),
                             'first_test_origin': fold.test_start,
                             'first_test_target': str(pd.Timestamp(fold.test_start) + horizon * STEP),
                             'last_test_target': str(pd.Timestamp(fold.test_end_exclusive) - STEP)})
    source_stats = {}
    for filename in ('metadata.json', 'centers.csv', 'grid_mapping.csv', 'center.csv',
                     'S01.csv', 'S02.csv', 'A01.csv', 'neighborhoods.npz'):
        stat = (spatial_dir / filename).stat()
        source_stats[filename] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    report = {'status': 'passed', 'scope': 'data integrity and metadata only; no model evaluation',
              'year': year, 'point_count': len(centers),
              'unique_center_grids': len(centers.drop_duplicates(['center_y', 'center_x'])),
              'time_steps_per_point': len(expected), 'rows_per_feature_set': len(center),
              'fold_manifest': str(manifest_path), 'fold_manifest_sha256': digest,
              'reference_check': reference_check, 'feature_sets': files,
              'source_stats': source_stats, 'report_dir': str(report_dir),
              'confirmation_policy': 'F04 data coverage checked; no predictive metrics computed'}
    report_dir.mkdir(parents=True, exist_ok=False)
    (report_dir / 'readiness.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame(coverage).to_csv(report_dir / 'fold_coverage.csv', index=False)
    (report_dir / 'fold_manifest.json').write_bytes(manifest_path.read_bytes())
    temporary = spatial_dir / 'READY.json.tmp'
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(spatial_dir / 'READY.json')
    print(f'[Readiness] PASS: {len(centers)} points, {len(expected)} timestamps, '
          f'{len(center)} rows per feature set; report={report_dir}', flush=True)
    return report, pd.DataFrame(coverage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spatial_dir', type=project_path, default=ROOT / 'data/CERRA/spatial_2024_full')
    parser.add_argument('--fold_manifest', type=project_path, default=ROOT / 'configs/seasonal_folds_2024.json')
    parser.add_argument('--report_dir', type=project_path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run through sbatch run_prepare_year.sh, not on the login node.')
    validate_annual(args.spatial_dir, args.fold_manifest, args.report_dir)


if __name__ == '__main__':
    main()
