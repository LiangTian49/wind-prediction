"""Extract aligned 10-m V/u/v from the existing GRIB through Slurm."""

import argparse
import json
import os
import pickle
from time import perf_counter

import eccodes
import numpy as np
import pandas as pd
import xarray as xr

from direction_pipeline import DATA_DIR, code_hashes, output_transaction, sha256, wind_components
from extract_cerra_spatial import prepare_time_axis
from temporal_folds import project_path, verify_readiness

SPATIAL = 'data/CERRA/spatial_2024_full'


def normalize_direction(values, packing_error):
    """Wrap only endpoint overshoots justified by this GRIB message's packing error."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(packing_error) or packing_error < 0 or not np.isfinite(values).all():
        raise ValueError('Missing/non-finite direction or invalid GRIB packing error.')
    tolerance = float(packing_error) + 8*np.spacing(360.)
    if (values < -tolerance).any() or (values > 360+tolerance).any():
        raise ValueError(f'Direction outside encoded tolerance: min={values.min()}, '
                         f'max={values.max()}, packing_error={packing_error}.')
    changed = (values < 0) | (values > 360)
    normalized = values.copy()
    normalized[changed] = np.mod(normalized[changed], 360.)
    return normalized, int(changed.sum())


def audit_product_index(grib, index):
    if index.stat().st_mtime_ns < grib.stat().st_mtime_ns:
        raise ValueError('GRIB index is older than source.')
    with index.open('rb') as stream:
        cache = pickle.load(stream)
    products = {}
    direction_offsets = {}
    for parameter in [207, 260260]:
        keys = []
        for raw, offsets in cache.field_ids_index:
            row = dict(zip(cache.index_keys, raw))
            if row['paramId'] != parameter:
                continue
            if (len(offsets) != 1 or row['dataType'] != 'an' or row['stepType'] != 'instant'
                    or row['step'] != 0 or row['endStep'] != 0
                    or row['typeOfLevel'] != 'heightAboveGround' or row['level:float'] != 10):
                raise ValueError('Ambiguous or incompatible wind product.')
            keys.append((row['time'], row['md5GridSection']))
            if parameter == 260260:
                direction_offsets[row['time']] = offsets[0]
        if len(keys) != 2928 or len(set(keys)) != 2928:
            raise ValueError('Expected 2928 unique records for each wind variable.')
        products[parameter] = set(keys)
    if products[207] != products[260260]:
        raise ValueError('Speed and direction do not share times and grid.')
    packing_errors = {}
    # Header-only reads avoid decoding a second copy of the direction fields.
    with grib.open('rb') as stream:
        for time, offset in direction_offsets.items():
            stream.seek(offset)
            message = eccodes.codes_grib_new_from_file(stream, headers_only=True)
            try:
                packing_errors[time] = float(eccodes.codes_get(message, 'packingError'))
            finally:
                eccodes.codes_release(message)
    audit = dict(records_per_variable=2928, parameters=[207, 260260], data_type='an',
                 height_m=10, step_hours=0, grid_hash=next(iter(products[207]))[1])
    return audit, packing_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=project_path, default=project_path(DATA_DIR))
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit run_extract_direction.sh.')
    started = perf_counter()
    spatial = project_path(SPATIAL)
    verify_readiness(spatial, 'configs/seasonal_folds_2024.json')
    provenance = json.loads(project_path('result/rs_hpc_v3/P0/grib_provenance.json').read_text())
    grib, index = project_path(provenance['source']), project_path(provenance['index'])
    legacy = {str((spatial / name).relative_to(project_path('.'))): sha256(spatial / name)
              for name in ['center.csv', 'A01.csv', 'centers.csv', 'grid_mapping.csv']}
    fingerprint = dict(source_size=grib.stat().st_size, source_mtime_ns=grib.stat().st_mtime_ns,
                       legacy_sources=legacy,
                       code=code_hashes(['extract_cerra_direction.py', 'direction_pipeline.py',
                                         'extract_cerra_spatial.py']))
    with output_transaction(args.output_dir, fingerprint) as work:
        if work is None:
            print(f'[Direction] Verified existing extraction: {args.output_dir}')
            return
        audit, packing_errors = audit_product_index(grib, index)
        mapping = pd.read_csv(spatial / 'grid_mapping.csv')
        selected = mapping[(mapping.kind == 'far') | ((mapping.dy == 0) & (mapping.dx == 0))].copy()
        centers = pd.read_csv(spatial / 'centers.csv')
        if (centers.point_id.duplicated().any() or len(centers) != 100
                or len(centers.drop_duplicates(['center_y', 'center_x'])) != 98
                or set(selected.point_id) != set(centers.point_id)):
            raise ValueError('Original 100-point / 98-grid mapping is required.')
        if selected.duplicated(['point_id', 'feature']).any():
            raise ValueError('Repeated feature mapping.')
        for row in centers.itertuples():
            center_row = selected[(selected.point_id == row.point_id)
                                  & (selected.dy == 0) & (selected.dx == 0)]
            if (len(center_row) != 1 or center_row.grid_y.iloc[0] != row.center_y
                    or center_row.grid_x.iloc[0] != row.center_x):
                raise ValueError('Center and neighborhood mappings differ.')
        if selected.groupby('point_id').size().ne(9).any() or len(selected) != 900:
            raise ValueError('Expected nine cells per original point.')
        cells = selected[['grid_y', 'grid_x']].drop_duplicates()
        lookup = {tuple(cell): i for i, cell in enumerate(cells.to_numpy())}
        with xr.open_dataset(grib, engine='cfgrib', backend_kwargs={'indexpath': ''}) as ds:
            speed_name = next(name for name in ds.data_vars if ds[name].attrs.get('GRIB_paramId') == 207)
            angle_name = next(name for name in ds.data_vars if ds[name].attrs.get('GRIB_paramId') == 260260)
            speed_ds = ds.rename({speed_name: 'si10'}) if speed_name != 'si10' else ds
            time_args = argparse.Namespace(year=2024, months=list(range(1, 13)), require_complete_months=True)
            speed, dates, units, selectors = prepare_time_axis(speed_ds, time_args)
            angle = ds[angle_name]
            if angle.attrs.get('units') != 'Degree true' or angle.dims != speed.dims:
                raise ValueError('Expected true-north direction on matching dimensions.')
            if any(not angle.coords[d].equals(speed.coords[d]) for d in speed.dims):
                raise ValueError('Variable coordinates differ.')
            lat, lon = ds.latitude.values, ds.longitude.values
            for row in selected.itertuples():
                if not (0 <= row.grid_y < lat.shape[0] and 0 <= row.grid_x < lat.shape[1]):
                    raise ValueError('Input grid outside source.')
                np.testing.assert_allclose([lat[row.grid_y, row.grid_x], lon[row.grid_y, row.grid_x]],
                                           [row.grid_lat, row.grid_lon], atol=1e-8, rtol=0)
            ys = xr.DataArray(cells.grid_y.to_numpy(), dims='cell')
            xs = xr.DataArray(cells.grid_x.to_numpy(), dims='cell')
            speeds = np.empty((len(dates), len(cells)), dtype=np.float64)
            angles = np.empty_like(speeds)
            raw_angles = np.empty_like(speeds)
            quality = []
            for i, selector in enumerate(selectors):
                speeds[i] = speed.isel(**selector).isel(y=ys, x=xs).values
                raw_angles[i] = angle.isel(**selector).isel(y=ys, x=xs).values
                packing_error = packing_errors[dates[i].value // 10**9]
                try:
                    angles[i], changed = normalize_direction(raw_angles[i], packing_error)
                    wind_components(speeds[i], angles[i])
                except ValueError as error:
                    diagnostic = dict(datetime=str(dates[i]), time_index=i,
                                      packing_error_deg=packing_error,
                                      speed_min=float(np.nanmin(speeds[i])),
                                      speed_max=float(np.nanmax(speeds[i])),
                                      raw_direction_min=float(np.nanmin(raw_angles[i])),
                                      raw_direction_max=float(np.nanmax(raw_angles[i])),
                                      reason=str(error))
                    (work / 'failure_diagnostic.json').write_text(json.dumps(diagnostic, indent=2))
                    raise ValueError(f'Direction extraction failed at {dates[i]}: {diagnostic}') from error
                quality.append(dict(datetime=str(dates[i]), packing_error_deg=packing_error,
                                    raw_min_deg=float(raw_angles[i].min()),
                                    raw_max_deg=float(raw_angles[i].max()),
                                    wrapped_unique_cells=changed))
                if i % 240 == 0:
                    print(f'[Direction] {i}/{len(dates)}', flush=True)
        u, v = wind_components(speeds, angles)
        np.testing.assert_allclose(np.hypot(u, v), speeds, atol=1e-8, rtol=1e-8)
        frames = []
        for point in centers.point_id:
            rows = selected[selected.point_id == point]
            frame = pd.DataFrame({'datetime': dates, 'point_id': point})
            for row in rows.itertuples():
                column = lookup[(row.grid_y, row.grid_x)]
                for prefix, values in [('ws_10', speeds), ('u_10', u), ('v_10', v),
                                       ('direction_10', angles), ('direction_raw_10', raw_angles)]:
                    name = row.feature.replace('ws_10', prefix, 1)
                    frame[name] = values[:, column]
            frames.append(frame)
        full = pd.concat(frames, ignore_index=True)
        archived = pd.read_csv(spatial / 'A01.csv', parse_dates=['datetime'])
        joined = full.merge(archived, on=['point_id', 'datetime'], validate='one_to_one', suffixes=('', '_old'))
        if len(joined) != 292800:
            raise ValueError('Incomplete legacy sample coverage.')
        max_difference = 0.
        for name in archived.columns.difference(['datetime', 'point_id']):
            difference = float(np.abs(joined[name] - joined[name + '_old']).max())
            max_difference = max(max_difference, difference)
            np.testing.assert_allclose(joined[name], joined[name + '_old'], atol=1e-6, rtol=0)
            # Preserve the exact archived V for bridge comparisons and Persistence.
            replacement = archived.set_index(['point_id', 'datetime'])[name]
            full[name] = replacement.reindex(pd.MultiIndex.from_frame(full[['point_id', 'datetime']])).to_numpy()
        # Recompute components against the archived V so hypot(u,v) remains consistent.
        for name in archived.columns.difference(['datetime', 'point_id']):
            angle_column = name.replace('ws_10', 'direction_10', 1)
            uu, vv = wind_components(full[name].to_numpy(), full[angle_column].to_numpy())
            full[name.replace('ws_10', 'u_10', 1)] = uu
            full[name.replace('ws_10', 'v_10', 1)] = vv
        full.to_csv(work / 'A01.csv', index=False)
        full[['datetime', 'point_id', 'ws_10', 'u_10', 'v_10', 'direction_10', 'direction_raw_10']].to_csv(
            work / 'T01.csv', index=False)
        pd.DataFrame(quality).to_csv(work / 'direction_quality.csv', index=False)
        centers.to_csv(work / 'centers.csv', index=False)
        selected.to_csv(work / 'grid_mapping.csv', index=False)
        metadata = dict(status='complete', experiment_version='rs_hpc_v3', time_steps=len(dates),
                        point_ids=100, unique_centers=98, rows=len(full), units=units,
                        legacy_sources=legacy, product_audit=audit, source_sha256=sha256(grib),
                        speed_max_abs_difference=max_difference, angle_units='degrees true north, meteorological from',
                        normalization='Only endpoint overshoots within per-message GRIB packingError are wrapped modulo 360.',
                        wrapped_unique_cell_times=sum(row['wrapped_unique_cells'] for row in quality),
                        raw_direction_min_deg=min(row['raw_min_deg'] for row in quality),
                        raw_direction_max_deg=max(row['raw_max_deg'] for row in quality),
                        formula='u=-V*sin(theta); v=-V*cos(theta)', speed_policy='archived V after raw-source tolerance check',
                        first_time=str(dates.min()), last_time=str(dates.max()),
                        elapsed_seconds=perf_counter()-started,
                        source_size=grib.stat().st_size, source_mtime_ns=grib.stat().st_mtime_ns)
        (work / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'[Direction] Completed: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
