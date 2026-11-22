"""Extract matched CERRA centers, regular neighborhoods and fixed distant controls.

Run through run_extract_cerra.sh. No full weather cube is loaded into memory.
Grid dy/dx describe array indices, not geographic north/east directions.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parent


def project_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Path must stay inside {ROOT}: {path}')
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grib', type=project_path,
                        default=ROOT / 'data/CERRA/dacdd4252851172d1e8fbd4b4def733a.grib')
    parser.add_argument('--reference_csv', type=project_path,
                        default=ROOT / 'data/CERRA/cerra_100_points_200km_summer_roi_2024.csv')
    parser.add_argument('--reference_mode', choices=['exact', 'reference_subset'], default='exact',
                        help='reference_subset requires every reference row within the larger extraction.')
    parser.add_argument('--reference_centers', type=project_path,
                        help='Verify the regenerated point/grid mapping against an existing centers.csv.')
    parser.add_argument('--require_complete_months', action='store_true',
                        help='Require every 3-hour timestamp in all requested calendar months.')
    parser.add_argument('--output_dir', type=project_path,
                        default=ROOT / 'data/CERRA/spatial_v1')
    parser.add_argument('--year', type=int, default=2024)
    parser.add_argument('--months', type=int, nargs='+', default=[6, 7, 8])
    parser.add_argument('--num_points', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lat_bounds', type=float, nargs=2, default=[49.5, 51.5])
    parser.add_argument('--lon_bounds', type=float, nargs=2, default=[8.5, 11.5])
    parser.add_argument('--radius', type=int, default=2, help='2 gives a 5x5 patch.')
    parser.add_argument('--far_offset', type=int, default=10,
                        help='Grid-index offset for 8 fixed distant controls; must exceed radius.')
    parser.add_argument('--chunk_steps', type=int, default=1,
                        help='Number of GRIB time fields decoded together.')
    args = parser.parse_args(argv)
    if args.num_points <= 0 or args.radius < 1 or args.chunk_steps <= 0:
        parser.error('num_points/chunk_steps must be positive; radius must be >= 1.')
    if args.far_offset <= args.radius:
        parser.error('far_offset must exceed radius.')
    if len(set(args.months)) != len(args.months) or any(month < 1 or month > 12 for month in args.months):
        parser.error('months must be in 1..12.')
    if any(bounds[0] >= bounds[1] for bounds in (args.lat_bounds, args.lon_bounds)):
        parser.error('Coordinate bounds must be increasing.')
    return args


def offset_name(dy, dx):
    def token(value):
        return f'm{abs(value)}' if value < 0 else f'p{value}'
    return f'ws_10_dy_{token(dy)}_dx_{token(dx)}'


def build_mapping(lat, lon, args):
    """Exactly reproduce getdata.py's random draw order and nearest-grid rule."""
    if lat.ndim != 2 or lat.shape != lon.shape:
        raise ValueError('Expected matching 2D latitude/longitude arrays.')
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError('Grid coordinates contain missing/non-finite values.')
    rng = np.random.RandomState(args.seed)
    target_lat = rng.uniform(*args.lat_bounds, args.num_points)
    target_lon = rng.uniform(*args.lon_bounds, args.num_points)
    local = [(dy, dx) for dy in range(-args.radius, args.radius + 1)
             for dx in range(-args.radius, args.radius + 1)]
    far = [(dy, dx) for dy in (-args.far_offset, 0, args.far_offset)
           for dx in (-args.far_offset, 0, args.far_offset) if (dy, dx) != (0, 0)]
    centers, mapping = [], []
    for point_id, (tlat, tlon) in enumerate(zip(target_lat, target_lon)):
        y, x = np.unravel_index(np.argmin((lat - tlat)**2 + (lon - tlon)**2), lat.shape)
        centers.append(dict(point_id=point_id, target_lat=tlat, target_lon=tlon,
                            center_y=int(y), center_x=int(x),
                            grid_lat=float(lat[y, x]), grid_lon=float(lon[y, x])))
        for kind, offsets in [('local', local), ('far', far)]:
            for dy, dx in offsets:
                yy, xx = int(y + dy), int(x + dx)
                if not (0 <= yy < lat.shape[0] and 0 <= xx < lat.shape[1]):
                    raise ValueError(f'point_id={point_id}: {kind} offset {(dy, dx)} outside grid.')
                feature = 'ws_10' if (dy, dx) == (0, 0) else offset_name(dy, dx)
                mapping.append(dict(point_id=point_id, kind=kind, dy=dy, dx=dx,
                                    grid_y=yy, grid_x=xx, grid_lat=float(lat[yy, xx]),
                                    grid_lon=float(lon[yy, xx]), feature=feature))
    return pd.DataFrame(centers), pd.DataFrame(mapping), local, far


def prepare_time_axis(ds, args):
    da = ds['si10']
    units = str(da.attrs.get('units', '')).strip()
    if units not in {'m s**-1', 'm s-1', 'm/s', 'm s^-1'}:
        raise ValueError(f'Unrecognized si10 units: {units!r}; expected m/s.')
    if not {'y', 'x', 'time'}.issubset(da.dims):
        raise ValueError(f'Expected time/y/x dimensions, got {da.dims}.')
    temporal_dims = [dim for dim in da.dims if dim not in ('y', 'x')]
    if set(temporal_dims).difference({'time', 'step'}):
        raise ValueError(f'Unsupported extra dimensions: {da.dims}')
    template = da.isel(y=0, x=0, drop=True)
    if 'valid_time' in ds:
        valid = ds['valid_time']
    elif 'step' in ds:
        valid = ds['time'] + ds['step']
    else:
        valid = ds['time']
    valid = valid.broadcast_like(template).transpose(*temporal_dims)
    timestamps = pd.DatetimeIndex(valid.values.reshape(-1))
    keep = np.flatnonzero((timestamps.year == args.year) & timestamps.month.isin(args.months))
    if not len(keep):
        raise ValueError('No valid times match requested year/months.')
    keep = keep[np.argsort(timestamps[keep].values)]
    dates = timestamps[keep]
    if dates.hasnans or dates.has_duplicates:
        raise ValueError('Missing/duplicate valid times; choose one forecast cycle/lead explicitly.')
    if len(dates) < 2 or not (np.diff(dates.values) == np.timedelta64(3, 'h')).all():
        raise ValueError('Selected valid times must form a continuous 3-hour sequence.')
    if args.require_complete_months:
        expected = pd.DatetimeIndex([])
        for month in sorted(args.months):
            first = pd.Timestamp(year=args.year, month=month, day=1)
            expected = expected.append(pd.date_range(
                first, first + pd.offsets.MonthBegin(1), freq='3h', inclusive='left'))
        if not dates.equals(expected):
            raise ValueError('Selected valid times do not cover every requested calendar month completely.')
    # Do not stack/reshape the weather backend: without Dask that can eagerly
    # materialize the full GRIB cube. Only unravel the small time coordinates.
    indices = np.unravel_index(keep, tuple(da.sizes[dim] for dim in temporal_dims))
    selectors = [dict(zip(temporal_dims, map(int, values))) for values in zip(*indices)]
    return da, dates, units, selectors


def check_reference(center_frame, reference_path, mode='exact'):
    reference = pd.read_csv(reference_path, usecols=['point_id', 'datetime', 'ws_10'])
    reference['datetime'] = pd.to_datetime(reference['datetime'])
    keys = ['point_id', 'datetime']
    if reference.duplicated(keys).any() or center_frame.duplicated(keys).any():
        raise ValueError('Reference CSV contains duplicate point/time pairs.')
    if mode not in {'exact', 'reference_subset'}:
        raise ValueError(f'Unknown reference comparison mode: {mode}')
    if set(reference.point_id) != set(center_frame.point_id):
        raise ValueError('Reference and extraction have different point IDs.')
    # In annual mode every old summer row must match, without requiring that
    # reference data also cover all newly extracted months.
    matched = center_frame.merge(reference, on=keys,
                                 how='outer' if mode == 'exact' else 'right',
                                 suffixes=('_new', '_old'), indicator=True,
                                 validate='one_to_one')
    if not (matched['_merge'] == 'both').all():
        raise ValueError('Reference and extracted center point/time keys differ; '
                         'check point generation parameters and valid times.')
    a = matched['ws_10_new'].to_numpy()
    b = matched['ws_10_old'].to_numpy()
    if not np.isfinite(b).all() or not np.allclose(a, b, rtol=1e-6, atol=1e-5):
        raise ValueError(f'Center wind differs from reference; max absolute difference: '
                         f'{np.max(np.abs(a - b))}')
    return {'matched_rows': len(matched), 'reference_rows': len(reference),
            'extracted_rows': len(center_frame), 'mode': mode,
            'max_abs_difference': float(np.max(np.abs(a - b)))}


def check_center_mapping(centers, reference_path):
    reference = pd.read_csv(reference_path).sort_values('point_id').reset_index(drop=True)
    actual = centers.sort_values('point_id').reset_index(drop=True)
    columns = ['point_id', 'target_lat', 'target_lon', 'center_y', 'center_x', 'grid_lat', 'grid_lon']
    if reference.point_id.duplicated().any() or actual.shape[0] != reference.shape[0]:
        raise ValueError('Center mapping point counts/IDs do not match.')
    for column in columns:
        if not np.allclose(actual[column], reference[column], rtol=0, atol=1e-8):
            raise ValueError(f'Center mapping changed: {column}')
    return {'matched_points': len(actual), 'reference': str(reference_path)}


def extract(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit run_extract_cerra.sh through sbatch; do not extract on login nodes.')
    if args.output_dir.exists():
        raise FileExistsError(f'Output already exists; choose a new --output_dir: {args.output_dir}')
    if not args.grib.is_file() or not args.reference_csv.is_file():
        raise FileNotFoundError('GRIB/reference CSV does not exist.')
    # Disable cfgrib index side effects; the source remains untouched.
    with xr.open_dataset(args.grib, engine='cfgrib', backend_kwargs={'indexpath': ''}) as ds:
        lat = ds['latitude'].transpose('y', 'x').values
        lon = ds['longitude'].transpose('y', 'x').values
        centers, mapping, local, far = build_mapping(lat, lon, args)
        mapping_check = (check_center_mapping(centers, args.reference_centers)
                         if args.reference_centers is not None else None)
        da, dates, units, selectors = prepare_time_axis(ds, args)
        cells = mapping[['grid_y', 'grid_x']].drop_duplicates()
        lookup = {tuple(pair): i for i, pair in enumerate(cells.to_numpy())}
        positions = np.array([lookup[(row.grid_y, row.grid_x)]
                              for row in mapping.itertuples()])
        ys = xr.DataArray(cells['grid_y'].to_numpy(), dims='cell')
        xs = xr.DataArray(cells['grid_x'].to_numpy(), dims='cell')
        sampled = np.empty((len(dates), len(cells)), dtype=np.float32)
        print(f'[Input] {len(dates)} times, {len(cells)} unique grid cells, units={units}', flush=True)
        # Read all requested cells together per time field, avoiding repeated
        # decoding of the same full GRIB field once per point/neighborhood.
        for start in range(0, len(dates), args.chunk_steps):
            stop = min(start + args.chunk_steps, len(dates))
            values = np.stack([
                da.isel(**selectors[index]).isel(y=ys, x=xs).transpose('cell').values
                for index in range(start, stop)])
            if values.shape != (stop - start, len(cells)) or not np.isfinite(values).all():
                raise ValueError(f'Missing values or unexpected shape at time chunk {start}.')
            if (values < 0).any():
                raise ValueError(f'Negative wind speeds at time chunk {start}.')
            sampled[start:stop] = values
            if start % 24 == 0 or stop == len(dates):
                print(f'[Extract] {stop}/{len(dates)} valid times', flush=True)
    all_values = sampled[:, positions].reshape(len(dates), args.num_points, len(local) + len(far))
    local_values = all_values[:, :, :len(local)].transpose(1, 0, 2)
    center_index = local.index((0, 0))
    center_frames = [pd.DataFrame({'datetime': dates, 'ws_10': local_values[p, :, center_index],
                                  'point_id': p}) for p in range(args.num_points)]
    center_frame = pd.concat(center_frames, ignore_index=True)
    reference_check = check_reference(center_frame, args.reference_csv, args.reference_mode)
    duplicate_centers = centers.duplicated(['center_y', 'center_x']).sum()
    print(f'[Check] {reference_check}; duplicate center mappings={duplicate_centers}', flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    centers.to_csv(args.output_dir / 'centers.csv', index=False)
    mapping.to_csv(args.output_dir / 'grid_mapping.csv', index=False)
    center_frame.to_csv(args.output_dir / 'center.csv', index=False)
    feature_sets = {'T01': ['ws_10', 'hour_sin', 'hour_cos']}
    for name, offsets in [('S01', [(dy, dx) for dy, dx in local if abs(dy) <= 1 and abs(dx) <= 1]),
                          ('S02', local), ('A01', [(0, 0)] + far)]:
        columns = ['ws_10' if offset == (0, 0) else offset_name(*offset) for offset in offsets]
        source_indices = [(local + far).index(offset) for offset in offsets]
        frames = []
        for point_id in range(args.num_points):
            frame = pd.DataFrame(all_values[:, point_id, :][:, source_indices], columns=columns)
            frame.insert(0, 'datetime', dates)
            frame.insert(1, 'point_id', point_id)
            frames.append(frame)
        pd.concat(frames, ignore_index=True).to_csv(args.output_dir / f'{name}.csv', index=False)
        feature_sets[name] = columns + ['hour_sin', 'hour_cos']
    side = 2 * args.radius + 1
    np.savez_compressed(args.output_dir / 'neighborhoods.npz',
                        ws_10=local_values.reshape(args.num_points, len(dates), side, side),
                        datetime=dates.values, point_id=centers['point_id'].to_numpy(),
                        dy=np.arange(-args.radius, args.radius + 1),
                        dx=np.arange(-args.radius, args.radius + 1))
    metadata = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    metadata.update(status='complete', units=units, reference_check=reference_check,
                    center_mapping_check=mapping_check,
                    first_valid_time=str(dates.min()), last_valid_time=str(dates.max()),
                    time_steps=len(dates),
                    monthly_time_counts={str(month): int((dates.month == month).sum())
                                         for month in sorted(args.months)},
                    duplicate_centers=int(duplicate_centers), feature_sets=feature_sets,
                    tensor_axes=['point_id', 'datetime', 'dy', 'dx'],
                    shape=[args.num_points, len(dates), side, side],
                    time_basis='GRIB valid_time; duplicate valid times rejected',
                    far_selection='8 fixed offsets from {-far_offset,0,+far_offset} squared, excluding center',
                    offset_semantics='array index directions; not geographic compass directions')
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'[Complete] Outputs: {args.output_dir}', flush=True)


if __name__ == '__main__':
    extract(parse_args())
