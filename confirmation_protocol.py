"""Versioned independent-year contracts; no training or weather decoding on import."""
import importlib.metadata
import json
import re
from itertools import product

import numpy as np
import pandas as pd

from direction_pipeline import code_hashes, sha256
from temporal_folds import FoldSpec, expected_fold_counts, project_path

DEFAULT_REQUEST = 'configs/confirmation_v1/request.json'
GROUPS = ['RS_S0', 'RS_S1', 'S_COORD', 'S_LC', 'S_DSM']
GRID_KEYS = ['gridType', 'Nx', 'Ny', 'DxInMetres', 'DyInMetres', 'LoVInDegrees',
             'LaDInDegrees', 'Latin1InDegrees', 'Latin2InDegrees',
             'latitudeOfFirstGridPointInDegrees', 'longitudeOfFirstGridPointInDegrees',
             'shapeOfTheEarth', 'scanningMode', 'radius']
SCIENCE_CODE = ['confirmation_protocol.py', 'prepare_confirmation_year.py',
                'train_confirmation_year.py', 'analyze_confirmation_year.py',
                'train.py', 'train_p3.py', 'p3b_contract.py', 'rs_static_contract.py',
                'temporal_folds.py', 'direction_pipeline.py', 'analyze_experiments.py',
                'analyze_p1_dates.py', 'train_p1_baseline.py', 'environment/activate_wind_env.sh',
                'slurm/run_confirmation_extract.sh', 'slurm/run_confirmation_year.sh',
                'slurm/run_confirmation_analysis.sh']


def read_json(path):
    return json.loads(project_path(path).read_text())


def write_new(path, value):
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def source_stats(paths):
    return {str(project_path(p)): dict(bytes=project_path(p).stat().st_size,
                                     mtime_ns=project_path(p).stat().st_mtime_ns) for p in paths}


def versions():
    return {name: importlib.metadata.version(name) for name in
            ['tensorflow', 'keras', 'numpy', 'pandas', 'scikit-learn', 'eccodes']}


def make_folds(year):
    if year not in (2023, 2025):
        raise ValueError('v1 permits 2025, or availability-based fallback 2023; never 2024.')
    folds = []
    for index, month in enumerate((4, 7, 10), 1):
        test = pd.Timestamp(year=year, month=month, day=1)
        validation = test - pd.Timedelta(days=28)
        fold = FoldSpec(f'C{index:02d}', str(validation-pd.Timedelta(days=60)),
                        str(validation), str(test), str(test+pd.Timedelta(days=28)),
                        role='locked_confirmation')
        folds.append(fold.as_dict())
    return folds


def task_list(request):
    return list(product([42, 43, 44], [f['fold_id'] for f in request['folds']], GROUPS))


def run_id(index, request):
    seed, fold, group = task_list(request)[index]
    return f'{index:02d}_{fold}_{group}_seed{seed}'


def validate_request(request):
    if not re.fullmatch(r'confirmation_[A-Za-z0-9_-]+', request['version']):
        raise ValueError('Unsafe version name.')
    if request['folds'] != make_folds(request['year']):
        raise ValueError('v1 uses three calendar windows with 60/28/28 days.')
    if request['groups'] != GROUPS or request['seeds'] != [42, 43, 44]:
        raise ValueError('v1 requires all five groups and three seeds.')
    settings = dict(primary=['RS_S1', 'RS_S0', 'RMSE'], secondary=['RS_S1', 'S_COORD'],
                    date_assignment='forecast origin UTC date', block_days=3, resamples=2000,
                    bootstrap_seed=20261119, mode='refit_new_year_only', training_fits=45,
                    window_aggregate='equal window mean of within-seed gains')
    if any(request.get(k) != v for k, v in settings.items()):
        raise ValueError('v1 fixes the primary estimand, date blocks and 45-fit refit protocol.')
    template_path = project_path(request['training_template'])
    if sha256(template_path) != request['training_template_sha256']:
        raise ValueError('Development training template changed.')
    template = read_json(template_path)
    expected = dict(lookback=56, pred_len=1, lr=.001, batch_size=64, epochs=100,
                    patience=10, direction_enabled=False,
                    architecture='LSTM64_Dropout0.4_StaticDense8_Concat_Dense16_Dense1')
    if any(template[k] != v for k, v in expected.items()):
        raise ValueError('Confirmation must retain the development training settings.')
    for name, digest in request['bound_metadata'].items():
        if sha256(project_path(name)) != digest:
            raise ValueError(f'Bound metadata changed: {name}')
    for name in ('data_dir', 'run_dir', 'result_dir', 'fold_manifest'):
        project_path(request[name])
    if len(task_list(request)) != 45:
        raise ValueError('Expected 45 fits.')
    return template


def validate_product(header, reference):
    if any(header.get(k) != reference[k] for k in GRID_KEYS):
        raise ValueError('Native grid/projection/scanning changed.')
    for key, value in dict(paramId=207, dataType='an', typeOfLevel='heightAboveGround',
                           level=10, stepType='instant', stepRange='0', units=reference['units']).items():
        if str(header.get(key)) != str(value):
            raise ValueError(f'Wrong wind product: {key}')
    if int(header['scanningMode']) & 48:
        raise ValueError('Column-major/alternating-row scanning is unsupported.')


def check_samples(samples, fold, lookback=56, horizon=1, n_points=100):
    expected = list(expected_fold_counts(fold, lookback, horizon, n_points).values())
    if [len(v[1]) for v in samples.values()] != expected:
        raise ValueError('Sample counts differ from fixed-date protocol.')
    for name, (array, labels, persistence, metadata) in samples.items():
        if array.shape != (len(labels), lookback, 11) or not all(
                np.isfinite(v).all() for v in [array, labels, persistence]):
            raise ValueError('Invalid dynamic shape/values.')
        origin, target = pd.to_datetime(metadata.forecast_origin), pd.to_datetime(metadata.target_time)
        start = pd.to_datetime(metadata.input_start)
        if not ((target-origin == pd.Timedelta(hours=3*horizon)).all()
                and (origin-start == pd.Timedelta(hours=3*(lookback-1))).all()):
            raise ValueError('Input/target causality violated.')
        lower, upper = {
            'train': (fold.train_start, fold.validation_start),
            'validation': (fold.validation_start, fold.test_start),
            'test': (fold.test_start, fold.test_end_exclusive)}[name]
        if not ((origin >= pd.Timestamp(lower)).all() and (target < pd.Timestamp(upper)).all()):
            raise ValueError('Partition target boundary violated.')
    return expected


def load_frozen(request_path, full=False):
    request_path = project_path(request_path)
    request = read_json(request_path)
    template = validate_request(request)
    frozen_path = request_path.parent/'FROZEN.json'
    frozen = read_json(frozen_path)
    if frozen.get('status') != 'frozen' or frozen['request_sha256'] != sha256(request_path):
        raise ValueError('Missing frozen protocol or request changed.')
    if frozen['fold_manifest_sha256'] != sha256(project_path(request['fold_manifest'])):
        raise ValueError('Frozen calendar manifest changed.')
    if frozen['code_sha256'] != code_hashes(SCIENCE_CODE) or frozen['versions'] != versions():
        raise ValueError('Frozen code or environment changed; do not silently resume another version.')
    data = project_path(request['data_dir'])
    if sha256(data/'completion.json') != frozen['extraction_completion_sha256']:
        raise ValueError('Extraction marker changed.')
    marker = read_json(data/'completion.json')
    for name, stat in frozen['extracted_stats'].items():
        if source_stats([data/name])[str(data/name)] != stat:
            raise ValueError(f'Extracted artifact changed: {name}')
    if source_stats(request['grib_files']) != frozen['raw_source_stats']:
        raise ValueError('Raw source changed after extraction/freeze.')
    if full:
        from direction_pipeline import verify_artifacts
        verify_artifacts(data)
    return request, template, frozen, marker


def static_array(metadata, mapping, table, static_scaler, coordinate_scaler, group):
    from p3b_contract import representation
    from rs_static_contract import static_inputs
    if group in ('RS_S0', 'RS_S1'):
        return static_inputs(metadata, mapping, table, static_scaler, group == 'RS_S1')
    if group not in GROUPS:
        raise ValueError('Unknown static condition.')
    return representation(metadata, mapping, table, static_scaler, coordinate_scaler, group)


def block_weights(rng, n_days, resamples=2000, block_days=3):
    starts = rng.integers(n_days, size=(resamples, int(np.ceil(n_days/block_days))))
    indices = ((starts[:, :, None]+np.arange(block_days)) % n_days).reshape(resamples, -1)[:, :n_days]
    return np.stack([(indices == day).sum(axis=1) for day in range(n_days)], axis=1)


def paired_gain(loss, reference, weights, metric):
    loss, reference = np.asarray(loss), np.asarray(reference)
    if not np.isfinite(loss).all() or not np.isfinite(reference).all() or (reference < 0).any() or (
            reference.sum() <= 0 or (weights @ reference <= 0).any()):
        raise ValueError('Invalid/zero reference loss; no silent division or dropped dates.')
    ratio = (weights @ loss)/(weights @ reference)
    point = loss.sum()/reference.sum()
    if metric == 'RMSE':
        point, ratio = np.sqrt(point), np.sqrt(ratio)
    return 100*(1-point), 100*(1-ratio)
