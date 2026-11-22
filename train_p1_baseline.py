"""Versioned direct-regression baselines using the frozen LSTM sample protocol."""

import argparse
import fcntl
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import pickle
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from analyze_experiments import attach_centers, collapse_grids, score
from temporal_folds import load_manifest, project_path, verify_readiness
from train import create_dataset, load_and_preprocess_data, prediction_frame

CONFIG = 'configs/rs_hpc_v3_p1.json'
OUTPUT = 'runs/rs_hpc_v3/P1'
KEYS = ['point_id', 'forecast_origin', 'target_time']


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def tasks(config):
    return list(product(config['folds'], config['forecast_steps'], config['models'], config['groups']))


def make_model(family, parameters, seed):
    if family == 'ridge':
        return Ridge(**parameters)
    if family == 'hist_gradient_boosting':
        return HistGradientBoostingRegressor(**parameters, early_stopping=False, random_state=seed)
    raise ValueError(f'Unknown family: {family}')


def select_model(family, candidates, seed, train_x, train_y, val_x, val_y):
    best = None
    trials = []
    for parameters in candidates:
        model = make_model(family, parameters, seed)
        started = perf_counter()
        model.fit(train_x, train_y)
        seconds = perf_counter() - started
        rmse = float(np.sqrt(np.mean((model.predict(val_x) - val_y) ** 2)))
        trials.append(dict(parameters=parameters, validation_RMSE=rmse, fit_seconds=seconds))
        if best is None or rmse < best[0]:
            best = (rmse, model, parameters)
    return best, trials


def verify_reference(frame, reference):
    if len(frame) != len(reference) or frame.duplicated(KEYS).any() or reference.duplicated(KEYS).any():
        raise ValueError('Invalid or unequal prediction coverage.')
    paired = frame.merge(reference, on=KEYS, validate='one_to_one', suffixes=('', '_legacy'))
    if len(paired) != len(frame):
        raise ValueError('Legacy forecast keys differ.')
    for name in ['y_true', 'persistence_pred']:
        np.testing.assert_allclose(paired[name], paired[f'{name}_legacy'], atol=1e-6, rtol=0)


def execute(index, config, output):
    started = perf_counter()
    fold_id, steps, family, group = tasks(config)[index]
    _, folds, fold_hash = load_manifest(config['fold_manifest'])
    fold = folds[fold_id]
    if fold.role != 'development':
        raise ValueError('P1 selects models on development folds only.')
    spatial = project_path(config['spatial_dir'])
    verify_readiness(spatial, config['fold_manifest'])
    spec = config['groups'][group]
    csv_path = spatial / spec['file']
    legacy = project_path(config['legacy_analysis'])
    sources = json.loads((legacy / 'source_runs.json').read_text())
    refs = [r for r in sources if r['fold_id'] == fold_id and r['pred_len'] == steps
            and r['experiment'] in ['T01', 'A01']]
    if {(r['experiment'], r['seed']) for r in refs} != set(product(['T01', 'A01'], range(42, 47))) or len(refs) != 10:
        raise ValueError('Incomplete reference runs.')
    if any(r['fold_manifest_sha256'] != fold_hash or r['input_len'] != config['lookback']
           or json.loads(r['features']) != config['groups'][r['experiment']]['features']
           for r in refs):
        raise ValueError('Reference fold, history length or features differ from P1.')
    code_hashes = {name: digest(project_path(name)) for name in
                   ['train_p1_baseline.py', 'train.py', 'temporal_folds.py', 'analyze_experiments.py']}
    fingerprints = dict(config_sha256=digest(project_path(CONFIG)), fold_sha256=fold_hash,
                        data_sha256=digest(csv_path), centers_sha256=digest(spatial / 'centers.csv'),
                        code_sha256=code_hashes, sklearn_version=sklearn.__version__,
                        reference_manifest_sha256=digest(legacy / 'source_runs.json'),
                        reference_prediction_sha256={r['run_id']: digest(project_path(r['run_dir']) / 'predictions.csv') for r in refs})
    run_id = f'{index:02d}_{fold_id}_{steps * 3}h_{family}_{group}'
    output.mkdir(parents=True, exist_ok=True)
    final = output / run_id
    with (output / f'{run_id}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if final.exists():
            saved = json.loads((final / 'result.json').read_text())
            if saved['fingerprints'] != fingerprints or any(
                digest(final / name) != sha for name, sha in saved['artifact_hashes'].items()
            ):
                raise ValueError(f'Existing result differs or is damaged: {final}. Preserve it in its version directory.')
            print(f'[P1] Verified completed configuration, skipping: {final}', flush=True)
            return
        # Partial attempts remain separate and can be inspected or retried.
        work = Path(tempfile.mkdtemp(prefix=f'.{run_id}_', dir=output))
        phase = perf_counter()
        df = load_and_preprocess_data(csv_path, spec['features'])
        data = create_dataset(df, config['lookback'], steps, spec['features'],
                              include_metadata=True, protocol=config['protocol'], fold=fold)
        xt, yt, xv, yv, xe, ye, persistence, scaler, split, meta = data
        xt, xv, xe = (x.reshape(len(x), -1) for x in (xt, xv, xe))
        preparation_seconds = perf_counter() - phase
        phase = perf_counter()
        (validation_rmse, model, parameters), trials = select_model(
            family, config['models'][family], config['seed'], xt, yt, xv, yv)
        selection_seconds = perf_counter() - phase
        phase = perf_counter()
        predicted = prediction_frame(meta, ye, model.predict(xe), persistence,
                                     run_id, config['seed'], group, fold_id)
        centers = pd.read_csv(spatial / 'centers.csv')
        predicted = attach_centers(predicted, centers)
        predicted['grid_id'] = predicted.center_y.astype(str) + ':' + predicted.center_x.astype(str)
        predicted['experiment_version'] = config['experiment_version']
        predicted['model_family'] = family
        grids = collapse_grids(predicted)
        metrics = score(grids.y_true, grids.y_pred, grids.persistence_pred)
        comparisons = []
        for ref in refs:
            frame = pd.read_csv(project_path(ref['run_dir']) / 'predictions.csv',
                                parse_dates=['forecast_origin', 'target_time', 'input_start'])
            verify_reference(predicted, frame)
            reference = collapse_grids(attach_centers(frame, centers))
            ref_scores = score(reference.y_true, reference.y_pred, reference.persistence_pred)
            comparisons.append(dict(reference=ref['experiment'], seed=ref['seed'], run_id=ref['run_id'],
                                    RMSE=ref_scores['RMSE'], MAE=ref_scores['MAE'],
                                    **{f'{m}_gain_pct': 100 * (1 - metrics[m] / ref_scores[m]) for m in ['RMSE', 'MAE']}))
        pd.DataFrame(comparisons).to_csv(work / 'legacy_comparisons.csv', index=False)
        predicted.to_csv(work / 'predictions.csv', index=False)
        with (work / 'model.pkl').open('wb') as handle:
            pickle.dump({'model': model, 'scaler': scaler, 'features': spec['features']}, handle)
        artifact_hashes = {name: digest(work / name) for name in
                           ['legacy_comparisons.csv', 'predictions.csv', 'model.pkl']}
        result = dict(experiment_version=config['experiment_version'], stage='P1', protocol=config['protocol'],
                      run_id=run_id, task=index, fold_id=fold_id, model_family=family, experiment=group,
                      lead_hours=3 * steps, seed=config['seed'], lookback=config['lookback'],
                      features=spec['features'], flatten_order='oldest first; channel order within time',
                      scaler='train-period MinMaxScaler shared with legacy window pipeline',
                      split=split, counts=[len(yt), len(yv), len(ye)], metrics=metrics,
                      validation_RMSE=validation_rmse, selected=parameters, trials=trials,
                      unique_grids=grids[['center_y', 'center_x']].drop_duplicates().shape[0],
                      fingerprints=fingerprints, artifact_hashes=artifact_hashes,
                      preparation_seconds=preparation_seconds, selection_seconds=selection_seconds,
                      evaluation_and_output_seconds=perf_counter()-phase,
                      elapsed_seconds=perf_counter()-started,
                      slurm_job_id=os.environ.get('SLURM_JOB_ID'), slurm_cpus=os.environ.get('SLURM_CPUS_PER_TASK'))
        (work / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        work.rename(final)
        print(f'[P1] Completed: {final}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--task', type=int)
    mode.add_argument('--list', action='store_true')
    args = parser.parse_args()
    config = json.loads(project_path(CONFIG).read_text())
    plan = tasks(config)
    if args.list:
        for index, task in enumerate(plan):
            print(index, *task)
        print(f'{len(plan)} selected fits; {sum(len(config["models"][t[2]]) for t in plan)} candidate fits including selected models.')
        return
    if not 0 <= args.task < len(plan):
        parser.error('Task outside configured range.')
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Real fitting requires a manually submitted Slurm job.')
    execute(args.task, config, project_path(OUTPUT))


if __name__ == '__main__':
    main()
