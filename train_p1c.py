"""Same-shape direction ablations on the existing v2_fixed sample protocol."""

import argparse
from itertools import product
import json
import os
import pickle
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analyze_experiments import attach_centers, collapse_grids, score
from direction_pipeline import (GROUPS, code_hashes, direction_features, mask_direction,
                                output_transaction, sha256, verify_direction_data)
from temporal_folds import load_manifest, project_path
from train import build_model, create_dataset, load_and_preprocess_data, prediction_frame
from train_p1_baseline import verify_reference

CONFIG = 'configs/rs_hpc_v3_p1c.json'
OUTPUT = 'runs/rs_hpc_v3/P1c'


def task_list(config):
    return list(product(config['folds'], config['groups']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--task', type=int)
    modes.add_argument('--list', action='store_true')
    args = parser.parse_args()
    config = json.loads(project_path(CONFIG).read_text())
    tasks = task_list(config)
    if args.list:
        for i, task in enumerate(tasks):
            print(i, *task)
        return
    if args.task not in range(len(tasks)) or not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit a valid array index through run_p1c.sh.')
    started = perf_counter()
    fold_id, group = tasks[args.task]
    input_group, enabled = GROUPS[group]
    metadata, marker = verify_direction_data(config['direction_dir'])
    directory = project_path(config['direction_dir'])
    _, folds, fold_sha = load_manifest(config['fold_manifest'])
    if folds[fold_id].role != 'development':
        raise ValueError('P1c is restricted to development folds.')
    p1 = json.loads(project_path('configs/rs_hpc_v3_p1.json').read_text())
    speed_features = [f for f in p1['groups'][input_group]['features'] if f.startswith('ws_10')]
    features = direction_features(speed_features)
    sources = json.loads(project_path('result/spatial_analysis/analysis_912234/source_runs.json').read_text())
    source = next(r for r in sources if r['fold_id'] == fold_id and r['pred_len'] == config['pred_len']
                  and r['experiment'] == input_group and r['seed'] == config['seed'])
    reference_file = project_path(source['run_dir']) / 'predictions.csv'
    fingerprint = dict(config_sha256=sha256(project_path(CONFIG)), fold_sha256=fold_sha,
                       data_marker_sha256=sha256(directory / 'completion.json'),
                       reference_sha256=sha256(reference_file),
                       feature_order=features, code=code_hashes([
                           'train_p1c.py', 'train.py', 'direction_pipeline.py', 'temporal_folds.py',
                           'analyze_experiments.py', 'train_p1_baseline.py']))
    run_id = f'{args.task:02d}_{fold_id}_{group}_seed{config["seed"]}'
    final = project_path(OUTPUT) / run_id
    with output_transaction(final, fingerprint) as work:
        if work is None:
            print(f'[P1c] Verified existing result: {final}')
            return
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, CSVLogger

        tf.keras.utils.set_random_seed(config['seed'])
        preparation_started = perf_counter()
        df = load_and_preprocess_data(directory / f'{input_group}.csv', features)
        data = create_dataset(df, config['lookback'], config['pred_len'], features,
                              include_metadata=True, protocol=config['protocol'], fold=folds[fold_id])
        xt, yt, xv, yv, xe, ye, persistence, scaler, split, identifiers = data
        xt, xv, xe = (mask_direction(x, features, enabled) for x in (xt, xv, xe))
        preparation_seconds = perf_counter() - preparation_started
        model = build_model(xt.shape[1:], SimpleNamespace(**config))
        fit_started = perf_counter()
        history = model.fit(xt, yt, validation_data=(xv, yv), epochs=config['epochs'],
                            batch_size=config['batch_size'], verbose=2,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=config['patience'],
                                                     restore_best_weights=True),
                                       CSVLogger(str(work / 'training_log.csv'))])
        fit_seconds = perf_counter() - fit_started
        validation_prediction = model.predict(xv, batch_size=config['batch_size'], verbose=0).ravel()
        predict_started = perf_counter()
        predicted = model.predict(xe, batch_size=config['batch_size'], verbose=0).ravel()
        prediction_seconds = perf_counter() - predict_started
        frame = prediction_frame(identifiers, ye, predicted, persistence,
                                 run_id, config['seed'], group, fold_id)
        reference = pd.read_csv(reference_file, parse_dates=['forecast_origin', 'target_time', 'input_start'])
        verify_reference(frame, reference)
        np.testing.assert_array_equal(pd.to_datetime(frame.input_start), pd.to_datetime(reference.input_start))
        frame = attach_centers(frame, pd.read_csv(directory / 'centers.csv'))
        frame['grid_id'] = frame.center_y.astype(str) + ':' + frame.center_x.astype(str)
        frame['experiment_version'] = config['experiment_version']
        frame['direction_enabled'] = enabled
        frame.to_csv(work / 'predictions.csv', index=False)
        unique = collapse_grids(frame)
        model.save(work / 'model.keras')
        with (work / 'preprocessor.pkl').open('wb') as stream:
            pickle.dump(dict(scaler=scaler, features=features, direction_enabled=enabled,
                             mask_stage='after scaling', lookback=config['lookback']), stream)
        (work / 'fold_manifest.json').write_bytes(project_path(config['fold_manifest']).read_bytes())
        (work / 'config.json').write_text(json.dumps(dict(config, fold_id=fold_id, group=group,
                                             features=features, direction_enabled=enabled,
                                             parameter_count=model.count_params(), input_shape=list(xt.shape[1:]),
                                             split=split, tensorflow_version=tf.__version__), indent=2))
        result = dict(status='completed', run_id=run_id, fold_id=fold_id, group=group,
                      seed=config['seed'], counts=[len(yt), len(yv), len(ye)],
                      unique_grids=unique[['center_y', 'center_x']].drop_duplicates().shape[0],
                      metrics=score(unique.y_true, unique.y_pred, unique.persistence_pred),
                      validation_RMSE=float(np.sqrt(np.mean((validation_prediction-yv)**2))),
                      validation_MAE=float(np.mean(np.abs(validation_prediction-yv))),
                      epochs_run=len(history.history['loss']),
                      best_epoch=int(np.argmin(history.history['val_loss']))+1,
                      parameter_count=model.count_params(), input_channels=len(features),
                      preparation_seconds=preparation_seconds, fit_seconds=fit_seconds,
                      prediction_seconds=prediction_seconds, elapsed_seconds=perf_counter()-started,
                      slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                      requested_worker_threads=os.environ.get('SLURM_CPUS_PER_TASK'))
        (work / 'result.json').write_text(json.dumps(result, indent=2))
    print(f'[P1c] Completed: {final}', flush=True)


if __name__ == '__main__':
    main()
