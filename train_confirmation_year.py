"""Frozen independent-year refits; causal windows and static contracts reused unchanged."""
import argparse
import json
import os
import pickle
from time import perf_counter

import numpy as np
import pandas as pd

from analyze_experiments import attach_centers, collapse_grids, score
from confirmation_protocol import (DEFAULT_REQUEST, GROUPS, check_samples, load_frozen,
    read_json, run_id, static_array, task_list)
from direction_pipeline import output_transaction, sha256
from p3b_contract import fit_coordinates
from rs_static_contract import STATIC_FEATURES, build_fusion, fit_static
from temporal_folds import FoldSpec, project_path
from train import load_and_preprocess_data, prediction_frame
from train_p3 import build_partitions, load_static_bundle, sample_signature


def verify_labels_and_persistence(df, samples):
    lookup = df.set_index(['point_id', 'datetime'], verify_integrity=True).ws_10
    for _, labels, persistence, metadata in samples.values():
        origin = pd.MultiIndex.from_arrays([metadata.point_id, metadata.forecast_origin])
        target = pd.MultiIndex.from_arrays([metadata.point_id, metadata.target_time])
        np.testing.assert_array_equal(persistence, lookup.loc[origin].to_numpy())
        np.testing.assert_array_equal(labels, lookup.loc[target].to_numpy())


def execute(args):
    started = perf_counter()
    request, template, frozen, _ = load_frozen(args.request, full=True)
    verification_seconds = perf_counter()-started
    seed, fold_id, group = task_list(request)[args.task]
    identity = run_id(args.task, request)
    fold = FoldSpec.from_dict(next(f for f in request['folds'] if f['fold_id'] == fold_id))
    if fold.role != 'locked_confirmation':
        raise ValueError('Only the frozen confirmation windows are allowed.')
    features = template['dynamic_groups']['A01']['features']
    data = project_path(request['data_dir'])
    ready = read_json(data/'READY.json')
    if ready['feature_order'] != features or ready['A01_sha256'] != sha256(data/'A01.csv'):
        raise ValueError('Frozen dynamic data/order mismatch.')
    fingerprint = dict(frozen_sha256=sha256(project_path(args.request).parent/'FROZEN.json'),
                       task=args.task, run_id=identity)
    final = project_path(request['run_dir'])/identity
    with output_transaction(final, fingerprint) as work:
        if work is None:
            print(f'Verified completed confirmation fit: {identity}')
            return
        import tensorflow as tf
        from tensorflow.keras.callbacks import CSVLogger, EarlyStopping

        tf.keras.utils.set_random_seed(seed)
        if tf.config.list_physical_devices('GPU'):
            raise ValueError('v1 timing protocol is CPU-only; use its Slurm wrapper.')
        preparation_started = perf_counter()
        table, mapping, _ = load_static_bundle(template)
        centers = pd.read_csv(data/'centers.csv')
        aligned = mapping.merge(centers, on='point_id', validate='one_to_one', suffixes=('', '_new'))
        if len(aligned) != 100:
            raise ValueError('Incomplete static/dynamic target mapping.')
        for name in ['center_y', 'center_x', 'grid_lat', 'grid_lon']:
            np.testing.assert_allclose(aligned[name], aligned[name+'_new'], atol=1e-8, rtol=0)
        df = load_and_preprocess_data(data/'A01.csv', features)
        samples, dynamic_scaler, split = build_partitions(df, features, fold, 56, 1)
        counts = check_samples(samples, fold)
        verify_labels_and_persistence(df, samples)
        training_points = samples['train'][3].point_id.unique()
        train_grids = sorted(set(mapping.set_index('point_id').loc[training_points, 'grid_id']))
        if len(train_grids) != 98:
            raise ValueError('Require all 98 training grids.')
        static_scaler = fit_static(table, train_grids)
        coordinate_scaler = fit_coordinates(mapping, train_grids)
        prepared = {name: dict(dynamic=value[0], static=static_array(value[3], mapping,
            table, static_scaler, coordinate_scaler, group)) for name, value in samples.items()}
        signatures = {name: sample_signature(v[3], v[1], v[2]) for name, v in samples.items()}
        preparation_seconds = perf_counter()-preparation_started
        model = build_fusion((56, 11))
        if model.count_params() != 20761:
            raise ValueError('Fusion architecture changed.')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=template['lr']),
                      loss='mse', metrics=['mae'])
        fit_started = perf_counter()
        history = model.fit(prepared['train'], samples['train'][1],
            validation_data=(prepared['validation'], samples['validation'][1]),
            epochs=template['epochs'], batch_size=template['batch_size'], verbose=2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=template['patience'], restore_best_weights=True),
                       CSVLogger(str(work/'training_log.csv'))])
        fit_seconds = perf_counter()-fit_started
        validation = model.predict(prepared['validation'], batch_size=64, verbose=0).ravel()
        prediction_started = perf_counter()
        predicted = model.predict(prepared['test'], batch_size=64, verbose=0).ravel()
        prediction_seconds = perf_counter()-prediction_started
        if not np.isfinite(validation).all() or not np.isfinite(predicted).all():
            raise ValueError('Nonfinite predictions.')
        _, labels, persistence, metadata = samples['test']
        frame = attach_centers(prediction_frame(metadata, labels, predicted, persistence,
                                              identity, seed, group, fold_id), centers)
        frame['experiment_version'] = request['version']
        frame['confirmation_year'] = request['year']
        unique = collapse_grids(frame)
        if len(unique[['center_y', 'center_x']].drop_duplicates()) != 98:
            raise ValueError('Target coverage changed.')
        metrics = score(unique.y_true, unique.y_pred, unique.persistence_pred)
        if not np.isfinite(list(metrics.values())).all():
            raise ValueError('Nonfinite scores.')
        frame.to_csv(work/'predictions.csv', index=False)
        model.save(work/'model.keras')
        with (work/'preprocessors.pkl').open('wb') as stream:
            pickle.dump(dict(dynamic_scaler=dynamic_scaler, static_scaler=static_scaler,
                coordinate_scaler=coordinate_scaler, training_grid_ids=train_grids,
                dynamic_features=features, raw_static_features=list(STATIC_FEATURES),
                static_mode=group, mask_stage='after scaling'), stream)
        (work/'config.json').write_text(json.dumps(dict(request=request, template=template,
            split=split, task=args.task, seed=seed, fold_id=fold_id, group=group), indent=2))
        (work/'FROZEN.json').write_text(json.dumps(frozen, indent=2))
        result = dict(status='completed', task=args.task, run_id=identity, seed=seed,
            fold_id=fold_id, group=group, year=request['year'], metrics=metrics,
            counts=counts, sample_signatures=signatures, unique_grids=98,
            parameter_count=model.count_params(), dynamic_channels=11, static_channels=14,
            epochs_run=len(history.history['loss']), best_epoch=int(np.argmin(history.history['val_loss']))+1,
            validation_RMSE=float(np.sqrt(np.mean((validation-samples['validation'][1])**2))),
            validation_MAE=float(np.mean(np.abs(validation-samples['validation'][1]))),
            verification_seconds=verification_seconds, preparation_seconds=preparation_seconds,
            fit_seconds=fit_seconds, prediction_seconds=prediction_seconds,
            elapsed_seconds=perf_counter()-started, execution_device='CPU',
            slurm_job_id=os.environ['SLURM_JOB_ID'], worker_threads=os.environ.get('SLURM_CPUS_PER_TASK'),
            timing_scope='fit includes epoch validation/callbacks; prediction is one full test model.predict '
                'including framework overhead, not warmed kernel latency; wall excludes queue/module launch')
        (work/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'[Independent-year fit] Completed {identity}; retain all outcomes.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', default=DEFAULT_REQUEST)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--task', type=int)
    mode.add_argument('--list', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.list:
        from confirmation_protocol import validate_request
        request = read_json(args.request)
        validate_request(request)
        for i, task in enumerate(task_list(request)):
            print(i, *task)
        return
    if args.preflight:
        request, template, _, _ = load_frozen(args.request)
        load_static_bundle(template)
        print(f'Metadata preflight passed: year={request["year"]}, 45 fits; NO training.')
        return
    if not os.environ.get('SLURM_JOB_ID') or args.task not in range(45):
        parser.error('Real fitting requires sbatch slurm/run_confirmation_year.sh with task 0–44.')
    execute(args)


if __name__ == '__main__':
    main()
