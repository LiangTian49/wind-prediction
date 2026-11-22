"""Spatial coordinate/land-cover/DSM controls; real fitting is Slurm-only."""

import argparse
import json
import os
import pickle
from time import perf_counter

import numpy as np
import pandas as pd

from analyze_experiments import attach_centers, collapse_grids, score
from direction_pipeline import code_hashes, output_transaction, sha256
from p3b_contract import (CONFIG, OUTPUT, references, representation, require_signatures,
                          runtime_versions, fit_coordinates, slot_names, tasks, validate_config)
from rs_static_contract import STATIC_FEATURES, build_fusion, fit_static
from temporal_folds import load_manifest, project_path, verify_readiness
from train import load_and_preprocess_data, prediction_frame
from train_p3 import build_partitions, load_static_bundle, sample_signature, reference_frame
from train_p1_baseline import verify_reference


def execute(index, config, sources):
    started = perf_counter()
    seed, fold_id, group = tasks(config)[index]
    reused = references(config, sources, fold_id, seed, full_artifacts=True)
    if len(reused) != 2:
        raise ValueError('Missing matched P3 spatial pair.')
    table, mapping, _ = load_static_bundle(config)
    spatial = project_path(config['spatial_dir'])
    verify_readiness(spatial, config['fold_manifest'])
    _, folds, fold_sha = load_manifest(config['fold_manifest'])
    fold = folds[fold_id]
    if fold.role != 'development':
        raise ValueError('P3b does not train winter confirmation or new test folds.')
    features = config['dynamic_groups']['A01']['features']
    dynamic_path = spatial/config['dynamic_groups']['A01']['file']
    if sha256(dynamic_path) != config['data_sha256']['A01']:
        raise ValueError('Frozen spatial data changed.')
    decision_path = project_path(config['direction_selection'])
    if sha256(decision_path) != config['direction_selection_sha256']:
        raise ValueError('Direction selection changed.')
    decision = json.loads(decision_path.read_text())
    if decision['direction_enabled'] or decision['status'] != 'validation_selected':
        raise ValueError('Direction selection does not match the no-direction protocol.')
    centers = pd.read_csv(spatial/'centers.csv')
    check = mapping.merge(centers, on='point_id', validate='one_to_one', suffixes=('', '_source'))
    if len(check) != 100:
        raise ValueError('Incomplete target-grid mapping.')
    for column in ['center_y', 'center_x', 'grid_lat', 'grid_lon']:
        np.testing.assert_allclose(check[column], check[column+'_source'], atol=1e-10, rtol=0)
    fingerprint = dict(config_sha256=sha256(project_path(CONFIG)), fold_sha256=fold_sha,
        static_ready_sha256=config['static_ready_sha256'], static_completion_sha256=config['static_completion_sha256'],
        dynamic_data_sha256=sha256(dynamic_path), direction_selection_sha256=sha256(decision_path),
        reference_manifest_sha256=config['reference_manifest_sha256'], runtime_versions=runtime_versions(),
        reference_markers={key:record['source']['completion_sha256'] for key, record in reused.items()},
        code=code_hashes(['train_p3b.py', 'p3b_contract.py', 'train_p3.py', 'train.py',
                          'rs_static_contract.py', 'temporal_folds.py', 'direction_pipeline.py',
                          'analyze_experiments.py', 'train_p1_baseline.py']))
    run_id = f'{index:02d}_{fold_id}_{group}_seed{seed}'
    final = project_path(OUTPUT)/run_id
    with output_transaction(final, fingerprint) as work:
        if work is None:
            print(f'[P3b] Verified completed run: {final}')
            return
        import tensorflow as tf
        from tensorflow.keras.callbacks import CSVLogger, EarlyStopping

        tf.keras.utils.set_random_seed(seed)
        preparation_started = perf_counter()
        df = load_and_preprocess_data(dynamic_path, features)
        samples, dynamic_scaler, split = build_partitions(df, features, fold, config['lookback'], config['pred_len'])
        signatures = {name:sample_signature(v[3], v[1], v[2]) for name, v in samples.items()}
        require_signatures(signatures, reused)
        train_points = samples['train'][3].point_id.unique()
        train_grids = sorted(set(mapping.set_index('point_id').loc[train_points, 'grid_id']))
        static_scaler = fit_static(table, train_grids)
        coordinate_scaler = fit_coordinates(mapping, train_grids)
        for record in reused.values():
            with (record['path']/'preprocessors.pkl').open('rb') as stream:
                old = pickle.load(stream)
            if (old['training_grid_ids'] != train_grids or old['dynamic_features'] != features
                    or old['static_features'] != list(STATIC_FEATURES)):
                raise ValueError('Training preprocessing provenance differs from matched P3.')
            for name, scaler, attrs in [('dynamic_scaler', dynamic_scaler, ['data_min_', 'data_max_', 'scale_', 'min_']),
                                        ('static_scaler', static_scaler, ['mean_', 'var_', 'scale_'])]:
                for attr in attrs:
                    np.testing.assert_array_equal(getattr(scaler, attr), getattr(old[name], attr))
        prepared = {name:{'dynamic':v[0], 'static':representation(v[3], mapping, table,
                    static_scaler, coordinate_scaler, group)} for name, v in samples.items()}
        preparation_seconds = perf_counter()-preparation_started
        # Validate archived sample keys and values before expensive fitting, not after it.
        _, labels, persistence, metadata = samples['test']
        probe = attach_centers(prediction_frame(metadata, labels, persistence, persistence,
                                               run_id, seed, group, fold_id), centers)
        for record in reused.values():
            other = reference_frame(record['path']/'predictions.csv', centers)
            verify_reference(probe, other)
            pd.testing.assert_series_equal(probe.input_start, other.input_start, check_names=False)
        print('[P3b] Pre-fit reference, scaler and all-partition sample checks passed.', flush=True)
        model = build_fusion(samples['train'][0].shape[1:])
        if model.count_params() != 20761:
            raise ValueError('P3b must retain the matched P3 fusion architecture.')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']), loss='mse', metrics=['mae'])
        fit_started = perf_counter()
        history = model.fit(prepared['train'], samples['train'][1],
            validation_data=(prepared['validation'], samples['validation'][1]),
            epochs=config['epochs'], batch_size=config['batch_size'], verbose=2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=config['patience'], restore_best_weights=True),
                       CSVLogger(str(work/'training_log.csv'))])
        fit_seconds = perf_counter()-fit_started
        validation = model.predict(prepared['validation'], batch_size=config['batch_size'], verbose=0).ravel()
        prediction_started = perf_counter()
        predicted = model.predict(prepared['test'], batch_size=config['batch_size'], verbose=0).ravel()
        prediction_seconds = perf_counter()-prediction_started
        frame = attach_centers(prediction_frame(metadata, labels, predicted, persistence,
                                                run_id, seed, group, fold_id), centers)
        frame['grid_id'] = frame.center_y.astype(str)+':'+frame.center_x.astype(str)
        frame['static_mode'] = group
        frame['experiment_version'] = config['experiment_version']
        unique = collapse_grids(frame)
        unique_grids = len(unique[['center_y', 'center_x']].drop_duplicates())
        if unique_grids != 98 or not np.isfinite(validation).all() or not np.isfinite(predicted).all():
            raise ValueError('Nonfinite predictions or changed target grid coverage.')
        metrics = score(unique.y_true, unique.y_pred, unique.persistence_pred)
        if not np.isfinite(list(metrics.values())).all():
            raise ValueError('Nonfinite evaluation metrics.')
        frame.to_csv(work/'predictions.csv', index=False)
        model.save(work/'model.keras')
        with (work/'preprocessors.pkl').open('wb') as stream:
            pickle.dump(dict(dynamic_scaler=dynamic_scaler, static_scaler=static_scaler,
                coordinate_scaler=coordinate_scaler, static_mode=group, static_slot_names=slot_names(group),
                dynamic_features=features, raw_static_features=list(STATIC_FEATURES), mask_stage='after scaling',
                training_grid_ids=train_grids), stream)
        (work/'reference_sources.json').write_text(json.dumps([r['source'] for r in reused.values()], indent=2))
        actual = dict(config, task=index, run_id=run_id, fold_id=fold_id, seed=seed, group=group,
            dynamic_features=features, dynamic_shape=[56, 11], static_shape=[14], static_slot_names=slot_names(group),
            parameter_count=model.count_params(), split=split, static_training_grids=train_grids,
            static_training_constant_columns=[n for n,v in zip(STATIC_FEATURES, static_scaler.var_) if v == 0],
            coordinate_training_constant_columns=[n for n,v in zip(['grid_lat','grid_lon'], coordinate_scaler.var_) if v == 0])
        (work/'config.json').write_text(json.dumps(actual, indent=2))
        for destination, source in [('fold_manifest.json', config['fold_manifest']),
                                    ('direction_selection.json', config['direction_selection']),
                                    ('static_READY.json', str(project_path(config['static_dir'])/'READY.json'))]:
            (work/destination).write_bytes(project_path(source).read_bytes())
        result = dict(status='completed', task=index, run_id=run_id, fold_id=fold_id, seed=seed, group=group,
            metrics=metrics, sample_signatures=signatures, counts=[len(v[1]) for v in samples.values()],
            unique_grids=unique_grids, parameter_count=model.count_params(), dynamic_channels=11, static_channels=14,
            validation_RMSE=float(np.sqrt(np.mean((validation-samples['validation'][1])**2))),
            validation_MAE=float(np.mean(np.abs(validation-samples['validation'][1]))),
            epochs_run=len(history.history['loss']), best_epoch=int(np.argmin(history.history['val_loss']))+1,
            preparation_seconds=preparation_seconds, fit_seconds=fit_seconds, prediction_seconds=prediction_seconds,
            elapsed_seconds=perf_counter()-started, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
            worker_threads=os.environ.get('SLURM_CPUS_PER_TASK'), execution_device='CPU')
        (work/'result.json').write_text(json.dumps(result, indent=2))
    print(f'[P3b] Completed: {final}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--task', type=int)
    modes.add_argument('--list', action='store_true')
    modes.add_argument('--preflight', action='store_true', help='Metadata-only; no windows or training')
    args = parser.parse_args()
    config = json.loads(project_path(CONFIG).read_text())
    sources = validate_config(config)
    if args.list:
        for i, task in enumerate(tasks(config)):
            print(i, *task)
        return
    if args.preflight:
        reused = references(config, sources)
        load_static_bundle(config)
        print(f'[P3b] Metadata preflight passed: {len(reused)} references; 27 new tasks. No fitting.')
        return
    if args.task not in range(27) or not os.environ.get('SLURM_JOB_ID'):
        parser.error('Real training requires sbatch slurm/run_p3b.sh with index 0–26.')
    execute(args.task, config, sources)


if __name__ == '__main__':
    main()
