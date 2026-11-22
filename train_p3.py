"""Paired static remote-sensing controls using the existing causal sequence builder."""

import argparse
from itertools import product
import json
import importlib.metadata
import os
import pickle
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from analyze_experiments import attach_centers, collapse_grids, score
from direction_pipeline import code_hashes, output_transaction, sha256, verify_artifacts
from rs_static_contract import RS_GROUPS, STATIC_FEATURES, build_fusion, fit_static, static_inputs
from temporal_folds import expected_fold_counts, load_manifest, project_path, select_fold_panel, verify_readiness
from train import build_test_sequences, create_sequences, load_and_preprocess_data, prediction_frame, split_dataframe
from train_p1_baseline import verify_reference

CONFIG = 'configs/rs_hpc_v3_p3.json'
OUTPUT = 'runs/rs_hpc_v3/P3'


def tasks(config):
    return list(product(config['seeds'], config['folds'], config['groups']))


def build_partitions(df, features, fold, lookback, horizon):
    """Reuse original scaling and window functions, retaining every partition's keys."""
    panel = select_fold_panel(df, fold)
    training, _, _, split = split_dataframe(panel, fold=fold)
    scaler = MinMaxScaler().fit(training[features])
    samples = {
        'train': create_sequences(training, lookback, horizon, features, scaler, 'P3 training'),
        'validation': create_sequences(panel, lookback, horizon, features, scaler, 'P3 validation',
                                       origin_start=split['validation_start'], target_before=split['test_start']),
        'test': build_test_sequences(panel, lookback, horizon, features, scaler, split, 'v2_fixed')
    }
    expected = expected_fold_counts(fold, lookback, horizon, training.point_id.nunique())
    if [len(value[1]) for value in samples.values()] != list(expected.values()):
        raise ValueError('P3 partition counts differ from the fixed-date protocol.')
    return samples, scaler, split


def sample_signature(metadata, labels, persistence):
    frame = metadata.copy()
    frame['y_true'] = labels
    frame['persistence_pred'] = persistence
    import hashlib
    return hashlib.sha256(frame.to_csv(index=False, date_format='%Y-%m-%d %H:%M:%S').encode()).hexdigest()


def load_static_bundle(config):
    directory = project_path(config['static_dir'])
    marker = verify_artifacts(directory)
    if (sha256(directory/'READY.json') != config['static_ready_sha256']
            or sha256(directory/'completion.json') != config['static_completion_sha256']):
        raise ValueError('Static dataset differs from the frozen P3 version.')
    ready = json.loads((directory/'READY.json').read_text())
    if ready['status'] != 'passed' or ready['features'] != list(STATIC_FEATURES):
        raise ValueError('Static features must pass P2 QA in the declared order.')
    for name, key in [('static_features.csv', 'static_features_sha256'),
                      ('point_grid_mapping.csv', 'mapping_sha256'), ('quality.csv', 'quality_sha256')]:
        if sha256(directory/name) != ready[key]:
            raise ValueError(f'Changed static input: {name}')
    table = pd.read_csv(directory/'static_features.csv')
    mapping = pd.read_csv(directory/'point_grid_mapping.csv')
    if (len(table) != 98 or len(mapping) != 100 or table.grid_id.duplicated().any()
            or mapping.point_id.duplicated().any() or set(table.grid_id) != set(mapping.grid_id)
            or not np.isfinite(table[list(STATIC_FEATURES)]).all().all()):
        raise ValueError('Invalid static feature coverage or mapping.')
    return table, mapping, marker


def reference_frame(path, centers):
    frame = pd.read_csv(path, parse_dates=['input_start', 'forecast_origin', 'target_time'])
    return attach_centers(frame, centers) if 'center_y' not in frame else frame


def execute(index, config):
    started = perf_counter()
    seed, fold_id, group = tasks(config)[index]
    input_group, enabled = RS_GROUPS[group]
    if config['direction_enabled'] or config['architecture'] != 'LSTM64_Dropout0.4_StaticDense8_Concat_Dense16_Dense1':
        raise ValueError('P3 v1 freezes V/time inputs and the declared fusion architecture.')
    decision_path = project_path(config['direction_selection'])
    if sha256(decision_path) != config['direction_selection_sha256']:
        raise ValueError('Direction selection changed after P3 was configured.')
    decision = json.loads(decision_path.read_text())
    if decision['direction_enabled'] or decision['status'] != 'validation_selected':
        raise ValueError('Dynamic representation differs from validation selection.')
    table, mapping, _ = load_static_bundle(config)
    spatial = project_path(config['spatial_dir'])
    verify_readiness(spatial, config['fold_manifest'])
    _, folds, fold_sha = load_manifest(config['fold_manifest'])
    fold = folds[fold_id]
    if fold.role != 'development':
        raise ValueError('P3 v1 trains development folds only.')
    features = config['dynamic_groups'][input_group]['features']
    dynamic_path = spatial/config['dynamic_groups'][input_group]['file']
    if sha256(dynamic_path) != config['data_sha256'][input_group]:
        raise ValueError('Dynamic data differs from frozen P3 input.')
    original_centers = pd.read_csv(spatial/'centers.csv')
    checks = mapping.merge(original_centers, on='point_id', validate='one_to_one', suffixes=('', '_old'))
    for column in ['center_y', 'center_x', 'grid_lat', 'grid_lon']:
        np.testing.assert_allclose(checks[column], checks[f'{column}_old'], atol=1e-10, rtol=0)
    sources = json.loads(project_path(config['legacy_sources']).read_text())
    bridges = [r for r in sources if r['fold_id'] == fold_id and r['seed'] == seed
               and r['experiment'] == input_group and r['pred_len'] == config['pred_len']]
    if (len(bridges) != 1 or bridges[0]['fold_manifest_sha256'] != fold_sha
            or bridges[0]['input_len'] != config['lookback']
            or json.loads(bridges[0]['features']) != features):
        raise ValueError('Legacy bridge is missing or uses a different temporal fold.')
    bridge_path = project_path(bridges[0]['run_dir'])/'predictions.csv'
    p1 = pd.read_csv(project_path(config['p1_summary']))
    selected = p1[(p1.fold_id == fold_id) & (p1.lead_hours == 3)]
    if len(selected) != 4 or selected.run_id_strong_center.nunique() != 1:
        raise ValueError('Validation-selected P1 center is incomplete.')
    center_id = selected.run_id_strong_center.iloc[0]
    center_candidates = selected[selected.experiment.eq('T01')].sort_values(['validation_RMSE', 'model_family'])
    if len(center_candidates) != 2 or center_id != center_candidates.run_id.iloc[0]:
        raise ValueError('P1 strong center must be chosen using validation only.')
    center_path = project_path('runs/rs_hpc_v3/P1')/center_id/'predictions.csv'
    center_result = json.loads((center_path.parent/'result.json').read_text())
    if sha256(center_path) != center_result['artifact_hashes']['predictions.csv']:
        raise ValueError('P1 center prediction artifact changed.')
    fingerprint = dict(config_sha256=sha256(project_path(CONFIG)), fold_sha256=fold_sha,
                       static_ready_sha256=config['static_ready_sha256'],
                       static_completion_sha256=config['static_completion_sha256'],
                       direction_selection_sha256=sha256(decision_path),
                       dynamic_data_sha256=sha256(dynamic_path),
                       bridge_predictions_sha256=sha256(bridge_path),
                       strong_center_predictions_sha256=sha256(center_path),
                       p1_summary_sha256=sha256(project_path(config['p1_summary'])),
                       runtime_versions={name:importlib.metadata.version(name)
                                         for name in ['tensorflow', 'keras', 'numpy', 'pandas', 'scikit-learn']},
                       code=code_hashes(['train_p3.py', 'train.py', 'rs_static_contract.py',
                                         'temporal_folds.py', 'direction_pipeline.py',
                                         'analyze_experiments.py', 'train_p1_baseline.py']))
    run_id = f'{index:02d}_{fold_id}_{group}_seed{seed}'
    final = project_path(OUTPUT)/run_id
    with output_transaction(final, fingerprint) as work:
        if work is None:
            print(f'[P3] Verified completed run: {final}')
            return
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, CSVLogger

        tf.keras.utils.set_random_seed(seed)
        preparation_started = perf_counter()
        df = load_and_preprocess_data(dynamic_path, features)
        samples, dynamic_scaler, split = build_partitions(df, features, fold, config['lookback'], config['pred_len'])
        training_points = samples['train'][3].point_id.unique()
        train_grids = mapping.set_index('point_id').loc[training_points, 'grid_id'].tolist()
        static_scaler = fit_static(table, train_grids)
        static_arrays = {partition: static_inputs(values[3], mapping, table, static_scaler, enabled)
                         for partition, values in samples.items()}
        signatures = {name: sample_signature(value[3], value[1], value[2]) for name, value in samples.items()}
        prepared = {}
        for name, value in samples.items():
            prepared[name] = {'dynamic': value[0], 'static': static_arrays[name]}
        preparation_seconds = perf_counter()-preparation_started
        model = build_fusion(samples['train'][0].shape[1:])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']), loss='mse', metrics=['mae'])
        expected_parameters = 18713 if input_group == 'T01' else 20761
        if model.count_params() != expected_parameters:
            raise ValueError('Fusion parameter count differs from the frozen architecture.')
        fit_started = perf_counter()
        history = model.fit(prepared['train'], samples['train'][1],
                            validation_data=(prepared['validation'], samples['validation'][1]),
                            epochs=config['epochs'], batch_size=config['batch_size'], verbose=2,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=config['patience'],
                                                     restore_best_weights=True),
                                       CSVLogger(str(work/'training_log.csv'))])
        fit_seconds = perf_counter()-fit_started
        validation = model.predict(prepared['validation'], batch_size=config['batch_size'], verbose=0).ravel()
        prediction_started = perf_counter()
        predicted = model.predict(prepared['test'], batch_size=config['batch_size'], verbose=0).ravel()
        prediction_seconds = perf_counter()-prediction_started
        _, labels, persistence, metadata = samples['test']
        frame = prediction_frame(metadata, labels, predicted, persistence, run_id, seed, group, fold_id)
        frame = attach_centers(frame, original_centers)
        frame['grid_id'] = frame.center_y.astype(str)+':'+frame.center_x.astype(str)
        frame['static_enabled'] = enabled
        frame['experiment_version'] = config['experiment_version']
        unique = collapse_grids(frame)
        actual_grids = len(unique[['center_y', 'center_x']].drop_duplicates())
        if actual_grids != 98:
            raise ValueError('P3 evaluation requires the original 98 unique target grids.')
        metrics = score(unique.y_true, unique.y_pred, unique.persistence_pred)
        comparisons = []
        for reference, path in [('legacy_'+input_group, bridge_path), ('validation_selected_P1_center', center_path)]:
            other = reference_frame(path, original_centers)
            verify_reference(frame, other)
            pd.testing.assert_series_equal(frame.input_start, other.input_start, check_names=False)
            scored = collapse_grids(other)
            reference_scores = score(scored.y_true, scored.y_pred, scored.persistence_pred)
            comparisons.append(dict(reference=reference, source_run_id=str(other.run_id.iloc[0]),
                                    source_seed=int(other.seed.iloc[0]),
                                    **{m: reference_scores[m] for m in ['RMSE', 'MAE']},
                                    **{f'{m}_gain_pct': 100*(1-metrics[m]/reference_scores[m]) for m in ['RMSE', 'MAE']}))
        frame.to_csv(work/'predictions.csv', index=False)
        pd.DataFrame(comparisons).to_csv(work/'bridge_comparisons.csv', index=False)
        model.save(work/'model.keras')
        with (work/'preprocessors.pkl').open('wb') as stream:
            pickle.dump(dict(dynamic_scaler=dynamic_scaler, static_scaler=static_scaler,
                             dynamic_features=features, static_features=list(STATIC_FEATURES),
                             static_enabled=enabled, disabled_value=0., mask_stage='after scaling',
                             training_grid_ids=sorted(set(train_grids))), stream)
        actual_config = dict(config, run_id=run_id, task=index, seed=seed, fold_id=fold_id, group=group,
                             static_enabled=enabled, dynamic_features=features, static_features=list(STATIC_FEATURES),
                             dynamic_shape=list(samples['train'][0].shape[1:]), static_shape=[14],
                             parameter_count=model.count_params(), split=split,
                             static_training_grids=sorted(set(train_grids)),
                             static_training_constant_columns=[name for name, var in zip(STATIC_FEATURES, static_scaler.var_) if var == 0],
                             tensorflow_version=tf.__version__)
        (work/'config.json').write_text(json.dumps(actual_config, indent=2))
        (work/'fold_manifest.json').write_bytes(project_path(config['fold_manifest']).read_bytes())
        (work/'direction_selection.json').write_bytes(decision_path.read_bytes())
        (work/'static_READY.json').write_bytes((project_path(config['static_dir'])/'READY.json').read_bytes())
        result = dict(status='completed', task=index, run_id=run_id, fold_id=fold_id, group=group, seed=seed,
                      static_enabled=enabled, metrics=metrics, sample_signatures=signatures,
                      counts=[len(v[1]) for v in samples.values()], unique_grids=actual_grids,
                      validation_RMSE=float(np.sqrt(np.mean((validation-samples['validation'][1])**2))),
                      validation_MAE=float(np.mean(np.abs(validation-samples['validation'][1]))),
                      epochs_run=len(history.history['loss']), best_epoch=int(np.argmin(history.history['val_loss']))+1,
                      parameter_count=model.count_params(), dynamic_channels=len(features), static_channels=14,
                      preparation_seconds=preparation_seconds, fit_seconds=fit_seconds,
                      prediction_seconds=prediction_seconds, elapsed_seconds=perf_counter()-started,
                      slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                      worker_threads=os.environ.get('SLURM_CPUS_PER_TASK'))
        (work/'result.json').write_text(json.dumps(result, indent=2))
    print(f'[P3] Completed: {final}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--task', type=int)
    modes.add_argument('--list', action='store_true')
    args = parser.parse_args()
    config = json.loads(project_path(CONFIG).read_text())
    if config['lookback'] != 56 or config['pred_len'] != 1 or config['protocol'] != 'v2_fixed':
        raise ValueError('P3 v1 fixes 56 observations and the 3h direct v2_fixed task.')
    plan = tasks(config)
    if args.list:
        for i, task in enumerate(plan):
            print(i, *task)
        print('Pilot 0-3; initial development 0-35 (36 total); optional formal extension 36-59 (24 more).')
        return
    if args.task not in range(len(plan)) or not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit a configured array task through slurm/run_p3.sh.')
    execute(args.task, config)


if __name__ == '__main__':
    main()
