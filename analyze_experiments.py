"""Export existing model predictions and summarize paired CERRA experiments.

Run with run_analysis.sh in Slurm. Never retrains models or refits saved scalers.
The default pilot/repetition protocol remains unchanged.
"""

import argparse
import gc
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from train import (PROTOCOLS, build_test_sequences, load_and_preprocess_data,
                   prediction_frame, split_dataframe)
from temporal_folds import FoldSpec, load_manifest


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ('T00', 'T01', 'S01', 'S02', 'A01')
KEYS = ['point_id', 'input_start', 'forecast_origin', 'target_time']
PROTOCOL_FIELDS = [
    'model_type', 'input_len', 'lr', 'batch_size', 'epochs', 'patience', 'dropout',
    'hidden_units', 'train_split_ratio', 'val_split_ratio', 'validation_start',
    'test_start', 'train_samples', 'val_samples', 'test_samples', 'evaluation_protocol']


def local_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Path must remain inside project: {path}')
    return path


def discover_runs(runs_root, protocol='v1'):
    records = []
    for table in sorted(runs_root.rglob('results.csv')):
        for row in pd.read_csv(table).to_dict('records'):
            if row.get('evaluation_protocol') != PROTOCOLS[protocol]:
                continue
            experiment = row.get('experiment')
            if experiment not in EXPERIMENTS:
                folder = table.parent.name
                experiment = 'T00' if folder.startswith('first_four_') else folder.split('_')[0]
            if experiment not in EXPERIMENTS:
                continue
            row['experiment'] = experiment
            row['run_dir'] = local_path(row['model_path']).parent
            records.append(row)
    if not records:
        raise ValueError('No recognized completed experiments found.')
    for row in records:
        if not isinstance(row.get('fold_id'), str) or not row['fold_id']:
            row['fold_id'] = 'pilot'
        if protocol == 'v2_fixed' and row['fold_id'] == 'pilot':
            raise ValueError('Fixed-date results must contain a fold_id.')
    rows = pd.DataFrame(records)
    if rows.duplicated(['fold_id', 'experiment', 'pred_len', 'seed']).any():
        raise ValueError('Multiple runs for the same experiment/lead/seed; '
                         'select one set of runs instead of silently choosing the best.')
    if not (rows['evaluation_protocol'] == PROTOCOLS[protocol]).all():
        raise ValueError('Mixed evaluation protocols cannot be combined.')
    fields = PROTOCOL_FIELDS + (['train_start', 'test_end_exclusive', 'fold_role',
                                'fold_manifest_sha256'] if protocol == 'v2_fixed' else [])
    for (fold_id, horizon), group in rows.groupby(['fold_id', 'pred_len']):
        for field in fields:
            if field not in group or group[field].nunique(dropna=False) != 1:
                raise ValueError(f'Inconsistent {field} in {fold_id}, horizon {horizon}.')
    if protocol == 'v2_fixed' and rows.fold_manifest_sha256.dropna().nunique() > 1:
        raise ValueError('Fixed-date runs use different versions of the fold manifest.')
    return records


def export_run(row):
    """Use only saved model/scaler and original test partition; no model fitting."""
    import tensorflow as tf

    run_dir = row['run_dir']
    output = run_dir / 'predictions.csv'
    if output.exists():
        return
    config = json.loads((run_dir / 'config.json').read_text())
    if config['run_id'] != row['run_id'] or config['seed'] != int(row['seed']):
        raise ValueError(f'Run config does not match results: {run_dir}')
    features = json.loads(row['features'])
    if features != config['features']:
        raise ValueError(f'Feature order differs from saved configuration: {run_dir}')
    df = load_and_preprocess_data(local_path(row['data_path']), features)
    protocol = config.get('protocol', 'v1')
    if protocol == 'v2_fixed':
        fold = FoldSpec.from_dict(config['fold_spec'])
        if fold.fold_id != row['fold_id']:
            raise ValueError('Saved configuration and results have different fold IDs.')
        _, _, _, split_info = split_dataframe(df, fold=fold)
    else:
        _, _, _, split_info = split_dataframe(
            df, float(row['train_split_ratio']), float(row['val_split_ratio']))
    if any(split_info[key] != row[key] for key in split_info):
        raise ValueError(f'Time boundaries changed: {run_dir}')
    with (run_dir / 'scaler.pkl').open('rb') as handle:
        scaler = pickle.load(handle)
    if list(scaler.feature_names_in_) != features:
        raise ValueError(f'Saved scaler feature order changed: {run_dir}')
    if PROTOCOLS[protocol] != row.get('evaluation_protocol', PROTOCOLS['v1']):
        raise ValueError(f'Configuration/result protocol mismatch: {run_dir}')
    X, y, baseline, metadata = build_test_sequences(
        df, int(row['input_len']), int(row['pred_len']), features, scaler, split_info, protocol)
    if len(y) != int(row['test_samples']):
        raise ValueError(f'Test sample count changed: {run_dir}')
    model = tf.keras.models.load_model(str(local_path(row['model_path'])), compile=False)
    pred = model.predict(X, batch_size=int(row['batch_size']), verbose=0).reshape(-1)
    metrics = score(y, pred, baseline)
    for key, value in metrics.items():
        if not np.isclose(value, float(row[key]), rtol=1e-4, atol=1e-5):
            raise ValueError(f'Replayed {key} differs from recorded value: {run_dir}')
    frame = prediction_frame(metadata, y, pred, baseline, row['run_id'],
                             int(row['seed']), row['experiment'], row.get('fold_id', 'pilot'))
    temporary = run_dir / 'predictions.csv.tmp'
    frame.to_csv(temporary, index=False)
    temporary.replace(output)
    del model, X, df
    tf.keras.backend.clear_session()
    gc.collect()
    print(f'[Export] {row["experiment"]} lead={row["pred_len"]} seed={row["seed"]}: {output}', flush=True)


def score(y, pred, baseline):
    y, pred, baseline = (np.asarray(a, dtype=float) for a in (y, pred, baseline))
    return {'RMSE': float(np.sqrt(np.mean((pred - y)**2))),
            'MAE': float(np.mean(np.abs(pred - y))),
            'persistence_RMSE': float(np.sqrt(np.mean((baseline - y)**2))),
            'persistence_MAE': float(np.mean(np.abs(baseline - y)))}


def attach_centers(frame, centers):
    """Point IDs remain distinct; unique-grid aggregation is an additional view."""
    if centers['point_id'].duplicated().any():
        raise ValueError('centers.csv has duplicate point IDs.')
    merged = frame.merge(centers, on='point_id', how='left', validate='many_to_one')
    if merged[['center_y', 'center_x', 'grid_lat', 'grid_lon']].isna().any().any():
        raise ValueError('Prediction point IDs are missing from centers.csv.')
    return merged


def collapse_grids(frame):
    """Count each center grid/time once; average equivalent point predictions.

Targets/baselines at a shared grid must agree before aggregation. This is
evaluation deduplication, not retraining on deduplicated training samples.
"""
    keys = ['center_y', 'center_x', 'input_start', 'forecast_origin', 'target_time']
    grouped = frame.groupby(keys, sort=True)
    spread = grouped[['y_true', 'persistence_pred']].max() - grouped[['y_true', 'persistence_pred']].min()
    if (spread.to_numpy() > 1e-5).any():
        raise ValueError('Duplicate center grids have inconsistent targets or baselines.')
    return grouped[['y_true', 'y_pred', 'persistence_pred', 'grid_lat', 'grid_lon']].mean().reset_index()


def read_predictions(row):
    frame = pd.read_csv(row['run_dir'] / 'predictions.csv',
                        parse_dates=['input_start', 'forecast_origin', 'target_time'])
    if frame.duplicated(KEYS).any() or len(frame) != int(row['test_samples']):
        raise ValueError(f'Duplicate/missing test windows: {row["run_dir"]}')
    if not (frame['run_id'] == row['run_id']).all() or not (frame['seed'] == int(row['seed'])).all():
        raise ValueError('Prediction run identity does not match results.')
    fold_id = row.get('fold_id', 'pilot')
    if 'fold_id' not in frame:
        if fold_id != 'pilot':
            raise ValueError('Fixed-date predictions are missing their fold_id.')
        frame['fold_id'] = 'pilot'
    if not (frame['fold_id'] == fold_id).all():
        raise ValueError('Prediction fold_id does not match results.')
    if row.get('evaluation_protocol') == PROTOCOLS['v2_fixed']:
        if (frame.input_start < pd.Timestamp(row['train_start'])).any():
            raise ValueError('Prediction uses context before the fixed fold training start.')
        if (frame.forecast_origin < pd.Timestamp(row['test_start'])).any():
            raise ValueError('Prediction origin precedes the test boundary.')
        if (frame.target_time >= pd.Timestamp(row['test_end_exclusive'])).any():
            raise ValueError('Prediction target crosses the exclusive test end.')
    if not np.isfinite(frame[['y_true', 'y_pred', 'persistence_pred']].to_numpy()).all():
        raise ValueError('Non-finite predictions.')
    hours = (frame['target_time'] - frame['forecast_origin']).dt.total_seconds() / 3600
    if not (hours == int(row['pred_len']) * 3).all():
        raise ValueError('Prediction lead/time alignment is incorrect.')
    metrics = score(frame.y_true, frame.y_pred, frame.persistence_pred)
    if any(not np.isclose(value, float(row[key]), rtol=1e-4, atol=1e-5)
           for key, value in metrics.items()):
        raise ValueError(f'Prediction metrics differ from results: {row["run_dir"]}')
    return frame.sort_values(KEYS).reset_index(drop=True)


def verify_alignment(frame, reference):
    if not frame[KEYS].equals(reference[KEYS]):
        raise ValueError('Experiments use different test point/time windows.')
    for column in ('y_true', 'persistence_pred'):
        if not np.allclose(frame[column], reference[column], rtol=1e-6, atol=1e-5):
            raise ValueError(f'Experiments have different {column} values.')


def analyze(records, centers, output_dir, expected_folds=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    protocols = {row.get('evaluation_protocol', PROTOCOLS['v1']) for row in records}
    if len(protocols) != 1:
        raise ValueError('Analysis must not mix V1 and V2 runs.')
    protocol_name = next(iter(protocols))
    for row in records:
        row.setdefault('fold_id', 'pilot')
    overall, per_point, per_grid, per_time, coverage = [], [], [], [], []
    references = {}
    # T01 is the pairing reference for every seed, lead and point.
    for row in records:
        if row['experiment'] == 'T01':
            references[(row['fold_id'], int(row['seed']), int(row['pred_len']))] = read_predictions(row)
    pilot_references = {(fold_id, horizon): frame for (fold_id, seed, horizon), frame in references.items()
                        if seed == 42}
    for row in records:
        frame = read_predictions(row)
        seed, horizon = int(row['seed']), int(row['pred_len'])
        reference = references.get((row['fold_id'], seed, horizon))
        if reference is not None:
            verify_alignment(frame, reference)
        if (row['fold_id'], horizon) in pilot_references:
            verify_alignment(frame, pilot_references[(row['fold_id'], horizon)])
        frame = attach_centers(frame, centers)
        grids = collapse_grids(frame)
        identity = {'fold_id': row['fold_id'], 'experiment': row['experiment'], 'seed': seed,
                    'lead_hours': horizon * 3, 'run_id': row['run_id']}
        for scope, data in [('point_weighted', frame), ('unique_grid', grids)]:
            overall.append({**identity, 'scope': scope, 'samples': len(data),
                            **score(data.y_true, data.y_pred, data.persistence_pred)})
        for point_id, data in frame.groupby('point_id'):
            per_point.append({**identity, 'point_id': point_id,
                              'center_y': data.center_y.iloc[0], 'center_x': data.center_x.iloc[0],
                              'grid_lat': data.grid_lat.iloc[0], 'grid_lon': data.grid_lon.iloc[0],
                              'samples': len(data), **score(data.y_true, data.y_pred, data.persistence_pred)})
        for (yy, xx), data in grids.groupby(['center_y', 'center_x']):
            per_grid.append({**identity, 'center_y': yy, 'center_x': xx,
                             'grid_lat': data.grid_lat.iloc[0], 'grid_lon': data.grid_lon.iloc[0],
                             'samples': len(data), **score(data.y_true, data.y_pred, data.persistence_pred)})
        for target_time, data in grids.groupby('target_time'):
            per_time.append({**identity, 'target_time': target_time,
                             'grids': len(data), **score(data.y_true, data.y_pred, data.persistence_pred)})
    overall = pd.DataFrame(overall)
    points, grids = pd.DataFrame(per_point), pd.DataFrame(per_grid)
    overall.to_csv(output_dir / 'per_run_metrics.csv', index=False)
    points.to_csv(output_dir / 'per_point_metrics.csv', index=False)
    grids.to_csv(output_dir / 'per_grid_metrics.csv', index=False)
    pd.DataFrame(per_time).to_csv(output_dir / 'per_time_metrics.csv', index=False)
    summary = overall.groupby(['fold_id', 'experiment', 'lead_hours', 'scope']).agg(
        n_seeds=('seed', 'nunique'), RMSE_mean=('RMSE', 'mean'), RMSE_std=('RMSE', 'std'),
        MAE_mean=('MAE', 'mean'), MAE_std=('MAE', 'std')).reset_index()
    summary.to_csv(output_dir / 'seed_summary.csv', index=False)
    baseline = overall[overall.experiment == 'T01']
    paired = overall[overall.experiment != 'T01'].merge(
        baseline[['fold_id', 'seed', 'lead_hours', 'scope', 'RMSE', 'MAE']],
        on=['fold_id', 'seed', 'lead_hours', 'scope'], suffixes=('', '_T01'), validate='many_to_one')
    for metric in ('RMSE', 'MAE'):
        paired[f'{metric}_improvement_pct'] = np.where(
            paired[f'{metric}_T01'] > 0,
            (paired[f'{metric}_T01'] - paired[metric]) / paired[f'{metric}_T01'] * 100, np.nan)
    paired.to_csv(output_dir / 'paired_seed_improvements.csv', index=False)
    paired.groupby(['fold_id', 'experiment', 'lead_hours', 'scope']).agg(
        n_pairs=('seed', 'nunique'), RMSE_gain_mean=('RMSE_improvement_pct', 'mean'),
        RMSE_gain_std=('RMSE_improvement_pct', 'std'),
        MAE_gain_mean=('MAE_improvement_pct', 'mean'),
        MAE_gain_std=('MAE_improvement_pct', 'std')).reset_index().to_csv(
            output_dir / 'paired_seed_summary.csv', index=False)
    # Report fractions over unique grids, so duplicate point mappings do not vote twice.
    grid_pairs = grids[grids.experiment != 'T01'].merge(
        grids[grids.experiment == 'T01'][['fold_id', 'seed', 'lead_hours', 'center_y', 'center_x', 'RMSE', 'MAE']],
        on=['fold_id', 'seed', 'lead_hours', 'center_y', 'center_x'], suffixes=('', '_T01'), validate='many_to_one')
    for metric in ('RMSE', 'MAE'):
        grid_pairs[f'{metric}_improved'] = grid_pairs[metric] < grid_pairs[f'{metric}_T01']
    grid_pairs.to_csv(output_dir / 'paired_grid_improvements.csv', index=False)
    if not grid_pairs.empty:
        grid_pairs.groupby(['fold_id', 'experiment', 'seed', 'lead_hours']).agg(
            grids=('center_y', 'size'), RMSE_improved_fraction=('RMSE_improved', 'mean'),
            MAE_improved_fraction=('MAE_improved', 'mean')).reset_index().to_csv(
                output_dir / 'grid_improvement_coverage.csv', index=False)
    # Expose missing runs instead of presenting partial arrays as five-seed results.
    fold_ids = list(expected_folds) if expected_folds else sorted(overall.fold_id.unique())
    for fold_id in fold_ids:
        role = expected_folds[fold_id].role if expected_folds else 'pilot'
        for experiment in EXPERIMENTS:
            if experiment == 'T00' and not (overall.experiment == 'T00').any():
                continue
            for lead in (3, 6, 12, 24):
                present = sorted(overall[(overall.fold_id == fold_id) & (overall.experiment == experiment)
                                         & (overall.lead_hours == lead)].seed.unique().tolist())
                expected = [42] if experiment == 'T00' else [42, 43, 44, 45, 46]
                coverage.append({'fold_id': fold_id, 'fold_role': role, 'experiment': experiment,
                                 'lead_hours': lead, 'seeds': present,
                                 'missing_seeds': sorted(set(expected) - set(present)),
                                 'status': 'reserved_confirmation' if role == 'locked_confirmation' and not present
                                 else ('complete' if set(expected).issubset(present) else 'incomplete')})
    (output_dir / 'coverage.json').write_text(json.dumps(coverage, indent=2), encoding='utf-8')
    manifest = [{key: str(value) if isinstance(value, Path) else (None if pd.isna(value) else value)
                 for key, value in row.items()}
                for row in records]
    (output_dir / 'source_runs.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output_dir / 'analysis_notes.txt').write_text(
        f'Evaluation protocol: {protocol_name}\n'
        'All metrics are m/s. Positive paired gains mean lower error than T01.\n'
        'Seed std is sample standard deviation (ddof=1), undefined for one seed.\n'
        'Seed variation is training variation, not temporal generalization uncertainty.\n'
        'Unique-grid metrics average predictions for duplicate point mappings before scoring; '
        'training data are unchanged.\n'
        'Same forecast lead is compared using exactly matched point/time keys and targets.\n'
        'T00, if present, is a single-seed supplementary ablation. '
        'See coverage.json for missing runs.\n'
        'V1 has short effective validation coverage. V2 expands historical context; '
        'both require additional independent temporal validation.\n', encoding='utf-8')
    for fold_id in sorted(overall.fold_id.unique()):
        destination = output_dir if fold_id == 'pilot' else output_dir / fold_id
        destination.mkdir(exist_ok=True)
        make_plot(summary[summary.fold_id == fold_id], overall[overall.fold_id == fold_id], destination)
    print(f'[Analysis] {len(records)} completed runs; outputs: {output_dir}', flush=True)


def make_plot(summary, overall, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ('RMSE', 'MAE')):
        for experiment in EXPERIMENTS:
            data = summary[(summary.experiment == experiment) & (summary.scope == 'unique_grid')]
            if not data.empty:
                ax.errorbar(data.lead_hours, data[f'{metric}_mean'],
                            yerr=data[f'{metric}_std'].fillna(0), marker='o', capsize=3, label=experiment)
        baseline = overall[(overall.experiment == 'T01') & (overall.scope == 'unique_grid')]
        baseline = baseline.groupby('lead_hours')[f'persistence_{metric}'].mean()
        ax.plot(baseline.index, baseline.values, '--', label='B01')
        ax.set(xlabel='Forecast lead (hours)', ylabel=f'{metric} (m/s)', xticks=[3, 6, 12, 24])
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle('Unique center grids; mean +/- seed standard deviation')
    fig.tight_layout()
    fig.savefig(output_dir / 'unique_grid_comparison.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=list(PROTOCOLS), default='v1')
    parser.add_argument('--fold_manifest', type=local_path)
    parser.add_argument('--fold_id', help='Analyze just one fixed-date fold.')
    parser.add_argument('--runs_root', type=local_path, default=ROOT / 'runs/ablation')
    parser.add_argument('--centers', type=local_path,
                        default=ROOT / 'data/CERRA/spatial_911958/centers.csv')
    parser.add_argument('--output_dir', type=local_path,
                        default=ROOT / 'result/spatial_analysis/repeated')
    parser.add_argument('--skip_export', action='store_true', help='Require prediction files to exist already.')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run through sbatch run_analysis.sh; model replay belongs on compute nodes.')
    records = discover_runs(args.runs_root, args.protocol)
    expected_folds = None
    if args.protocol == 'v2_fixed':
        if args.fold_manifest:
            _, expected_folds, digest = load_manifest(args.fold_manifest)
            if any(row.get('fold_manifest_sha256') != digest for row in records):
                parser.error('Results do not match the supplied fold manifest version.')
        else:
            expected_folds = {}
            for row in records:
                if isinstance(row.get('fold_manifest_sha256'), str) and row['fold_manifest_sha256']:
                    parser.error('Manifest-based runs require their --fold_manifest for analysis.')
                config = json.loads((row['run_dir'] / 'config.json').read_text())
                fold = FoldSpec.from_dict(config['fold_spec'])
                expected_folds[fold.fold_id] = fold
        if args.fold_id:
            if args.fold_id not in expected_folds:
                parser.error('Unknown fold_id.')
            expected_folds = {args.fold_id: expected_folds[args.fold_id]}
            records = [row for row in records if row['fold_id'] == args.fold_id]
            if not records:
                parser.error('No completed runs for the requested fold.')
    elif args.fold_id or args.fold_manifest:
        parser.error('Fold arguments require --protocol v2_fixed.')
    if not args.skip_export:
        for row in records:
            export_run(row)
    analyze(records, pd.read_csv(args.centers), args.output_dir, expected_folds)


if __name__ == '__main__':
    main()
