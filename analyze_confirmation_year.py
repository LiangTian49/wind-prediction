"""Audit confirmation coverage and calculate the frozen equal-window paired estimand."""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

from analyze_experiments import collapse_grids, score
from confirmation_protocol import (DEFAULT_REQUEST, GROUPS, block_weights, load_frozen,
    paired_gain, read_json, run_id, task_list)
from direction_pipeline import output_transaction, sha256, verify_artifacts
from p3b_contract import COMPARISONS
from temporal_folds import project_path


def collect(request, frozen_sha, expected):
    root = project_path(request['run_dir'])
    missing, rows, signatures, scalers = [], [], {}, {}
    for index, (seed, fold, group) in enumerate(task_list(request)):
        if expected == 'pilot' and index >= 5:
            continue
        identity = run_id(index, request)
        path = root/identity
        if not path.exists():
            missing.append(index)
            continue
        marker = verify_artifacts(path)
        if marker['fingerprint'] != dict(frozen_sha256=frozen_sha, task=index, run_id=identity):
            raise ValueError('Mixed frozen protocol/run identity.')
        result = read_json(path/'result.json')
        expected_values = dict(status='completed', task=index, run_id=identity, seed=seed,
            fold_id=fold, group=group, year=request['year'], counts=[42400, 22300, 22300],
            unique_grids=98, parameter_count=20761, dynamic_channels=11, static_channels=14)
        if any(result.get(k) != v for k, v in expected_values.items()):
            raise ValueError(f'Incomplete/mismatched confirmation result: {identity}')
        if fold in signatures and result['sample_signatures'] != signatures[fold]:
            raise ValueError('All seeds/static conditions must have identical per-partition samples.')
        signatures[fold] = result['sample_signatures']
        with (path/'preprocessors.pkl').open('rb') as stream:
            pre = pickle.load(stream)
        attrs = {'dynamic_scaler': ['data_min_', 'data_max_', 'scale_', 'min_'],
                 'static_scaler': ['mean_', 'var_', 'scale_'],
                 'coordinate_scaler': ['mean_', 'var_', 'scale_']}
        if fold in scalers:
            old = scalers[fold]
            if pre['training_grid_ids'] != old['training_grid_ids']:
                raise ValueError('Training grids changed between static controls.')
            for name, names in attrs.items():
                for attribute in names:
                    np.testing.assert_array_equal(getattr(pre[name], attribute), getattr(old[name], attribute))
        scalers[fold] = pre
        values = list(result['metrics'].values()) + [result['validation_RMSE'], result['validation_MAE']]
        if not np.isfinite(values).all():
            raise ValueError('Missing/nonfinite metrics.')
        rows.append({k: result[k] for k in ['task', 'run_id', 'seed', 'fold_id', 'group', 'year',
            'validation_RMSE', 'validation_MAE', 'epochs_run', 'best_epoch', 'fit_seconds',
            'prediction_seconds', 'elapsed_seconds']} | result['metrics'] |
            dict(run_dir=str(path), completion_sha256=sha256(path/'completion.json')))
    count = 5 if expected == 'pilot' else 45
    print(f'[Confirmation coverage] {len(rows)}/{count}; missing task indices={missing}', flush=True)
    if missing or len(rows) != count:
        raise ValueError('Incomplete confirmation: no pooled conclusion or intervals are published.')
    return pd.DataFrame(rows)


def daily_frame(raw, fold):
    frame = collapse_grids(raw).sort_values(['center_y', 'center_x', 'forecast_origin', 'target_time']).reset_index(drop=True)
    if len(frame) != 98*223 or frame.duplicated(['center_y', 'center_x', 'forecast_origin', 'target_time']).any():
        raise ValueError('Duplicate/missing unique-grid prediction keys.')
    origin, target = pd.to_datetime(frame.forecast_origin), pd.to_datetime(frame.target_time)
    if not ((origin >= pd.Timestamp(fold['test_start'])).all() and
            (target < pd.Timestamp(fold['test_end_exclusive'])).all() and
            (target-origin == pd.Timedelta(hours=3)).all()):
        raise ValueError('Forecast boundaries changed.')
    frame['date'] = origin.dt.normalize()
    error = frame.y_pred-frame.y_true
    persistence = frame.persistence_pred-frame.y_true
    frame['squared'], frame['absolute'] = error**2, error.abs()
    frame['persistence_squared'], frame['persistence_absolute'] = persistence**2, persistence.abs()
    daily = frame.groupby('date')[['squared', 'absolute', 'persistence_squared', 'persistence_absolute']].sum()
    daily['count'] = frame.groupby('date').size()
    if len(daily) != 28:
        raise ValueError('Expected all 28 origin dates; no date removal.')
    return frame, daily


def analyze(summary, request, work):
    rng = np.random.default_rng(request['bootstrap_seed'])
    pairs = COMPARISONS + [(g, 'Persistence') for g in GROUPS]
    intervals, contributions, aggregate = [], [], {}
    for fold in request['folds']:
        daily, baseline = {}, None
        block = summary[summary.fold_id.eq(fold['fold_id'])]
        for row in block.itertuples():
            raw = pd.read_csv(project_path(row.run_dir)/'predictions.csv',
                              parse_dates=['input_start', 'forecast_origin', 'target_time'])
            if (len(raw) != 22300 or raw.duplicated(['point_id', 'forecast_origin', 'target_time']).any()
                    or not (raw.run_id == row.run_id).all() or not (raw.seed == row.seed).all()
                    or not (raw.fold_id == row.fold_id).all()):
                raise ValueError('Prediction identity/point count differs from saved result.')
            frame, errors = daily_frame(raw, fold)
            keys = ['center_y', 'center_x', 'input_start', 'forecast_origin', 'target_time']
            if baseline is None:
                baseline = frame
            pd.testing.assert_frame_equal(frame[keys], baseline[keys])
            np.testing.assert_allclose(frame[['y_true', 'persistence_pred']], baseline[['y_true', 'persistence_pred']],
                                       rtol=0, atol=1e-6)
            reconstructed = score(frame.y_true, frame.y_pred, frame.persistence_pred)
            for key, value in reconstructed.items():
                if not np.isclose(value, getattr(row, key), rtol=0, atol=1e-8):
                    raise ValueError('Archived score reconstruction mismatch.')
            daily[row.seed, row.group] = errors
        days = next(iter(daily.values())).index
        if any(not value.index.equals(days) for value in daily.values()):
            raise ValueError('Date coverage differs.')
        weights = block_weights(rng, len(days), request['resamples'], request['block_days'])
        for model, reference in pairs:
            for metric, column in [('RMSE', 'squared'), ('MAE', 'absolute')]:
                points, draws = [], []
                for seed in request['seeds']:
                    errors = daily[seed, model]
                    ref_column = 'persistence_'+column if reference == 'Persistence' else column
                    ref = errors if reference == 'Persistence' else daily[seed, reference]
                    np.testing.assert_array_equal(errors['count'], ref['count'])
                    point, distribution = paired_gain(errors[column], ref[ref_column], weights, metric)
                    points.append(point)
                    draws.append(distribution)
                    low, high = np.quantile(distribution, [.025, .975])
                    intervals.append(dict(fold_id=fold['fold_id'], model=model, reference=reference,
                        metric=metric, seed=str(seed), gain_pct=point, date_p025_pct=low, date_p975_pct=high))
                    for date in days:
                        contributions.append(dict(fold_id=fold['fold_id'], model=model, reference=reference,
                            metric=metric, seed=seed, date=str(date.date()), count=int(errors.loc[date, 'count']),
                            model_loss=float(errors.loc[date, column]), reference_loss=float(ref.loc[date, ref_column]),
                            loss_reduction=float(ref.loc[date, ref_column]-errors.loc[date, column])))
                point, distribution = float(np.mean(points)), np.mean(draws, axis=0)
                low, high = np.quantile(distribution, [.025, .975])
                intervals.append(dict(fold_id=fold['fold_id'], model=model, reference=reference,
                    metric=metric, seed='fixed_seed_mean', gain_pct=point, date_p025_pct=low, date_p975_pct=high))
                aggregate.setdefault((model, reference, metric), []).append((point, distribution))
    for (model, reference, metric), folds in aggregate.items():
        point = float(np.mean([v[0] for v in folds]))
        distribution = np.mean([v[1] for v in folds], axis=0)
        low, high = np.quantile(distribution, [.025, .975])
        intervals.append(dict(fold_id='equal_window_mean', model=model, reference=reference,
            metric=metric, seed='fixed_seed_mean', gain_pct=point, date_p025_pct=low, date_p975_pct=high))
    result = pd.DataFrame(intervals)
    if len(result) != 312 or len(contributions) != 6048 or result.isna().any().any():
        raise ValueError('Incomplete date/aggregate analysis.')
    result.to_csv(work/'date_intervals.csv', index=False)
    pd.DataFrame(contributions).to_csv(work/'daily_loss_contributions.csv', index=False)
    individual = result[result.seed.ne('fixed_seed_mean') & result.fold_id.ne('equal_window_mean')]
    individual.groupby(['fold_id', 'model', 'reference', 'metric']).gain_pct.agg(['mean', 'std']).to_csv(
        work/'paired_seed_summary.csv')
    primary = result[(result.fold_id == 'equal_window_mean') & (result.model == 'RS_S1') &
                     (result.reference == 'RS_S0') & (result.metric == 'RMSE')].iloc[0].to_dict()
    review = dict(status='completed', primary=primary,
        primary_interval_entirely_positive=bool(primary['date_p025_pct'] > 0),
        interpretation='Report all outcomes, including no confirmation or deterioration. '
            'Equal-window mean of paired within-seed gains; independent date draws per window; '
            'shared draws across models/seeds/metrics within window. Origin-date circular 3-day blocks, '
            '2000 draws; fixed-seed mean, no prediction ensemble, no seed resampling. '
            'Intervals conditional on selected windows and fixed fitted models; secondary intervals unadjusted.',
        rows=len(result), daily_rows=len(contributions))
    (work/'CONFIRMATION_REVIEW.json').write_text(json.dumps(review, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', default=DEFAULT_REQUEST)
    parser.add_argument('--expected', choices=['pilot', 'full'], default='full')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        parser.error('Use sbatch slurm/run_confirmation_analysis.sh.')
    if args.expected == 'pilot' and not args.summarize_only:
        parser.error('Pilot audits coverage only, never select models/windows from pilot scores.')
    request, _, _, _ = load_frozen(args.request, full=True)
    frozen_sha = sha256(project_path(args.request).parent/'FROZEN.json')
    summary = collect(request, frozen_sha, args.expected)
    output = project_path(request['result_dir'])/f'{args.expected}_{job}'
    fingerprint = dict(frozen_sha256=frozen_sha, expected=args.expected,
        summarize_only=args.summarize_only,
        run_markers=dict(zip(summary.run_id, summary.completion_sha256)))
    with output_transaction(output, fingerprint) as work:
        if work is None:
            return
        summary.to_csv(work/'absolute_results.csv', index=False)
        summary.groupby(['fold_id', 'group'])[['RMSE', 'MAE']].agg(['mean', 'std']).to_csv(work/'absolute_seed_summary.csv')
        if not args.summarize_only:
            analyze(summary, request, work)
        (work/'scope.json').write_text(json.dumps(dict(status='completed', fits=len(summary),
            frozen_sha256=frozen_sha, expected=args.expected, analysis=not args.summarize_only), indent=2))
    print(f'[Confirmation audit/analysis] {output}')


if __name__ == '__main__':
    main()
