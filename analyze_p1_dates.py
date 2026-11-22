"""Paired date-block analysis of completed P1 baselines."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_experiments import collapse_grids
from temporal_folds import project_path
from train_p1_baseline import OUTPUT, digest

KEYS = ['center_y', 'center_x', 'forecast_origin', 'target_time']


def daily_errors(frame):
    data = collapse_grids(frame)
    data['date'] = data.target_time.dt.normalize()
    error = data.y_pred - data.y_true
    persistence_error = data.persistence_pred - data.y_true
    data['squared'] = error**2
    data['absolute'] = error.abs()
    data['persistence_squared'] = persistence_error**2
    data['persistence_absolute'] = persistence_error.abs()
    grouped = data.groupby('date')[['squared', 'absolute', 'persistence_squared', 'persistence_absolute']].sum()
    grouped['count'] = data.groupby('date').size()
    return data, grouped


def main():
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        raise SystemExit('Submit run_p1_analysis.sh.')
    root = project_path(OUTPUT)
    summary = pd.read_csv(root / 'summary.csv')
    if len(summary) != 48 or summary.isna().any().any():
        raise ValueError('Require a complete P1 summary.')
    output = project_path(f'result/rs_hpc_v3/P1/analysis_{job}')
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261111)
    results, manifests = [], []
    for (fold, lead), block in summary.groupby(['fold_id', 'lead_hours']):
        frames, daily = {}, {}
        for run_id in block.run_id:
            run = root / run_id
            saved = json.loads((run / 'result.json').read_text())
            if any(digest(run / name) != sha for name, sha in saved['artifact_hashes'].items()):
                raise ValueError(f'Damaged artifacts: {run}')
            frame = pd.read_csv(run / 'predictions.csv',
                                parse_dates=['forecast_origin', 'target_time', 'input_start'])
            frames[run_id], daily[run_id] = daily_errors(frame)
            manifests.append(dict(run_id=run_id, predictions_sha256=saved['artifact_hashes']['predictions.csv']))
        first = frames[block.run_id.iloc[0]].sort_values(KEYS).reset_index(drop=True)
        for frame in frames.values():
            matched = frame.sort_values(KEYS).reset_index(drop=True)
            pd.testing.assert_frame_equal(first[KEYS], matched[KEYS])
            np.testing.assert_allclose(first[['y_true', 'persistence_pred']],
                                       matched[['y_true', 'persistence_pred']], atol=1e-6, rtol=0)
        days = daily[block.run_id.iloc[0]].index
        n_days = len(days)
        starts = rng.integers(0, n_days, size=(2000, int(np.ceil(n_days / 3))))
        indices = ((starts[:, :, None] + np.arange(3)) % n_days).reshape(2000, -1)[:, :n_days]
        weights = np.stack([(indices == day).sum(axis=1) for day in range(n_days)], axis=1)
        for row in block.itertuples():
            errors = daily[row.run_id].reindex(days)
            for reference in ['same_family_center', 'strong_center', 'Persistence']:
                if reference == 'Persistence':
                    ref = errors.rename(columns={'squared': 'unused_squared', 'absolute': 'unused_absolute',
                                                 'persistence_squared': 'squared', 'persistence_absolute': 'absolute'})
                    reference_id = 'Persistence'
                else:
                    reference_id = (block[(block.model_family == row.model_family)
                                          & block.experiment.eq('T01')].run_id.iloc[0]
                                    if reference == 'same_family_center' else row.run_id_strong_center)
                    ref = daily[reference_id].reindex(days)
                np.testing.assert_array_equal(errors['count'], ref['count'])
                for metric, column in [('RMSE', 'squared'), ('MAE', 'absolute')]:
                    # Matching counts cancel in each paired error ratio.
                    ratio = (weights @ errors[column].to_numpy()) / (weights @ ref[column].to_numpy())
                    point_ratio = errors[column].sum() / ref[column].sum()
                    if metric == 'RMSE':
                        ratio, point_ratio = np.sqrt(ratio), np.sqrt(point_ratio)
                    gains = 100 * (1 - ratio)
                    point = 100 * (1 - point_ratio)
                    suffix = 'persistence' if reference == 'Persistence' else reference
                    expected = getattr(row, f'{metric}_gain_vs_{suffix}_pct')
                    if not np.isclose(point, expected, atol=1e-8):
                        raise ValueError('Reconstructed gain differs from completed P1 summary.')
                    low, high = np.quantile(gains, [.025, .975])
                    results.append(dict(fold_id=fold, lead_hours=lead, model_family=row.model_family,
                                        experiment=row.experiment, reference=reference,
                                        reference_id=reference_id, metric=metric, gain_pct=point,
                                        date_p025_pct=low, date_p975_pct=high, dates=n_days,
                                        seed=42, block_days=3, resamples=2000))
    result = pd.DataFrame(results)
    if len(result) != 288 or result.isna().any().any():
        raise ValueError('Incomplete date analysis.')
    result.to_csv(output / 'date_intervals.csv', index=False)
    (output / 'sources.json').write_text(json.dumps(manifests, indent=2))
    (output / 'method.json').write_text(json.dumps(dict(
        status='completed', configurations=48, interval_rows=288,
        rng_seed=20261111, block_days=3, resamples=2000,
        method='Circular date blocks shared across all four P1 models and both metrics within each fold/lead.',
        interpretation='Post hoc window-conditional intervals. One fitted seed per baseline. No multiplicity adjustment.',
        summary_sha256=digest(root / 'summary.csv'), code_sha256=digest(Path(__file__).resolve())
    ), indent=2))
    print(f'[P1 dates] Completed: {output}')


if __name__ == '__main__':
    main()
