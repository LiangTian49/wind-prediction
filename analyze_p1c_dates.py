"""Date-paired descriptive analysis of all twelve direction ablations."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_p1_dates import KEYS, daily_errors
from direction_pipeline import code_hashes, output_transaction, sha256, verify_artifacts
from temporal_folds import project_path
from train_p1c import CONFIG, OUTPUT

PAIRS = [('D_T1', 'D_T0'), ('D_S1', 'D_S0'), ('D_S1', 'D_T1')]


def paired_gain(errors, reference, column, metric, weights):
    """Recompute pooled RMSE/MAE for every paired date-block sample."""
    np.testing.assert_array_equal(errors['count'], reference['count'])
    ratio = (weights @ errors[column].to_numpy()) / (weights @ reference[column].to_numpy())
    point_ratio = errors[column].sum() / reference[column].sum()
    if metric == 'RMSE':
        ratio, point_ratio = np.sqrt(ratio), np.sqrt(point_ratio)
    gain = 100*(1-ratio)
    low, high = np.quantile(gain, [.025, .975])
    return float(100*(1-point_ratio)), float(low), float(high)


def main():
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        raise SystemExit('Submit run_p1c_analysis.sh.')
    root = project_path(OUTPUT)
    summary = pd.read_csv(root / 'summary.csv')
    selection = json.loads((root / 'dynamic_selection.json').read_text())
    if len(summary) != 12 or summary.isna().any().any():
        raise ValueError('Require twelve complete direction configurations.')
    manifests = {row.run_id: verify_artifacts(root / row.run_id) for row in summary.itertuples()}
    fingerprint = dict(config_sha256=sha256(project_path(CONFIG)),
                       summary_sha256=sha256(root / 'summary.csv'),
                       validation_selection_sha256=sha256(root / 'dynamic_selection.json'),
                       code=code_hashes(['analyze_p1c_dates.py', 'analyze_p1_dates.py',
                                         'analyze_experiments.py', 'direction_pipeline.py']),
                       run_markers={name: sha256(root / name / 'completion.json') for name in manifests})
    final = project_path(f'result/rs_hpc_v3/P1c/analysis_{job}')
    with output_transaction(final, fingerprint) as work:
        if work is None:
            print(f'[P1c dates] Verified existing analysis: {final}')
            return
        rng = np.random.default_rng(20261113)
        intervals = []
        for fold, block in summary.groupby('fold_id'):
            frames, daily = {}, {}
            for row in block.itertuples():
                frame = pd.read_csv(root / row.run_id / 'predictions.csv',
                                    parse_dates=['input_start', 'forecast_origin', 'target_time'])
                frames[row.group], daily[row.group] = daily_errors(frame)
                metrics = {'RMSE': np.sqrt(daily[row.group].squared.sum()/daily[row.group]['count'].sum()),
                           'MAE': daily[row.group].absolute.sum()/daily[row.group]['count'].sum()}
                for metric, value in metrics.items():
                    if not np.isclose(value, getattr(row, metric), atol=1e-8, rtol=0):
                        raise ValueError('Stored metrics differ from prediction reconstruction.')
            first = frames['D_T0'].sort_values(KEYS).reset_index(drop=True)
            for frame in frames.values():
                current = frame.sort_values(KEYS).reset_index(drop=True)
                pd.testing.assert_frame_equal(first[KEYS], current[KEYS])
                np.testing.assert_allclose(first[['y_true', 'persistence_pred']],
                                           current[['y_true', 'persistence_pred']], atol=1e-6, rtol=0)
            days = daily['D_T0'].index
            starts = rng.integers(0, len(days), size=(2000, int(np.ceil(len(days)/3))))
            indices = ((starts[:, :, None]+np.arange(3)) % len(days)).reshape(2000, -1)[:, :len(days)]
            weights = np.stack([(indices == i).sum(axis=1) for i in range(len(days))], axis=1)
            comparisons = PAIRS + [(group, 'Persistence') for group in frames]
            for model, reference in comparisons:
                errors = daily[model].reindex(days)
                if reference == 'Persistence':
                    ref = pd.DataFrame(dict(squared=errors.persistence_squared,
                                             absolute=errors.persistence_absolute, count=errors['count']))
                else:
                    ref = daily[reference].reindex(days)
                for metric, column in [('RMSE', 'squared'), ('MAE', 'absolute')]:
                    point, low, high = paired_gain(errors, ref, column, metric, weights)
                    intervals.append(dict(fold_id=fold, model=model, reference=reference, metric=metric,
                                          gain_pct=point, date_p025_pct=low, date_p975_pct=high,
                                          seed=42, lead_hours=3, block_days=3, resamples=2000, dates=len(days)))
        table = pd.DataFrame(intervals)
        if len(table) != 42 or table.isna().any().any():
            raise ValueError('Incomplete date analysis.')
        table.to_csv(work / 'date_intervals.csv', index=False)
        summary.to_csv(work / 'absolute_results.csv', index=False)
        (work / 'validation_selection.json').write_text(json.dumps(selection, indent=2))
        lines = ['# P1c paired date analysis', '',
                 'Three-day circular blocks, 2000 replicates, RNG seed 20261113. '
                 'All models and both metrics share the same date indices within each fold. '
                 'These are post hoc window-conditional intervals from one fitted seed, '
                 'without multiplicity correction.', '',
                 '| Fold | Model | Reference | Metric | Gain % | 95% date interval |',
                 '|---|---|---|---|---|---|']
        for row in table.itertuples():
            lines.append(f'| {row.fold_id} | {row.model} | {row.reference} | {row.metric} | '
                         f'{row.gain_pct:.2f} | [{row.date_p025_pct:.2f}, {row.date_p975_pct:.2f}] |')
        lines += ['', f'Validation selection: direction_enabled={selection["direction_enabled"]}; '
                  f'equal-fold validation gain={selection["equal_fold_mean_gain_pct"]:.4f}%. '
                  'Test scores and date intervals do not change that rule.', '']
        (work / 'REVIEW.md').write_text('\n'.join(lines))
        (work / 'method.json').write_text(json.dumps(dict(
            status='completed', configurations=12, interval_rows=42,
            block_days=3, resamples=2000, rng_seed=20261113,
            selection_uses='registered development validation rule only'), indent=2))
    print(f'[P1c dates] Completed: {final}')


if __name__ == '__main__':
    main()
