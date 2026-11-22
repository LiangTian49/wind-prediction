"""Post hoc circular date-block bootstrap of existing paired error tables."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--block_days', type=int, default=3)
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261106)
    args = parser.parse_args()
    source = (ROOT / args.analysis_dir).resolve()
    output = (ROOT / args.output_dir).resolve()
    if not source.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        raise ValueError('Paths must remain inside project.')
    if args.block_days < 1 or args.resamples < 100:
        raise ValueError('Require positive block length and >=100 resamples.')
    output.mkdir(parents=True, exist_ok=True)
    time = pd.read_csv(source / 'per_time_metrics.csv')
    if not time.grids.eq(98).all():
        raise ValueError('Expected the same 98 grids at each timestamp.')
    time['date'] = pd.to_datetime(time.target_time).dt.normalize()
    keys = ['fold_id', 'seed', 'lead_hours', 'target_time']
    if time.duplicated(keys + ['experiment']).any():
        raise ValueError('Duplicate paired keys.')
    baseline = time[time.experiment.eq('T01')][keys + ['RMSE', 'MAE']]
    time = time.merge(baseline, on=keys, suffixes=('', '_T01'), validate='many_to_one')
    rng = np.random.default_rng(args.seed)
    results = []
    for (fold, experiment, lead), block in time.groupby(['fold_id', 'experiment', 'lead_hours']):
        if sorted(block.seed.unique()) != [42, 43, 44, 45, 46]:
            raise ValueError('Require all five seeds.')
        dates = sorted(block.date.unique())
        n_dates = len(dates)
        if args.block_days > n_dates:
            raise ValueError('Block exceeds available dates.')
        # Resample dates together across grids and all five training seeds.
        # Consecutive days within circular blocks preserve some serial dependence.
        starts = rng.integers(0, n_dates, size=(args.resamples, int(np.ceil(n_dates / args.block_days))))
        sampled = ((starts[:, :, None] + np.arange(args.block_days)) % n_dates).reshape(args.resamples, -1)[:, :n_dates]
        weights = np.stack([(sampled == day).sum(axis=1) for day in range(n_dates)], axis=1)
        daily_count = block[block.seed.eq(42)].groupby('date').size().reindex(dates).to_numpy()
        denominator = weights @ daily_count
        for reference in ['T01', 'Persistence']:
            for metric in ['RMSE', 'MAE']:
                model_col = block[metric] ** 2 if metric == 'RMSE' else block[metric]
                ref_name = f'{metric}_T01' if reference == 'T01' else f'persistence_{metric}'
                ref_col = block[ref_name] ** 2 if metric == 'RMSE' else block[ref_name]
                work = block.assign(model_error=model_col, reference_error=ref_col)
                gains, originals = [], []
                for _, seed_data in work.groupby('seed'):
                    daily = seed_data.groupby('date')[['model_error', 'reference_error']].sum().reindex(dates)
                    if daily.isna().any().any():
                        raise ValueError('Mismatched seed/date coverage.')
                    model_error = (weights @ daily.model_error.to_numpy()) / denominator
                    reference_error = (weights @ daily.reference_error.to_numpy()) / denominator
                    ratio = model_error / reference_error
                    original_ratio = daily.model_error.sum() / daily.reference_error.sum()
                    if metric == 'RMSE':
                        ratio, original_ratio = np.sqrt(ratio), np.sqrt(original_ratio)
                    gains.append(100 * (1 - ratio))
                    originals.append(100 * (1 - original_ratio))
                bootstrap = np.mean(gains, axis=0)
                low, high = np.quantile(bootstrap, [0.025, 0.975])
                results.append(dict(fold_id=fold, experiment=experiment, lead_hours=lead,
                                    reference=reference, metric=metric, gain_mean=np.mean(originals),
                                    seed_std=np.std(originals, ddof=1), ci_low=low, ci_high=high,
                                    dates=n_dates, block_days=args.block_days, resamples=args.resamples))
    frame = pd.DataFrame(results)
    frame.to_csv(output / 'date_block_intervals.csv', index=False)
    (output / 'README.md').write_text(
        '# Post hoc date-block uncertainty analysis\n\n'
        'Uses completed 98-grid error tables, with no model fitting or new predictions. '
        'Circular moving blocks resample consecutive target dates jointly across grids and seeds. '
        'Each replicate recomputes pooled errors per seed and averages the five paired percentage gains. '
        'Partial boundary dates retain their timestamp weights.\n\n'
        'Percentile 95% intervals describe within-window date variation conditional on fitted models; '
        'they do not measure cross-year or cross-region generalization. Training-seed SD is separate. '
        'Three-day blocks are a post hoc diagnostic choice, not a preregistered significance test. '
        'Many comparisons are reported without multiplicity correction: do not use intervals to select '
        'a winning configuration or claim confirmatory statistical significance. '
        'The winter fold has already been evaluated and must not be reused for tuning.\n'
    )
    print(f'Wrote {len(frame)} interval rows to {output}')


if __name__ == '__main__':
    main()
