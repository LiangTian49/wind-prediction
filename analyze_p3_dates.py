"""Paired, window-conditional date intervals for the frozen 36-run P3 development set."""
import json
import os

import numpy as np
import pandas as pd

from analyze_p1_dates import KEYS, daily_errors
from direction_pipeline import code_hashes, output_transaction, sha256, verify_artifacts
from summarize_p3 import COMPARISONS
from temporal_folds import project_path
from train_p3 import CONFIG, OUTPUT


def gain_distribution(errors, reference, column, metric, weights):
    np.testing.assert_array_equal(errors['count'], reference['count'])
    ratio = (weights @ errors[column].to_numpy()) / (weights @ reference[column].to_numpy())
    point = errors[column].sum() / reference[column].sum()
    if metric == 'RMSE':
        ratio, point = np.sqrt(ratio), np.sqrt(point)
    return 100*(1-point), 100*(1-ratio)


def main():
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        raise SystemExit('Submit slurm/run_p3_analysis.sh.')
    root = project_path(OUTPUT)
    summary = pd.read_csv(root/'summary.csv')
    if len(summary) != 36 or set(summary.task) != set(range(36)):
        raise ValueError('Require exactly the frozen initial 36 configurations.')
    fingerprint = dict(config_sha256=sha256(project_path(CONFIG)),
                       summary_sha256=sha256(root/'summary.csv'),
                       run_markers={r:sha256(root/r/'completion.json') for r in summary.run_id},
                       code=code_hashes(['analyze_p3_dates.py', 'analyze_p1_dates.py',
                                         'analyze_experiments.py', 'direction_pipeline.py']))
    final = project_path(f'result/rs_hpc_v3/P3/analysis_{job}')
    with output_transaction(final, fingerprint) as work:
        if work is None:
            return
        rng = np.random.default_rng(20261115)
        intervals, daily_rows = [], []
        pairs = COMPARISONS + [(g, 'Persistence') for g in ['RS_T0', 'RS_T1', 'RS_S0', 'RS_S1']]
        for fold, block in summary.groupby('fold_id'):
            daily, baseline = {}, None
            for row in block.itertuples():
                verify_artifacts(root/row.run_id)
                raw = pd.read_csv(root/row.run_id/'predictions.csv',
                                  parse_dates=['forecast_origin', 'target_time', 'input_start'])
                frame, errors = daily_errors(raw)
                frame = frame.sort_values(KEYS).reset_index(drop=True)
                if baseline is None:
                    baseline = frame
                pd.testing.assert_frame_equal(baseline[KEYS+['input_start']], frame[KEYS+['input_start']])
                np.testing.assert_allclose(baseline[['y_true', 'persistence_pred']],
                                           frame[['y_true', 'persistence_pred']], atol=1e-6, rtol=0)
                for metric, column in [('RMSE', 'squared'), ('MAE', 'absolute')]:
                    value = errors[column].sum()/errors['count'].sum()
                    if metric == 'RMSE':
                        value = np.sqrt(value)
                    if not np.isclose(value, getattr(row, metric), atol=1e-8, rtol=0):
                        raise ValueError('Archived metric reconstruction mismatch.')
                daily[row.seed, row.group] = errors
            days = next(iter(daily.values())).index
            if any(not d.index.equals(days) for d in daily.values()):
                raise ValueError('Unmatched date coverage.')
            starts = rng.integers(0, len(days), size=(2000, int(np.ceil(len(days)/3))))
            indices = ((starts[:, :, None]+np.arange(3)) % len(days)).reshape(2000, -1)[:, :len(days)]
            weights = np.stack([(indices == i).sum(axis=1) for i in range(len(days))], axis=1)
            for model, reference in pairs:
                for metric, column in [('RMSE', 'squared'), ('MAE', 'absolute')]:
                    points, replicates = [], []
                    for seed in sorted(block.seed.unique()):
                        errors = daily[seed, model]
                        ref = daily[seed, reference] if reference != 'Persistence' else pd.DataFrame(
                            dict(squared=errors.persistence_squared, absolute=errors.persistence_absolute,
                                 count=errors['count']))
                        point, replicate = gain_distribution(errors, ref, column, metric, weights)
                        points.append(point)
                        replicates.append(replicate)
                        low, high = np.quantile(replicate, [.025, .975])
                        intervals.append(dict(fold_id=fold, model=model, reference=reference, metric=metric,
                                              seed=str(seed), gain_pct=point, date_p025_pct=low, date_p975_pct=high))
                        for date in days:
                            daily_rows.append(dict(fold_id=fold, seed=int(seed), model=model, reference=reference,
                                metric=metric, date=str(date.date()), count=int(errors.loc[date, 'count']),
                                model_loss=float(errors.loc[date, column]), reference_loss=float(ref.loc[date, column]),
                                loss_reduction=float(ref.loc[date, column]-errors.loc[date, column])))
                    low, high = np.quantile(np.mean(replicates, axis=0), [.025, .975])
                    intervals.append(dict(fold_id=fold, model=model, reference=reference, metric=metric,
                        seed='fixed_seed_mean', gain_pct=float(np.mean(points)), date_p025_pct=low, date_p975_pct=high))
        pd.DataFrame(intervals).to_csv(work/'date_intervals.csv', index=False)
        pd.DataFrame(daily_rows).to_csv(work/'daily_loss_contributions.csv', index=False)
        summary.to_csv(work/'absolute_results.csv', index=False)
        (work/'method.json').write_text(json.dumps(dict(configurations=36, interval_rows=len(intervals),
            block_days=3, resamples=2000, rng_seed=20261115,
            date_assignment='target date', scope='post hoc window-conditional, no multiplicity correction',
            seed_treatment='Separate seed results and fixed-three-seed mean paired gains; shared date draws. '
                           'No prediction ensemble, no seed resampling; date intervals exclude seed uncertainty.',
            daily_loss='Squared error for RMSE, absolute error for MAE. Positive reduction favors model.'), indent=2))
    print(f'[P3 dates] Completed: {final}')


if __name__ == '__main__':
    main()
