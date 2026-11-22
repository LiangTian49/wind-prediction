"""Batch-only source-level audit of archived forecasts and winter 6h diagnostics."""

import json
import os

import numpy as np
import pandas as pd

from analyze_experiments import attach_centers, collapse_grids, score
from temporal_folds import load_manifest, project_path, select_fold_panel

LEGACY = 'result/spatial_analysis/analysis_912234'
SPATIAL = 'data/CERRA/spatial_2024_full'
MANIFEST = 'configs/seasonal_folds_2024.json'


def main():
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        raise SystemExit('Submit run_p0_audit.sh via Slurm.')
    output = project_path(f'result/rs_hpc_v3/P0/audit_{job}')
    output.mkdir(parents=True, exist_ok=False)
    center = project_path(SPATIAL)
    panel = pd.read_csv(center / 'center.csv', parse_dates=['datetime'])
    lookup = panel.set_index(['point_id', 'datetime'], verify_integrity=True).ws_10
    centers = pd.read_csv(center / 'centers.csv')
    _, folds, _ = load_manifest(MANIFEST)
    sources = json.loads((project_path(LEGACY) / 'source_runs.json').read_text())
    winter = select_fold_panel(panel, folds['F04'])
    fold = folds['F04']
    start_val, start_test = pd.Timestamp(fold.validation_start), pd.Timestamp(fold.test_start)
    winter['segment'] = np.select([winter.datetime < start_val, winter.datetime < start_test],
                                  ['train', 'validation'], default='test')
    winter = attach_centers(winter, centers).drop_duplicates(['center_y', 'center_x', 'datetime'])
    distributions = []
    for segment, block in winter.groupby('segment'):
        block = block.sort_values(['center_y', 'center_x', 'datetime'])
        wind = block.ws_10
        lag = block.groupby(['center_y', 'center_x']).ws_10.diff(2).dropna()
        distributions.append(dict(segment=segment, rows=len(wind), wind_mean=wind.mean(), wind_sd=wind.std(),
                                  wind_p10=wind.quantile(.1), wind_p50=wind.median(), wind_p90=wind.quantile(.9),
                                  six_hour_change_sd=lag.std(), six_hour_persistence_RMSE=np.sqrt((lag**2).mean()),
                                  six_hour_persistence_MAE=lag.abs().mean()))
    pd.DataFrame(distributions).to_csv(output / 'winter_distribution.csv', index=False)
    thresholds = winter[winter.segment.eq('train')].ws_10.quantile([.5, .9]).to_list()
    audit, breakdown = [], []
    for source in sources:
        run = project_path(source['run_dir'])
        pred = pd.read_csv(run / 'predictions.csv', parse_dates=['input_start', 'forecast_origin', 'target_time'])
        fold = folds[source['fold_id']]
        lead = source['pred_len']
        assert len(pred) == source['test_samples']
        assert not pred.duplicated(['point_id', 'forecast_origin', 'target_time']).any()
        assert pred.forecast_origin.ge(pd.Timestamp(fold.test_start)).all()
        assert pred.input_start.ge(pd.Timestamp(fold.train_start)).all()
        assert pred.target_time.lt(pd.Timestamp(fold.test_end_exclusive)).all()
        assert pred.target_time.sub(pred.forecast_origin).eq(pd.Timedelta(hours=lead*3)).all()
        assert pred.forecast_origin.sub(pred.input_start).eq(pd.Timedelta(hours=165)).all()
        for time_col, value_col in [('forecast_origin', 'persistence_pred'), ('target_time', 'y_true')]:
            keys = pd.MultiIndex.from_arrays([pred.point_id, pred[time_col]], names=lookup.index.names)
            np.testing.assert_allclose(pred[value_col], lookup.reindex(keys), rtol=0, atol=1e-6)
        audit.append(dict(run_id=source['run_id'], fold_id=source['fold_id'], rows=len(pred), status='passed'))
        if source['fold_id'] != 'F04' or lead != 2:
            continue
        grids = collapse_grids(attach_centers(pred, centers))
        grids['origin_hour'] = grids.forecast_origin.dt.hour
        grids['origin_wind_bin'] = pd.cut(grids.persistence_pred, [-np.inf, *thresholds, np.inf], duplicates='drop')
        for key in ['origin_hour', 'origin_wind_bin']:
            for value, group in grids.groupby(key, observed=True):
                breakdown.append(dict(experiment=source['experiment'], seed=source['seed'],
                                      stratum=key, level=str(value), rows=len(group),
                                      bias=(group.y_pred-group.y_true).mean(),
                                      **score(group.y_true, group.y_pred, group.persistence_pred)))
    assert len(audit) == 320
    pd.DataFrame(audit).to_csv(output / 'prediction_audit.csv', index=False)
    pd.DataFrame(breakdown).to_csv(output / 'winter_6h_error_breakdown.csv', index=False)
    (output / 'summary.json').write_text(json.dumps(dict(status='passed', archived_predictions=320,
                                                       wind_bin_training_thresholds=thresholds), indent=2))
    print(f'[P0] PASS: {output}')


if __name__ == '__main__':
    main()
