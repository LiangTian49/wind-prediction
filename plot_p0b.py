"""Batch review of archived grid errors and fixed, complete regional cases."""

import json
import os
from time import perf_counter
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from analyze_experiments import attach_centers, collapse_grids, score
from direction_pipeline import code_hashes, output_transaction, sha256
from temporal_folds import load_manifest, project_path
from train import build_model

LEGACY = 'result/spatial_analysis/analysis_912234'
P1 = 'runs/rs_hpc_v3/P1'


def great_circle(lat1, lon1, lat2, lon2, radius=6371.229):
    a, b = np.deg2rad([lat1, lat2])
    delta = np.deg2rad(lon2-lon1)
    term = np.sin((b-a)/2)**2 + np.cos(a)*np.cos(b)*np.sin(delta/2)**2
    return float(2*radius*np.arcsin(np.sqrt(np.clip(term, 0, 1))))


def aligned_case(frames, hours=72):
    keys = ['center_y', 'center_x', 'forecast_origin', 'target_time']
    ordered = {}
    for name, frame in frames.items():
        data = frame.sort_values(keys).reset_index(drop=True)
        if data.duplicated(keys).any():
            raise ValueError('Duplicate case keys.')
        ordered[name] = data
    first = next(iter(ordered.values()))
    for data in ordered.values():
        pd.testing.assert_frame_equal(first[keys], data[keys])
        np.testing.assert_allclose(first.y_true, data.y_true, atol=1e-6, rtol=0)
        np.testing.assert_allclose(first.persistence_pred, data.persistence_pred, atol=1e-6, rtol=0)
    expected_grids = first[['center_y', 'center_x']].drop_duplicates().shape[0]
    origins = pd.DatetimeIndex(first.forecast_origin.unique()).sort_values()
    for start in origins:
        wanted = pd.date_range(start, periods=hours//3, freq='3h')
        part = first[first.forecast_origin.isin(wanted)]
        counts = part.groupby('forecast_origin').size().reindex(wanted, fill_value=0)
        if counts.eq(expected_grids).all():
            return {name: frame[frame.forecast_origin.isin(wanted)].copy()
                    for name, frame in ordered.items()}, start
    raise ValueError('No complete common 72-hour case.')


def map_and_series(frames, fold, output, grid_list):
    case, start = aligned_case(frames)
    names = list(case)
    first = case[names[0]]
    instant = first[first.forecast_origin.eq(start)].sort_values(['center_y', 'center_x'])
    value_max = max(float(frame[frame.forecast_origin.eq(start)].y_pred.max()) for frame in case.values())
    value_max = max(value_max, float(instant.y_true.max()), float(instant.persistence_pred.max()))
    error_max = max(float(np.abs(frame[frame.forecast_origin.eq(start)].y_pred -
                                  frame[frame.forecast_origin.eq(start)].y_true).max()) for frame in case.values())
    error_max = max(error_max, float(np.abs(instant.persistence_pred-instant.y_true).max()), .01)
    fig, axes = plt.subplots(2, len(names)+1, figsize=(3.3*(len(names)+1), 6.8), layout='constrained')
    axes[0, 0].scatter(instant.grid_lon, instant.grid_lat, c=instant.y_true,
                       vmin=0, vmax=value_max, cmap='viridis', s=18)
    axes[0, 0].set_title('Reanalysis reference (m/s)')
    axes[1, 0].scatter(instant.grid_lon, instant.grid_lat, c=instant.persistence_pred-instant.y_true,
                       vmin=-error_max, vmax=error_max, cmap='RdBu_r', s=18)
    axes[1, 0].set_title('Persistence bias (m/s)')
    for column, name in enumerate(names, 1):
        frame = case[name]
        snapshot = frame[frame.forecast_origin.eq(start)]
        values = axes[0, column].scatter(snapshot.grid_lon, snapshot.grid_lat, c=snapshot.y_pred,
                                         vmin=0, vmax=value_max, cmap='viridis', s=18)
        errors = axes[1, column].scatter(snapshot.grid_lon, snapshot.grid_lat,
                                         c=snapshot.y_pred-snapshot.y_true,
                                         vmin=-error_max, vmax=error_max, cmap='RdBu_r', s=18)
        axes[0, column].set_title(f'{name} prediction')
        axes[1, column].set_title(f'{name} bias')
    for axis in axes.flat:
        axis.set(xlabel='Longitude (degrees E)', ylabel='Latitude (degrees N)',
                 xlim=(8.35, 11.65), ylim=(49.4, 51.6))
        axis.set_aspect(1/np.cos(np.deg2rad(50.5)))
        axis.grid(alpha=.2)
    fig.colorbar(values, ax=axes[0, :].tolist(), label='Wind speed (m/s)')
    fig.colorbar(errors, ax=axes[1, :].tolist(), label='Prediction minus reference (m/s)')
    fig.suptitle(f'{fold}, seed 42, lead 3h, origin {start}, target {start+pd.Timedelta(hours=3)}')
    for suffix in ['png', 'svg']:
        fig.savefig(output / f'{fold}_maps.{suffix}', dpi=180)
    plt.close(fig)
    grid_list = grid_list.sort_values(['grid_lon', 'grid_lat', 'center_y', 'center_x']).reset_index(drop=True)
    chosen = grid_list.iloc[[0, len(grid_list)//2, len(grid_list)-1]]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), layout='constrained')
    locations = []
    for axis, row in zip(axes, chosen.itertuples()):
        locations.append(dict(grid_id=f'{row.center_y}:{row.center_x}', lat=row.grid_lat, lon=row.grid_lon))
        reference = first[(first.center_y == row.center_y) & (first.center_x == row.center_x)].sort_values('target_time')
        axis.plot(reference.target_time, reference.y_true, color='black', label='Reanalysis reference')
        axis.plot(reference.target_time, reference.persistence_pred, '--', color='0.5', label='Persistence')
        for name, frame in case.items():
            local = frame[(frame.center_y == row.center_y) & (frame.center_x == row.center_x)].sort_values('target_time')
            axis.plot(local.target_time, local.y_pred, label=name, alpha=.8)
        axis.set(ylabel='Wind speed (m/s)', title=f'Native grid {row.center_y}:{row.center_x}')
        axis.grid(alpha=.2)
    axes[0].legend(ncol=3, fontsize=8)
    fig.suptitle(f'{fold}, lead 3h, seed 42; first complete 72h origin interval')
    for suffix in ['png', 'svg']:
        fig.savefig(output / f'{fold}_series.{suffix}', dpi=180)
    plt.close(fig)
    return dict(fold_id=fold, seed=42, lead_hours=3, origin_start=str(start),
                origin_end_exclusive=str(start+pd.Timedelta(hours=72)), models=names, locations=locations,
                selection='Earliest common complete 72h; longitude-sorted first/middle/last target cells')


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Submit run_p0b.sh.')
    started = perf_counter()
    sources = json.loads(project_path(f'{LEGACY}/source_runs.json').read_text())
    _, folds, _ = load_manifest('configs/seasonal_folds_2024.json')
    centers_path = project_path('data/CERRA/spatial_2024_full/centers.csv')
    centers = pd.read_csv(centers_path)
    centers['grid_id'] = centers.center_y.astype(str) + ':' + centers.center_x.astype(str)
    grids = centers.drop_duplicates('grid_id')
    p1_paths = sorted(project_path(P1).glob('*/result.json'))
    records = []
    for source in sources:
        records.append(dict(fold=source['fold_id'], group=source['experiment'], family='lstm',
                            seed=source['seed'], lead=source['pred_len']*3, run_id=source['run_id'],
                            expected_rows=source['test_samples'],
                            path=project_path(source['run_dir']) / 'predictions.csv'))
    for path in p1_paths:
        result = json.loads(path.read_text())
        if any(sha256(path.parent / name) != expected
               for name, expected in result['artifact_hashes'].items()):
            raise ValueError(f'Changed P1 artifact: {path}')
        records.append(dict(fold=result['fold_id'], group=result['experiment'], family=result['model_family'],
                            seed=result['seed'], lead=result['lead_hours'], run_id=result['run_id'],
                            expected_rows=result['counts'][2],
                            path=path.parent / 'predictions.csv'))
    if len(sources) != 320 or len(p1_paths) != 48:
        raise ValueError('Require complete archived LSTM and P1 runs.')
    fingerprint = dict(code=code_hashes(['plot_p0b.py', 'direction_pipeline.py', 'train.py', 'analyze_experiments.py']),
                       centers=sha256(centers_path),
                       predictions={record['run_id']: sha256(record['path']) for record in records},
                       p1_summary=sha256(project_path(f'{P1}/summary.csv')),
                       fold_manifest=sha256(project_path('configs/seasonal_folds_2024.json')))
    final = project_path('result/rs_hpc_v3/P0b/v1')
    with output_transaction(final, fingerprint) as work:
        if work is None:
            print(f'[P0b] Verified existing report: {final}')
            return
        centers.to_csv(work / 'point_grid_mapping.csv', index=False)
        grids.to_csv(work / 'unique_grids.csv', index=False)
        pd.read_csv(project_path('data/CERRA/spatial_2024_full/grid_mapping.csv')).to_csv(work / 'neighborhoods.csv', index=False)
        metrics, pooled, spreads = [], [], []
        cases = {fold: {} for fold in folds}
        p1_summary = pd.read_csv(project_path(f'{P1}/summary.csv'))
        for record in records:
            frame = pd.read_csv(record['path'], parse_dates=['input_start', 'forecast_origin', 'target_time'])
            keys = ['point_id', 'forecast_origin', 'target_time']
            if (len(frame) != record['expected_rows'] or frame.duplicated(keys).any()
                    or not frame.run_id.eq(record['run_id']).all()
                    or not frame.seed.eq(record['seed']).all()
                    or not frame.fold_id.eq(record['fold']).all()):
                raise ValueError(f'Prediction identity or sample coverage differs: {record["run_id"]}')
            if not frame.target_time.sub(frame.forecast_origin).eq(pd.Timedelta(hours=record['lead'])).all():
                raise ValueError('Prediction horizon mismatch.')
            if 'center_y' not in frame:
                frame = attach_centers(frame, centers.drop(columns='grid_id'))
            group_keys = ['center_y', 'center_x', 'forecast_origin', 'target_time']
            spread = frame.groupby(group_keys).y_pred.agg(['min', 'max'])
            max_spread = float((spread['max']-spread['min']).max())
            if max_spread > 1e-4:
                raise ValueError(f'Duplicate-center prediction spread requires investigation: {record["run_id"]}')
            spreads.append(dict(run_id=record['run_id'], max_duplicate_prediction_spread=max_spread))
            unique = collapse_grids(frame)
            identity = {k: record[k] for k in ['fold', 'group', 'family', 'seed', 'lead', 'run_id']}
            pooled.append(dict(identity, metric_scope='pooled_unique_grid',
                               **score(unique.y_true, unique.y_pred, unique.persistence_pred)))
            for (yy, xx), local in unique.groupby(['center_y', 'center_x']):
                metrics.append(dict(identity, grid_id=f'{yy}:{xx}', grid_lat=local.grid_lat.iloc[0],
                                    grid_lon=local.grid_lon.iloc[0], samples=len(local),
                                    Bias=float((local.y_pred-local.y_true).mean()),
                                    **score(local.y_true, local.y_pred, local.persistence_pred)))
            if record['seed'] == 42 and record['lead'] == 3:
                if record['family'] == 'lstm' and record['group'] in ['T01', 'A01']:
                    cases[record['fold']][f'LSTM {record["group"]}'] = unique
                if record['family'] != 'lstm':
                    selected = p1_summary[(p1_summary.fold_id == record['fold']) & (p1_summary.lead_hours == 3)]
                    if record['run_id'] == selected.run_id_strong_center.iloc[0]:
                        cases[record['fold']]['Validation-selected P1 center'] = unique
            print(f'[P0b] {record["run_id"]}', flush=True)
        table = pd.DataFrame(metrics)
        match = ['fold', 'family', 'seed', 'lead', 'grid_id']
        baseline = table[table.group.eq('T01')][match + ['RMSE', 'MAE']]
        skill = table.merge(baseline, on=match, validate='many_to_one', suffixes=('', '_T01'))
        for metric in ['RMSE', 'MAE']:
            skill[f'{metric}_gain_vs_T01_pct'] = 100*(1-skill[metric]/skill[f'{metric}_T01'])
            skill[f'{metric}_gain_vs_Persistence_pct'] = 100*(1-skill[metric]/skill[f'persistence_{metric}'])
        table.to_csv(work / 'grid_metrics.csv', index=False)
        skill.to_csv(work / 'grid_skill.csv', index=False)
        pd.DataFrame(pooled).to_csv(work / 'pooled_metrics.csv', index=False)
        pd.DataFrame(spreads).to_csv(work / 'duplicate_grid_audit.csv', index=False)
        stats = skill.groupby(['fold', 'family', 'group', 'seed', 'lead']).agg(
            RMSE_improved_fraction=('RMSE_gain_vs_T01_pct', lambda x: (x>0).mean()),
            MAE_improved_fraction=('MAE_gain_vs_T01_pct', lambda x: (x>0).mean()),
            grid_RMSE_arithmetic_mean=('RMSE', 'mean'),
            RMSE_gain_p10=('RMSE_gain_vs_T01_pct', lambda x: x.quantile(.1)),
            RMSE_gain_median=('RMSE_gain_vs_T01_pct', 'median'),
            RMSE_gain_p90=('RMSE_gain_vs_T01_pct', lambda x: x.quantile(.9)))
        stats.to_csv(work / 'grid_coverage.csv')
        figure_manifest = []
        for fold, frame_set in cases.items():
            figure_manifest.append(map_and_series(frame_set, fold, work, grids))
        # Exact layer structure is taken from the production builder.
        architectures = {}
        for name, channels in [('T01', 3), ('S01', 11), ('S02', 27), ('A01', 11)]:
            model = build_model((56, channels), SimpleNamespace(model_type='lstm', dropout=.4, lr=.001))
            architectures[name] = dict(input_shape=[None, 56, channels], output_shape=list(model.output_shape),
                                       parameter_count=model.count_params(),
                                       layers=[dict(name=layer.__class__.__name__, config=layer.get_config()) for layer in model.layers])
        (work / 'architectures.json').write_text(json.dumps(architectures, indent=2))
        fig, axis = plt.subplots(figsize=(11, 2.4), layout='constrained')
        axis.axis('off')
        for i, label in enumerate(['(B,56,F)', 'LSTM(64)', 'Dropout(0.4)', 'Dense(1)', '(B,1)']):
            x=i*.2
            axis.add_patch(Rectangle((x,.35),.16,.3,fill=False))
            axis.text(x+.08,.5,label,ha='center',va='center')
            if i<4:
                axis.annotate('',xy=(x+.2,.5),xytext=(x+.16,.5),arrowprops=dict(arrowstyle='->'))
        axis.set(xlim=(-.02,1),ylim=(0,1),title='Shared sequence-to-one model; one direct model per lead')
        fig.savefig(work / 'lstm_architecture.svg')
        plt.close(fig)
        (work / 'figure_manifest.json').write_text(json.dumps(figure_manifest, indent=2))
        dimensions = dict(
            nominal_roi_south_width_km=great_circle(49.5,8.5,49.5,11.5),
            nominal_roi_north_width_km=great_circle(51.5,8.5,51.5,11.5),
            nominal_roi_height_km=great_circle(49.5,8.5,51.5,8.5),
            unique_grids=len(grids), point_ids=len(centers),
            distance_method='Great-circle on GRIB sphere radius 6371229 m',
            coverage='Selected target grid points; spatially correlated known locations')
        (work / 'roi_dimensions.json').write_text(json.dumps(dimensions, indent=2))
        (work / 'roi_prediction_audit.md').write_text(
            '# Regional prediction audit\n\n'
            'The archived model uses shared weights and scalar output per target window. '
            'Evaluation deduplicates mapped centers within each seed and forecast time. '
            'The 98 target cells are shown at real coordinates with blank unpredicted locations. '
            'Cases use seed 42 and the earliest common complete 72-hour origin interval in each fold. '
            'The plotted strong P1 center is chosen by validation RMSE. '
            'Bias is prediction minus reanalysis reference. Maps share value and error scales. '
            'Grid RMSE arithmetic means and pooled RMSE are saved under distinct names. '
            'No cross-seed prediction ensemble is used.\n')
        (work / 'summary.json').write_text(json.dumps(dict(status='completed', archived_runs=len(records),
                   grid_metric_rows=len(table), cases=4, elapsed_seconds=perf_counter()-started), indent=2))
    print(f'[P0b] Completed: {final}')


if __name__ == '__main__':
    main()
