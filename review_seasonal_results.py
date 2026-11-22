"""Review small, existing analysis tables; never load models or weather arrays."""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def project_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Path outside project: {path}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis_dir', required=True)
    args = parser.parse_args()
    directory = project_path(args.analysis_dir)
    sources = json.loads((directory / 'source_runs.json').read_text())
    coverage = json.loads((directory / 'coverage.json').read_text())
    freeze = json.loads((ROOT / 'configs/confirmation_freeze_2024.json').read_text())
    for name, expected in {**freeze['code_sha256'], **freeze['development_report_sha256']}.items():
        assert hashlib.sha256(project_path(name).read_bytes()).hexdigest() == expected, name
    digest = hashlib.sha256(project_path(freeze['fold_manifest']).read_bytes()).hexdigest()
    assert digest == freeze['fold_manifest_sha256']
    expected_keys = set(product(['F01', 'F02', 'F03', 'F04'], freeze['groups'], [1, 2, 4, 8], freeze['seeds']))
    actual_keys = [(r['fold_id'], r['experiment'], r['pred_len'], r['seed']) for r in sources]
    assert len(actual_keys) == len(set(actual_keys)) == 320
    assert set(actual_keys) == expected_keys
    assert len(coverage) == 64 and all(c['status'] == 'complete' for c in coverage)
    counts = {1: (42400, 22300), 2: (42300, 22200), 4: (42100, 22000), 8: (41700, 21600)}
    audit = []
    for record in sources:
        run_dir = project_path(record['run_dir'])
        config = json.loads((run_dir / 'config.json').read_text())
        for key, value in freeze['hyperparameters'].items():
            assert config[key] == value, (run_dir, key)
        assert config['fold_manifest_sha256'] == digest
        assert config['allow_confirmation'] == (record['fold_id'] == 'F04')
        train_count, test_count = counts[record['pred_len']]
        assert record['train_samples'] == train_count
        assert record['val_samples'] == record['test_samples'] == test_count
        assert pd.Timestamp(record['test_forecast_origin_min']) >= pd.Timestamp(record['test_start'])
        assert pd.Timestamp(record['test_target_time_max']) < pd.Timestamp(record['test_end_exclusive'])
        assert pd.Timestamp(record['validation_target_time_max']) < pd.Timestamp(record['test_start'])
        assert project_path(record['model_path']).is_file()
        assert (run_dir / 'predictions.csv').is_file()
        audit.append({key: record[key] for key in ['fold_id', 'experiment', 'pred_len', 'seed', 'run_id', 'train_samples', 'val_samples', 'test_samples']})
    pd.DataFrame(audit).to_csv(directory / 'completion_config_audit.csv', index=False)

    paired = pd.read_csv(directory / 'paired_seed_improvements.csv')
    paired = paired[paired.scope.eq('unique_grid')]
    wins = paired.groupby(['fold_id', 'experiment', 'lead_hours']).agg(
        rmse_seed_wins=('RMSE_improvement_pct', lambda x: int((x > 0).sum())),
        mae_seed_wins=('MAE_improvement_pct', lambda x: int((x > 0).sum())),
    ).reset_index()
    summary = pd.read_csv(directory / 'paired_seed_summary.csv')
    summary = summary[summary.scope.eq('unique_grid')].merge(wins)
    summary.to_csv(directory / 'paper_paired_results.csv', index=False)

    per_run = pd.read_csv(directory / 'per_run_metrics.csv')
    per_run = per_run[per_run.scope.eq('unique_grid')].copy()
    for metric in ['RMSE', 'MAE']:
        per_run[f'{metric}_gain_vs_persistence'] = 100 * (1 - per_run[metric] / per_run[f'persistence_{metric}'])
    per_run.to_csv(directory / 'persistence_comparison.csv', index=False)
    persistence_summary = per_run.groupby(['fold_id', 'experiment', 'lead_hours']).agg(
        RMSE_gain_mean=('RMSE_gain_vs_persistence', 'mean'),
        RMSE_gain_std=('RMSE_gain_vs_persistence', 'std'),
        MAE_gain_mean=('MAE_gain_vs_persistence', 'mean'),
        MAE_gain_std=('MAE_gain_vs_persistence', 'std'),
    ).reset_index()
    persistence_summary.to_csv(directory / 'persistence_summary.csv', index=False)
    metrics = pd.read_csv(directory / 'seed_summary.csv')
    metrics = metrics[metrics.scope.eq('unique_grid')]
    metrics.to_csv(directory / 'paper_absolute_results.csv', index=False)

    # These compact tables already average duplicate grid predictions before scoring.
    # Equal grid counts at every time make mean RMSE**2 an exact pooled MSE.
    time = pd.read_csv(directory / 'per_time_metrics.csv')
    assert time.grids.eq(98).all()
    keys = ['fold_id', 'seed', 'lead_hours', 'target_time']
    assert not time.duplicated(keys + ['experiment']).any()
    baseline = time[time.experiment.eq('T01')][keys + ['RMSE', 'MAE']]
    time = time[~time.experiment.eq('T01')].merge(baseline, on=keys, suffixes=('', '_T01'), validate='many_to_one')
    assert len(time) == 52860
    time['date'] = pd.to_datetime(time.target_time).dt.strftime('%Y-%m-%d')
    time['mse'] = time.RMSE**2
    time['mse_T01'] = time.RMSE_T01**2
    time['mse_reduction'] = time.mse_T01 - time.mse
    daily = time.groupby(['fold_id', 'experiment', 'lead_hours', 'date'], as_index=False).agg(
        mean_mse_reduction=('mse_reduction', 'mean'), timestamps_across_seeds=('mse', 'size'))
    daily.to_csv(directory / 'daily_mse_reduction.csv', index=False)
    rows = []
    for (fold, experiment, lead), block in time.groupby(['fold_id', 'experiment', 'lead_hours']):
        dates = daily[(daily.fold_id == fold) & (daily.experiment == experiment) & (daily.lead_hours == lead)]
        # Rank by total squared-error reduction, including partial boundary days.
        ranked = block.groupby('date').mse_reduction.sum().sort_values(ascending=False)
        for removed in [0, 1, 3]:
            excluded = ranked.head(removed).index.tolist()
            retained = block[~block.date.isin(excluded)]
            gains = []
            mae_gains = []
            for _, seed_data in retained.groupby('seed'):
                gains.append(100 * (1 - np.sqrt(seed_data.mse.mean() / seed_data.mse_T01.mean())))
                mae_gains.append(100 * (1 - seed_data.MAE.mean() / seed_data.MAE_T01.mean()))
            rows.append(dict(fold_id=fold, experiment=experiment, lead_hours=lead,
                             removed_top_dates=removed, excluded_dates=';'.join(excluded),
                             positive_dates=int((dates.mean_mse_reduction > 0).sum()), total_dates=len(dates),
                             RMSE_gain_mean=np.mean(gains), RMSE_gain_std=np.std(gains, ddof=1),
                             MAE_gain_mean=np.mean(mae_gains), MAE_gain_std=np.std(mae_gains, ddof=1)))
    sensitivity = pd.DataFrame(rows)
    # Check reconstruction against the completed analysis before writing conclusions.
    check = sensitivity[sensitivity.removed_top_dates.eq(0)].merge(summary, on=['fold_id', 'experiment', 'lead_hours'])
    assert np.allclose(check.RMSE_gain_mean_x, check.RMSE_gain_mean_y, atol=1e-8)
    assert np.allclose(check.MAE_gain_mean_x, check.MAE_gain_mean_y, atol=1e-8)
    sensitivity.to_csv(directory / 'date_sensitivity.csv', index=False)
    lines = [
        '# 四折320次实验最终审查', '',
        '来源：确认数组912229、汇总912234，均COMPLETED/0:0。', '',
        '320个唯一配置齐全：四折×四组×四提前量×五种子。冻结代码和开发报告摘要保持一致；',
        '逐配置超参数、样本数、记录的时间边界及模型/预测文件存在性检查通过。',
        '原汇总已检查预测配对和指标；本次复核小型汇总表，没有重跑训练或推理。', '',
        '以下均为98个唯一网格的指标；收益为相对同种子T01的百分比，正值表示改善。',
        '±表示五个训练种子的样本标准差（百分点），不代表独立天气事件的不确定性。', '',
        '| 折 | 组 | 提前量 | RMSE收益% | MAE收益% | RMSE改善种子数 |',
        '|---|---|---|---|---|---|',
    ]
    for r in summary.itertuples():
        lines.append(f'| {r.fold_id} | {r.experiment} | {r.lead_hours}h | {r.RMSE_gain_mean:.2f} ± {r.RMSE_gain_std:.2f} | {r.MAE_gain_mean:.2f} ± {r.MAE_gain_std:.2f} | {r.rmse_seed_wins}/5 |')
    lines += ['', '## 冬季收益的日期敏感性（事后诊断）', '',
              '按五种子的总平方误差改善排序，剔除贡献最大的日期后重新计算。',
              '这是描述性稳健性诊断，不能用来筛选模型或删除正式评估中的困难日期。', '',
              '| 组 | 提前量 | 改善日期数 | 全部RMSE收益% | 去最优1日 | 去最优3日 |',
              '|---|---|---|---|---|---|']
    for (experiment, lead), b in sensitivity[sensitivity.fold_id.eq('F04')].groupby(['experiment', 'lead_hours']):
        b = b.set_index('removed_top_dates')
        lines.append(f'| {experiment} | {lead}h | {int(b.loc[0,"positive_dates"])}/{int(b.loc[0,"total_dates"])} | {b.loc[0,"RMSE_gain_mean"]:.2f} | {b.loc[1,"RMSE_gain_mean"]:.2f} | {b.loc[3,"RMSE_gain_mean"]:.2f} |')
    lines += ['', '## 冬季相对Persistence的收益', '',
              '正值为优于Persistence。以下为各训练种子收益的平均值，不能与相对T01收益混用。', '',
              '| 组 | 提前量 | RMSE收益% | MAE收益% |', '|---|---|---|---|']
    for r in persistence_summary[persistence_summary.fold_id.eq('F04')].itertuples():
        lines.append(f'| {r.experiment} | {r.lead_hours}h | {r.RMSE_gain_mean:.2f} | {r.MAE_gain_mean:.2f} |')
    lines += ['', '## 论文结论与下一步', '',
              '- A01的3h收益在三个开发窗口和冬季确认窗口均为正，支持该区域、该协议下较远空间上下文对短提前量的价值。',
              '- 冬季6h局部邻域收益更大；空间范围的作用依赖时段和提前量，不能声称A01始终最佳。',
              '- 冬季T01在6h的RMSE比Persistence差约9.47%；S02虽相对T01改善9.16%，相对Persistence仅改善0.56%，MAE仍差0.76%。必须同时报告这两个参照。',
              '- 保留24h的弱收益和负收益，不根据冬季确认得分重新调参。',
              '- 四折代表2024年四个28天窗口，不是全年逐时测试；同年同区域结果不能代表跨年或跨区域泛化。',
              '- 本实验比较CERRA内部输入空间范围，不能单独证明相对ERA5的分辨率因果优势。',
              '- 首先将paper_absolute_results.csv、paper_paired_results.csv及既有按折图用于论文结果段；保留Persistence基线。',
              '- 结合date_sensitivity.csv和grid_improvement_coverage.csv报告日期/网格覆盖；任何新增极端风或天气机制分析标为事后探索。',
              '- 目前无需补跑或新增大批训练；如扩展模型、物理特征或区域，应建立新的开发/确认划分，F04已经使用。', '']
    (directory / 'FINAL_SEASONAL_REVIEW.md').write_text('\n'.join(lines))
    print('PASS: 320 configurations; frozen sources; counts/boundaries/artifacts; reconstructed paired scores.')
    print(sensitivity[(sensitivity.fold_id == 'F04') & (sensitivity.removed_top_dates.isin([0, 3]))].to_string(index=False))
    print(per_run[per_run.fold_id.eq('F04')].groupby(['experiment', 'lead_hours'])[['RMSE_gain_vs_persistence', 'MAE_gain_vs_persistence']].mean().round(3))


if __name__ == '__main__':
    main()
