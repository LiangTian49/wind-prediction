"""Slurm-only paired origin-date analysis of matched history and horizon controls."""
import json
import os

import numpy as np
import pandas as pd

from analyze_experiments import collapse_grids
from analyze_p3_dates import gain_distribution
from direction_pipeline import code_hashes, output_transaction, sha256, verify_artifacts
from temporal_controls import CONFIGS,OUTPUTS,anchor_references,load_config,run_name,tasks,validate_saved
from temporal_folds import project_path

ORIGINS = ['point_id','forecast_origin']
GRID_ORIGINS = ['center_y','center_x','forecast_origin']


def label(length,lead,group):
    return f'L{length}_H{lead*3}_{group}'


def comparisons(stage):
    lengths,leads = ([8,24,56],[1]) if stage=='history' else ([56],[1,2])
    pairs = [(label(length,lead,'RS_S1'),label(length,lead,'RS_S0')) for length in lengths for lead in leads]
    if stage=='history':
        pairs += [(label(length,1,group),label(56,1,group)) for length in [8,24] for group in ['RS_S0','RS_S1']]
    pairs += [(label(length,lead,group),'Persistence') for length in lengths for lead in leads for group in ['RS_S0','RS_S1']]
    return pairs


def interaction_keys(stage):
    if stage=='history':
        return [(label(length,1,'RS_S1'),label(length,1,'RS_S0'),
                 label(56,1,'RS_S1'),label(56,1,'RS_S0')) for length in [8,24]]
    return [(label(56,2,'RS_S1'),label(56,2,'RS_S0'),label(56,1,'RS_S1'),label(56,1,'RS_S0'))]


def match_origins(template,frame):
    if template.duplicated(ORIGINS).any() or frame.duplicated(ORIGINS).any():
        raise ValueError('Duplicate point forecast origins.')
    matched = template[ORIGINS].merge(frame,on=ORIGINS,validate='one_to_one')
    if len(matched)!=len(template):
        raise ValueError('Incomplete common-origin reference.')
    return matched


def origin_errors(frame,length,lead):
    if (not ((frame.forecast_origin-frame.input_start)==pd.Timedelta(hours=3*(length-1))).all()
            or not ((frame.target_time-frame.forecast_origin)==pd.Timedelta(hours=3*lead)).all()):
        raise ValueError('Actual input span or lead time mismatch.')
    collapsed = collapse_grids(frame).sort_values(GRID_ORIGINS).reset_index(drop=True)
    collapsed['date'] = collapsed.forecast_origin.dt.normalize()
    error = collapsed.y_pred-collapsed.y_true
    persistence_error = collapsed.persistence_pred-collapsed.y_true
    for column,values in [('squared',error**2),('absolute',error.abs()),
                           ('persistence_squared',persistence_error**2),('persistence_absolute',persistence_error.abs())]:
        collapsed[column]=values
    daily = collapsed.groupby('date')[['squared','absolute','persistence_squared','persistence_absolute']].sum()
    daily['count']=collapsed.groupby('date').size()
    return collapsed,daily


def assert_aligned(frames):
    baseline = next(iter(frames.values()))
    for frame in frames.values():
        pd.testing.assert_frame_equal(baseline[GRID_ORIGINS],frame[GRID_ORIGINS])
        np.testing.assert_allclose(baseline.persistence_pred,frame.persistence_pred,atol=1e-6,rtol=0)
    by_lead = {}
    for name,frame in frames.items():
        lead = name.split('_H')[1].split('_')[0]
        if lead in by_lead:
            other=by_lead[lead]
            pd.testing.assert_series_equal(other.target_time,frame.target_time,check_names=False)
            np.testing.assert_allclose(other.y_true,frame.y_true,atol=1e-6,rtol=0)
        else:
            by_lead[lead]=frame


def circular_weights(rng,days,resamples=2000):
    starts=rng.integers(0,days,size=(resamples,int(np.ceil(days/3))))
    indices=((starts[:,:,None]+np.arange(3))%days).reshape(resamples,-1)[:,:days]
    return np.stack([(indices==day).sum(axis=1) for day in range(days)],axis=1)


def load_records(stage,summary):
    config,base,sources=load_config(stage)
    reused=anchor_references(base,sources)
    expected={}
    for index,task in enumerate(tasks(config)):
        r,path,_=validate_saved(stage,config,index,task,reused)
        expected[r['run_id']]=(str(path),r['fold_id'],r['seed'],r['group'],r['input_len'],r['pred_len'])
    for record in reused.values():
        r=record['result']
        verify_artifacts(record['path'])
        expected[r['run_id']]=(str(record['path']),r['fold_id'],r['seed'],r['group'],56,1)
    if len(summary)!=len(expected) or summary.run_id.duplicated().any() or set(summary.run_id)!=set(expected):
        raise ValueError('Require all registered new and reused temporal results.')
    for row in summary.itertuples():
        actual=(str(project_path(row.source_run_dir)),row.fold_id,row.seed,row.group,row.input_len,row.pred_len)
        if actual!=expected[row.run_id]:
            raise ValueError('Wrong source or temporal identity in summary.')
    return config


def main():
    job=os.environ.get('SLURM_JOB_ID')
    if not job:
        raise SystemExit('Submit slurm/run_temporal_analysis.sh.')
    summaries={stage:pd.read_csv(project_path(OUTPUTS[stage])/'combined_results.csv') for stage in ['history','horizon']}
    for stage,summary in summaries.items():
        load_records(stage,summary)
    fingerprint=dict(configs={stage:sha256(project_path(path)) for stage,path in CONFIGS.items()},
        summaries={stage:sha256(project_path(OUTPUTS[stage])/'combined_results.csv') for stage in summaries},
        run_markers={stage+':'+r.run_id:sha256(project_path(r.source_run_dir)/'completion.json')
                     for stage,summary in summaries.items() for r in summary.itertuples()},
        code=code_hashes(['analyze_temporal_controls.py','temporal_controls.py','summarize_temporal_controls.py',
                          'analyze_p3_dates.py','analyze_experiments.py','direction_pipeline.py']))
    final=project_path(f'result/rs_hpc_v3/temporal_controls/analysis_{job}')
    with output_transaction(final,fingerprint) as work:
        if work is None:
            return
        intervals,interactions,contributions=[],[],[]
        for stage,summary in summaries.items():
            rng=np.random.default_rng(20261115)
            for fold,block in summary.groupby('fold_id'):
                daily={}
                for seed,seed_block in block.groupby('seed'):
                    raw={label(r.input_len,r.pred_len,r.group):pd.read_csv(project_path(r.source_run_dir)/'predictions.csv',
                        parse_dates=['input_start','forecast_origin','target_time']) for r in seed_block.itertuples()}
                    if stage=='horizon':
                        template=raw[label(56,2,'RS_S0')]
                        raw={key:match_origins(template,frame) for key,frame in raw.items()}
                    frames={}
                    for row in seed_block.itertuples():
                        name=label(row.input_len,row.pred_len,row.group)
                        frames[name],daily[seed,name]=origin_errors(raw[name],row.input_len,row.pred_len)
                        errors=daily[seed,name]
                        for metric,column in [('RMSE','squared'),('MAE','absolute')]:
                            value=errors[column].sum()/errors['count'].sum()
                            if metric=='RMSE':value=np.sqrt(value)
                            if not np.isclose(value,getattr(row,metric),atol=1e-8,rtol=0):
                                raise ValueError('Reconstructed metrics differ from matched summary.')
                    assert_aligned(frames)
                days=next(iter(daily.values())).index
                if len(days)!=28 or any(not d.index.equals(days) for d in daily.values()):
                    raise ValueError('Unmatched date coverage.')
                weights=circular_weights(rng,len(days))
                seeds=sorted(block.seed.unique())
                draws,points={},{}
                for model,reference in comparisons(stage):
                    for metric,column in [('RMSE','squared'),('MAE','absolute')]:
                        fit_points,fit_draws=[],[]
                        for seed in seeds:
                            errors=daily[seed,model]
                            ref=daily[seed,reference] if reference!='Persistence' else pd.DataFrame(dict(
                                squared=errors.persistence_squared,absolute=errors.persistence_absolute,count=errors['count']))
                            point,replicate=gain_distribution(errors,ref,column,metric,weights)
                            draws[seed,model,reference,metric]=replicate
                            points[seed,model,reference,metric]=point
                            fit_points.append(point);fit_draws.append(replicate)
                            low,high=np.quantile(replicate,[.025,.975])
                            intervals.append(dict(stage=stage,fold_id=fold,model=model,reference=reference,metric=metric,
                                seed=str(seed),gain_pct=point,date_p025_pct=low,date_p975_pct=high))
                            for date in days:
                                contributions.append(dict(stage=stage,fold_id=fold,seed=int(seed),model=model,
                                    reference=reference,metric=metric,date=str(date.date()),count=int(errors.loc[date,'count']),
                                    model_loss=float(errors.loc[date,column]),reference_loss=float(ref.loc[date,column]),
                                    loss_reduction=float(ref.loc[date,column]-errors.loc[date,column])))
                        low,high=np.quantile(np.mean(fit_draws,axis=0),[.025,.975])
                        intervals.append(dict(stage=stage,fold_id=fold,model=model,reference=reference,metric=metric,
                            seed='fixed_seed_mean',gain_pct=float(np.mean(fit_points)),date_p025_pct=low,date_p975_pct=high))
                for model,reference,base_model,base_ref in interaction_keys(stage):
                    for metric in ['RMSE','MAE']:
                        effect_draws,effect_points=[],[]
                        for seed in seeds:
                            difference=draws[seed,model,reference,metric]-draws[seed,base_model,base_ref,metric]
                            point=points[seed,model,reference,metric]-points[seed,base_model,base_ref,metric]
                            effect_draws.append(difference);effect_points.append(point)
                            low,high=np.quantile(difference,[.025,.975])
                            interactions.append(dict(stage=stage,fold_id=fold,model=model,reference=reference,
                                baseline_model=base_model,baseline_reference=base_ref,metric=metric,seed=str(seed),
                                RS_gain_difference_pp=point,date_p025_pp=low,date_p975_pp=high))
                        low,high=np.quantile(np.mean(effect_draws,axis=0),[.025,.975])
                        interactions.append(dict(stage=stage,fold_id=fold,model=model,reference=reference,
                            baseline_model=base_model,baseline_reference=base_ref,metric=metric,seed='fixed_seed_mean',
                            RS_gain_difference_pp=float(np.mean(effect_points)),date_p025_pp=low,date_p975_pp=high))
        if (len(intervals),len(interactions),len(contributions))!=(456,72,9576):
            raise ValueError('Incomplete temporal date analysis.')
        for records in [intervals,interactions,contributions]:
            if pd.DataFrame(records).isna().any().any():
                raise ValueError('Missing analysis values.')
        pd.DataFrame(intervals).to_csv(work/'date_intervals.csv',index=False)
        pd.DataFrame(interactions).to_csv(work/'RS_gain_interactions.csv',index=False)
        pd.DataFrame(contributions).to_csv(work/'daily_loss_contributions.csv',index=False)
        for stage,summary in summaries.items():summary.to_csv(work/f'{stage}_absolute_results.csv',index=False)
        (work/'method.json').write_text(json.dumps(dict(unique_fits=72,new_fits=54,reused_unique_fits=18,
            evaluation_rows=90,interval_rows=456,interaction_rows=72,daily_rows=9576,
            block_days=3,resamples=2000,rng_seed=20261115,date_assignment='forecast_origin date',
            scope='Post hoc window-conditional descriptive intervals, no multiplicity correction; no seed resampling or prediction ensemble',
            date_rule_change='Previous P3/P3b analyses used target date. This version explicitly uses origin date for both stages to align leads; old outputs retained.',
            interaction='Difference in paired RS percentage improvements, in percentage points; NOT a percentage change in errors',
            coverage='History: all partitions matched to original 56-step/3h anchor. Horizon: 3h and 6h test origins matched, labels differ; native training/validation embargoes retained.'),indent=2))
    print(f'[Temporal dates] Completed: {final}')


if __name__=='__main__':
    main()
