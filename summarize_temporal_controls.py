"""Verify temporal controls and report paired RS effects; no forecast decoding."""
import argparse
import json

import pandas as pd

from direction_pipeline import verify_artifacts
from temporal_controls import (CONFIGS,OUTPUTS,anchor_references,load_config,run_name,tasks,validate_saved)
from temporal_folds import project_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['history','horizon'],required=True)
    parser.add_argument('--expected',choices=['pilot','full'],default='pilot')
    parser.add_argument('--require_complete',action='store_true')
    args = parser.parse_args()
    config,base,sources = load_config(args.stage)
    reused = anchor_references(base,sources)
    root = project_path(OUTPUTS[args.stage])
    expected = len(config['pilot_task_indices']) if args.expected=='pilot' else len(tasks(config))
    rows,results,markers = [],[],[]
    for index,task in enumerate(tasks(config)):
        if not (root/run_name(index,task)).exists():
            continue
        r,path,marker = validate_saved(args.stage,config,index,task,reused)
        results.append(r)
        markers.append(marker['fingerprint'])
        rows.append({k:r[k] for k in ['task','run_id','fold_id','seed','group','input_len','pred_len',
                    'validation_RMSE','validation_MAE','epochs_run','parameter_count']}
                    | r['metrics'] | dict(train_raw_samples=r['counts'][0],validation_raw_samples=r['counts'][1],
                    test_raw_samples=r['counts'][2],source_stage=args.stage,source_run_dir=str(path.relative_to(project_path('.')))))
    selected = sum(r['task']<expected for r in results)
    print(f'{selected}/{expected} new {args.stage} fits complete; {len(results)} total new; 18 registered P3 references')
    if args.require_complete and selected != expected:
        raise SystemExit(1)
    if not results:
        return
    for key in ['code','runtime_versions','static_ready_sha256','static_completion_sha256',
                'dynamic_data_sha256','direction_selection_sha256','fold_sha256']:
        if any(fp[key]!=markers[0][key] for fp in markers):
            raise ValueError(f'Mixed scientific versions: {key}')
    pd.DataFrame(rows).to_csv(root/'new_results.csv',index=False)
    blocks = {(r['fold_id'],r['seed']) for r in results}
    for record in reused.values():
        source = record['source']
        if (source['fold_id'],source['seed']) not in blocks:
            continue
        verify_artifacts(record['path'])
        r = record['result']
        metrics = r['metrics']
        if args.stage=='horizon':
            matched = [v['matched_3h_references'][r['group']] for v in results
                       if v['fold_id']==r['fold_id'] and v['seed']==r['seed']]
            if any(v!=matched[0] for v in matched) or matched[0]['raw_samples']!=22200:
                raise ValueError('Inconsistent common-origin 3h reference metrics.')
            metrics = matched[0]['metrics']
        evaluation_count = 22300 if args.stage=='history' else 22200
        rows.append({k:r[k] for k in ['run_id','fold_id','seed','group','validation_RMSE',
                                      'validation_MAE','epochs_run','parameter_count']}
                    | metrics | dict(task=-1,input_len=56,pred_len=1,train_raw_samples=42400,
                                     validation_raw_samples=22300,test_raw_samples=evaluation_count,source_stage='P3_reused',
                                     source_run_dir=source['run_dir']))
    table = pd.DataFrame(rows)
    if table.duplicated(['fold_id','seed','input_len','pred_len','group']).any():
        raise ValueError('Duplicate control identity.')
    table.to_csv(root/'combined_results.csv',index=False)
    paired = []
    for (fold,seed,length,lead),block in table.groupby(['fold_id','seed','input_len','pred_len']):
        indexed = block.set_index('group')
        if set(indexed.index)=={'RS_S0','RS_S1'}:
            paired.append(dict(fold_id=fold,seed=seed,input_len=length,pred_len=lead,
                model='RS_S1',reference='RS_S0',comparison='RS_effect',
                **{f'{m}_gain_pct':100*(1-indexed.loc['RS_S1',m]/indexed.loc['RS_S0',m]) for m in ['RMSE','MAE']}))
    if args.stage=='history':
        for row in table[table.input_len.ne(56)].itertuples():
            ref = table[(table.fold_id==row.fold_id)&(table.seed==row.seed)&(table.group==row.group)&table.input_len.eq(56)]
            if len(ref)!=1:
                raise ValueError('Missing matched 56-step reference.')
            paired.append(dict(fold_id=row.fold_id,seed=row.seed,input_len=row.input_len,pred_len=1,
                model=row.group,reference=row.group+'_L56',comparison='history_vs_56',
                **{f'{m}_gain_pct':100*(1-getattr(row,m)/ref.iloc[0][m]) for m in ['RMSE','MAE']}))
    frame = pd.DataFrame(paired)
    frame.to_csv(root/'paired_results.csv',index=False)
    frame.groupby(['fold_id','comparison','input_len','pred_len','model','reference'])[
        ['RMSE_gain_pct','MAE_gain_pct']].agg(['mean','std']).to_csv(root/'paired_seed_summary.csv')
    print(f'Saved {len(table)} combined rows and {len(frame)} paired comparisons in {root}')
    if args.stage=='horizon':
        print('3h test metrics use the matched 6h origin subset; native validation scores are not re-matched. '
              'Training sample counts differ by the necessary target-boundary embargo.')


if __name__ == '__main__':
    main()
