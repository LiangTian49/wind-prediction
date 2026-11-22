"""Matched-history and lead-time controls without modifying frozen data/model code."""
import json
from itertools import product

import numpy as np
import pandas as pd

from direction_pipeline import code_hashes, sha256, verify_artifacts
from p3b_contract import references, validate_config as validate_p3b_config
from temporal_folds import project_path, STEP
from train_p3 import build_partitions, sample_signature

CONFIGS = {stage:f'configs/rs_hpc_v3_{stage}_v1.json' for stage in ['history','horizon']}
OUTPUTS = {'history':'runs/rs_hpc_v3/P1b_history/v1', 'horizon':'runs/rs_hpc_v3/P3c_horizon/v1'}
GROUPS = ['RS_S0','RS_S1']


def tasks(config):
    return list(product(config['seeds'],config['folds'],config['input_lengths'],config['groups']))


def run_name(index, task):
    seed,fold,length,group = task
    return f'{index:02d}_{fold}_L{length}_{group}_seed{seed}'


def load_config(stage):
    config = json.loads(project_path(CONFIGS[stage]).read_text())
    base_path = project_path(config['anchor_config'])
    if sha256(base_path) != config['anchor_config_sha256']:
        raise ValueError('Frozen P3b anchor configuration changed.')
    base = json.loads(base_path.read_text())
    sources = validate_p3b_config(base)
    keys = ['protocol','fold_manifest','spatial_dir','static_dir','dynamic_groups','lr','batch_size',
            'epochs','patience','architecture','direction_enabled','direction_selection',
            'reference_manifest','reference_manifest_sha256',
            'direction_selection_sha256','static_ready_sha256','static_completion_sha256','data_sha256']
    if any(config[k] != base[k] for k in keys):
        raise ValueError('A matched input/training setting changed.')
    if (config['stage'] != stage or config['groups'] != GROUPS or config['seeds'] != [42,43,44]
            or config['folds'] != ['F01','F02','F03'] or config['anchor_length'] != 56
            or config['input_lengths'] != ([8,24] if stage == 'history' else [56])
            or config['pred_len'] != (1 if stage == 'history' else 2)):
        raise ValueError('Temporal control task budget or protocol changed.')
    if stage == 'horizon' and config['history_config_sha256'] != sha256(project_path(CONFIGS['history'])):
        raise ValueError('Registered history-stage configuration changed.')
    return config,base,sources


def control_partitions(df, features, fold, length, horizon, anchor_length=56):
    """Construct the original 3h anchor once, preserving its eligible origins."""
    if not 0 < length <= anchor_length or horizon not in [1,2] or (horizon == 2 and length != anchor_length):
        raise ValueError('Unsupported registered history/horizon.')
    anchor,scaler,split = build_partitions(df,features,fold,anchor_length,1)
    signatures = {name:sample_signature(v[3],v[1],v[2]) for name,v in anchor.items()}
    samples = {}
    for name,(x,y,persistence,metadata) in anchor.items():
        if horizon == 1:
            actual = metadata.copy()
            actual['input_start'] = actual.forecast_origin-(length-1)*STEP
            samples[name] = (x[:,-length:,:],y,persistence,actual)
        else:
            keep,following = [],[]
            for _,block in metadata.groupby('point_id',sort=False):
                indices = block.index.to_numpy()
                if len(indices)<2:
                    raise ValueError('No eligible 6h windows for one point/partition.')
                keep.extend(indices[:-1])
                following.extend(indices[1:])
            keep,following = np.asarray(keep),np.asarray(following)
            actual = metadata.iloc[keep].reset_index(drop=True).copy()
            actual['target_time'] = metadata.iloc[following].target_time.to_numpy()
            samples[name] = (x[keep],y[following],persistence[keep],actual)
        actual = samples[name][3]
        if not ((actual.target_time-actual.forecast_origin) == horizon*STEP).all():
            raise ValueError('Target lead mismatch.')
    return samples,scaler,split,signatures


def anchor_references(base,sources,fold=None,seed=None,full=False):
    return references(base,sources,fold,seed,full_artifacts=full)


def validate_saved(stage, config, index, task, reused):
    path = project_path(OUTPUTS[stage])/run_name(index,task)
    marker = verify_artifacts(path)
    fp = marker['fingerprint']
    if (marker['status'] != 'completed' or fp['config_sha256'] != sha256(project_path(CONFIGS[stage]))
            or fp['code'] != code_hashes(list(fp['code']))):
        raise ValueError('Changed temporal-control version.')
    result = json.loads((path/'result.json').read_text())
    seed,fold,length,group = task
    expected = [42400,22300,22300] if stage == 'history' else [42300,22200,22200]
    if (result['task'] != index or result['run_id'] != path.name or result['stage'] != stage
            or result['seed'] != seed or result['fold_id'] != fold or result['group'] != group
            or result['input_len'] != length or result['pred_len'] != config['pred_len']
            or result['counts'] != expected or result['unique_grids'] != 98
            or result['parameter_count'] != 20761 or result['dynamic_channels'] != 11
            or result['static_channels'] != 14):
        raise ValueError('Temporal-control coverage or identity mismatch.')
    matched = {k:r for k,r in reused.items() if r['source']['seed']==seed and r['source']['fold_id']==fold}
    if len(matched) != 2 or fp['reference_markers'] != {k:r['source']['completion_sha256'] for k,r in matched.items()}:
        raise ValueError('Changed anchor references.')
    for r in matched.values():
        if (result['anchor_signatures'] != r['result']['sample_signatures']
                or fp['runtime_versions'] != r['marker']['fingerprint']['runtime_versions']):
            raise ValueError('Anchor samples or runtime versions differ.')
    values = list(result['metrics'].values())+[result['validation_RMSE'],result['validation_MAE']]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite metrics.')
    return result,path,marker


def require_history_complete():
    config,base,sources = load_config('history')
    reused = anchor_references(base,sources)
    for index,task in enumerate(tasks(config)):
        path = project_path(OUTPUTS['history'])/run_name(index,task)
        if not path.exists():
            raise ValueError('Finish and verify all 36 history fits before starting the 6h stage.')
        validate_saved('history',config,index,task,reused)


def matched_reference_metrics(probe, other, horizon):
    """3h and 6h share origins/Persistence, not future labels or target times."""
    from analyze_experiments import collapse_grids, score
    from train_p1_baseline import verify_reference

    keys = ['point_id','forecast_origin']
    if probe.duplicated(keys).any() or other.duplicated(keys).any():
        raise ValueError('Duplicate forecast origins.')
    matched = probe[keys].merge(other,on=keys,validate='one_to_one')
    if len(matched) != len(probe):
        raise ValueError('Incomplete common-origin reference.')
    np.testing.assert_allclose(probe.persistence_pred,matched.persistence_pred,atol=1e-6,rtol=0)
    if horizon == 1:
        verify_reference(probe,matched)
    elif horizon == 2:
        if not ((probe.target_time-matched.target_time) == STEP).all():
            raise ValueError('3h/6h target-time alignment mismatch.')
        pd.testing.assert_series_equal(probe.input_start,matched.input_start,check_names=False)
    else:
        raise ValueError('Unsupported lead.')
    unique = collapse_grids(matched)
    return dict(metrics=score(unique.y_true,unique.y_pred,unique.persistence_pred),
                raw_samples=len(matched),origin_signature=sample_signature(matched[keys],np.zeros(len(matched)),matched.persistence_pred))
