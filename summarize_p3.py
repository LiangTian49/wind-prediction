"""Verify P3 output versions and summarize the predeclared paired comparisons."""

import argparse
import json

import pandas as pd

from direction_pipeline import sha256, verify_artifacts
from rs_static_contract import RS_GROUPS
from temporal_folds import project_path
from train_p3 import CONFIG, OUTPUT, tasks

COMPARISONS = [('RS_S1', 'RS_S0'), ('RS_T1', 'RS_T0'),
               ('RS_S0', 'RS_T0'), ('RS_S1', 'RS_T1')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected', choices=['pilot', 'initial', 'formal'], default='pilot')
    parser.add_argument('--require_complete', action='store_true')
    args = parser.parse_args()
    config = json.loads(project_path(CONFIG).read_text())
    expected = {'pilot': 4, 'initial': config['initial_task_count'], 'formal': config['optional_formal_task_count']}[args.expected]
    root = project_path(OUTPUT)
    records, signatures, fingerprints = [], {}, []
    for index, (seed, fold, group) in enumerate(tasks(config)):
        path = root/f'{index:02d}_{fold}_{group}_seed{seed}'
        if not path.exists():
            continue
        marker = verify_artifacts(path)
        if marker['fingerprint']['config_sha256'] != sha256(project_path(CONFIG)):
            raise ValueError(f'Changed P3 configuration: {path}')
        result = json.loads((path/'result.json').read_text())
        if (result['task'] != index or result['seed'] != seed or result['fold_id'] != fold
                or result['group'] != group or result['counts'] != [42400, 22300, 22300]
                or result['unique_grids'] != 98 or result['static_channels'] != 14
                or result['static_enabled'] != RS_GROUPS[group][1]):
            raise ValueError(f'P3 identity, structure or sample coverage mismatch: {path}')
        input_group = RS_GROUPS[group][0]
        if (result['dynamic_channels'] != (3 if input_group == 'T01' else 11)
                or result['parameter_count'] != (18713 if input_group == 'T01' else 20761)):
            raise ValueError(f'Unexpected fusion architecture: {path}')
        key = (fold, seed)
        if key in signatures and signatures[key] != result['sample_signatures']:
            raise ValueError('Paired groups have different sample keys or labels.')
        signatures[key] = result['sample_signatures']
        fingerprints.append(marker['fingerprint'])
        records.append({key: result[key] for key in ['task', 'run_id', 'fold_id', 'group', 'seed',
                        'static_enabled', 'validation_RMSE', 'validation_MAE', 'parameter_count', 'epochs_run']}
                       | result['metrics'])
    selected = [row for row in records if row['task'] < expected]
    print(f'{len(selected)}/{expected} configurations complete for {args.expected}; {len(records)} total saved')
    if args.require_complete and len(selected) != expected:
        raise SystemExit(1)
    if not records:
        return
    for key in ['fold_sha256', 'static_ready_sha256', 'static_completion_sha256',
                'direction_selection_sha256', 'runtime_versions', 'code']:
        if any(value[key] != fingerprints[0][key] for value in fingerprints):
            raise ValueError(f'Mixed scientific versions: {key}')
    table = pd.DataFrame(records)
    table.to_csv(root/'summary.csv', index=False)
    pairs, interaction = [], []
    for (fold, seed), block in table.groupby(['fold_id', 'seed']):
        block = block.set_index('group')
        for model, reference in COMPARISONS:
            if model not in block.index or reference not in block.index:
                continue
            pairs.append(dict(fold_id=fold, seed=seed, model=model, reference=reference,
                              **{f'{metric}_gain_pct':100*(1-block.loc[model, metric]/block.loc[reference, metric])
                                 for metric in ['RMSE', 'MAE']}))
        if set(block.index) == set(RS_GROUPS):
            interaction.append(dict(fold_id=fold, seed=seed,
                **{f'{metric}_RS_gain_interaction_mps':
                   (block.loc['RS_S0', metric]-block.loc['RS_S1', metric])-
                   (block.loc['RS_T0', metric]-block.loc['RS_T1', metric]) for metric in ['RMSE', 'MAE']}))
    pd.DataFrame(pairs).to_csv(root/'paired_results.csv', index=False)
    pd.DataFrame(interaction).to_csv(root/'interaction_results.csv', index=False)
    print(f'Saved summary and paired tables in {root}')


if __name__ == '__main__':
    main()
