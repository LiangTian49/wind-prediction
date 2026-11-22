"""Validate completed direction ablations and select dynamic representation on validation only."""

import argparse
import json

import pandas as pd

from direction_pipeline import sha256, verify_artifacts
from temporal_folds import project_path
from train_p1c import CONFIG, OUTPUT, task_list


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require_complete', action='store_true')
    args = parser.parse_args()
    config = json.loads(project_path(CONFIG).read_text())
    rows = []
    fingerprints = []
    for i, (fold, group) in enumerate(task_list(config)):
        path = project_path(OUTPUT) / f'{i:02d}_{fold}_{group}_seed{config["seed"]}'
        if not path.exists():
            continue
        marker = verify_artifacts(path)
        if marker['fingerprint']['config_sha256'] != sha256(project_path(CONFIG)):
            raise ValueError('Result configuration differs from current P1c plan.')
        result = json.loads((path / 'result.json').read_text())
        if result['counts'] != [42400, 22300, 22300] or result['unique_grids'] != 98:
            raise ValueError(f'Unexpected sample coverage: {path}')
        expected_width = 5 if group.startswith('D_T') else 29
        expected_parameters = 4*64*(expected_width+64+1)+65
        if (result['fold_id'] != fold or result['group'] != group
                or result['input_channels'] != expected_width
                or result['parameter_count'] != expected_parameters):
            raise ValueError(f'Group identity or architecture mismatch: {path}')
        fingerprints.append(marker['fingerprint'])
        rows.append({k: result[k] for k in ['run_id', 'fold_id', 'group', 'validation_RMSE', 'validation_MAE',
                                           'parameter_count', 'input_channels', 'epochs_run']} | result['metrics'])
    print(f'{len(rows)}/12 direction configurations complete')
    if args.require_complete and len(rows) != 12:
        raise SystemExit(1)
    if not rows:
        return
    # Shared code, data and protocol must match across groups; reference files differ.
    for key in ['config_sha256', 'fold_sha256', 'data_marker_sha256', 'code']:
        if any(f[key] != fingerprints[0][key] for f in fingerprints):
            raise ValueError(f'Mixed experiment versions: {key}')
    table = pd.DataFrame(rows)
    table.to_csv(project_path(OUTPUT) / 'summary.csv', index=False)
    if len(rows) == 12:
        off = table[table.group.eq('D_S0')].set_index('fold_id')
        on = table[table.group.eq('D_S1')].set_index('fold_id')
        gain = 100*(1-on.validation_RMSE/off.validation_RMSE)
        selection = dict(status='validation_selected', rule=config['direction_selection'],
                         validation_gain_by_fold=gain.to_dict(), equal_fold_mean_gain_pct=float(gain.mean()),
                         direction_enabled=bool(gain.mean()>0),
                         scope='Development selection; new-time confirmation still required.')
        (project_path(OUTPUT) / 'dynamic_selection.json').write_text(json.dumps(selection, indent=2))


if __name__ == '__main__':
    main()
