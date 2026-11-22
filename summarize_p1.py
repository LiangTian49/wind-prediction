"""Summarize completed P1 artifacts and choose center references by validation RMSE."""

import argparse
import json

import pandas as pd

from temporal_folds import project_path
from train_p1_baseline import CONFIG, OUTPUT, digest, tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require_complete', action='store_true')
    args = parser.parse_args()
    config_path = project_path(CONFIG)
    config = json.loads(config_path.read_text())
    config_sha = digest(config_path)
    root = project_path(OUTPUT)
    records = []
    for index, (fold, steps, family, group) in enumerate(tasks(config)):
        run_id = f'{index:02d}_{fold}_{steps * 3}h_{family}_{group}'
        path = root / run_id
        if not path.exists():
            continue
        result = json.loads((path / 'result.json').read_text())
        if (result['run_id'] != run_id or result['fingerprints']['config_sha256'] != config_sha
                or any(digest(path / name) != sha for name, sha in result['artifact_hashes'].items())):
            raise ValueError(f'Artifact or version mismatch: {path}')
        records.append({k: result[k] for k in ['run_id', 'fold_id', 'lead_hours', 'model_family',
                                               'experiment', 'validation_RMSE']} |
                       result['metrics'] | {'candidate_fits': len(result['trials'])})
    expected = len(tasks(config))
    print(f'{len(records)}/{expected} P1 configurations complete')
    if args.require_complete and len(records) != expected:
        raise SystemExit(1)
    if not records:
        return
    table = pd.DataFrame(records)
    keys = ['fold_id', 'lead_hours']
    # Omit a fold/lead from selection until both center families have completed.
    centers = table[table.experiment.eq('T01')]
    centers = centers[centers.groupby(keys).run_id.transform('size').eq(len(config['models']))]
    chosen = centers.sort_values(['validation_RMSE', 'model_family']).drop_duplicates(keys)
    table = table.merge(chosen[keys + ['run_id', 'RMSE', 'MAE']], how='left', on=keys,
                        suffixes=('', '_strong_center'), validate='many_to_one')
    same_family = table[table.experiment.eq('T01')][keys + ['model_family', 'RMSE', 'MAE']]
    table = table.merge(same_family, how='left', on=keys + ['model_family'],
                        suffixes=('', '_same_family_center'), validate='many_to_one')
    for metric in ['RMSE', 'MAE']:
        for ref in ['strong_center', 'same_family_center', 'persistence']:
            ref_col = f'persistence_{metric}' if ref == 'persistence' else f'{metric}_{ref}'
            table[f'{metric}_gain_vs_{ref}_pct'] = 100 * (1 - table[metric] / table[ref_col])
    table.to_csv(root / 'summary.csv', index=False)
    print(f'{table.candidate_fits.sum()} candidate fits represented; summary at {root / "summary.csv"}')


if __name__ == '__main__':
    main()
