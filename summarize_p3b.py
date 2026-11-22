"""Verify new P3b fits and reuse the 18 separately archived P3 spatial references."""
import argparse
import json

import numpy as np
import pandas as pd

from direction_pipeline import code_hashes, sha256, verify_artifacts
from p3b_contract import (CONFIG, OUTPUT, COMPARISONS, references, require_signatures,
                          slot_names, tasks, validate_config)
from temporal_folds import project_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected', choices=['pilot', 'full'], default='pilot')
    parser.add_argument('--require_complete', action='store_true')
    args = parser.parse_args()
    config = json.loads(project_path(CONFIG).read_text())
    reused = references(config, validate_config(config))
    root = project_path(OUTPUT)
    expected = 3 if args.expected == 'pilot' else 27
    rows, fingerprints = [], []
    for index, (seed, fold, group) in enumerate(tasks(config)):
        run_id = f'{index:02d}_{fold}_{group}_seed{seed}'
        path = root/run_id
        if not path.exists():
            continue
        marker = verify_artifacts(path)
        fp = marker['fingerprint']
        if (marker['status'] != 'completed' or fp['config_sha256'] != sha256(project_path(CONFIG))
                or fp['reference_manifest_sha256'] != config['reference_manifest_sha256']
                or fp['code'] != code_hashes(list(fp['code']))):
            raise ValueError(f'Changed P3b scientific version: {path}')
        result = json.loads((path/'result.json').read_text())
        actual = json.loads((path/'config.json').read_text())
        if (result['task'] != index or result['run_id'] != run_id or result['fold_id'] != fold
                or result['seed'] != seed or result['group'] != group or result['status'] != 'completed'
                or result['counts'] != [42400, 22300, 22300] or result['unique_grids'] != 98
                or result['parameter_count'] != 20761 or result['dynamic_channels'] != 11
                or result['static_channels'] != 14 or actual['static_slot_names'] != slot_names(group)):
            raise ValueError(f'Identity, coverage or representation mismatch: {path}')
        matched = {k:r for k,r in reused.items() if r['source']['seed'] == seed and r['source']['fold_id'] == fold}
        require_signatures(result['sample_signatures'], matched)
        if (fp['reference_markers'] != {k:r['source']['completion_sha256'] for k,r in matched.items()}
                or any(fp['runtime_versions'] != r['marker']['fingerprint']['runtime_versions'] for r in matched.values())):
            raise ValueError('Reference provenance or software versions differ.')
        values = [result[k] for k in ['validation_RMSE', 'validation_MAE']] + list(result['metrics'].values())
        if not np.isfinite(values).all():
            raise ValueError('Missing/nonfinite scores.')
        fingerprints.append(fp)
        rows.append({k:result[k] for k in ['task', 'run_id', 'fold_id', 'group', 'seed',
                                         'validation_RMSE', 'validation_MAE', 'parameter_count', 'epochs_run']}
                    | result['metrics'] | dict(source_stage='P3b', source_run_dir=str(path.relative_to(project_path('.')))))
    selected = sum(row['task'] < expected for row in rows)
    print(f'{selected}/{expected} new configurations complete for {args.expected}; {len(rows)} total new; 18 P3 references')
    if args.require_complete and selected != expected:
        raise SystemExit(1)
    if not rows:
        return
    for key in ['fold_sha256', 'static_ready_sha256', 'static_completion_sha256',
                'dynamic_data_sha256', 'direction_selection_sha256', 'runtime_versions', 'code']:
        if any(fp[key] != fingerprints[0][key] for fp in fingerprints):
            raise ValueError(f'Mixed P3b versions: {key}')
    for record in reused.values():
        verify_artifacts(record['path'])
    new_table = pd.DataFrame(rows)
    new_table.to_csv(root/'new_results.csv', index=False)
    # Include only fold/seed blocks with new fits; reference-only blocks cannot masquerade as completed controls.
    blocks = set(zip(new_table.fold_id, new_table.seed))
    for record in reused.values():
        r = record['result']
        if (r['fold_id'], r['seed']) in blocks:
            rows.append({k:r[k] for k in ['run_id', 'fold_id', 'group', 'seed', 'validation_RMSE',
                                          'validation_MAE', 'parameter_count', 'epochs_run']}
                        | r['metrics'] | dict(task=-1, source_stage='P3_reused', source_run_dir=record['source']['run_dir']))
    table = pd.DataFrame(rows)
    table.to_csv(root/'combined_results.csv', index=False)
    pairs = []
    for (fold, seed), block in table.groupby(['fold_id', 'seed']):
        block = block.set_index('group', verify_integrity=True)
        for model, reference in COMPARISONS:
            if model in block.index and reference in block.index:
                pairs.append(dict(fold_id=fold, seed=seed, model=model, reference=reference,
                    **{f'{metric}_gain_pct':100*(1-block.loc[model, metric]/block.loc[reference, metric])
                       for metric in ['RMSE', 'MAE']}))
    paired = pd.DataFrame(pairs)
    paired.to_csv(root/'paired_results.csv', index=False)
    paired.groupby(['fold_id', 'model', 'reference'])[['RMSE_gain_pct','MAE_gain_pct']].agg(['mean','std']).to_csv(
        root/'paired_seed_summary.csv')
    print(f'Saved new, reused, paired and seed-summary tables: {root}')


if __name__ == '__main__':
    main()
