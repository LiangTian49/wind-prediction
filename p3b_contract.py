"""Frozen 14-slot static representations and P3 reference checks for P3b."""

import json
from itertools import product

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from direction_pipeline import code_hashes, sha256, verify_artifacts
from rs_static_contract import STATIC_FEATURES, static_inputs
from temporal_folds import project_path

CONFIG = 'configs/rs_hpc_v3_p3b_v1.json'
OUTPUT = 'runs/rs_hpc_v3/P3b/v1'
GROUPS = ('S_COORD', 'S_LC', 'S_DSM')
COMPARISONS = [('RS_S1', 'RS_S0'), ('S_COORD', 'RS_S0'), ('RS_S1', 'S_COORD'),
               ('S_LC', 'RS_S0'), ('S_DSM', 'RS_S0'),
               ('RS_S1', 'S_LC'), ('RS_S1', 'S_DSM')]


def tasks(config):
    return list(product(config['seeds'], config['folds'], config['groups']))


def slot_names(group):
    if group == 'S_COORD':
        return ['grid_lat', 'grid_lon'] + ['disabled']*12
    if group == 'S_LC':
        return list(STATIC_FEATURES[:11]) + ['disabled']*3
    if group == 'S_DSM':
        return ['disabled']*11 + list(STATIC_FEATURES[11:])
    raise ValueError(f'Unknown P3b representation: {group}')


def coordinate_table(mapping):
    if mapping.point_id.duplicated().any():
        raise ValueError('Duplicate point identifiers.')
    values = mapping[['grid_id', 'grid_lat', 'grid_lon']].drop_duplicates()
    if values.grid_id.duplicated().any() or not np.isfinite(values[['grid_lat', 'grid_lon']]).all().all():
        raise ValueError('Inconsistent or missing coordinates for one target grid.')
    if not values.grid_lat.between(-90, 90).all() or not values.grid_lon.between(-180, 180).all():
        raise ValueError('Require coordinates in geographic degrees.')
    return values.set_index('grid_id', verify_integrity=True)


def fit_coordinates(mapping, train_grid_ids):
    coordinates = coordinate_table(mapping)
    training = coordinates.loc[sorted(set(train_grid_ids)), ['grid_lat', 'grid_lon']]
    if training.empty:
        raise ValueError('No training coordinates.')
    return StandardScaler().fit(training)


def representation(metadata, mapping, table, static_scaler, coordinate_scaler, group):
    """Preserve window ordering; mask standardized rather than raw channels."""
    slot_names(group)
    if group == 'S_COORD':
        grids = mapping.set_index('point_id', verify_integrity=True).loc[metadata.point_id, 'grid_id']
        values = coordinate_table(mapping).loc[grids, ['grid_lat', 'grid_lon']]
        result = np.zeros((len(metadata), 14), dtype=np.float32)
        result[:, :2] = coordinate_scaler.transform(values).astype(np.float32)
    else:
        result = static_inputs(metadata, mapping, table, static_scaler, True)
        if group == 'S_LC':
            result[:, 11:] = 0
        else:
            result[:, :11] = 0
    if result.shape != (len(metadata), 14) or not np.isfinite(result).all():
        raise ValueError('Invalid P3b static tensor.')
    return result


def runtime_versions():
    import importlib.metadata
    return {name: importlib.metadata.version(name)
            for name in ['tensorflow', 'keras', 'numpy', 'pandas', 'scikit-learn']}


def validate_config(config):
    parent_path = project_path(config['parent_config'])
    if sha256(parent_path) != config['parent_config_sha256']:
        raise ValueError('Frozen parent configuration changed.')
    parent = json.loads(parent_path.read_text())
    keys = ['protocol', 'fold_manifest', 'spatial_dir', 'static_dir', 'dynamic_groups',
            'lookback', 'pred_len', 'lr', 'batch_size', 'epochs', 'patience', 'architecture',
            'direction_enabled', 'direction_selection', 'direction_selection_sha256',
            'static_ready_sha256', 'static_completion_sha256', 'data_sha256']
    if any(config[k] != parent[k] for k in keys):
        raise ValueError('P3b changes a matched P3 input or training setting.')
    if (config['groups'] != list(GROUPS) or config['seeds'] != [42, 43, 44]
            or config['folds'] != ['F01', 'F02', 'F03'] or config['direction_enabled']
            or config['protocol'] != 'v2_fixed' or config['lookback'] != 56 or config['pred_len'] != 1):
        raise ValueError('P3b v1 fixes 27 tasks, spatial inputs and a 3h direct target.')
    manifest = project_path(config['reference_manifest'])
    if sha256(manifest) != config['reference_manifest_sha256']:
        raise ValueError('P3 reuse manifest changed.')
    sources = json.loads(manifest.read_text())['references']
    identities = [(r['seed'], r['fold_id'], r['group']) for r in sources]
    expected = set(product(config['seeds'], config['folds'], ['RS_S0', 'RS_S1']))
    if len(sources) != 18 or set(identities) != expected:
        raise ValueError('Require exactly 18 frozen P3 spatial references.')
    return sources


def validate_reference_metadata(config, source, marker, actual, result, versions):
    """Check scientific compatibility without equating new and legacy code hashes."""
    fp = marker['fingerprint']
    if (marker['status'] != 'completed' or fp['config_sha256'] != config['parent_config_sha256']
            or fp['runtime_versions'] != versions
            or fp['fold_sha256'] != sha256(project_path(config['fold_manifest']))
            or fp['code'] != code_hashes(list(fp['code']))):
        raise ValueError('Reference environment, protocol or frozen code changed.')
    for key in ['static_ready_sha256', 'static_completion_sha256', 'direction_selection_sha256']:
        if fp[key] != config[key]:
            raise ValueError(f'P3 reference mismatch: {key}')
    if fp['dynamic_data_sha256'] != config['data_sha256']['A01']:
        raise ValueError('Reference uses different dynamic data.')
    for key in ['lookback', 'pred_len', 'lr', 'batch_size', 'epochs', 'patience', 'architecture']:
        if actual[key] != config[key]:
            raise ValueError(f'P3 training mismatch: {key}')
    enabled = source['group'] == 'RS_S1'
    if (actual['dynamic_features'] != config['dynamic_groups']['A01']['features']
            or actual['static_features'] != list(STATIC_FEATURES)
            or actual['dynamic_shape'] != [56, 11] or actual['static_shape'] != [14]
            or result['counts'] != [42400, 22300, 22300] or result['unique_grids'] != 98
            or result['parameter_count'] != 20761 or result['static_enabled'] != enabled):
        raise ValueError('P3 reference structure or coverage changed.')
    for key in ['run_id', 'fold_id', 'seed', 'group']:
        if actual[key] != source[key] or result[key] != source[key]:
            raise ValueError(f'P3 reference identity mismatch: {key}')


def references(config, sources, fold=None, seed=None, full_artifacts=False):
    """Metadata-only preflight on login nodes; full artifact checks within Slurm."""
    selected = [r for r in sources if (fold is None or r['fold_id'] == fold)
                and (seed is None or r['seed'] == seed)]
    result = {}
    for source in selected:
        path = project_path(source['run_dir'])
        if sha256(path/'completion.json') != source['completion_sha256']:
            raise ValueError('P3 reference completion marker changed.')
        marker = verify_artifacts(path) if full_artifacts else json.loads((path/'completion.json').read_text())
        for name in ['config.json', 'result.json']:
            if sha256(path/name) != marker['artifacts'][name]:
                raise ValueError('P3 reference metadata changed.')
        actual = json.loads((path/'config.json').read_text())
        saved = json.loads((path/'result.json').read_text())
        validate_reference_metadata(config, source, marker, actual, saved, runtime_versions())
        result[source['run_id']] = dict(source=source, marker=marker, config=actual, result=saved, path=path)
    return result


def require_signatures(signatures, reused):
    for record in reused.values():
        if signatures != record['result']['sample_signatures']:
            raise ValueError('P3b samples/labels/Persistence differ from P3 before fitting.')
