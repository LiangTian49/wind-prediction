"""Shared validation, feature ordering and atomic output for direction experiments."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from temporal_folds import project_path

GROUPS = {'D_T0': ('T01', False), 'D_T1': ('T01', True),
          'D_S0': ('A01', False), 'D_S1': ('A01', True)}
DATA_DIR = 'data/CERRA/direction_rs_hpc_v3'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def wind_components(speed, direction):
    speed, direction = np.broadcast_arrays(speed, direction)
    if (not np.isfinite(speed).all() or not np.isfinite(direction).all()
            or (speed < 0).any() or (direction < 0).any() or (direction > 360).any()):
        raise ValueError('Require finite nonnegative speed and direction in [0,360].')
    radians = np.deg2rad(direction)
    return -speed * np.sin(radians), -speed * np.cos(radians)


def direction_features(speed_features):
    return [name for speed in speed_features
            for name in (speed, speed.replace('ws_10', 'u_10', 1), speed.replace('ws_10', 'v_10', 1))
            ] + ['hour_sin', 'hour_cos']


def mask_direction(array, features, enabled):
    if array.shape[-1] != len(features):
        raise ValueError('Feature dimension mismatch.')
    if not enabled:
        array = array.copy()
        indices = [i for i, name in enumerate(features) if name.startswith(('u_10', 'v_10'))]
        array[..., indices] = 0
    return array


def verify_artifacts(directory):
    marker = json.loads((directory / 'completion.json').read_text())
    for name, expected in marker['artifacts'].items():
        path = project_path(directory / name)
        if sha256(path) != expected:
            raise ValueError(f'Artifact changed: {path}')
    return marker


@contextmanager
def output_transaction(final, fingerprint):
    """Lock one final path; preserve partial attempts and publish by directory rename."""
    final = project_path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    with final.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if final.exists():
            marker = verify_artifacts(final)
            if marker['fingerprint'] != fingerprint:
                raise ValueError(f'Existing version differs: {final}')
            yield None
            return
        work = Path(tempfile.mkdtemp(prefix=f'.{final.name}_', dir=final.parent))
        yield work
        files = sorted(p for p in work.iterdir() if p.is_file())
        marker = dict(status='completed', fingerprint=fingerprint,
                      artifacts={p.name: sha256(p) for p in files})
        (work / 'completion.json').write_text(json.dumps(marker, indent=2))
        work.rename(final)


def verify_direction_data(directory):
    directory = project_path(directory)
    marker = verify_artifacts(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    if metadata['status'] != 'complete' or metadata['time_steps'] != 2928:
        raise ValueError('Direction extraction is incomplete.')
    for name, expected in metadata['legacy_sources'].items():
        if sha256(project_path(name)) != expected:
            raise ValueError(f'Legacy mapping/data changed: {name}')
    return metadata, marker


def code_hashes(names):
    return {name: sha256(project_path(name)) for name in names}
