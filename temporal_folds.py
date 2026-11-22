"""Fixed-date fold definitions and metadata-only boundary checks (no model IO)."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
STEP = pd.Timedelta(hours=3)
DATE_FIELDS = ('train_start', 'validation_start', 'test_start', 'test_end_exclusive')


def project_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Path must remain inside project: {path}')
    return path


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    train_start: str
    validation_start: str
    test_start: str
    test_end_exclusive: str
    role: str = 'development'
    season_representative: str = ''

    def __post_init__(self):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', self.fold_id):
            raise ValueError('fold_id must contain only letters, digits, underscores or hyphens.')
        if self.role not in {'development', 'locked_confirmation'}:
            raise ValueError(f'Unknown fold role: {self.role}')
        dates = [pd.Timestamp(getattr(self, field)) for field in DATE_FIELDS]
        if any(pd.isna(date) or date.tzinfo is not None or date != date.floor('3h') for date in dates):
            raise ValueError('Fold dates must be non-missing, timezone-naive UTC on the 3-hour grid.')
        if not all(a < b for a, b in zip(dates, dates[1:])):
            raise ValueError('Require train_start < validation_start < test_start < test_end_exclusive.')
        for field, date in zip(DATE_FIELDS, dates):
            object.__setattr__(self, field, str(date))

    @classmethod
    def from_dict(cls, values):
        missing = set(('fold_id', *DATE_FIELDS)).difference(values)
        if missing:
            raise ValueError(f'Missing fold fields: {sorted(missing)}')
        fields = ('fold_id', *DATE_FIELDS, 'role', 'season_representative')
        return cls(**{field: values[field] for field in fields if field in values})

    def as_dict(self):
        return {field: getattr(self, field) for field in
                ('fold_id', *DATE_FIELDS, 'role', 'season_representative')}


def load_manifest(path):
    path = project_path(path)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    folds = [FoldSpec.from_dict(row) for row in manifest['folds']]
    if not folds or len({fold.fold_id for fold in folds}) != len(folds):
        raise ValueError('Manifest needs nonempty folds with unique fold_id values.')
    for fold in folds:
        dates = [pd.Timestamp(getattr(fold, field)) for field in DATE_FIELDS]
        for key, a, b in zip(('train_days', 'validation_days', 'test_days'), dates, dates[1:]):
            if key in manifest and b - a != pd.Timedelta(days=manifest[key]):
                raise ValueError(f'{fold.fold_id}: dates do not match {key}.')
    ordered = sorted(folds, key=lambda fold: pd.Timestamp(fold.test_start))
    if any(pd.Timestamp(a.test_end_exclusive) > pd.Timestamp(b.test_start)
           for a, b in zip(ordered, ordered[1:])):
        raise ValueError('Test windows of fixed folds must not overlap.')
    return manifest, {fold.fold_id: fold for fold in folds}, hashlib.sha256(raw).hexdigest()


def select_fold_panel(df, fold):
    """Require complete observations for every original point throughout the fold."""
    start, end = pd.Timestamp(fold.train_start), pd.Timestamp(fold.test_end_exclusive)
    selected = df[(df['datetime'] >= start) & (df['datetime'] < end)].copy()
    expected = pd.date_range(start, end, freq='3h', inclusive='left').to_numpy()
    all_points = set(df['point_id'].unique())
    if not all_points or set(selected['point_id'].unique()) != all_points:
        raise ValueError(f'{fold.fold_id}: points are absent from the fixed-date interval.')
    for point_id, group in selected.groupby('point_id'):
        dates = group['datetime'].sort_values().to_numpy()
        if not np.array_equal(dates, expected):
            raise ValueError(f'{fold.fold_id}: point {point_id} has missing/duplicate/out-of-grid timestamps.')
    return selected.sort_values(['point_id', 'datetime']).reset_index(drop=True)


def expected_fold_counts(fold, input_len, horizon, n_points):
    if min(input_len, horizon, n_points) <= 0:
        raise ValueError('input_len, horizon and point count must be positive.')
    dates = [pd.Timestamp(getattr(fold, field)) for field in DATE_FIELDS]
    counts = [int((b - a) / STEP) for a, b in zip(dates, dates[1:])]
    samples = [counts[0] - input_len - horizon + 1, counts[1] - horizon, counts[2] - horizon]
    if min(samples) <= 0:
        raise ValueError(f'{fold.fold_id}: insufficient training/validation/test history.')
    return dict(zip(('train_samples', 'val_samples', 'test_samples'),
                    (count * n_points for count in samples)))


def verify_readiness(spatial_dir, manifest_path):
    """Cheap pre-submission/job check; never reads weather arrays."""
    spatial_dir = project_path(spatial_dir)
    marker = spatial_dir / 'READY.json'
    if not marker.is_file():
        raise ValueError('Missing READY.json; run annual extraction and readiness validation first.')
    report = json.loads(marker.read_text())
    _, _, digest = load_manifest(manifest_path)
    if report.get('status') != 'passed' or report.get('fold_manifest_sha256') != digest:
        raise ValueError('Annual readiness marker does not match the current fold manifest.')
    for filename, expected in report['source_stats'].items():
        path = project_path(spatial_dir / filename)
        if path.parent != spatial_dir:
            raise ValueError('Invalid source path in readiness marker.')
        stat = path.stat()
        if stat.st_size != expected['bytes'] or stat.st_mtime_ns != expected['mtime_ns']:
            raise ValueError(f'Annual data changed after readiness validation: {filename}')
    return report
