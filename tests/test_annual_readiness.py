"""Tiny synthetic annual panels; no real GRIB or model execution."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from extract_cerra_spatial import offset_name
from temporal_folds import verify_readiness
from validate_annual_data import validate_annual


ROOT = Path(__file__).resolve().parents[1]


class AnnualReadinessTests(unittest.TestCase):
    def test_annual_panel_all_folds_and_changed_source_detection(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            spatial = root / 'annual'
            spatial.mkdir()
            dates = pd.date_range('2024-01-01', '2025-01-01', freq='3h', inclusive='left')
            centers = pd.DataFrame({'point_id': [0, 1], 'target_lat': [50., 51.],
                                    'target_lon': [9., 10.], 'center_y': [100, 120], 'center_x': [100, 120],
                                    'grid_lat': [50., 51.], 'grid_lon': [9., 10.]})
            centers.to_csv(spatial / 'centers.csv', index=False)
            reference_centers = root / 'old_centers.csv'
            centers.to_csv(reference_centers, index=False)
            local = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]
            far = [(dy, dx) for dy in (-10, 0, 10) for dx in (-10, 0, 10) if (dy, dx) != (0, 0)]
            names = {offset: 'ws_10' if offset == (0, 0) else offset_name(*offset)
                     for offset in local + far}
            values = np.empty((2, len(dates), 5, 5), dtype=np.float32)
            for point in (0, 1):
                for index, _ in enumerate(local):
                    values[point, :, index // 5, index % 5] = (
                        2 + point + np.arange(len(dates)) / len(dates) + index / 100)
            center = pd.concat([pd.DataFrame({'datetime': dates, 'point_id': point,
                                              'ws_10': values[point, :, 2, 2]})
                                for point in (0, 1)], ignore_index=True)
            center.to_csv(spatial / 'center.csv', index=False)
            reference = root / 'summer.csv'
            center[center.datetime.dt.month.isin([6, 7, 8])].to_csv(reference, index=False)
            pd.DataFrame({'point_id': [0, 1], 'kind': ['local', 'local'],
                          'dy': [0, 0], 'dx': [0, 0]}).to_csv(spatial / 'grid_mapping.csv', index=False)
            features = {'T01': ['ws_10', 'hour_sin', 'hour_cos']}
            for experiment, offsets in [('S01', [(dy, dx) for dy, dx in local if abs(dy) <= 1 and abs(dx) <= 1]),
                                         ('S02', local), ('A01', [(0, 0)] + far)]:
                frames = []
                for point in (0, 1):
                    frame = pd.DataFrame({'datetime': dates, 'point_id': point})
                    for offset in offsets:
                        if offset in local:
                            index = local.index(offset)
                            frame[names[offset]] = values[point, :, index // 5, index % 5]
                        else:
                            frame[names[offset]] = values[point, :, 2, 2] + 0.5
                    frames.append(frame)
                pd.concat(frames, ignore_index=True).to_csv(spatial / f'{experiment}.csv', index=False)
                features[experiment] = [names[offset] for offset in offsets] + ['hour_sin', 'hour_cos']
            np.savez_compressed(spatial / 'neighborhoods.npz', ws_10=values, datetime=dates.to_numpy(),
                                point_id=np.array([0, 1]), dy=np.arange(-2, 3), dx=np.arange(-2, 3))
            metadata = {'status': 'complete', 'year': 2024, 'months': list(range(1, 13)),
                        'require_complete_months': True, 'units': 'm s**-1', 'radius': 2,
                        'reference_csv': str(reference), 'reference_check': {'matched_rows': 1472},
                        'feature_sets': features}
            (spatial / 'metadata.json').write_text(json.dumps(metadata))
            manifest = json.loads((ROOT / 'configs/seasonal_folds_2024.json').read_text())
            manifest['point_mapping'] = str(reference_centers)
            path = root / 'manifest.json'
            path.write_text(json.dumps(manifest))
            report, coverage = validate_annual(spatial, path, root / 'report')
            self.assertEqual(report['time_steps_per_point'], 2928)
            self.assertEqual(report['rows_per_feature_set'], 5856)
            self.assertEqual((root / 'report/fold_manifest.json').read_bytes(), path.read_bytes())
            self.assertEqual(len(coverage), 16)
            counts = coverage[(coverage.fold_id == 'F01') & (coverage.lead_hours == 3)].iloc[0]
            self.assertEqual((counts.train_samples, counts.val_samples, counts.test_samples), (848, 446, 446))
            self.assertEqual(verify_readiness(spatial, path)['status'], 'passed')
            # A changed file must invalidate the marker before any training runs.
            with (spatial / 'metadata.json').open('a') as handle:
                handle.write('\n')
            with self.assertRaisesRegex(ValueError, 'changed after'):
                verify_readiness(spatial, path)


if __name__ == '__main__':
    unittest.main()
