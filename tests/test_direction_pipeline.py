"""Synthetic checks for direction signs, causal inputs, masking and region cases."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from direction_pipeline import (direction_features, mask_direction, output_transaction,
                                verify_artifacts, wind_components)
from extract_cerra_direction import normalize_direction
from plot_p0b import aligned_case, great_circle
from temporal_folds import FoldSpec, project_path
from train import create_dataset
from train_p1c import task_list


class DirectionChecks(unittest.TestCase):
    def test_grib_packing_endpoint_wrap_and_bad_data_rejection(self):
        raw = np.array([0., 90., 360., 360.0001851564739, -0.0002])
        original = raw.copy()
        normalized, count = normalize_direction(raw, 0.003906250007275958)
        self.assertEqual(count, 2)
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_allclose(normalized, [0., 90., 360., .0001851564739, 359.9998], atol=1e-10)
        uu, vv = wind_components(np.full(5, 2.), normalized)
        np.testing.assert_allclose(np.hypot(uu, vv), 2.)
        for value in [361., -1., np.nan, np.inf]:
            with self.assertRaises(ValueError):
                normalize_direction([value], 0.003906250007275958)
        with self.assertRaises(ValueError):
            normalize_direction([360.0001851564739], 0.)

    def test_cardinal_signs_and_magnitude(self):
        u, v = wind_components(np.full(5, 2.), [0, 90, 180, 270, 360])
        np.testing.assert_allclose(u, [0, -2, 0, 2, 0], atol=1e-12)
        np.testing.assert_allclose(v, [-2, 0, 2, 0, -2], atol=1e-12)
        np.testing.assert_allclose(np.hypot(u, v), 2)
        for angle in [-1, 361, np.nan]:
            with self.assertRaises(ValueError):
                wind_components(1, angle)
        np.testing.assert_array_equal(wind_components(0, 100), [-0., 0.])

    def test_mask_after_scaling_and_train_only_causality(self):
        features = direction_features(['ws_10'])
        self.assertEqual(len(features), 5)
        self.assertEqual(len(direction_features([f'ws_10_{i}' for i in range(9)])), 29)
        dates = pd.date_range('2024-01-01', periods=80, freq='3h')
        panel = pd.concat([pd.DataFrame(dict(
            point_id=point, datetime=dates, ws_10=np.arange(80)+point*100,
            u_10=-np.arange(80)-point*100, v_10=np.arange(80)/2,
            hour_sin=np.sin(np.arange(80)), hour_cos=np.cos(np.arange(80))))
            for point in [0, 1]], ignore_index=True)
        fold = FoldSpec('X', *(str(dates[i]) for i in [8, 32, 48, 64]))
        with contextlib.redirect_stdout(io.StringIO()):
            original = create_dataset(panel, 4, 1, features, include_metadata=True,
                                      protocol='v2_fixed', fold=fold)
            changed = panel.copy()
            changed.loc[changed.datetime > dates[48], ['u_10', 'v_10']] += 100000
            future = create_dataset(changed, 4, 1, features, include_metadata=True,
                                    protocol='v2_fixed', fold=fold)
        for index in [0, 1, 2, 3]:
            np.testing.assert_array_equal(original[index], future[index])
        np.testing.assert_array_equal(original[4][0], future[4][0])
        np.testing.assert_array_equal(original[7].data_max_, future[7].data_max_)
        self.assertEqual(original[6][0], 48)
        self.assertEqual(original[6][len(original[6])//2], 148)
        off = mask_direction(original[0], features, False)
        np.testing.assert_array_equal(off[..., 1:3], 0)
        np.testing.assert_array_equal(off[..., [0, 3, 4]], original[0][..., [0, 3, 4]])
        np.testing.assert_array_equal(mask_direction(original[0], features, True), original[0])

    def test_atomic_retry_skip_and_damage(self):
        # Retain this tiny fixture under the project for diagnostic review.
        base = Path(tempfile.mkdtemp(prefix='direction_test_', dir=project_path('result')))
        final = base / 'done'
        with self.assertRaisesRegex(RuntimeError, 'synthetic'):
            with output_transaction(final, {'version': 1}) as work:
                (work / 'value.txt').write_text('partial')
                raise RuntimeError('synthetic interruption')
        self.assertFalse(final.exists())
        with output_transaction(final, {'version': 1}) as work:
            (work / 'value.txt').write_text('complete')
        with output_transaction(final, {'version': 1}) as work:
            self.assertIsNone(work)
        with self.assertRaises(ValueError):
            with output_transaction(final, {'version': 2}):
                pass
        (final / 'value.txt').write_text('changed')
        with self.assertRaises(ValueError):
            verify_artifacts(final)

    def test_common_case_uses_time_not_error(self):
        origins = pd.date_range('2024-04-01', periods=30, freq='3h')
        frame = pd.DataFrame(dict(center_y=np.repeat([1, 2], 30), center_x=1,
                                  forecast_origin=list(origins)*2,
                                  target_time=list(origins+pd.Timedelta(hours=3))*2,
                                  y_true=1., y_pred=100., persistence_pred=2.))
        cases, start = aligned_case({'a': frame, 'b': frame.assign(y_pred=-100.)})
        self.assertEqual(start, origins[0])
        self.assertEqual(len(cases['a']), 48)
        with self.assertRaises(AssertionError):
            aligned_case({'a': frame, 'b': frame.iloc[:-1]})
        self.assertGreater(great_circle(49.5, 8.5, 51.5, 8.5), 220)
        self.assertEqual(great_circle(50, 10, 50, 10), 0)

    def test_task_budget(self):
        config = json.loads(project_path('configs/rs_hpc_v3_p1c.json').read_text())
        tasks = task_list(config)
        self.assertEqual(len(set(tasks)), 12)
        self.assertEqual([task[0] for task in tasks[:4]], ['F01']*4)


if __name__ == '__main__':
    unittest.main()
