"""Small synthetic protocol checks; no TensorFlow fits, GRIB scan or benchmarks."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from confirmation_protocol import (GROUPS, block_weights, check_samples, load_frozen,
                                  make_folds, paired_gain, static_array, task_list, validate_product)
from p3b_contract import fit_coordinates
from rs_static_contract import STATIC_FEATURES, fit_static
from temporal_folds import FoldSpec, expected_fold_counts
from train_confirmation_year import verify_labels_and_persistence
from train_p3 import build_partitions


class ConfirmationTests(unittest.TestCase):
    def test_calendar_and_45_task_identity(self):
        for year in (2023, 2025):
            folds = make_folds(year)
            self.assertEqual(pd.Timestamp(folds[0]['validation_start'])-
                             pd.Timestamp(folds[0]['train_start']), pd.Timedelta(days=60))
            self.assertTrue(folds[0]['train_start'].startswith(f'{year}-01-03'))
            self.assertEqual(list(expected_fold_counts(FoldSpec.from_dict(folds[0]), 56, 1, 100).values()),
                             [42400, 22300, 22300])
            tasks = task_list(dict(folds=folds))
            self.assertEqual(len(set(tasks)), 45)
            self.assertEqual([v[2] for v in tasks[:5]], GROUPS)
        with self.assertRaises(ValueError):
            make_folds(2024)

    def test_product_rejects_forecast_height_units_and_scanning(self):
        from confirmation_protocol import GRID_KEYS
        grid = {k: 1 for k in GRID_KEYS}
        grid.update(scanningMode=64, units='m s**-1')
        header = dict(grid, paramId=207, dataType='an', typeOfLevel='heightAboveGround',
                      level=10, stepType='instant', stepRange='0')
        validate_product(header, grid)
        for key, value in [('dataType', 'fc'), ('level', 100), ('units', 'knots'), ('stepRange', '3'),
                           ('Nx', 2), ('scanningMode', 96)]:
            with self.assertRaises(ValueError):
                validate_product(dict(header, **{key: value}), grid)

    def test_causal_windows_scaler_and_pointwise_persistence(self):
        fold = FoldSpec('tiny', '2025-01-01', '2025-01-01 12:00',
                        '2025-01-01 18:00', '2025-01-02', role='locked_confirmation')
        dates = pd.date_range('2025-01-01', periods=8, freq='3h')
        features = ['ws_10'] + [f'n{i}' for i in range(8)] + ['hour_sin', 'hour_cos']
        rows = []
        for point in (0, 1):
            for i, time in enumerate(dates):
                rows.append(dict(datetime=time, point_id=point,
                                 **{name: point*100+i for name in features}))
        df = pd.DataFrame(rows)
        samples, scaler, _ = build_partitions(df, features, fold, 2, 1)
        self.assertEqual(check_samples(samples, fold, 2, 1, n_points=2), [4, 2, 2])
        verify_labels_and_persistence(df, samples)
        self.assertEqual(scaler.data_max_[0], 103)
        damaged = copy.deepcopy(samples)
        damaged['test'][2][1] = damaged['test'][2][0]
        with self.assertRaises(AssertionError):
            verify_labels_and_persistence(df, damaged)
        damaged = copy.deepcopy(samples)
        damaged['validation'][3].loc[0, 'target_time'] = pd.Timestamp(fold.test_start)
        with self.assertRaises(ValueError):
            check_samples(damaged, fold, 2, 1, n_points=2)

    def test_static_masks_and_point_join_order(self):
        table = pd.DataFrame([dict(grid_id=f'g{i}', **{f: i+j for j, f in enumerate(STATIC_FEATURES)})
                              for i in range(3)])
        mapping = pd.DataFrame(dict(point_id=[0, 1, 2, 3], grid_id=['g0', 'g1', 'g2', 'g0'],
                                    grid_lat=[50, 51, 52, 50], grid_lon=[8, 9, 10, 8]))
        static = fit_static(table, ['g0', 'g1', 'g2', 'g0'])
        coord = fit_coordinates(mapping, ['g0', 'g1', 'g2'])
        metadata = pd.DataFrame(dict(point_id=[2, 3, 1]))
        arrays = {g: static_array(metadata, mapping, table, static, coord, g) for g in GROUPS}
        np.testing.assert_array_equal(arrays['RS_S0'], np.zeros((3, 14)))
        np.testing.assert_array_equal(arrays['S_LC'][:, 11:], 0)
        np.testing.assert_array_equal(arrays['S_DSM'][:, :11], 0)
        np.testing.assert_array_equal(arrays['S_COORD'][:, 2:], 0)
        self.assertGreater(arrays['RS_S1'][0, 0], arrays['RS_S1'][1, 0])
        self.assertEqual(float(static.mean_[0]), 1)

    def test_date_pairing_equal_window_estimand_and_zero_reference(self):
        weights = block_weights(np.random.default_rng(1), 4, 10, 3)
        np.testing.assert_array_equal(weights.sum(axis=1), 4)
        point, draws = paired_gain([1, 1, 1, 1], [4, 4, 4, 4], weights, 'RMSE')
        self.assertEqual(point, 50)
        np.testing.assert_array_equal(draws, 50)
        # Equal-window gains differ from pooling losses across windows.
        a = paired_gain([1]*4, [4]*4, weights, 'RMSE')[0]
        b = paired_gain([81]*4, [100]*4, weights, 'RMSE')[0]
        self.assertAlmostEqual(np.mean([a, b]), 30)
        with self.assertRaises(ValueError):
            paired_gain([1]*4, [0]*4, weights, 'RMSE')

    def test_draft_and_mutated_frozen_request_rejected(self):
        with patch('confirmation_protocol.validate_request', return_value={}), \
             patch('confirmation_protocol.read_json', side_effect=[{}, {'status': 'draft_not_frozen'}]):
            with self.assertRaises(ValueError):
                load_frozen('configs/confirmation_v1/request.json')
        with patch('confirmation_protocol.validate_request', return_value={}), \
             patch('confirmation_protocol.read_json', side_effect=[{}, {'status': 'frozen', 'request_sha256': 'old'}]), \
             patch('confirmation_protocol.sha256', return_value='changed'):
            with self.assertRaises(ValueError):
                load_frozen('configs/confirmation_v1/request.json')

    def test_real_jobs_rejected_on_login(self):
        from prepare_confirmation_year import extract
        from argparse import Namespace
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'sbatch'):
                extract(Namespace())


if __name__ == '__main__':
    unittest.main()
