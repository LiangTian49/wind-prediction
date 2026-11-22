"""Small checks for the new baseline and planned static feature contracts."""

import json
import unittest

import numpy as np
import pandas as pd

from rs_static_contract import RS_GROUPS, STATIC_FEATURES, fit_static, static_inputs
from temporal_folds import project_path
from train_p1_baseline import CONFIG, make_model, select_model, tasks, verify_reference


class BaselineTests(unittest.TestCase):
    def test_config_counts_and_explicit_validation(self):
        config = json.loads(project_path(CONFIG).read_text())
        plan = tasks(config)
        self.assertEqual(len(set(plan)), 48)
        self.assertEqual(sum(len(config['models'][task[2]]) for task in plan), 144)
        self.assertTrue(all(task[0] != 'F04' for task in plan))
        self.assertFalse(make_model('hist_gradient_boosting', {}, 42).early_stopping)
        x = np.arange(20, dtype=float).reshape(-1, 1)
        best, trials = select_model('ridge', [{'alpha': 100}, {'alpha': .1}], 42,
                                    x[:14], x[:14, 0], x[14:], x[14:, 0])
        self.assertEqual(best[2], {'alpha': .1})
        self.assertEqual(len(trials), 2)

    def test_reference_alignment_detects_wrong_point_persistence(self):
        frame = pd.DataFrame(dict(point_id=[1, 2], forecast_origin=[1, 1],
                                  target_time=[2, 2], y_true=[5., 8.],
                                  persistence_pred=[4., 6.]))
        verify_reference(frame, frame.iloc[::-1])
        changed = frame.assign(persistence_pred=[6., 4.])
        with self.assertRaises(AssertionError):
            verify_reference(frame, changed)

    def test_static_joins_and_training_only_scaling(self):
        table = pd.DataFrame([[0.]*14, [2.]*14, [1000.]*14], columns=STATIC_FEATURES)
        table['grid_id'] = ['a', 'b', 'held_out']
        mapping = pd.DataFrame(dict(point_id=[2, 0, 1], grid_id=['b', 'a', 'a']))
        windows = pd.DataFrame(dict(point_id=[1, 2, 0, 1]))
        scaler = fit_static(table, ['a', 'b', 'a'])
        np.testing.assert_array_equal(scaler.mean_, np.ones(14))
        enabled = static_inputs(windows, mapping, table.iloc[::-1], scaler, True)
        disabled = static_inputs(windows, None, None, None, False)
        np.testing.assert_array_equal(enabled[:, 0], [-1, 1, -1, -1])
        np.testing.assert_array_equal(disabled, np.zeros_like(enabled))
        self.assertEqual(RS_GROUPS['RS_T0'][0], RS_GROUPS['RS_T1'][0])
        self.assertEqual(RS_GROUPS['RS_S0'][0], RS_GROUPS['RS_S1'][0])


if __name__ == '__main__':
    unittest.main()
