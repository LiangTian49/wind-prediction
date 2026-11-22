"""Lightweight causal-window and same-architecture checks for P3."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rs_static_contract import STATIC_FEATURES, build_fusion, fit_static, static_inputs
from temporal_folds import FoldSpec, project_path
from train import create_dataset
from train_p3 import CONFIG, build_partitions, sample_signature, tasks


class P3Checks(unittest.TestCase):
    def test_new_partitions_equal_legacy_and_ignore_future_changes(self):
        dates = pd.date_range('2024-01-01', periods=80, freq='3h')
        frame = pd.concat([pd.DataFrame(dict(point_id=point, datetime=dates, ws_10=np.arange(80)+point*100,
                                            hour_sin=np.sin(np.arange(80)), hour_cos=np.cos(np.arange(80))))
                           for point in [0, 1]], ignore_index=True)
        fold = FoldSpec('X', *(str(dates[i]) for i in [8, 32, 48, 64]))
        features = ['ws_10', 'hour_sin', 'hour_cos']
        with contextlib.redirect_stdout(io.StringIO()):
            samples, scaler, _ = build_partitions(frame, features, fold, 4, 2)
            legacy = create_dataset(frame, 4, 2, features, include_metadata=True, protocol='v2_fixed', fold=fold)
            changed = frame.copy()
            changed.loc[changed.datetime > dates[48], features] += 10000
            future, future_scaler, _ = build_partitions(changed, features, fold, 4, 2)
        for i, name in enumerate(['train', 'validation', 'test']):
            np.testing.assert_array_equal(samples[name][0], legacy[2*i])
            np.testing.assert_array_equal(samples[name][1], legacy[2*i+1])
            metadata = samples[name][3]
            expected = frame.set_index(['point_id', 'datetime']).ws_10.reindex(
                pd.MultiIndex.from_arrays([metadata.point_id, metadata.forecast_origin])).to_numpy()
            np.testing.assert_array_equal(samples[name][2], expected)
        pd.testing.assert_frame_equal(samples['test'][3], legacy[9])
        for name in ['train', 'validation']:
            np.testing.assert_array_equal(samples[name][0], future[name][0])
            np.testing.assert_array_equal(samples[name][1], future[name][1])
        np.testing.assert_array_equal(samples['test'][0][0], future['test'][0][0])
        np.testing.assert_array_equal(scaler.data_max_, future_scaler.data_max_)
        self.assertLess(samples['train'][3].target_time.max(), dates[32])
        self.assertLess(samples['validation'][3].target_time.max(), dates[48])

    def test_budget_and_sample_signature(self):
        config = json.loads(project_path(CONFIG).read_text())
        plan = tasks(config)
        self.assertEqual(len(set(plan)), 60)
        self.assertEqual(plan[:4], [(42, 'F01', name) for name in ['RS_T0', 'RS_T1', 'RS_S0', 'RS_S1']])
        self.assertEqual({seed for seed, _, _ in plan[:36]}, {42, 43, 44})
        self.assertEqual({seed for seed, _, _ in plan[36:]}, {45, 46})
        self.assertNotIn('F04', {fold for _, fold, _ in plan})
        metadata = pd.DataFrame(dict(point_id=[0, 1], forecast_origin=[1, 1], target_time=[2, 2]))
        signature = sample_signature(metadata, [1., 2.], [0., 1.])
        self.assertNotEqual(signature, sample_signature(metadata, [1., 3.], [0., 1.]))

    def test_train_grid_scaling_keyed_join_and_disabled_information(self):
        table = pd.DataFrame([[0.]*14, [2.]*14, [1000.]*14], columns=STATIC_FEATURES)
        table['grid_id'] = ['a', 'b', 'withheld']
        mapping = pd.DataFrame(dict(point_id=[0, 2, 1], grid_id=['a', 'b', 'a']))
        metadata = pd.DataFrame(dict(point_id=[1, 0, 2, 1]))
        scaler = fit_static(table, ['a', 'b', 'a'])
        np.testing.assert_array_equal(scaler.mean_, 1.)
        enabled = static_inputs(metadata, mapping, table.iloc[::-1], scaler, True)
        np.testing.assert_array_equal(enabled[:, 0], [-1, -1, 1, -1])
        disabled = static_inputs(metadata, None, None, None, False)
        np.testing.assert_array_equal(disabled, 0.)
        self.assertEqual(disabled.shape, enabled.shape)

    def test_fusion_inputs_structure_and_saved_model_roundtrip(self):
        import tensorflow as tf

        for channels, parameters in [(3, 18713), (11, 20761)]:
            tf.keras.utils.set_random_seed(42)
            off = build_fusion((56, channels))
            tf.keras.utils.set_random_seed(42)
            on = build_fusion((56, channels))
            self.assertEqual(off.count_params(), parameters)
            self.assertEqual(on.count_params(), parameters)
            self.assertEqual(off.input_shape, on.input_shape)
            self.assertEqual(off.input_shape, [(None, 56, channels), (None, 14)])
            x = np.ones((2, 56, channels), dtype=np.float32)
            zero = static_inputs(pd.DataFrame(dict(point_id=[1, 2])), None, None, None, False)
            prediction = off({'dynamic':x, 'static':zero}, training=False).numpy()
            self.assertEqual(prediction.shape, (2, 1))
            off.compile(optimizer=tf.keras.optimizers.Adam(.001), loss='mse')
            self.assertTrue(np.isfinite(off.test_on_batch({'dynamic':x, 'static':zero}, np.ones(2))))
            with tempfile.TemporaryDirectory(dir=project_path('.')) as directory:
                path = Path(directory)/'synthetic.keras'
                off.save(path)
                loaded = tf.keras.models.load_model(path, compile=False)
                np.testing.assert_allclose(loaded({'dynamic':x, 'static':zero}, training=False), prediction)


if __name__ == '__main__':
    unittest.main()
