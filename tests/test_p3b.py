"""Synthetic P3b representation, provenance and plan checks; no model fitting."""
import copy
import json
import unittest

import numpy as np
import pandas as pd

from p3b_contract import (CONFIG, coordinate_table, fit_coordinates, references,
                          representation, require_signatures, slot_names, tasks, validate_config,
                          validate_reference_metadata, runtime_versions)
from rs_static_contract import STATIC_FEATURES, fit_static, static_inputs
from temporal_folds import project_path


class P3bChecks(unittest.TestCase):
    def setUp(self):
        self.mapping = pd.DataFrame(dict(point_id=[0,1,2,3], grid_id=['a','a','b','held'],
                                        grid_lat=[50.,50.,52.,80.], grid_lon=[8.,8.,12.,100.]))
        self.table = pd.DataFrame([[0.]*14, [2.]*14, [1000.]*14], columns=STATIC_FEATURES)
        self.table['grid_id'] = ['a','b','held']
        self.meta = pd.DataFrame(dict(point_id=[2,1,0,2]))
        self.static = fit_static(self.table, ['a','b','a'])
        self.coords = fit_coordinates(self.mapping, ['a','b','a'])

    def test_keyed_coordinate_join_train_only_and_padding(self):
        np.testing.assert_array_equal(self.coords.mean_, [51,10])
        value = representation(self.meta, self.mapping.iloc[::-1], self.table, self.static, self.coords, 'S_COORD')
        np.testing.assert_array_equal(value[:,:2], [[1,1],[-1,-1],[-1,-1],[1,1]])
        np.testing.assert_array_equal(value[:,2:], 0.)
        changed = self.mapping.copy()
        changed.loc[changed.grid_id.eq('held'), ['grid_lat','grid_lon']] = [89,179]
        again = fit_coordinates(changed, ['a','b'])
        np.testing.assert_array_equal(again.mean_, self.coords.mean_)
        np.testing.assert_array_equal(again.scale_, self.coords.scale_)
        self.assertEqual(slot_names('S_COORD')[:2], ['grid_lat','grid_lon'])

    def test_landcover_and_dsm_masks_after_scaling_preserve_inputs(self):
        full = static_inputs(self.meta, self.mapping, self.table, self.static, True)
        original = self.table.copy(deep=True)
        for group in ['S_LC','S_DSM']:
            value = representation(self.meta, self.mapping, self.table.iloc[::-1], self.static, self.coords, group)
            if group == 'S_LC':
                np.testing.assert_array_equal(value[:,:11], full[:,:11])
                np.testing.assert_array_equal(value[:,11:], 0.)
            else:
                np.testing.assert_array_equal(value[:,:11], 0.)
                np.testing.assert_array_equal(value[:,11:], full[:,11:])
            self.assertEqual(value.dtype, np.float32)
            self.assertEqual(value.shape, (4,14))
            # Grid a's raw zeros standardize to -1; an inactive raw zero would not suffice.
            self.assertTrue((full[1] == -1).all())
        pd.testing.assert_frame_equal(original, self.table)
        np.testing.assert_array_equal(static_inputs(self.meta,None,None,None,False), np.zeros_like(full))

    def test_bad_join_and_coordinates_fail(self):
        bad = self.mapping.copy()
        bad.loc[1,'grid_lat'] = 51
        with self.assertRaises(ValueError):
            coordinate_table(bad)
        bad.loc[1,'grid_lat'] = float('nan')
        with self.assertRaises(ValueError):
            coordinate_table(bad)
        with self.assertRaises(KeyError):
            representation(pd.DataFrame(dict(point_id=[999])), self.mapping, self.table,
                           self.static,self.coords,'S_COORD')
        with self.assertRaises(ValueError):
            slot_names('unknown')

    def test_plan_and_real_metadata_preflight(self):
        config = json.loads(project_path(CONFIG).read_text())
        sources = validate_config(config)
        plan = tasks(config)
        self.assertEqual(len(set(plan)),27)
        self.assertEqual(plan[:3],[(42,'F01',g) for g in ['S_COORD','S_LC','S_DSM']])
        self.assertEqual(plan[-1],(44,'F03','S_DSM'))
        self.assertNotIn('F04',{fold for _,fold,_ in plan})
        records = references(config,sources,fold='F01',seed=42)
        self.assertEqual(len(records),2)
        record = next(iter(records.values()))
        require_signatures(record['result']['sample_signatures'],records)
        changed = dict(record['result']['sample_signatures'],train='changed label or key')
        with self.assertRaises(ValueError):
            require_signatures(changed,records)
        actual = copy.deepcopy(record['config'])
        actual['lr'] *= 2
        with self.assertRaises(ValueError):
            validate_reference_metadata(config,record['source'],record['marker'],actual,record['result'],runtime_versions())
        actual = copy.deepcopy(record['config'])
        actual['dynamic_features'] = list(reversed(actual['dynamic_features']))
        with self.assertRaises(ValueError):
            validate_reference_metadata(config,record['source'],record['marker'],actual,record['result'],runtime_versions())
        invalid = dict(config, lr=.02)
        with self.assertRaises(ValueError):
            validate_config(invalid)

    def test_same_initial_model_forward_and_save_load_for_all_masks(self):
        import tempfile
        from pathlib import Path
        import tensorflow as tf
        from rs_static_contract import build_fusion

        tf.keras.utils.set_random_seed(42)
        model = build_fusion((56, 11))
        self.assertEqual(model.count_params(), 20761)
        self.assertEqual(model.input_shape, [(None, 56, 11), (None, 14)])
        dynamic = np.ones((len(self.meta), 56, 11), dtype=np.float32)
        initial = [weight.numpy().copy() for weight in model.weights]
        for group in ['S_COORD', 'S_LC', 'S_DSM']:
            array = representation(self.meta, self.mapping, self.table, self.static, self.coords, group)
            prediction = model({'dynamic':dynamic, 'static':array}, training=False).numpy()
            self.assertEqual(prediction.shape, (4,1))
            self.assertTrue(np.isfinite(prediction).all())
            with tempfile.TemporaryDirectory(dir=project_path('.')) as directory:
                path = Path(directory)/'synthetic.keras'
                model.save(path)
                loaded = tf.keras.models.load_model(path, compile=False)
                np.testing.assert_allclose(loaded({'dynamic':dynamic,'static':array}, training=False), prediction)
        for before, after in zip(initial, model.weights):
            np.testing.assert_array_equal(before, after.numpy())
