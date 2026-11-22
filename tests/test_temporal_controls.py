"""Synthetic sample matching, label embargo and lead-alignment tests; no real fitting."""
import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from temporal_controls import (control_partitions, load_config, matched_reference_metrics, tasks)
from temporal_folds import FoldSpec
from train import prediction_frame
from train_p3 import build_partitions


class TemporalControlChecks(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range('2024-01-01',periods=400,freq='3h')
        self.df = pd.concat([pd.DataFrame(dict(point_id=p,datetime=self.dates,
                    ws_10=np.arange(400,dtype=float)+p*1000,
                    hour_sin=np.sin(np.arange(400)),hour_cos=np.cos(np.arange(400)))) for p in [0,1]],ignore_index=True)
        self.features = ['ws_10','hour_sin','hour_cos']
        self.fold = FoldSpec('X',*(str(self.dates[k]) for k in [0,128,256,384]))

    def build(self,length,horizon=1,df=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return control_partitions(self.df if df is None else df,self.features,self.fold,length,horizon)

    def test_all_partitions_same_labels_origins_counts_and_true_input_start(self):
        full,scaler,_,anchor = self.build(56)
        for length in [8,24]:
            short,other_scaler,_,signatures = self.build(length)
            self.assertEqual(signatures,anchor)
            np.testing.assert_array_equal(other_scaler.data_max_,scaler.data_max_)
            for partition in full:
                x,y,persistence,metadata = short[partition]
                np.testing.assert_array_equal(x,full[partition][0][:,-length:,:])
                np.testing.assert_array_equal(y,full[partition][1])
                np.testing.assert_array_equal(persistence,full[partition][2])
                pd.testing.assert_frame_equal(metadata[['point_id','forecast_origin','target_time']],
                                              full[partition][3][['point_id','forecast_origin','target_time']])
                self.assertTrue(((metadata.forecast_origin-metadata.input_start)==pd.Timedelta(hours=3*(length-1))).all())
                source = self.df.set_index(['point_id','datetime']).ws_10.reindex(
                    pd.MultiIndex.from_arrays([metadata.point_id,metadata.forecast_origin])).to_numpy()
                np.testing.assert_array_equal(persistence,source)

    def test_derived_6h_exactly_matches_native_window_builder(self):
        samples,scaler,_,anchor = self.build(56,2)
        with contextlib.redirect_stdout(io.StringIO()):
            native,other_scaler,_ = build_partitions(self.df,self.features,self.fold,56,2)
        _,_,_,anchor3h = self.build(56,1)
        self.assertEqual(anchor,anchor3h)
        np.testing.assert_array_equal(scaler.data_min_,other_scaler.data_min_)
        for name in samples:
            for i in [0,1,2]:
                np.testing.assert_array_equal(samples[name][i],native[name][i])
            pd.testing.assert_frame_equal(samples[name][3],native[name][3])
            self.assertTrue(((samples[name][3].target_time-samples[name][3].forecast_origin)==pd.Timedelta(hours=6)).all())
        self.assertLess(samples['train'][3].target_time.max(),self.dates[128])
        self.assertLess(samples['validation'][3].target_time.max(),self.dates[256])
        self.assertLess(samples['test'][3].target_time.max(),self.dates[384])

    def test_future_changes_do_not_change_train_validation_or_scaling(self):
        original,scaler,_,_ = self.build(8)
        changed = self.df.copy()
        changed.loc[changed.datetime>self.dates[256],self.features]+=10000
        future,other_scaler,_,_ = self.build(8,df=changed)
        for part in ['train','validation']:
            for k in [0,1,2]:
                np.testing.assert_array_equal(original[part][k],future[part][k])
        np.testing.assert_array_equal(scaler.data_max_,other_scaler.data_max_)
        np.testing.assert_array_equal(original['test'][0][0],future['test'][0][0])

    def test_origin_matching_uses_distinct_targets_and_rejects_bad_persistence(self):
        short,_,_,_ = self.build(8)
        anchor,_,_,_ = self.build(56)
        six,_,_,_ = self.build(56,2)
        def frame(samples):
            _,y,p,m = samples['test']
            result=prediction_frame(m,y,p,p,'synthetic',42,'RS_S0')
            result['center_y']=result.point_id
            result['center_x']=0
            result['grid_lat']=50.+result.point_id
            result['grid_lon']=10.
            return result
        reference=frame(anchor)
        matched_reference_metrics(frame(short),reference,1)
        score=matched_reference_metrics(frame(six),reference,2)
        self.assertEqual(score['raw_samples'],len(six['test'][1]))
        wrong=reference.copy()
        wrong.loc[0,'persistence_pred']+=1
        with self.assertRaises(AssertionError):
            matched_reference_metrics(frame(six),wrong,2)
        wrong=reference.copy()
        wrong['target_time']+=pd.Timedelta(hours=3)
        with self.assertRaises(ValueError):
            matched_reference_metrics(frame(six),wrong,2)

    def test_registered_budget_and_stage_order(self):
        history,_,_=load_config('history')
        horizon,_,_=load_config('horizon')
        plan=tasks(history)
        self.assertEqual(len(set(plan)),36)
        self.assertEqual(plan[:4],[(42,'F01',8,'RS_S0'),(42,'F01',8,'RS_S1'),
                                  (42,'F01',24,'RS_S0'),(42,'F01',24,'RS_S1')])
        self.assertEqual(len(tasks(horizon)),18)
        self.assertEqual(tasks(horizon)[:2],[(42,'F01',56,'RS_S0'),(42,'F01',56,'RS_S1')])
        with self.assertRaises(ValueError):
            self.build(8,2)

    def test_6h_gate_rejects_missing_history_stage(self):
        import tempfile
        from unittest.mock import patch
        from temporal_controls import OUTPUTS, require_history_complete
        from temporal_folds import project_path

        with tempfile.TemporaryDirectory(dir=project_path('.')) as directory:
            with patch.dict(OUTPUTS,history=directory):
                with self.assertRaises(ValueError):
                    require_history_complete()

    def test_short_history_models_keep_parameters_and_saved_predictions(self):
        import tempfile
        from pathlib import Path
        import tensorflow as tf
        from rs_static_contract import build_fusion
        from temporal_folds import project_path

        for length in [8,24]:
            tf.keras.utils.set_random_seed(42)
            model=build_fusion((length,11))
            self.assertEqual(model.count_params(),20761)
            dynamic=np.ones((2,length,11),dtype=np.float32)
            static=np.zeros((2,14),dtype=np.float32)
            prediction=model({'dynamic':dynamic,'static':static},training=False).numpy()
            self.assertTrue(np.isfinite(prediction).all())
            with tempfile.TemporaryDirectory(dir=project_path('.')) as directory:
                path=Path(directory)/'synthetic.keras'
                model.save(path)
                loaded=tf.keras.models.load_model(path,compile=False)
                np.testing.assert_allclose(loaded({'dynamic':dynamic,'static':static},training=False),prediction)
