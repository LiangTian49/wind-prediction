"""Synthetic origin-date matching and resampling checks; no archive analysis."""
import unittest
import numpy as np
import pandas as pd
from analyze_temporal_controls import (assert_aligned,circular_weights,comparisons,
                                      interaction_keys,label,match_origins,origin_errors)
from analyze_p3_dates import gain_distribution


class TemporalDateChecks(unittest.TestCase):
    def frame(self,length,lead):
        records=[]
        for point in [0,1]:
            for index,origin in enumerate(pd.date_range('2024-04-01',periods=3,freq='3h')):
                wind=float(point*100+index)
                records.append(dict(point_id=point,center_y=point,center_x=0,grid_lat=50.+point,grid_lon=10.,
                    input_start=origin-pd.Timedelta(hours=3*(length-1)),forecast_origin=origin,
                    target_time=origin+pd.Timedelta(hours=3*lead),y_true=wind+lead,y_pred=wind,persistence_pred=wind))
        return pd.DataFrame(records)

    def test_common_origin_matching_does_not_require_equal_future_labels(self):
        source=self.frame(56,1).sample(frac=1,random_state=42)
        template=self.frame(56,2).iloc[[0,1,3,4]]
        result=match_origins(template,source)
        self.assertEqual(len(result),4)
        self.assertTrue(((template.target_time.to_numpy()-result.target_time.to_numpy())==np.timedelta64(3,'h')).all())
        with self.assertRaises(ValueError):
            match_origins(template,source.iloc[:1])
        with self.assertRaises(ValueError):
            match_origins(template,pd.concat([source,source.iloc[[0]]]))

    def test_length_and_lead_alignment_checks_and_origin_date(self):
        frames={}
        for length,lead in [(8,1),(56,1),(56,2)]:
            collapsed,daily=origin_errors(self.frame(length,lead),length,lead)
            frames[label(length,lead,'RS_S0')]=collapsed
            self.assertEqual(daily['count'].sum(),6)
            self.assertEqual(daily.squared.sum(),6*lead**2)
        assert_aligned(frames)
        crossing=self.frame(56,1)
        for column in ['input_start','forecast_origin','target_time']:
            crossing[column]+=pd.Timedelta(hours=21)
        _,daily=origin_errors(crossing,56,1)
        self.assertIn(pd.Timestamp('2024-04-01'),daily.index)
        self.assertEqual(daily.loc[pd.Timestamp('2024-04-01'),'count'],2)
        bad=frames.copy()
        bad['L56_H3_RS_S1']=frames['L56_H3_RS_S0'].copy()
        bad['L56_H3_RS_S1']['y_true']+=1
        with self.assertRaises(AssertionError):
            assert_aligned(bad)
        with self.assertRaises(ValueError):
            origin_errors(self.frame(8,1),56,1)

    def test_shared_bootstrap_known_gains_and_registered_row_budgets(self):
        weights=circular_weights(np.random.default_rng(42),28)
        np.testing.assert_array_equal(weights.sum(axis=1),28)
        ref=pd.DataFrame(dict(squared=np.arange(1,29,dtype=float),absolute=np.arange(1,29,dtype=float),count=[1]*28))
        model=ref.copy();model['squared']*=.25;model['absolute']*=.5
        for metric,column in [('RMSE','squared'),('MAE','absolute')]:
            point,draws=gain_distribution(model,ref,column,metric,weights)
            self.assertAlmostEqual(point,50.)
            np.testing.assert_allclose(draws,50.)
        self.assertEqual(len(comparisons('history')),13)
        self.assertEqual(len(comparisons('horizon')),6)
        self.assertEqual(sum(len(comparisons(stage)) for stage in ['history','horizon'])*3*2*4,456)
        self.assertEqual(sum(len(interaction_keys(stage)) for stage in ['history','horizon'])*3*2*4,72)
