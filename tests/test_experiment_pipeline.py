"""Lightweight synthetic tests; no TensorFlow initialization or real weather IO."""

import tempfile
import unittest
import json
import pickle
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from analyze_experiments import (analyze, attach_centers, collapse_grids,
                                 export_run, read_predictions, score, verify_alignment)
from train import create_dataset, prediction_frame


class ExperimentPipelineTests(unittest.TestCase):
    def test_windows_scaling_and_identifiers_are_unchanged(self):
        dates = pd.date_range('2024-06-01', periods=100, freq='3h')
        df = pd.concat([pd.DataFrame({'datetime': dates, 'point_id': point,
                                     'ws_10': np.arange(100) + point * 100})
                        for point in (0, 1)], ignore_index=True)
        result = create_dataset(df, 4, 2, ['ws_10'], include_metadata=True)
        X, y, Xv, yv, Xt, yt, baseline, scaler, splits, metadata = result
        self.assertEqual(X.shape, (130, 4, 1))
        self.assertEqual(Xv.shape, (10, 4, 1))
        self.assertEqual(Xt.shape, (30, 4, 1))
        self.assertEqual(scaler.data_max_[0], 169)
        for array, targets, start, count in [(X, y, 0, 65), (Xv, yv, 70, 5), (Xt, yt, 80, 15)]:
            expected_X, expected_y = [], []
            for point in (0, 1):
                values = np.arange(100) + point * 100
                for i in range(start, start + count):
                    expected_X.append(((values[i:i + 4]) / 169).reshape(4, 1))
                    expected_y.append(values[i + 5])
            np.testing.assert_allclose(array, expected_X)
            np.testing.assert_array_equal(targets, expected_y)
        self.assertEqual(pd.Timestamp(splits['test_start']), dates[80])
        self.assertEqual(metadata.iloc[0].forecast_origin, dates[83])
        self.assertEqual(metadata.iloc[0].target_time, dates[85])
        self.assertEqual(metadata.iloc[15].point_id, 1)
        np.testing.assert_array_equal(baseline[:15], np.arange(83, 98))
        np.testing.assert_array_equal(baseline[15:], np.arange(183, 198))
        frame = prediction_frame(metadata, yt, yt + 0.2, baseline, 'synthetic', 42, 'T01')
        self.assertTrue((frame.lead_hours == 6).all())
        np.testing.assert_array_equal(frame.y_true, yt)

    def make_frame(self, prediction):
        metadata = pd.DataFrame({'point_id': [0, 1],
                                 'input_start': pd.to_datetime(['2024-06-01'] * 2),
                                 'forecast_origin': pd.to_datetime(['2024-06-02'] * 2),
                                 'target_time': pd.to_datetime(['2024-06-02 06:00'] * 2)})
        return prediction_frame(metadata, [1, 1], prediction, [0, 0], 'test', 42, 'T01')

    def test_unique_grids_and_pair_alignment(self):
        centers = pd.DataFrame({'point_id': [0, 1], 'center_y': [5, 5], 'center_x': [8, 8],
                                'grid_lat': [50., 50.], 'grid_lon': [9., 9.]})
        frame = self.make_frame([1, 3])
        unique = collapse_grids(attach_centers(frame, centers))
        self.assertEqual(len(unique), 1)
        self.assertEqual(unique.y_pred.iloc[0], 2)
        verify_alignment(frame, self.make_frame([2, 2]))
        changed = frame.copy()
        changed.loc[1, 'target_time'] += pd.Timedelta(hours=3)
        with self.assertRaisesRegex(ValueError, 'different test'):
            verify_alignment(changed, frame)
        changed = frame.copy()
        changed.loc[1, 'y_true'] = 9
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            collapse_grids(attach_centers(changed, centers))

    def test_summary_pairing_and_single_seed_reporting(self):
        centers = pd.DataFrame({'point_id': [0, 1], 'center_y': [5, 5], 'center_x': [8, 8],
                                'grid_lat': [50., 50.], 'grid_lon': [9., 9.]})
        frames = {'T01': self.make_frame([2, 2]), 'S01': self.make_frame([1.5, 1.5])}
        records = [{'experiment': name, 'seed': 42, 'pred_len': 2, 'run_id': name}
                   for name in frames]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            output = Path(directory)
            with patch('analyze_experiments.read_predictions', side_effect=lambda r: frames[r['experiment']]), \
                    patch('analyze_experiments.make_plot'):
                analyze(records, centers, output)
            summary = pd.read_csv(output / 'seed_summary.csv')
            self.assertTrue((summary.n_seeds == 1).all())
            self.assertTrue(summary.RMSE_std.isna().all())
            paired = pd.read_csv(output / 'paired_seed_improvements.csv')
            np.testing.assert_allclose(paired.RMSE_improvement_pct, 50)
            unique = pd.read_csv(output / 'per_run_metrics.csv')
            self.assertTrue((unique[unique.scope == 'unique_grid'].samples == 1).all())

    def test_replay_uses_saved_scaler_without_fitting(self):
        dates = pd.date_range('2024-06-01', periods=100, freq='3h')
        df = pd.concat([pd.DataFrame({'datetime': dates, 'point_id': point,
                                     'ws_10': np.arange(100) + point * 100})
                        for point in (0, 1)], ignore_index=True)
        result = create_dataset(df, 4, 2, ['ws_10'])
        _, _, _, _, X, y, baseline, scaler, split_info = result
        pred = baseline + 0.2
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            source = root / 'source.csv'
            df.to_csv(source, index=False)
            (root / 'config.json').write_text(json.dumps(
                {'run_id': 'replay', 'seed': 42, 'features': ['ws_10']}))
            with (root / 'scaler.pkl').open('wb') as handle:
                pickle.dump(scaler, handle)
            row = {'run_dir': root, 'run_id': 'replay', 'seed': 42, 'experiment': 'T01',
                   'features': '["ws_10"]', 'data_path': str(source), 'model_path': str(root / 'model.keras'),
                   'input_len': 4, 'pred_len': 2, 'train_split_ratio': 0.7, 'val_split_ratio': 0.1,
                   'batch_size': 64, 'test_samples': len(y), **split_info, **score(y, pred, baseline)}
            class FakeModel:
                def predict(self, values, **kwargs):
                    np.testing.assert_allclose(values, X)
                    return pred[:, None]
            fake_tf = SimpleNamespace(keras=SimpleNamespace(
                models=SimpleNamespace(load_model=lambda *a, **k: FakeModel()),
                backend=SimpleNamespace(clear_session=lambda: None)))
            with patch.dict('sys.modules', {'tensorflow': fake_tf}), \
                    patch('sklearn.preprocessing.MinMaxScaler.fit', side_effect=AssertionError('No refitting')):
                export_run(row)
            frame = read_predictions(row)
            np.testing.assert_allclose(frame.y_pred, pred)
            self.assertTrue((frame.lead_hours == 6).all())


if __name__ == '__main__':
    unittest.main()
