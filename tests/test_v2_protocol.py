"""Small checks of time-boundary embargoes, past context and V1 compatibility."""

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from train import build_test_sequences, create_dataset, split_dataframe
from train import PROTOCOLS
from analyze_experiments import PROTOCOL_FIELDS, discover_runs


class V2ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range('2024-06-01', periods=100, freq='3h')
        self.df = pd.concat([pd.DataFrame({'datetime': self.dates, 'point_id': point,
                                          'ws_10': np.arange(100) + 100 * point})
                             for point in (0, 1)], ignore_index=True)

    def test_past_context_labels_and_training_unchanged(self):
        for horizon in (1, 2, 4, 8):
            with self.subTest(horizon=horizon):
                v2 = create_dataset(self.df, 4, horizon, ['ws_10'],
                                    include_metadata=True, protocol='v2')
                X, y, Xv, yv, Xt, yt, baseline, scaler, info, metadata = v2
                if horizon <= 4:
                    v1 = create_dataset(self.df, 4, horizon, ['ws_10'], include_metadata=True)
                    np.testing.assert_array_equal(X, v1[0])
                    np.testing.assert_array_equal(y, v1[1])
                    np.testing.assert_array_equal(scaler.data_max_, v1[7].data_max_)
                self.assertEqual(len(y), 2 * (70 - 4 - horizon + 1))
                self.assertEqual(scaler.data_max_[0], 169)
                self.assertEqual(len(yv), 2 * (10 - horizon))
                self.assertEqual(len(yt), 2 * (20 - horizon))
                # Train labels are earlier than validation origins.
                self.assertLess(y[:len(y) // 2].max(), 70)
                self.assertEqual(pd.Timestamp(info['validation_forecast_origin_min']), self.dates[70])
                self.assertEqual(pd.Timestamp(info['validation_target_time_min']), self.dates[70 + horizon])
                # All validation labels precede every test forecast origin.
                self.assertLess(pd.Timestamp(info['validation_target_time_max']), self.dates[80])
                self.assertEqual(metadata.forecast_origin.min(), self.dates[80])
                self.assertEqual(metadata.target_time.min(), self.dates[80 + horizon])
                self.assertEqual(metadata.input_start.min(), self.dates[77])
                self.assertTrue(((metadata.target_time - metadata.forecast_origin)
                                 == pd.Timedelta(hours=3 * horizon)).all())
                # First test window uses observed history 77..80, never the future.
                np.testing.assert_allclose(scaler.inverse_transform(Xt[0]).ravel(), np.arange(77, 81))
                self.assertEqual(baseline[0], 80)
                self.assertEqual(yt[0], 80 + horizon)
                self.assertEqual(metadata.iloc[len(yt) // 2].point_id, 1)
                # Model replay must reconstruct precisely the same test set.
                _, _, _, split = split_dataframe(self.df)
                replay = build_test_sequences(self.df, 4, horizon, ['ws_10'], scaler, split, 'v2')
                np.testing.assert_array_equal(replay[0], Xt)
                np.testing.assert_array_equal(replay[1], yt)
                pd.testing.assert_frame_equal(replay[3], metadata)

    def test_future_observations_cannot_change_training_validation_or_first_input(self):
        original = create_dataset(self.df, 4, 2, ['ws_10'], protocol='v2')
        changed = self.df.copy()
        changed.loc[changed.datetime > self.dates[80], 'ws_10'] += 10000
        perturbed = create_dataset(changed, 4, 2, ['ws_10'], protocol='v2')
        for index in (0, 1, 2, 3):
            np.testing.assert_array_equal(original[index], perturbed[index])
        np.testing.assert_array_equal(original[4][0], perturbed[4][0])
        np.testing.assert_array_equal(original[7].data_max_, perturbed[7].data_max_)
        self.assertNotEqual(original[5][0], perturbed[5][0])

    def test_analysis_keeps_protocols_separate(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            rows = []
            for protocol in ('v1', 'v2'):
                row = {field: 1 for field in PROTOCOL_FIELDS}
                row.update(experiment='T01', pred_len=1, seed=42, model_type='lstm',
                           evaluation_protocol=PROTOCOLS[protocol],
                           model_path=str(root / protocol / 'model.keras'))
                rows.append(row)
            pd.DataFrame(rows).to_csv(root / 'results.csv', index=False)
            for protocol in ('v1', 'v2'):
                found = discover_runs(root, protocol)
                self.assertEqual(len(found), 1)
                self.assertEqual(found[0]['evaluation_protocol'], PROTOCOLS[protocol])


if __name__ == '__main__':
    unittest.main()
