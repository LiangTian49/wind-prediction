"""Synthetic fold isolation and annual-reference checks; no real model or GRIB IO."""

import contextlib
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from analyze_experiments import (PROTOCOL_FIELDS, analyze, discover_runs, export_run,
                                 read_predictions, score)
from extract_cerra_spatial import check_reference, parse_args as extraction_args, prepare_time_axis
from temporal_folds import FoldSpec, STEP, expected_fold_counts, load_manifest, select_fold_panel
from train import PROTOCOLS, build_test_sequences, create_dataset, parse_args, prediction_frame


ROOT = Path(__file__).resolve().parents[1]


class FixedFoldTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range('2024-01-01', periods=80, freq='3h')
        self.df = pd.concat([pd.DataFrame({'point_id': point, 'datetime': self.dates,
                                          'ws_10': np.arange(80) + point * 1000})
                             for point in (0, 1)], ignore_index=True)
        self.fold = FoldSpec('X01', *(str(self.dates[i]) for i in (8, 32, 48, 64)))

    def test_exact_boundaries_context_and_scaler(self):
        outside = (self.df.datetime < self.dates[8]) | (self.df.datetime >= self.dates[64])
        self.df.loc[outside, 'ws_10'] = 90000
        for horizon in (1, 2, 4, 8):
            with self.subTest(horizon=horizon):
                X, y, Xv, yv, Xt, yt, baseline, scaler, info, meta = create_dataset(
                    self.df, 4, horizon, ['ws_10'], protocol='v2_fixed',
                    fold=self.fold, include_metadata=True)
                expected = expected_fold_counts(self.fold, 4, horizon, 2)
                self.assertEqual((len(y), len(yv), len(yt)), tuple(expected.values()))
                self.assertEqual(scaler.data_min_[0], 8)
                self.assertEqual(scaler.data_max_[0], 1031)
                np.testing.assert_allclose(scaler.inverse_transform(X[0]).ravel(), np.arange(8, 12))
                np.testing.assert_allclose(scaler.inverse_transform(Xt[0]).ravel(), np.arange(45, 49))
                self.assertEqual(baseline[0], 48)
                self.assertEqual(yt[0], 48 + horizon)
                self.assertEqual(meta.forecast_origin.min(), self.dates[48])
                self.assertEqual(meta.target_time.max(), self.dates[63])
                self.assertEqual(pd.Timestamp(info['validation_forecast_origin_min']), self.dates[32])
                self.assertEqual(pd.Timestamp(info['validation_target_time_max']), self.dates[47])
                replay = build_test_sequences(self.df, 4, horizon, ['ws_10'], scaler, info, 'v2_fixed')
                np.testing.assert_array_equal(replay[0], Xt)
                pd.testing.assert_frame_equal(replay[3], meta)

    def test_future_and_outside_fold_values_do_not_change_fitting(self):
        original = create_dataset(self.df, 4, 2, ['ws_10'], protocol='v2_fixed', fold=self.fold)
        altered = self.df.copy()
        altered.loc[altered.datetime > self.dates[48], 'ws_10'] += 200000
        altered.loc[altered.datetime < self.dates[8], 'ws_10'] += 300000
        changed = create_dataset(altered, 4, 2, ['ws_10'], protocol='v2_fixed', fold=self.fold)
        for index in (0, 1, 2, 3):
            np.testing.assert_array_equal(original[index], changed[index])
        np.testing.assert_array_equal(original[4][0], changed[4][0])
        np.testing.assert_array_equal(original[7].data_max_, changed[7].data_max_)

    def test_missing_time_invalid_dates_and_cli(self):
        missing = self.df.drop(self.df[(self.df.point_id == 1) & (self.df.datetime == self.dates[40])].index)
        with self.assertRaisesRegex(ValueError, 'missing/duplicate'):
            select_fold_panel(missing, self.fold)
        for values in [('2024-01-01 01:00', '2024-02-01', '2024-03-01', '2024-04-01'),
                       ('2024-02-01', '2024-01-01', '2024-03-01', '2024-04-01')]:
            with self.assertRaises(ValueError):
                FoldSpec('bad', *values)
        args = parse_args(['--protocol', 'v2_fixed', '--fold_id', 'X01',
                           '--train_start', self.fold.train_start,
                           '--validation_start', self.fold.validation_start,
                           '--test_start', self.fold.test_start, '--test_end', self.fold.test_end_exclusive])
        self.assertEqual(args.fold_spec, self.fold.as_dict())
        canonical = ROOT / 'configs/seasonal_folds_2024.json'
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--protocol', 'v2_fixed', '--fold_manifest', str(canonical), '--fold_id', 'F04'])
        args = parse_args(['--protocol', 'v2_fixed', '--fold_manifest', str(canonical),
                           '--fold_id', 'F04', '--allow_confirmation'])
        self.assertEqual(args.fold_spec['role'], 'locked_confirmation')
        self.assertEqual(len(args.fold_manifest_sha256), 64)

    def test_fixed_model_replay_without_refitting(self):
        result = create_dataset(self.df, 4, 2, ['ws_10'], protocol='v2_fixed', fold=self.fold)
        _, _, _, _, X, y, baseline, scaler, info = result
        pred = baseline + 0.5
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            source = root / 'source.csv'
            self.df.to_csv(source, index=False)
            config = {'run_id': 'fixed-replay', 'seed': 42, 'features': ['ws_10'],
                      'protocol': 'v2_fixed', 'fold_spec': self.fold.as_dict()}
            (root / 'config.json').write_text(json.dumps(config))
            with (root / 'scaler.pkl').open('wb') as handle:
                pickle.dump(scaler, handle)
            row = {'run_dir': root, 'run_id': 'fixed-replay', 'fold_id': 'X01', 'seed': 42,
                   'experiment': 'T01', 'features': '["ws_10"]', 'data_path': str(source),
                   'model_path': str(root / 'model.keras'), 'input_len': 4, 'pred_len': 2,
                   'batch_size': 64, 'test_samples': len(y), **info,
                   'evaluation_protocol': PROTOCOLS['v2_fixed'], **score(y, pred, baseline)}

            class FakeModel:
                def predict(self, values, **kwargs):
                    np.testing.assert_array_equal(values, X)
                    return pred[:, None]

            tf = SimpleNamespace(keras=SimpleNamespace(
                models=SimpleNamespace(load_model=lambda *a, **k: FakeModel()),
                backend=SimpleNamespace(clear_session=lambda: None)))
            with patch.dict('sys.modules', {'tensorflow': tf}), \
                    patch('sklearn.preprocessing.MinMaxScaler.fit', side_effect=AssertionError('No refitting')):
                export_run(row)
            frame = read_predictions(row)
            self.assertTrue((frame.fold_id == 'X01').all())
            self.assertLess(frame.target_time.max(), pd.Timestamp(self.fold.test_end_exclusive))


class FoldAnalysisTests(unittest.TestCase):
    def test_same_seed_and_horizon_in_different_folds_are_not_duplicates(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            rows = []
            for fold_id in ('F01', 'F02'):
                row = {field: 1 for field in PROTOCOL_FIELDS}
                row.update(fold_id=fold_id, experiment='T01', pred_len=1, seed=42,
                           model_type='lstm', evaluation_protocol=PROTOCOLS['v2_fixed'],
                           train_start='2024-01-01', test_end_exclusive='2024-05-01',
                           fold_role='development', fold_manifest_sha256='same',
                           model_path=str(root / fold_id / 'model.keras'),
                           train_split_ratio=None, val_split_ratio=None)
                rows.append(row)
            pd.DataFrame(rows).to_csv(root / 'results.csv', index=False)
            self.assertEqual(len(discover_runs(root, 'v2_fixed')), 2)
            pd.DataFrame(rows + [rows[0]]).to_csv(root / 'results.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'Multiple runs'):
                discover_runs(root, 'v2_fixed')

    def test_fold_pairing_cannot_mix_reference_models(self):
        centers = pd.DataFrame({'point_id': [0], 'center_y': [5], 'center_x': [8],
                                'grid_lat': [50.], 'grid_lon': [9.]})
        records, frames = [], {}
        for fold_id, start, spatial_pred in [('F01', '2024-04-01', 1.5), ('F02', '2024-07-01', 3.)]:
            for experiment, pred in [('T01', 2.), ('S01', spatial_pred)]:
                run_id = f'{fold_id}_{experiment}'
                metadata = pd.DataFrame({'point_id': [0], 'input_start': [pd.Timestamp(start)],
                                         'forecast_origin': [pd.Timestamp(start) + STEP],
                                         'target_time': [pd.Timestamp(start) + 3 * STEP]})
                frames[run_id] = prediction_frame(metadata, [1.], [pred], [0.], run_id,
                                                  42, experiment, fold_id)
                records.append({'run_id': run_id, 'fold_id': fold_id, 'experiment': experiment,
                                'seed': 42, 'pred_len': 2, 'evaluation_protocol': PROTOCOLS['v2_fixed']})
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            with patch('analyze_experiments.read_predictions', side_effect=lambda row: frames[row['run_id']]), \
                    patch('analyze_experiments.make_plot'):
                analyze(records, centers, output)
            paired = pd.read_csv(output / 'paired_seed_improvements.csv')
            np.testing.assert_allclose(paired[paired.fold_id == 'F01'].RMSE_improvement_pct, 50)
            np.testing.assert_allclose(paired[paired.fold_id == 'F02'].RMSE_improvement_pct, -100)


class AnnualReferenceTests(unittest.TestCase):
    def test_reference_subset_requires_all_old_rows_and_identical_values(self):
        dates = pd.date_range('2024-05-31', periods=16, freq='3h')
        frame = pd.DataFrame({'point_id': 0, 'datetime': dates, 'ws_10': np.arange(16, dtype=float)})
        reference = frame[frame.datetime >= '2024-06-01']
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / 'summer.csv'
            reference.to_csv(path, index=False)
            checked = check_reference(frame, path, 'reference_subset')
            self.assertEqual(checked['matched_rows'], len(reference))
            with self.assertRaises(ValueError):
                check_reference(frame, path, 'exact')
            with self.assertRaises(ValueError):
                check_reference(frame.iloc[:-1], path, 'reference_subset')
            changed = frame.copy()
            changed.loc[15, 'ws_10'] += 100
            with self.assertRaisesRegex(ValueError, 'differs'):
                check_reference(changed, path, 'reference_subset')

    def test_complete_year_including_leap_day_and_missing_end(self):
        args = extraction_args(['--months', *map(str, range(1, 13)), '--require_complete_months'])
        dates = pd.date_range('2024-01-01', '2025-01-01', freq='3h', inclusive='left')
        ds = xr.Dataset({'si10': (('time', 'y', 'x'), np.ones((len(dates), 1, 1)),
                                 {'units': 'm s**-1'})}, coords={'time': dates})
        _, selected, _, _ = prepare_time_axis(ds, args)
        self.assertEqual(len(selected), 2928)
        with self.assertRaisesRegex(ValueError, 'completely'):
            prepare_time_axis(ds.isel(time=slice(None, -1)), args)


if __name__ == '__main__':
    unittest.main()
